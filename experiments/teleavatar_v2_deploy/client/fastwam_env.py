#!/usr/bin/env python3
"""FastWAM Environment wrapper for openpi_client.runtime framework.

Adapts the existing TeleavatarROS2Interface to work with OpenPI's
standardized Environment interface.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ros2_interface import TeleavatarROS2Interface

logger = logging.getLogger("fastwam_env")


class FastWAMEnvironment:
    """Environment wrapper for FastWAM TA2 robot using openpi_client.runtime framework.

    This environment:
    - Uses existing TeleavatarROS2Interface for ROS2 + RTP communication
    - Adapts observations to OpenPI format
    - Handles sensor outages gracefully
    - Supports action chunk execution with interpolation
    """

    def __init__(
        self,
        task: Optional[str] = None,
        prompt: Optional[str] = None,
        control_frequency: float = 20.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
        rtp_port: int = 8890,
        rtp_decoder: str = "nvh265dec",
        enable_video: bool = True,
        swap_grippers: bool = False,
        min_grasp_hold_s: float = 0.0,
        grasp_close_trigger: float = 0.5,
        recovery_ramp_s: float = 0.75,
        outage_poll_hz: float = 100.0,
    ):
        """Initialize FastWAM environment.

        Args:
            task: Task-library key to condition on (a key of the server's
                taskmap.json). Sent verbatim to the server, which raises on an
                unknown name rather than silently using its startup default.
                None omits the field, keeping the server's startup instruction.
            prompt: Language instruction. Only usable when the server was
                started with --load-text-encoder; otherwise the server rejects
                it. Prefer `task`.
            control_frequency: Policy action rate (Hz)
            interp_frequency: ROS2 publish rate (Hz)
            interpolate: Enable joint interpolation
            rtp_port: RTP video stream port
            rtp_decoder: GStreamer decoder (nvh265dec)
            enable_video: Enable RTP video (False for debug)
            swap_grippers: Swap left/right gripper commands
            min_grasp_hold_s: Minimum grasp hold duration
            grasp_close_trigger: Trigger threshold for grasp
            recovery_ramp_s: Ramp duration after sensor outage
            outage_poll_hz: Rate to check sensors during outage
        """
        self._task = task
        self._prompt = prompt
        self._control_frequency = control_frequency
        self._recovery_ramp_s = recovery_ramp_s
        self._outage_poll_period = 1.0 / max(outage_poll_hz, 1.0)

        # Track recovery state
        self._agent = None
        self._pending_recovery = False

        # Initialize ROS2 interface
        self._ros_interface: Optional[TeleavatarROS2Interface] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._ros_executor: Optional[MultiThreadedExecutor] = None
        self._ros_stop_requested = threading.Event()
        self._ros_init_error: Optional[BaseException] = None
        self._ros_close_lock = threading.Lock()
        try:
            self._init_ros2(
                control_frequency=control_frequency,
                interp_frequency=interp_frequency,
                interpolate=interpolate,
                rtp_port=rtp_port,
                rtp_decoder=rtp_decoder,
                enable_video=enable_video,
                swap_grippers=swap_grippers,
                min_grasp_hold_s=min_grasp_hold_s,
                grasp_close_trigger=grasp_close_trigger,
            )
        except BaseException:
            # __init__ has no usable instance to return, so clean up here rather
            # than leaving ROS/GStreamer native threads behind during an error.
            self.close()
            raise

        logger.info(f"FastWAMEnvironment initialized with prompt: '{prompt}'")

    def attach_agent(self, agent) -> None:
        """Register agent for action chunk invalidation after sensor outages.

        ActionChunkBroker caches actions computed from observations. After an
        outage, those observations are stale and the cached chunk must be
        discarded.
        """
        self._agent = agent
        logger.info("Agent attached to environment")

    def _init_ros2(self, **kwargs):
        """Initialize ROS2 in background thread and wait for initial sensor data."""
        spin_started = threading.Event()

        def ros_spin():
            executor = None
            interface = None
            try:
                rclpy.init()
                if self._ros_stop_requested.is_set():
                    return

                interface = TeleavatarROS2Interface(**kwargs)
                self._ros_interface = interface

                # Spin in background
                executor = MultiThreadedExecutor()
                executor.add_node(interface)
                with self._ros_close_lock:
                    self._ros_executor = executor

                spin_started.set()
                if self._ros_stop_requested.is_set():
                    return
                executor.spin()
            except BaseException as exc:
                self._ros_init_error = exc
                logger.exception("ROS2 interface thread failed")
            finally:
                spin_started.set()
                if executor is not None:
                    try:
                        executor.shutdown()
                    except Exception:
                        logger.exception("Failed to shut down ROS2 executor")
                if interface is not None:
                    try:
                        interface.destroy_node()
                    except Exception:
                        logger.exception("Failed to destroy ROS2 interface")
                with self._ros_close_lock:
                    if self._ros_executor is executor:
                        self._ros_executor = None
                if rclpy.ok():
                    rclpy.shutdown()

        self._ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self._ros_thread.start()

        # Wait for interface to be created
        timeout = 10.0
        start_time = time.time()
        while (
            self._ros_interface is None
            and self._ros_init_error is None
            and time.time() - start_time < timeout
        ):
            time.sleep(0.1)

        if self._ros_init_error is not None:
            raise RuntimeError("Failed to initialize ROS2 interface") from self._ros_init_error

        if self._ros_interface is None:
            raise RuntimeError("Failed to initialize ROS2 interface")

        logger.info("ROS2 interface created, waiting for executor to start...")

        if not spin_started.wait(timeout=5.0):
            raise RuntimeError("ROS2 executor failed to start")

        if self._ros_init_error is not None:
            raise RuntimeError("ROS2 interface thread failed") from self._ros_init_error

        logger.info("ROS2 executor started, waiting for initial sensor data...")

        # Wait for initial data (RTP video + joint states)
        if not self._ros_interface.wait_for_initial_data(timeout=30.0):
            raise RuntimeError(
                "Failed to receive initial sensor data. Check:\n"
                "  - RTP stream: robot pushing to this host:8890\n"
                "  - GStreamer: gst-inspect-1.0 nvh265dec\n"
                f"  - ROS2: ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '<unset>')}, "
                "ros2 topic list\n"
                "  - Joint states: ros2 topic echo /left_arm/joint_states --once"
            )

        logger.info("ROS2 interface ready with sensor data")

    def close(self) -> None:
        """Stop ROS2 and RTP resources; safe to call during failed init."""
        self._ros_stop_requested.set()

        with self._ros_close_lock:
            executor = self._ros_executor
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:
                logger.exception("Failed to request ROS2 executor shutdown")

        thread = self._ros_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("ROS2 thread did not exit within 3s")

    def reset(self) -> None:
        """Reset environment (no-op for real robot deployment)."""
        logger.info("Environment reset called (no-op for real robot)")

    def is_episode_complete(self) -> bool:
        """Check if episode is complete.

        For real robot, episodes never complete automatically - must be
        terminated by user (Ctrl+C).
        """
        return False

    def get_observation(self) -> Dict:
        """Get current observation from robot sensors.

        Blocks until sensors are fresh. If any sensor is missing or stale,
        freezes the robot and waits for recovery.

        Returns:
            Dictionary with OpenPI-style keys:
                - 'observation/state': np.ndarray (72,) float32
                - 'observation/images/head_camera': np.ndarray (H, W, 3) uint8
                - 'observation/images/left_color': np.ndarray (H, W, 3) uint8
                - 'observation/images/right_color': np.ndarray (H, W, 3) uint8
                - 'prompt': str
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        # Get raw observation
        raw_obs = self._ros_interface.get_observation()

        # Handle sensor outage
        if raw_obs is None:
            raw_obs = self._wait_for_fresh_observation()

        # Convert to OpenPI format
        # Images: already HWC uint8 from ROS2 interface
        # State: build 72-d proprio
        images = raw_obs['images']
        state = raw_obs['state']

        obs = {
            'observation/state': state,
            'observation/images/head_camera': images['head_camera'],
            'observation/images/left_color': images['left_color'],
            'observation/images/right_color': images['right_color'],
        }
        # Omit both when unset so the server keeps its startup instruction.
        if self._task:
            obs['task'] = self._task
        if self._prompt:
            obs['prompt'] = self._prompt
        return obs

    def _wait_for_fresh_observation(self) -> Dict:
        """Block until sensors are live again, freezing robot meanwhile.

        The robot freezes automatically: by not returning, the control loop
        stops calling apply_action, so the ROS2 interpolation timer keeps
        republishing the last command and the enable heartbeat stays alive.

        This never raises - waiting is better than crashing and losing the
        enable heartbeat.
        """
        outage_start = time.time()
        logger.error(
            "SENSOR_OUTAGE_START - Freezing robot at last commanded pose. "
            "Waiting for recovery..."
        )
        last_log = outage_start

        while True:
            time.sleep(self._outage_poll_period)
            raw_obs = self._ros_interface.get_observation()

            if raw_obs is not None:
                duration = time.time() - outage_start
                logger.warning(
                    f"SENSOR_OUTAGE_END duration={duration:.2f}s - "
                    f"Discarding cached actions and easing back in over {self._recovery_ramp_s:.2f}s"
                )

                # Invalidate cached action chunk
                if self._agent is not None:
                    self._agent.reset()

                self._pending_recovery = True
                return raw_obs

            now = time.time()
            if now - last_log >= 5.0:
                logger.error(f"Still frozen: sensors unavailable for {now - outage_start:.1f}s")
                last_log = now

    def apply_action(self, action: Dict) -> None:
        """Apply action to robot.

        Args:
            action: Dictionary containing 'actions' key with 16-dim action array
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        if 'actions' not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        # Ensure correct shape
        if actions.shape != (16,):
            raise ValueError(f"Expected 16-dim action, got shape {actions.shape}")

        # Use longer ramp after outage to avoid jerky motion
        ramp_duration = self._recovery_ramp_s if self._pending_recovery else None
        self._pending_recovery = False

        # Publish to ROS2
        self._ros_interface.publish_action16(actions, ramp_duration=ramp_duration)

    def set_prompt(self, prompt: str):
        """Update the language instruction prompt.

        Args:
            prompt: New language instruction
        """
        self._prompt = prompt
        logger.info(f"Updated prompt to: '{prompt}'")
