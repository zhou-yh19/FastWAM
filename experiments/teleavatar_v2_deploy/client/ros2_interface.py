#!/usr/bin/env python3
"""ROS2 + RTP interface for FastWAM TA2 deploy (runs on the robot-side control machine).

Cameras: RTP H265 composite on port 8890 (see rtp_video_interface.py).

Policy images are monocular, matching the *_mono training datasets: each camera key
carries only the LEFT eye of that camera's side-by-side stereo pair.

  head_camera  <- head_left_eye
  left_color   <- left_wrist_left_eye
  right_color  <- right_wrist_left_eye

Server then runs the same compose_ta2_mosaic as training.

State: client still builds raw 72-d observation.state (server slices to 14-d).
Missing EE/chassis/kinco are zero-filled.

Actions from server are 16-d denormalized:
  [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
  policy-rate arm targets -> interpolation-rate joint_cmd;
  grip efforts -> latest trigger target via TA2 curve.
"""

from __future__ import annotations

import pathlib
import time
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from rtp_video_interface import RTPH265VideoInterface
from fastwam_rosbag_bridge import FastWamBagPublisher

# Training-aligned monocular mapping: policy key -> the RTP split view it reads.
# Left eye only. The *_mono training datasets crop each TA2 camera's side-by-side stereo
# pair to its left eye, because the right eye duplicated the viewpoint without adding
# per-eye detail. rtp_video_interface already publishes every eye as its own entry, so
# this is a straight dict pick, not a crop.
_POLICY_TO_RTP: Dict[str, str] = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_left_eye",
    "right_color": "right_wrist_left_eye",
}


def gripper_effort_to_trigger(effort: float) -> float:
    """Dataset gripper effort (Nm) -> platform [0,1] trigger (openpi teleavatar_v2)."""
    effort = float(effort)
    if effort > 0:
        trigger = 0.10 * (1.0 - effort / 2.0)
    else:
        trigger = 0.10 - effort * 0.90 / 1.6
    return float(np.clip(trigger, 0.0, 1.0))


class TeleavatarROS2Interface(Node):
    def __init__(
        self,
        node_name: str = "fastwam_ta2_client",
        rtp_port: int = 8890,
        rtp_payload: int = 96,
        rtp_decoder: str = "nvh265dec",
        enable_video: bool = True,
        swap_grippers: bool = False,
        min_grasp_hold_s: float = 0.0,
        grasp_close_trigger: float = 0.5,
        grasp_hold_trigger: float = 0.95,
        control_frequency: float = 20.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
        sensor_timeout: float = 1.0,
        arm_config_path: Optional[pathlib.Path] = None,
    ):
        print("[client] ROS Node.__init__ ...", flush=True)
        super().__init__(node_name)
        print("[client] ROS Node ready", flush=True)
        self.logger = self.get_logger()
        self.lock = Lock()
        self.sensor_timeout = sensor_timeout
        self.swap_grippers = bool(swap_grippers)
        self.min_grasp_hold_s = float(min_grasp_hold_s)
        self.grasp_close_trigger = float(grasp_close_trigger)
        self.grasp_hold_trigger = float(np.clip(grasp_hold_trigger, 0.0, 1.0))
        self._grip_hold_until = {"left": 0.0, "right": 0.0}
        self._interpolate = bool(interpolate)
        self._ctrl_period = 1.0 / max(float(control_frequency), 1e-3)
        self._interp_period = 1.0 / max(float(interp_frequency), 1.0)
        self._cmd_lock = Lock()
        self._publish_lock = Lock()
        self._command_stream_stopped = False
        self._ramp_from: Optional[np.ndarray] = None
        self._ramp_to: Optional[np.ndarray] = None
        self._ramp_t0: Optional[float] = None
        self._ramp_duration = self._ctrl_period
        self._last_cmd_pos: Optional[np.ndarray] = None
        self._gripper_target = np.zeros(2, dtype=np.float64)
        self._have_target = False
        self._enable_counter = 0
        command_rate = float(interp_frequency) if self._interpolate else float(control_frequency)
        self._enable_every = max(1, int(round(command_rate / 50.0)))

        cfg_path = arm_config_path or (pathlib.Path(__file__).resolve().parent / "arm_config.yml")
        arm_config = yaml.safe_load(open(cfg_path))
        self.joint_lower = {
            "left_arm": np.array(arm_config["arms"]["left_arm"]["lower"]),
            "right_arm": np.array(arm_config["arms"]["right_arm"]["lower"]),
        }
        self.joint_upper = {
            "left_arm": np.array(arm_config["arms"]["left_arm"]["upper"]),
            "right_arm": np.array(arm_config["arms"]["right_arm"]["upper"]),
        }

        self.latest_joint_states: Dict[str, JointState] = {}
        self.joint_timestamps: Dict[str, float] = {}
        self.latest_ee: Dict[str, Pose] = {}
        self.ee_timestamps: Dict[str, float] = {}
        self.latest_gripper: Dict[str, JointState] = {}
        self.gripper_timestamps: Dict[str, float] = {}
        self.latest_chassis: Optional[JointState] = None
        self.chassis_timestamp: float = 0.0

        self.left_joint_names = [f"l_joint{i}" for i in range(1, 8)]
        self.right_joint_names = [f"r_joint{i}" for i in range(1, 8)]

        self._video = None
        if enable_video:
            print(f"[client] Starting RTP port={rtp_port} decoder={rtp_decoder}", flush=True)
            self._video = RTPH265VideoInterface(
                port=rtp_port, payload=rtp_payload, decoder=rtp_decoder
            )
            self._video.start()
            print("[client] RTP start() returned", flush=True)
        else:
            print("[client] Video disabled (--skip-video)", flush=True)

        self._setup_subscribers()
        self._setup_publishers()
        # RTP is not a ROS topic. Republish decoded policy views as compressed
        # images so rosbag2 can record them alongside joint states.
        self._bag_bridge = FastWamBagPublisher(self)
        if self._video is not None:
            self._bag_bridge.start_image_publisher(self._bag_image_source)
        if self._interpolate:
            self._interp_timer = self.create_timer(self._interp_period, self._interp_publish)
            self.logger.info(
                f"des_q interpolation ON: {control_frequency:.0f} Hz target -> "
                f"{interp_frequency:.0f} Hz ROS2 publish "
                f"(ramp {self._ctrl_period * 1e3:.0f} ms)"
            )
        else:
            self._interp_timer = None
            self.logger.info(
                f"des_q interpolation OFF: publishing raw targets at "
                f"{control_frequency:.0f} Hz"
            )
        self.logger.info(f"TeleavatarROS2Interface ready (RTP={enable_video} + ROS2)")
        print("[client] TeleavatarROS2Interface ready", flush=True)

    def _setup_subscribers(self) -> None:
        self.create_subscription(JointState, "/left_arm/joint_states", lambda m: self._js_cb(m, "left_arm"), 10)
        self.create_subscription(JointState, "/right_arm/joint_states", lambda m: self._js_cb(m, "right_arm"), 10)
        self.create_subscription(JointState, "/left_gripper/joint_states", lambda m: self._grip_cb(m, "left"), 10)
        self.create_subscription(JointState, "/right_gripper/joint_states", lambda m: self._grip_cb(m, "right"), 10)
        self.create_subscription(JointState, "/chassis/joint_states", self._chassis_cb, 10)
        self.create_subscription(Pose, "/left_arm/current_ee_pose", lambda m: self._ee_cb(m, "left"), 10)
        self.create_subscription(Pose, "/right_arm/current_ee_pose", lambda m: self._ee_cb(m, "right"), 10)

    def _setup_publishers(self) -> None:
        self.pubs = {
            # TA2 API mode consumes commands below /api; joint_states remain
            # under /<group>/joint_states and are subscribed above.
            "left_arm": self.create_publisher(JointState, "/api/left_arm/joint_cmd", 10),
            "right_arm": self.create_publisher(JointState, "/api/right_arm/joint_cmd", 10),
            "left_gripper": self.create_publisher(Float32, "/api/left_gripper/cmd", 10),
            "right_gripper": self.create_publisher(Float32, "/api/right_gripper/cmd", 10),
        }
        self.enable_pub = self.create_publisher(Float32, "/api/fsm/enable", 10)

    def _js_cb(self, msg: JointState, key: str) -> None:
        with self.lock:
            self.latest_joint_states[key] = msg
            self.joint_timestamps[key] = time.time()

    def _grip_cb(self, msg: JointState, key: str) -> None:
        with self.lock:
            self.latest_gripper[key] = msg
            self.gripper_timestamps[key] = time.time()

    def _chassis_cb(self, msg: JointState) -> None:
        with self.lock:
            self.latest_chassis = msg
            self.chassis_timestamp = time.time()

    def _ee_cb(self, msg: Pose, key: str) -> None:
        with self.lock:
            self.latest_ee[key] = msg
            self.ee_timestamps[key] = time.time()

    def destroy_node(self) -> None:
        try:
            self.stop_command_stream()
            # Stop republishing before the decoder goes away, so the poll thread
            # cannot touch a torn-down pipeline.
            self._bag_bridge.stop_image_publisher()
            if self._video is not None:
                self._video.stop()
        finally:
            super().destroy_node()

    def wait_for_initial_data(self, timeout: float = 15.0) -> bool:
        required = ["left_arm", "right_arm"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._video is None:
                video_ok = True
            else:
                video_ok = self._video.has_initial_frame()
            with self.lock:
                joints_ok = all(k in self.latest_joint_states for k in required)
            if video_ok and joints_ok:
                self._log_command_connections()
                self.logger.info(f"Sensors ready (video={self._video is not None})")
                return True
            time.sleep(0.1)
        with self.lock:
            joints_present = {
                key: key in self.latest_joint_states for key in required
            }
        video_ok = self._video is None or self._video.has_initial_frame()
        if self._video is None:
            rtp_packets = h265_buffers = 0
        else:
            rtp_packets = self._video._rtp_packet_count
            h265_buffers = self._video._h265_buffer_count
        self.logger.error(
            "Timeout waiting for initial sensors: "
            f"video={video_ok} left_arm={joints_present['left_arm']} "
            f"right_arm={joints_present['right_arm']} "
            f"rtp_packets={rtp_packets} h265_buffers={h265_buffers}"
        )
        return False

    def _log_command_connections(self) -> None:
        """Report whether a remote controller is subscribed to command topics."""
        publishers = {
            **self.pubs,
            "enable": self.enable_pub,
        }
        counts = {
            name: publisher.get_subscription_count()
            for name, publisher in publishers.items()
        }
        self.logger.info(
            "Command topic subscribers: "
            f"left_arm={counts['left_arm']} right_arm={counts['right_arm']} "
            f"left_gripper={counts['left_gripper']} "
            f"right_gripper={counts['right_gripper']} "
            f"enable={counts['enable']}"
        )
        if counts["left_arm"] == 0 or counts["right_arm"] == 0:
            self.logger.warning(
                "No subscriber detected on one or both arm command topics. "
                "Verify API mode and the /api/*/joint_cmd topic names."
            )

    @staticmethod
    def _field(msg: Optional[JointState], field: str, n: int, preferred_names: Optional[list] = None) -> np.ndarray:
        """Read joint vector; prefer name matching (l_joint1..7) over raw array order."""
        out = np.zeros(n, dtype=np.float32)
        if msg is None:
            return out
        data = list(getattr(msg, field, []) or [])
        names = list(getattr(msg, "name", []) or [])
        if preferred_names and names and len(data) == len(names):
            name_to_val = {str(nm): float(data[i]) for i, nm in enumerate(names)}
            matched = 0
            for i, nm in enumerate(preferred_names[:n]):
                if nm in name_to_val:
                    out[i] = name_to_val[nm]
                    matched += 1
            if matched == n:
                return out
        m = min(n, len(data))
        if m:
            out[:m] = np.asarray(data[:m], dtype=np.float32)
        return out

    def build_proprio_72(self) -> np.ndarray:
        """Match the LeRobot 72-d observation.state layout."""
        with self.lock:
            left = self.latest_joint_states.get("left_arm")
            right = self.latest_joint_states.get("right_arm")
            lg = self.latest_gripper.get("left")
            rg = self.latest_gripper.get("right")
            chassis = self.latest_chassis
            lee = self.latest_ee.get("left")
            ree = self.latest_ee.get("right")

        state = np.zeros(72, dtype=np.float32)
        # positions 0-15
        state[0:7] = self._field(left, "position", 7, self.left_joint_names)
        state[7] = self._field(lg, "position", 1)[0]
        state[8:15] = self._field(right, "position", 7, self.right_joint_names)
        state[15] = self._field(rg, "position", 1)[0]
        # velocities 16-31
        state[16:23] = self._field(left, "velocity", 7, self.left_joint_names)
        state[23] = self._field(lg, "velocity", 1)[0]
        state[24:31] = self._field(right, "velocity", 7, self.right_joint_names)
        state[31] = self._field(rg, "velocity", 1)[0]
        # efforts 32-47
        state[32:39] = self._field(left, "effort", 7, self.left_joint_names)
        state[39] = self._field(lg, "effort", 1)[0]
        state[40:47] = self._field(right, "effort", 7, self.right_joint_names)
        state[47] = self._field(rg, "effort", 1)[0]
        # EE 48-61
        if lee is not None:
            state[48:51] = [lee.position.x, lee.position.y, lee.position.z]
            state[51:55] = [lee.orientation.x, lee.orientation.y, lee.orientation.z, lee.orientation.w]
        if ree is not None:
            state[55:58] = [ree.position.x, ree.position.y, ree.position.z]
            state[58:62] = [ree.orientation.x, ree.orientation.y, ree.orientation.z, ree.orientation.w]
        # chassis 62-70
        if chassis is not None:
            state[62:65] = self._field(chassis, "position", 3)
            state[65:68] = self._field(chassis, "velocity", 3)
            state[68:71] = self._field(chassis, "effort", 3)
        # kinco 71 left 0
        return state

    def _bag_image_source(
        self,
        since: Dict[str, float],
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Feed FastWamBagPublisher's thread with newly decoded policy views.

        Translates between the bag's camera keys and the RTP split-view names, and
        only hands over frames the publisher has not recorded yet.
        """
        if self._video is None:
            return {}
        since_by_view = {
            view: since[key]
            for key, view in _POLICY_TO_RTP.items()
            if key in since
        }
        fresh = self._video.get_views_if_newer(_POLICY_TO_RTP.values(), since_by_view)
        return {
            key: fresh[view]
            for key, view in _POLICY_TO_RTP.items()
            if view in fresh
        }

    def get_observation(self) -> Optional[Dict]:
        now = time.time()
        rtp_images, rtp_stamps = self._video.get_latest_images_with_timestamps()

        required_views = list(_POLICY_TO_RTP.values())
        for view in required_views:
            stamp = rtp_stamps.get(view)
            if view not in rtp_images or stamp is None or now - stamp > self.sensor_timeout:
                self.logger.error(f"video stale/missing: {view}", throttle_duration_sec=1.0)
                return None

        joint_stamps: Dict[str, float] = {}
        with self.lock:
            for g in ("left_arm", "right_arm"):
                stamp = self.joint_timestamps.get(g)
                if stamp is None or now - stamp > self.sensor_timeout:
                    self.logger.error(f"joints stale/missing: {g}", throttle_duration_sec=1.0)
                    return None
                joint_stamps[g] = float(stamp)

        # Match training: each policy key carries a single (left) eye.
        images = {key: rtp_images[view] for key, view in _POLICY_TO_RTP.items()}
        # Bag republishing runs in FastWamBagPublisher's own thread; encoding it
        # here would put ~20 ms of JPEG work per iteration on the control loop.
        view_stamps = {view: float(rtp_stamps[view]) for view in required_views}
        all_stamps = list(view_stamps.values()) + list(joint_stamps.values())
        timing = {
            "t_assembled": now,
            "image_stamps": view_stamps,
            "joint_stamps": joint_stamps,
            "image_age_ms": {k: (now - v) * 1000.0 for k, v in view_stamps.items()},
            "joint_age_ms": {k: (now - v) * 1000.0 for k, v in joint_stamps.items()},
            "skew_ms": (max(all_stamps) - min(all_stamps)) * 1000.0,
            "image_vs_joint_ms": (
                min(view_stamps.values()) - min(joint_stamps.values())
            )
            * 1000.0,
        }
        return {
            "images": images,
            "state": self.build_proprio_72(),
            "timing": timing,
        }

    def _clamp(self, arm: str, pos: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(pos, dtype=np.float64), self.joint_lower[arm], self.joint_upper[arm])

    def publish_enable(self, value: float = 1.0) -> None:
        msg = Float32()
        msg.data = float(value)
        self.enable_pub.publish(msg)

    def publish_action16(
        self,
        action: np.ndarray,
        ramp_duration: Optional[float] = None,
    ) -> None:
        """Set one 16-d policy action as the latest actuator target.

        Arms ramp from the last published position to the new target over one
        policy period and are republished by the interpolation timer. Grippers
        are not interpolated; the timer republishes the latest effort target.
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (16,):
            self.logger.error(f"expected action shape (16,), got {a.shape}")
            return

        arm14 = np.concatenate([a[0:7], a[8:15]]).astype(np.float64)
        grip2 = np.asarray([a[7], a[15]], dtype=np.float64)
        if not self._interpolate:
            self._publish_cmd(arm14, grip2)
            return

        now = time.monotonic()
        with self._cmd_lock:
            if self._last_cmd_pos is None:
                measured = self._current_arm_positions()
                self._last_cmd_pos = measured if measured is not None else arm14.copy()
            self._ramp_from = self._last_cmd_pos.copy()
            self._ramp_to = arm14
            self._ramp_t0 = now
            self._ramp_duration = max(
                self._ctrl_period if ramp_duration is None else float(ramp_duration),
                1e-3,
            )
            self._gripper_target = grip2
            self._have_target = True

    def _current_arm_positions(self) -> Optional[np.ndarray]:
        with self.lock:
            left_state = self.latest_joint_states.get("left_arm")
            right_state = self.latest_joint_states.get("right_arm")
        if left_state is None or right_state is None:
            return None
        return np.concatenate(
            [
                self._field(left_state, "position", 7, self.left_joint_names),
                self._field(right_state, "position", 7, self.right_joint_names),
            ]
        ).astype(np.float64)

    def _interp_publish(self) -> None:
        with self._cmd_lock:
            if not self._have_target:
                return
            assert self._ramp_t0 is not None
            assert self._ramp_from is not None
            assert self._ramp_to is not None
            alpha = (time.monotonic() - self._ramp_t0) / self._ramp_duration
            alpha = float(np.clip(alpha, 0.0, 1.0))
            arm14 = self._ramp_from + alpha * (self._ramp_to - self._ramp_from)
            self._last_cmd_pos = arm14.copy()
            grip2 = self._gripper_target.copy()
        self._publish_cmd(arm14, grip2)

    def _publish_cmd(self, arm14: np.ndarray, grip2: np.ndarray) -> None:
        """Publish one arm position frame plus the latest gripper targets."""
        with self._publish_lock:
            if self._command_stream_stopped:
                return
            self._publish_cmd_locked(arm14, grip2)

    def _publish_cmd_locked(self, arm14: np.ndarray, grip2: np.ndarray) -> None:
        left_pos = self._clamp("left_arm", arm14[0:7])
        right_pos = self._clamp("right_arm", arm14[7:14])
        stamp = self.get_clock().now().to_msg()

        self._enable_counter += 1
        if self._enable_counter % self._enable_every == 0:
            self.publish_enable(1.0)

        left = JointState()
        left.header.stamp = stamp
        left.header.frame_id = "left_arm"
        left.name = list(self.left_joint_names)
        left.position = left_pos.tolist()
        left.velocity = [0.0] * 7
        left.effort = [0.0] * 7
        self.pubs["left_arm"].publish(left)

        right = JointState()
        right.header.stamp = stamp
        right.header.frame_id = "right_arm"
        right.name = list(self.right_joint_names)
        right.position = right_pos.tolist()
        right.velocity = [0.0] * 7
        right.effort = [0.0] * 7
        self.pubs["right_arm"].publish(right)

        left_eff = float(grip2[0])
        right_eff = float(grip2[1])
        if self.swap_grippers:
            left_eff, right_eff = right_eff, left_eff

        lg_trig = float(np.clip(gripper_effort_to_trigger(left_eff), 0.0, 1.0))
        rg_trig = float(np.clip(gripper_effort_to_trigger(right_eff), 0.0, 1.0))
        lg_trig, lg_held = self._apply_grasp_hold("left", lg_trig)
        rg_trig, rg_held = self._apply_grasp_hold("right", rg_trig)

        lg = Float32()
        lg.data = lg_trig
        self.pubs["left_gripper"].publish(lg)
        rg = Float32()
        rg.data = rg_trig
        self.pubs["right_gripper"].publish(rg)

        hold_tag = ""
        if lg_held or rg_held:
            hold_tag = f" holdL={int(lg_held)} holdR={int(rg_held)}"
        self.logger.info(
            f"pub16 Lq0={float(left_pos[0]):.3f} Rq0={float(right_pos[0]):.3f} "
            f"Lgrip_eff={left_eff:.3f}->trig={lg.data:.3f} "
            f"Rgrip_eff={right_eff:.3f}->trig={rg.data:.3f}{hold_tag}",
            throttle_duration_sec=1.0,
        )

    def stop_command_stream(self) -> None:
        """Stop the high-rate command timer before disabling the robot."""
        if self._interp_timer is not None:
            self._interp_timer.cancel()
        with self._cmd_lock:
            self._have_target = False
        # Drain an in-flight timer callback before the caller publishes enable=0.
        with self._publish_lock:
            self._command_stream_stopped = True

    def _apply_grasp_hold(self, side: str, trigger: float) -> Tuple[float, bool]:
        """Latch close commands so policy open-flickers cannot drop the object."""
        trig = float(np.clip(trigger, 0.0, 1.0))
        hold_s = float(self.min_grasp_hold_s)
        if hold_s <= 0.0:
            return trig, False

        now = time.time()
        thresh = float(self.grasp_close_trigger)
        hold_trig = float(self.grasp_hold_trigger)

        if trig >= thresh:
            # (Re)arm hold window on every close command.
            self._grip_hold_until[side] = max(self._grip_hold_until.get(side, 0.0), now + hold_s)
            return max(trig, hold_trig), False

        until = float(self._grip_hold_until.get(side, 0.0))
        if now < until:
            return hold_trig, True
        return trig, False

    # Backward-compatible alias
    publish_action72 = publish_action16

    def hold_current_joints(self) -> None:
        """Safe sync: publish current arm joints (no big jump)."""
        with self.lock:
            left = self.latest_joint_states.get("left_arm")
            right = self.latest_joint_states.get("right_arm")
        if left is None or right is None:
            return
        a = np.zeros(16, dtype=np.float32)
        a[0:7] = self._field(left, "position", 7, self.left_joint_names)
        a[8:15] = self._field(right, "position", 7, self.right_joint_names)
        # effort~0 -> trigger 0.10 (neutral)
        a[7] = 0.0
        a[15] = 0.0
        self.publish_action16(a)
