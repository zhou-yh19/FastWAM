#!/usr/bin/env python3
"""FastWAM TA2 WebSocket client using OpenPI runtime framework.

WebSocket-based client with msgpack serialization for better performance.
Compatible with serve_policy_ws.py server.

Example:
    # Start server first (on GPU machine)
    python server/serve_policy_ws.py --checkpoint ... --dataset-stats ...

    # Then run client (on robot control machine)
    python client/run_client_ws.py \\
        --server-host 192.168.1.100 \\
        --server-port 8000 \\
        --control-hz 20 \\
        --publish-hz 200 \\
        --open-loop-horizon 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("fastwam_ws_client")


class ChunkRecordingPolicy:
    """Republish each policy chunk to /fastwam/policy/* so rosbag captures it.

    Sits between the WebSocket policy and ActionChunkBroker. The broker only calls
    its inner policy when the current chunk is exhausted, so exactly one call lands
    here per real inference -- with the full [T, 16] array, before the broker slices
    it down to a single step. Hooking the broker's *output* instead would publish one
    row per control tick and lose the chunk structure that analyze_deploy_bag.py
    reconstructs the "predicted" trace from.

    Without this, /fastwam/policy/action_chunk and /fastwam/policy/inference_ms never
    enter the bag and analyze_deploy_bag.py can only do commanded-vs-measured -- which
    is what happened to every bag recorded before this was added.
    """

    def __init__(self, policy, bridge_getter) -> None:
        self._policy = policy
        self._bridge_getter = bridge_getter
        self._warned = False

    def infer(self, obs):
        result = self._policy.infer(obs)
        self._publish(result)
        return result

    def _publish(self, result) -> None:
        # Recording is strictly best-effort: a bag-publishing failure must never break
        # the control loop, because the arms are moving.
        try:
            bridge = self._bridge_getter()
            if bridge is None:
                return
            actions = result.get("actions")
            if actions is None:
                return
            bridge.publish_action_chunk(actions)
            # total_ms is the server's own end-to-end inference time, the same quantity
            # the HTTP client published as infer_s * 1000. Keeping it identical means
            # inference_ms stays comparable against bags recorded before the switch.
            timing = result.get("server_timing") or {}
            inference_ms = timing.get("total_ms", timing.get("infer_ms", 0.0))
            bridge.publish_inference_time(float(inference_ms))
        except Exception:
            if not self._warned:
                logger.warning(
                    "Failed to publish policy chunk for rosbag; continuing without "
                    "policy topics. This bag will not support predicted-vs-commanded "
                    "analysis.",
                    exc_info=True,
                )
                self._warned = True

    def reset(self) -> None:
        self._policy.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM WebSocket Client")

    # Server connection
    parser.add_argument("--server-host", type=str, required=True, help="Policy server hostname/IP")
    parser.add_argument("--server-port", type=int, default=8000, help="Policy server port")

    # Control settings
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="Policy action rate (Hz), must match dataset fps",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=200.0,
        help="ROS2 joint command publish rate (Hz)",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable joint interpolation (publish raw targets at control-hz)",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=32,
        help="Number of actions to execute before querying policy again",
    )

    # Task settings
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task library key to condition on (from taskmap.json on the "
        "server). Omit to use whatever instruction the server was started with. This is "
        "NOT free-form text -- it must match a key the server loaded, or inference fails "
        "loudly rather than silently running the wrong tower height.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Free-form instruction, recorded in the observation but NOT used to pick the "
        "task. Use --task for that.",
    )

    # RTP settings
    parser.add_argument("--rtp-port", type=int, default=8890, help="RTP video stream port")
    parser.add_argument(
        "--decoder",
        type=str,
        default="nvh265dec",
        help="GStreamer H265 decoder",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip RTP video (debug only, cannot infer without images)",
    )

    # Robot settings
    parser.add_argument(
        "--swap-grippers",
        action="store_true",
        help="Swap left/right gripper commands (debug)",
    )
    parser.add_argument(
        "--min-grasp-hold-s",
        type=float,
        default=0.0,
        help="Minimum grasp hold duration (seconds, 0=follow policy)",
    )
    parser.add_argument(
        "--grasp-close-trigger",
        type=float,
        default=0.5,
        help="Trigger threshold to count as grasp close",
    )

    # Episode settings
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Maximum steps per episode (0=unlimited)",
    )

    # Safety
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Infer only, do not publish motion commands",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=2.0,
        help="Seconds to hold current pose before starting policy",
    )

    args = parser.parse_args()

    # Validation
    if args.control_hz <= 0:
        parser.error(f"--control-hz must be positive, got {args.control_hz}")
    if args.publish_hz <= 0:
        parser.error(f"--publish-hz must be positive, got {args.publish_hz}")
    if not args.no_interpolate and args.publish_hz < args.control_hz:
        parser.error(
            f"--publish-hz ({args.publish_hz}) must be >= --control-hz ({args.control_hz})"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )

    logger.info("=" * 70)
    logger.info("FastWAM TA2 WebSocket Client")
    logger.info("=" * 70)
    logger.info(f"Server: ws://{args.server_host}:{args.server_port}")
    logger.info(f"Control frequency: {args.control_hz} Hz")
    if not args.no_interpolate:
        logger.info(f"Joint interpolation: {args.publish_hz} Hz")
    else:
        logger.info("Joint interpolation: OFF (raw commands)")
    logger.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logger.info(f"Task: {args.task if args.task else '(server default)'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 70)

    # Import heavy dependencies after argument parsing
    try:
        from openpi_client import websocket_client_policy
        from openpi_client import action_chunk_broker
        logger.info("Using openpi_client package")
    except ImportError as exc:
        missing = exc.name or "unknown"
        logger.error(
            "Cannot import WebSocket client dependency %r with %s. Install with:\n"
            "  %s -m pip install --user msgpack websockets msgpack-numpy "
            "typing-extensions dm-tree",
            missing,
            sys.executable,
            sys.executable,
        )
        sys.exit(1)

    from fastwam_env import FastWAMEnvironment

    # Check server health
    import urllib.request
    server_url = f"http://{args.server_host}:{args.server_port}"
    try:
        with urllib.request.urlopen(f"{server_url}/healthz", timeout=5) as resp:
            logger.info(f"Server health check: {resp.read().decode().strip()}")
    except Exception as exc:
        logger.error(
            f"Cannot reach policy server {server_url} ({exc}). "
            "Start serve_policy_ws.py first."
        )
        sys.exit(1)

    # Create WebSocket client policy
    logger.info("Connecting to policy server...")
    ws_policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.server_host,
        port=args.server_port,
    )

    # Log server metadata
    metadata = ws_policy.get_server_metadata()
    logger.info(f"Connected! Server metadata: {metadata}")

    # Fail now, not mid-episode: the server raises on an unknown task name, but that
    # error would surface on the first infer -- after the arms have already been
    # enabled and held. The task library is in the metadata, so check it up front.
    available = metadata.get("available_tasks") or []
    if args.task:
        if available and args.task not in available:
            logger.error(
                "--task %r is not in the server's task library %s. "
                "Check taskmap.json on the server side.",
                args.task,
                sorted(available),
            )
            sys.exit(1)
        logger.info("Task %r found in server library", args.task)
    else:
        logger.warning(
            "No --task given: the server will use its startup instruction (%r). "
            "Pass --task <key> to select one from the server's task library.",
            metadata.get("active_task"),
        )

    # Create environment before wrapping the policy: ChunkRecordingPolicy needs to
    # reach the environment's bag bridge, and the broker must sit outermost.
    logger.info("Initializing FastWAM environment...")
    environment = FastWAMEnvironment(
        prompt=args.prompt,
        task=args.task,
        control_frequency=args.control_hz,
        interp_frequency=args.publish_hz,
        interpolate=not args.no_interpolate,
        rtp_port=args.rtp_port,
        rtp_decoder=args.decoder,
        enable_video=not args.skip_video,
        swap_grippers=args.swap_grippers,
        min_grasp_hold_s=args.min_grasp_hold_s,
        grasp_close_trigger=args.grasp_close_trigger,
    )

    # Resolve the bridge lazily: _ros_interface is assigned from the ROS spin thread,
    # so binding it now could capture None even though it becomes available later.
    def _bag_bridge():
        interface = environment._ros_interface
        return getattr(interface, "_bag_bridge", None) if interface else None

    # Wrap with action chunk broker. Recording goes *inside* the broker so it sees
    # one call per real inference with the full chunk.
    logger.info(f"Wrapping policy with ActionChunkBroker (horizon={args.open_loop_horizon})")
    chunked_policy = action_chunk_broker.ActionChunkBroker(
        policy=ChunkRecordingPolicy(ws_policy, _bag_bridge),
        action_horizon=args.open_loop_horizon,
    )

    # Create a simple policy agent (without using openpi_client.runtime for now)
    # This allows us to have more control over the loop
    logger.info("Starting control loop...")

    if args.dry_run:
        logger.warning("DRY RUN MODE - Actions will be logged but not executed")

    try:
        import time
        import numpy as np

        # Initial hold
        logger.info(f"Holding current pose for {args.hold_s:.1f}s...")
        if not args.dry_run:
            for _ in range(int(args.hold_s * args.control_hz)):
                environment._ros_interface.hold_current_joints()
                time.sleep(1.0 / args.control_hz)

        # Main control loop
        step = 0
        episode = 0
        dt = 1.0 / args.control_hz

        while episode < args.num_episodes:
            loop_start = time.time()

            # Get observation
            obs = environment.get_observation()

            # Get action from policy (ActionChunkBroker handles chunking)
            action_dict = chunked_policy.infer(obs)
            action = action_dict['actions']

            if args.dry_run:
                logger.info(
                    f"[DRY RUN] step={step} "
                    f"L_q0={action[0]:.3f} R_q0={action[8]:.3f} "
                    f"L_grip={action[7]:.3f} R_grip={action[15]:.3f}"
                )
            else:
                # Apply action
                environment.apply_action({'actions': action})

            step += 1

            # Check episode termination
            if args.max_episode_steps > 0 and step >= args.max_episode_steps:
                logger.info(f"Episode {episode} complete (max steps reached)")
                step = 0
                episode += 1
                if episode < args.num_episodes:
                    environment.reset()
                    chunked_policy.reset()

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nStopping (Ctrl+C pressed)...")
    finally:
        logger.info("Stopping robot...")
        if not args.dry_run and environment._ros_interface:
            environment._ros_interface.stop_command_stream()
            environment._ros_interface.publish_enable(0.0)
        environment.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
