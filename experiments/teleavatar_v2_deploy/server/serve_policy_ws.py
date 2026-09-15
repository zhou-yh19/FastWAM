#!/usr/bin/env python3
"""FastWAM TA2 WebSocket policy server.

WebSocket-based policy server using OpenPI protocol (msgpack + numpy).
Provides better performance than HTTP + JSON + base64.

Example:
    conda activate fastwam
    cd <FastWAM repo root>
    export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
    python experiments/teleavatar_v2_deploy/server/serve_policy_ws.py \\
        --task <config under configs/task/, as trained> \\
        --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_NNNNNN.pt \\
        --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import websockets.asyncio.server as _server
import websockets.frames

# Add project root to path. Must match serve_policy.py: parents[3] is the repo root
# holding configs/ and src/ -- hydra composes from configs/, so an off-by-one here
# fails at initialize_config_dir rather than anywhere obvious.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import after path setup
from fastwam_policy_wrapper import FastWAMPolicyWrapper

logger = logging.getLogger("fastwam_ws_server")


# Lazy import to avoid loading heavy dependencies before argument parsing
def _lazy_import_msgpack():
    """Lazy import msgpack_numpy to avoid early dependency loading."""
    global msgpack_numpy
    try:
        # Try openpi_client package first
        from openpi_client import msgpack_numpy
    except ImportError:
        # Fallback to local copy
        logger.warning("openpi_client not found, using msgpack-numpy directly")
        import msgpack
        import msgpack_numpy as _msgpack_numpy
        _msgpack_numpy.patch()
        msgpack_numpy = _msgpack_numpy
    return msgpack_numpy


def _lazy_import_policy():
    """Lazy import FastWAMTA2Policy to avoid loading PyTorch before arguments are parsed."""
    # Import the existing serve_policy module to reuse FastWAMTA2Policy
    sys.path.insert(0, str(Path(__file__).parent))
    from serve_policy import FastWAMTA2Policy
    return FastWAMTA2Policy


class WebsocketPolicyServer:
    """WebSocket server for FastWAM policy inference.

    Protocol compatible with OpenPI WebsocketClientPolicy.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Run the server (blocking)."""
        asyncio.run(self.run())

    async def run(self):
        """Async server main loop."""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            addr = server.sockets[0].getsockname() if server.sockets else (self._host, self._port)
            logger.info(f"WebSocket server listening on ws://{addr[0]}:{addr[1]}")
            logger.info(f"Health check available at http://{addr[0]}:{addr[1]}/healthz")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """Handle WebSocket connection."""
        logger.info(f"Connection from {websocket.remote_address} opened")

        msgpack_numpy = _lazy_import_msgpack()
        packer = msgpack_numpy.Packer()

        # Send metadata on connection
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # Receive observation
                raw_data = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw_data)

                # Run inference
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Add timing info. The wrapper already put the per-stage breakdown
                # (decode/image/model/denorm) here, so merge rather than assign --
                # overwriting it left the client with a single opaque number and no way
                # to tell a slow GPU from a slow JPEG decode.
                timing = dict(action.get("server_timing") or {})
                timing["infer_ms"] = infer_time * 1000
                if prev_total_time is not None:
                    timing["prev_total_ms"] = prev_total_time * 1000
                action["server_timing"] = timing

                # Send action
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.exception("Error in inference handler")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> Optional[_server.Response]:
    """Health check endpoint handler."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM WebSocket Policy Server")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-stats", type=Path, required=True, help="Path to dataset stats JSON")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task config name under configs/task/ (without .yaml). Must be the config the "
             "checkpoint was trained with -- hydra recomposes data/model dims from it, so a "
             "mismatch fails at load time.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--action-horizon", type=int, default=32, help="Action chunk length")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Dummy inferences to run after loading",
    )
    parser.add_argument(
        "--num-video-frames",
        type=int,
        default=None,
        help="Joint video length (default: from task config)",
    )
    parser.add_argument(
        "--prompt-task",
        type=str,
        default=None,
        help="Instruction to condition on (default: from dataset meta/tasks.jsonl)",
    )
    parser.add_argument(
        "--text-embed",
        type=Path,
        default=None,
        help="Precomputed T5 context .pt file",
    )
    parser.add_argument(
        "--task-map",
        type=Path,
        default=None,
        help="JSON mapping task names to instructions or hashes",
    )
    parser.add_argument(
        "--load-text-encoder",
        action="store_true",
        help="Load T5 encoder (11 GiB) instead of using cached embeddings",
    )
    parser.add_argument(
        "--no-compile-denoise",
        action="store_true",
        help="Disable torch.compile for action denoising",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        nargs="*",
        default=None,
        help="num_inference_steps values for CUDA Graph warmup",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )

    logger.info("=" * 70)
    logger.info("FastWAM WebSocket Policy Server")
    logger.info("=" * 70)
    logger.info(f"Protocol: WebSocket + msgpack_numpy")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Server: ws://{args.host}:{args.port}")
    logger.info("=" * 70)

    # Import and initialize policy (heavy dependencies loaded here)
    logger.info("Loading FastWAMTA2Policy...")
    FastWAMTA2Policy = _lazy_import_policy()

    policy = FastWAMTA2Policy(
        checkpoint=args.checkpoint.expanduser().resolve(),
        dataset_stats=args.dataset_stats.expanduser().resolve(),
        task_name=args.task,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        num_video_frames=args.num_video_frames,
        prompt_task=args.prompt_task,
        text_embed=args.text_embed,
        load_text_encoder=args.load_text_encoder,
        task_map=args.task_map,
        compile_denoise=not args.no_compile_denoise,
        warmup_steps=tuple(args.warmup_steps or ()),
    )

    logger.info("Running warmup iterations...")
    policy.warmup(args.warmup_iters)
    logger.info("Warmup complete")

    # Wrap policy for OpenPI compatibility
    wrapped_policy = FastWAMPolicyWrapper(policy)

    # Build metadata
    metadata = {
        "task": args.task,
        "action_horizon": args.action_horizon,
        "action_dim": 16,
        "num_inference_steps": args.num_inference_steps,
        "device": args.device,
        "available_tasks": list(policy.task_library.keys()) if hasattr(policy, 'task_library') else [],
        "active_task": policy.active_task if hasattr(policy, 'active_task') else None,
    }

    # Create and run server
    server = WebsocketPolicyServer(
        policy=wrapped_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )

    logger.info("Starting WebSocket server...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)")


if __name__ == "__main__":
    main()
