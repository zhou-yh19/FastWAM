#!/usr/bin/env python3
"""Check WebSocket server connectivity and message format.

    python test_websocket_connection.py --host <gpu-host> --port 8000

Sends one synthetic observation; never touches ROS 2, so no arm moves.
"""

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_connection(host: str, port: int) -> bool:
    """Connect, fetch metadata, run one inference."""

    try:
        # Dependencies live in the client interpreter, not conda
        try:
            from openpi_client import websocket_client_policy
            from openpi_client import msgpack_numpy
            logger.info("[ok] openpi_client available")
        except ImportError:
            logger.error("[x] openpi_client not installed")
            logger.error("    pip install websockets msgpack-numpy typing-extensions dm-tree")
            return False

        logger.info(f"connecting to ws://{host}:{port} ...")
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )
        logger.info("[ok] connected")

        # Metadata arrives on connect
        metadata = policy.get_server_metadata()
        logger.info(f"[ok] server metadata: {metadata}")

        logger.info("building a synthetic observation ...")
        obs = {
            'observation/state': np.zeros(72, dtype=np.float32),
            'observation/images/head_camera': np.zeros((256, 512, 3), dtype=np.uint8),
            'observation/images/left_color': np.zeros((80, 256, 3), dtype=np.uint8),
            'observation/images/right_color': np.zeros((80, 256, 3), dtype=np.uint8),
            'prompt': 'test',
        }
        logger.info("[ok] observation built")

        logger.info("sending inference request ...")
        start_time = time.time()
        result = policy.infer(obs)
        elapsed = time.time() - start_time
        logger.info(f"[ok] inference returned in {elapsed*1000:.1f} ms")

        # Validate the response
        if 'actions' not in result:
            logger.error(f"[x] response has no 'actions' key: {result.keys()}")
            return False

        actions = result['actions']
        logger.info(f"[ok] actions shape: {actions.shape}")
        logger.info(f"[ok] actions dtype: {actions.dtype}")
        logger.info(f"[ok] actions range: [{actions.min():.3f}, {actions.max():.3f}]")

        # Per-stage server timing, when reported
        if 'server_timing' in result:
            timing = result['server_timing']
            logger.info(f"[ok] server timing: {timing}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("[ok] all checks passed")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"[x] failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Check WebSocket server connectivity")
    parser.add_argument("--host", type=str, required=True, help="server hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="server port")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WebSocket connection test")
    logger.info("=" * 60)
    logger.info(f"target: ws://{args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("")

    success = test_connection(args.host, args.port)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
