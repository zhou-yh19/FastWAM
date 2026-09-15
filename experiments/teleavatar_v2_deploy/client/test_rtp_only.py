#!/usr/bin/env python3
"""Isolate RTP/GStreamer crash without ROS2."""

from __future__ import annotations

import argparse
import time

from rtp_video_interface import RTPH265VideoInterface


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--decoder", type=str, default="avdec_h265")
    p.add_argument("--timeout", type=float, default=15.0)
    args = p.parse_args()

    iface = None
    try:
        print(f"[test] create interface decoder={args.decoder!r}", flush=True)
        iface = RTPH265VideoInterface(port=args.port, decoder=args.decoder)
        print("[test] start()", flush=True)
        iface.start()
        print("[test] waiting for first frame...", flush=True)
        ok = iface.wait_for_initial_data(timeout=args.timeout)
        print(f"[test] first_frame={ok}", flush=True)
        if ok:
            imgs = iface.get_latest_images()
            for k, v in imgs.items():
                print(f"  {k}: {None if v is None else v.shape}", flush=True)
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("[test] interrupted", flush=True)
    finally:
        if iface is not None:
            iface.stop()
        print("[test] done", flush=True)


if __name__ == "__main__":
    main()
