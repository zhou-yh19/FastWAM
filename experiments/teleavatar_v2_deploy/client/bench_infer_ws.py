#!/usr/bin/env python3
"""Measure obs -> action latency over the WebSocket deploy path.

`bench_infer_rtt.py` benchmarks the HTTP server (`POST /infer`). The WebSocket path
is a different transport (msgpack over a persistent connection) with a different
serialization cost and no per-request connection setup, so its numbers do not carry
over. This script drives `serve_policy_ws.py` through the same `openpi_client`
stack `run_client_ws.py` uses, on the same interpreter, so what it reports is what
the control loop will actually see.

Run it against an already-running server (./start_local_serve_ws.sh), with the robot
idle -- it sends synthetic frames and never touches ROS2, so no arm ever moves.

    ./bench_infer_ws.py                      # 30 reps, mono geometry, server default task
    ./bench_infer_ws.py --task <taskmap-key> -n 100
    ./bench_infer_ws.py --image-dir <deploy_logs/run_*/images/infer_000000>

The request can select `--num-inference-steps`. Start the server with
`--warmup-steps 8 10 12` before comparing those values so every CUDA Graph is warm.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Single-eye (left) RTP frame sizes, i.e. what ros2_interface._POLICY_TO_RTP now hands
# the client. The server letterboxes these down to the mono tiles (256x256 / 80x128),
# so sending anything else would measure a different resize than deployment does.
MONO_SHAPES = {
    "head_camera": (960, 960, 3),
    "left_color": (400, 640, 3),
    "right_color": (400, 640, 3),
}
# Pre-mono SBS sizes, for comparing against the old stereo checkpoint.
STEREO_SHAPES = {
    "head_camera": (960, 1920, 3),
    "left_color": (400, 1280, 3),
    "right_color": (400, 1280, 3),
}


def synth_images(shapes: dict[str, tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """Smooth checkerboard frames -- compress like a real scene, unlike random noise."""
    rng = np.random.default_rng(0)
    out = {}
    for key, hw in shapes.items():
        yy, xx = np.mgrid[0 : hw[0], 0 : hw[1]]
        base = ((xx // 16 + yy // 16) % 2) * 40 + 80
        img = np.stack([base, base + 10, base + 20], axis=-1).astype(np.int16)
        img += rng.integers(-8, 9, size=hw, dtype=np.int16)
        out[key] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def load_images(image_dir: Path) -> dict[str, np.ndarray]:
    from PIL import Image

    out = {}
    for key in MONO_SHAPES:
        path = image_dir / f"{key}.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        out[key] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return out


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile; avoids numpy's interpolation on tiny samples."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def summarize(name: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"{name:24} (no samples)"
    return (
        f"{name:24} mean {statistics.fmean(values):7.1f}  p50 {pct(values, 50):7.1f}  "
        f"p95 {pct(values, 95):7.1f}  p99 {pct(values, 99):7.1f}  "
        f"max {max(values):7.1f}  {unit}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("-n", "--repeats", type=int, default=30, help="measured requests")
    p.add_argument("--warmup", type=int, default=3, help="discarded requests first")
    p.add_argument(
        "--task",
        default=None,
        help="taskmap key to condition on; omit to use the server's startup instruction",
    )
    p.add_argument("--num-inference-steps", type=int, default=10,
                   help="denoising steps requested from the server")
    p.add_argument("--geometry", choices=("mono", "stereo"), default="mono",
                   help="synthetic frame sizes; mono = single left eye (current deploy)")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="load head_camera/left_color/right_color .jpg instead of synthesizing")
    p.add_argument("--state-dim", type=int, choices=(48, 72), default=72)
    p.add_argument("--control-hz", type=float, default=20.0, help="for the budget report")
    p.add_argument("--open-loop-horizon", type=int, default=20, help="for the budget report")
    p.add_argument("--csv", type=Path, default=None, help="write per-request rows here")
    args = p.parse_args(argv)

    try:
        import websockets.sync.client
        from openpi_client import msgpack_numpy
    except ImportError:
        print("websockets/msgpack/openpi_client missing. Run with the fastwam interpreter:\n"
              "  python bench_infer_ws.py ...",
              file=sys.stderr)
        return 1

    images = load_images(args.image_dir) if args.image_dir else synth_images(
        MONO_SHAPES if args.geometry == "mono" else STEREO_SHAPES
    )
    obs = {f"observation/images/{k}": v for k, v in images.items()}
    obs["observation/state"] = np.zeros(args.state_dim, dtype=np.float32)
    # Omit the key entirely when no task was given: sending task=None would ask the server
    # to look up a taskmap entry named "None" instead of falling back to its startup prompt.
    if args.task is not None:
        obs["task"] = args.task
    obs["num_inference_steps"] = args.num_inference_steps

    payload_mb = sum(v.nbytes for v in images.values()) / 1e6
    print("=" * 74)
    print(f"WS latency  {args.server_host}:{args.server_port}  task={args.task}  "
          f"denoise_steps={args.num_inference_steps}")
    for key, img in images.items():
        print(f"  {key:14} {img.shape[1]}x{img.shape[0]}")
    print(f"  raw image payload  {payload_mb:.2f} MB/obs")
    print("=" * 74)

    uri = f"ws://{args.server_host}:{args.server_port}"
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
    meta = msgpack_numpy.unpackb(ws.recv())
    packer = msgpack_numpy.Packer()
    available = meta.get("available_tasks") or []
    if args.task is not None and available and args.task not in available:
        print(f"task {args.task!r} not served. available: {available}", file=sys.stderr)
        return 1
    print(f"server metadata: {meta}\n")

    rtt, pack_ms, send_ms, recv_ms, unpack_ms = [], [], [], [], []
    server_infer, overhead = [], []
    detail: dict[str, list[float]] = {}
    rows = []

    total = args.warmup + args.repeats
    for i in range(total):
        t0 = time.perf_counter_ns()
        payload = packer.pack(obs)
        t1 = time.perf_counter_ns()
        ws.send(payload)
        t2 = time.perf_counter_ns()
        response = ws.recv()
        t3 = time.perf_counter_ns()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        t4 = time.perf_counter_ns()

        actual_steps = int(result.get("num_inference_steps", -1))
        if actual_steps != args.num_inference_steps:
            raise RuntimeError(
                f"server used num_inference_steps={actual_steps}, requested "
                f"{args.num_inference_steps}; restart with the updated WebSocket server"
            )

        phases = [(t1 - t0) / 1e6, (t2 - t1) / 1e6,
                  (t3 - t2) / 1e6, (t4 - t3) / 1e6]
        rtt_ms = (t4 - t0) / 1e6

        timing = result.get("server_timing") or {}
        infer_ms = float(timing.get("infer_ms", 0.0))
        measured = i >= args.warmup
        tag = "warmup" if not measured else f"{i - args.warmup + 1}/{args.repeats}"
        print(f"  [{tag:>7}] total {rtt_ms:7.1f}  pack {phases[0]:6.1f}  "
              f"send {phases[1]:6.1f}  recv {phases[2]:7.1f}  unpack {phases[3]:5.1f}  "
              f"server {infer_ms:7.1f} ms")
        if not measured:
            continue

        rtt.append(rtt_ms)
        pack_ms.append(phases[0])
        send_ms.append(phases[1])
        recv_ms.append(phases[2])
        unpack_ms.append(phases[3])
        server_infer.append(infer_ms)
        # Everything the server did not spend inside infer(): msgpack encode/decode on
        # both ends, the socket itself, and any queueing.
        overhead.append(rtt_ms - infer_ms)
        for key, value in timing.items():
            if key != "infer_ms" and isinstance(value, (int, float)):
                detail.setdefault(key, []).append(float(value))
        rows.append((rtt_ms, *phases, infer_ms, rtt_ms - infer_ms))

    actions = np.asarray(result["actions"])
    print(f"\naction chunk: {actions.shape}  (horizon x dim)")
    print("-" * 74)
    print(summarize("end-to-end RTT", rtt))
    print(summarize("client msgpack pack", pack_ms))
    print(summarize("client socket send", send_ms))
    print(summarize("client wait response", recv_ms))
    print(summarize("client msgpack unpack", unpack_ms))
    print(summarize("server infer", server_infer))
    print(summarize("transport+encode", overhead))
    for key in sorted(detail):
        print(summarize(f"  server {key}", detail[key]))
    print("-" * 74)

    # What the control loop actually feels. run_client_ws.py infers synchronously, so
    # a blocking inference stalls the loop for that long every open_loop_horizon steps.
    period_ms = 1000.0 / args.control_hz
    p99 = pct(rtt, 99)
    steps = p99 / period_ms
    replan_s = args.open_loop_horizon / args.control_hz
    print(f"control period      {period_ms:.1f} ms  ({args.control_hz:.0f} Hz)")
    print(f"replan interval     {replan_s:.2f} s  (every {args.open_loop_horizon} steps)")
    print(f"p99 RTT consumes    {steps:.1f} control steps "
          f"({p99 / (replan_s * 1000) * 100:.1f}% of the replan window)")
    if p99 > replan_s * 1000:
        print("  !! inference is slower than the replan window -- the loop can never keep up")
    elif steps > 1.0:
        print("  note: inference blocks the loop for more than one control step; the 200 Hz\n"
              "        interp timer keeps publishing meanwhile, so the arm holds, not stalls")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as fh:
            fh.write("task,num_inference_steps,rtt_ms,pack_ms,send_ms,recv_wait_ms,unpack_ms,server_infer_ms,overhead_ms\n")
            for row in rows:
                values = ",".join(f"{value:.3f}" for value in row)
                fh.write(f"{args.task},{args.num_inference_steps},{values}\n")
        print(f"\nwrote {args.csv}")
    ws.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
