#!/usr/bin/env python3
"""Reconstruct the operation footage from a FastWAM TA2 deploy rosbag.

Decodes /fastwam/observation/*/compressed back into video. Playback is placed on
the recording's real time axis: output is constant-rate, and each output frame
holds the newest frame whose stamp has arrived, so the video runs for the same
wall-clock duration as the episode and recording stalls show up as visible
freezes instead of being silently sped up.

Needs no ROS install and no OpenCV (rosbags + Pillow + PyAV).

Example:
  python experiments/teleavatar_v2_deploy/server/bag_to_video.py \\
    --bag fastwam_bags/fastwam_ta2_20260908_182846_0.db3 --individual
"""

from __future__ import annotations

import argparse
import bisect
import io
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

CAMERA_TOPICS: Dict[str, str] = {
    "/fastwam/observation/head_camera/compressed": "head_camera",
    "/fastwam/observation/left_color/compressed": "left_color",
    "/fastwam/observation/right_color/compressed": "right_color",
}
# Left to right in the mosaic: scene view first, then the two wrists.
PANEL_ORDER = ("head_camera", "left_color", "right_color")
REPLAN_TOPIC = "/fastwam/policy/action_chunk"
FSM_TOPIC = "/api/fsm/enable"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> Any:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def read_frames(
    bag: Path,
) -> Tuple[Dict[str, List[Tuple[float, bytes]]], List[float], List[Tuple[float, float]]]:
    """Pull JPEG payloads plus replan/FSM marks, all on one epoch time axis.

    Camera frames are keyed by header stamp (capture time). Replans are keyed by
    bag write time, which is when the chunk was actually published.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    frames: Dict[str, List[Tuple[float, bytes]]] = {c: [] for c in CAMERA_TOPICS.values()}
    replans: List[float] = []
    fsm: List[Tuple[float, float]] = []

    with AnyReader([bag], default_typestore=typestore) as reader:
        for conn, t_ns, raw in reader.messages():
            camera = CAMERA_TOPICS.get(conn.topic)
            if camera is not None:
                msg = reader.deserialize(raw, conn.msgtype)
                stamp = msg.header.stamp
                epoch = int(stamp.sec) + int(stamp.nanosec) / 1e9
                if epoch <= 0.0:  # unset header stamp: fall back to bag time
                    epoch = int(t_ns) / 1e9
                frames[camera].append((epoch, bytes(msg.data)))
            elif conn.topic == REPLAN_TOPIC:
                replans.append(int(t_ns) / 1e9)
            elif conn.topic == FSM_TOPIC:
                msg = reader.deserialize(raw, conn.msgtype)
                fsm.append((int(t_ns) / 1e9, float(msg.data)))

    for series in frames.values():
        series.sort(key=lambda x: x[0])
    replans.sort()
    fsm.sort(key=lambda x: x[0])
    return frames, replans, fsm


def panel_sizes(
    sample: Dict[str, Image.Image],
    cameras: Sequence[str],
    height: int,
) -> Dict[str, Tuple[int, int]]:
    """Scale every panel to a common height, keeping each camera's aspect ratio."""
    sizes: Dict[str, Tuple[int, int]] = {}
    for camera in cameras:
        img = sample[camera]
        w = max(2, int(round(img.width * height / img.height)))
        sizes[camera] = (w - (w % 2), height)  # even width keeps x264 happy
    return sizes


def build_schedule(
    stamps: Sequence[float],
    fps: float,
    t_start: float,
    t_end: float,
) -> List[int]:
    """Constant-rate output schedule: index of the newest frame at each output tick.

    Holding the previous frame across a gap is what makes a recording stall read
    as a freeze, and keeps the video's duration equal to the episode's.
    """
    n_out = max(1, int(round((t_end - t_start) * fps)) + 1)
    schedule: List[int] = []
    for k in range(n_out):
        t = t_start + k / fps
        idx = bisect.bisect_right(stamps, t) - 1
        schedule.append(max(0, idx))
    return schedule


def draw_overlay(
    canvas: Image.Image,
    *,
    strip_h: int,
    panels: Sequence[Tuple[str, int]],
    t_rel: float,
    duration: float,
    src_index: int,
    n_src: int,
    held: bool,
    replan_active: bool,
    fsm_off: bool,
    font: Any,
    font_small: Any,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, strip_h - 1], fill=(16, 16, 20))
    draw.text(
        (8, strip_h // 2),
        f"t={t_rel:6.2f}s / {duration:.2f}s   frame {src_index + 1}/{n_src}",
        font=font,
        fill=(235, 235, 235),
        anchor="lm",
    )

    x = canvas.width - 8
    if fsm_off:
        draw.text((x, strip_h // 2), "FSM OFF", font=font, fill=(255, 96, 96), anchor="rm")
        x -= 95
    if replan_active:
        draw.text((x, strip_h // 2), "REPLAN", font=font, fill=(120, 220, 255), anchor="rm")
        x -= 90
    if held:
        # No new frame arrived for this tick -- the recording, not the robot, stalled.
        draw.text((x, strip_h // 2), "REC STALL", font=font, fill=(255, 190, 60), anchor="rm")

    for label, x0 in panels:
        draw.text(
            (x0 + 6, strip_h + 5),
            label,
            font=font_small,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


class Encoder:
    """Constant-rate H.264 writer."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        self.container = av.open(str(path), mode="w")
        rate = Fraction(fps).limit_denominator(1000)
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "veryfast"}

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def render(
    bag: Path,
    out: Path,
    *,
    fps: float = 20.0,
    height: int = 384,
    crf: int = 23,
    overlay: bool = True,
    individual: bool = False,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> None:
    print(f"reading {bag} ...", flush=True)
    frames, replans, fsm = read_frames(bag)
    present = [c for c in PANEL_ORDER if frames.get(c)]
    if not present:
        raise SystemExit(f"no camera frames in bag: {bag}")

    counts = {c: len(frames[c]) for c in present}
    if len(set(counts.values())) != 1:
        print(f"[warn] frame counts differ per camera: {counts}")

    t0_all = min(frames[c][0][0] for c in present)
    t1_all = max(frames[c][-1][0] for c in present)
    t_start = t0_all if start is None else t0_all + float(start)
    t_end = t1_all if end is None else min(t1_all, t0_all + float(end))
    if t_end <= t_start:
        raise SystemExit(f"empty time range: start={start} end={end}")
    duration = t_end - t_start
    print(
        f"cameras={present}  frames={counts}  "
        f"episode={t1_all - t0_all:.2f}s  rendering={duration:.2f}s @ {fps:g} fps"
    )

    stamps = {c: [s for s, _ in frames[c]] for c in present}
    schedules = {c: build_schedule(stamps[c], fps, t_start, t_end) for c in present}
    n_out = min(len(schedules[c]) for c in present)

    sample = {c: Image.open(io.BytesIO(frames[c][0][1])) for c in present}
    sizes = panel_sizes(sample, present, height)
    strip_h = 30 if overlay else 0
    offsets: Dict[str, int] = {}
    x = 0
    for camera in present:
        offsets[camera] = x
        x += sizes[camera][0]
    mosaic_w = x + (x % 2)
    mosaic_h = height + strip_h
    mosaic_h += mosaic_h % 2

    font = _load_font(17)
    font_small = _load_font(14)
    panel_labels = [(c, offsets[c]) for c in present]

    out.mkdir(parents=True, exist_ok=True)
    mosaic_path = out / "video_3cam.mp4"
    print(f"  mosaic {mosaic_w}x{mosaic_h} -> {mosaic_path.name}")
    mosaic_enc = Encoder(mosaic_path, mosaic_w, mosaic_h, fps, crf)
    solo_enc: Dict[str, Encoder] = {}
    if individual:
        for camera in present:
            w, h = sizes[camera]
            path = out / f"video_{camera}.mp4"
            print(f"  solo   {w}x{h} -> {path.name}")
            solo_enc[camera] = Encoder(path, w, h, fps, crf)

    replan_hold = 0.25  # seconds the REPLAN badge stays lit after a chunk arrives
    decoded: Dict[str, Tuple[int, Image.Image]] = {}
    prev_index: Dict[str, int] = {}
    held_total = 0
    t_render = time.time()

    for k in range(n_out):
        t_abs = t_start + k / fps
        t_rel = t_abs - t0_all
        canvas = Image.new("RGB", (mosaic_w, mosaic_h), (16, 16, 20))
        held_any = False

        for camera in present:
            idx = schedules[camera][k]
            cached = decoded.get(camera)
            if cached is None or cached[0] != idx:
                img = Image.open(io.BytesIO(frames[camera][idx][1])).convert("RGB")
                if img.size != sizes[camera]:
                    img = img.resize(sizes[camera], Image.BILINEAR)
                decoded[camera] = (idx, img)
            else:
                img = cached[1]
            if prev_index.get(camera) == idx:
                held_any = True
            prev_index[camera] = idx
            canvas.paste(img, (offsets[camera], strip_h))
            if camera in solo_enc:
                solo_enc[camera].write(np.asarray(img))

        if held_any:
            held_total += 1

        if overlay:
            ri = bisect.bisect_right(replans, t_abs) - 1
            replan_active = ri >= 0 and (t_abs - replans[ri]) <= replan_hold
            fi = bisect.bisect_right([t for t, _ in fsm], t_abs) - 1
            fsm_off = fi >= 0 and fsm[fi][1] <= 0.5
            ref = present[0]
            draw_overlay(
                canvas,
                strip_h=strip_h,
                panels=panel_labels,
                t_rel=t_rel,
                duration=t1_all - t0_all,
                src_index=schedules[ref][k],
                n_src=counts[ref],
                held=held_any,
                replan_active=replan_active,
                fsm_off=fsm_off,
                font=font,
                font_small=font_small,
            )

        mosaic_enc.write(np.asarray(canvas))
        if (k + 1) % 200 == 0 or k + 1 == n_out:
            print(f"    {k + 1}/{n_out} frames", flush=True)

    mosaic_enc.close()
    for enc in solo_enc.values():
        enc.close()

    print(
        f"done in {time.time() - t_render:.1f}s: {n_out} output frames, "
        f"{held_total} held ({held_total / n_out * 100:.1f}% -- recording gaps), "
        f"{len(replans)} replans"
    )
    for path in sorted(out.glob("video_*.mp4")):
        print(f"  {path}  {path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag2 dir or .db3 file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: <bag>_analysis next to the bag)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="output frame rate; playback stays real-time regardless (default 20)",
    )
    parser.add_argument("--height", type=int, default=384, help="panel height in px")
    parser.add_argument("--crf", type=int, default=23, help="x264 quality, lower = better")
    parser.add_argument("--no-overlay", action="store_true", help="omit the info strip")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="also write one mp4 per camera (no overlay)",
    )
    parser.add_argument("--start", type=float, default=None, help="trim start (s from episode t0)")
    parser.add_argument("--end", type=float, default=None, help="trim end (s from episode t0)")
    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    out = args.out.resolve() if args.out else bag.parent / f"{bag.name}_analysis"
    render(
        bag,
        out,
        fps=args.fps,
        height=args.height,
        crf=args.crf,
        overlay=not args.no_overlay,
        individual=args.individual,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
