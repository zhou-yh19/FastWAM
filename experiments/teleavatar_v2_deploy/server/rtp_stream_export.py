#!/usr/bin/env python3
"""Stream recorded FastWAM RTP JPEG topics to files and preview videos.

This exporter keeps only one synchronized camera group in memory. It writes the
original JPEG payload for each camera, so it is considerably lighter than an
offline-inference export that also stores decoded numpy tensors and proprioception.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


CAMS = ("head_camera", "left_color", "right_color")
TOPIC_TO_CAM = {f"/fastwam/observation/{cam}/compressed": cam for cam in CAMS}


def _stamp_ns(msg: object, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value else int(fallback_ns)


def _decode(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def _mosaic(images: Dict[str, Image.Image]) -> Image.Image:
    """Build the TA2 stereo canvas: head above two wrist views."""
    canvas = Image.new("RGB", (512, 384), (128, 128, 128))
    canvas.paste(_letterbox(images["head_camera"], 512, 256), (0, 0))
    canvas.paste(_letterbox(images["left_color"], 256, 80), (0, 256))
    canvas.paste(_letterbox(images["right_color"], 256, 80), (256, 256))
    return canvas


def _append_video(writer: Optional[object], frame: Image.Image) -> object:
    if writer is None:
        raise RuntimeError("video writer must be initialized before appending")
    writer.append_data(np.asarray(frame, dtype=np.uint8))
    return writer


def export_rtp_streaming(bag: Path, out_dir: Path, fps: float = 20.0) -> dict:
    """Export camera topics from one bag without retaining the whole bag in RAM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_root = out_dir / "camera_frames"
    for cam in CAMS:
        (frame_root / cam).mkdir(parents=True, exist_ok=True)

    writers: Dict[str, object] = {}
    pending: Dict[int, Dict[str, object]] = {}
    frame_counts = {cam: 0 for cam in CAMS}
    rows: List[dict] = []
    contact: List[Image.Image] = []
    first_ns: Optional[int] = None
    stamp_history: List[int] = []

    def ensure_writer(name: str) -> object:
        writer = writers.get(name)
        if writer is None:
            writer = imageio.get_writer(
                str(out_dir / f"recorded_{name}_full.mp4"),
                fps=max(1, round(fps)),
                codec="libx264",
                format="FFMPEG",
                pixelformat="yuv420p",
            )
            writers[name] = writer
        return writer

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    try:
        with AnyReader([bag], default_typestore=typestore) as reader:
            for conn, t_ns, raw in reader.messages():
                cam = TOPIC_TO_CAM.get(conn.topic)
                if cam is None:
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                payload = bytes(msg.data)
                stamp_ns = _stamp_ns(msg, int(t_ns))
                if first_ns is None:
                    first_ns = stamp_ns
                stamp_history.append(stamp_ns)

                index = frame_counts[cam]
                rel = Path("camera_frames") / cam / f"frame_{index:06d}.jpg"
                (out_dir / rel).write_bytes(payload)
                frame_counts[cam] += 1
                image = _decode(payload)
                _append_video(ensure_writer(cam), image)

                group = pending.setdefault(stamp_ns, {})
                group[cam] = payload
                group[f"{cam}_path"] = str(rel)
                if all(name in group for name in CAMS):
                    frame_id = len(rows)
                    decoded = {name: _decode(group[name]) for name in CAMS}  # type: ignore[arg-type]
                    mosaic = _mosaic(decoded)
                    if len(contact) < 12:
                        contact.append(mosaic.copy())
                    _append_video(ensure_writer("obs_mosaic"), mosaic)
                    rows.append(
                        {
                            "id": frame_id,
                            "stamp_ns": stamp_ns,
                            "t_sec": ((stamp_ns - first_ns) / 1e9) if first_ns else 0.0,
                            "camera_paths": {
                                name: str(group[f"{name}_path"])
                                for name in CAMS
                            },
                        }
                    )
                    del pending[stamp_ns]
    finally:
        for writer in writers.values():
            writer.close()

    if contact:
        cols = min(4, len(contact))
        tile_w, tile_h = 256, 192
        sheet = Image.new("RGB", (cols * tile_w, ((len(contact) + cols - 1) // cols) * tile_h), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)
        for index, image in enumerate(contact):
            tile = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            x = (index % cols) * tile_w
            y = (index // cols) * tile_h
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + 4), f"{index:04d}", fill=(255, 255, 255))
        sheet.save(out_dir / "viz_obs_mosaic_sheet.jpg", quality=90)

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "frame_index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "stamp_ns", "t_sec", *CAMS])
        for row in rows:
            writer.writerow([row["id"], row["stamp_ns"], row["t_sec"], *[row["camera_paths"][cam] for cam in CAMS]])

    if len(stamp_history) > 1:
        dt = np.diff(np.asarray(sorted(stamp_history), dtype=np.float64)) / 1e9
        valid = dt[(dt > 0.001) & (dt < 2.0)]
        measured_fps = float(1.0 / np.median(valid)) if len(valid) else float(fps)
    else:
        measured_fps = float(fps)
    meta = {
        "bag": str(bag),
        "cameras": list(CAMS),
        "camera_frame_counts": frame_counts,
        "synchronized_rtp_samples": len(rows),
        "fps": measured_fps,
        "format": "original JPEG payloads in camera_frames plus H264 preview videos",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_rtp_streaming(args.bag, args.out), indent=2))
