#!/usr/bin/env python3
"""Give a transcoded dataset its own meta/ carrying the real (downscaled) video size.

For speed, transcode.sh symlinks meta/ back to the source dataset -- but that
info.json still advertises the source resolution, and LeRobot reads those fields.
This copies it and rewrites them. data/ stays a symlink: transcoding does not touch
the parquet files.

Usage:
    python scripts/patch_transcoded_meta.py --mode lowres data/<dataset>_lowres
    python scripts/patch_transcoded_meta.py --mode mono   data/<dataset>_mono
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# Target geometry, keyed by LeRobot feature name, as (height, width).
# Must stay in lockstep with filter_for() in transcode.sh.
EXPECTED_BY_MODE = {
    "lowres": {                                     # downscaled side-by-side stereo
        "observation.images.head_camera": (256, 512),
        "observation.images.left_color": (80, 256),
        "observation.images.right_color": (80, 256),
    },
    "mono": {                                       # left eye only
        "observation.images.head_camera": (256, 256),
        "observation.images.left_color": (80, 128),
        "observation.images.right_color": (80, 128),
    },
}


def probe(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = (int(x) for x in out.split("x")[:2])
    return h, w


def patch_dataset(ds_dir: Path, mode: str, dry_run: bool = False) -> None:
    EXPECTED = EXPECTED_BY_MODE[mode]
    meta = ds_dir / "meta"
    src_meta = Path(meta.resolve()) if meta.is_symlink() else meta
    if not src_meta.is_dir():
        raise FileNotFoundError(f"no meta/ to read for {ds_dir}")

    info_path = src_meta / "info.json"
    info = json.loads(info_path.read_text())

    changed = []
    for key, (exp_h, exp_w) in EXPECTED.items():
        feat = info.get("features", {}).get(key)
        if feat is None:
            continue
        # Cross-check against a real transcoded file rather than trusting the table.
        sample = next((ds_dir / "videos").rglob(f"*{key.split('.')[-1]}/*.mp4"), None)
        if sample is not None:
            got_h, got_w = probe(sample)
            if (got_h, got_w) != (exp_h, exp_w):
                raise ValueError(
                    f"{ds_dir.name}/{key}: transcoded file is {got_w}x{got_h}, "
                    f"expected {exp_w}x{exp_h}. Refusing to write a wrong info.json."
                )
        feat["shape"] = [exp_h, exp_w, 3]
        vinfo = feat.setdefault("info", {})
        vinfo["video.height"], vinfo["video.width"] = exp_h, exp_w
        vinfo["video.codec"] = "h264"          # transcode re-encodes with libx264
        vinfo["video.pix_fmt"] = "yuv420p"
        changed.append(f"{key} -> {exp_w}x{exp_h}")

    if dry_run:
        print(f"[dry-run] {ds_dir.name}: " + "; ".join(changed))
        return

    # Replace the symlink with a real directory so the patched info.json cannot leak
    # back into the source dataset.
    if meta.is_symlink():
        meta.unlink()
    if not meta.exists():
        shutil.copytree(src_meta, meta)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    print(f"{ds_dir.name}: " + "; ".join(changed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--mode", choices=sorted(EXPECTED_BY_MODE), required=True,
                    help="same MODE as transcode.sh: lowres=downscaled SBS, mono=left eye only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for ds in args.datasets:
        patch_dataset(ds, args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
