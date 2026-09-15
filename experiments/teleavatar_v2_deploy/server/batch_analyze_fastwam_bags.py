#!/usr/bin/env python3
"""Batch-export FastWAM deployment bags.

For each rosbag2 directory this command writes:

* ``joint_analysis/``: joint commands, measured states, gripper data, end-effector
  pose CSVs, and comparison plots;
* ``rtp/export/``: the recorded RTP JPEG streams, one synchronized sample directory
  per camera timestamp, mosaics, and MP4 previews.

The command is intentionally prediction-free. The 2026-09-07 bags contain the
published arm commands and camera observations, but do not contain policy action
chunks, so no offline prediction is inferred from missing topics.

Example::

    conda activate fastwam
    python experiments/teleavatar_v2_deploy/server/batch_analyze_fastwam_bags.py \
        --date 20260907
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SERVER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_deploy_bag import run_action_analysis  # noqa: E402
from bag_visualize_and_predict import (  # noqa: E402
    export_bag,
    make_full_recorded_videos,
    make_overview_strip,
)
from rtp_stream_export import export_rtp_streaming  # noqa: E402


def _resolve_bags(args: argparse.Namespace) -> List[Path]:
    if args.bag:
        bags = [Path(value).expanduser().resolve() for value in args.bag]
    else:
        pattern = f"fastwam_ta2_{args.date}_*"
        bags = sorted((PROJECT_ROOT / "fastwam_bags").glob(pattern))
    bags = [bag for bag in bags if bag.is_dir()]
    if not bags:
        raise SystemExit("No rosbag2 directories found for the requested inputs")
    return bags


def _write_summary(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        fields = [
            "bag",
            "duration_s",
            "joint_cmd_left",
            "joint_cmd_right",
            "joint_state_left",
            "joint_state_right",
            "ee_pose_left",
            "ee_pose_right",
            "rtp_head",
            "rtp_left_wrist",
            "rtp_right_wrist",
            "synchronized_rtp_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _csv_count_and_end(path: Path) -> tuple[int, float]:
    """Return data-row count and final t_sec from an existing analysis CSV."""
    if not path.is_file():
        return 0, 0.0
    count = 0
    end = 0.0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            count += 1
            try:
                end = max(end, float(row[0]))
            except (IndexError, ValueError):
                pass
    return count, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        action="append",
        help="rosbag2 directory; repeat for multiple bags (default: all bags for --date)",
    )
    parser.add_argument("--date", default="20260907", help="date suffix used for discovery")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output root (default: fastwam_bags/batch_analysis_<date>)",
    )
    parser.add_argument("--match-ms", type=float, default=2500.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete joint/RTP outputs already present under --out",
    )
    parser.add_argument(
        "--stream-rtp",
        action="store_true",
        help="use the low-memory streaming RTP exporter under rtp/streaming",
    )
    args = parser.parse_args()

    bags = _resolve_bags(args)
    out_root = (
        args.out.expanduser().resolve()
        if args.out is not None
        else PROJECT_ROOT / "fastwam_bags" / f"batch_analysis_{args.date}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"Analyzing {len(bags)} bags -> {out_root}")
    for index, bag in enumerate(bags, start=1):
        bag_out = out_root / bag.name
        joint_out = bag_out / "joint_analysis"
        rtp_out = bag_out / "rtp"
        joint_complete = (joint_out / "ee_pose_left.csv").is_file() and (
            joint_out / "joint_states_left.csv"
        ).is_file()
        if args.resume and joint_complete:
            print(f"[{index}/{len(bags)}] {bag.name}: reuse joint analysis")
            data = None
        else:
            print(f"[{index}/{len(bags)}] {bag.name}: joints and poses")
            data = run_action_analysis(bag, joint_out, control_hz=args.control_hz)

        streaming_dir = rtp_out / "streaming"
        legacy_dir = rtp_out / "export"
        # On resume, accept either exporter so a batch can be continued after a
        # slow legacy export was replaced by the streaming implementation.
        if args.stream_rtp and args.resume and not (streaming_dir / "meta.json").is_file() and (
            legacy_dir / "meta.json"
        ).is_file():
            export_dir = legacy_dir
        else:
            export_dir = streaming_dir if args.stream_rtp else legacy_dir
        rtp_complete = (export_dir / "meta.json").is_file()
        if args.resume and rtp_complete:
            print(f"[{index}/{len(bags)}] {bag.name}: reuse RTP export")
            rtp_meta = json.loads((export_dir / "meta.json").read_text())
            image_rows = []
            if "camera_frame_counts" not in rtp_meta:
                # Legacy export metadata records only the number of complete
                # synchronized samples; each sample contains all three cameras.
                sample_count = int(rtp_meta.get("num_samples", 0))
                if sample_count == 0 and (export_dir / "samples.jsonl").is_file():
                    sample_count = sum(
                        1
                        for line in (export_dir / "samples.jsonl").read_text().splitlines()
                        if line.strip()
                    )
                rtp_meta["camera_frame_counts"] = {cam: sample_count for cam in ("head_camera", "left_color", "right_color")}
                rtp_meta["synchronized_rtp_samples"] = sample_count
        elif args.stream_rtp:
            print(f"[{index}/{len(bags)}] {bag.name}: streaming RTP images and videos")
            rtp_meta = export_rtp_streaming(bag, export_dir)
            image_rows = []
        else:
            print(f"[{index}/{len(bags)}] {bag.name}: RTP images and videos")
            image_rows = export_bag(bag, export_dir, match_ms=args.match_ms)
            make_overview_strip(export_dir, image_rows)
            make_full_recorded_videos(export_dir, image_rows)
            rtp_meta = {
                "camera_frame_counts": {
                    cam: sum(
                        (export_dir / "samples" / row["dir"].split("/")[-1] / f"{cam}.jpg").is_file()
                        for row in image_rows
                    )
                    for cam in ("head_camera", "left_color", "right_color")
                },
                "synchronized_rtp_samples": len(image_rows),
            }

        if data is None:
            cmd_l, end_l = _csv_count_and_end(joint_out / "joint_cmd_left.csv")
            cmd_r, end_r = _csv_count_and_end(joint_out / "joint_cmd_right.csv")
            state_l, end_sl = _csv_count_and_end(joint_out / "joint_states_left.csv")
            state_r, end_sr = _csv_count_and_end(joint_out / "joint_states_right.csv")
            ee_l, end_ee_l = _csv_count_and_end(joint_out / "ee_pose_left.csv")
            ee_r, end_ee_r = _csv_count_and_end(joint_out / "ee_pose_right.csv")
            duration_s = max(end_l, end_r, end_sl, end_sr, end_ee_l, end_ee_r)
        else:
            cmd_l, cmd_r = len(data["left_cmd"]), len(data["right_cmd"])
            state_l, state_r = len(data["left_meas"]), len(data["right_meas"])
            ee_l, ee_r = len(data["left_ee"]), len(data["right_ee"])
            duration_s = max(
                [t for t, _ in data["left_meas"]]
                + [t for t, _ in data["right_meas"]]
                + [t for t, _ in data["left_ee"]]
                + [t for t, _ in data["right_ee"]]
                + [0.0]
            )

        summary_rows.append(
            {
                "bag": bag.name,
                "duration_s": float(duration_s),
                "joint_cmd_left": cmd_l,
                "joint_cmd_right": cmd_r,
                "joint_state_left": state_l,
                "joint_state_right": state_r,
                "ee_pose_left": ee_l,
                "ee_pose_right": ee_r,
                "rtp_head": int(rtp_meta.get("camera_frame_counts", {}).get("head_camera", 0)),
                "rtp_left_wrist": int(rtp_meta.get("camera_frame_counts", {}).get("left_color", 0)),
                "rtp_right_wrist": int(rtp_meta.get("camera_frame_counts", {}).get("right_color", 0)),
                "synchronized_rtp_samples": int(rtp_meta.get("synchronized_rtp_samples", len(image_rows))),
            }
        )

    _write_summary(out_root / "summary.json", summary_rows)
    print(f"Wrote batch summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
