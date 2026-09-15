#!/usr/bin/env python3
"""Offline re-infer from an exported FastWAM bag folder (no ROS required).

Input per sample: observation SBS JPEGs + proprio72 (+ optional recorded_action).
Output: predicted mosaic video frames + predicted action chunk.

Example (on GPU host, with serve_policy code / weights available):

  cd <FastWAM repo root>
  conda activate fastwam
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

  python experiments/teleavatar_v2_deploy/server/offline_infer_from_export.py \\
    --export bag_exports/<run> \\
    --task <config under configs/task/> \\
    --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_NNNNNN.pt \\
    --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json \\
    --out bag_offline_out/<run> \\
    --num-inference-steps 15 \\
    --return-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SERVER_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SRC_ROOT, SERVER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from serve_policy import FastWAMTA2Policy  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402

CAMS = ("head_camera", "left_color", "right_color")


def _load_sample(export_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    sample_dir = export_dir / row["dir"]
    images = {}
    for cam in CAMS:
        path = sample_dir / f"{cam}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        images[cam] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    proprio = np.load(sample_dir / "proprio72.npy").astype(np.float32)
    recorded = None
    rec_path = sample_dir / "recorded_action.npy"
    if rec_path.is_file():
        recorded = np.load(rec_path).astype(np.float32)
    return {"images": images, "proprio": proprio, "recorded_action": recorded, "dir": sample_dir}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True, help="folder from export_bag_for_offline.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task config name under configs/task/ (without .yaml)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument("--return-video", action="store_true", default=True)
    parser.add_argument("--no-return-video", action="store_false", dest="return_video")
    parser.add_argument("--max-samples", type=int, default=0, help="0=all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--prompt-task",
        type=str,
        default=None,
        help="instruction to condition on; must match the training data byte for byte "
        "(default: read from the task dataset's meta/tasks.jsonl)",
    )
    parser.add_argument(
        "--text-embed",
        type=Path,
        default=None,
        help="precomputed T5 context .pt, bypassing the prompt->cache lookup",
    )
    parser.add_argument(
        "--load-text-encoder",
        action="store_true",
        help="load the 11 GiB T5 and encode the prompt on the fly instead of reading the cache",
    )
    parser.add_argument(
        "--condition-video-on-recorded-action",
        action="store_true",
        help="if set and recorded_action exists, pass it into infer_joint as video condition "
        "(experimental; default is open-loop joint predict of video+action)",
    )
    args = parser.parse_args()

    export_dir = args.export.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (export_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = rows[args.start :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"no samples in {export_dir}")

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
    )

    summary: List[Dict[str, Any]] = []
    for row in rows:
        sid = int(row["id"])
        sample = _load_sample(export_dir, row)
        payload: Dict[str, Any] = {
            "images": {
                # serve_policy expects jpeg b64; reuse its helpers via encode here
            },
            "proprio": sample["proprio"].reshape(-1).tolist(),
            "action_horizon": int(args.action_horizon),
            "num_inference_steps": int(args.num_inference_steps),
            "num_video_frames": int(args.num_video_frames),
            "return_video": bool(args.return_video),
            "seed": 0,
        }
        # Encode images to b64 jpeg for the existing Policy.infer API.
        import base64
        import io

        for cam, arr in sample["images"].items():
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            payload.setdefault("images", {})[cam] = base64.b64encode(buf.getvalue()).decode("ascii")

        # Optional video conditioning on recorded action is not exposed via HTTP payload yet;
        # keep default joint open-loop. (Hook reserved for future.)
        _ = args.condition_video_on_recorded_action

        out = policy.infer(payload)
        pred_action = np.asarray(out["action"], dtype=np.float32)

        sample_out = out_dir / f"{sid:06d}"
        sample_out.mkdir(parents=True, exist_ok=True)
        np.save(sample_out / "pred_action.npy", pred_action)
        # Save obs copy for side-by-side review
        for cam, arr in sample["images"].items():
            Image.fromarray(arr).save(sample_out / f"obs_{cam}.jpg", quality=90)
        if sample["recorded_action"] is not None:
            np.save(sample_out / "recorded_action.npy", sample["recorded_action"])
            # quick numeric compare on overlapping length
            ra = sample["recorded_action"]
            t = min(ra.shape[0], pred_action.shape[0])
            d = int(min(ra.shape[1], pred_action.shape[1])) if ra.ndim == 2 else 0
            if t > 0 and d > 0:
                err = np.mean(np.abs(pred_action[:t, :d] - ra[:t, :d]))
            else:
                err = None
        else:
            err = None

        n_vid = 0
        if args.return_video and out.get("pred_video"):
            frames = []
            for ti, b64 in enumerate(out["pred_video"]):
                raw = base64.b64decode(b64)
                frame = Image.open(io.BytesIO(raw)).convert("RGB")
                frame.save(sample_out / f"pred_video_{ti:02d}.jpg", quality=90)
                frames.append(frame)
            n_vid = len(frames)
            if frames:
                save_mp4(frames, str(sample_out / "pred_video.mp4"), fps=8)

        rec = {
            "id": sid,
            "infer_s": out.get("infer_s"),
            "mode": out.get("mode"),
            "pred_action_shape": list(pred_action.shape),
            "pred_video_frames": n_vid,
            "recorded_action_l1": err,
            "out_dir": str(sample_out),
        }
        summary.append(rec)
        print(
            f"[{sid:04d}] mode={rec['mode']} infer_s={rec['infer_s']:.2f} "
            f"video={n_vid} action_l1_vs_recorded={err}"
        )

    (out_dir / "summary.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in summary),
        encoding="utf-8",
    )
    print(f"Done. {len(summary)} samples -> {out_dir}")


if __name__ == "__main__":
    main()
