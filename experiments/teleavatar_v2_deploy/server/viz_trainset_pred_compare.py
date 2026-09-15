#!/usr/bin/env python3
"""Sample train windows, run infer_joint, save GT vs pred mosaic @ 5fps."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastwam.datasets.lerobot.transforms.image import compose_ta2_mosaic  # noqa: E402
from fastwam.utils import misc  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from serve_policy import FastWAMTA2Policy  # noqa: E402

CAMS = ("head_camera", "left_color", "right_color")
VIDEO_IDX = list(range(0, 33, 4))  # 9 frames for num_frames=33, ratio=4


def chw_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"expected CHW, got {tuple(x.shape)}")
    arr = x.permute(1, 2, 0).numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def mosaic_hwc(images: dict) -> np.ndarray:
    cams = []
    for k in CAMS:
        t = torch.from_numpy(np.ascontiguousarray(images[k])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cams.append(t)
    m = compose_ta2_mosaic(cams)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    return (m * 255.0).astype(np.uint8)


def raw_to_action16(gt_raw: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [gt_raw[:, 0:7], gt_raw[:, 39:40], gt_raw[:, 8:15], gt_raw[:, 47:48]],
        axis=-1,
    ).astype(np.float32)


def jpeg_b64(hwc: np.ndarray, quality: int = 90) -> str:
    buf = io.BytesIO()
    Image.fromarray(hwc, mode="RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def hstack_labeled(left: np.ndarray, right: np.ndarray, title_l: str, title_r: str) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    bar = 28
    canvas = Image.new("RGB", (left.shape[1] + right.shape[1], h + bar), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(left), (0, bar))
    canvas.paste(Image.fromarray(right), (left.shape[1], bar))
    draw.text((4, 4), title_l, fill=(240, 240, 240))
    draw.text((left.shape[1] + 4, 4), title_r, fill=(240, 240, 240))
    return np.asarray(canvas)


def plot_actions(gt: np.ndarray, pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(gt.shape[0])
    for ax, sl, name in (
        (axes[0], slice(0, 7), "L arm joints"),
        (axes[1], slice(8, 15), "R arm joints"),
        (axes[2], [7, 15], "grippers (effort)"),
    ):
        for j, idx in enumerate(range(sl.start, sl.stop) if isinstance(sl, slice) else sl):
            ax.plot(t, gt[:, idx], color="C0", alpha=0.35 if j else 1.0, label="gt" if j == 0 else None)
            ax.plot(t, pred[:, idx], color="C1", alpha=0.35 if j else 1.0, label="pred" if j == 0 else None)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if name.startswith("L"):
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("action step (20 Hz)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        help="Task config name under configs/task/ (without .yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="runs/<task>/<RUN_ID>/checkpoints/weights/step_NNNNNN.pt",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        required=True,
        help="runs/<task>/<RUN_ID>/dataset_stats.json -- must come from the same run "
             "as --checkpoint, or normalization silently drifts",
    )
    parser.add_argument("--out", type=Path, default=Path("bag_offline_out/trainset_pred"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=5.0, help="playback fps for compare mp4")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    register_default_resolvers()
    misc.register_work_dir(str(PROJECT_ROOT / "runs" / "_deploy_test"))

    configs_root = PROJECT_ROOT / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task}",
                f"data.train.pretrained_norm_stats={args.dataset_stats.resolve()}",
                "data.train.val_set_proportion=0.0",
            ],
        )
    OmegaConf.resolve(cfg)
    ds = instantiate(cfg.data.train)
    base = ds.lerobot_dataset

    rng = np.random.default_rng(args.seed)
    n = min(int(args.num_samples), len(ds))
    idxs = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    print(f"dataset len={len(ds)} sample idxs={idxs}")

    out_root = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    policy = FastWAMTA2Policy(
        checkpoint=args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint,
        dataset_stats=args.dataset_stats if args.dataset_stats.is_absolute() else PROJECT_ROOT / args.dataset_stats,
        task_name=args.task,
        device=args.device,
        action_horizon=32,
        num_inference_steps=args.num_inference_steps,
        num_video_frames=9,
    )

    summary = []
    saved_processor = base.processor
    try:
        base.processor = None
        for si, index in enumerate(idxs):
            raw = base[index]
            # t0 images for policy input
            images0 = {k: chw_to_hwc_uint8(raw["images"][k][0]) for k in CAMS}
            # GT mosaics at video sample times
            gt_frames = []
            for ti in VIDEO_IDX:
                imgs = {k: chw_to_hwc_uint8(raw["images"][k][ti]) for k in CAMS}
                gt_frames.append(mosaic_hwc(imgs))

            state_key = list(raw["state"].keys())[0]
            action_key = list(raw["action"].keys())[0]
            proprio = raw["state"][state_key][0].detach().cpu().numpy().astype(np.float32).reshape(-1)
            gt_action = raw_to_action16(raw["action"][action_key].detach().cpu().numpy().astype(np.float32))
            gt_action = gt_action[:32]

            payload = {
                "images": {k: jpeg_b64(v) for k, v in images0.items()},
                "proprio": proprio.tolist(),
                "action_horizon": 32,
                "num_inference_steps": args.num_inference_steps,
                "num_video_frames": 9,
                "return_video": True,
                "seed": 0,
            }
            out = policy.infer(payload)
            pred_action = np.asarray(out["action"], dtype=np.float32)[:32]
            pred_b64 = out.get("pred_video") or []
            pred_frames = []
            for b64 in pred_b64:
                pred_frames.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")))

            t = min(len(gt_frames), len(pred_frames))
            compare = [
                hstack_labeled(gt_frames[i], pred_frames[i], f"GT t={VIDEO_IDX[i]}", f"Pred t={i}")
                for i in range(t)
            ]

            sample_dir = out_root / f"sample_{si:02d}_idx{index}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(sample_dir / "compare_gt_vs_pred_5fps.mp4", compare, fps=float(args.fps), codec="libx264", quality=8)
            imageio.mimsave(sample_dir / "gt_mosaic_5fps.mp4", gt_frames[:t], fps=float(args.fps), codec="libx264", quality=8)
            imageio.mimsave(sample_dir / "pred_mosaic_5fps.mp4", pred_frames[:t], fps=float(args.fps), codec="libx264", quality=8)
            Image.fromarray(compare[0]).save(sample_dir / "compare_t0.jpg", quality=92)
            np.save(sample_dir / "gt_action.npy", gt_action)
            np.save(sample_dir / "pred_action.npy", pred_action)
            action_l1 = float(np.mean(np.abs(pred_action - gt_action)))
            joint_l1 = float(np.mean(np.abs(pred_action[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14]] - gt_action[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14]])))
            plot_actions(
                gt_action,
                pred_action,
                sample_dir / "action_compare.png",
                f"idx={index} action_L1={action_l1:.4f} joint14_L1={joint_l1:.4f}",
            )

            row = {
                "sample": si,
                "index": int(index),
                "infer_s": float(out.get("infer_s", -1)),
                "action_l1": action_l1,
                "joint14_l1": joint_l1,
                "dir": str(sample_dir.relative_to(out_root)),
            }
            summary.append(row)
            print(
                f"[{si+1}/{n}] idx={index} action_L1={action_l1:.4f} joint14_L1={joint_l1:.4f} "
                f"infer_s={row['infer_s']:.2f} -> {sample_dir}"
            )
    finally:
        base.processor = saved_processor

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_root}")
    print(
        "mean action_L1={:.4f} joint14_L1={:.4f}".format(
            float(np.mean([r["action_l1"] for r in summary])),
            float(np.mean([r["joint14_l1"] for r in summary])),
        )
    )


if __name__ == "__main__":
    main()
