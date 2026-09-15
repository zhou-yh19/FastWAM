#!/usr/bin/env python3
"""Read FastWAM TA2 rosbag2 on the GPU host (no ROS install needed).

Steps this script covers:
  1) Export observation images + proprio + recorded actions
  2) Quick visualization (obs mosaic strips, action/latency plots)
  3) Offline infer_joint with your checkpoint → pred mosaic video + pred action
  4) Compare pred frames vs real mosaics from the bag

Example:
  cd <FastWAM repo root>
  conda activate fastwam
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints\"
  export CUDA_VISIBLE_DEVICES=0

  python experiments/teleavatar_v2_deploy/server/bag_visualize_and_predict.py \\
    --bag ../fastwam_bags/fastwam_ta2_20260804_110907 \\
    --task <config under configs/task/> \\
    --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_NNNNNN.pt \\
    --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json \\
    --out bag_offline_out/fastwam_ta2_20260804_110907 \\
    --max-samples 8 \\
    --num-inference-steps 15
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SERVER_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(SRC_ROOT), str(SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

import torch  # noqa: E402

from fastwam.datasets.lerobot.transforms.image import compose_ta2_mosaic  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402
from serve_policy import FastWAMTA2Policy  # noqa: E402

try:
    from analyze_deploy_bag import run_action_analysis  # noqa: E402
except ImportError:
    run_action_analysis = None  # type: ignore

CAMS = ("head_camera", "left_color", "right_color")
LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]


def _field_from_js(msg: Any, field: str, n: int, preferred: Optional[List[str]] = None) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    raw = getattr(msg, field, None)
    if raw is None:
        data = []
    else:
        data = list(raw)
    names_raw = getattr(msg, "name", None)
    names = list(names_raw) if names_raw is not None else []
    if preferred and names and len(data) == len(names):
        name_to_val = {str(nm): float(data[i]) for i, nm in enumerate(names)}
        if all(nm in name_to_val for nm in preferred[:n]):
            for i, nm in enumerate(preferred[:n]):
                out[i] = name_to_val[nm]
            return out
    m = min(n, len(data))
    if m:
        out[:m] = np.asarray(data[:m], dtype=np.float32)
    return out


def _build_proprio72(left_arm, right_arm, left_grip, right_grip, left_ee, right_ee) -> np.ndarray:
    state = np.zeros(72, dtype=np.float32)
    if left_arm is not None:
        state[0:7] = _field_from_js(left_arm, "position", 7, LEFT_JOINTS)
        state[16:23] = _field_from_js(left_arm, "velocity", 7, LEFT_JOINTS)
        state[32:39] = _field_from_js(left_arm, "effort", 7, LEFT_JOINTS)
    if left_grip is not None:
        state[7] = _field_from_js(left_grip, "position", 1)[0]
        state[23] = _field_from_js(left_grip, "velocity", 1)[0]
        state[39] = _field_from_js(left_grip, "effort", 1)[0]
    if right_arm is not None:
        state[8:15] = _field_from_js(right_arm, "position", 7, RIGHT_JOINTS)
        state[24:31] = _field_from_js(right_arm, "velocity", 7, RIGHT_JOINTS)
        state[40:47] = _field_from_js(right_arm, "effort", 7, RIGHT_JOINTS)
    if right_grip is not None:
        state[15] = _field_from_js(right_grip, "position", 1)[0]
        state[31] = _field_from_js(right_grip, "velocity", 1)[0]
        state[47] = _field_from_js(right_grip, "effort", 1)[0]
    if left_ee is not None:
        state[48:51] = [left_ee.position.x, left_ee.position.y, left_ee.position.z]
        state[51:55] = [
            left_ee.orientation.x,
            left_ee.orientation.y,
            left_ee.orientation.z,
            left_ee.orientation.w,
        ]
    if right_ee is not None:
        state[55:58] = [right_ee.position.x, right_ee.position.y, right_ee.position.z]
        state[58:62] = [
            right_ee.orientation.x,
            right_ee.orientation.y,
            right_ee.orientation.z,
            right_ee.orientation.w,
        ]
    return state


def _nearest(timeline: List[Tuple[int, Any]], t_ns: int, max_dt_ns: int):
    if not timeline:
        return None
    best = min(timeline, key=lambda x: abs(x[0] - t_ns))
    if abs(best[0] - t_ns) > max_dt_ns:
        return None
    return best[1]


def _match_after(
    stamp_ns: int,
    timeline: List[Tuple[int, Any]],
    *,
    max_after_ns: int,
    max_before_ns: int = 0,
) -> Optional[Tuple[int, Any]]:
    """Pick the earliest timeline entry at or after stamp_ns (causal pairing).

    Deploy flow: obs is published, then infer completes and action/infer_ms follow.
    Matching by nearest absolute time can attach a *previous* infer's action to the
    next obs; this helper avoids that misalignment.
    """
    if not timeline:
        return None
    lo = stamp_ns - max_before_ns
    hi = stamp_ns + max_after_ns
    candidates = [(t, v) for t, v in timeline if lo <= t <= hi and t >= stamp_ns]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0] - stamp_ns)


def _jpeg_bytes_to_rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def _action_from_multiarray(msg: Any) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.float32)
    dims = list(getattr(msg.layout, "dim", []) or [])
    if len(dims) >= 2:
        t, d = int(dims[0].size), int(dims[1].size)
        if t * d == data.size:
            return data.reshape(t, d)
    if data.size % 16 == 0:
        return data.reshape(-1, 16)
    return data.reshape(-1, 1)


def mosaic_from_cams(images: Dict[str, np.ndarray]) -> np.ndarray:
    """Build TA2 mosaic uint8 HWC from three SBS observation images."""
    cams = []
    for key in CAMS:
        arr = images[key]
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
        cams.append(t)
    mosaic01 = compose_ta2_mosaic(cams)[0]  # [3,H,W]
    out = (mosaic01.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return np.ascontiguousarray(out)


def hstack_labeled(frames: List[np.ndarray], labels: List[str]) -> Image.Image:
    imgs = [Image.fromarray(f) for f in frames]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (w, h + 28), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(imgs, labels):
        canvas.paste(im, (x, 28))
        draw.text((x + 4, 4), lab, fill=(240, 240, 240))
        x += im.width
    return canvas


def export_bag(bag: Path, out_dir: Path, match_ms: float = 1500.0) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    obs_groups: Dict[int, Dict[str, Any]] = {}
    actions: List[Tuple[int, np.ndarray]] = []
    infer_ms: List[Tuple[int, float]] = []
    left_arm_tl: List[Tuple[int, Any]] = []
    right_arm_tl: List[Tuple[int, Any]] = []
    left_grip_tl: List[Tuple[int, Any]] = []
    right_grip_tl: List[Tuple[int, Any]] = []
    left_ee_tl: List[Tuple[int, Any]] = []
    right_ee_tl: List[Tuple[int, Any]] = []

    topic_to_cam = {f"/fastwam/observation/{c}/compressed": c for c in CAMS}

    with AnyReader([bag], default_typestore=typestore) as reader:
        for conn, t_ns, raw in reader.messages():
            topic = conn.topic
            msg = reader.deserialize(raw, conn.msgtype)
            if topic in topic_to_cam:
                cam = topic_to_cam[topic]
                stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
                g = obs_groups.setdefault(stamp_ns, {"stamp_ns": stamp_ns, "images": {}})
                g["images"][cam] = _jpeg_bytes_to_rgb(bytes(msg.data))
            elif topic == "/fastwam/policy/action_chunk":
                actions.append((int(t_ns), _action_from_multiarray(msg)))
            elif topic == "/fastwam/policy/inference_ms":
                infer_ms.append((int(t_ns), float(msg.data)))
            elif topic == "/left_arm/joint_states":
                left_arm_tl.append((int(t_ns), msg))
            elif topic == "/right_arm/joint_states":
                right_arm_tl.append((int(t_ns), msg))
            elif topic == "/left_gripper/joint_states":
                left_grip_tl.append((int(t_ns), msg))
            elif topic == "/right_gripper/joint_states":
                right_grip_tl.append((int(t_ns), msg))
            elif topic == "/left_arm/current_ee_pose":
                left_ee_tl.append((int(t_ns), msg))
            elif topic == "/right_arm/current_ee_pose":
                right_ee_tl.append((int(t_ns), msg))

    max_dt = int(match_ms * 1e6)
    complete = [
        (stamp, g)
        for stamp, g in sorted(obs_groups.items())
        if all(c in g["images"] for c in CAMS)
    ]
    print(f"[export] complete obs groups={len(complete)} action_chunks={len(actions)}")

    rows: List[Dict[str, Any]] = []
    for idx, (stamp_ns, g) in enumerate(complete):
        sample_rel = f"samples/{idx:06d}"
        sample_dir = out_dir / sample_rel
        sample_dir.mkdir(parents=True, exist_ok=True)
        for cam in CAMS:
            Image.fromarray(g["images"][cam]).save(sample_dir / f"{cam}.jpg", quality=92)
        mosaic = mosaic_from_cams(g["images"])
        Image.fromarray(mosaic).save(sample_dir / "obs_mosaic.jpg", quality=92)
        np.save(sample_dir / "obs_mosaic.npy", mosaic)

        proprio = _build_proprio72(
            _nearest(left_arm_tl, stamp_ns, max_dt),
            _nearest(right_arm_tl, stamp_ns, max_dt),
            _nearest(left_grip_tl, stamp_ns, max_dt),
            _nearest(right_grip_tl, stamp_ns, max_dt),
            _nearest(left_ee_tl, stamp_ns, max_dt),
            _nearest(right_ee_tl, stamp_ns, max_dt),
        )
        np.save(sample_dir / "proprio72.npy", proprio)

        recorded_action = None
        action_ts_ns = None
        if actions:
            pick = _match_after(stamp_ns, actions, max_after_ns=max_dt)
            if pick is not None:
                action_ts_ns, recorded_action = int(pick[0]), pick[1]
                np.save(sample_dir / "recorded_action.npy", recorded_action)

        lat = None
        infer_ts_ns = None
        if infer_ms:
            pick_ms = _match_after(stamp_ns, infer_ms, max_after_ns=max_dt)
            if pick_ms is not None:
                infer_ts_ns, lat = int(pick_ms[0]), float(pick_ms[1])

        row = {
            "id": idx,
            "stamp_ns": stamp_ns,
            "action_ts_ns": action_ts_ns,
            "infer_ts_ns": infer_ts_ns,
            "action_lag_ms": (action_ts_ns - stamp_ns) / 1e6 if action_ts_ns else None,
            "dir": sample_rel,
            "has_recorded_action": recorded_action is not None,
            "inference_ms": lat,
        }
        rows.append(row)

    (out_dir / "samples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "bag": str(bag),
                "num_samples": len(rows),
                "note": (
                    "obs_mosaic is the TA2 mosaic built from bag observation cameras. "
                    "Bag video was usually NOT recorded online; offline infer regenerates it. "
                    "Future-frame GT uses later bag mosaics as coarse real frames "
                    "(bag only stores one obs per infer, not full fps video)."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def _estimate_obs_fps(rows: List[Dict[str, Any]], default: float = 2.0) -> float:
    if len(rows) < 2:
        return default
    dts = []
    for a, b in zip(rows, rows[1:]):
        dt = (int(b["stamp_ns"]) - int(a["stamp_ns"])) / 1e9
        if 0.05 < dt < 5.0:
            dts.append(dt)
    if not dts:
        return default
    return float(max(0.5, min(10.0, 1.0 / np.median(dts))))


def make_full_recorded_videos(export_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Stitch all bag observation frames into full-session mp4s (sparse ~infer rate)."""
    if not rows:
        return
    fps = _estimate_obs_fps(rows)
    mosaic_frames = []
    cam_frames: Dict[str, List[Image.Image]] = {c: [] for c in CAMS}
    for row in rows:
        sample_dir = export_dir / row["dir"]
        mp = sample_dir / "obs_mosaic.jpg"
        if mp.is_file():
            mosaic_frames.append(Image.open(mp).convert("RGB"))
        for cam in CAMS:
            cp = sample_dir / f"{cam}.jpg"
            if cp.is_file():
                cam_frames[cam].append(Image.open(cp).convert("RGB"))
    if mosaic_frames:
        out = export_dir / "recorded_obs_mosaic_full.mp4"
        save_mp4(mosaic_frames, str(out), fps=fps)
        print(f"[viz] wrote {out} ({len(mosaic_frames)} frames @ {fps:.2f} fps)")
    for cam, frames in cam_frames.items():
        if not frames:
            continue
        out = export_dir / f"recorded_{cam}_full.mp4"
        save_mp4(frames, str(out), fps=fps)
        print(f"[viz] wrote {out} ({len(frames)} frames @ {fps:.2f} fps)")


def make_overview_strip(out_dir: Path, rows: List[Dict[str, Any]], max_n: int = 12) -> None:
    """Save a contact-sheet of observation mosaics."""
    imgs = []
    for row in rows[:max_n]:
        p = out_dir / row["dir"] / "obs_mosaic.jpg"
        if p.is_file():
            im = Image.open(p).convert("RGB")
            im = im.resize((256, 192))
            imgs.append(im)
    if not imgs:
        return
    cols = min(4, len(imgs))
    rows_n = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows_n * 192), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * 256, (i // cols) * 192))
    sheet.save(out_dir / "viz_obs_mosaic_sheet.jpg", quality=90)
    print(f"[viz] wrote {out_dir / 'viz_obs_mosaic_sheet.jpg'}")


def run_predict_and_compare(
    rows: List[Dict[str, Any]],
    export_dir: Path,
    out_dir: Path,
    policy: FastWAMTA2Policy,
    *,
    return_video: bool,
    action_horizon: int,
    num_inference_steps: int,
    num_video_frames: int,
) -> None:
    pred_root = out_dir / "predictions"
    pred_root.mkdir(exist_ok=True)
    summary = []

    for i, row in enumerate(rows):
        sid = int(row["id"])
        sample_dir = export_dir / row["dir"]
        images = {
            cam: np.asarray(Image.open(sample_dir / f"{cam}.jpg").convert("RGB"), dtype=np.uint8)
            for cam in CAMS
        }
        proprio = np.load(sample_dir / "proprio72.npy").astype(np.float32)

        payload: Dict[str, Any] = {
            "images": {},
            "proprio": proprio.reshape(-1).tolist(),
            "action_horizon": int(action_horizon),
            "num_inference_steps": int(num_inference_steps),
            "num_video_frames": int(num_video_frames),
            "return_video": bool(return_video),
            "seed": 0,
        }
        for cam, arr in images.items():
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            payload["images"][cam] = base64.b64encode(buf.getvalue()).decode("ascii")

        out = policy.infer(payload)
        pred_action = np.asarray(out["action"], dtype=np.float32)
        sample_out = pred_root / f"{sid:06d}"
        sample_out.mkdir(exist_ok=True)
        np.save(sample_out / "pred_action.npy", pred_action)

        recorded = None
        rec_path = sample_dir / "recorded_action.npy"
        action_l1 = None
        if rec_path.is_file():
            recorded = np.load(rec_path).astype(np.float32)
            np.save(sample_out / "recorded_action.npy", recorded)
            t = min(recorded.shape[0], pred_action.shape[0])
            d = min(recorded.shape[1], pred_action.shape[1])
            action_l1 = float(np.mean(np.abs(pred_action[:t, :d] - recorded[:t, :d])))

        psnr_list = []
        if return_video and out.get("pred_video"):
            pred_frames = []
            for ti, b64 in enumerate(out["pred_video"]):
                fr = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                fr.save(sample_out / f"pred_{ti:02d}.jpg", quality=90)
                pred_frames.append(np.asarray(fr, dtype=np.uint8))

            # Only t=0 has timestamp-aligned GT (conditioning obs mosaic).
            # Bag stores one obs per infer, not dense video — do not compare future pred
            # frames to later samples (different wall-clock times / wrong alignment).
            real_t0 = np.load(sample_dir / "obs_mosaic.npy")
            compare_frames = []
            for ti, pred in enumerate(pred_frames):
                if ti == 0:
                    real = real_t0
                    if real.shape[:2] != pred.shape[:2]:
                        real = np.asarray(
                            Image.fromarray(real).resize(
                                (pred.shape[1], pred.shape[0]), Image.BILINEAR
                            )
                        )
                    mse = np.mean((pred.astype(np.float32) - real.astype(np.float32)) ** 2)
                    psnr = float(10.0 * np.log10((255.0**2) / max(mse, 1e-6)))
                    psnr_list.append(psnr)
                    panel = hstack_labeled(
                        [real, pred], ["real_t0(obs)", f"pred_t{ti:02d} PSNR={psnr:.1f}"]
                    )
                    panel.save(sample_out / f"compare_t{ti:02d}.jpg", quality=90)
                    compare_frames.append(panel)
                else:
                    panel = hstack_labeled([pred], [f"pred_t{ti:02d} (no aligned GT)"])
                    panel.save(sample_out / f"compare_t{ti:02d}.jpg", quality=90)

            if pred_frames:
                save_mp4(
                    [Image.fromarray(f) for f in pred_frames],
                    str(sample_out / "pred_video.mp4"),
                    fps=8,
                )
            if compare_frames:
                save_mp4(compare_frames, str(sample_out / "compare_pred_vs_real_t0.mp4"), fps=4)
                compare_frames[0].save(sample_out / "compare_t00_strip.jpg", quality=90)

        rec = {
            "id": sid,
            "infer_s": out.get("infer_s"),
            "mode": out.get("mode"),
            "action_l1_vs_recorded": action_l1,
            "video_psnr_mean": float(np.mean(psnr_list)) if psnr_list else None,
            "video_psnr_t0": psnr_list[0] if psnr_list else None,
            "out": str(sample_out),
        }
        summary.append(rec)
        print(
            f"[{sid:04d}] infer_s={rec['infer_s']:.2f}s "
            f"action_l1={action_l1} psnr_t0={rec['video_psnr_t0']}"
        )

    (pred_root / "summary.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in summary), encoding="utf-8"
    )
    print(f"[pred] summary -> {pred_root / 'summary.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True, help="rosbag2 directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=None)
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
    parser.add_argument("--skip-predict", action="store_true", help="only export+viz")
    parser.add_argument("--no-video", action="store_true", help="action-only offline infer")
    parser.add_argument(
        "--match-ms",
        type=float,
        default=2500.0,
        help="Max ms after obs stamp to pair action/infer_ms (causal match)",
    )
    parser.add_argument(
        "--skip-action-viz",
        action="store_true",
        help="skip published vs actual action plots",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="policy Hz for action_chunk timeline in action_viz",
    )
    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    if not bag.is_dir():
        raise SystemExit(f"bag not found: {bag}")

    export_dir = out_dir / "export"
    rows = export_bag(bag, export_dir, match_ms=args.match_ms)
    make_overview_strip(export_dir, rows)
    make_full_recorded_videos(export_dir, rows)

    rows = rows[args.start :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.skip_predict:
        print("skip predict (--skip-predict). Open export/ for images.")
    else:
        if args.checkpoint is None or args.dataset_stats is None:
            raise SystemExit("Need --checkpoint and --dataset-stats for prediction")

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
        run_predict_and_compare(
            rows,
            export_dir,
            out_dir,
            policy,
            return_video=not args.no_video,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            num_video_frames=args.num_video_frames,
        )

    if not args.skip_action_viz and run_action_analysis is not None:
        action_viz_dir = out_dir / "action_viz"
        pred_dir = out_dir / "predictions" if (out_dir / "predictions").is_dir() else None
        print(f"[action_viz] -> {action_viz_dir}")
        run_action_analysis(
            bag,
            action_viz_dir,
            control_hz=args.control_hz,
            export_dir=export_dir,
            pred_dir=pred_dir,
        )


if __name__ == "__main__":
    main()
