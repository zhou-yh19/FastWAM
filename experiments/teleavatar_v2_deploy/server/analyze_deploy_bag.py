#!/usr/bin/env python3
"""Analyze a FastWAM TA2 deploy rosbag: predicted vs commanded vs measured.

Produces the same artifact set as
  fastwam_bags/.../fastwam_ta2_*_analysis/
without needing a ROS install (uses rosbags).

Example:
  cd <FastWAM repo root>
  conda activate fastwam
  python experiments/teleavatar_v2_deploy/server/analyze_deploy_bag.py \\
    --bag ../fastwam_bags/fastwam_ta2_20260804_180355 \\
    --out ../fastwam_bags/fastwam_ta2_20260804_180355_analysis
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]
CLOSE_THRESH_NM = -0.3
ACTION_LABELS = (
    [f"L_q{i}" for i in range(7)]
    + ["L_grip_eff"]
    + [f"R_q{i}" for i in range(7)]
    + ["R_grip_eff"]
)
CAMERA_TOPICS = {
    "/fastwam/observation/head_camera/compressed": "head_camera",
    "/fastwam/observation/left_color/compressed": "left_color",
    "/fastwam/observation/right_color/compressed": "right_color",
}
FSM_ON_THRESH = 0.5


def _field_from_js(
    msg: Any,
    field: str,
    n: int,
    preferred: Optional[Sequence[str]] = None,
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    raw = getattr(msg, field, None)
    data = list(raw) if raw is not None else []
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


def _pose_from_msg(msg: Any) -> np.ndarray:
    """geometry_msgs/Pose -> [x, y, z, qx, qy, qz, qw]."""
    p = msg.position
    o = msg.orientation
    return np.array(
        [p.x, p.y, p.z, o.x, o.y, o.z, o.w],
        dtype=np.float64,
    )


def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """[N,4] quaternions (x, y, z, w) -> [N,3] roll/pitch/yaw in rad (ZYX intrinsic)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    x, y, z, w = (q / norm).T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def _header_epoch(msg: Any) -> Optional[float]:
    """Absolute header stamp in seconds, or None when unset/zero."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nsec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nsec == 0:
        return None
    return sec + nsec / 1e9


def _fsm_spans(
    fsm: Sequence[Tuple[float, float]],
    thresh: float = FSM_ON_THRESH,
) -> List[Tuple[float, float]]:
    """Contiguous [t_start, t_end] windows where the FSM enable flag is high."""
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for t, v in fsm:
        if v > thresh and start is None:
            start = t
        elif v <= thresh and start is not None:
            spans.append((start, t))
            start = None
    if start is not None and fsm:
        spans.append((start, fsm[-1][0]))
    return spans


def _shade_fsm(ax: Any, spans: Sequence[Tuple[float, float]], *, label: bool = False) -> None:
    """Shade FSM-enabled windows so plots show when the policy was actually in control."""
    for i, (a, b) in enumerate(spans):
        ax.axvspan(
            a,
            b,
            color="green",
            alpha=0.07,
            zorder=0,
            label="fsm enabled" if (label and i == 0) else None,
        )


def _nearest(
    timeline: Sequence[Tuple[float, np.ndarray]],
    t: float,
) -> Optional[np.ndarray]:
    if not timeline:
        return None
    # timeline sorted by t
    idx = int(np.searchsorted([x[0] for x in timeline], t))
    cands = []
    if 0 <= idx < len(timeline):
        cands.append(timeline[idx])
    if 0 <= idx - 1 < len(timeline):
        cands.append(timeline[idx - 1])
    if not cands:
        return None
    best = min(cands, key=lambda x: abs(x[0] - t))
    return best[1]


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_bag(
    bag: Path,
    control_hz: float,
) -> Dict[str, Any]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    left_cmd: List[Tuple[float, np.ndarray]] = []
    right_cmd: List[Tuple[float, np.ndarray]] = []
    left_meas: List[Tuple[float, np.ndarray]] = []
    right_meas: List[Tuple[float, np.ndarray]] = []
    left_grip_cmd: List[Tuple[float, float]] = []
    right_grip_cmd: List[Tuple[float, float]] = []
    left_grip_meas: List[Tuple[float, float]] = []
    right_grip_meas: List[Tuple[float, float]] = []
    actions: List[Tuple[float, np.ndarray]] = []
    infer_ms: List[Tuple[float, float]] = []
    left_ee: List[Tuple[float, np.ndarray]] = []
    right_ee: List[Tuple[float, np.ndarray]] = []
    fsm: List[Tuple[float, float]] = []
    cam_meta: Dict[str, List[Tuple[float, int, str, Optional[float]]]] = {
        name: [] for name in CAMERA_TOPICS.values()
    }
    t0_ns: Optional[int] = None

    want = {
        "/api/left_arm/joint_cmd",
        "/api/right_arm/joint_cmd",
        "/api/left_gripper/cmd",
        "/api/right_gripper/cmd",
        "/left_arm/joint_states",
        "/right_arm/joint_states",
        "/left_gripper/joint_states",
        "/right_gripper/joint_states",
        "/fastwam/policy/action_chunk",
        "/fastwam/policy/inference_ms",
        "/left_arm/current_ee_pose",
        "/right_arm/current_ee_pose",
        "/api/fsm/enable",
        *CAMERA_TOPICS,
    }

    with AnyReader([bag], default_typestore=typestore) as reader:
        for conn, t_ns, raw in reader.messages():
            if conn.topic not in want:
                continue
            if t0_ns is None:
                t0_ns = int(t_ns)
            t_sec = (int(t_ns) - t0_ns) / 1e9
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/api/left_arm/joint_cmd":
                left_cmd.append((t_sec, _field_from_js(msg, "position", 7, LEFT_JOINTS)))
            elif conn.topic == "/api/right_arm/joint_cmd":
                right_cmd.append((t_sec, _field_from_js(msg, "position", 7, RIGHT_JOINTS)))
            elif conn.topic == "/api/left_gripper/cmd":
                left_grip_cmd.append((t_sec, float(msg.data)))
            elif conn.topic == "/api/right_gripper/cmd":
                right_grip_cmd.append((t_sec, float(msg.data)))
            elif conn.topic == "/left_arm/joint_states":
                left_meas.append((t_sec, _field_from_js(msg, "position", 7, LEFT_JOINTS)))
            elif conn.topic == "/right_arm/joint_states":
                right_meas.append((t_sec, _field_from_js(msg, "position", 7, RIGHT_JOINTS)))
            elif conn.topic == "/left_gripper/joint_states":
                left_grip_meas.append((t_sec, float(_field_from_js(msg, "position", 1)[0])))
            elif conn.topic == "/right_gripper/joint_states":
                right_grip_meas.append((t_sec, float(_field_from_js(msg, "position", 1)[0])))
            elif conn.topic == "/fastwam/policy/action_chunk":
                actions.append((t_sec, _action_from_multiarray(msg)))
            elif conn.topic == "/fastwam/policy/inference_ms":
                infer_ms.append((t_sec, float(msg.data)))
            elif conn.topic == "/left_arm/current_ee_pose":
                left_ee.append((t_sec, _pose_from_msg(msg)))
            elif conn.topic == "/right_arm/current_ee_pose":
                right_ee.append((t_sec, _pose_from_msg(msg)))
            elif conn.topic == "/api/fsm/enable":
                fsm.append((t_sec, float(msg.data)))
            elif conn.topic in CAMERA_TOPICS:
                epoch = _header_epoch(msg)
                lat_ms = None if epoch is None else (int(t_ns) / 1e9 - epoch) * 1000.0
                cam_meta[CAMERA_TOPICS[conn.topic]].append(
                    (t_sec, int(len(msg.data)), str(getattr(msg, "format", "")), lat_ms)
                )

    if t0_ns is None:
        raise RuntimeError(f"No relevant messages in bag: {bag}")

    n = min(len(actions), len(infer_ms))
    if len(actions) != len(infer_ms):
        print(
            f"[warn] action_chunks={len(actions)} inference_ms={len(infer_ms)}; "
            f"pairing first {n}"
        )

    inferences = []
    chunks_full = []
    for i in range(n):
        t_infer, chunk = actions[i]
        _, ms = infer_ms[i]
        if chunk.ndim != 2 or chunk.shape[1] < 16:
            raise ValueError(f"action_chunk[{i}] expected [T,16+], got {chunk.shape}")
        step0 = chunk[0]
        inferences.append(
            {
                "infer_id": i,
                "t_sec": t_infer,
                "inference_ms": ms,
                "L_q": step0[0:7].astype(np.float64),
                "L_grip_eff": float(step0[7]),
                "R_q": step0[8:15].astype(np.float64),
                "R_grip_eff": float(step0[15]),
                "chunk": chunk,
            }
        )
        for step in range(chunk.shape[0]):
            a = chunk[step]
            t_step = t_infer + step / float(control_hz)
            chunks_full.append(
                [
                    i,
                    step,
                    t_step,
                    *a[0:7].tolist(),
                    float(a[7]),
                    *a[8:15].tolist(),
                    float(a[15]),
                ]
            )

    return {
        "t0_ns": t0_ns,
        "joint_cmd": left_cmd,
        "joint_states": left_meas,
        "left_cmd": left_cmd,
        "right_cmd": right_cmd,
        "left_meas": left_meas,
        "right_meas": right_meas,
        "left_grip_cmd": left_grip_cmd,
        "right_grip_cmd": right_grip_cmd,
        "left_grip_meas": left_grip_meas,
        "right_grip_meas": right_grip_meas,
        "inferences": inferences,
        "chunks_full": chunks_full,
        "control_hz": control_hz,
        "left_ee": left_ee,
        "right_ee": right_ee,
        "fsm": fsm,
        "cam_meta": cam_meta,
    }


def export_csvs(data: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(
        out / "joint_cmd_left.csv",
        ["t_sec", "q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        [[t, *q.tolist()] for t, q in data["joint_cmd"]],
    )
    _write_csv(
        out / "joint_states_left.csv",
        ["t_sec", "q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        [[t, *q.tolist()] for t, q in data["joint_states"]],
    )
    _write_csv(
        out / "joint_cmd_right.csv",
        ["t_sec", "q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        [[t, *q.tolist()] for t, q in data["right_cmd"]],
    )
    _write_csv(
        out / "joint_states_right.csv",
        ["t_sec", "q0", "q1", "q2", "q3", "q4", "q5", "q6"],
        [[t, *q.tolist()] for t, q in data["right_meas"]],
    )
    _write_csv(
        out / "gripper_cmd.csv",
        ["t_sec", "left_trig", "right_trig"],
        _merge_gripper_series(data["left_grip_cmd"], data["right_grip_cmd"]),
    )
    _write_csv(
        out / "gripper_states.csv",
        ["t_sec", "left_pos", "right_pos"],
        _merge_gripper_series(data["left_grip_meas"], data["right_grip_meas"]),
    )

    inf_rows = []
    align_rows = []
    for inf in data["inferences"]:
        t = inf["t_sec"]
        L = inf["L_q"]
        R = inf["R_q"]
        inf_rows.append(
            [
                inf["infer_id"],
                t,
                inf["inference_ms"],
                *L.tolist(),
                inf["L_grip_eff"],
                *R.tolist(),
                inf["R_grip_eff"],
            ]
        )
        cmd = _nearest(data["joint_cmd"], t)
        meas = _nearest(data["joint_states"], t)
        if cmd is None or meas is None:
            continue
        pred0 = float(L[0])
        cmd0 = float(cmd[0])
        meas0 = float(meas[0])
        align_rows.append(
            [
                inf["infer_id"],
                t,
                inf["inference_ms"],
                pred0,
                cmd0,
                meas0,
                pred0 - cmd0,
                pred0 - meas0,
                cmd0 - meas0,
            ]
        )

    _write_csv(
        out / "inferences.csv",
        [
            "infer_id",
            "t_sec",
            "inference_ms",
            "L_q0",
            "L_q1",
            "L_q2",
            "L_q3",
            "L_q4",
            "L_q5",
            "L_q6",
            "L_grip_eff",
            "R_q0",
            "R_q1",
            "R_q2",
            "R_q3",
            "R_q4",
            "R_q5",
            "R_q6",
            "R_grip_eff",
        ],
        inf_rows,
    )
    _write_csv(
        out / "infer_align_left_arm.csv",
        [
            "infer_id",
            "t_sec",
            "inference_ms",
            "pred_L_q0",
            "cmd_L_q0",
            "meas_L_q0",
            "pred_minus_cmd_q0",
            "pred_minus_meas_q0",
            "cmd_minus_meas_q0",
        ],
        align_rows,
    )
    _write_csv(
        out / "action_chunks_full.csv",
        [
            "infer_id",
            "step",
            "t_sec",
            "L_q0",
            "L_q1",
            "L_q2",
            "L_q3",
            "L_q4",
            "L_q5",
            "L_q6",
            "L_grip_eff",
            "R_q0",
            "R_q1",
            "R_q2",
            "R_q3",
            "R_q4",
            "R_q5",
            "R_q6",
            "R_grip_eff",
        ],
        data["chunks_full"],
    )

    for side in ("left", "right"):
        series = data[f"{side}_ee"]
        ee_rows: List[List[float]] = []
        if series:
            rpy = _quat_to_euler(np.stack([p[3:7] for _, p in series], axis=0))
            for (t, p), e in zip(series, rpy):
                ee_rows.append([t, *p.tolist(), *e.tolist()])
        _write_csv(
            out / f"ee_pose_{side}.csv",
            ["t_sec", "x", "y", "z", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw"],
            ee_rows,
        )

    _write_csv(
        out / "fsm_enable.csv",
        ["t_sec", "enable"],
        [[t, v] for t, v in data["fsm"]],
    )

    cam_summary = []
    for name, frames in data["cam_meta"].items():
        cam_rows = []
        prev_t: Optional[float] = None
        for t, nbytes, fmt, lat in frames:
            cam_rows.append(
                [
                    t,
                    "" if prev_t is None else (t - prev_t) * 1000.0,
                    nbytes,
                    fmt,
                    "" if lat is None else lat,
                ]
            )
            prev_t = t
        _write_csv(
            out / f"camera_frames_{name}.csv",
            ["t_sec", "dt_ms", "size_bytes", "format", "latency_ms"],
            cam_rows,
        )
        cam_summary.append(_camera_summary_row(name, frames))
    _write_csv(
        out / "camera_summary.csv",
        [
            "camera",
            "n_frames",
            "duration_s",
            "mean_fps",
            "dt_p50_ms",
            "dt_p95_ms",
            "dt_max_ms",
            "mean_size_kb",
            "mean_latency_ms",
        ],
        cam_summary,
    )


def _camera_summary_row(
    name: str,
    frames: Sequence[Tuple[float, int, str, Optional[float]]],
) -> List[Any]:
    if not frames:
        return [name, 0, 0.0, 0.0, "", "", "", "", ""]
    t = np.array([f[0] for f in frames], dtype=np.float64)
    sizes = np.array([f[1] for f in frames], dtype=np.float64)
    lats = np.array([f[3] for f in frames if f[3] is not None], dtype=np.float64)
    dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dt_ms = np.diff(t) * 1000.0 if t.size > 1 else np.array([])
    return [
        name,
        len(frames),
        round(dur, 3),
        round((len(frames) - 1) / dur, 2) if dur > 0 else 0.0,
        round(float(np.median(dt_ms)), 2) if dt_ms.size else "",
        round(float(np.percentile(dt_ms, 95)), 2) if dt_ms.size else "",
        round(float(dt_ms.max()), 2) if dt_ms.size else "",
        round(float(sizes.mean()) / 1024.0, 1),
        round(float(lats.mean()), 2) if lats.size else "",
    ]


def _camera_fps(frames: Sequence[Tuple[float, int, str, Optional[float]]]) -> float:
    """Median-interval fps, used as the default playback rate for exported video."""
    if len(frames) < 2:
        return 15.0
    dt = np.diff(np.array([f[0] for f in frames], dtype=np.float64))
    med = float(np.median(dt))
    return 1.0 / med if med > 0 else 15.0


def _merge_gripper_series(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
) -> List[List[float]]:
    """Merge left/right scalar series onto a common time grid (union of timestamps)."""
    times = sorted({t for t, _ in left} | {t for t, _ in right})
    left_map = dict(left)
    right_map = dict(right)
    rows = []
    last_l = last_r = 0.0
    for t in times:
        if t in left_map:
            last_l = left_map[t]
        if t in right_map:
            last_r = right_map[t]
        rows.append([t, last_l, last_r])
    return rows


def _arm_publish_vs_actual_plot(
    cmd: Sequence[Tuple[float, np.ndarray]],
    meas: Sequence[Tuple[float, np.ndarray]],
    *,
    title: str,
    ylabel_prefix: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=120, sharex=True)
    t_cmd = np.array([t for t, _ in cmd], dtype=np.float64) if cmd else np.array([])
    q_cmd = np.stack([q for _, q in cmd], axis=0) if cmd else np.zeros((0, 7))
    t_meas = np.array([t for t, _ in meas], dtype=np.float64) if meas else np.array([])
    q_meas = np.stack([q for _, q in meas], axis=0) if meas else np.zeros((0, 7))
    for j, ax in enumerate(axes):
        if len(t_cmd):
            ax.plot(t_cmd, q_cmd[:, j], color="C0", lw=1.0, label="published (joint_cmd)")
        if len(t_meas):
            ax.plot(t_meas, q_meas[:, j], color="C1", lw=0.9, alpha=0.85, label="actual (joint_states)")
        ax.set_ylabel(f"{ylabel_prefix}{j + 1}")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _tracking_error_plot(
    cmd: Sequence[Tuple[float, np.ndarray]],
    meas: Sequence[Tuple[float, np.ndarray]],
    *,
    title: str,
    out_path: Path,
) -> None:
    if not cmd or not meas:
        return
    t_cmd = np.array([t for t, _ in cmd], dtype=np.float64)
    q_cmd = np.stack([q for _, q in cmd], axis=0)
    errs = []
    for t, q in zip(t_cmd, q_cmd):
        m = _nearest(meas, float(t))
        if m is None:
            continue
        errs.append((t, np.abs(q - m)))
    if not errs:
        return
    t_err = np.array([e[0] for e in errs])
    e = np.stack([e[1] for e in errs], axis=0)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=120, sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t_err, e[:, j], color="C3", lw=0.9)
        ax.set_ylabel(f"|cmd-meas| j{j + 1}")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_published_vs_actual(data: Dict[str, Any], out: Path) -> None:
    """Published joint_cmd / gripper cmd vs measured joint_states."""
    out.mkdir(parents=True, exist_ok=True)
    _arm_publish_vs_actual_plot(
        data["left_cmd"],
        data["left_meas"],
        title="Left arm: published joint_cmd vs actual joint_states",
        ylabel_prefix="L_j",
        out_path=out / "published_vs_actual_left_arm.png",
    )
    _arm_publish_vs_actual_plot(
        data["right_cmd"],
        data["right_meas"],
        title="Right arm: published joint_cmd vs actual joint_states",
        ylabel_prefix="R_j",
        out_path=out / "published_vs_actual_right_arm.png",
    )
    _tracking_error_plot(
        data["left_cmd"],
        data["left_meas"],
        title="Left arm tracking error |published - actual|",
        out_path=out / "tracking_error_left_arm.png",
    )
    _tracking_error_plot(
        data["right_cmd"],
        data["right_meas"],
        title="Right arm tracking error |published - actual|",
        out_path=out / "tracking_error_right_arm.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=120, sharex=True)
    for ax, side, cmd, meas in (
        (axes[0], "left", data["left_grip_cmd"], data["left_grip_meas"]),
        (axes[1], "right", data["right_grip_cmd"], data["right_grip_meas"]),
    ):
        if cmd:
            tc = [t for t, _ in cmd]
            vc = [v for _, v in cmd]
            ax.plot(tc, vc, color="C0", lw=1.0, label=f"{side} grip cmd (trigger)")
        if meas:
            tm = [t for t, _ in meas]
            vm = [v for _, v in meas]
            ax.plot(tm, vm, color="C1", lw=0.9, alpha=0.85, label=f"{side} grip state (pos)")
        ax.set_ylabel(side)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_title("Gripper: published cmd vs actual joint_states")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "published_vs_actual_grippers.png")
    plt.close(fig)


def plot_policy_vs_published_vs_actual(data: Dict[str, Any], out: Path) -> None:
    """Policy action_chunk (20 Hz) vs published cmd vs actual at infer boundaries."""
    out.mkdir(parents=True, exist_ok=True)
    infs = data["inferences"]
    if not infs:
        return
    control_hz = float(data["control_hz"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=120, sharex=True)
    arm_data = (
        (data["left_cmd"], data["left_meas"], 0),
        (data["right_cmd"], data["right_meas"], 8),
    )
    labels = ("L_joint1 (q0)", "R_joint1 (q0)")

    chunk_l: List[float] = []
    chunk_r: List[float] = []
    chunk_t_l: List[float] = []
    chunk_t_r: List[float] = []
    infer_t: List[float] = []
    infer_l: List[float] = []
    infer_r: List[float] = []
    for inf in infs:
        t0 = inf["t_sec"]
        infer_t.append(t0)
        infer_l.append(float(inf["L_q"][0]))
        infer_r.append(float(inf["R_q"][0]))
        for step in range(inf["chunk"].shape[0]):
            t_step = t0 + step / control_hz
            chunk_t_l.append(t_step)
            chunk_l.append(float(inf["chunk"][step, 0]))
            chunk_t_r.append(t_step)
            chunk_r.append(float(inf["chunk"][step, 8]))

    for ax, (cmd, meas, _), label, ct, cv, it, iv in zip(
        axes,
        arm_data,
        labels,
        (chunk_t_l, chunk_t_r),
        (chunk_l, chunk_r),
        (infer_t, infer_t),
        (infer_l, infer_r),
    ):
        if cmd:
            t_cmd = np.array([t for t, _ in cmd], dtype=np.float64)
            q_cmd = np.stack([q for _, q in cmd], axis=0)
            ax.plot(t_cmd, q_cmd[:, 0], color="C0", lw=0.8, alpha=0.7, label="published cmd q0")
        if meas:
            t_meas = np.array([t for t, _ in meas], dtype=np.float64)
            q_meas = np.stack([q for _, q in meas], axis=0)
            ax.plot(t_meas, q_meas[:, 0], color="C1", lw=0.7, alpha=0.7, label="actual q0")
        if ct:
            ax.scatter(ct, cv, s=6, c="C3", alpha=0.5, label="policy chunk @20Hz", zorder=4)
        if it:
            ax.scatter(it, iv, s=28, c="C4", marker="x", label="policy step0 @infer", zorder=5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_title("Policy action_chunk vs published cmd vs actual (joint q0)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "policy_chunk_vs_published_vs_actual_q0.png")
    plt.close(fig)

    # Per-replan: policy step0 vs nearest published vs nearest actual
    pred_cmd_err = []
    pred_meas_err = []
    pub_meas_err = []
    for inf in infs:
        t = inf["t_sec"]
        cmd = _nearest(data["left_cmd"], t)
        meas = _nearest(data["left_meas"], t)
        if cmd is None or meas is None:
            continue
        p0 = float(inf["L_q"][0])
        pred_cmd_err.append(abs(p0 - float(cmd[0])))
        pred_meas_err.append(abs(p0 - float(meas[0])))
        pub_meas_err.append(abs(float(cmd[0]) - float(meas[0])))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    labels = ["|policy-meas|", "|published-meas|", "|policy-published|"]
    vals = [
        float(np.mean(pred_meas_err)) if pred_meas_err else 0.0,
        float(np.mean(pub_meas_err)) if pub_meas_err else 0.0,
        float(np.mean(pred_cmd_err)) if pred_cmd_err else 0.0,
    ]
    ax.bar(labels, vals, color=["C3", "C0", "C2"])
    ax.set_ylabel("mean abs error (rad) @ infer, L_q0")
    ax.set_title("Action alignment at replan times (L_joint1 q0)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "action_alignment_errors_L_q0.png")
    plt.close(fig)


def plot_ee_poses(data: Dict[str, Any], out: Path) -> None:
    """End-effector pose from /{side}_arm/current_ee_pose: components + 3D path."""
    if not data["left_ee"] and not data["right_ee"]:
        return
    out.mkdir(parents=True, exist_ok=True)
    spans = _fsm_spans(data["fsm"])

    for side, color in (("left", "C0"), ("right", "C1")):
        series = data[f"{side}_ee"]
        if not series:
            continue
        t = np.array([x[0] for x in series], dtype=np.float64)
        p = np.stack([x[1] for x in series], axis=0)
        rpy = _quat_to_euler(p[:, 3:7])
        rows = [
            ("x (m)", p[:, 0]),
            ("y (m)", p[:, 1]),
            ("z (m)", p[:, 2]),
            ("roll (rad)", rpy[:, 0]),
            ("pitch (rad)", rpy[:, 1]),
            ("yaw (rad)", rpy[:, 2]),
        ]
        fig, axes = plt.subplots(6, 1, figsize=(12, 13), dpi=120, sharex=True)
        for j, (ax, (lbl, v)) in enumerate(zip(axes, rows)):
            _shade_fsm(ax, spans, label=(j == 0))
            ax.plot(t, v, color=color, lw=0.9)
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.legend(loc="upper right", fontsize=7)
        axes[0].set_title(
            f"{side.capitalize()} arm end-effector pose (/{side}_arm/current_ee_pose)"
        )
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out / f"ee_pose_{side}.png")
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=120, sharex=True)
    for side, color in (("left", "C0"), ("right", "C1")):
        series = data[f"{side}_ee"]
        if not series:
            continue
        t = np.array([x[0] for x in series], dtype=np.float64)
        p = np.stack([x[1] for x in series], axis=0)
        for j, ax in enumerate(axes):
            ax.plot(t, p[:, j], color=color, lw=0.9, label=side)
    for j, ax in enumerate(axes):
        _shade_fsm(ax, spans, label=(j == 0))
        ax.set_ylabel(f"{'xyz'[j]} (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_title("End-effector position: left vs right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "ee_pose_xyz_both.png")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7), dpi=120)
    ax3d = fig.add_subplot(111, projection="3d")
    for side, color in (("left", "C0"), ("right", "C1")):
        series = data[f"{side}_ee"]
        if not series:
            continue
        p = np.stack([x[1] for x in series], axis=0)
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=0.7, alpha=0.8, label=side)
        ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], color=color, marker="o", s=32)
        ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], color=color, marker="^", s=32)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("End-effector 3D trajectory (o = start, ^ = end)")
    ax3d.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ee_trajectory_3d.png")
    plt.close(fig)


def plot_cameras(data: Dict[str, Any], out: Path) -> None:
    """Camera stream health: frame intervals, payload size, publish latency."""
    cams = {k: v for k, v in data["cam_meta"].items() if v}
    if not cams:
        return
    out.mkdir(parents=True, exist_ok=True)
    spans = _fsm_spans(data["fsm"])
    n = len(cams)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), dpi=120, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, frames) in zip(axes, cams.items()):
        t = np.array([f[0] for f in frames], dtype=np.float64)
        _shade_fsm(ax, spans, label=False)
        if t.size > 1:
            dt_ms = np.diff(t) * 1000.0
            med = float(np.median(dt_ms))
            ax.plot(t[1:], dt_ms, color="C0", lw=0.7)
            ax.axhline(
                med,
                color="C3",
                ls="--",
                lw=1.0,
                label=f"median {med:.1f} ms ({1000.0 / med:.1f} fps)" if med > 0 else "median",
            )
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylabel(f"{name}\ndt (ms)")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Camera frame intervals (gaps = dropped / stalled frames)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "camera_frame_intervals.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=120)
    _shade_fsm(ax, spans, label=True)
    for i, (name, frames) in enumerate(cams.items()):
        t = np.array([f[0] for f in frames], dtype=np.float64)
        kb = np.array([f[1] for f in frames], dtype=np.float64) / 1024.0
        ax.plot(t, kb, color=f"C{i}", lw=0.7, alpha=0.85, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("JPEG size (KB)")
    ax.set_title("Camera frame payload size")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "camera_frame_sizes.png")
    plt.close(fig)

    has_lat = any(any(f[3] is not None for f in fr) for fr in cams.values())
    if has_lat:
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=120)
        _shade_fsm(ax, spans, label=True)
        for i, (name, frames) in enumerate(cams.items()):
            pts = [(f[0], f[3]) for f in frames if f[3] is not None]
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=f"C{i}",
                lw=0.7,
                alpha=0.85,
                label=f"{name} (mean {np.mean([p[1] for p in pts]):.1f} ms)",
            )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("bag_time - header_stamp (ms)")
        ax.set_title("Camera publish latency (capture stamp to bag write)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "camera_latency.png")
        plt.close(fig)


def plot_fsm(data: Dict[str, Any], out: Path) -> None:
    """FSM enable flag with the replan times overlaid."""
    fsm = data["fsm"]
    if not fsm:
        return
    out.mkdir(parents=True, exist_ok=True)
    t = np.array([x[0] for x in fsm], dtype=np.float64)
    v = np.array([x[1] for x in fsm], dtype=np.float64)
    spans = _fsm_spans(fsm)

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=120)
    _shade_fsm(ax, spans, label=True)
    ax.step(t, v, where="post", color="C2", lw=1.1, label="/api/fsm/enable")
    infs = data["inferences"]
    if infs:
        ax.scatter(
            [i["t_sec"] for i in infs],
            np.full(len(infs), 1.05),
            marker="|",
            s=60,
            color="C3",
            label="replan",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("enable")
    ax.set_ylim(-0.15, 1.25)
    total_on = sum(b - a for a, b in spans)
    ax.set_title(
        f"FSM enable ({len(spans)} active window(s), {total_on:.1f}s enabled)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fsm_enable.png")
    plt.close(fig)


def extract_cameras(
    bag: Path,
    out: Path,
    cam_meta: Dict[str, List[Tuple[float, int, str, Optional[float]]]],
    *,
    dump_frames: bool = False,
    video: bool = False,
    video_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> None:
    """Optionally write per-frame JPEGs and/or an mp4 per camera."""
    if not (dump_frames or video):
        return
    cv2 = None
    if video:
        try:
            import cv2 as _cv2

            cv2 = _cv2
        except ImportError:
            print("[camera] opencv-python not installed; skipping --video")
            video = False
    if not (dump_frames or video):
        return

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    counts = {name: 0 for name in CAMERA_TOPICS.values()}
    writers: Dict[str, Any] = {}
    frame_dirs: Dict[str, Path] = {}
    if dump_frames:
        for name in CAMERA_TOPICS.values():
            d = out / "frames" / name
            d.mkdir(parents=True, exist_ok=True)
            frame_dirs[name] = d

    with AnyReader([bag], default_typestore=typestore) as reader:
        for conn, _t_ns, raw in reader.messages():
            name = CAMERA_TOPICS.get(conn.topic)
            if name is None:
                continue
            if max_frames is not None and counts[name] >= max_frames:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            buf = np.asarray(msg.data, dtype=np.uint8)
            idx = counts[name]
            counts[name] += 1
            if dump_frames:
                fmt = str(getattr(msg, "format", "")).lower()
                ext = "png" if "png" in fmt else "jpg"
                (frame_dirs[name] / f"{idx:06d}.{ext}").write_bytes(buf.tobytes())
            if video:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                writer = writers.get(name)
                if writer is None:
                    fps = video_fps or _camera_fps(cam_meta.get(name, []))
                    writer = cv2.VideoWriter(
                        str(out / f"video_{name}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        (img.shape[1], img.shape[0]),
                    )
                    writers[name] = writer
                writer.write(img)

    for writer in writers.values():
        writer.release()
    for name, cnt in counts.items():
        if cnt:
            what = []
            if dump_frames:
                what.append(f"frames/{name}/")
            if video:
                what.append(f"video_{name}.mp4")
            print(f"[camera] {name}: {cnt} frames -> {', '.join(what)}")


def plot_offline_pred_vs_recorded(export_dir: Path, pred_dir: Path, out: Path) -> None:
    """Compare offline re-infer pred_action vs online recorded action_chunk per sample."""
    samples_path = export_dir / "samples.jsonl"
    if not samples_path.is_file() or not pred_dir.is_dir():
        return
    out.mkdir(parents=True, exist_ok=True)
    import json

    rows = [json.loads(l) for l in samples_path.read_text().splitlines()]
    pairs = []
    for row in rows:
        sid = int(row["id"])
        rec_path = export_dir / row["dir"] / "recorded_action.npy"
        pred_path = pred_dir / f"{sid:06d}" / "pred_action.npy"
        if not rec_path.is_file() or not pred_path.is_file():
            continue
        rec = np.load(rec_path).astype(np.float32)
        pred = np.load(pred_path).astype(np.float32)
        t = min(rec.shape[0], pred.shape[0])
        d = min(rec.shape[1], pred.shape[1])
        pairs.append((sid, rec[:t, :d], pred[:t, :d]))

    if not pairs:
        print("[action_viz] no offline pred/recorded pairs found")
        return

    # Mean L1 per action dim across all samples.
    dim_err = np.zeros(16, dtype=np.float64)
    for _, rec, pred in pairs:
        dim_err += np.mean(np.abs(pred - rec), axis=0)
    dim_err /= len(pairs)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    ax.bar(range(16), dim_err, color="C0")
    ax.set_xticks(range(16))
    ax.set_xticklabels(ACTION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean |offline_pred - online_recorded|")
    ax.set_title(f"Offline vs online action (n={len(pairs)} replans)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "offline_pred_vs_recorded_mean_l1.png")
    plt.close(fig)

    # Horizon trajectories for L_q0 and R_q0 (first 12 replans).
    show = pairs[: min(12, len(pairs))]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=120, sharex=True)
    for sid, rec, pred in show:
        axes[0].plot(rec[:, 0], color="C1", alpha=0.35, lw=0.9)
        axes[0].plot(pred[:, 0], color="C3", alpha=0.35, lw=0.9)
        axes[1].plot(rec[:, 8], color="C1", alpha=0.35, lw=0.9)
        axes[1].plot(pred[:, 8], color="C3", alpha=0.35, lw=0.9)
    axes[0].plot([], [], color="C1", label="online recorded")
    axes[0].plot([], [], color="C3", label="offline pred")
    axes[0].set_ylabel("L_q0")
    axes[1].set_ylabel("R_q0")
    axes[0].set_title("Action horizon: online recorded vs offline pred (first replans)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("policy step")
    fig.tight_layout()
    fig.savefig(out / "offline_pred_vs_recorded_horizon_q0.png")
    plt.close(fig)

    _write_csv(
        out / "offline_pred_vs_recorded_per_dim.csv",
        ["dim", "label", "mean_l1"],
        [[i, ACTION_LABELS[i], float(dim_err[i])] for i in range(16)],
    )
    print(f"[action_viz] offline pred vs recorded -> {out}")


def plot_all(data: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = data["joint_cmd"]
    meas = data["joint_states"]
    infs = data["inferences"]
    n_replan = len(infs)
    spans = _fsm_spans(data.get("fsm", []))

    t_cmd = np.array([t for t, _ in cmd], dtype=np.float64) if cmd else np.array([])
    q_cmd = np.stack([q for _, q in cmd], axis=0) if cmd else np.zeros((0, 7))
    t_meas = np.array([t for t, _ in meas], dtype=np.float64) if meas else np.array([])
    q_meas = np.stack([q for _, q in meas], axis=0) if meas else np.zeros((0, 7))
    t_pred = np.array([inf["t_sec"] for inf in infs], dtype=np.float64)
    q_pred = np.stack([inf["L_q"] for inf in infs], axis=0) if infs else np.zeros((0, 7))
    ms = np.array([inf["inference_ms"] for inf in infs], dtype=np.float64)
    Lg = np.array([inf["L_grip_eff"] for inf in infs], dtype=np.float64)
    Rg = np.array([inf["R_grip_eff"] for inf in infs], dtype=np.float64)

    # --- compare joint 0 ---
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
    _shade_fsm(ax, spans, label=True)
    if len(t_cmd):
        ax.plot(t_cmd, q_cmd[:, 0], color="C0", lw=1.0, label="/api/left_arm/joint_cmd q0")
    if len(t_meas):
        ax.plot(t_meas, q_meas[:, 0], color="C1", lw=1.2, label="/left_arm/joint_states q0")
    if len(t_pred):
        ax.scatter(
            t_pred,
            q_pred[:, 0],
            color="C3",
            s=22,
            zorder=5,
            label="action_chunk step0 pred q0",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L_joint1 (rad)")
    ax.set_title("Left arm joint 0: predicted vs commanded vs measured")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_L_joint0.png")
    plt.close(fig)

    # --- all 7 joints ---
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=120, sharex=True)
    for j, ax in enumerate(axes):
        _shade_fsm(ax, spans, label=(j == 0))
        if len(t_cmd):
            ax.plot(t_cmd, q_cmd[:, j], color="C0", lw=0.9, label="cmd")
        if len(t_meas):
            ax.plot(t_meas, q_meas[:, j], color="C1", lw=1.0, label="meas")
        if len(t_pred):
            ax.scatter(t_pred, q_pred[:, j], color="C3", s=12, zorder=5, label="pred@infer")
        ax.set_ylabel(f"L_j{j+1}")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax.set_title("Left arm: cmd vs meas vs action_chunk step0")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "compare_L_all_joints.png")
    plt.close(fig)

    # --- gripper effort step0 ---
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
    if len(t_pred):
        ax.plot(t_pred, Lg, "o-", color="C0", ms=4, lw=1.0, label="L grip effort (pred step0)")
        ax.plot(t_pred, Rg, "s-", color="C1", ms=4, lw=1.0, label="R grip effort (pred step0)")
    ax.axhline(
        CLOSE_THRESH_NM,
        color="gray",
        ls="--",
        lw=1.2,
        label=f"close thresh ({CLOSE_THRESH_NM} Nm)",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("effort (Nm)")
    ax.set_title("Gripper effort in action_chunk (first step per replan)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gripper_effort_pred.png")
    plt.close(fig)

    # --- inference latency ---
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=140)
    if len(t_pred):
        ax.plot(t_pred, ms, "o-", color="C0", ms=4, lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("inference_ms")
    ax.set_title(f"Inference latency ({n_replan} replans)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "inference_ms.png")
    plt.close(fig)


def _print_stream_summary(data: Dict[str, Any]) -> None:
    """Cameras, end-effector poses and FSM state."""
    for name, frames in data["cam_meta"].items():
        if not frames:
            print(f"camera {name}: no frames")
            continue
        row = _camera_summary_row(name, frames)
        lat = f" lat_mean={row[8]}ms" if row[8] != "" else ""
        print(
            f"camera {name}: n={row[1]} dur={row[2]}s fps={row[3]} "
            f"dt_p50={row[4]}ms p95={row[5]}ms max={row[6]}ms "
            f"size_mean={row[7]}KB{lat}"
        )

    for side in ("left", "right"):
        series = data[f"{side}_ee"]
        if not series:
            print(f"{side} ee_pose: no samples")
            continue
        p = np.stack([x[1] for x in series], axis=0)
        span = p[:, :3].max(axis=0) - p[:, :3].min(axis=0)
        path_len = float(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1).sum())
        print(
            f"{side} ee_pose: n={len(series)} "
            f"xyz_range=[{span[0]:.3f},{span[1]:.3f},{span[2]:.3f}]m "
            f"path_len={path_len:.3f}m"
        )

    fsm = data["fsm"]
    if fsm:
        spans = _fsm_spans(fsm)
        total_on = sum(b - a for a, b in spans)
        total = fsm[-1][0] - fsm[0][0]
        pct = (total_on / total * 100.0) if total > 0 else 0.0
        print(
            f"fsm enable: n={len(fsm)} windows={len(spans)} "
            f"enabled={total_on:.1f}s / {total:.1f}s ({pct:.1f}%)"
        )
    else:
        print("fsm enable: no samples")


def print_summary(data: Dict[str, Any]) -> None:
    _print_stream_summary(data)
    infs = data["inferences"]
    if not infs:
        print("No inferences found.")
        return
    ms = np.array([i["inference_ms"] for i in infs])
    Lg = np.array([i["L_grip_eff"] for i in infs])
    Rg = np.array([i["R_grip_eff"] for i in infs])
    align_path_vals = []
    for inf in infs:
        cmd = _nearest(data["joint_cmd"], inf["t_sec"])
        meas = _nearest(data["joint_states"], inf["t_sec"])
        if cmd is None or meas is None:
            continue
        pred0 = float(inf["L_q"][0])
        align_path_vals.append(
            (abs(pred0 - float(meas[0])), abs(float(cmd[0]) - float(meas[0])))
        )
    print(
        f"replans={len(infs)}  t=[{infs[0]['t_sec']:.3f},{infs[-1]['t_sec']:.3f}]s  "
        f"infer_ms mean={ms.mean():.1f} median={np.median(ms):.1f} "
        f"min={ms.min():.1f} max={ms.max():.1f}"
    )
    print(
        f"gripper step0 below {CLOSE_THRESH_NM}: "
        f"L={(Lg < CLOSE_THRESH_NM).mean()*100:.1f}%  "
        f"R={(Rg < CLOSE_THRESH_NM).mean()*100:.1f}%"
    )
    if align_path_vals:
        pm = np.mean([a[0] for a in align_path_vals])
        cm = np.mean([a[1] for a in align_path_vals])
        print(f"at infer times |pred-meas| q0 mean={pm:.4f}  |cmd-meas| q0 mean={cm:.4f}")


def run_action_analysis(
    bag: Path,
    out: Path,
    *,
    control_hz: float = 20.0,
    export_dir: Optional[Path] = None,
    pred_dir: Optional[Path] = None,
    dump_frames: bool = False,
    video: bool = False,
    video_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Full action analysis: published vs actual + policy chunk + optional offline compare."""
    out.mkdir(parents=True, exist_ok=True)
    data = load_bag(bag, control_hz=control_hz)
    export_csvs(data, out)
    plot_all(data, out)
    plot_published_vs_actual(data, out)
    plot_policy_vs_published_vs_actual(data, out)
    plot_ee_poses(data, out)
    plot_cameras(data, out)
    plot_fsm(data, out)
    extract_cameras(
        bag,
        out,
        data["cam_meta"],
        dump_frames=dump_frames,
        video=video,
        video_fps=video_fps,
        max_frames=max_frames,
    )
    if export_dir is not None and pred_dir is not None:
        plot_offline_pred_vs_recorded(export_dir, pred_dir, out)
    print_summary(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag2 directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output analysis directory (default: <bag>_analysis next to bag)",
    )
    parser.add_argument("--control-hz", type=float, default=20.0,
                        help="policy rate used to place action_chunk steps on the time axis "
                             "(20 for 20 fps bags; 45 for legacy 45 fps ones)")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="bag_offline_out/.../export for offline pred vs recorded compare",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="bag_offline_out/.../predictions for offline pred vs recorded compare",
    )
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="write every camera frame as jpg under <out>/frames/<camera>/",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="encode one mp4 per camera (needs opencv-python)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="mp4 playback rate (default: measured median rate per camera)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap frames per camera for --dump-frames/--video",
    )
    args = parser.parse_args()

    bag = args.bag.resolve()
    out = args.out.resolve() if args.out else bag.parent / f"{bag.name}_analysis"
    export_dir = args.export_dir.resolve() if args.export_dir else None
    pred_dir = args.pred_dir.resolve() if args.pred_dir else None

    print(f"bag={bag}")
    print(f"out={out}")
    run_action_analysis(
        bag,
        out,
        control_hz=args.control_hz,
        export_dir=export_dir,
        pred_dir=pred_dir,
        dump_frames=args.dump_frames,
        video=args.video,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
    )
    print(f"wrote analysis -> {out}")


if __name__ == "__main__":
    main()
