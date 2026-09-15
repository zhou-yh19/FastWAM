#!/usr/bin/env python3
"""Per-joint / per-gripper error report for a FastWAM TA2 deploy bag.

Three independent error layers are separated, because they have different causes:

  A. Servo tracking      published /api/*/joint_cmd  vs  measured /*/joint_states
                         -> how well the arm follows what we told it (robot-side)
  B. Policy open loop    action_chunk[k]             vs  measured at t_infer + k/20
                         -> how well the model predicts the future (model-side),
                            benchmarked against a persistence (hold) baseline
  C. Gripper             policy effort -> trigger cmd -> measured position
                         -> force command vs position feedback, so compared as
                            binary open/close agreement + transition latency

Example:
  cd <FastWAM repo root>
  conda activate fastwam
  python experiments/teleavatar_v2_deploy/server/joint_error_report.py \\
    --bag ../fastwam_bags/fastwam_ta2_20260818_142420 \\
    --out ../fastwam_bags/fastwam_ta2_20260818_142420_offline/joint_errors
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_deploy_bag import load_bag  # noqa: E402

RAD2DEG = 180.0 / np.pi
ARM_LABELS = [f"L_j{i}" for i in range(1, 8)] + [f"R_j{i}" for i in range(1, 8)]
# Gripper trigger >= this means "commanded to close" (effort <= 0).
# See gripper_effort_to_trigger in fastwam/datasets/lerobot/transforms/teleavatar.py
CLOSE_TRIGGER = 0.10


# --------------------------------------------------------------------------- utils


def _series(pairs: Sequence[Tuple[float, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """[(t, vec)] -> (t[N], v[N, D]) sorted by t."""
    if not pairs:
        return np.zeros(0), np.zeros((0, 0))
    t = np.asarray([p[0] for p in pairs], dtype=np.float64)
    v = np.stack([np.atleast_1d(np.asarray(p[1], dtype=np.float64)) for p in pairs], axis=0)
    order = np.argsort(t)
    return t[order], v[order]


def _interp(t_query: np.ndarray, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    """Linear interpolation of a [N, D] series onto t_query. 200 Hz feedback -> exact enough."""
    if t_src.size == 0:
        return np.full((t_query.size, v_src.shape[1] if v_src.ndim > 1 else 1), np.nan)
    out = np.empty((t_query.size, v_src.shape[1]), dtype=np.float64)
    for d in range(v_src.shape[1]):
        out[:, d] = np.interp(t_query, t_src, v_src[:, d])
    # Mark extrapolation as NaN so it never silently pollutes the statistics.
    bad = (t_query < t_src[0]) | (t_query > t_src[-1])
    out[bad] = np.nan
    return out


def _stats(err: np.ndarray) -> Dict[str, float]:
    """Signed error vector -> the usual descriptive set. NaNs dropped."""
    e = err[np.isfinite(err)]
    if e.size == 0:
        return {k: float("nan") for k in ("n", "bias", "mae", "med", "rms", "p95", "max")}
    a = np.abs(e)
    return {
        "n": float(e.size),
        "bias": float(e.mean()),
        "mae": float(a.mean()),
        "med": float(np.median(a)),
        "rms": float(np.sqrt((e**2).mean())),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def _skill(mae_deg: float, hold_deg: float, floor_deg: float = 0.05) -> float:
    """Skill score vs the persistence baseline, in percent.

    Undefined at k=0, where the baseline is 'the state we just measured' and its
    error is identically zero -- dividing by it would print a meaningless -1e11%.
    """
    if not np.isfinite(hold_deg) or hold_deg < floor_deg:
        return float("nan")
    return 100.0 * (1.0 - mae_deg / hold_deg)


# ----------------------------------------------------------------- A. servo tracking


def servo_tracking(data: Dict[str, Any], lag_ms_grid: np.ndarray) -> Dict[str, Any]:
    """Published joint_cmd vs measured joint_states, per joint, with lag compensation."""
    out: Dict[str, Any] = {"per_joint": [], "lag_curves": {}}
    for side, cmd_key, meas_key, off in (("L", "left_cmd", "left_meas", 0),
                                         ("R", "right_cmd", "right_meas", 7)):
        t_cmd, q_cmd = _series(data[cmd_key])
        t_meas, q_meas = _series(data[meas_key])
        if t_cmd.size == 0 or t_meas.size == 0:
            continue

        raw = q_cmd - _interp(t_cmd, t_meas, q_meas)  # [N, 7] signed, rad

        # Sweep a pure time shift: cmd(t) vs meas(t + tau). The minimiser is the
        # effective actuation delay; the residual is tracking error that a delay
        # cannot explain (stiffness, saturation, model error).
        rms_curve = np.zeros((lag_ms_grid.size, 7))
        for li, lag_ms in enumerate(lag_ms_grid):
            shifted = _interp(t_cmd + lag_ms / 1000.0, t_meas, q_meas)
            e = q_cmd - shifted
            for j in range(7):
                col = e[:, j][np.isfinite(e[:, j])]
                rms_curve[li, j] = np.sqrt((col**2).mean()) if col.size else np.nan
        out["lag_curves"][side] = (lag_ms_grid, rms_curve)

        for j in range(7):
            s = _stats(raw[:, j])
            best = int(np.nanargmin(rms_curve[:, j]))
            out["per_joint"].append({
                "joint": ARM_LABELS[off + j],
                **s,
                "best_lag_ms": float(lag_ms_grid[best]),
                "rms_after_lag": float(rms_curve[best, j]),
                "rms_explained_pct": float(
                    100.0 * (1.0 - rms_curve[best, j] / s["rms"]) if s["rms"] > 0 else np.nan
                ),
            })
    return out


# ------------------------------------------------------- B. policy open-loop accuracy


def policy_open_loop(data: Dict[str, Any], control_hz: float) -> Dict[str, Any]:
    """action_chunk[k] vs the joint state actually reached at t_infer + k/control_hz.

    Compared against a persistence baseline (hold the state observed at t_infer),
    which is the bar a useful policy has to clear.
    """
    infs = data["inferences"]
    t_lm, q_lm = _series(data["left_meas"])
    t_rm, q_rm = _series(data["right_meas"])
    if not infs or t_lm.size == 0 or t_rm.size == 0:
        return {}

    horizon = int(infs[0]["chunk"].shape[0])
    n_inf = len(infs)
    err = np.full((n_inf, horizon, 14), np.nan)
    hold = np.full((n_inf, horizon, 14), np.nan)

    t_inf = np.array([inf["t_sec"] for inf in infs])
    base_l = _interp(t_inf, t_lm, q_lm)   # state at replan time
    base_r = _interp(t_inf, t_rm, q_rm)

    for i, inf in enumerate(infs):
        chunk = inf["chunk"]
        t_k = inf["t_sec"] + np.arange(horizon) / control_hz
        meas = np.concatenate([_interp(t_k, t_lm, q_lm), _interp(t_k, t_rm, q_rm)], axis=1)
        pred = np.concatenate([chunk[:horizon, 0:7], chunk[:horizon, 8:15]], axis=1)
        err[i] = pred - meas
        hold[i] = np.concatenate([base_l[i], base_r[i]])[None, :] - meas

    # How much of the horizon was actually executed before the next replan.
    periods = np.diff(t_inf)
    n_exec = int(round(float(np.median(periods)) * control_hz)) if periods.size else horizon

    return {
        "err": err, "hold": hold, "horizon": horizon,
        "n_exec": min(n_exec, horizon), "n_inf": n_inf,
        "mae_k_joint": np.nanmean(np.abs(err), axis=0),    # [horizon, 14]
        "hold_k_joint": np.nanmean(np.abs(hold), axis=0),
    }


# ------------------------------------------------------------------------ C. gripper


def gripper_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger command (force) vs position feedback.

    The gripper is force controlled, so position does NOT go to zero on a successful
    grasp: it stops at the width of the grasped object. The measured position is in
    fact trimodal -- open (~1.0), holding a block (~0.5), closed empty (~0.0) -- so a
    single "is closed" threshold would score every successful grasp as a failure.
    Thresholds are therefore derived per side from the observed open extreme, and the
    close events are additionally classified into grasped-object vs closed-empty.
    """
    res: Dict[str, Any] = {"sides": {}}
    for side, cmd_key, meas_key, eff_key in (
        ("left", "left_grip_cmd", "left_grip_meas", "L_grip_eff"),
        ("right", "right_grip_cmd", "right_grip_meas", "R_grip_eff"),
    ):
        t_c, v_c = _series([(t, np.array([v])) for t, v in data[cmd_key]])
        t_m, v_m = _series([(t, np.array([v])) for t, v in data[meas_key]])
        if t_c.size == 0 or t_m.size == 0:
            continue
        trig = v_c[:, 0]
        pos = v_m[:, 0]
        pos_open = float(np.nanpercentile(pos, 99))
        th_engage = 0.75 * pos_open   # below this the jaws have actually moved in
        th_empty = 0.20 * pos_open    # below this nothing is held

        pos_at_cmd = _interp(t_c, t_m, v_m)[:, 0]
        cmd_closed = trig >= CLOSE_TRIGGER
        meas_engaged = pos_at_cmd <= th_engage
        ok = np.isfinite(pos_at_cmd)
        agree = float((cmd_closed[ok] == meas_engaged[ok]).mean() * 100.0)
        tp = int((cmd_closed[ok] & meas_engaged[ok]).sum())
        tn = int((~cmd_closed[ok] & ~meas_engaged[ok]).sum())
        fp = int((cmd_closed[ok] & ~meas_engaged[ok]).sum())
        fn = int((~cmd_closed[ok] & meas_engaged[ok]).sum())

        # Three-way occupancy over the whole session.
        fin = np.isfinite(pos)
        state_open = float((pos[fin] > th_engage).mean() * 100.0)
        state_grasp = float(((pos[fin] <= th_engage) & (pos[fin] > th_empty)).mean() * 100.0)
        state_closed = float((pos[fin] <= th_empty).mean() * 100.0)

        # Correlation between commanded closing force and how far the jaws travelled.
        travel = np.clip(1.0 - pos_at_cmd / max(pos_open, 1e-6), 0.0, 1.0)
        best_r, best_lag = -2.0, 0.0
        for lag in np.arange(-100, 1001, 20):
            shifted = _interp(t_c + lag / 1000.0, t_m, v_m)[:, 0]
            tv = np.clip(1.0 - shifted / max(pos_open, 1e-6), 0.0, 1.0)
            m = np.isfinite(tv)
            if m.sum() > 10 and np.std(trig[m]) > 1e-9 and np.std(tv[m]) > 1e-9:
                r = float(np.corrcoef(trig[m], tv[m])[0, 1])
                if r > best_r:
                    best_r, best_lag = r, float(lag)

        edges_close, edges_open, grasp_events = [], [], []
        d = np.diff(cmd_closed.astype(int))
        for idx in np.flatnonzero(d == 1):
            t0 = t_c[idx + 1]
            fut = (t_m >= t0) & (t_m <= t0 + 3.0)
            if not fut.any():
                continue
            pf, tf = pos[fut], t_m[fut]
            hit = np.flatnonzero(pf <= th_engage)
            if hit.size:
                edges_close.append((tf[hit[0]] - t0) * 1000.0)
            # Settled position ~1 s after the command tells us what was grasped.
            settle = (tf >= t0 + 0.8) & (tf <= t0 + 1.5)
            if settle.any():
                sp = float(np.median(pf[settle]))
                kind = ("open" if sp > th_engage
                        else "grasped_object" if sp > th_empty else "closed_empty")
                grasp_events.append((float(t0), sp, kind))
        for idx in np.flatnonzero(d == -1):
            t0 = t_c[idx + 1]
            fut = (t_m >= t0) & (t_m <= t0 + 3.0)
            if not fut.any():
                continue
            hit = np.flatnonzero(pos[fut] > th_engage)
            if hit.size:
                edges_open.append((t_m[fut][hit[0]] - t0) * 1000.0)

        eff = np.array([inf[eff_key] for inf in data["inferences"]], dtype=np.float64)
        res["sides"][side] = {
            "pos_open": pos_open, "th_engage": th_engage, "th_empty": th_empty,
            "agree_pct": agree, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "cmd_closed_pct": float(cmd_closed[ok].mean() * 100.0),
            "meas_engaged_pct": float(meas_engaged[ok].mean() * 100.0),
            "state_open_pct": state_open, "state_grasp_pct": state_grasp,
            "state_closed_pct": state_closed,
            "best_r": best_r, "best_lag_ms": best_lag,
            "close_lat_ms": np.array(edges_close),
            "open_lat_ms": np.array(edges_open),
            "n_close_edges": int((d == 1).sum()), "n_open_edges": int((d == -1).sum()),
            "n_close_responded": len(edges_close), "n_open_responded": len(edges_open),
            "grasp_events": grasp_events,
            "n_grasped_object": sum(1 for g in grasp_events if g[2] == "grasped_object"),
            "n_closed_empty": sum(1 for g in grasp_events if g[2] == "closed_empty"),
            "n_no_response": sum(1 for g in grasp_events if g[2] == "open"),
            "effort": eff,
            "trigger_series": (t_c, trig),
            "pos_series": (t_m, pos),
        }
    return res


# ---------------------------------------------------------------------------- plots


def plot_servo(servo: Dict[str, Any], out: Path) -> None:
    rows = servo["per_joint"]
    if not rows:
        return
    labels = [r["joint"] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=130)
    w = 0.26
    axes[0].bar(x - w, [r["mae"] * RAD2DEG for r in rows], w, label="MAE", color="C0")
    axes[0].bar(x, [r["rms"] * RAD2DEG for r in rows], w, label="RMS", color="C1")
    axes[0].bar(x + w, [r["p95"] * RAD2DEG for r in rows], w, label="P95", color="C3")
    axes[0].set_ylabel("tracking error (deg)")
    axes[0].set_title("A. Servo tracking error: published joint_cmd vs measured joint_states")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - w / 2, [r["rms"] * RAD2DEG for r in rows], w, label="RMS raw", color="C1")
    axes[1].bar(x + w / 2, [r["rms_after_lag"] * RAD2DEG for r in rows], w,
                label="RMS after lag compensation", color="C2")
    for i, r in enumerate(rows):
        axes[1].annotate(f"{r['best_lag_ms']:.0f}ms", (i, r["rms"] * RAD2DEG),
                         ha="center", va="bottom", fontsize=6.5)
    axes[1].set_ylabel("RMS (deg)")
    axes[1].set_title("Pure actuation delay removed -> residual is real tracking error")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "A_servo_tracking_error.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130, sharey=True)
    for ax, side in zip(axes, ("L", "R")):
        if side not in servo["lag_curves"]:
            continue
        grid, curve = servo["lag_curves"][side]
        for j in range(7):
            ax.plot(grid, curve[:, j] * RAD2DEG, lw=1.1, label=f"{side}_j{j+1}")
        ax.set_xlabel("assumed actuation delay (ms)")
        ax.set_title(f"{side} arm: RMS vs time shift")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("RMS error (deg)")
    fig.tight_layout()
    fig.savefig(out / "A_lag_sweep.png")
    plt.close(fig)


def plot_openloop(ol: Dict[str, Any], control_hz: float, out: Path) -> None:
    if not ol:
        return
    mae = ol["mae_k_joint"] * RAD2DEG      # [horizon, 14]
    hold = ol["hold_k_joint"] * RAD2DEG
    horizon, n_exec = ol["horizon"], ol["n_exec"]
    t_axis = np.arange(horizon) / control_hz

    fig, ax = plt.subplots(figsize=(13, 5), dpi=130)
    im = ax.imshow(mae.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, t_axis[-1], -0.5, 13.5])
    ax.axvline(n_exec / control_hz, color="cyan", ls="--", lw=1.5,
               label=f"median replan ({n_exec} steps)")
    ax.set_yticks(range(14))
    ax.set_yticklabels(ARM_LABELS, fontsize=8)
    ax.set_xlabel("open-loop horizon (s)")
    ax.set_title("B. Policy open-loop error |action_chunk[k] - measured| (deg)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="MAE (deg)")
    fig.tight_layout()
    fig.savefig(out / "B_openloop_error_heatmap.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
    axes[0].plot(t_axis, mae.mean(axis=1), "o-", ms=3, color="C3", label="policy")
    axes[0].plot(t_axis, hold.mean(axis=1), "s--", ms=3, color="C7", label="persistence (hold)")
    axes[0].axvline(n_exec / control_hz, color="C0", ls=":", lw=1.5, label="median replan")
    axes[0].set_xlabel("open-loop horizon (s)")
    axes[0].set_ylabel("MAE over 14 joints (deg)")
    axes[0].set_title("Policy vs persistence baseline")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # k=0's persistence baseline is identically zero, so skill is undefined there.
    skill = np.array([_skill(mae[k].mean(), hold[k].mean()) for k in range(horizon)])
    valid = np.isfinite(skill)
    axes[1].plot(t_axis[valid], skill[valid], "o-", ms=3, color="C2")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].axvline(n_exec / control_hz, color="C0", ls=":", lw=1.5)
    axes[1].fill_between(t_axis[valid], 0, skill[valid],
                         where=skill[valid] >= 0, color="C2", alpha=0.2)
    axes[1].fill_between(t_axis[valid], 0, skill[valid],
                         where=skill[valid] < 0, color="C3", alpha=0.2)
    axes[1].set_xlabel("open-loop horizon (s)")
    axes[1].set_ylabel("skill score vs hold (%)")
    axes[1].set_title("> 0 means the policy beats holding still")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "B_openloop_vs_persistence.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 7, figsize=(18, 6), dpi=120, sharex=True)
    for j in range(14):
        ax = axes[j // 7, j % 7]
        ax.plot(t_axis, mae[:, j], color="C3", lw=1.2, label="policy")
        ax.plot(t_axis, hold[:, j], color="C7", lw=1.0, ls="--", label="hold")
        ax.set_title(ARM_LABELS[j], fontsize=9)
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(fontsize=7)
        if j % 7 == 0:
            ax.set_ylabel("MAE (deg)")
        if j // 7 == 1:
            ax.set_xlabel("horizon (s)")
    fig.suptitle("B. Per-joint open-loop error growth", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "B_openloop_per_joint.png")
    plt.close(fig)


def plot_gripper(gr: Dict[str, Any], data: Dict[str, Any], out: Path) -> None:
    if not gr.get("sides"):
        return
    t_inf = np.array([inf["t_sec"] for inf in data["inferences"]])

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), dpi=130, sharex=True)
    for side, color in (("left", "C0"), ("right", "C1")):
        s = gr["sides"].get(side)
        if not s:
            continue
        tc, trig = s["trigger_series"]
        tm, pos = s["pos_series"]
        axes[0].plot(tc, trig, color=color, lw=1.0, label=f"{side} trigger cmd")
        axes[1].plot(tm, pos, color=color, lw=0.9, label=f"{side} measured position")
        axes[1].axhline(s["th_engage"], color=color, ls="--", lw=0.8, alpha=0.7)
        axes[1].axhline(s["th_empty"], color=color, ls=":", lw=0.8, alpha=0.7)
        for t0, sp, kind in s["grasp_events"]:
            axes[1].scatter([t0], [sp], s=34, zorder=5, edgecolor="k", linewidth=0.5,
                            color={"grasped_object": "lime", "closed_empty": "red",
                                   "open": "gray"}[kind])
        axes[2].plot(t_inf, s["effort"], "o-", ms=3, color=color, lw=1.0,
                     label=f"{side} policy effort (Nm)")
    axes[0].axhline(CLOSE_TRIGGER, color="gray", ls="--", lw=1,
                    label=f"close threshold ({CLOSE_TRIGGER})")
    axes[2].axhline(0.0, color="gray", ls="--", lw=1, label="effort=0 -> trigger 0.10")
    axes[0].set_ylabel("trigger [0,1]")
    axes[1].set_ylabel("position")
    axes[2].set_ylabel("effort (Nm)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title("C. Gripper: commanded force vs position feedback vs policy effort\n"
                      "dashed = engage threshold, dotted = empty threshold; "
                      "dots = settled position after a close command "
                      "(green grasped / red empty / gray no response)", fontsize=10)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "C_gripper_cmd_vs_state.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=130)
    for k, (title, key) in enumerate((("close cmd -> jaws engaged", "close_lat_ms"),
                                      ("open cmd -> jaws opened", "open_lat_ms"))):
        drew = False
        for side, color in (("left", "C0"), ("right", "C1")):
            s = gr["sides"].get(side)
            if s is None or s[key].size == 0:
                continue
            axes[k].hist(s[key], bins=20, alpha=0.55, color=color,
                         label=f"{side} (n={s[key].size}, med={np.median(s[key]):.0f}ms)")
            drew = True
        axes[k].set_xlabel("response latency (ms)")
        axes[k].set_title(title)
        axes[k].grid(True, alpha=0.3)
        if drew:
            axes[k].legend(fontsize=8)
    axes[0].set_ylabel("count")

    # Grasp outcome breakdown per side.
    sides = list(gr["sides"].keys())
    x = np.arange(len(sides))
    w = 0.26
    for off, key, lab, col in ((-w, "n_grasped_object", "grasped object", "C2"),
                               (0.0, "n_closed_empty", "closed empty", "C3"),
                               (w, "n_no_response", "no response", "C7")):
        axes[2].bar(x + off, [gr["sides"][s][key] for s in sides], w, label=lab, color=col)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sides)
    axes[2].set_ylabel("close events")
    axes[2].set_title("Outcome of each close command")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "C_gripper_latency.png")
    plt.close(fig)


# --------------------------------------------------------------------------- report


def write_report(bag: Path, data: Dict[str, Any], servo: Dict[str, Any],
                 ol: Dict[str, Any], gr: Dict[str, Any], control_hz: float,
                 out: Path) -> Dict[str, Any]:
    infs = data["inferences"]
    ms = np.array([i["inference_ms"] for i in infs]) if infs else np.zeros(0)
    periods = np.diff([i["t_sec"] for i in infs]) if len(infs) > 1 else np.zeros(0)

    L: List[str] = [f"# Joint / gripper error report - `{bag.name}`", ""]
    L.append(f"- replans: **{len(infs)}**, duration {infs[-1]['t_sec'] - infs[0]['t_sec']:.1f} s"
             if infs else "- no inference records")
    if ms.size:
        L.append(f"- online inference median **{np.median(ms):.0f} ms**, "
                 f"implied denoising steps ~ **{(np.median(ms) - 233.7) / 40.64:.1f}**")
    if periods.size:
        L.append(f"- replan period median **{np.median(periods):.3f} s**, "
                 f"action chunk covers {ol.get('horizon', 32) / control_hz:.2f} s, "
                 f"executed **{ol.get('n_exec', 0)}/{ol.get('horizon', 32)}** steps")
    L += ["", "---", "", "## A. Servo tracking error (published command vs measured feedback)", "",
          "Per-joint comparison of `/api/*/joint_cmd` against `/*/joint_states`. "
          "`best_lag` is the pure delay minimising RMS; `rms_after_lag` is the residual "
          "once that delay is removed.", ""]
    rows = []
    for r in servo["per_joint"]:
        rows.append([r["joint"], f"{r['bias'] * RAD2DEG:+.3f}", f"{r['mae'] * RAD2DEG:.3f}",
                     f"{r['med'] * RAD2DEG:.3f}", f"{r['rms'] * RAD2DEG:.3f}",
                     f"{r['p95'] * RAD2DEG:.3f}", f"{r['max'] * RAD2DEG:.3f}",
                     f"{r['best_lag_ms']:.0f}", f"{r['rms_after_lag'] * RAD2DEG:.3f}",
                     f"{r['rms_explained_pct']:.0f}%"])
    L.append(_md_table(["joint", "bias(deg)", "MAE(deg)", "median(deg)", "RMS(deg)",
                        "P95(deg)", "max(deg)", "best_lag(ms)", "RMS_after_lag(deg)",
                        "lag-explained fraction"], rows))

    if ol:
        mae = ol["mae_k_joint"] * RAD2DEG
        hold = ol["hold_k_joint"] * RAD2DEG
        n_exec = ol["n_exec"]
        L += ["", "---", "", "## B. Policy open-loop error (action_chunk step k vs measured joints at that time)", "",
              "The baseline is persistence: hold the joints as they were at replan time. "
              "Only skill > 0 means the model beats doing nothing.", ""]
        ks = [k for k in (0, 2, 5, 10, 15, 20, 31) if k < ol["horizon"]]
        rows = [["horizon step", "time(s)"] + ARM_LABELS + ["mean of 14", "hold mean", "skill%"]]
        body = []
        for k in ks:
            sk = _skill(mae[k].mean(), hold[k].mean())
            body.append([f"k={k}{' *' if k < n_exec else ''}", f"{k / control_hz:.2f}"]
                        + [f"{mae[k, j]:.2f}" for j in range(14)]
                        + [f"**{mae[k].mean():.2f}**", f"{hold[k].mean():.2f}",
                           "n/a" if not np.isfinite(sk) else f"{sk:+.0f}%"])
        L.append(_md_table(rows[0], body))
        L.append("")
        L.append(f"`*` = this step is actually executed before the next replan (median {n_exec} steps). "
                 f"Units: deg. The hold baseline is 0 at k=0, so skill is undefined there.")
        e0, ee = mae[0].mean(), mae[min(n_exec, ol["horizon"] - 1)].mean()
        L.append("")
        L.append(f"- step 0 MAE **{e0:.2f} deg**, end of execution (k={n_exec}) **{ee:.2f} deg**, "
                 f"open-loop drift **{ee / max(e0, 1e-9):.1f}x**")
        sk_exec = _skill(mae[1:n_exec + 1].mean(), hold[1:n_exec + 1].mean())
        L.append(f"- skill score against persistence over the executed range (k=1..{n_exec}): "
                 f"**{sk_exec:+.1f}%**")

    if gr.get("sides"):
        L += ["", "---", "", "## C. Gripper", "",
              "The policy outputs effort (N*m); `effort_to_trigger` maps it to a trigger in "
              "[0,1] for publishing. Feedback is position.",
              "",
              "The gripper is **force**-controlled: on a successful grasp the position settles at the "
              "object width rather than returning to zero. Measured positions are trimodal "
              "(open / holding an object / closed on nothing), so thresholds derive from each "
              "side's `pos_open` (p99):",
              "",
              "- open: `pos > 0.75*pos_open`",
              "- holding: `0.20*pos_open < pos <= 0.75*pos_open`",
              "- closed on nothing: `pos <= 0.20*pos_open`", ""]
        rows = []
        for side, s in gr["sides"].items():
            rows.append([side, f"{s['pos_open']:.3f}", f"{s['th_engage']:.3f}",
                         f"{s['th_empty']:.3f}", f"{s['state_open_pct']:.1f}%",
                         f"{s['state_grasp_pct']:.1f}%", f"{s['state_closed_pct']:.1f}%"])
        L.append(_md_table(["side", "pos_open", "engage thr", "empty thr",
                            "open %", "holding %", "empty %"], rows))

        L += ["", "### Command vs feedback agreement (close commanded vs gripper actually closing)", ""]
        rows = []
        for side, s in gr["sides"].items():
            cl, op = s["close_lat_ms"], s["open_lat_ms"]
            rows.append([side, f"{s['agree_pct']:.1f}%", f"{s['cmd_closed_pct']:.1f}%",
                         f"{s['meas_engaged_pct']:.1f}%",
                         f"{s['tp']}/{s['tn']}/{s['fp']}/{s['fn']}",
                         f"{s['best_r']:.3f}", f"{s['best_lag_ms']:.0f}",
                         f"{np.median(cl):.0f}" if cl.size else "n/a",
                         f"{np.median(op):.0f}" if op.size else "n/a"])
        L.append(_md_table(["side", "agreement", "commanded close %", "measured close %",
                            "TP/TN/FP/FN", "best r", "best lag(ms)",
                            "close latency median(ms)", "open latency median(ms)"], rows))

        L += ["", "### Outcome of each close command", ""]
        rows = []
        for side, s in gr["sides"].items():
            n = s["n_close_edges"]
            rows.append([side, n, s["n_grasped_object"], s["n_closed_empty"],
                         s["n_no_response"],
                         f"{100.0 * s['n_grasped_object'] / n:.0f}%" if n else "n/a"])
        L.append(_md_table(["side", "close commands", "held object", "closed empty", "no response",
                            "hold rate"], rows))

        rows = []
        for side, s in gr["sides"].items():
            e = s["effort"]
            rows.append([side, f"{e.min():+.3f}", f"{e.max():+.3f}", f"{e.mean():+.3f}",
                         f"{(e <= 0).mean() * 100:.1f}%"])
        L += ["", "### Gripper effort from the policy (step 0 of each replan)", ""]
        L.append(_md_table(["side", "min(Nm)", "max(Nm)", "mean(Nm)",
                            "effort<=0 % (intent to close)"], rows))

    L += ["", "---", "", "## Figures", "",
          "- `A_servo_tracking_error.png` / `A_lag_sweep.png`",
          "- `B_openloop_error_heatmap.png` / `B_openloop_vs_persistence.png` / `B_openloop_per_joint.png`",
          "- `C_gripper_cmd_vs_state.png` / `C_gripper_latency.png`", ""]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")

    summary = {
        "bag": bag.name,
        "n_replans": len(infs),
        "infer_ms_median": float(np.median(ms)) if ms.size else None,
        "implied_steps": float((np.median(ms) - 233.7) / 40.64) if ms.size else None,
        "replan_period_median": float(np.median(periods)) if periods.size else None,
        "servo_rms_deg_mean": float(np.mean([r["rms"] for r in servo["per_joint"]]) * RAD2DEG),
        "servo_best_lag_ms_median": float(np.median([r["best_lag_ms"] for r in servo["per_joint"]])),
    }
    if ol:
        mae = ol["mae_k_joint"] * RAD2DEG
        hold = ol["hold_k_joint"] * RAD2DEG
        ne = ol["n_exec"]
        summary.update({
            "n_exec_steps": ne,
            "openloop_mae_deg_k0": float(mae[0].mean()),
            "openloop_mae_deg_kexec": float(mae[min(ne, ol["horizon"] - 1)].mean()),
            "skill_vs_hold_exec_pct": _skill(mae[1:ne + 1].mean(), hold[1:ne + 1].mean()),
        })
    for side, s in gr.get("sides", {}).items():
        summary[f"grip_{side}_agree_pct"] = s["agree_pct"]
        summary[f"grip_{side}_close_events"] = s["n_close_edges"]
        summary[f"grip_{side}_grasped_object"] = s["n_grasped_object"]
        summary[f"grip_{side}_closed_empty"] = s["n_closed_empty"]
        summary[f"grip_{side}_close_lat_ms"] = (
            float(np.median(s["close_lat_ms"])) if s["close_lat_ms"].size else None)
    return summary


def run(bag: Path, out: Path, control_hz: float) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    data = load_bag(bag, control_hz=control_hz)

    servo = servo_tracking(data, np.arange(-100, 401, 10, dtype=np.float64))
    ol = policy_open_loop(data, control_hz)
    gr = gripper_analysis(data)

    _write_csv(out / "A_servo_tracking_per_joint.csv",
               ["joint", "n", "bias_rad", "mae_rad", "median_rad", "rms_rad", "p95_rad",
                "max_rad", "mae_deg", "rms_deg", "p95_deg", "max_deg",
                "best_lag_ms", "rms_after_lag_rad", "rms_after_lag_deg", "rms_explained_pct"],
               [[r["joint"], int(r["n"]), r["bias"], r["mae"], r["med"], r["rms"], r["p95"],
                 r["max"], r["mae"] * RAD2DEG, r["rms"] * RAD2DEG, r["p95"] * RAD2DEG,
                 r["max"] * RAD2DEG, r["best_lag_ms"], r["rms_after_lag"],
                 r["rms_after_lag"] * RAD2DEG, r["rms_explained_pct"]]
                for r in servo["per_joint"]])

    if ol:
        mae, hold = ol["mae_k_joint"], ol["hold_k_joint"]
        _write_csv(out / "B_openloop_error_by_step.csv",
                   ["step_k", "t_sec"] + [f"{l}_mae_deg" for l in ARM_LABELS]
                   + ["mean14_mae_deg", "mean14_hold_deg", "skill_pct", "executed"],
                   [[k, k / control_hz] + [mae[k, j] * RAD2DEG for j in range(14)]
                    + [mae[k].mean() * RAD2DEG, hold[k].mean() * RAD2DEG,
                       _skill(mae[k].mean() * RAD2DEG, hold[k].mean() * RAD2DEG),
                       int(k < ol["n_exec"])]
                    for k in range(ol["horizon"])])

    if gr.get("sides"):
        _write_csv(out / "C_gripper_summary.csv",
                   ["side", "pos_open", "th_engage", "th_empty", "state_open_pct",
                    "state_grasp_pct", "state_closed_pct", "agree_pct", "cmd_closed_pct",
                    "meas_engaged_pct", "tp", "tn", "fp", "fn", "best_r", "best_lag_ms",
                    "n_close_events", "n_grasped_object", "n_closed_empty", "n_no_response",
                    "close_lat_ms_median", "open_lat_ms_median",
                    "effort_min", "effort_max", "effort_mean", "effort_neg_pct"],
                   [[side, s["pos_open"], s["th_engage"], s["th_empty"],
                     s["state_open_pct"], s["state_grasp_pct"], s["state_closed_pct"],
                     s["agree_pct"], s["cmd_closed_pct"], s["meas_engaged_pct"],
                     s["tp"], s["tn"], s["fp"], s["fn"], s["best_r"], s["best_lag_ms"],
                     s["n_close_edges"], s["n_grasped_object"], s["n_closed_empty"],
                     s["n_no_response"],
                     float(np.median(s["close_lat_ms"])) if s["close_lat_ms"].size else "",
                     float(np.median(s["open_lat_ms"])) if s["open_lat_ms"].size else "",
                     float(s["effort"].min()), float(s["effort"].max()),
                     float(s["effort"].mean()), float((s["effort"] <= 0).mean() * 100)]
                    for side, s in gr["sides"].items()])
        _write_csv(out / "C_gripper_close_events.csv",
                   ["side", "t_sec", "settled_pos", "outcome"],
                   [[side, t0, sp, kind]
                    for side, s in gr["sides"].items()
                    for t0, sp, kind in s["grasp_events"]])

    plot_servo(servo, out)
    plot_openloop(ol, control_hz, out)
    plot_gripper(gr, data, out)
    summary = write_report(bag, data, servo, ol, gr, control_hz, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(f"[joint_error] {bag.name} -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, action="append", required=True,
                    help="rosbag2 directory (repeatable)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir; with multiple bags this is the parent")
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--cross-summary", type=Path, default=None,
                    help="write a cross-bag comparison CSV/markdown here")
    args = ap.parse_args()

    summaries = []
    for bag in args.bag:
        bag = bag.expanduser().resolve()
        if len(args.bag) == 1 and args.out is not None:
            out = args.out.expanduser().resolve()
        else:
            parent = args.out.expanduser().resolve() if args.out else bag.parent
            out = parent / f"{bag.name}_offline" / "joint_errors"
        summaries.append(run(bag, out, args.control_hz))

    if args.cross_summary and summaries:
        dst = args.cross_summary.expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)
        keys = list(summaries[0].keys())
        _write_csv(dst / "cross_bag_summary.csv", keys,
                   [[s.get(k, "") for k in keys] for s in summaries])
        lines = ["# Cross-bag comparison", ""]
        rows = [[k] + [(f"{s.get(k):.3f}" if isinstance(s.get(k), float) else str(s.get(k)))
                       for s in summaries] for k in keys if k != "bag"]
        lines.append(_md_table(["metric"] + [s["bag"] for s in summaries], rows))
        (dst / "cross_bag_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[joint_error] cross summary -> {dst}")


if __name__ == "__main__":
    main()
