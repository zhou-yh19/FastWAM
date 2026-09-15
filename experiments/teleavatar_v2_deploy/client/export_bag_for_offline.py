#!/usr/bin/env python3
"""Export a FastWAM TA2 rosbag2 into a portable folder for remote offline infer.

Run on the Ubuntu client host (needs ROS2 Humble + rosbag2_py):

  source /opt/ros/humble/setup.bash
  python3 export_bag_for_offline.py \\
    --bag $HOME/fastwam_bags/fastwam_ta2_YYYYMMDD_HHMMSS \\
    --out ~/fastwam_bag_exports/run1

Produces:
  out/
    meta.json
    samples.jsonl
    samples/000000/{head_camera,left_color,right_color}.jpg
    samples/000000/recorded_action.npy   # if /fastwam/policy/action_chunk nearby
    samples/000000/proprio72.npy         # reconstructed from joint/gripper/ee near stamp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Need ROS2 Python libs. Run with:\n"
        "  source /opt/ros/humble/setup.bash\n"
        f"Original import error: {exc}"
    ) from exc

CAMS = ("head_camera", "left_color", "right_color")
LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]


def _open_reader(bag_path: Path) -> SequentialReader:
    reader = SequentialReader()
    storage_id = "sqlite3"
    if list(bag_path.glob("*.mcap")):
        storage_id = "mcap"
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def _field_from_js(msg: Any, field: str, n: int, preferred: Optional[List[str]] = None) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    data = list(getattr(msg, field, []) or [])
    names = list(getattr(msg, "name", []) or [])
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


def _build_proprio72(
    left_arm: Any,
    right_arm: Any,
    left_grip: Any,
    right_grip: Any,
    left_ee: Any,
    right_ee: Any,
) -> np.ndarray:
    state = np.zeros(72, dtype=np.float32)
    state[0:7] = _field_from_js(left_arm, "position", 7, LEFT_JOINTS)
    state[7] = _field_from_js(left_grip, "position", 1)[0] if left_grip is not None else 0.0
    state[8:15] = _field_from_js(right_arm, "position", 7, RIGHT_JOINTS)
    state[15] = _field_from_js(right_grip, "position", 1)[0] if right_grip is not None else 0.0
    state[16:23] = _field_from_js(left_arm, "velocity", 7, LEFT_JOINTS)
    state[23] = _field_from_js(left_grip, "velocity", 1)[0] if left_grip is not None else 0.0
    state[24:31] = _field_from_js(right_arm, "velocity", 7, RIGHT_JOINTS)
    state[31] = _field_from_js(right_grip, "velocity", 1)[0] if right_grip is not None else 0.0
    state[32:39] = _field_from_js(left_arm, "effort", 7, LEFT_JOINTS)
    state[39] = _field_from_js(left_grip, "effort", 1)[0] if left_grip is not None else 0.0
    state[40:47] = _field_from_js(right_arm, "effort", 7, RIGHT_JOINTS)
    state[47] = _field_from_js(right_grip, "effort", 1)[0] if right_grip is not None else 0.0
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


def _nearest(timeline: List[Tuple[int, Any]], t_ns: int, max_dt_ns: int) -> Optional[Any]:
    if not timeline:
        return None
    best = min(timeline, key=lambda x: abs(x[0] - t_ns))
    if abs(best[0] - t_ns) > max_dt_ns:
        return None
    return best[1]


def _decode_jpeg(msg: Any) -> np.ndarray:
    from io import BytesIO

    img = Image.open(BytesIO(bytes(msg.data))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _action_from_multiarray(msg: Any) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.float32)
    dims = list(msg.layout.dim) if msg.layout.dim else []
    if len(dims) >= 2:
        t, d = int(dims[0].size), int(dims[1].size)
        if t * d == data.size:
            return data.reshape(t, d)
    # fallback: assume D=16
    if data.size % 16 == 0:
        return data.reshape(-1, 16)
    return data.reshape(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True, help="rosbag2 directory")
    parser.add_argument("--out", type=Path, required=True, help="export output directory")
    parser.add_argument(
        "--match-ms",
        type=float,
        default=1500.0,
        help="max |dt| to pair action_chunk / joint_states with an observation stamp",
    )
    args = parser.parse_args()

    bag_path = args.bag.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    if not bag_path.is_dir():
        raise SystemExit(f"bag not found: {bag_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    obs_groups: Dict[int, Dict[str, Any]] = {}  # stamp_ns -> cams
    actions: List[Tuple[int, np.ndarray]] = []
    left_arm_tl: List[Tuple[int, Any]] = []
    right_arm_tl: List[Tuple[int, Any]] = []
    left_grip_tl: List[Tuple[int, Any]] = []
    right_grip_tl: List[Tuple[int, Any]] = []
    left_ee_tl: List[Tuple[int, Any]] = []
    right_ee_tl: List[Tuple[int, Any]] = []

    topic_to_cam = {
        f"/fastwam/observation/{c}/compressed": c for c in CAMS
    }

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in type_map:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if topic in topic_to_cam:
            cam = topic_to_cam[topic]
            stamp = msg.header.stamp
            stamp_ns = int(stamp.sec) * 10**9 + int(stamp.nanosec)
            g = obs_groups.setdefault(stamp_ns, {"stamp_ns": stamp_ns, "images": {}})
            g["images"][cam] = _decode_jpeg(msg)
        elif topic == "/fastwam/policy/action_chunk":
            actions.append((int(t_ns), _action_from_multiarray(msg)))
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

    max_dt = int(args.match_ms * 1e6)
    rows: List[Dict[str, Any]] = []
    complete = [
        (stamp, g)
        for stamp, g in sorted(obs_groups.items())
        if all(c in g["images"] for c in CAMS)
    ]
    print(f"complete observation groups: {len(complete)}  action_chunks: {len(actions)}")

    for idx, (stamp_ns, g) in enumerate(complete):
        sample_rel = f"samples/{idx:06d}"
        sample_dir = out_dir / sample_rel
        sample_dir.mkdir(parents=True, exist_ok=True)
        for cam in CAMS:
            Image.fromarray(g["images"][cam]).save(sample_dir / f"{cam}.jpg", quality=92)

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
        if actions:
            # action_chunk is published just after infer; prefer first action after obs stamp
            after = [a for a in actions if a[0] >= stamp_ns - max_dt]
            pick = after[0] if after else min(actions, key=lambda x: abs(x[0] - stamp_ns))
            if abs(pick[0] - stamp_ns) <= max_dt * 2:
                recorded_action = pick[1]
                np.save(sample_dir / "recorded_action.npy", recorded_action)

        row = {
            "id": idx,
            "stamp_ns": stamp_ns,
            "dir": sample_rel,
            "has_recorded_action": recorded_action is not None,
            "recorded_action_shape": None
            if recorded_action is None
            else list(recorded_action.shape),
            "proprio_Lq0": float(proprio[0]),
            "proprio_Rq0": float(proprio[8]),
        }
        rows.append(row)
        print(
            f"[{idx:04d}] stamp={stamp_ns} action="
            f"{None if recorded_action is None else tuple(recorded_action.shape)}"
        )

    (out_dir / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    meta = {
        "bag": str(bag_path),
        "num_samples": len(rows),
        "cameras": list(CAMS),
        "match_ms": args.match_ms,
        "note": (
            "Images are the FastWAM observation SBS frames published at infer time. "
            "proprio72 is reconstructed from nearest robot joint/gripper/ee topics. "
            "recorded_action is the online policy action_chunk if present."
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} samples -> {out_dir}")


if __name__ == "__main__":
    main()
