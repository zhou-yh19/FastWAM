"""Teleavatar / TA2 action-state selection (aligned with openpi teleavatar_v2).

Raw dataset layout (72-d shared by action & state):
  positions[0:16]  = [L_arm(7), L_grip_pos, R_arm(7), R_grip_pos]
  velocities[16:32]
  efforts[32:48]   = [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
  + optional EE / chassis / ...

Model formats (openpi convention):
  state  14-d = [L_arm(7), R_arm(7)]
  action 16-d = [L_arm(7), L_grip, R_arm(7), R_grip]

Gripper channels in action are force-control: dataset stores effort (Nm);
before normalization we convert effort -> platform trigger in [0, 1];
backward converts trigger -> effort for robot-side publishing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def gripper_effort_to_trigger(effort: torch.Tensor) -> torch.Tensor:
    """Dataset gripper effort (Nm) -> platform trigger [≈0, 1] (TA2 / openpi v2)."""
    return torch.where(
        effort > 0,
        0.10 * (1.0 - effort / 2.0),
        0.10 - effort * 0.90 / 1.6,
    )


def gripper_trigger_to_effort(trigger: torch.Tensor) -> torch.Tensor:
    """Platform trigger [0, 1] -> gripper effort (Nm) (TA2 / openpi v2)."""
    return torch.where(
        trigger < 0.10,
        2.0 * (1.0 - trigger / 0.10),
        -1.6 * (trigger - 0.10) / 0.90,
    )


class TeleavatarSelectTransform:
    """Slice 72-d Teleavatar vectors to 14-d state / 16-d action.

    Args:
        keys: shape_meta keys to transform (usually ["default"]).
        convert_gripper_to_trigger: if True, map action gripper effort->trigger
            on forward (before norm) and trigger->effort on backward.
    """

    # Indices into the raw 72-d vector.
    STATE_IDX = list(range(0, 7)) + list(range(8, 15))  # 14
    ACTION_ARM_L = list(range(0, 7))
    ACTION_ARM_R = list(range(8, 15))
    ACTION_GRIP_L = 39
    ACTION_GRIP_R = 47

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        convert_gripper_to_trigger: bool = True,
    ):
        self.keys = keys if keys is not None else ["default"]
        self.convert_gripper_to_trigger = bool(convert_gripper_to_trigger)

    def forward(self, batch: Dict) -> Dict:
        for k in self.keys:
            if "action" in batch and k in batch["action"]:
                a = batch["action"][k]
                assert a.shape[-1] >= 48, f"action '{k}' too short: {a.shape}"
                selected = torch.cat(
                    [
                        a[..., self.ACTION_ARM_L],
                        a[..., self.ACTION_GRIP_L : self.ACTION_GRIP_L + 1],
                        a[..., self.ACTION_ARM_R],
                        a[..., self.ACTION_GRIP_R : self.ACTION_GRIP_R + 1],
                    ],
                    dim=-1,
                )
                if self.convert_gripper_to_trigger:
                    selected = selected.clone()
                    selected[..., 7] = gripper_effort_to_trigger(selected[..., 7])
                    selected[..., 15] = gripper_effort_to_trigger(selected[..., 15])
                batch["action"][k] = selected

            if "state" in batch and k in batch["state"]:
                s = batch["state"][k]
                assert s.shape[-1] >= 15, f"state '{k}' too short: {s.shape}"
                idx = torch.as_tensor(self.STATE_IDX, device=s.device, dtype=torch.long)
                batch["state"][k] = s.index_select(dim=-1, index=idx)

        return batch

    def backward(self, batch: Dict) -> Dict:
        """Convert action gripper trigger -> effort; keep 16-d (do not expand to 72)."""
        if "action" not in batch or not self.convert_gripper_to_trigger:
            return batch
        for k in self.keys:
            if k not in batch["action"]:
                continue
            a = batch["action"][k]
            assert a.shape[-1] == 16, f"expected 16-d action for backward, got {a.shape}"
            a = a.clone()
            a[..., 7] = gripper_trigger_to_effort(a[..., 7])
            a[..., 15] = gripper_trigger_to_effort(a[..., 15])
            batch["action"][k] = a
        return batch
