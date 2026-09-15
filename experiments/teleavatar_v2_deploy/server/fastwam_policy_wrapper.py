#!/usr/bin/env python3
"""BasePolicy wrapper for FastWAMTA2Policy to work with OpenPI WebSocket server.

This adapter converts between OpenPI's standardized observation/action format
and FastWAM's existing inference interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("fastwam_policy_wrapper")


class FastWAMPolicyWrapper:
    """Wraps FastWAMTA2Policy to conform to OpenPI BasePolicy interface.

    OpenPI expects:
        infer(obs: Dict) -> Dict

    Where obs contains:
        - 'observation/images/<key>': np.ndarray (H, W, C) uint8
        - 'observation/state': np.ndarray (N,) float32
        - 'prompt': str (optional)

    And returns:
        - 'actions': np.ndarray (T, action_dim) float32
    """

    def __init__(self, policy: Any) -> None:
        """Initialize wrapper.

        Args:
            policy: FastWAMTA2Policy instance
        """
        self._policy = policy
        logger.info("FastWAMPolicyWrapper initialized")

    def infer(self, obs: Dict) -> Dict:
        """Run inference on observation.

        Args:
            obs: Dictionary with OpenPI-style observation keys:
                - 'observation/images/head_camera': np.ndarray
                - 'observation/images/left_color': np.ndarray
                - 'observation/images/right_color': np.ndarray
                - 'observation/state': np.ndarray (48 or 72-d)
                - 'prompt': str (optional, for task switching)

        Returns:
            Dictionary with:
                - 'actions': np.ndarray (T, 16) denormalized actions
        """
        # Extract images from OpenPI format
        images = {}
        for key in ('head_camera', 'left_color', 'right_color'):
            obs_key = f'observation/images/{key}'
            if obs_key in obs:
                images[key] = obs[obs_key]

        if not images:
            raise ValueError("No images found in observation")

        # Extract state (48 or 72-d)
        state = obs.get('observation/state')
        if state is None:
            raise ValueError("No state found in observation")

        state = np.asarray(state, dtype=np.float32).reshape(-1)

        # Expand 48-d to 72-d if needed (FastWAM server expects 72-d)
        if state.shape[0] == 48:
            # OpenPI sends 48-d: [pos(16), vel(16), eff(16)]
            # FastWAM expects 72-d: [pos(16), vel(16), eff(16), EE(14), chassis(9), kinco(1)]
            state_72 = np.zeros(72, dtype=np.float32)
            state_72[:48] = state
            state = state_72
        elif state.shape[0] != 72:
            raise ValueError(f"Expected state dim 48 or 72, got {state.shape[0]}")

        # Build FastWAM-style payload
        payload = {
            'images': images,
            'proprio': state.tolist(),
            'action_horizon': self._policy.action_horizon,
            'return_video': False,
        }

        num_inference_steps = int(
            obs.get('num_inference_steps', self._policy.num_inference_steps)
        )
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}"
            )
        payload['num_inference_steps'] = num_inference_steps

        # Task selection is explicit: the client sends the library key it wants
        # (a key of taskmap.json). Passing it straight through means an
        # unknown name raises in _resolve_text_kwargs rather than silently falling
        # back to the startup instruction -- commanding a 5-level tower while the
        # model is conditioned on a 3-level one looks like a policy failure, not a
        # config bug. `prompt` is deliberately NOT used to pick the task: the
        # library keys are short names, not instruction text, so matching them
        # against a sentence can only guess.
        task = obs.get('task')
        if task:
            payload['task'] = str(task)

        # Run inference through existing FastWAM logic
        result = self._policy.infer(payload)

        # Convert to OpenPI format
        action = np.asarray(result['action'], dtype=np.float32)  # [T, 16]

        return {
            'actions': action,
            'num_inference_steps': num_inference_steps,
            'task': result.get('task'),
            # Optional: pass through timing info for debugging
            'server_timing': result.get('timings_ms', {}),
        }

    def reset(self) -> None:
        """Reset policy state (no-op for FastWAM stateless inference)."""
        pass
