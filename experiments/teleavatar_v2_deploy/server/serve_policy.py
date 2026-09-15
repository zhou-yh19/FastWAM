#!/usr/bin/env python3
"""FastWAM TA2 policy implementation (GPU).

This module is a library, not an entry point. `FastWAMTA2Policy` is loaded by
`serve_policy_ws.py`, which is the only supported way to serve it. The HTTP server that
used to live at the bottom of this file (BaseHTTPRequestHandler + `POST /infer`) was
removed when deployment moved to WebSocket + msgpack -- running both risked two processes
fighting over port 8000 and the same GPU.

The client sends three camera images + 72-d proprio; the policy slices to 14-d state via
TeleavatarSelectTransform and returns a denormalized action chunk
[T, 16] = [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
(gripper trigger->effort applied in transform.backward after denorm).

The T5 text encoder is NOT loaded: the deployed instruction is a constant, so its embedding
is read from the same `text_embeds_cache` training used (11 GiB saved, and the conditioning
is byte-identical to training). Pass `--load-text-encoder` to go back to encoding on the fly.

On a host without `data/` (e.g. a 24 GB deploy box), copy the single ~1 MB embedding over and
point at it directly with `--text-embed`; nothing else from the dataset is needed.

See docs/GUIDE.md for how to start the server.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.image import compose_ta2_mosaic
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger("fastwam_ta2_server")

# A --task-map value is treated as a cache hash when it looks like one, else as instruction text.
_HASH_RE = re.compile(r"[0-9a-f]{8,64}")


def _decode_jpeg_b64(image: Any) -> np.ndarray:
    """Decode a base64 JPEG, or pass through an already-decoded HWC uint8 array.

    The HTTP client sends base64 JPEG strings. The WebSocket client sends raw numpy
    arrays instead -- msgpack carries them natively, so JPEG round-tripping there would
    only add an encode on the client and a decode here for no benefit.
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB array, got shape {image.shape}")
        return np.ascontiguousarray(image, dtype=np.uint8)
    raw = base64.b64decode(image)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _hwc_uint8_to_tchw01(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB uint8, got {image.shape}")
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
    return t


def _pil_to_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _np_rgb_to_jpeg_b64(image: np.ndarray, quality: int = 85) -> str:
    return _pil_to_jpeg_b64(Image.fromarray(image.astype(np.uint8), mode="RGB"), quality=quality)


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def _text_embed_filename(prompt: str, context_len: int) -> str:
    """Cache filename for a formatted prompt.

    Mirrors RobotVideoDataset._get_cached_text_context, so the tensor served here is the exact
    one training consumed for the same instruction.
    """
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{int(context_len)}.wan22ti2v5b.pt"


def _instruction_from_dataset(dataset_dir: Path) -> Optional[str]:
    """First task string recorded in a LeRobot dataset, or None if unavailable."""
    tasks = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks.is_file():
        return None
    for line in tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return str(json.loads(line)["task"])
    return None


def _instructions_from_task_cfg(cfg) -> list[str]:
    """Distinct instructions recorded across the task's training datasets, in config order.

    Reading these from the data rather than a hardcoded default is what keeps the deployed
    conditioning byte-identical after a dataset is regenerated (one such rebuild
    silently dropped three spaces, which changes the sha256 and thus the embedding).
    """
    found: list[str] = []
    for dataset_dir in cfg.data.train.dataset_dirs:
        instruction = _instruction_from_dataset((PROJECT_ROOT / str(dataset_dir)).resolve())
        if instruction is not None and instruction not in found:
            found.append(instruction)
    return found


class FastWAMTA2Policy:
    def __init__(
        self,
        checkpoint: Path,
        dataset_stats: Path,
        task_name: str,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: int = 32,
        num_inference_steps: int = 20,
        num_video_frames: Optional[int] = None,
        prompt_task: Optional[str] = None,
        text_embed: Optional[Path] = None,
        load_text_encoder: bool = False,
        task_map: Optional[Path] = None,
        compile_denoise: bool = True,
        warmup_steps: tuple[int, ...] = (),
    ) -> None:
        self.task_map_path = task_map
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._infer_lock = threading.Lock()
        if mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception as exc:
                logger.warning("Could not set torch deployment backend knobs: %s", exc)

        configs_root = PROJECT_ROOT / "configs"
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        # Must compose via root `train` so task can override /data and /model.
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(
                config_name="train",
                overrides=[f"task={task_name}"],
            )

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = bool(load_text_encoder)
        self.has_text_encoder = bool(load_text_encoder)
        # Deployment restores every expert weight from `--checkpoint`, so reading the pretrained
        # backbones first is wasted work: 19 GiB of fp32 Wan2.2 DiT shards plus the 2 GiB
        # ActionDiT payload, all overwritten by load_checkpoint() moments later. Skipping them
        # cuts startup I/O and keeps the host RAM peak to the checkpoint itself. The VAE is
        # unaffected -- it is absent from the checkpoint and still loads from its own file.
        #
        # The checkpoint is verified below to cover every expert parameter, so the random init
        # this flag leaves behind never survives into inference.
        model_cfg.skip_dit_load_from_pretrain = True
        # Resolve the conditioning *before* building the model: a missing embedding should fail
        # in a second, not after loading 12 GiB of weights.
        if prompt_task is None and text_embed is None:
            prompt_task = self._autoresolve_prompt_task(cfg)
        self.prompt_task = prompt_task
        self.prompt = None if prompt_task is None else DEFAULT_PROMPT.format(task=prompt_task)

        text_embed_path: Optional[Path] = None
        if self.has_text_encoder:
            if self.prompt is None:
                raise ValueError(
                    "--load-text-encoder needs an instruction to encode, but --prompt-task was "
                    "not given and no meta/tasks.jsonl was found under "
                    f"{list(cfg.data.train.dataset_dirs)}."
                )
        else:
            text_embed_path = self._resolve_text_embed(cfg, text_embed, prompt_task)

        self.model = instantiate(model_cfg, model_dtype=self.dtype, device=str(self.device))
        try:
            # Pretrained backbones were skipped above, so an uncovered parameter would fall back
            # to random init instead of pretrained weights. Fail instead of serving that.
            self.model.load_checkpoint(str(checkpoint), require_full_coverage=True)
        except RuntimeError as exc:
            msg = str(exc)
            if "size mismatch" in msg and ("1024, 72" in msg or "Size([72" in msg or "shape torch.Size([72" in msg):
                raise RuntimeError(
                    f"Checkpoint action dim does not match the model built from "
                    f"--task {task_name} (a 72-d checkpoint against a 16-d config is the "
                    f"usual cause).\n"
                    f"The checkpoint, the dataset_stats and the task config must all come "
                    f"from the same run -- do not mix them:\n"
                    f"  --task <config under configs/task/, as trained> \\\n"
                    f"  --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_NNNNNN.pt \\\n"
                    f"  --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json\n"
                    f"Got: {checkpoint}\n"
                    f"Original error: {exc}"
                ) from exc
            raise
        self.model = self.model.to(self.device).eval()

        processor_cfg = cfg.data.train.processor
        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        stats = load_dataset_stats_from_json(str(dataset_stats))
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        # Match training video length: indices 0,4,...,32 → 9 frames at ratio 4.
        if num_video_frames is None:
            data_nf = int(cfg.data.train.num_frames)
            ratio = int(cfg.data.train.action_video_freq_ratio)
            num_video_frames = 1 + (data_nf - 1) // ratio
        self.num_video_frames = int(num_video_frames)
        if self.num_video_frames % 4 != 1:
            raise ValueError(
                f"num_video_frames must satisfy T%4==1, got {self.num_video_frames}"
            )
        if self.has_text_encoder:
            # Encode on the fly: 11 GiB of T5 for a constant string, kept only as an A/B escape
            # hatch against the cached path.
            self.text_kwargs: dict[str, Any] = {"prompt": self.prompt}
        else:
            context, context_mask = self._load_cached_context(text_embed_path)
            self.text_kwargs = {
                "prompt": None,
                "context": context,
                "context_mask": context_mask,
            }
        self._load_task_library(cfg, text_embed_path)
        self._mosaic_canvas_size = tuple(int(x) for x in self.processor.ta2_canvas_size)
        # Echo the resolved mosaic geometry: this is the one number to eyeball when swapping
        # checkpoints, since a stereo/mono mismatch is otherwise invisible at runtime.
        logger.info(
            "TA2 mosaic geometry from config: canvas=%dx%d head=%dx%d wrist=%dx%d",
            self.processor.ta2_canvas_size[0],
            self.processor.ta2_canvas_size[1],
            self.processor.ta2_head_tile[0],
            self.processor.ta2_head_tile[1],
            self.processor.ta2_wrist_tile[0],
            self.processor.ta2_wrist_tile[1],
        )
        logger.info(
            "Loaded ckpt=%s stats=%s horizon=%d video_frames=%d text_encoder=%s embed=%s prompt=%s",
            checkpoint,
            dataset_stats,
            action_horizon,
            self.num_video_frames,
            self.has_text_encoder,
            text_embed_path,
            self.prompt,
        )

        if compile_denoise:
            steps = tuple(warmup_steps) or (self.num_inference_steps,)
            self._enable_compiled_denoise(steps)

    def _pin_rope_freqs_to_device(self) -> None:
        """Move plain-attribute RoPE tables onto the compute device."""
        for name, module in (
            ("action_expert", getattr(self.model, "action_expert", None)),
            ("video_expert", getattr(self.model, "video_expert", None)),
        ):
            freqs = getattr(module, "freqs", None) if module is not None else None
            if freqs is None:
                continue
            if isinstance(freqs, torch.Tensor):
                if freqs.device != self.device:
                    module.freqs = freqs.to(self.device)
                    logger.info("moved %s.freqs -> %s", name, self.device)
            elif isinstance(freqs, (list, tuple)):
                moved = [
                    value.to(self.device)
                    if isinstance(value, torch.Tensor) and value.device != self.device
                    else value
                    for value in freqs
                ]
                module.freqs = type(freqs)(moved)
                logger.info("moved %s.freqs (%d tables) -> %s", name, len(moved), self.device)

    def _enable_compiled_denoise(self, warmup_steps: tuple[int, ...]) -> None:
        """Compile the launch-bound action denoiser and warm its CUDA Graph."""
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled denoise requires torch.compile (PyTorch 2.x).")
        if self.device.type != "cuda":
            logger.warning("Skipping compiled denoise on non-CUDA device: %s", self.device)
            return

        self._pin_rope_freqs_to_device()
        original = self.model._predict_action_noise_with_cache
        self.model._predict_action_noise_with_cache = torch.compile(
            original,
            mode="reduce-overhead",
            dynamic=False,
        )
        logger.info("compiled action denoise step (mode=reduce-overhead)")

        canvas_h, canvas_w = (int(value) for value in self.processor.ta2_canvas_size)
        dummy_image = torch.zeros(
            (1, 3, canvas_h, canvas_w), device=self.device, dtype=self.dtype
        )
        dummy_proprio = None
        if getattr(self.model, "proprio_dim", None):
            dummy_proprio = torch.zeros(
                (1, int(self.model.proprio_dim)), device=self.device, dtype=self.dtype
            )

        for steps in dict.fromkeys(int(value) for value in warmup_steps):
            if steps <= 0:
                raise ValueError(f"warmup num_inference_steps must be positive, got {steps}")
            t0 = time.perf_counter()
            with torch.inference_mode():
                self.model.infer_action(
                    input_image=dummy_image,
                    action_horizon=self.action_horizon,
                    proprio=dummy_proprio,
                    num_inference_steps=steps,
                    seed=0,
                    **self.text_kwargs,
                )
            logger.info(
                "warmed num_inference_steps=%d in %.1f s",
                steps,
                time.perf_counter() - t0,
            )

    def _sync_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self, iters: int = 1) -> None:
        """Run dummy action-only requests to initialize CUDA kernels before serving."""
        n = int(iters)
        if n <= 0:
            return
        if self.device.type != "cuda":
            logger.info("Skipping startup warmup on non-CUDA device: %s", self.device)
            return
        # Warm the same path production uses: the three-camera branch, not the pre-composed
        # "mosaic" shortcut. Sizes are the real single-eye RTP frames, so the CUDA-graph
        # specialization matches real requests and a geometry regression fails at startup
        # instead of silently serving a canvas the model was never trained on.
        head = np.full((960, 960, 3), 128, dtype=np.uint8)
        wrist = np.full((400, 640, 3), 128, dtype=np.uint8)
        payload = {
            "images": {
                "head_camera": _np_rgb_to_jpeg_b64(head, quality=85),
                "left_color": _np_rgb_to_jpeg_b64(wrist, quality=85),
                "right_color": _np_rgb_to_jpeg_b64(wrist, quality=85),
            },
            "proprio": [0.0] * 72,
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "return_video": False,
        }
        logger.info("Running %d startup warmup inference(s)...", n)
        for i in range(n):
            t0 = time.perf_counter()
            out = self.infer(payload)
            logger.info(
                "warmup %d/%d mode=%s total=%.0fms model=%.0fms",
                i + 1,
                n,
                out.get("mode"),
                float(out["timings_ms"].get("total_ms", 0.0)),
                float(out["timings_ms"].get("model_ms", 0.0)),
            )
            logger.debug("warmup wall %.3fs", time.perf_counter() - t0)

    @staticmethod
    def _autoresolve_prompt_task(cfg) -> str:
        """Infer the instruction from the task's own datasets, or refuse to guess.

        A multi-dataset task trained on several distinct instructions (one config spanning
        two- through eight-level towers) has no single default: the instruction *is* the task
        being commanded, so picking dataset_dirs[0] would silently deploy the wrong one.
        """
        candidates = _instructions_from_task_cfg(cfg)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"Cannot read the instruction: no meta/tasks.jsonl under "
                f"{[str(d) for d in cfg.data.train.dataset_dirs]} (the datasets are probably "
                f"not on this host). Pass --prompt-task, or --text-embed to skip the lookup."
            )
        listed = "\n".join(f"    {c!r}" for c in candidates)
        raise ValueError(
            f"This task trained on {len(candidates)} distinct instructions, and the "
            f"instruction is part of what you are commanding -- pick one explicitly with "
            f"--prompt-task:\n{listed}"
        )

    def _resolve_text_embed(
        self,
        cfg,
        text_embed: Optional[Path],
        prompt_task: Optional[str],
    ) -> Path:
        """Locate the precomputed T5 context for this deployment.

        Training never ran T5 either -- RobotVideoDataset reads these same files, keyed by
        sha256 of the formatted prompt (see `_get_cached_text_context`). An explicit
        `--text-embed` wins; otherwise the instruction is hashed against the cache dir named by
        the composed config, so switching `--task` switches to that run's embeddings.
        """
        if text_embed is not None:
            path = text_embed.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"--text-embed not found: {path}")
            return path

        data_cfg = cfg.data.train
        cache_dir = (PROJECT_ROOT / str(data_cfg.text_embedding_cache_dir)).resolve()
        context_len = int(data_cfg.context_len)
        if prompt_task is None:
            raise ValueError(
                f"Cannot determine the instruction: no --prompt-task given and no "
                f"meta/tasks.jsonl found under {list(data_cfg.dataset_dirs)}. "
                f"Pass --prompt-task, or --text-embed to skip the lookup entirely."
            )

        path = cache_dir / _text_embed_filename(DEFAULT_PROMPT.format(task=prompt_task), context_len)
        if not path.is_file():
            available = sorted(p.name for p in cache_dir.glob("*.pt")) if cache_dir.is_dir() else []
            raise FileNotFoundError(
                f"No cached text embedding for this instruction.\n"
                f"  cache dir : {cache_dir}\n"
                f"  expected  : {path.name}\n"
                f"  available : {available or '(none)'}\n"
                f"  prompt    : {prompt_task!r}\n"
                f"The instruction must match the training data byte for byte (see "
                f"<dataset>/meta/tasks.jsonl -- whitespace counts). Pass --text-embed to point "
                f"at a file directly, or run scripts/precompute_text_embeds.py to add this string."
            )
        return path

    def _load_cached_context(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a precomputed T5 context, reproducing the dataset's post-processing.

        RobotVideoDataset zeroes the padded positions and then sets the mask to all-ones, and
        encode_prompt does the same; skipping it would feed the model a mask training never
        produced.
        """
        payload = torch.load(str(path), map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Bad text embed cache {path}: "
                f"context={tuple(context.shape)} mask={tuple(mask.shape)}"
            )
        context = context.clone()
        context[~mask] = 0.0
        mask = torch.ones_like(mask)
        return (
            context.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            mask.unsqueeze(0).to(device=self.device),
        )

    def _load_task_library(self, cfg, active_embed: Optional[Path]) -> None:
        """Preload every cached T5 context so a request can pick its instruction by name.

        The cache is keyed by sha256 of the formatted prompt, so the instruction text cannot be
        recovered from the filename. Names therefore come from `--task-map` when supplied and
        fall back to the hash prefix, which is still stable and selectable. Each entry is a
        self-contained (context, mask) pair -- exactly what the single-instruction path feeds the
        model -- so switching between them needs no text encoder.
        """
        self.task_library: dict[str, dict] = {}
        self.active_task: Optional[str] = None

        cache_dir = (PROJECT_ROOT / str(cfg.data.train.text_embedding_cache_dir)).resolve()
        if not cache_dir.is_dir():
            logger.warning("Text embed cache dir not found, multi-task disabled: %s", cache_dir)
            return

        alias: dict[str, str] = {}
        if self.task_map_path is not None:
            raw = json.loads(self.task_map_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"--task-map must be a JSON object, got {type(raw).__name__}")
            for name, value in raw.items():
                # Accept either a hash (prefix) or the full instruction text.
                token = str(value)
                digest = (
                    token
                    if _HASH_RE.fullmatch(token)
                    else hashlib.sha256(DEFAULT_PROMPT.format(task=token).encode("utf-8")).hexdigest()
                )
                alias[digest] = str(name)

        for path in sorted(cache_dir.glob("*.pt")):
            digest = path.name.split(".")[0]
            name = alias.get(digest)
            if name is None:
                name = next(
                    (n for d, n in alias.items() if digest.startswith(d)),
                    digest[:12],
                )
            context, mask = self._load_cached_context(path)
            self.task_library[name] = {
                "context": context,
                "mask": mask,
                "digest": digest,
                "path": str(path),
            }
            if active_embed is not None and path.resolve() == active_embed.resolve():
                self.active_task = name

        logger.info(
            "Task library: %d instruction(s) available %s (default=%s)",
            len(self.task_library),
            sorted(self.task_library),
            self.active_task,
        )

    def _resolve_text_kwargs(self, payload: dict) -> tuple[dict, Optional[str]]:
        """Pick the conditioning for one request.

        Precedence: explicit `prompt` (needs T5) > `task` name from the library > the
        instruction chosen at startup.
        """
        req_prompt = payload.get("prompt")
        if req_prompt:
            if not self.has_text_encoder:
                raise ValueError(
                    "Per-request `prompt` needs the T5 text encoder, which is not loaded. "
                    "Use `task` to pick a precomputed instruction "
                    f"({sorted(self.task_library)}), restart with --prompt-task/--text-embed, "
                    "or add --load-text-encoder to encode on the fly (+11 GiB)."
                )
            return {"prompt": str(req_prompt)}, None

        task = payload.get("task")
        if task is None:
            return self.text_kwargs, self.active_task

        name = str(task)
        entry = self.task_library.get(name)
        if entry is None:
            matches = [k for k in self.task_library if k.startswith(name)]
            if len(matches) > 1:
                raise ValueError(f"`task`={name!r} is ambiguous: {sorted(matches)}")
            if not matches:
                raise ValueError(
                    f"Unknown `task`={name!r}. Available: {sorted(self.task_library)}"
                )
            name = matches[0]
            entry = self.task_library[name]
        return (
            {"prompt": None, "context": entry["context"], "context_mask": entry["mask"]},
            name,
        )

    def _normalize_proprio(self, state: np.ndarray) -> torch.Tensor:
        """Client still sends raw 72-d; processor selects 14-d then normalizes."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 72:
            raise ValueError(f"proprio must be raw 72-d from robot, got {state.shape}")
        state_meta = self.processor.shape_meta["state"]
        state_key = state_meta[0]["key"]
        batch = {"state": {state_key: torch.as_tensor(state).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Denorm 16-d model action, then trigger->effort on gripper dims.

        Only action is present at deploy time; normalizer.backward must not require state.
        """
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self.processor.shape_meta["action"][0]["key"]
        batch = {"action": {action_key: action.to(dtype=torch.float32, device="cpu")}}
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for trans in reversed(self.processor.action_state_transforms):
                batch = trans.backward(batch)
        return batch["action"][action_key].numpy()[0]

    def _build_mosaic(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        if "mosaic" in images:
            image = images["mosaic"]
            expected_h, expected_w = self._mosaic_canvas_size
            if tuple(image.shape[:2]) != (expected_h, expected_w):
                raise ValueError(
                    f"mosaic image must be {expected_w}x{expected_h}, got "
                    f"{image.shape[1]}x{image.shape[0]}"
                )
            mosaic01 = _hwc_uint8_to_tchw01(image)
            mosaic = mosaic01 * 2.0 - 1.0
            return mosaic.to(device=self.device, dtype=self.dtype)

        # Training camera order: head, left_color, right_color
        for k in ("head_camera", "left_color", "right_color"):
            if k not in images:
                raise KeyError(f"missing image key: {k}")
        cams = [_hwc_uint8_to_tchw01(images[k]) for k in ("head_camera", "left_color", "right_color")]
        # Geometry must come from the checkpoint's processor, not compose_ta2_mosaic's
        # defaults: a config/default mismatch would silently shift every tile and the model
        # would keep serving plausible-looking garbage.
        mosaic01 = compose_ta2_mosaic(
            cams,
            canvas_size=tuple(int(v) for v in self.processor.ta2_canvas_size),
            head_tile=tuple(int(v) for v in self.processor.ta2_head_tile),
            wrist_tile=tuple(int(v) for v in self.processor.ta2_wrist_tile),
            fill=float(self.processor.ta2_fill),
        )  # [1, 3, canvas_h, canvas_w] in [0, 1]
        mosaic = mosaic01 * 2.0 - 1.0
        return mosaic.to(device=self.device, dtype=self.dtype)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        queue_t0 = time.perf_counter()
        with self._infer_lock:
            queue_ms = (time.perf_counter() - queue_t0) * 1000.0
            return self._infer_locked(payload, queue_ms=queue_ms)

    @torch.inference_mode()
    def _infer_locked(self, payload: dict[str, Any], queue_ms: float = 0.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        timings_ms: dict[str, float] = {"queue_ms": float(queue_ms)}

        t_decode0 = time.perf_counter()
        image_payload = payload["images"]
        if "mosaic" in image_payload:
            images = {"mosaic": _decode_jpeg_b64(image_payload["mosaic"])}
        else:
            images = {
                "head_camera": _decode_jpeg_b64(image_payload["head_camera"]),
                "left_color": _decode_jpeg_b64(image_payload["left_color"]),
                "right_color": _decode_jpeg_b64(image_payload["right_color"]),
            }
        timings_ms["decode_ms"] = (time.perf_counter() - t_decode0) * 1000.0

        t_state0 = time.perf_counter()
        proprio = self._normalize_proprio(np.asarray(payload["proprio"], dtype=np.float32))
        timings_ms["state_ms"] = (time.perf_counter() - t_state0) * 1000.0

        t_image0 = time.perf_counter()
        image_tensor = self._build_mosaic(images)
        self._sync_cuda()
        timings_ms["image_ms"] = (time.perf_counter() - t_image0) * 1000.0

        text_kwargs, task_name = self._resolve_text_kwargs(payload)
        action_horizon = int(payload.get("action_horizon", self.action_horizon))
        num_inference_steps = int(payload.get("num_inference_steps", self.num_inference_steps))
        seed = payload.get("seed", None)
        return_video = _as_bool(payload.get("return_video"), default=False)
        num_video_frames = int(payload.get("num_video_frames", self.num_video_frames))
        jpeg_quality = int(payload.get("pred_video_jpeg_quality", 85))

        self._sync_cuda()
        t_model0 = time.perf_counter()
        if return_video:
            # Joint video+action rollout (same path as trainer.evaluate).
            # Disable the internal action-only cross-check (would double cost / require seed).
            if action_horizon % (num_video_frames - 1) != 0:
                raise ValueError(
                    f"action_horizon={action_horizon} must be divisible by "
                    f"num_video_frames-1={num_video_frames - 1}"
                )
            pred = self.model.infer_joint(
                input_image=image_tensor,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                proprio=proprio,
                num_inference_steps=num_inference_steps,
                seed=seed,
                text_cfg_scale=1.0,
                test_action_with_infer_action=False,
                **text_kwargs,
            )
            mode = "infer_joint"
        else:
            pred = self.model.infer_action(
                input_image=image_tensor,
                action_horizon=action_horizon,
                proprio=proprio,
                num_inference_steps=num_inference_steps,
                seed=seed,
                **text_kwargs,
            )
            mode = "infer_action"
        self._sync_cuda()
        timings_ms["model_ms"] = (time.perf_counter() - t_model0) * 1000.0

        t_denorm0 = time.perf_counter()
        action = self._denormalize_action(pred["action"])
        timings_ms["denorm_ms"] = (time.perf_counter() - t_denorm0) * 1000.0
        dt = time.perf_counter() - t0
        timings_ms["total_ms"] = dt * 1000.0
        out: dict[str, Any] = {
            "action": action.astype(np.float32).tolist(),  # [T,16] effort on grip dims
            "infer_s": dt,
            "action_horizon": int(action.shape[0]),
            "action_dim": int(action.shape[1]),
            "mode": mode,
            # Echo the instruction actually conditioned on, so a bag records which task ran.
            "task": task_name,
            "timings_ms": timings_ms,
        }
        if return_video:
            # List[PIL.Image] mosaic rollouts, typically 9 x 384x512.
            frames = pred.get("video") or []
            out["pred_video"] = [_pil_to_jpeg_b64(f, quality=jpeg_quality) for f in frames]
            out["num_video_frames"] = int(len(frames))
            if frames:
                out["pred_video_hw"] = [int(frames[0].size[1]), int(frames[0].size[0])]
            logger.info(
                "infer_joint ok action=%s video_frames=%d infer_s=%.2f",
                tuple(action.shape),
                len(frames),
                dt,
            )
        return out
