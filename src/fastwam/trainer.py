import logging
import json
import inspect
import os
import re
import traceback
from datetime import timedelta
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)

# Rank-local logger: `get_logger` disables non-zero ranks, which is right for progress
# spam but wrong for a crash report -- the rank that fails is usually not rank 0.
_rank_logger = logging.getLogger(f"{__name__}.rank")


class _CollectiveGuard:
    """Convert a single-rank exception into a whole-job abort.

    See `Wan22Trainer._guard_collectives`. On the way out of a `with` block that raised, this
    prints the traceback tagged with the failing rank and then calls `dist.destroy_process_
    group()`, which makes the peers' in-flight collectives fail fast instead of hanging for
    the full watchdog timeout. The original exception is left to propagate.
    """

    def __init__(self, trainer, what: str):
        self._trainer = trainer
        self._what = what

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        rank = self._trainer.accelerator.process_index
        _rank_logger.error(
            "[rank %d] Exception inside `%s`, which sits between collectives. "
            "Aborting the process group so peers fail fast instead of hanging until the "
            "NCCL watchdog fires.\n%s",
            rank,
            self._what,
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pragma: no cover - best effort teardown on an already-failing path
            _rank_logger.exception("[rank %d] Failed to destroy the process group.", rank)
        return False


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        # Eval-only knobs. Read with defaults so older configs/checkpoints still load.
        # `eval_seed` is deliberately decoupled from `cfg.seed` (which drives training data
        # order) so that pinning the eval set does not perturb training.
        self.eval_seed = int(getattr(cfg, "eval_seed", 12345))
        self.eval_num_loss_timesteps = int(getattr(cfg, "eval_num_loss_timesteps", 5))
        self.eval_num_loss_samples = int(getattr(cfg, "eval_num_loss_samples", 2))
        if self.eval_num_loss_timesteps < 1:
            raise ValueError(
                f"`eval_num_loss_timesteps` must be >= 1, got {self.eval_num_loss_timesteps}."
            )
        if self.eval_num_loss_samples < 1:
            raise ValueError(
                f"`eval_num_loss_samples` must be >= 1, got {self.eval_num_loss_samples}."
            )
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        # After accelerate-state resume: keep weights/opt momentum/data progress,
        # but rebuild LR schedule from cfg.learning_rate for the remaining steps.
        self.resume_reinit_lr = bool(getattr(cfg, "resume_reinit_lr", False))
        # Warmup fraction used when rebuilding that schedule. The default 5% suits dropping to
        # a *lower* peak after a plateau. A pure anneal-to-zero tail wants 0.0 instead: there
        # the peak is the LR the checkpoint already carries, so re-warming would first crash
        # the LR to peak/warmup_steps and climb back, undoing the anneal for those steps.
        self.resume_warmup_frac = float(getattr(cfg, "resume_warmup_frac", 0.05))
        if not 0.0 <= self.resume_warmup_frac < 1.0:
            raise ValueError(
                f"`resume_warmup_frac` must be in [0.0, 1.0), got {self.resume_warmup_frac}."
            )
        additional_steps = getattr(cfg, "additional_steps", None)
        self.additional_steps = int(additional_steps) if additional_steps is not None else None
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # How long a collective may stall before the NCCL watchdog aborts the job. The old
        # default was DeepSpeed's 30 min, which meant a rank-divergence hang burned half an
        # hour of 8 GPUs before saying anything.
        self.dist_timeout_sec = int(getattr(cfg, "dist_timeout_sec", 600))
        if self.dist_timeout_sec < 60:
            raise ValueError(
                f"`dist_timeout_sec` must be >= 60, got {self.dist_timeout_sec}."
            )
        # `InitProcessGroupKwargs.to_kwargs()` only forwards fields that *differ* from its
        # own defaults, and its nccl default is exactly 600s. Requesting 600 would therefore
        # forward nothing and let DeepSpeed fall back to its own 1800s default -- the silent
        # trap behind the `Timeout(ms)=1800000` in the step-10500 hang. Nudge by 1s so the
        # value always survives the filter and the configured number is the one in effect.
        forwarded_timeout = self.dist_timeout_sec
        if forwarded_timeout == 600:
            forwarded_timeout = 601
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(seconds=forwarded_timeout))
            ],
        )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")
        if self.resume_reinit_lr or self.additional_steps is not None:
            logger.warning(
                "resume_reinit_lr / additional_steps only apply to accelerate state/ resume; "
                "weight-only .pt resume already builds a fresh LR schedule and resets data progress."
            )

    def _maybe_reinit_lr_after_state_resume(self) -> None:
        """Keep Adam momentum + dataloader cursor; optionally rebuild LR at cfg.learning_rate.

        Use after a loss plateau when you want a lower peak LR without reshuffling data.
        Extend the run with additional_steps=N (preferred) or a larger absolute max_steps.
        """
        if self.additional_steps is not None:
            self.max_steps = int(self.global_step) + int(self.additional_steps)
            logger.info(
                "Extended max_steps -> %d (global_step=%d + additional_steps=%d)",
                self.max_steps,
                self.global_step,
                self.additional_steps,
            )

        if not self.resume_reinit_lr:
            if self.max_steps is not None and self.global_step >= self.max_steps:
                logger.warning(
                    "Resumed at global_step=%d with max_steps=%d; training will exit immediately. "
                    "Set additional_steps=N or a larger max_steps to continue.",
                    self.global_step,
                    self.max_steps,
                )
            return

        if self.max_steps is None:
            raise ValueError(
                "resume_reinit_lr=true requires max_steps (or additional_steps) so the "
                "remaining cosine/warmup horizon is well-defined."
            )
        if self.global_step >= self.max_steps:
            raise ValueError(
                f"Cannot reinit LR: global_step={self.global_step} >= max_steps={self.max_steps}. "
                "Pass additional_steps=N (recommended) or set max_steps > current step."
            )

        remaining = int(self.max_steps) - int(self.global_step)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
            group["initial_lr"] = self.learning_rate

        warmup_steps = int(remaining * self.resume_warmup_frac)
        self.scheduler = self._build_scheduler(
            scheduler_type=self.cfg.lr_scheduler_type,
            total_train_steps=remaining,
            warmup_steps=warmup_steps,
        )
        logger.info(
            "Rebuilt LR after state resume: peak_lr=%s remaining_steps=%d warmup=%d "
            "(warmup_frac=%.3f, optimizer momentum + dataloader progress kept)",
            self.learning_rate,
            remaining,
            warmup_steps,
            self.resume_warmup_frac,
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    def _barrier(self):
        """`wait_for_everyone` that does not depend on any per-rank value."""
        self.accelerator.wait_for_everyone()

    def _train_metric_keys(self, loss_dict):
        """Key order for the packed per-step metric gather, pinned on the first optimizer step.

        The gathered row must have the same width on every rank at every step. Pinning the key
        set the first time we see it and then requiring an exact match turns a drifting
        `loss_dict` into an immediate, readable error on the rank that drifted, instead of a
        silent shape mismatch that hangs the whole group inside NCCL.
        """
        keys = tuple(sorted(loss_dict))
        pinned = getattr(self, "_pinned_train_metric_keys", None)
        if pinned is None:
            self._pinned_train_metric_keys = keys
            return keys
        if keys != pinned:
            raise RuntimeError(
                f"[rank {self.accelerator.process_index}] `training_loss` returned loss_dict "
                f"keys {keys}, but this run pinned {pinned} at its first step. The gathered "
                "metric width must be identical on every rank and every step."
            )
        return pinned

    def _agree_bool(self, local_ok: bool, what: str) -> bool:
        """Turn a per-rank boolean into a value every rank agrees on.

        Any `if <local condition>` around a collective is a deadlock: the ranks that take
        the branch enqueue work the others never do, and NCCL waits until the watchdog
        kills the job. Routing the condition through a single fixed-shape all-reduce means
        every rank leaves this function with the *same* answer, so the branch is safe.

        Returns True only when the condition held on every rank.
        """
        flag = torch.tensor(
            [1.0 if local_ok else 0.0],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        agreed = self.accelerator.reduce(flag, reduction="sum")
        num_ok = int(agreed.item())
        world = self.accelerator.num_processes
        if 0 < num_ok < world:
            logger.warning(
                "Rank divergence on `%s`: %d/%d ranks reported True. "
                "Treating it as False everywhere to keep collectives aligned.",
                what,
                num_ok,
                world,
            )
        return num_ok == world

    def _guard_collectives(self, what: str):
        """Abort the whole job if one rank raises where others are about to communicate.

        A bare exception on a single rank is as fatal as a shape mismatch and looks worse in
        the logs: the raising rank unwinds and stops calling collectives, the other seven sit
        in an all-reduce until the watchdog fires, and the traceback that explains it is
        buried in whichever rank's stream nobody is tailing. This logs the traceback with its
        rank, then tears the process group down so every rank exits promptly with a cause.
        """
        return _CollectiveGuard(self, what)

    def _fixed_eval_indices(self):
        """Sample indices used by `evaluate`, pinned for the lifetime of the run.

        The seed deliberately excludes `global_step`, so every eval scores the *same*
        samples. Without this, the step-to-step movement of the eval curves is dominated
        by which samples happened to be drawn rather than by the model changing.
        """
        num_samples = min(self.eval_num_loss_samples, len(self.val_dataset))
        rng = torch.Generator(device="cpu").manual_seed(self.eval_seed + self.accelerator.process_index)
        # Collisions are possible but harmless (they just reweight the average) and
        # vanishingly rare for realistic dataset sizes.
        return torch.randint(0, len(self.val_dataset), (num_samples,), generator=rng).tolist()

    # Loss components broken out on the eval grid, in a fixed order so the gathered metric
    # tensor has the same layout on every rank.
    EVAL_LOSS_COMPONENTS = ("total", "loss_video", "loss_action")

    def _fixed_grid_val_loss(self, model, samples):
        """`val_loss` on a deterministic timestep grid, split into video/action.

        `training_loss` normally draws a fresh random timestep per call, and the flow-matching
        loss varies strongly with that timestep -- the `training_weight` alone has a ~68% CV
        under the training distribution, which is most of the noise in the logged curve. Here
        the timesteps come from `build_training_t_grid`, a deterministic quantile grid over
        that same distribution, and the noise is pinned by a seeded generator. Same weights +
        same samples => same number, every time.

        The split matters because `loss_total = loss_video + loss_action` and the two move on
        very different scales; a combined number hides what the video branch is doing.

        Returns `{component: per_timestep_means}` for *every* component in
        `EVAL_LOSS_COMPONENTS`, with NaN in the slots the model did not report. The width is
        therefore identical on every rank, which is what makes the gather in `evaluate` safe.
        """
        t_grid = model.train_video_scheduler.build_training_t_grid(
            self.eval_num_loss_timesteps,
            device=model.device,
            dtype=torch.float32,
        )
        # A generator must sit on the exact device the noise is drawn on. `model.device` is
        # normally `cuda:<local_rank>`, but a bare `cuda` would silently resolve to the
        # current device and mismatch on non-zero ranks.
        generator_device = model.device
        if generator_device.type == "cuda" and generator_device.index is None:
            generator_device = torch.device("cuda", torch.cuda.current_device())

        per_timestep = {key: [] for key in self.EVAL_LOSS_COMPONENTS}
        for t in t_grid:
            accumulated = {key: [] for key in self.EVAL_LOSS_COMPONENTS}
            for sample_idx, sample in enumerate(samples):
                # Re-seeded per (timestep, sample), so a given sample sees the same noise at
                # every timestep and at every eval -- common random numbers across the grid.
                generator = torch.Generator(device=generator_device).manual_seed(
                    self.eval_seed + sample_idx
                )
                with self.accelerator.autocast():
                    loss, loss_dict = model.training_loss(
                        sample,
                        timestep_video=t,
                        timestep_action=t,
                        generator=generator,
                    )
                accumulated["total"].append(loss.float().item())
                for key in self.EVAL_LOSS_COMPONENTS[1:]:
                    if key in loss_dict:
                        accumulated[key].append(float(loss_dict[key]))
            for key, values in accumulated.items():
                # A component the model does not report stays NaN here. Unlike before, the
                # entry is still *present*: dropping keys locally made the gathered tensor's
                # width rank-dependent, which is a deadlock (see `evaluate`). NaN is the
                # sentinel and it is filtered after the gather, identically on every rank.
                per_timestep[key].append(sum(values) / len(values) if values else float("nan"))

        # Always the full `EVAL_LOSS_COMPONENTS` x `eval_num_loss_timesteps` grid, so the
        # layout is a function of config alone -- never of what this rank's samples contained.
        assert all(
            len(values) == len(t_grid) for values in per_timestep.values()
        ), f"eval loss grid is ragged: { {k: len(v) for k, v in per_timestep.items()} }"
        return per_timestep

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        eval_indices = self._fixed_eval_indices()
        samples = [self._to_batched_eval_sample(self.val_dataset[i]) for i in eval_indices]

        # 1. training loss, on a fixed timestep grid over all pinned samples.
        loss_by_component = self._fixed_grid_val_loss(model, samples)
        val_loss_per_timestep = loss_by_component["total"]
        val_loss = sum(val_loss_per_timestep) / len(val_loss_per_timestep)

        # The rollout below is far more expensive than the loss, so it stays on one sample.
        sample = samples[0]
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        # --- Gathered metric layout -------------------------------------------------------
        # The width below is a pure function of `EVAL_LOSS_COMPONENTS` and
        # `eval_num_loss_timesteps`, both of which come from config and are therefore
        # identical on every rank. Nothing here may depend on what *this* rank's samples
        # happened to contain: a rank-dependent width makes the all-gather shapes disagree,
        # which hangs every rank until the NCCL watchdog kills the job. That is exactly the
        # failure that killed the 2026-08-29 run at step 10500.
        #
        #   [0:7]   scalar video/loss metrics
        #   [7:9]   action_l2, action_l1 (NaN when this rank has no action metrics)
        #   [9:]    per-component per-timestep losses, `EVAL_LOSS_COMPONENTS` order,
        #           `eval_num_loss_timesteps` values each, NaN where not reported
        num_t = self.eval_num_loss_timesteps
        component_block = []
        for key in self.EVAL_LOSS_COMPONENTS:
            values = loss_by_component.get(key) or []
            if len(values) != num_t:
                raise RuntimeError(
                    f"Eval loss component `{key}` has {len(values)} timesteps, expected "
                    f"{num_t}. The gathered metric width must not vary across ranks."
                )
            component_block.extend(float(v) for v in values)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                # NaN rather than -1.0: a missing value must not be averaged in as data.
                float(action_l2) if action_l2 is not None else float("nan"),
                float(action_l1) if action_l1 is not None else float("nan"),
                *component_block,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        expected_width = 9 + len(self.EVAL_LOSS_COMPONENTS) * num_t
        assert local_metrics.shape == (1, expected_width), (
            f"eval metric tensor is {tuple(local_metrics.shape)}, expected (1, {expected_width})"
        )

        # `_agree_bool` is itself a fixed-shape collective, so it is safe to call before the
        # gather; it replaces the old `action_l2 is not None` local branch below.
        has_action_metrics = self._agree_bool(action_l2 is not None, "eval action metrics")

        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if has_action_metrics else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if has_action_metrics else None

        # Unpack the fixed-width block. Filtering happens *here*, on the gathered result, so
        # every rank drops exactly the same components.
        component_means = {}
        offset = 9
        for key in self.EVAL_LOSS_COMPONENTS:
            block = gathered_metrics[:, offset : offset + num_t]
            offset += num_t
            if bool(torch.isnan(block).any()):
                # Not reported by the model (or not by some rank) -- skip it everywhere.
                continue
            component_means[key] = block.mean(dim=0).tolist()

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "val_loss_per_timestep": [float(v) for v in component_means.get("total", [])],
            # e.g. {"loss_video": {"mean": .., "per_timestep": [..]}, "loss_action": {...}}
            "val_loss_components": {
                key: {
                    "mean": sum(values) / len(values),
                    "per_timestep": [float(v) for v in values],
                }
                for key, values in component_means.items()
                if key != "total" and values
            },
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        # The rank-0-only writes below sit between barriers: if one of them raises (disk full,
        # bad path, serialization error) rank 0 stops calling collectives while the other
        # ranks wait in `wait_for_everyone`, and the job hangs rather than reporting the disk
        # error. The guard turns that into a prompt, attributable failure.
        with self._guard_collectives("save_checkpoint"):
            self._barrier()
            ckpt_path = None
            if self.accelerator.is_main_process:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self._barrier()

            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self._barrier()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self._maybe_reinit_lr_after_state_resume()
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )
        self._maybe_reinit_lr_after_state_resume()
        self.accelerator.wait_for_everyone()

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            step_start_time = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            # Time spent blocked on the dataloader. Under DDP the fast ranks then burn this
            # same wall-clock spinning inside the NCCL collective waiting for the slow one,
            # so this is gathered as both mean and max below: max >> mean means a single
            # straggler rank, both large means the input pipeline is globally too slow.
            data_wait_sec = time.perf_counter() - step_start_time

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    # One gather of a fixed-width row, not one gather per `loss_dict` key.
                    # The old loop issued `len(loss_dict)` collectives, so a rank whose
                    # `loss_dict` had a different key set would issue a different *number* of
                    # collectives and desynchronise the group for the rest of the run --
                    # matching the step-10500 hang, where six ranks had enqueued one more
                    # collective (52558) than ranks 0 and 4 (52557).
                    metric_keys = self._train_metric_keys(loss_dict)
                    global_loss_row = torch.tensor(
                        [
                            float(loss.detach()),
                            float(grad_norm),
                            *[float(loss_dict[key]) for key in metric_keys],
                        ],
                        device=loss.device,
                        dtype=torch.float32,
                    ).reshape(1, -1)
                    gathered_loss_row = self.accelerator.gather(global_loss_row)
                    row_means = gathered_loss_row.mean(dim=0)
                    global_loss = float(row_means[0].item())
                    global_grad_norm = float(row_means[1].item())
                    global_loss_metrics = {
                        key: float(row_means[2 + i].item()) for i, key in enumerate(metric_keys)
                    }

                    # Measured after the gather above, whose `.item()` forces a CUDA sync,
                    # so this covers real compute rather than just kernel launches.
                    step_total_sec = time.perf_counter() - step_start_time
                    timing_tensor = torch.tensor(
                        [data_wait_sec, step_total_sec], device=loss.device, dtype=torch.float32
                    ).reshape(1, 2)
                    gathered_timing = self.accelerator.gather(timing_tensor)
                    data_wait_mean = float(gathered_timing[:, 0].mean().item())
                    data_wait_max = float(gathered_timing[:, 0].max().item())
                    step_total_mean = float(gathered_timing[:, 1].mean().item())
                    # The slowest rank sets the pace for the whole step, so its wait is what
                    # actually costs wall-clock.
                    compute_sec = max(0.0, step_total_mean - data_wait_max)
                    data_wait_frac = (data_wait_max / step_total_mean) if step_total_mean > 0 else 0.0

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        # `global_step` is an optimizer step; include accumulation in
                        # throughput so samples/s reports effective samples processed.
                        effective_batch_size = (
                            self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * effective_batch_size,
                            eta_str,
                        )
                        description += " data_wait=%.1fs/%.1fs(mean/max) compute=%.1fs stall=%.0f%%" % (
                            data_wait_mean,
                            data_wait_max,
                            compute_sec,
                            100.0 * data_wait_frac,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * effective_batch_size,
                            "performance/data_wait_sec_mean": data_wait_mean,
                            "performance/data_wait_sec_max": data_wait_max,
                            "performance/data_wait_frac": data_wait_frac,
                            "performance/compute_sec": compute_sec,
                            "performance/step_total_sec": step_total_mean,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        # `evaluate` runs inference, VAE decode and mp4 encode per rank before
                        # its gather. Anything that raises in there (a corrupt val shard, a
                        # bad frame, a full disk) would otherwise leave the peers stuck in the
                        # gather until the watchdog fires.
                        with self._guard_collectives("evaluate"):
                            metrics = self.evaluate()
                        self._barrier()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            components = metrics.get("val_loss_components") or {}
                            for key in sorted(components):
                                description += " val_%s=%.4f" % (key, components[key]["mean"])
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            per_timestep = metrics.get("val_loss_per_timestep") or []
                            if per_timestep:
                                description += " val_loss_by_t=[%s]" % ", ".join(
                                    "%.4f" % v for v in per_timestep
                                )
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            # Bucketed by noise level, low-noise first (see
                            # `build_training_t_grid`): shows *where* the model improves.
                            for i, value in enumerate(per_timestep):
                                eval_payload[f"eval/val_loss_t{i}"] = float(value)
                            # Video/action split -- the combined val_loss is dominated by
                            # whichever branch is larger, so each gets its own curve.
                            for key, entry in components.items():
                                eval_payload[f"eval/val_{key}"] = float(entry["mean"])
                                for i, value in enumerate(entry["per_timestep"]):
                                    eval_payload[f"eval/val_{key}_t{i}"] = float(value)
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
