#!/usr/bin/env python3
"""Cache the deploy prompt's T5 embedding so inference can skip the text encoder.

The umT5-XXL text encoder is ~11 GB in bf16 -- roughly as large as the video +
action DiTs combined. Deployment always uses a single fixed prompt, so encoding it
once and reusing the tensor drops the resident footprint from ~24 GB to ~13 GB.
That is the difference between fitting and OOM-ing when the GPUs are shared.

`FastWAMTA2Policy` reads this cache when constructed with load_text_encoder=False.
The cache is file-compatible with scripts/precompute_text_embeds.py (same sha256
naming, same {context, mask} payload), so entries written by either are usable.

Encode the deploy prompt (defaults to CPU so it never touches a busy GPU):

  cd <FastWAM repo root>
  conda activate fastwam
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  python experiments/teleavatar_v2_deploy/server/prompt_embed_cache.py \\
    --cache-dir data/text_embeds_cache/<your_task> \\
    --prompt-task '<instruction as recorded in meta/tasks.jsonl>'
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128



def model_id_to_enc_id(model_id: str = DEFAULT_MODEL_ID) -> str:
    base = str(model_id).split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", base.lower()) or "textenc"


def cache_path(cache_dir: Path, prompt: str, context_len: int = DEFAULT_CONTEXT_LEN,
               model_id: str = DEFAULT_MODEL_ID) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{hashed}.t5_len{context_len}.{model_id_to_enc_id(model_id)}.pt"


def load_cached(cache_dir: Path, prompt: str, context_len: int = DEFAULT_CONTEXT_LEN,
                model_id: str = DEFAULT_MODEL_ID) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (context[L,D], mask[L]) if cached, else None."""
    path = cache_path(cache_dir, prompt, context_len, model_id)
    if not path.is_file():
        return None
    payload = torch.load(str(path), map_location="cpu")
    return payload["context"], payload["mask"]


def encode_and_cache(
    cache_dir: Path,
    prompt: str,
    *,
    context_len: int = DEFAULT_CONTEXT_LEN,
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_model_id: str = DEFAULT_TOKENIZER_MODEL_ID,
    redirect_common_files: bool = True,
    device: str = "cpu",
    overwrite: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load only the text encoder + tokenizer, encode one prompt, cache the result."""
    path = cache_path(cache_dir, prompt, context_len, model_id)
    if path.is_file() and not overwrite:
        payload = torch.load(str(path), map_location="cpu")
        return payload["context"], payload["mask"]

    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    # float32 on CPU: bf16 matmul on CPU is slow and this runs exactly once.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=dtype, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=context_len,
                                     clean="whitespace")

    with torch.no_grad():
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device=device, dtype=torch.bool)
        context = text_encoder(ids, mask)

    ctx = context[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    msk = mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save({"context": ctx, "mask": msk}, str(tmp))
    os.replace(tmp, path)

    del text_encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ctx, msk


def main() -> None:
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="text_embedding_cache_dir from your data config")
    ap.add_argument(
        "--prompt-task", type=str, required=True,
        help="Instruction to encode, exactly as recorded in the dataset's meta/tasks.jsonl. "
             "The cache file is named after sha256 of the formatted prompt, so any wording "
             "difference (including whitespace) produces a file the trainer will not find.",
    )
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu keeps this off a busy GPU; it only runs once")
    ap.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompt = DEFAULT_PROMPT.format(task=args.prompt_task)
    path = cache_path(args.cache_dir, prompt, args.context_len)
    print(f"prompt : {prompt}")
    print(f"cache  : {path}")
    print(f"exists : {path.is_file()}")
    ctx, msk = encode_and_cache(args.cache_dir, prompt, context_len=args.context_len,
                                device=args.device, overwrite=args.overwrite)
    print(f"context={tuple(ctx.shape)} {ctx.dtype}  mask={tuple(msk.shape)} "
          f"valid_tokens={int(msk.sum())}")
    print("ok")


if __name__ == "__main__":
    main()
