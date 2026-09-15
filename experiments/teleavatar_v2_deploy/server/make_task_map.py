#!/usr/bin/env python3
"""Generate a --task-map for serve_policy.py from LeRobot dataset metadata.

The text-embed cache is keyed by sha256 of the formatted prompt, so a cache file cannot be
traced back to its instruction. This script goes the other way: it reads the instructions a
dataset actually recorded, hashes each one, and keeps the ones that have a cached embedding.
Names that resolve to nothing are reported rather than silently written, because a name the
server cannot serve is worse than a missing one.

Instructions are read from each dataset's `meta/tasks.jsonl` (LeRobot 2.x layout), so this
works for any dataset without hardcoding its wording. Task-map keys default to the dataset
directory name with the common `_lerobot*` / `_mono` / `_lowres` suffixes stripped; a dataset
recording several instructions gets one key per `task_index`.

Usage:
    # One key per dataset, names derived from the directory
    python experiments/teleavatar_v2_deploy/server/make_task_map.py \
        --dataset-dir data/<your_dataset>_mono \
        --cache-dir data/text_embeds_cache/<your_task> \
        --out taskmap.json

    # Several datasets at once
    python .../make_task_map.py \
        --dataset-dir data/<ds_a>_mono --dataset-dir data/<ds_b>_mono \
        --cache-dir data/text_embeds_cache/<your_task>

    # Override or add a name explicitly (repeatable); wins over the derived name
    python .../make_task_map.py --dataset-dir ... --cache-dir ... \
        --instruction 'short_key=Full instruction text as recorded in the dataset.'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402

# Suffixes the transcode pipeline appends; they describe the video variant, not the task, so
# they are stripped when deriving a task-map key from a directory name.
_DIR_SUFFIX_RE = re.compile(
    r"(_lerobot(_\d+fps)?|_mono|_lowres|_stereo|_\d+fps)+$",
    re.IGNORECASE,
)


def digest_for(instruction: str) -> str:
    """sha256 of the fully formatted prompt -- the text-embed cache's file stem."""
    return hashlib.sha256(DEFAULT_PROMPT.format(task=instruction).encode("utf-8")).hexdigest()


def key_for_dataset(dataset_dir: Path) -> str:
    """Short, stable task-map key derived from a dataset directory name."""
    return _DIR_SUFFIX_RE.sub("", dataset_dir.name) or dataset_dir.name


def instructions_from_dataset(dataset_dir: Path) -> list[tuple[int, str]]:
    """[(task_index, instruction)] from meta/tasks.jsonl, ordered by task_index."""
    tasks_file = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_file.is_file():
        raise SystemExit(
            f"no meta/tasks.jsonl under {dataset_dir} -- not a LeRobot 2.x dataset directory?"
        )
    found: list[tuple[int, str]] = []
    for line_no, line in enumerate(tasks_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{tasks_file}:{line_no}: invalid JSON: {exc}") from exc
        instruction = row.get("task")
        if not instruction:
            raise SystemExit(f"{tasks_file}:{line_no}: row has no 'task' field")
        found.append((int(row.get("task_index", len(found))), instruction))
    if not found:
        raise SystemExit(f"{tasks_file} is empty")
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="LeRobot dataset directory to read meta/tasks.jsonl from; repeatable",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="directory holding the precomputed *.pt text embeddings "
             "(the text_embedding_cache_dir from your data config)",
    )
    parser.add_argument(
        "--instruction",
        action="append",
        default=[],
        metavar="KEY=TEXT",
        help="explicit name -> instruction pair; repeatable, overrides derived names",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "taskmap.json")
    args = parser.parse_args()

    if not args.dataset_dir and not args.instruction:
        raise SystemExit("nothing to do: pass at least one --dataset-dir or --instruction")

    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"cache dir not found: {cache_dir}")
    cached = {p.name.split(".")[0]: p.name for p in sorted(cache_dir.glob("*.pt"))}
    if not cached:
        raise SystemExit(f"no *.pt embeddings under {cache_dir}")

    # name -> instruction. Datasets first, explicit pairs last so they win.
    wanted: dict[str, str] = {}
    for dataset_dir in args.dataset_dir:
        dataset_dir = dataset_dir.expanduser().resolve()
        if not dataset_dir.is_dir():
            raise SystemExit(f"dataset dir not found: {dataset_dir}")
        base = key_for_dataset(dataset_dir)
        rows = instructions_from_dataset(dataset_dir)
        for task_index, instruction in rows:
            # A single-task dataset keeps the bare name; multi-task ones get a suffix so two
            # instructions from one directory cannot silently collide.
            name = base if len(rows) == 1 else f"{base}_{task_index}"
            if name in wanted and wanted[name] != instruction:
                raise SystemExit(
                    f"name collision on '{name}': two different instructions map to it.\n"
                    f"  {wanted[name]!r}\n  {instruction!r}\n"
                    f"Disambiguate with --instruction '<key>=<text>'."
                )
            wanted[name] = instruction

    for pair in args.instruction:
        if "=" not in pair:
            raise SystemExit(f"--instruction needs KEY=TEXT, got: {pair!r}")
        name, instruction = pair.split("=", 1)
        name, instruction = name.strip(), instruction.strip()
        if not name or not instruction:
            raise SystemExit(f"--instruction needs a non-empty key and text, got: {pair!r}")
        wanted[name] = instruction

    task_map: dict[str, str] = {}
    used: set[str] = set()
    missing: list[str] = []
    width = max((len(n) for n in wanted), default=0)

    for name, instruction in sorted(wanted.items()):
        digest = digest_for(instruction)
        if digest in cached:
            task_map[name] = digest
            used.add(digest)
            print(f"  {name:{width}s} -> {digest[:16]}  ({instruction[:48]}...)")
        else:
            missing.append(name)
            print(f"  {name:{width}s} -> NO CACHED EMBEDDING (skipped)")

    args.out.write_text(json.dumps(task_map, indent=2, ensure_ascii=False) + "\n")
    print()
    print(f"wrote {len(task_map)} task(s) to {args.out}")
    if missing:
        print(
            f"missing (no precomputed embedding): {missing}\n"
            f"  run scripts/precompute_text_embeds.py for the task whose "
            f"text_embedding_cache_dir is {cache_dir}"
        )
    leftover = sorted(set(cached) - used)
    if leftover:
        print("cached but unidentified (reachable by hash prefix, unnamed):")
        for digest in leftover:
            print(f"  {digest[:16]}")


if __name__ == "__main__":
    main()
