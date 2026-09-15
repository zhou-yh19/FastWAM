#!/usr/bin/env python3
"""Keep only the newest N accelerate state/ checkpoints of a run.

A ZeRO-1 state directory for this 5B model is ~80 GB (8 optimizer shards + model states),
so a 40k-step run at save_every=2000 would write ~1.5 TB. The weights/*.pt files are ~12 GB
each and are what you actually deploy from, so those are never touched -- only the resumable
optimizer states are pruned.

Safety rules:
  * only ever removes directories matching state/step_XXXXXX inside the given run
  * always keeps the newest `--keep` states, so an in-progress save is never a candidate
  * skips a directory that has no trainer_state.json (a partially written save)
  * --dry-run prints what would go without touching anything

Run once, or with --watch to poll while training runs.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

STEP_DIR = re.compile(r"^step[_-](\d+)$")


def disk_free_gb(path: Path) -> float:
    out = subprocess.run(["df", "-BG", "--output=avail", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip().splitlines()[-1].rstrip("G"))


def find_states(run_dir: Path) -> list[tuple[int, Path]]:
    state_root = run_dir / "checkpoints" / "state"
    if not state_root.is_dir():
        return []
    found = []
    for d in state_root.iterdir():
        m = STEP_DIR.match(d.name)
        if d.is_dir() and m:
            found.append((int(m.group(1)), d))
    return sorted(found)


def prune(run_dir: Path, keep: int, dry_run: bool) -> int:
    states = find_states(run_dir)
    if len(states) <= keep:
        return 0

    doomed = states[:-keep]
    freed = 0
    for step, d in doomed:
        # A save writes trainer_state.json last; without it the dir may be mid-write.
        # Newest `keep` are already excluded, so this only guards against odd leftovers.
        if not (d / "trainer_state.json").exists():
            print(f"  skip step_{step:06d} (no trainer_state.json -- incomplete?)")
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        if dry_run:
            print(f"  [dry-run] would remove {d} ({size / 2**30:.1f} GB)")
        else:
            shutil.rmtree(d)
            print(f"  removed {d} ({size / 2**30:.1f} GB)")
        freed += size
    return freed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="a runs/<task>/<timestamp> directory")
    ap.add_argument("--keep", type=int, default=2, help="how many newest states to retain")
    ap.add_argument("--watch", type=int, default=0, metavar="SEC",
                    help="poll every SEC seconds instead of running once")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.keep < 1:
        raise SystemExit("--keep must be >= 1 (the newest state is needed to resume)")
    if not args.run_dir.is_dir():
        raise SystemExit(f"not a directory: {args.run_dir}")

    while True:
        states = find_states(args.run_dir)
        stamp = time.strftime("%F %T")
        if states:
            kept = [f"step_{s:06d}" for s, _ in states[-args.keep:]]
            print(f"{stamp} states={[s for s, _ in states]} keep={kept} "
                  f"free={disk_free_gb(args.run_dir):.0f}G", flush=True)
        freed = prune(args.run_dir, args.keep, args.dry_run)
        if freed:
            print(f"{stamp} freed {freed / 2**30:.1f} GB, "
                  f"now {disk_free_gb(args.run_dir):.0f}G free", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
