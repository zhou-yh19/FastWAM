#!/bin/bash
# Start the FastWAM inference server (WebSocket + msgpack).
#
#   TASK=<task> bash start_local_serve_ws.sh [num_inference_steps]
#
# RUN and STEP default to the newest run and the highest step, so the common case needs
# no paths. Override any of them, or give full paths and skip RUN/STEP entirely.
#
#   TASK           required. Config name under configs/task/ (no .yaml). Must be the one
#                  the checkpoint was trained with -- hydra rebuilds the model dims from it.
#   RUN            run directory. Default: newest under runs/$TASK.
#   STEP           filename under weights/. Default: highest step in that run.
#   CHECKPOINT     explicit weights path (overrides RUN/STEP).
#   DATASET_STATS  explicit stats path. Default: $RUN/dataset_stats.json.
#   TASK_MAP       taskmap.json path; skipped if absent.
#   PROMPT_TASK    startup instruction; resolved from the dataset meta if unset.
#   PROJECT_ROOT   repo root. Default: this script's directory.
#   PYTHON_BIN     interpreter. Default: python.
#   HOST / PORT    listen address. Default: 0.0.0.0 / 8000.
#
# Only one server per host: port 8000 is fixed and two processes would contend for it
# and for the GPU.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PROJECT_ROOT/checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_STEPS="${1:-10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TASK="${TASK:?required: TASK=<config name under configs/task/, without .yaml>}"

# Validate TASK before looking for a run: otherwise a typo reports "no runs found" and
# points at the wrong problem.
[[ -f "configs/task/$TASK.yaml" ]] || {
  echo "not found: configs/task/$TASK.yaml -- typo? Available:" >&2
  ls -1 configs/task/*.yaml 2>/dev/null | sed 's|.*/||; s|\.yaml$||; s|^|    |' >&2
  exit 1
}

# ---- locate the run --------------------------------------------------------- #
if [[ -z "${RUN:-}" ]]; then
  RUN="$(ls -td "runs/$TASK"/*/ 2>/dev/null | head -1 || true)"
  RUN="${RUN%/}"
  [[ -n "$RUN" ]] || {
    echo "no runs under runs/$TASK/ -- pass RUN=runs/$TASK/<RUN_ID>" >&2
    exit 1
  }
  echo "RUN unset, using newest: $RUN"
fi

# ---- locate the weights ----------------------------------------------------- #
if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ -z "${STEP:-}" ]]; then
    # Sort by step number, not mtime: a re-copied file has a misleading mtime.
    STEP="$(ls -1 "$RUN/checkpoints/weights"/step_*.pt 2>/dev/null | sed 's|.*/||' | sort -V | tail -1 || true)"
    [[ -n "$STEP" ]] || { echo "no step_*.pt under $RUN/checkpoints/weights/" >&2; exit 1; }
    echo "STEP unset, using highest: $STEP"
  fi
  CHECKPOINT="$RUN/checkpoints/weights/$STEP"
fi
DATASET_STATS="${DATASET_STATS:-$RUN/dataset_stats.json}"

[[ -f "$CHECKPOINT" ]]    || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$DATASET_STATS" ]] || { echo "dataset_stats not found: $DATASET_STATS" >&2; exit 1; }

# ---- optional arguments ----------------------------------------------------- #
# taskmap.json comes from make_task_map.py, is generated per deployment and is not in
# version control. Without it the server still runs; you just cannot switch instructions
# by name.
EXTRA=()
TASK_MAP="${TASK_MAP:-taskmap.json}"
if [[ -f "$TASK_MAP" ]]; then
  EXTRA+=(--task-map "$TASK_MAP")
else
  echo "note: $TASK_MAP absent, not passing --task-map."
  echo "      To select instructions by name, generate it with"
  echo "      experiments/teleavatar_v2_deploy/server/make_task_map.py"
fi
if [[ -n "${PROMPT_TASK:-}" ]]; then
  EXTRA+=(--prompt-task "$PROMPT_TASK")
fi

echo "================================================"
echo "FastWAM inference server (WebSocket)"
echo "Task:          $TASK"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:    $CHECKPOINT"
echo "Dataset stats: $DATASET_STATS"
echo "Listening:     ws://$HOST:$PORT"
echo "Health:        http://$HOST:$PORT/healthz   (/healthz, not /health)"
echo "Denoise steps: $NUM_STEPS"
echo "CUDA Graph warmup: 8 10 12"
echo "================================================"
echo "Wait for 'WebSocket server listening on ws://$HOST:$PORT' before starting the client"
echo "================================================"

exec "$PYTHON_BIN" experiments/teleavatar_v2_deploy/server/serve_policy_ws.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --dataset-stats "$DATASET_STATS" \
  "${EXTRA[@]}" \
  --num-inference-steps "$NUM_STEPS" \
  --warmup-steps 8 10 12 \
  --host "$HOST" \
  --port "$PORT"
