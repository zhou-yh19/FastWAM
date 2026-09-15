#!/usr/bin/env bash
# Generic training launcher. Extra arguments are passed straight to hydra, so resume
# and LR anneal need no separate config file.
#
#   bash scripts/launch_train.sh <task-config> [hydra overrides...]
#
#   # resume
#   bash scripts/launch_train.sh <task-config> \
#     resume=./runs/<task>/<RUN_ID>/checkpoints/state/step_<N> \
#     additional_steps=<M>
#
#   # LR anneal (learning_rate = the LR in effect at the resume point, not the peak)
#   bash scripts/launch_train.sh <task-config> \
#     resume=./runs/<task>/<RUN_ID>/checkpoints/state/step_<N> \
#     resume_reinit_lr=true additional_steps=<M> \
#     resume_warmup_frac=0.0 learning_rate=<lr>
#
#   # stage / GPU count
#   ZERO=2 NPROC=4 bash scripts/launch_train.sh <task-config>
#
# Opens two tmux windows: 0=train, 1=prune. The pruner is not optional -- each ZeRO
# state snapshot is ~80 GiB and will fill the disk.
set -euo pipefail

TASK="${1:?usage: bash scripts/launch_train.sh <task-config> [hydra overrides...]}"
shift
OVERRIDES=("$@")

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZERO="${ZERO:-1}"                       # 1 or 2
NPROC="${NPROC:-8}"                     # GPUs per node
KEEP="${KEEP:-2}"                       # ZeRO state snapshots to retain
SESSION="${SESSION:-fastwam_${TASK}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fastwam}"

# ---- Pre-flight: failing here is far cheaper than failing after 8 GPUs spin up ----
[[ -f "${PROJ}/configs/task/${TASK}.yaml" ]] \
  || { echo "not found: configs/task/${TASK}.yaml" >&2; exit 1; }

# `# @package _global_` is a hydra directive; without it composition fails with
# "Could not override 'data@task.data'" -- the most common mistake here.
head -1 "${PROJ}/configs/task/${TASK}.yaml" | grep -q '@package _global_' \
  || { echo "configs/task/${TASK}.yaml: first line must be '# @package _global_'" >&2; exit 1; }

[[ -f "${PROJ}/scripts/train_zero${ZERO}.sh" ]] \
  || { echo "ZERO=${ZERO} is not valid (use 1 or 2)" >&2; exit 1; }

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
  echo "No wandb credentials: api.wandb.ai absent from ~/.netrc and WANDB_API_KEY unset." >&2
  echo "Run 'wandb login', or pass wandb.enabled=false as an override." >&2
  exit 1
fi

if [[ -z "${DRY_RUN:-}" ]] && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists: tmux attach -t ${SESSION}" >&2
  exit 1
fi

# ---- Pin RUN_ID so the pruner knows which directory to watch ----
cd "${PROJ}"
mkdir -p logs/train
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="runs/${TASK}/${RUN_ID}"
LOG="logs/train/${TASK}_${RUN_ID}.log"

printf 'session=%s  zero=%s  nproc=%s\nrun_dir=%s\nlog=%s\n' \
  "${SESSION}" "${ZERO}" "${NPROC}" "${RUN_DIR}" "${LOG}"
((${#OVERRIDES[@]})) && printf 'overrides=%s\n' "${OVERRIDES[*]}"
echo

ACTIVATE="source ${CONDA_SH} && conda activate ${CONDA_ENV}"

# DRY_RUN=1 prints the command without starting tmux or claiming GPUs.
if [[ -n "${DRY_RUN:-}" ]]; then
  cat <<EOF
[DRY_RUN] nothing will be started. This would run:

  bash scripts/train_zero${ZERO}.sh ${NPROC} task=${TASK} ${OVERRIDES[*]}

which expands (inside train_zero${ZERO}.sh) to:

  accelerate launch \\
    --config_file scripts/accelerate_configs/accelerate_zero${ZERO}_ds.yaml \\
    --num_processes ${NPROC} \\
    scripts/train.py \\
    output_dir=./${RUN_DIR} \\
    wandb.name=${TASK} \\
    task=${TASK} ${OVERRIDES[*]}

DeepSpeed config: scripts/ds_configs/ds_zero${ZERO}_config.json ("stage": ${ZERO})
Pruner:           python scripts/prune_states.py ${RUN_DIR} --keep ${KEEP} --watch 300
EOF
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n train -c "${PROJ}"
tmux send-keys -t "${SESSION}:train" \
  "${ACTIVATE} && export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 RUN_ID='${RUN_ID}' && \
bash scripts/train_zero${ZERO}.sh ${NPROC} task=${TASK} ${OVERRIDES[*]} 2>&1 | tee '${LOG}'" C-m

# Each ZeRO state snapshot is ~80 GiB. The sleep lets the trainer create the run dir first.
tmux new-window -t "${SESSION}" -n prune -c "${PROJ}"
tmux send-keys -t "${SESSION}:prune" \
  "${ACTIVATE} && sleep 300 && \
python scripts/prune_states.py '${RUN_DIR}' --keep ${KEEP} --watch 300 \
  2>&1 | tee -a 'logs/train/${TASK}_prune_${RUN_ID}.log'" C-m

echo "Attach:  tmux attach -t ${SESSION}"
echo "Windows: 0=train 1=prune   (Ctrl-b 0 / Ctrl-b 1 to switch, Ctrl-b d to detach)"
