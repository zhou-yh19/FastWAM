#!/bin/bash
# Download a checkpoint from KS3. Run this on the DEPLOY MACHINE (public endpoint).
# The three values must match what push_checkpoint_ks3.sh used exactly.
#
#   TASK=<config name under configs/task/> \
#   RUN_ID=<directory under runs/<TASK>/> \
#   STEP=<filename under weights/, e.g. step_<N>.pt> \
#   KS3_BUCKET=ks3://<your-bucket>/<prefix> \
#   PROJECT_ROOT=/path/to/FastWAM \
#   bash pull_checkpoint.sh
set -euo pipefail

# ---- the run to download ----
TASK="${TASK:?required: same task config name as the push side}"
RUN_ID="${RUN_ID:?required: same RUN_ID as the push side}"
STEP="${STEP:?required: same STEP as the push side, e.g. step_<N>.pt}"

# The layout is fixed: serve_policy.py resolves everything relative to PROJECT_ROOT.
# Defaults to this script's directory; pass PROJECT_ROOT if the machine differs.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---- KS3 ----
KS3_BUCKET="${KS3_BUCKET:?required: same KS3 bucket as the push side}"
KS3_PREFIX="$KS3_BUCKET/$TASK/$RUN_ID"
ENDPOINT="${ENDPOINT:-ks3-cn-beijing.ksyuncs.com}"   # public endpoint
KS3_OPTS=(-f -j 24 --bigfile-threshold=104857600 --parallel=32 \
  --log-path="$PROJECT_ROOT/logs/ks3" -e "$ENDPOINT")

LOCAL_RUN_DIR="$PROJECT_ROOT/runs/$TASK/$RUN_ID"
CONFIG_STAGE="$PROJECT_ROOT/configs_incoming_$RUN_ID"   # staged, never overwrites local configs
# The VAE must land in this exact layout: serving sets
# DIFFSYNTH_MODEL_BASE_PATH=$PROJECT_ROOT/checkpoints and the loader appends
# DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors itself.
LOCAL_BASE_DIR="$PROJECT_ROOT/checkpoints"

echo "==== 0. check what is on KS3 ===="
echo "task=$TASK  run_id=$RUN_ID  step=$STEP"
echo "ks3=$KS3_PREFIX"
echo
# Fail here if push never ran or a value is wrong, rather than downloading nothing
ks3util ls "$KS3_PREFIX/" -e "$ENDPOINT"

mkdir -p "$LOCAL_RUN_DIR/checkpoints/weights" "$CONFIG_STAGE" \
  "$LOCAL_BASE_DIR/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

echo
echo "==== 1. weights + stats + config (~12G) ===="
ks3util cp -r "${KS3_OPTS[@]}" "$KS3_PREFIX/run/" "$LOCAL_RUN_DIR/"

# Read the embeds dir from the config.yaml just downloaded, not a hardcoded name.
EMBED_REL="$(grep -m1 'text_embedding_cache_dir:' "$LOCAL_RUN_DIR/config.yaml" \
  | sed 's/.*text_embedding_cache_dir:[[:space:]]*//; s#^\./##; s/[[:space:]]*$//')"
[[ -n "$EMBED_REL" ]] || { echo "no text_embedding_cache_dir in config.yaml" >&2; exit 1; }
LOCAL_EMBED_DIR="$PROJECT_ROOT/$EMBED_REL"
mkdir -p "$LOCAL_EMBED_DIR"

echo
echo "==== 2. text embeddings -> $EMBED_REL ===="
ks3util cp -r "${KS3_OPTS[@]}" "$KS3_PREFIX/embeds/" "$LOCAL_EMBED_DIR/"

echo
echo "==== 3. configs -> staging area ===="
# serving composes from PROJECT_ROOT/configs via hydra using task=$TASK; it does not
# read the run's config.yaml. configs/task/$TASK.yaml must exist locally.
ks3util cp -r "${KS3_OPTS[@]}" "$KS3_PREFIX/configs/" "$CONFIG_STAGE/"

echo
echo "==== 4. VAE (1.4G) + taskmap.json ===="
# The VAE is not inside the .pt (save_checkpoint stores only mot + proprio_encoder),
# so it must be fetched separately -- serving will not start without it.
# DiT/ActionDiT are skipped via skip_dit_load_from_pretrain=True.
ks3util cp -r "${KS3_OPTS[@]}" "$KS3_PREFIX/base/" "$LOCAL_BASE_DIR/"
# taskmap.json may not exist on the push side either; not finding it is fine.
ks3util cp "${KS3_OPTS[@]}" "$KS3_PREFIX/taskmap.json" "$PROJECT_ROOT/taskmap.json" \
  || echo "note: no taskmap.json on KS3, skipping. Generate one locally if needed."

echo
echo "==== 5. verify ===="
ls -lh "$LOCAL_RUN_DIR/checkpoints/weights/$STEP" \
       "$LOCAL_RUN_DIR/dataset_stats.json" \
       "$LOCAL_RUN_DIR/config.yaml" \
       "$LOCAL_BASE_DIR/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
# taskmap is optional -- listed separately so set -e does not abort the check.
[[ -f "$PROJECT_ROOT/taskmap.json" ]] && ls -lh "$PROJECT_ROOT/taskmap.json"
echo "embeds: $(ls -1 "$LOCAL_EMBED_DIR" | wc -l) files, $(du -sh "$LOCAL_EMBED_DIR" | cut -f1)"

echo
echo "Difference between the downloaded configs and yours (review before merging):"
diff -rq "$PROJECT_ROOT/configs" "$CONFIG_STAGE" || true

echo
echo "==== next ===="
echo "1) review the diff above; at minimum merge this task config:"
echo "     cp $CONFIG_STAGE/task/$TASK.yaml $PROJECT_ROOT/configs/task/"
echo "   (also copy any new data/model config it references)"
echo "2) start the server:"
echo "     cd $PROJECT_ROOT"
echo "     python experiments/teleavatar_v2_deploy/server/serve_policy_ws.py \\"
echo "       --task $TASK \\"
echo "       --checkpoint runs/$TASK/$RUN_ID/checkpoints/weights/$STEP \\"
echo "       --dataset-stats runs/$TASK/$RUN_ID/dataset_stats.json \\"
if [[ -f "$PROJECT_ROOT/taskmap.json" ]]; then
  echo "       --task-map taskmap.json --num-inference-steps 10"
else
  echo "       --num-inference-steps 10"
fi
