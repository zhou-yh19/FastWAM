#!/bin/bash
# Upload a checkpoint to KS3. Run this on the TRAINING SERVER (internal endpoint).
#
#   TASK=<config name under configs/task/> \
#   RUN_ID=<directory under runs/<TASK>/> \
#   STEP=<filename under weights/, e.g. step_<N>.pt> \
#   KS3_BUCKET=ks3://<your-bucket>/<prefix> \
#   bash push_checkpoint_ks3.sh
#
# Run pull_checkpoint.sh on the deploy machine with the SAME three values: KS3_PREFIX
# is derived from them, so the two ends cannot drift apart.
#
# Sent:     weights .pt (12G), dataset_stats.json, config.yaml, text embeds, configs/,
#           VAE (1.4G), taskmap.json if present
# Not sent: DiT shards (19G) + ActionDiT (2G) -- serving skips them via
#           skip_dit_load_from_pretrain=True; T5 encoder (11G) -- only needed with
#           --load-text-encoder
# The VAE must be sent: save_checkpoint stores only mot + proprio_encoder.
set -euo pipefail

# ---- the run to upload ----
TASK="${TASK:?required: config name under configs/task/ (no .yaml)}"
RUN_ID="${RUN_ID:?required: directory name under runs/$TASK/}"
STEP="${STEP:?required: filename under weights/, e.g. step_<N>.pt}"

# Repo root defaults to this script's directory, so it travels between machines.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_DIR="${RUN_DIR:-$REPO/runs/$TASK/$RUN_ID}"
VAE_FILE="$REPO/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
TASK_MAP="$REPO/taskmap.json"

# ---- KS3 ----
KS3_BUCKET="${KS3_BUCKET:?required: KS3 bucket, e.g. ks3://<bucket>/<prefix>}"
KS3_PREFIX="$KS3_BUCKET/$TASK/$RUN_ID"
ENDPOINT="${ENDPOINT:-ks3-cn-beijing-internal.ksyuncs.com}"   # internal endpoint
# Pin --log-path: without it ks3util creates logs/info.log in whatever the current
# working directory happens to be.
KS3_OPTS=(-f -j 24 --bigfile-threshold=104857600 --parallel=32 \
  --log-path="$REPO/logs/ks3" -e "$ENDPOINT")

echo "==== 0. pre-flight ===="
echo "task=$TASK  run_id=$RUN_ID  step=$STEP"
echo "ks3=$KS3_PREFIX"
echo
ls -lh "$RUN_DIR/checkpoints/weights/$STEP" "$RUN_DIR/dataset_stats.json" "$RUN_DIR/config.yaml"
ls -lh "$VAE_FILE"
# taskmap.json comes from make_task_map.py, is generated per deployment and is not in
# version control. Absent is fine unless you need to select instructions by name.
if [[ -f "$TASK_MAP" ]]; then
  ls -lh "$TASK_MAP"
else
  echo "note: $TASK_MAP absent, skipping. Generate it with make_task_map.py if needed."
fi

# Read the embeds dir from the run's own config snapshot rather than hardcoding it.
EMBED_REL="$(grep -m1 'text_embedding_cache_dir:' "$RUN_DIR/config.yaml" \
  | sed 's/.*text_embedding_cache_dir:[[:space:]]*//; s#^\./##; s/[[:space:]]*$//')"
[[ -n "$EMBED_REL" ]] || { echo "no text_embedding_cache_dir in config.yaml" >&2; exit 1; }
EMBED_DIR="$REPO/$EMBED_REL"
[[ -d "$EMBED_DIR" ]] || { echo "embeds dir not found: $EMBED_DIR" >&2; exit 1; }
echo "embeds: $EMBED_REL — $(ls -1 "$EMBED_DIR" | wc -l) files, $(du -sh "$EMBED_DIR" | cut -f1)"
ls -d "$REPO/configs"

# The KS3 layout mirrors the deploy machine, so the download side is one cp -r.
echo
echo "==== 1. weights (~12G) ===="
ks3util cp "${KS3_OPTS[@]}" \
  "$RUN_DIR/checkpoints/weights/$STEP" \
  "$KS3_PREFIX/run/checkpoints/weights/$STEP"

echo
echo "==== 2. dataset_stats.json + config.yaml ===="
ks3util cp "${KS3_OPTS[@]}" "$RUN_DIR/dataset_stats.json" "$KS3_PREFIX/run/dataset_stats.json"
ks3util cp "${KS3_OPTS[@]}" "$RUN_DIR/config.yaml"        "$KS3_PREFIX/run/config.yaml"

echo
echo "==== 3. text embeddings ===="
ks3util cp -r "${KS3_OPTS[@]}" "$EMBED_DIR/" "$KS3_PREFIX/embeds/"

echo
echo "==== 4. configs (serving recomposes via hydra from task=$TASK) ===="
ks3util cp -r "${KS3_OPTS[@]}" "$REPO/configs/" "$KS3_PREFIX/configs/"

echo
echo "==== 5. VAE (1.4G) + taskmap.json if present ===="
ks3util cp "${KS3_OPTS[@]}" "$VAE_FILE" \
  "$KS3_PREFIX/base/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
if [[ -f "$TASK_MAP" ]]; then
  ks3util cp "${KS3_OPTS[@]}" "$TASK_MAP" "$KS3_PREFIX/taskmap.json"
fi

echo
echo "==== 6. verify the KS3 listing ===="
# ls on a prefix is recursive by default (-d lists only the current level)
ks3util ls "$KS3_PREFIX/" -e "$ENDPOINT"

echo
echo "==== next: run this on the deploy machine with identical values ===="
echo "  TASK=$TASK \\"
echo "  RUN_ID=$RUN_ID \\"
echo "  STEP=$STEP \\"
echo "  bash pull_checkpoint.sh"
