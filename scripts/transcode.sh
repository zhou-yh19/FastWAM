#!/usr/bin/env bash
# Transcode LeRobot dataset video down to the resolution training actually consumes.
#
# Two stages
# ----------
#   MODE=lowres  original side-by-side stereo -> downscaled side-by-side stereo
#                head_camera  3840x1920 → 512x256
#                left/right   2560x800  → 256x80
#
#   MODE=mono    stereo -> left eye only (crop the left half, then scale to the tile)
#                head_camera  512x256 → 256x256
#                left/right   256x80  → 128x80
#
# Why: the source is 3840x1920 HEVC and decoding one 33-frame window costs ~3.2
# CPU-seconds, which starves the GPU (~94% of wall time spent waiting on data). The
# source aspect ratio matches the target tiles exactly, so rescaling is equivalent to
# the letterbox the processor would apply anyway.
#
# Usage
# -----
#   # stage 1: original -> _lowres
#   DATASETS="<ds_a> <ds_b>" bash scripts/transcode.sh
#
#   # stage 2: _lowres -> _mono (what training reads)
#   DATASETS="<ds_a> <ds_b>" MODE=mono bash scripts/transcode.sh
#
#   # tune parallelism
#   DATASETS=<ds> JOBS=20 MODE=mono bash scripts/transcode.sh
#
# DATASETS takes dataset BASE names (no _lowres / _mono suffix), relative to SRC_ROOT.
# Leave it empty to discover every dataset carrying this stage's source suffix.
#
# Afterwards patch_transcoded_meta.py is mandatory: without it info.json still
# advertises the source resolution and LeRobot reads the wrong geometry.
#   python scripts/patch_transcoded_meta.py --mode mono data/<ds>_mono
#
# Output keeps the LeRobot layout: meta/ and data/ are symlinked back to the source,
# only videos/ is rewritten. Re-running skips outputs that already exist and are intact.
set -uo pipefail

MODE="${MODE:-lowres}"                  # lowres | mono
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-${REPO_ROOT}/data}"
CRF="${CRF:-18}"
JOBS="${JOBS:-40}"
THREADS="${THREADS:-4}"
# Space-separated dataset base names. Empty = auto-discover, see below.
DATASETS="${DATASETS:-}"

case "$MODE" in
  lowres)
    SRC_SUFFIX="${SRC_SUFFIX:-}"        # original datasets carry no suffix
    DST_SUFFIX="${DST_SUFFIX:-_lowres}"
    ;;
  mono)
    SRC_SUFFIX="${SRC_SUFFIX:-_lowres}" # cropping from the downscaled copy beats 4K
    DST_SUFFIX="${DST_SUFFIX:-_mono}"
    ;;
  *)
    echo "MODE must be lowres or mono, got: $MODE" >&2; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/transcode$([[ $MODE == mono ]] && echo _mono)}"

# Per-camera ffmpeg -vf filter. lowres scales only; mono crops the left half first.
filter_for() {
  case "$MODE" in
    lowres)
      case "$1" in
        *head_camera*)              echo "scale=512:256:flags=lanczos" ;;
        *left_color*|*right_color*) echo "scale=256:80:flags=lanczos" ;;
        *)                          echo "" ;;
      esac ;;
    mono)
      # crop=iw/2:ih:0:0 takes the left half of the stereo pair = left eye
      case "$1" in
        *head_camera*)              echo "crop=iw/2:ih:0:0,scale=256:256:flags=lanczos" ;;
        *left_color*|*right_color*) echo "crop=iw/2:ih:0:0,scale=128:80:flags=lanczos" ;;
        *)                          echo "" ;;
      esac ;;
  esac
}

transcode_one() {
  local src="$1" dst="$2" filter="$3" crf="$4" threads="$5"
  # Skip files already finished and not truncated by an interrupted run
  if [[ -s "$dst" ]] && ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
       -of csv=p=0 "$dst" >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  # -g 1 makes every frame a keyframe, matching the all-intra source: training seeks to
  # random 33-frame windows, and with an inter GOP the seek cost grows with the offset.
  if ffmpeg -y -v error -nostdin -threads "$threads" -i "$src" \
       -vf "$filter" -c:v libx264 -crf "$crf" -preset fast \
       -g 1 -threads "$threads" -pix_fmt yuv420p -an \
       -f mp4 "$dst.part" </dev/null 2>>"$LOG_DIR/ffmpeg_errors.log"; then
    mv -f "$dst.part" "$dst"
  else
    rm -f "$dst.part"
    echo "FAILED $src" >>"$LOG_DIR/failed.txt"
    return 1
  fi
}
export -f transcode_one filter_for

mkdir -p "$LOG_DIR"
export LOG_DIR CRF THREADS MODE
: >"$LOG_DIR/failed.txt"

# Auto-discovery. The lowres stage has no source suffix to filter on, so it falls back
# to "every directory holding videos/ whose name lacks _lowres/_mono".
if [[ -z "$DATASETS" ]]; then
  if [[ -n "$SRC_SUFFIX" ]]; then
    DATASETS="$(
      find -L "$SRC_ROOT" -maxdepth 1 -type d -name "*${SRC_SUFFIX}" -printf '%f\n' 2>/dev/null \
        | sed "s|${SRC_SUFFIX}\$||" | sort
    )"
  else
    DATASETS="$(
      find -L "$SRC_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
        | grep -vE '_(lowres|mono)$' | while read -r d; do
            [[ -d "$SRC_ROOT/$d/videos" ]] && echo "$d"
          done | sort
    )"
  fi
  [[ -n "$DATASETS" ]] || {
    echo "nothing to do under SRC_ROOT=$SRC_ROOT (source suffix '${SRC_SUFFIX:-none}')." >&2
    echo "Pass them explicitly: DATASETS=\"<ds_a> <ds_b>\" MODE=$MODE bash scripts/transcode.sh" >&2
    exit 1
  }
  echo "DATASETS unset, discovered: $(echo "$DATASETS" | tr '\n' ' ')"
fi

# Build the work list first so progress is countable
WORK="$LOG_DIR/worklist.txt"
: >"$WORK"
for ds in $DATASETS; do
  src_ds="$SRC_ROOT/${ds}${SRC_SUFFIX}"
  dst_ds="$SRC_ROOT/${ds}${DST_SUFFIX}"
  [[ -d "$src_ds" ]] || { echo "skip missing $src_ds"; continue; }

  mkdir -p "$dst_ds"
  # meta/ and data/ are small and unchanged -- share them by symlink rather than copy.
  # patch_transcoded_meta.py later replaces meta with a real directory to fix info.json.
  for sub in meta data; do
    [[ -e "$dst_ds/$sub" ]] || ln -s "$src_ds/$sub" "$dst_ds/$sub"
  done

  while IFS= read -r src; do
    rel="${src#$src_ds/}"
    filter="$(filter_for "$rel")"
    [[ -n "$filter" ]] || continue
    printf '%s\t%s\t%s\n' "$src" "$dst_ds/$rel" "$filter" >>"$WORK"
  done < <(find "$src_ds/videos" -name '*.mp4' | sort)
done

TOTAL=$(wc -l <"$WORK")
echo "$(date '+%F %T') MODE=$MODE  $TOTAL videos, $JOBS x $THREADS threads, CRF=$CRF"
echo "  ${SRC_SUFFIX:-<original>} -> ${DST_SUFFIX}, logs: $LOG_DIR"

xargs -a "$WORK" -d '\n' -P "$JOBS" -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r src dst filter <<<"{}"
  transcode_one "$src" "$dst" "$filter" "$CRF" "$THREADS"
'

NFAIL=$(wc -l <"$LOG_DIR/failed.txt")
echo "$(date '+%F %T') done. $NFAIL failed (see $LOG_DIR/failed.txt)"
if (( NFAIL == 0 )); then
  echo
  echo "Next -- fix meta/info.json, or LeRobot still reports the source resolution:"
  for ds in $DATASETS; do
    echo "  python scripts/patch_transcoded_meta.py --mode $MODE ${SRC_ROOT}/${ds}${DST_SUFFIX}"
  done
fi
