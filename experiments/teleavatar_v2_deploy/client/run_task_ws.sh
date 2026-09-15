#!/bin/bash
# Run the FastWAM WebSocket client, conditioned on one task-library key.
#
#   ./run_task_ws.sh <taskmap-key>              # e.g. ./run_task_ws.sh my_task
#   ./run_task_ws.sh <taskmap-key> --dry-run    # infer only, no motion
#   ./run_task_ws.sh                            # omit the key: use the server's
#                                               # startup instruction
#
# The key must exist in the server's task library (the taskmap.json passed to
# serve_policy_ws.py via --task-map). Generate one with:
#   python experiments/teleavatar_v2_deploy/server/make_task_map.py \
#     --dataset-dir data/<your_dataset>_mono \
#     --cache-dir data/text_embeds_cache/<your_task>
#
# The server is started once (start_local_serve_ws.sh) and serves every key in its
# library; this script only selects which instruction the policy is conditioned on.
#
# Environment variables
#   PROJECT_ROOT   FastWAM repo root. Default: three levels up from this script.
#   CLIENT_DIR     Directory holding run_client_ws.py. Default: this script's dir.
#   BAG_ROOT       Where rosbag2 writes. Default: $HOME/fastwam_bags.
#   PYTHON_BIN     Interpreter for the client. Default: /usr/bin/python3 -- see below.
#   ROS_SETUP      ROS 2 setup.bash. Default: /opt/ros/humble/setup.bash.
#   ROS_DOMAIN_ID  Default: 19.
#   SERVER_HOST / SERVER_PORT   Default: 127.0.0.1 / 8000.
#
# Why /usr/bin/python3 and not conda: ROS 2 Humble's C extensions are built for the
# system interpreter (3.10). Conda's python and the uv-managed one both lack rclpy/gi.
# rclpy comes from setup.bash's PYTHONPATH, not from dist-packages, so sourcing it
# below is required, not optional.
#
# Parameter mapping from the removed HTTP client:
#   --control-hz 20            -> same
#   --publish-hz 200           -> same
#   --replan-steps 20          -> --open-loop-horizon 20 (same meaning: policy steps
#                                 executed open-loop before re-querying)
#   --action-horizon 32        -> SERVER-side, set in start_local_serve_ws.sh
#   --num-inference-steps 10   -> SERVER-side, `./start_local_serve_ws.sh 10`
#   --prefetch-remaining 0     -> no equivalent: this client infers synchronously in
#                                 the control loop, so there is no background prefetch.
#                                 The executed trajectory matches; what differs is that
#                                 here the loop *blocks* for inference every
#                                 --open-loop-horizon steps instead of overlapping it.
#
# No `set -u`: ROS 2's setup.bash reads AMENT_TRACE_SETUP_FILES and friends while they
# are still unset, so nounset aborts the sourcing before ROS is on the path.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="${CLIENT_DIR:-$SCRIPT_DIR}"
BAG_ROOT="${BAG_ROOT:-$HOME/fastwam_bags}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"

# A leading `--flag` means no task key was given -- pass everything through and let the
# server fall back to its startup instruction.
TASK_ARGS=()
if [[ $# -gt 0 && "$1" != -* ]]; then
    TASK_ARGS=(--task "$1")
    shift
fi

cd "$CLIENT_DIR"

[[ -f "$ROS_SETUP" ]] || { echo "ROS setup not found: $ROS_SETUP (set ROS_SETUP=)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-19}"

# Record decoded RTP frames (republished by ros2_interface), the policy's action chunks,
# and robot state for the lifetime of this run. rosbag2 writes a directory under BAG_ROOT.
# The /fastwam/policy/* pair is what analyze_deploy_bag.py needs for its
# predicted-vs-commanded-vs-measured comparison; without them the bag only supports
# commanded-vs-measured.
mkdir -p "$BAG_ROOT"
BAG_NAME="fastwam_ta2_$(date +%Y%m%d_%H%M%S)"
echo "Recording rosbag: $BAG_ROOT/$BAG_NAME"
ros2 bag record -o "$BAG_ROOT/$BAG_NAME" \
  /fastwam/observation/head_camera/compressed \
  /fastwam/observation/left_color/compressed \
  /fastwam/observation/right_color/compressed \
  /fastwam/policy/action_chunk /fastwam/policy/inference_ms \
  /left_arm/joint_states /right_arm/joint_states \
  /left_gripper/joint_states /right_gripper/joint_states \
  /left_arm/current_ee_pose /right_arm/current_ee_pose \
  /api/left_arm/joint_cmd /api/right_arm/joint_cmd \
  /api/left_gripper/cmd /api/right_gripper/cmd /api/fsm/enable \
  >"$BAG_ROOT/${BAG_NAME}.log" 2>&1 &
BAG_PID=$!
trap 'kill -INT "$BAG_PID" 2>/dev/null || true; wait "$BAG_PID" 2>/dev/null || true; echo "Rosbag saved: $BAG_ROOT/$BAG_NAME"' EXIT
sleep 1

# openpi_client / websockets / msgpack are expected in this interpreter's user site
# (pip install --user), since conda's python cannot import rclpy.
exec "$PYTHON_BIN" -u run_client_ws.py \
  --server-host "$SERVER_HOST" \
  --server-port "$SERVER_PORT" \
  "${TASK_ARGS[@]}" \
  --decoder nvh265dec \
  --control-hz 20 \
  --publish-hz 200 \
  --open-loop-horizon 20 \
  "$@"
