#!/usr/bin/env bash
# Record FastWAM TA2 deploy bag on the local ROS2 client host.
#
# Standalone alternative to the recording run_task_ws.sh already starts for you --
# use this only when driving the client some other way. Start run_client_ws.py (or
# run_task_ws.sh --dry-run) in another terminal; it publishes the /fastwam/* topics.
#
# pred_video/* and episode/event are listed for compatibility with older HTTP-era bags;
# the WebSocket client does not emit them, so they stay empty.
#
# Usage: bash record_fastwam_bag.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-$HOME/FastWAM_TA2/lusiwei/fastwam_bags}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

: "${ROS_DOMAIN_ID:=29}"
BAG_NAME="fastwam_ta2_$(date +%Y%m%d_%H%M%S)"

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Writing ${OUT_DIR}/${BAG_NAME}"
echo "Start run_client_ws.py in another terminal (or use run_task_ws.sh, which records itself)."

ros2 bag record -o "${BAG_NAME}" \
  /fastwam/observation/head_camera/compressed \
  /fastwam/observation/left_color/compressed \
  /fastwam/observation/right_color/compressed \
  /fastwam/policy/action_chunk \
  /fastwam/policy/inference_ms \
  /fastwam/policy/pred_video/compressed \
  /fastwam/policy/pred_video/meta \
  /fastwam/episode/event \
  /left_arm/joint_states \
  /right_arm/joint_states \
  /left_gripper/joint_states \
  /right_gripper/joint_states \
  /left_arm/current_ee_pose \
  /right_arm/current_ee_pose \
  /api/left_arm/joint_cmd \
  /api/right_arm/joint_cmd \
  /api/left_gripper/cmd \
  /api/right_gripper/cmd \
  /api/fsm/enable
