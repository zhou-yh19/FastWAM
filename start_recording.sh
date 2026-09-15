#!/bin/bash
# Record a FastWAM rosbag.
#
#   [BAG_ROOT=...] [ROS_DOMAIN_ID=...] bash start_recording.sh
#
# Standalone alternative to the recording run_task_ws.sh starts on its own. Use this
# when driving the robot some other way; the client must already be publishing the
# /fastwam/* topics. Ctrl+C to stop.
#
#   BAG_ROOT        where bags are written. Default: $HOME/fastwam_bags
#   ROS_DOMAIN_ID   must match the robot. Default: 29
#   ROS_SETUP       ROS 2 setup.bash. Default: /opt/ros/humble/setup.bash

set -e

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[[ -f "$ROS_SETUP" ]] || { echo "ROS setup not found: $ROS_SETUP (set ROS_SETUP=)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-29}"

BAG_ROOT="${BAG_ROOT:-$HOME/fastwam_bags}"
mkdir -p "$BAG_ROOT"
cd "$BAG_ROOT"

BAG_NAME="fastwam_ta2_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Recording rosbag"
echo "Output:  $BAG_ROOT/$BAG_NAME"
echo "Domain:  $ROS_DOMAIN_ID"
echo "Ctrl+C to stop"
echo "================================================"

# /fastwam/policy/* is what enables predicted-vs-commanded-vs-measured analysis;
# without it a bag only supports commanded-vs-measured.
ros2 bag record -o "${BAG_NAME}" \
  /fastwam/observation/head_camera/compressed \
  /fastwam/observation/left_color/compressed \
  /fastwam/observation/right_color/compressed \
  /fastwam/policy/action_chunk \
  /fastwam/policy/inference_ms \
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
