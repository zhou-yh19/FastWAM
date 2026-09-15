#!/bin/bash
# Thin wrapper over start_local_serve_ws.sh at the repo root.
#
# Kept only because check_deployment.sh and older notes point here. All the real logic
# -- argument parsing, run/step discovery, path validation -- lives in the root script,
# so the two cannot drift apart.
#
#   TASK=<config under configs/task/> bash start_server.sh [num_inference_steps]
#
# See start_local_serve_ws.sh for every supported variable.

set -euo pipefail

# server/ -> teleavatar_v2_deploy/ -> experiments/ -> repo root
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ROOT_SCRIPT="$PROJECT_ROOT/start_local_serve_ws.sh"

[[ -f "$ROOT_SCRIPT" ]] || {
  echo "not found: $ROOT_SCRIPT -- is PROJECT_ROOT right? (got: $PROJECT_ROOT)" >&2
  exit 1
}

# This entry point has always defaulted to 8 steps; the root script defaults to 10.
# Pass it explicitly so behaviour is unchanged.
exec env PROJECT_ROOT="$PROJECT_ROOT" bash "$ROOT_SCRIPT" "${1:-8}"
