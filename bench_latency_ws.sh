#!/bin/bash
# Inference latency benchmark over the WebSocket path.
#
# Requires a running server (bash start_local_serve_ws.sh). Sends synthetic frames and
# never touches ROS 2, so no arm moves.
#
#   bash bench_latency_ws.sh                    # one point: server default task, 10 steps
#   bash bench_latency_ws.sh <taskmap-key> 12   # one point: given task and step count
#   bash bench_latency_ws.sh --sweep            # SWEEP_TASKS x SWEEP_STEPS
#
# Variables: REPEATS(30) WARMUP(3) HOST(127.0.0.1) PORT(8000)
#            SWEEP_TASKS(server's available_tasks) SWEEP_STEPS("8 10 12")
#            OUTPUT_DIR(logs/latency/ws_<timestamp>)
#
# Step count is per request -- no server restart needed.
# The server honours num_inference_steps from the payload, and bench_infer_ws.py
# verifies the server actually used the requested value.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$PROJECT_DIR/experiments/teleavatar_v2_deploy/client/bench_infer_ws.py"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REPEATS="${REPEATS:-30}"
WARMUP="${WARMUP:-3}"

[[ -f "$BENCHMARK" ]] || { echo "not found: $BENCHMARK" >&2; exit 1; }

# Fail immediately if the server is down, rather than after 30 connection timeouts
if ! curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/healthz" >/dev/null; then
    echo "ws://${HOST}:${PORT} not ready -- start it with bash start_local_serve_ws.sh" >&2
    exit 1
fi

run_one() {   # task steps csv   -- empty task means no --task (server default)
    "$PYTHON_BIN" -u "$BENCHMARK" \
        --server-host "$HOST" --server-port "$PORT" \
        ${1:+--task "$1"} --num-inference-steps "$2" \
        --warmup "$WARMUP" --repeats "$REPEATS" \
        ${3:+--csv "$3"}
}

if [[ "${1:-}" != "--sweep" ]]; then
    # ---- single point ----
    run_one "${1:-}" "${2:-10}" ""
    exit 0
fi

# ---- sweep ----
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/logs/latency/ws_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
COMBINED="$OUTPUT_DIR/all_samples.csv"

echo "task,num_inference_steps,rtt_ms,pack_ms,send_ms,recv_wait_ms,unpack_ms,server_infer_ms,overhead_ms" \
  > "$COMBINED"

# Which tasks to sweep: ask the server for available_tasks (it ships them in the
# connection metadata), falling back to a single run against its default instruction.
if [[ -z "${SWEEP_TASKS:-}" ]]; then
    SWEEP_TASKS="$(
        "$PYTHON_BIN" - "$HOST" "$PORT" <<'PYEOF' 2>/dev/null || true
import sys
try:
    import websockets.sync.client
    from openpi_client import msgpack_numpy
    with websockets.sync.client.connect(
        f"ws://{sys.argv[1]}:{sys.argv[2]}", compression=None, max_size=None
    ) as ws:
        print(" ".join(msgpack_numpy.unpackb(ws.recv()).get("available_tasks") or []))
except Exception:
    pass
PYEOF
    )"
    if [[ -n "$SWEEP_TASKS" ]]; then
        echo "SWEEP_TASKS unset, using the server's available_tasks: $SWEEP_TASKS"
    else
        # An empty string is a meaningful value here: run_one omits --task for it.
        SWEEP_TASKS=""
        echo "server reported no available_tasks; sweeping one group (server default)."
    fi
fi

for STEPS in ${SWEEP_STEPS:-8 10 12}; do
    # :-__default__ keeps the loop running one pass when SWEEP_TASKS is empty
    for TASK_KEY in ${SWEEP_TASKS:-__default__}; do
        if [[ "$TASK_KEY" == "__default__" ]]; then
            TASK_ARG=""; NAME="default_steps${STEPS}"
        else
            TASK_ARG="$TASK_KEY"; NAME="${TASK_KEY}_steps${STEPS}"
        fi
        echo "[$NAME] repeats=$REPEATS warmup=$WARMUP"
        run_one "$TASK_ARG" "$STEPS" "$OUTPUT_DIR/${NAME}.csv" | tee "$OUTPUT_DIR/${NAME}.log"
        tail -n +2 "$OUTPUT_DIR/${NAME}.csv" >> "$COMBINED"
    done
done

echo "Done:"
echo "  per-group: $OUTPUT_DIR/<task>_steps<N>.csv"
echo "  all samples: $COMBINED"
