#!/bin/bash
# Pre-flight check: verify deploy files and client dependencies.

echo "=========================================="
echo "FastWAM TA2 WebSocket deploy check"
echo "=========================================="
echo ""

# Deploy dir = this script's directory, so it travels between machines
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -d "$DEPLOY_DIR" ]; then
    echo "[x] deploy dir not found: $DEPLOY_DIR"
    exit 1
fi

cd "$DEPLOY_DIR" || exit 1
echo "[ok] deploy dir: $DEPLOY_DIR"
echo ""

# Files
echo "Checking files..."
FILES=(
    "server/serve_policy_ws.py"
    "server/fastwam_policy_wrapper.py"
    "server/start_server.sh"
    "client/run_client_ws.py"
    "client/fastwam_env.py"
    "client/run_task_ws.sh"
    "server/serve_policy_ws.py"
    "../../docs/GUIDE.md"
)

MISSING=0
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  [x] missing: $file"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "[x] $MISSING file(s) missing"
    exit 1
fi

echo ""
echo "[ok] all files present"
echo ""

# Client Python dependencies
echo "Checking Python dependencies..."
DEPS=("websockets" "msgpack")
MISSING_DEPS=0

for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "  ✓ $dep"
    else
        echo "  [x] not installed: $dep"
        MISSING_DEPS=$((MISSING_DEPS + 1))
    fi
done

if [ $MISSING_DEPS -gt 0 ]; then
    echo ""
    echo "[!] $MISSING_DEPS dependency/dependencies missing"
    echo "    pip install websockets msgpack-numpy typing-extensions dm-tree"
else
    echo ""
    echo "[ok] all dependencies installed"
fi

echo ""
echo "=========================================="
echo "Check complete"
echo "=========================================="
echo ""
echo "Next:"
echo "1. install anything reported missing above"
echo "2. start the server (GPU host): TASK=<task> bash start_local_serve_ws.sh"
echo "3. test the link: python3 test_websocket_connection.py --host <gpu-host>"
echo "4. start the client (robot host): cd client && ./run_task_ws.sh <taskmap-key> --dry-run"
echo ""
echo "Reference:"
echo "  - docs/GUIDE.md: training and deployment reference"
echo ""
