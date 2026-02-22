#!/usr/bin/env bash
# Aragora Demo Launcher
# One-command local launch: starts backend + frontend, opens browser.
#
# Usage:
#   ./scripts/demo.sh              # Start both backend and frontend
#   ./scripts/demo.sh --backend    # Backend only (API on :8080)
#   ./scripts/demo.sh --frontend   # Frontend only (assumes backend running)
#   ./scripts/demo.sh --stop       # Stop any running demo processes
#
# Environment:
#   ARAGORA_API_PORT    Backend API port (default: 8080)
#   ARAGORA_WS_PORT     WebSocket port  (default: 8765)
#   FRONTEND_PORT       Frontend port   (default: 3000)
#   NO_BROWSER          Skip opening browser if set

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT_DIR/.demo"
API_PORT="${ARAGORA_API_PORT:-8080}"
WS_PORT="${ARAGORA_WS_PORT:-8765}"
FE_PORT="${FRONTEND_PORT:-3000}"
BACKEND_ONLY=false
FRONTEND_ONLY=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { printf "\033[1;36m[demo]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[demo]\033[0m %s\n" "$*" >&2; }
ok()  { printf "\033[1;32m[demo]\033[0m %s\n" "$*"; }

cleanup() {
    log "Shutting down..."
    if [[ -f "$PID_DIR/backend.pid" ]]; then
        kill "$(cat "$PID_DIR/backend.pid")" 2>/dev/null || true
        rm -f "$PID_DIR/backend.pid"
    fi
    if [[ -f "$PID_DIR/frontend.pid" ]]; then
        kill "$(cat "$PID_DIR/frontend.pid")" 2>/dev/null || true
        rm -f "$PID_DIR/frontend.pid"
    fi
    ok "Demo stopped."
}

stop_demo() {
    if [[ -d "$PID_DIR" ]]; then
        cleanup
        rmdir "$PID_DIR" 2>/dev/null || true
    else
        log "No running demo found."
    fi
    exit 0
}

wait_for_url() {
    local url=$1 label=$2 timeout=${3:-30}
    local elapsed=0
    while ! curl -sf "$url" >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $timeout ]]; then
            err "$label did not start within ${timeout}s"
            return 1
        fi
    done
    ok "$label ready at $url"
}

open_browser() {
    local url=$1
    if [[ -n "${NO_BROWSER:-}" ]]; then return; fi
    if command -v open >/dev/null 2>&1; then
        open "$url"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url"
    fi
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prereqs() {
    local ok=true

    if ! command -v python3 >/dev/null 2>&1; then
        err "python3 not found. Install Python 3.11+."
        ok=false
    fi

    if ! $FRONTEND_ONLY; then
        if ! python3 -c "import aragora" 2>/dev/null; then
            err "aragora package not installed. Run: pip install -e ."
            ok=false
        fi
    fi

    if ! $BACKEND_ONLY; then
        if ! command -v node >/dev/null 2>&1; then
            err "node not found. Install Node.js 18+."
            ok=false
        fi
        if ! command -v npm >/dev/null 2>&1; then
            err "npm not found. Install Node.js 18+."
            ok=false
        fi
    fi

    if ! $ok; then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------

start_backend() {
    # Kill stale processes on our ports (API + WS range)
    for p in $API_PORT $WS_PORT $((WS_PORT+1)) $((WS_PORT+2)) $((WS_PORT+3)); do
        lsof -ti :"$p" 2>/dev/null | xargs kill 2>/dev/null || true
    done
    sleep 1

    log "Starting backend (API :$API_PORT, WS :$WS_PORT)..."

    ARAGORA_OFFLINE=true \
    ARAGORA_DEMO_MODE=true \
    ARAGORA_DB_BACKEND=sqlite \
    ARAGORA_ALLOWED_ORIGINS="*" \
    python3 -m aragora.server \
        --offline \
        --api-port "$API_PORT" \
        --ws-port "$WS_PORT" \
        > "$PID_DIR/backend.log" 2>&1 &

    echo $! > "$PID_DIR/backend.pid"
    wait_for_url "http://localhost:$API_PORT/healthz" "Backend API" 90
}

start_frontend() {
    local live_dir="$ROOT_DIR/aragora/live"

    if [[ ! -d "$live_dir/node_modules" ]]; then
        log "Installing frontend dependencies..."
        (cd "$live_dir" && npm install --silent)
    fi

    log "Starting frontend (:$FE_PORT)..."

    NEXT_PUBLIC_API_URL="http://localhost:$API_PORT" \
    NEXT_PUBLIC_WS_URL="ws://localhost:$WS_PORT/ws" \
    PORT="$FE_PORT" \
    npm --prefix "$live_dir" run dev \
        > "$PID_DIR/frontend.log" 2>&1 &

    echo $! > "$PID_DIR/frontend.pid"
    wait_for_url "http://localhost:$FE_PORT" "Frontend" 60
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

for arg in "$@"; do
    case "$arg" in
        --stop)      stop_demo ;;
        --backend)   BACKEND_ONLY=true ;;
        --frontend)  FRONTEND_ONLY=true ;;
        --help|-h)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *)
            err "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

check_prereqs

# Set up PID directory and trap
mkdir -p "$PID_DIR"
trap cleanup EXIT INT TERM

if ! $FRONTEND_ONLY; then
    start_backend
fi

if ! $BACKEND_ONLY; then
    start_frontend
fi

echo ""
ok "============================================"
ok "  Aragora Demo Running"
ok "============================================"
if ! $FRONTEND_ONLY; then
    ok "  API:       http://localhost:$API_PORT"
    ok "  WebSocket: ws://localhost:$WS_PORT/ws"
    ok "  Health:    http://localhost:$API_PORT/healthz"
fi
if ! $BACKEND_ONLY; then
    ok "  Dashboard: http://localhost:$FE_PORT"
fi
ok ""
ok "  Press Ctrl+C to stop"
ok "============================================"
echo ""

if ! $BACKEND_ONLY; then
    open_browser "http://localhost:$FE_PORT"
fi

# Wait for either process to exit
wait
