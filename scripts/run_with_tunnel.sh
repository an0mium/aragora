#!/bin/bash
# Run nomic loop with Cloudflare Tunnel for public access
#
# This exposes your local WebSocket server to the internet via Cloudflare Tunnel.
# live.aragora.ai connects to wss://api.aragora.ai which routes through this tunnel.
#
# Prerequisites:
#   - cloudflared installed (brew install cloudflared)
#   - Cloudflare account authenticated (cloudflared tunnel login)
#   - Tunnel "aragora-live" created and DNS routed to api.aragora.ai
#
# Usage:
#   ./scripts/run_with_tunnel.sh [cycles]

set -e

CYCLES=${1:-3}
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TUNNEL_NAME="aragora-live"
TOKEN_FILE="$HOME/.cloudflared/aragora-live.token"

echo "========================================"
echo "ARAGORA NOMIC LOOP WITH PERSISTENT TUNNEL"
echo "========================================"

# Check for cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "ERROR: cloudflared not found. Install with: brew install cloudflared"
    exit 1
fi

# Check if tunnel token exists
if [ ! -f "$TOKEN_FILE" ]; then
    echo "Getting tunnel token..."
    cloudflared tunnel token "$TUNNEL_NAME" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi

TOKEN=$(cat "$TOKEN_FILE")

# Check if tunnel is already running
EXISTING_PID=$(pgrep -f "cloudflared.*$TUNNEL_NAME" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Tunnel already running (PID: $EXISTING_PID)"
    TUNNEL_PID=""
else
    echo "Starting Cloudflare Tunnel (aragora-live -> api.aragora.ai)..."
    # Run with config file for proper ingress routing
    cloudflared tunnel --config "$HOME/.cloudflared/config.yml" run "$TUNNEL_NAME" &
    TUNNEL_PID=$!

    # Give tunnel time to establish
    sleep 3
    echo "Tunnel started (PID: $TUNNEL_PID)"
fi

echo ""
echo "Tunnel: $TUNNEL_NAME"
echo "Public URL: wss://api.aragora.ai"
echo "Dashboard: https://live.aragora.ai"
echo ""
echo "========================================"
echo ""

# Run nomic loop
cd "$PROJECT_DIR"
python scripts/run_nomic_with_stream.py run --cycles "$CYCLES"

# Cleanup (only if we started the tunnel)
if [ -n "$TUNNEL_PID" ]; then
    echo "Stopping tunnel..."
    kill $TUNNEL_PID 2>/dev/null || true
fi
