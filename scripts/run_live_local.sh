#!/bin/bash
# Run the live dashboard locally, connected to local nomic loop
#
# This allows you to watch the nomic loop at http://localhost:3000
# while it runs locally (instead of needing api.aragora.ai)
#
# Usage:
#   1. In one terminal: python scripts/run_nomic_with_stream.py run --cycles 3
#   2. In another terminal: ./scripts/run_live_local.sh

set -e

cd "$(dirname "$0")/../aragora/live"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Check if build exists
if [ ! -d "out" ] && [ ! -d ".next" ]; then
    echo "Building dashboard..."
    npm run build
fi

echo "========================================"
echo "ARAGORA LIVE DASHBOARD (LOCAL)"
echo "========================================"
echo "Dashboard: http://localhost:3000"
echo "WebSocket: ws://localhost:8765/ws"
echo ""
echo "Make sure nomic loop is running:"
echo "  python scripts/run_nomic_with_stream.py run --cycles 3"
echo "========================================"
echo ""

# Run dev server with local WebSocket URL
NEXT_PUBLIC_WS_URL=ws://localhost:8765/ws \
NEXT_PUBLIC_API_URL=http://localhost:8080 \
npm run dev
