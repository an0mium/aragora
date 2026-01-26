#!/bin/bash
# WebSocket connectivity verification script
# Tests WebSocket handshake against aragora.ai

set -e

WS_URL="${1:-wss://api.aragora.ai/ws}"
HTTP_URL="${WS_URL/wss:/https:}"
HTTP_URL="${HTTP_URL/ws:/http:}"

echo "=== WebSocket Verification ==="
echo "Target: $WS_URL"
echo ""

# Generate random WebSocket key
WS_KEY=$(openssl rand -base64 16)

echo "1. Testing HTTP/1.1 upgrade handshake..."
RESPONSE=$(curl --http1.1 -i -s -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: $WS_KEY" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Origin: https://aragora.ai" \
  --max-time 5 \
  "$HTTP_URL" 2>&1 || true)

echo "$RESPONSE" | head -20
echo ""

# Check for 101 Switching Protocols
if echo "$RESPONSE" | grep -q "101"; then
  echo "SUCCESS: WebSocket upgrade accepted (101 Switching Protocols)"
  exit 0
elif echo "$RESPONSE" | grep -q "426"; then
  echo "FAIL: HTTP 426 - Upgrade Required (proxy not forwarding Upgrade header)"
  echo ""
  echo "Fix: Enable WebSocket support in Cloudflare/nginx/LB"
  exit 1
elif echo "$RESPONSE" | grep -q "403"; then
  echo "FAIL: HTTP 403 - Forbidden (check CORS/origin settings)"
  exit 1
elif echo "$RESPONSE" | grep -q "401"; then
  echo "FAIL: HTTP 401 - Unauthorized (check auth requirements)"
  exit 1
else
  echo "UNKNOWN: Check response above"
  exit 1
fi
