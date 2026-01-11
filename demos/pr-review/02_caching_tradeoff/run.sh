#!/bin/bash
# Demo 2: Caching Tradeoff
#
# This demo shows how Aragora surfaces split opinions when agents
# disagree about architectural tradeoffs.
#
# Expected output: Agents have different opinions on caching approach

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFF_FILE="$SCRIPT_DIR/diff.patch"

echo "================================================"
echo "Demo 2: Caching Tradeoff Discussion"
echo "================================================"
echo ""
echo "This diff adds in-memory caching with tradeoffs."
echo "Expected: Agents will have split opinions on the approach."
echo ""
echo "Running aragora review..."
echo ""

cat "$DIFF_FILE" | aragora review --focus performance,quality

echo ""
echo "================================================"
echo "Demo complete!"
echo ""
echo "Note: Split opinions help you make informed decisions"
echo "Compare output with expected_comment.md"
