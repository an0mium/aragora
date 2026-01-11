#!/bin/bash
# Demo 3: Async Conversion Issues
#
# This demo shows how Aragora catches blocking operations in async code,
# a common mistake when converting sync code to async.
#
# Expected output: Both agents flag blocking I/O in async functions

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFF_FILE="$SCRIPT_DIR/diff.patch"

echo "================================================"
echo "Demo 3: Async Conversion Issues"
echo "================================================"
echo ""
echo "This diff converts to async but uses blocking requests."
echo "Expected: Agents should flag blocking I/O as HIGH severity."
echo ""
echo "Running aragora review..."
echo ""

cat "$DIFF_FILE" | aragora review --focus performance

echo ""
echo "================================================"
echo "Demo complete!"
echo ""
echo "Compare output with expected_comment.md"
