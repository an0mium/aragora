#!/bin/bash
# Demo 4: Error Handling Issues
#
# This demo shows how Aragora catches missing error handling and
# potential runtime errors that could cause production issues.
#
# Expected output: Unanimous agreement on NoneType and session errors

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFF_FILE="$SCRIPT_DIR/diff.patch"

echo "================================================"
echo "Demo 4: Error Handling Issues"
echo "================================================"
echo ""
echo "This diff removes error handling and introduces bugs."
echo "Expected: Agents should unanimously flag critical issues."
echo ""
echo "Running aragora review..."
echo ""

cat "$DIFF_FILE" | aragora review --focus quality

echo ""
echo "================================================"
echo "Demo complete!"
echo ""
echo "Compare output with expected_comment.md"
