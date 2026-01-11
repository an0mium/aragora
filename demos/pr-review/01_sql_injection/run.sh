#!/bin/bash
# Demo 1: SQL Injection Detection
#
# This demo shows how Aragora's multi-agent review catches SQL injection
# vulnerabilities that both agents unanimously agree on.
#
# Expected output: Both agents flag the SQL injection as CRITICAL

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFF_FILE="$SCRIPT_DIR/diff.patch"

echo "================================================"
echo "Demo 1: SQL Injection Detection"
echo "================================================"
echo ""
echo "This diff introduces two SQL injection vulnerabilities."
echo "Expected: All agents should unanimously flag these as CRITICAL."
echo ""
echo "Running aragora review..."
echo ""

cat "$DIFF_FILE" | aragora review --focus security

echo ""
echo "================================================"
echo "Demo complete!"
echo ""
echo "Compare output with expected_comment.md"
