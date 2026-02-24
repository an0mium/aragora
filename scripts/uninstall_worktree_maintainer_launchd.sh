#!/usr/bin/env bash
# Uninstall the codex worktree maintainer launchd job.

set -euo pipefail

LABEL="com.aragora.codex-worktree-maintainer"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
rm -f "${PLIST_PATH}"

echo "Removed launchd job: ${LABEL}"
echo "Plist removed: ${PLIST_PATH}"
