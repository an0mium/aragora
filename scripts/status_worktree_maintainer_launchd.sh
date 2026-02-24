#!/usr/bin/env bash
# Show status of the codex worktree maintainer launchd job.

set -euo pipefail

LABEL="com.aragora.codex-worktree-maintainer"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ -f "${PLIST_PATH}" ]]; then
    echo "plist: present (${PLIST_PATH})"
else
    echo "plist: missing (${PLIST_PATH})"
fi

echo "--- launchctl ---"
launchctl list | grep "${LABEL}" || echo "job not loaded"
