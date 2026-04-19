#!/usr/bin/env bash
# Show status of the codex automation publisher launchd job.

set -euo pipefail

LABEL="com.aragora.codex-automation-publisher"
LAUNCHD_DOMAIN="gui/$(id -u)"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ -f "${PLIST_PATH}" ]]; then
    echo "plist: present (${PLIST_PATH})"
else
    echo "plist: missing (${PLIST_PATH})"
fi

echo "--- launchctl ---"
launchctl print "${LAUNCHD_DOMAIN}/${LABEL}" 2>/dev/null | sed -n '1,20p' || echo "job not loaded"
