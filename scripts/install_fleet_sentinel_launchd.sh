#!/usr/bin/env bash
# Install (or dry-run render) the launchd unit for the fleet sentinel.
#
# Steering Leverage Operating Plan v2, Pillar 6 / Phase 0.1.
# Runs scripts/fleet_sentinel.py from the repo root every 600 seconds,
# logging to .aragora/overnight/fleet-sentinel.log.
#
# Usage:
#   ./scripts/install_fleet_sentinel_launchd.sh --dry-run    # print plist only
#   ./scripts/install_fleet_sentinel_launchd.sh --install    # write + launchctl load
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${ARAGORA_PYTHON:-${PYTHON_BIN:-$(command -v python3)}}"
INTERVAL="${INTERVAL:-600}"
LABEL="com.aragora.fleet-sentinel"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/fleet-sentinel.log"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
# launchd agents get a minimal PATH without /opt/homebrew/bin; the sentinel's
# gh_auth check needs `gh` on PATH or it reports status unknown (exit 2).
SENTINEL_PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

render_plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${REPO_ROOT}/scripts/fleet_sentinel.py</string>
        <string>--json</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${SENTINEL_PATH}</string>
    </dict>
    <key>StartInterval</key>
    <integer>${INTERVAL}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_PATH}</string>
</dict>
</plist>
EOF
}

usage() {
  echo "usage: $0 --dry-run | --install" >&2
  exit 64
}

[ $# -eq 1 ] || usage

case "$1" in
  --dry-run)
    render_plist
    ;;
  --install)
    mkdir -p "$(dirname "${PLIST_PATH}")" "$(dirname "${LOG_PATH}")"
    render_plist > "${PLIST_PATH}"
    plutil -lint "${PLIST_PATH}"
    launchctl unload "${PLIST_PATH}" 2>/dev/null || true
    launchctl load "${PLIST_PATH}"
    echo "installed ${LABEL} (every ${INTERVAL}s) -> ${PLIST_PATH}"
    echo "logs: ${LOG_PATH}"
    ;;
  *)
    usage
    ;;
esac
