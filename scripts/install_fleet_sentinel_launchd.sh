#!/usr/bin/env bash
# Install (or dry-run render) the launchd unit for the fleet sentinel.
#
# Steering Leverage Operating Plan v2, Pillar 6 / Phase 0.1.
# Runs scripts/fleet_sentinel.py from the repo root every 600 seconds,
# logging to .aragora/overnight/fleet-sentinel.log.
#
# The render bakes a full PATH into EnvironmentVariables: launchd jobs get a
# minimal environment, and on 2026-06-10 the live sentinel's gh_auth check
# died with FileNotFoundError('gh') (exit 2, blind) because /opt/homebrew/bin
# was not on PATH.  The live plist was hand-patched that day; reinstalls must
# not regress it.
#
# A default macOS notification channel is wired through the sentinel's
# --notify-cmd template ({summary} placeholder).  Override or disable it:
#
# Usage:
#   ./scripts/install_fleet_sentinel_launchd.sh --dry-run                # print plist only
#   ./scripts/install_fleet_sentinel_launchd.sh --install                # write + launchctl load
#   ./scripts/install_fleet_sentinel_launchd.sh --dry-run --notify-cmd ''        # no notifications
#   ./scripts/install_fleet_sentinel_launchd.sh --install --notify-cmd 'my-notifier {summary}'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${ARAGORA_PYTHON:-${PYTHON_BIN:-$(command -v python3)}}"
INTERVAL="${INTERVAL:-600}"
LABEL="com.aragora.fleet-sentinel"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/fleet-sentinel.log"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LAUNCHD_PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# Default notify channel: macOS notification via osascript.  Conforms to
# fleet_sentinel.py's --notify-cmd contract: {summary} embedded in a larger
# token is substituted (quote/backslash-sanitized) at breach time.
DEFAULT_NOTIFY_CMD='osascript -e "display notification \"{summary}\" with title \"Aragora Fleet Sentinel\""'

xml_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

render_plist() {
  local notify_block=""
  if [ -n "${NOTIFY_CMD}" ]; then
    notify_block="$(printf '        <string>--notify-cmd</string>\n        <string>%s</string>' "$(xml_escape "${NOTIFY_CMD}")")"
  fi
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
${notify_block}
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${LAUNCHD_PATH}</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
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
  echo "usage: $0 (--dry-run | --install) [--notify-cmd CMD]   (--notify-cmd '' disables notifications)" >&2
  exit 64
}

MODE=""
NOTIFY_CMD="${DEFAULT_NOTIFY_CMD}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--install)
      [ -z "${MODE}" ] || usage
      MODE="$1"
      ;;
    --notify-cmd)
      [ $# -ge 2 ] || usage
      NOTIFY_CMD="$2"
      shift
      ;;
    *)
      usage
      ;;
  esac
  shift
done
[ -n "${MODE}" ] || usage

case "${MODE}" in
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
    if [ -n "${NOTIFY_CMD}" ]; then
      echo "notify: ${NOTIFY_CMD}"
    else
      echo "notify: disabled"
    fi
    ;;
esac
