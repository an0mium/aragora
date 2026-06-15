#!/usr/bin/env bash
# Install a launchd job that runs the docs-site sync drift detector once a day.
#
# The detector (scripts/docs_sync_drift_detector.py) is a bounded single-shot
# governed loop iteration: launchd provides the cadence, the script provides
# the iteration. No KeepAlive - one fire per StartCalendarInterval.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.aragora.docs-sync-drift"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/docs-sync-drift-launchd.log"
HOUR="${DOCS_DRIFT_HOUR:-7}"
MINUTE="${DOCS_DRIFT_MINUTE:-45}"
CHECK_ONLY=false

usage() {
    cat <<'EOF'
Usage: ./scripts/install_docs_drift_launchd.sh [options]

Options:
  --hour <0-23>       Daily fire hour (default: 7)
  --minute <0-59>     Daily fire minute (default: 45)
  --log-path <file>   Log file path (default: .aragora/overnight/docs-sync-drift-launchd.log)
  --check-only        Install in report-only mode (omit --apply; no PRs opened)
  --help              Show this help
EOF
}

validate_integer() {
    local label="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "${label} must be numeric" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hour)
            HOUR="${2:-$HOUR}"
            shift 2
            ;;
        --minute)
            MINUTE="${2:-$MINUTE}"
            shift 2
            ;;
        --log-path)
            LOG_PATH="${2:-$LOG_PATH}"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

validate_integer "hour" "${HOUR}"
validate_integer "minute" "${MINUTE}"
if (( HOUR > 23 )) || (( MINUTE > 59 )); then
    echo "hour must be 0-23 and minute 0-59" >&2
    exit 2
fi

mkdir -p "$(dirname "${PLIST_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"

apply_flag="--apply"
if [[ "${CHECK_ONLY}" == true ]]; then
    apply_flag=""
fi
command_string="cd \"${REPO_ROOT}\" && export PATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$HOME/.pyenv/shims:\$PATH\" && exec python3 scripts/docs_sync_drift_detector.py ${apply_flag} --json"
command_xml="${command_string//&/&amp;}"

cat >"${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>${command_xml}</string>
  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${HOUR}</integer>
    <key>Minute</key>
    <integer>${MINUTE}</integer>
  </dict>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>StandardOutPath</key>
  <string>${LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_PATH}</string>
</dict>
</plist>
EOF

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl load "${PLIST_PATH}"

echo "Installed launchd job: ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Log: ${LOG_PATH}"
echo "Fires daily at $(printf '%02d:%02d' "${HOUR}" "${MINUTE}")"
if [[ "${CHECK_ONLY}" == true ]]; then
    echo "Mode: check-only (no PRs opened)"
else
    echo "Mode: apply (opens at most one sync PR; never merges)"
fi
