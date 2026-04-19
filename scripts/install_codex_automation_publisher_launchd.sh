#!/usr/bin/env bash
# Install a launchd job that runs the codex automation publisher bridge on an interval.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.aragora.codex-automation-publisher"
INTERVAL_SECONDS=300
LOG_PATH="${REPO_ROOT}/.aragora/overnight/codex-automation-publisher.log"

usage() {
    cat <<'EOF'
Usage: ./scripts/install_codex_automation_publisher_launchd.sh [options]

Options:
  --interval-seconds <n>        launchd StartInterval (default: 300)
  --log-path <file>             Log file path (default: .aragora/overnight/codex-automation-publisher.log)
  --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval-seconds)
            INTERVAL_SECONDS="${2:-300}"
            shift 2
            ;;
        --log-path)
            LOG_PATH="${2:-}"
            shift 2
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

if ! [[ "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "interval must be numeric" >&2
    exit 2
fi

PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
mkdir -p "$(dirname "${PLIST_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"

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
    <string>cd "${REPO_ROOT}" &amp;&amp; ./scripts/run_codex_automation_publisher.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>${INTERVAL_SECONDS}</integer>
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
echo "Interval: ${INTERVAL_SECONDS}s"
echo "Log: ${LOG_PATH}"
