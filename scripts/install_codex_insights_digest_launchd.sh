#!/usr/bin/env bash
# Install a launchd job that emits one Codex insights digest per interval.
#
# Adapts the same shape as scripts/install_codex_automation_publisher_launchd.sh
# so operators familiar with the publisher have one mental model for both.
#
# Defaults: every 3600 seconds (1 hour), logs to .aragora/overnight/codex-insights-digest.log.

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

first_worktree_root() {
    git -C "${SCRIPT_REPO_ROOT}" worktree list --porcelain 2>/dev/null \
        | awk 'NR == 1 && $1 == "worktree" { sub(/^worktree /, ""); print; exit }'
}

REPO_ROOT="${ARAGORA_CODEX_INSIGHTS_REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" ]]; then
    CANONICAL_REPO_ROOT="$(first_worktree_root || true)"
    if [[ -n "${CANONICAL_REPO_ROOT}" && -f "${CANONICAL_REPO_ROOT}/scripts/run_codex_insights_digest.sh" ]]; then
        REPO_ROOT="${CANONICAL_REPO_ROOT}"
    else
        REPO_ROOT="${SCRIPT_REPO_ROOT}"
    fi
fi
LABEL="com.aragora.codex-insights-digest"
LAUNCHD_DOMAIN="gui/$(id -u)"
INTERVAL_SECONDS=3600
LOG_PATH="${REPO_ROOT}/.aragora/overnight/codex-insights-digest.log"
SINCE="1h"
INGEST_KM="1"

usage() {
    cat <<'EOF'
Usage: ./scripts/install_codex_insights_digest_launchd.sh [options]

Options:
  --interval-seconds <n>        launchd StartInterval (default: 3600 = hourly)
  --since <duration>            ARAGORA_CODEX_INSIGHTS_SINCE window passed to digest (default: 1h)
  --no-ingest-km                Disable km ingestion of each digest (default: enabled)
  --log-path <file>             Log file path (default: .aragora/overnight/codex-insights-digest.log)
  --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval-seconds)
            INTERVAL_SECONDS="${2:-3600}"
            shift 2
            ;;
        --since)
            SINCE="${2:-1h}"
            shift 2
            ;;
        --no-ingest-km)
            INGEST_KM="0"
            shift
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
    <string>cd "${REPO_ROOT}" &amp;&amp; ARAGORA_CODEX_INSIGHTS_SINCE="${SINCE}" ARAGORA_CODEX_INSIGHTS_INGEST_KM="${INGEST_KM}" ./scripts/run_codex_insights_digest.sh</string>
  </array>
  <key>StartInterval</key>
  <integer>${INTERVAL_SECONDS}</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>StandardOutPath</key>
  <string>${LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_PATH}</string>
</dict>
</plist>
EOF

launchctl bootout "${LAUNCHD_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
launchctl bootstrap "${LAUNCHD_DOMAIN}" "${PLIST_PATH}"

echo "Installed launchd job: ${LABEL}"
echo "Plist:    ${PLIST_PATH}"
echo "Log:      ${LOG_PATH}"
echo "Interval: ${INTERVAL_SECONDS}s"
echo "Since:    ${SINCE}"
echo "KM ingest: ${INGEST_KM}"
echo
echo "Inspect with:  python3 scripts/probe_boss_loop_launchd.py --no-kickstart --label ${LABEL}"
echo "Uninstall:     launchctl bootout ${LAUNCHD_DOMAIN}/${LABEL}; rm '${PLIST_PATH}'"
