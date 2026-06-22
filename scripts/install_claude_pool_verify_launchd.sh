#!/usr/bin/env bash
# Install a launchd job that periodically verifies the Claude profile pool and
# refreshes .aragora/claude_pool_health.json (consumed by review/debate routing).
#
# Probing an expired access token makes the CLI refresh it from the (valid)
# refresh token, so this hourly job keeps valid-refresh profiles alive WITHOUT a
# browser login. Revoked refresh tokens (duplicate/same-org/cross-machine seats)
# still need a one-time `scripts/claude_profiles_bootstrap.sh login <profile>` or
# a long-lived `claude setup-token`; the job's log + non-zero exit flag those.
set -euo pipefail

LABEL="com.aragora.claude-pool-verify"
INTERVAL_SECONDS=3600  # hourly: matches the routing snapshot TTL (1h)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${HOME}/.aragora/claude-pool-verify.log"
PYTHON_BIN="${ARAGORA_PYTHON:-python3}"
# launchd runs with a minimal PATH (/usr/bin:/bin:...) that lacks ~/.local/bin
# (where the `claude` CLI lives) and Homebrew (gh/git). Without these on PATH the
# profile probe silently fails and every profile is misreported as expired, so
# set an explicit PATH for the job. Override with --path if your tools differ.
BIN_PATH="${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--interval-seconds <n>] [--python <path>] [--path <PATH>]
  --interval-seconds <n>   launchd StartInterval (default: ${INTERVAL_SECONDS})
  --python <path>          Python interpreter (default: ${PYTHON_BIN})
  --path <PATH>            PATH for the job (must include the claude CLI dir)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-seconds) INTERVAL_SECONDS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --path) BIN_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
mkdir -p "$(dirname "$PLIST_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

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
    <string>export PATH="${BIN_PATH}" &amp;&amp; cd "${REPO_ROOT}" &amp;&amp; "${PYTHON_BIN}" scripts/claude_pool_verify.py</string>
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
echo "Uninstall: launchctl unload \"${PLIST_PATH}\" && rm \"${PLIST_PATH}\""
