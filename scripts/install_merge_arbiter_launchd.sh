#!/usr/bin/env bash
# Install a launchd job that keeps swarm merge-arbiter running on macOS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.aragora.swarm-merge-arbiter"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/merge-arbiter-launchd.log"
BOSS_REPO="${BOSS_REPO:-synaptent/aragora}"
BRANCH_PREFIXES="${BRANCH_PREFIXES:-boss-harvest}"
INTERVAL_SECONDS="${MERGE_ARBITER_INTERVAL_SECONDS:-120}"
MAX_HOURS="${MERGE_ARBITER_MAX_HOURS:-12}"
MAX_CONSECUTIVE_FAILURES="${MERGE_ARBITER_MAX_CONSECUTIVE_FAILURES:-3}"
THROTTLE_SECONDS="${MERGE_ARBITER_THROTTLE_SECONDS:-300}"
ARAGORA_USER_ID="${ARAGORA_USER_ID:-${USER}}"
ARAGORA_WORKSPACE_ID="${ARAGORA_WORKSPACE_ID:-aragora}"
KEEPALIVE=true
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: ./scripts/install_merge_arbiter_launchd.sh [options]

Options:
  --repo <owner/repo>             GitHub repo for merge-arbiter polling (default: synaptent/aragora)
  --branch-prefix <csv>           Branch prefixes to manage (default: boss-harvest)
  --interval-seconds <n>          Merge-arbiter polling interval seconds (default: 120)
  --max-hours <n>                 Maximum runtime hours before recycle (default: 12)
  --max-consecutive-failures <n>  Stop after N hard failures (default: 3)
  --user-id <id>                  Export ARAGORA_USER_ID for the service
  --workspace-id <id>             Export ARAGORA_WORKSPACE_ID for the service
  --throttle-seconds <n>          launchd throttle interval after exits (default: 300)
  --log-path <file>               Log file path (default: .aragora/overnight/merge-arbiter-launchd.log)
  --dry-run                       Install the service in dry-run mode
  --no-keepalive                  Do not auto-restart the service after exits
  --help                          Show this help
EOF
}

resolve_python_bin() {
    local candidates=()
    local candidate=""
    local python_cmd=""

    if [[ -n "${ARAGORA_PYTHON:-}" ]]; then
        candidates+=("${ARAGORA_PYTHON}")
    fi
    if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
        candidates+=("${REPO_ROOT}/.venv/bin/python3")
    fi
    if python_cmd="$(command -v python3 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    if python_cmd="$(command -v python 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
            continue
        fi
        if "${candidate}" -c 'import pydantic' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if command -v pyenv >/dev/null 2>&1; then
        candidate="$(pyenv which python3 2>/dev/null || true)"
        if [[ -n "${candidate}" && -x "${candidate}" ]] && "${candidate}" -c 'import pydantic' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    echo "No usable python interpreter with pydantic found for merge-arbiter launchd install." >&2
    exit 2
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
        --repo)
            BOSS_REPO="${2:-$BOSS_REPO}"
            shift 2
            ;;
        --branch-prefix)
            BRANCH_PREFIXES="${2:-$BRANCH_PREFIXES}"
            shift 2
            ;;
        --interval-seconds)
            INTERVAL_SECONDS="${2:-$INTERVAL_SECONDS}"
            shift 2
            ;;
        --max-hours)
            MAX_HOURS="${2:-$MAX_HOURS}"
            shift 2
            ;;
        --max-consecutive-failures)
            MAX_CONSECUTIVE_FAILURES="${2:-$MAX_CONSECUTIVE_FAILURES}"
            shift 2
            ;;
        --user-id)
            ARAGORA_USER_ID="${2:-$ARAGORA_USER_ID}"
            shift 2
            ;;
        --workspace-id)
            ARAGORA_WORKSPACE_ID="${2:-$ARAGORA_WORKSPACE_ID}"
            shift 2
            ;;
        --throttle-seconds)
            THROTTLE_SECONDS="${2:-$THROTTLE_SECONDS}"
            shift 2
            ;;
        --log-path)
            LOG_PATH="${2:-$LOG_PATH}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-keepalive)
            KEEPALIVE=false
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

validate_integer "interval-seconds" "${INTERVAL_SECONDS}"
validate_integer "max-consecutive-failures" "${MAX_CONSECUTIVE_FAILURES}"
validate_integer "throttle-seconds" "${THROTTLE_SECONDS}"
if ! [[ "${MAX_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "max-hours must be numeric" >&2
    exit 2
fi

mkdir -p "$(dirname "${PLIST_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"
mkdir -p "${REPO_ROOT}/.aragora/overnight"

PYTHON_BIN="$(resolve_python_bin)"
PYTHON_DIR="$(dirname "${PYTHON_BIN}")"
command_string="cd \"${REPO_ROOT}\" && export PATH=\"${PYTHON_DIR}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH\" && export ARAGORA_USER_ID=\"${ARAGORA_USER_ID}\" && export ARAGORA_WORKSPACE_ID=\"${ARAGORA_WORKSPACE_ID}\" && export ARAGORA_PYTHON=\"${PYTHON_BIN}\" && exec \"${PYTHON_BIN}\" -u -m aragora.cli.main swarm merge-arbiter --boss-repo \"${BOSS_REPO}\" --branch-prefix \"${BRANCH_PREFIXES}\" --interval \"${INTERVAL_SECONDS}\" --max-hours \"${MAX_HOURS}\" --max-consecutive-failures \"${MAX_CONSECUTIVE_FAILURES}\""
if [[ "${DRY_RUN}" == true ]]; then
    command_string="${command_string} --dry-run"
fi
command_xml="${command_string//&/&amp;}"

keepalive_block=""
if [[ "${KEEPALIVE}" == true ]]; then
    keepalive_block=$'  <key>KeepAlive</key>\n  <true/>\n  <key>ThrottleInterval</key>\n  <integer>'"${THROTTLE_SECONDS}"$'</integer>\n'
fi

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
  <true/>
${keepalive_block}  <key>WorkingDirectory</key>
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
echo "Python: ${PYTHON_BIN}"
echo "Branch prefixes: ${BRANCH_PREFIXES}"
