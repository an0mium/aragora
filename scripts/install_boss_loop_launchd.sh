#!/usr/bin/env bash
# Install a launchd job that keeps swarm boss-loop running on macOS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.aragora.swarm-boss-loop"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/boss-loop-launchd.log"
BOSS_REPO="${BOSS_REPO:-synaptent/aragora}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
WORKER_MODEL="${WORKER_MODEL:-claude}"
REVIEW_MODEL="${REVIEW_MODEL:-codex}"
CLAUDE_RUNNER_PROFILES="${CLAUDE_RUNNER_PROFILES:-}"
BOSS_LABELS_RAW="${BOSS_LABELS:-boss-ready}"
MAX_TICKS="${BOSS_MAX_TICKS:-25}"
INTERVAL_SECONDS="${BOSS_INTERVAL_SECONDS:-60}"
MAX_HOURS="${BOSS_MAX_HOURS:-12}"
MAX_CONSECUTIVE_FAILURES="${BOSS_MAX_CONSECUTIVE_FAILURES:-12}"
MAX_PARALLEL_DISPATCHES="${BOSS_MAX_PARALLEL_DISPATCHES:-4}"
AUTONOMY_MODE="${BOSS_AUTONOMY_MODE:-full-auto}"
THROTTLE_SECONDS="${BOSS_THROTTLE_SECONDS:-300}"
ARAGORA_USER_ID="${ARAGORA_USER_ID:-${USER}}"
ARAGORA_WORKSPACE_ID="${ARAGORA_WORKSPACE_ID:-aragora}"
ARAGORA_CLAUDE_PROFILE="${ARAGORA_CLAUDE_PROFILE:-}"
KEEPALIVE=true
PING_PONG=false
LABELS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/install_boss_loop_launchd.sh [options]

Options:
  --repo <owner/repo>             GitHub repo for boss-loop issue feed (default: synaptent/aragora)
  --target-branch <branch>        Target branch for boss-loop deliverables (default: main)
  --label <label>                 Label filter for boss-loop issue selection (repeatable)
  --worker-model <model>          Worker model (default: claude)
  --review-model <model>          Review model (default: codex)
  --claude-runner-profiles <csv>  Preferred Claude profiles for boss-loop routing
  --max-ticks <n>                 Maximum boss-loop iterations before recycle (default: 25)
  --interval-seconds <n>          Boss-loop polling interval seconds (default: 60)
  --max-hours <n>                 Maximum runtime hours before recycle (default: 12)
  --max-consecutive-failures <n>  Stop after N hard failures (default: 12)
  --max-parallel-dispatches <n>   Maximum parallel boss-loop dispatches (default: 4)
  --autonomy <mode>               Autonomy mode passed to boss-loop (default: full-auto)
  --ping-pong                     Enable ping-pong retry mode
  --user-id <id>                  Export ARAGORA_USER_ID for the service
  --workspace-id <id>             Export ARAGORA_WORKSPACE_ID for the service
  --claude-profile <name>         Export ARAGORA_CLAUDE_PROFILE for the service
  --throttle-seconds <n>          launchd throttle interval after exits (default: 300)
  --log-path <file>               Log file path (default: .aragora/overnight/boss-loop-launchd.log)
  --no-keepalive                  Do not auto-restart the service after exits
  --help                          Show this help
EOF
}

trim_text() {
    printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
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
        --target-branch)
            TARGET_BRANCH="${2:-$TARGET_BRANCH}"
            shift 2
            ;;
        --label)
            LABELS+=("$(trim_text "${2:-}")")
            shift 2
            ;;
        --worker-model)
            WORKER_MODEL="${2:-$WORKER_MODEL}"
            shift 2
            ;;
        --review-model)
            REVIEW_MODEL="${2:-$REVIEW_MODEL}"
            shift 2
            ;;
        --claude-runner-profiles)
            CLAUDE_RUNNER_PROFILES="${2:-$CLAUDE_RUNNER_PROFILES}"
            shift 2
            ;;
        --max-ticks)
            MAX_TICKS="${2:-$MAX_TICKS}"
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
        --max-parallel-dispatches)
            MAX_PARALLEL_DISPATCHES="${2:-$MAX_PARALLEL_DISPATCHES}"
            shift 2
            ;;
        --autonomy)
            AUTONOMY_MODE="${2:-$AUTONOMY_MODE}"
            shift 2
            ;;
        --ping-pong)
            PING_PONG=true
            shift
            ;;
        --user-id)
            ARAGORA_USER_ID="${2:-$ARAGORA_USER_ID}"
            shift 2
            ;;
        --workspace-id)
            ARAGORA_WORKSPACE_ID="${2:-$ARAGORA_WORKSPACE_ID}"
            shift 2
            ;;
        --claude-profile)
            ARAGORA_CLAUDE_PROFILE="${2:-$ARAGORA_CLAUDE_PROFILE}"
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

if [[ ${#LABELS[@]} -eq 0 ]]; then
    IFS=',' read -r -a raw_labels <<< "${BOSS_LABELS_RAW}"
    for raw_label in "${raw_labels[@]}"; do
        trimmed_label="$(trim_text "${raw_label}")"
        if [[ -n "${trimmed_label}" ]]; then
            LABELS+=("${trimmed_label}")
        fi
    done
fi

if [[ ${#LABELS[@]} -eq 0 ]]; then
    echo "At least one --label (or BOSS_LABELS env var) is required." >&2
    exit 2
fi

validate_integer "max-ticks" "${MAX_TICKS}"
validate_integer "interval-seconds" "${INTERVAL_SECONDS}"
validate_integer "max-consecutive-failures" "${MAX_CONSECUTIVE_FAILURES}"
validate_integer "max-parallel-dispatches" "${MAX_PARALLEL_DISPATCHES}"
validate_integer "throttle-seconds" "${THROTTLE_SECONDS}"
if ! [[ "${MAX_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "max-hours must be numeric" >&2
    exit 2
fi

mkdir -p "$(dirname "${PLIST_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"
mkdir -p "${REPO_ROOT}/.aragora/overnight"

VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
command_string="cd \"${REPO_ROOT}\" && export PATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH\" && export ARAGORA_USER_ID=\"${ARAGORA_USER_ID}\" && export ARAGORA_WORKSPACE_ID=\"${ARAGORA_WORKSPACE_ID}\" && source \"${VENV_ACTIVATE}\""
if [[ -n "${ARAGORA_CLAUDE_PROFILE}" ]]; then
    command_string="${command_string} && export ARAGORA_CLAUDE_PROFILE=\"${ARAGORA_CLAUDE_PROFILE}\""
fi
command_string="${command_string} && exec python3 -u -m aragora.cli.main swarm boss-loop --boss-repo \"${BOSS_REPO}\" --target-branch \"${TARGET_BRANCH}\" --worker-model \"${WORKER_MODEL}\" --review-model \"${REVIEW_MODEL}\""
for label in "${LABELS[@]}"; do
    command_string="${command_string} --label \"${label}\""
done
if [[ -n "${CLAUDE_RUNNER_PROFILES}" ]]; then
    command_string="${command_string} --claude-runner-profiles \"${CLAUDE_RUNNER_PROFILES}\""
fi
command_string="${command_string} --max-ticks \"${MAX_TICKS}\" --interval \"${INTERVAL_SECONDS}\" --max-consecutive-failures \"${MAX_CONSECUTIVE_FAILURES}\" --autonomy \"${AUTONOMY_MODE}\" --max-hours \"${MAX_HOURS}\" --boss-max-parallel-dispatches \"${MAX_PARALLEL_DISPATCHES}\""
if [[ "${PING_PONG}" == true ]]; then
    command_string="${command_string} --ping-pong"
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
echo "Boss repo: ${BOSS_REPO}"
echo "Labels: ${LABELS[*]}"
