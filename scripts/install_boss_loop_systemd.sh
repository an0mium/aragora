#!/usr/bin/env bash
# Install a systemd --user unit that keeps swarm boss-loop running on Linux.
#
# Mirrors scripts/install_boss_loop_launchd.sh (macOS) so the boss loop is no
# longer a launchd monoculture. Same env contract: BOSS_INTERVAL_SECONDS,
# BOSS_MAX_HOURS, BOSS_MAX_CONSECUTIVE_FAILURES, BOSS_THROTTLE_SECONDS,
# BOSS_LABELS, ARAGORA_* passthrough (incl. ARAGORA_TIER4_TRUSTED_OPERATORS
# when present in the installer's environment).
#
# Default mode is --dry-run: prints the wrapper script, .service and .timer
# unit text to stdout without touching the filesystem. --install writes to
# ~/.config/systemd/user/, reloads the daemon, and enables the units.
# The interpreter is resolved at LAUNCH time by the wrapper (via
# scripts/aragora_runtime.sh); no interpreter path is baked into any unit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_NAME="aragora-boss-loop"
UNIT_DIR="${HOME}/.config/systemd/user"
WRAPPER_PATH="${REPO_ROOT}/.aragora/overnight/${UNIT_NAME}-wrapper.sh"
LOG_PATH="${REPO_ROOT}/.aragora/overnight/boss-loop-systemd.log"
BOSS_REPO="${BOSS_REPO:-synaptent/aragora}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
WORKER_MODEL="${WORKER_MODEL:-codex}"
REVIEW_MODEL="${REVIEW_MODEL:-codex}"
CLAUDE_RUNNER_PROFILES="${CLAUDE_RUNNER_PROFILES:-}"
BOSS_LABELS_RAW="${BOSS_LABELS:-boss-ready}"
MAX_TICKS="${BOSS_MAX_TICKS:-50}"
INTERVAL_SECONDS="${BOSS_INTERVAL_SECONDS:-60}"
MAX_HOURS="${BOSS_MAX_HOURS:-7}"
MAX_CONSECUTIVE_FAILURES="${BOSS_MAX_CONSECUTIVE_FAILURES:-12}"
MAX_PARALLEL_DISPATCHES="${BOSS_MAX_PARALLEL_DISPATCHES:-1}"
AUTONOMY_MODE="${BOSS_AUTONOMY_MODE:-full-auto}"
BOSS_POST_LOOP_ISSUE_REFILL="${BOSS_POST_LOOP_ISSUE_REFILL:-1}"
BOSS_POST_LOOP_MAX_ISSUES="${BOSS_POST_LOOP_MAX_ISSUES:-20}"
BOSS_POST_LOOP_DRY_RUN="${BOSS_POST_LOOP_DRY_RUN:-0}"
THROTTLE_SECONDS="${BOSS_THROTTLE_SECONDS:-300}"
ARAGORA_USER_ID="${ARAGORA_USER_ID:-${USER:-aragora}}"
ARAGORA_WORKSPACE_ID="${ARAGORA_WORKSPACE_ID:-aragora}"
ARAGORA_CLAUDE_PROFILE="${ARAGORA_CLAUDE_PROFILE:-}"
ARAGORA_DEV_COORDINATION_DB="${ARAGORA_DEV_COORDINATION_DB:-}"
MODE="dry-run"
RESTART_POLICY="on-failure"
PING_PONG=false
LABELS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/install_boss_loop_systemd.sh [options]

Modes:
  --dry-run                       Print wrapper + .service + .timer text (default)
  --install                       Write units to ~/.config/systemd/user/ and enable
                                  (Linux only; refused on darwin — use the launchd installer)

Options (env contract mirrors install_boss_loop_launchd.sh):
  --repo <owner/repo>             GitHub repo for boss-loop issue feed (default: synaptent/aragora)
  --target-branch <branch>        Target branch for boss-loop deliverables (default: main)
  --label <label>                 Label filter for boss-loop issue selection (repeatable)
  --worker-model <model>          Worker model (default: codex)
  --review-model <model>          Review model (default: codex)
  --claude-runner-profiles <csv>  Preferred Claude profiles for boss-loop routing
  --max-ticks <n>                 Maximum boss-loop iterations before recycle (default: 50)
  --interval-seconds <n>          Boss-loop polling interval seconds (default: 60)
  --max-hours <n>                 Maximum runtime hours before recycle (default: 7)
  --max-consecutive-failures <n>  Stop after N hard failures (default: 12)
  --max-parallel-dispatches <n>   Maximum parallel boss-loop dispatches (default: 1)
  --autonomy <mode>               Autonomy mode passed to boss-loop (default: full-auto)
  --ping-pong                     Enable ping-pong retry mode
  --post-loop-max-issues <n>      Create up to N fresh issues after a clean exit (default: 20)
  --post-loop-dry-run             Preview post-loop issue generation without creating issues
  --no-post-loop-issue-refill     Disable post-loop boss-ready issue generation
  --user-id <id>                  Export ARAGORA_USER_ID for the service
  --workspace-id <id>             Export ARAGORA_WORKSPACE_ID for the service
  --claude-profile <name>         Export ARAGORA_CLAUDE_PROFILE for the service
  --coordination-db <path>        Export ARAGORA_DEV_COORDINATION_DB for shared handoff
  --throttle-seconds <n>          Restart backoff seconds after exits (default: 300)
  --log-path <file>               Log file path (default: .aragora/overnight/boss-loop-systemd.log)
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
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --install)
            MODE="install"
            shift
            ;;
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
        --post-loop-max-issues)
            BOSS_POST_LOOP_MAX_ISSUES="${2:-$BOSS_POST_LOOP_MAX_ISSUES}"
            shift 2
            ;;
        --post-loop-dry-run)
            BOSS_POST_LOOP_DRY_RUN=1
            shift
            ;;
        --no-post-loop-issue-refill)
            BOSS_POST_LOOP_ISSUE_REFILL=0
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
        --coordination-db)
            ARAGORA_DEV_COORDINATION_DB="${2:-$ARAGORA_DEV_COORDINATION_DB}"
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
            RESTART_POLICY="no"
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

if [[ "${MODE}" == "install" && "$(uname -s)" == "Darwin" ]]; then
    echo "Refusing --install on darwin (macOS): systemd is not available here." >&2
    echo "Use ./scripts/install_boss_loop_launchd.sh on macOS instead." >&2
    exit 1
fi

# Boss-loop runner flags (identical contract to the launchd installer).
runner_args="--boss-repo \"${BOSS_REPO}\" --target-branch \"${TARGET_BRANCH}\" --worker-model \"${WORKER_MODEL}\" --review-model \"${REVIEW_MODEL}\""
for label in "${LABELS[@]}"; do
    runner_args="${runner_args} --label \"${label}\""
done
if [[ -n "${CLAUDE_RUNNER_PROFILES}" ]]; then
    runner_args="${runner_args} --claude-runner-profiles \"${CLAUDE_RUNNER_PROFILES}\""
fi
runner_args="${runner_args} --max-ticks \"${MAX_TICKS}\" --interval \"${INTERVAL_SECONDS}\" --max-consecutive-failures \"${MAX_CONSECUTIVE_FAILURES}\" --autonomy \"${AUTONOMY_MODE}\" --max-hours \"${MAX_HOURS}\" --boss-max-parallel-dispatches \"${MAX_PARALLEL_DISPATCHES}\""
if [[ "${PING_PONG}" == true ]]; then
    runner_args="${runner_args} --ping-pong"
fi

# Wrapper: resolves the interpreter at LAUNCH time via aragora_runtime.sh so a
# moved/rebuilt .venv degrades gracefully. No interpreter path is baked into
# the unit or this wrapper.
render_wrapper() {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' '# Generated by scripts/install_boss_loop_systemd.sh — do not edit by hand.'
    printf '%s\n' 'set -euo pipefail'
    printf 'REPO_ROOT="%s"\n' "${REPO_ROOT}"
    printf '%s\n' 'export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.pyenv/shims:${HOME}/.local/bin:${PATH}"'
    printf '%s\n' '# shellcheck source=scripts/aragora_runtime.sh'
    printf '%s\n' 'source "${REPO_ROOT}/scripts/aragora_runtime.sh"'
    printf '%s\n' '# Launch-time interpreter validation; run_boss_cycle.sh re-resolves on exec.'
    printf '%s\n' 'PYTHON_BIN="$(resolve_aragora_python '\''import pydantic'\'' '\''boss-loop'\'' '\''boss-loop systemd wrapper'\'')"'
    printf '%s\n' 'echo "boss-loop systemd wrapper using interpreter: ${PYTHON_BIN}"'
    printf '%s\n' 'cd "${REPO_ROOT}"'
    printf 'exec "${REPO_ROOT}/scripts/run_boss_cycle.sh" %s\n' "${runner_args}"
}

# Exponential restart backoff (RestartSteps/RestartMaxDelaySec) requires
# systemd >= 254; emit the directives commented out on older/unknown hosts.
render_restart_backoff() {
    local ver=""
    if command -v systemctl >/dev/null 2>&1; then
        ver="$(systemctl --version 2>/dev/null | awk 'NR==1 {print $2}')"
    fi
    if [[ "${ver}" =~ ^[0-9]+$ ]] && (( ver >= 254 )); then
        printf '%s\n' '# Exponential backoff: RestartSec -> RestartMaxDelaySec over RestartSteps.'
        printf '%s\n' 'RestartMaxDelaySec=3600'
        printf '%s\n' 'RestartSteps=6'
    else
        printf '# RestartMaxDelaySec/RestartSteps need systemd >= 254 (detected: %s).\n' "${ver:-none}"
        printf '%s\n' '# Uncomment after confirming with: systemctl --version'
        printf '%s\n' '# RestartMaxDelaySec=3600'
        printf '%s\n' '# RestartSteps=6'
    fi
}

render_service() {
    local start_limit_interval=$(( MAX_CONSECUTIVE_FAILURES * THROTTLE_SECONDS ))
    printf '%s\n' '[Unit]'
    printf '%s\n' 'Description=Aragora swarm boss-loop (mirrors com.aragora.swarm-boss-loop launchd job)'
    printf '%s\n' 'After=network-online.target'
    printf '%s\n' 'Wants=network-online.target'
    printf '%s\n' '# Mirror launchd MAX_CONSECUTIVE_FAILURES: stop restarting after N failures'
    printf '%s\n' '# inside the window, then the timer re-kicks the service later.'
    printf 'StartLimitIntervalSec=%s\n' "${start_limit_interval}"
    printf 'StartLimitBurst=%s\n' "${MAX_CONSECUTIVE_FAILURES}"
    printf '\n'
    printf '%s\n' '[Service]'
    printf '%s\n' 'Type=simple'
    printf 'WorkingDirectory=%s\n' "${REPO_ROOT}"
    printf 'ExecStart=%s\n' "${WRAPPER_PATH}"
    printf 'Restart=%s\n' "${RESTART_POLICY}"
    printf 'RestartSec=%s\n' "${THROTTLE_SECONDS}"
    render_restart_backoff
    printf 'Environment="ARAGORA_USER_ID=%s"\n' "${ARAGORA_USER_ID}"
    printf 'Environment="ARAGORA_WORKSPACE_ID=%s"\n' "${ARAGORA_WORKSPACE_ID}"
    printf 'Environment="ARAGORA_POST_LOOP_ISSUE_REFILL=%s"\n' "${BOSS_POST_LOOP_ISSUE_REFILL}"
    printf 'Environment="ARAGORA_POST_LOOP_MAX_ISSUES=%s"\n' "${BOSS_POST_LOOP_MAX_ISSUES}"
    printf 'Environment="ARAGORA_POST_LOOP_DRY_RUN=%s"\n' "${BOSS_POST_LOOP_DRY_RUN}"
    if [[ -n "${ARAGORA_CLAUDE_PROFILE}" ]]; then
        printf 'Environment="ARAGORA_CLAUDE_PROFILE=%s"\n' "${ARAGORA_CLAUDE_PROFILE}"
    fi
    if [[ -n "${ARAGORA_DEV_COORDINATION_DB}" ]]; then
        printf 'Environment="ARAGORA_DEV_COORDINATION_DB=%s"\n' "${ARAGORA_DEV_COORDINATION_DB}"
    fi
    if [[ -n "${ARAGORA_TIER4_TRUSTED_OPERATORS:-}" ]]; then
        printf 'Environment="ARAGORA_TIER4_TRUSTED_OPERATORS=%s"\n' "${ARAGORA_TIER4_TRUSTED_OPERATORS}"
    fi
    printf 'StandardOutput=append:%s\n' "${LOG_PATH}"
    printf 'StandardError=append:%s\n' "${LOG_PATH}"
    printf '\n'
    printf '%s\n' '[Install]'
    printf '%s\n' 'WantedBy=default.target'
}

render_timer() {
    printf '%s\n' '[Unit]'
    printf '%s\n' 'Description=Re-kick aragora boss-loop after boot or start-limit lockout'
    printf '\n'
    printf '%s\n' '[Timer]'
    printf 'Unit=%s.service\n' "${UNIT_NAME}"
    printf '%s\n' 'OnBootSec=60'
    printf '%s\n' '# Safety net: if the service hit StartLimitBurst and went inactive, restart it'
    printf '%s\n' '# once the start-limit window has lapsed (mirrors launchd KeepAlive+Throttle).'
    printf 'OnUnitInactiveSec=%s\n' "$(( MAX_CONSECUTIVE_FAILURES * THROTTLE_SECONDS + THROTTLE_SECONDS ))"
    printf '%s\n' 'Persistent=true'
    printf '\n'
    printf '%s\n' '[Install]'
    printf '%s\n' 'WantedBy=timers.target'
}

if [[ "${MODE}" == "dry-run" ]]; then
    echo "# ==== wrapper: ${WRAPPER_PATH} ===="
    render_wrapper
    echo
    echo "# ==== unit: ${UNIT_DIR}/${UNIT_NAME}.service ===="
    render_service
    echo
    echo "# ==== unit: ${UNIT_DIR}/${UNIT_NAME}.timer ===="
    render_timer
    echo
    echo "# Dry run only. Re-run with --install on a Linux host to activate." >&2
    exit 0
fi

mkdir -p "${UNIT_DIR}"
mkdir -p "$(dirname "${WRAPPER_PATH}")"
mkdir -p "$(dirname "${LOG_PATH}")"

render_wrapper >"${WRAPPER_PATH}"
chmod 0755 "${WRAPPER_PATH}"
render_service >"${UNIT_DIR}/${UNIT_NAME}.service"
render_timer >"${UNIT_DIR}/${UNIT_NAME}.timer"

systemctl --user daemon-reload
systemctl --user enable --now "${UNIT_NAME}.timer"
systemctl --user enable --now "${UNIT_NAME}.service"

echo "Installed systemd user units: ${UNIT_NAME}.service, ${UNIT_NAME}.timer"
echo "Wrapper: ${WRAPPER_PATH}"
echo "Log: ${LOG_PATH}"
echo "Interpreter is resolved at runtime via scripts/aragora_runtime.sh"
echo "Boss repo: ${BOSS_REPO}"
echo "Labels: ${LABELS[*]}"
