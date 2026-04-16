#!/usr/bin/env bash
# Launch a Codex or Claude session inside a named tmux pane.
#
# Usage:
#   ./scripts/tmux_session_launcher.sh --name codex-conductor --agent codex --prompt-file /tmp/prompt.md
#   ./scripts/tmux_session_launcher.sh --name claude-worker --agent claude --autonomous --prompt "Fix tests"
#   ./scripts/tmux_session_launcher.sh --name codex-qa --agent codex --autonomous --prompt "Fix the tests"
#   ./scripts/tmux_session_launcher.sh --list
#   ./scripts/tmux_session_launcher.sh --kill codex-conductor
#
# Flags:
#   --autonomous   Grant full permissions (Claude: --dangerously-skip-permissions, Codex: --full-auto)
#                  Required for agents to run Bash, edit files, etc. in tmux lanes.
#
# Each session gets:
#   - a dedicated tmux window in the "aragora" session
#   - an isolated git worktree (via codex_session.sh or claude-wt)
#   - output logged to ~/.aragora/tmux-sessions/<name>.log
#   - a metadata file at ~/.aragora/tmux-sessions/<name>.meta.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TMUX_SESSION="aragora"
LOG_DIR="${HOME}/.aragora/tmux-sessions"
mkdir -p "${LOG_DIR}"

send_prompt_to_target() {
    local target="$1"
    local prompt="$2"
    if [[ "$(echo "${prompt}" | wc -l)" -gt 1 ]]; then
        local buffer_name="aragora-prompt-launch-${NAME}-$$-$(date +%s%N)"
        tmux set-buffer -b "${buffer_name}" "${prompt}"
        tmux paste-buffer -b "${buffer_name}" -t "${target}"
        tmux send-keys -t "${target}" "" Enter
        tmux delete-buffer -b "${buffer_name}" 2>/dev/null || true
    else
        tmux send-keys -t "${target}" "${prompt}" Enter
    fi
    echo "Prompt sent to '${NAME}' (${#prompt} chars)"
}

wait_for_agent_ready() {
    local agent="$1"
    local log_file="$2"
    local timeout_seconds="$3"
    local pattern=""
    local elapsed=0

    case "${agent}" in
        codex)
            pattern='OpenAI Codex|Use /skills to list available skills|Improve documentation in @filename|Find and fix a bug in @filename|Explain this codebase|Use /rename to rename your threads'
            ;;
        claude)
            pattern='Claude Code|ctrl\+g to edit in VS Code|don'"'"'t ask on'
            ;;
    esac

    if [[ -z "${pattern}" ]]; then
        return 1
    fi

    while (( elapsed < timeout_seconds )); do
        if [[ -f "${log_file}" ]] && grep -Eq "${pattern}" "${log_file}"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1
}

default_init_wait_seconds() {
    local agent="$1"
    case "${agent}" in
        codex)
            echo "60"
            ;;
        claude)
            echo "30"
            ;;
        *)
            echo "30"
            ;;
    esac
}

# --- argument parsing ---
NAME=""
AGENT="codex"
PROMPT=""
PROMPT_FILE=""
ACTION="launch"
AUTONOMOUS="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     NAME="$2"; shift 2 ;;
        --agent)    AGENT="$2"; shift 2 ;;
        --prompt)   PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --autonomous) AUTONOMOUS="1"; shift ;;
        --list)     ACTION="list"; shift ;;
        --kill)     ACTION="kill"; NAME="$2"; shift 2 ;;
        --status)   ACTION="status"; shift ;;
        *)          echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# --- list action ---
if [[ "${ACTION}" == "list" || "${ACTION}" == "status" ]]; then
    if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "No aragora tmux session running."
        exit 0
    fi
    echo "=== Aragora tmux panes ==="
    tmux list-windows -t "${TMUX_SESSION}" -F '#{window_index} #{window_name} #{pane_current_command}' 2>/dev/null || true

    echo ""
    echo "=== Session metadata ==="
    for meta in "${LOG_DIR}"/*.meta.json; do
        [[ -f "${meta}" ]] || continue
        basename "${meta}" .meta.json
        python3 -c "
import json, sys
d = json.load(open('${meta}'))
print(f\"  agent={d.get('agent','-')} started={d.get('started','-')} pid={d.get('pane_pid','-')}\")
" 2>/dev/null || true
    done
    exit 0
fi

# --- kill action ---
if [[ "${ACTION}" == "kill" ]]; then
    if [[ -z "${NAME}" ]]; then
        echo "Usage: --kill <name>" >&2; exit 1
    fi
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        # Find and kill the window by name
        window_id=$(tmux list-windows -t "${TMUX_SESSION}" -F '#{window_index} #{window_name}' \
            | awk -v name="${NAME}" '$2 == name { print $1; exit }')
        if [[ -n "${window_id}" ]]; then
            tmux kill-window -t "${TMUX_SESSION}:${window_id}"
            echo "Killed tmux window: ${NAME}"
        else
            echo "Window '${NAME}' not found."
        fi
    fi
    rm -f "${LOG_DIR}/${NAME}.meta.json"
    exit 0
fi

# --- launch action ---
if [[ -z "${NAME}" ]]; then
    NAME="${AGENT}-$(date +%H%M%S)"
fi

LOG_FILE="${LOG_DIR}/${NAME}.log"
META_FILE="${LOG_DIR}/${NAME}.meta.json"

# Ensure tmux session exists
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${TMUX_SESSION}" -n "_control" -x 200 -y 50
    echo "Created tmux session: ${TMUX_SESSION}"
fi

# Build the launch command
# When --autonomous is set:
#   - Claude gets ARAGORA_ADMIN_APPROVED=1 → --dangerously-skip-permissions (can run Bash)
#   - Codex gets --full-auto approval mode
if [[ "${AGENT}" == "codex" ]]; then
    if [[ "${AUTONOMOUS}" == "1" ]]; then
        LAUNCH_CMD="cd '${REPO_ROOT}' && ./scripts/codex_session.sh --agent codex --base main --full-auto"
    else
        LAUNCH_CMD="cd '${REPO_ROOT}' && ./scripts/codex_session.sh --agent codex --base main"
    fi
elif [[ "${AGENT}" == "claude" ]]; then
    if [[ "${AUTONOMOUS}" == "1" ]]; then
        LAUNCH_CMD="cd '${REPO_ROOT}' && ARAGORA_ADMIN_APPROVED=1 ./scripts/claude-wt"
    else
        LAUNCH_CMD="cd '${REPO_ROOT}' && ./scripts/claude-wt"
    fi
else
    echo "Unknown agent: ${AGENT}. Use 'codex' or 'claude'." >&2
    exit 1
fi

# If a prompt file is specified, we'll feed it after launch
if [[ -n "${PROMPT_FILE}" && -f "${PROMPT_FILE}" ]]; then
    PROMPT="$(cat "${PROMPT_FILE}")"
fi

# Create new tmux window with logging
WINDOW_TARGET="$(tmux new-window -P -F '#{window_id}' -t "${TMUX_SESSION}" -n "${NAME}")"
tmux pipe-pane -t "${WINDOW_TARGET}" -o "cat >> '${LOG_FILE}'"

# Send the launch command
tmux send-keys -t "${WINDOW_TARGET}" "${LAUNCH_CMD}" Enter

# Write metadata (avoid embedding prompt content in Python literal)
python3 - "${NAME}" "${AGENT}" "${LOG_FILE}" "${REPO_ROOT}" "${PROMPT_FILE}" "${META_FILE}" "${PROMPT:+yes}" "${WINDOW_TARGET}" <<'PYEOF'
import json, datetime, sys
name, agent, log_file, repo_root, prompt_file, meta_file, has_prompt, window_target = sys.argv[1:9]
meta = {
    "name": name,
    "agent": agent,
    "started": datetime.datetime.now().isoformat(),
    "log_file": log_file,
    "repo_root": repo_root,
    "tmux_window_target": window_target,
    "prompt_file": prompt_file or None,
    "has_prompt": bool(has_prompt),
}
with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

echo "Launched '${NAME}' (${AGENT}) in tmux session '${TMUX_SESSION}'"
echo "  Log: ${LOG_FILE}"
echo "  Meta: ${META_FILE}"

# If there's a prompt to send, wait for the session to initialize then send it
if [[ -n "${PROMPT}" ]]; then
    INIT_WAIT_SECONDS="${ARAGORA_TMUX_INIT_WAIT_SECONDS:-$(default_init_wait_seconds "${AGENT}")}"
    SEND_ON_TIMEOUT="${ARAGORA_TMUX_SEND_ON_TIMEOUT:-0}"
    echo "Waiting up to ${INIT_WAIT_SECONDS}s for ${AGENT} readiness before sending prompt..."
    if wait_for_agent_ready "${AGENT}" "${LOG_FILE}" "${INIT_WAIT_SECONDS}"; then
        echo "Readiness markers detected for ${NAME}."
        send_prompt_to_target "${WINDOW_TARGET}" "${PROMPT}"
    else
        if [[ "${SEND_ON_TIMEOUT}" == "1" ]]; then
            echo "Timed out waiting for readiness markers for ${NAME}; sending prompt anyway because ARAGORA_TMUX_SEND_ON_TIMEOUT=1."
            send_prompt_to_target "${WINDOW_TARGET}" "${PROMPT}"
        else
            echo "Timed out waiting for readiness markers for ${NAME}; prompt not sent."
            echo "Re-send once ready with: ${SCRIPT_DIR}/tmux_send_prompt.sh --name ${NAME} --prompt '...'"
        fi
    fi
fi
