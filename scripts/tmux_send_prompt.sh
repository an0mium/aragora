#!/usr/bin/env bash
# Send a prompt to a named tmux pane in the aragora session.
#
# Usage:
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt "Fix the bug in spec.py"
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt-file /tmp/prompt.md
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt-file /tmp/prompt.md --wait 30
#
# The prompt is sent line-by-line via tmux send-keys. For multi-line prompts
# from a file, the content is piped through tmux's paste buffer to avoid
# shell interpretation issues.

set -euo pipefail

TMUX_SESSION="aragora"
NAME=""
PROMPT=""
PROMPT_FILE=""
WAIT_SECONDS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)        NAME="$2"; shift 2 ;;
        --prompt)      PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --wait)        WAIT_SECONDS="$2"; shift 2 ;;
        *)             echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${NAME}" ]]; then
    echo "Usage: --name <pane-name> --prompt <text> | --prompt-file <path>" >&2
    exit 1
fi

# Resolve prompt content
if [[ -n "${PROMPT_FILE}" && -f "${PROMPT_FILE}" ]]; then
    PROMPT="$(cat "${PROMPT_FILE}")"
fi

if [[ -z "${PROMPT}" ]]; then
    echo "No prompt specified. Use --prompt or --prompt-file." >&2
    exit 1
fi

# Verify the tmux window exists
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "No aragora tmux session. Launch one first with tmux_session_launcher.sh" >&2
    exit 1
fi

window_id=$(tmux list-windows -t "${TMUX_SESSION}" -F '#{window_index} #{window_name}' \
    | awk -v name="${NAME}" '$2 == name { print $1; exit }')

if [[ -z "${window_id}" ]]; then
    echo "Window '${NAME}' not found in tmux session '${TMUX_SESSION}'." >&2
    echo "Available windows:"
    tmux list-windows -t "${TMUX_SESSION}" -F '  #{window_name}' 2>/dev/null
    exit 1
fi

TARGET="${TMUX_SESSION}:${window_id}"

# For multi-line prompts, use tmux paste buffer to avoid shell escaping issues
if [[ "$(echo "${PROMPT}" | wc -l)" -gt 1 ]]; then
    BUFFER_NAME="aragora-prompt-${NAME}-$$-$(date +%s%N)"
    tmux set-buffer -b "${BUFFER_NAME}" "${PROMPT}"
    tmux paste-buffer -b "${BUFFER_NAME}" -t "${TARGET}"
    tmux send-keys -t "${TARGET}" "" Enter
    tmux delete-buffer -b "${BUFFER_NAME}" 2>/dev/null || true
else
    tmux send-keys -t "${TARGET}" "${PROMPT}" Enter
fi

echo "Prompt sent to '${NAME}' (${#PROMPT} chars)"

# Optional wait + tail
if [[ "${WAIT_SECONDS}" -gt 0 ]]; then
    LOG_FILE="${HOME}/.aragora/tmux-sessions/${NAME}.log"
    if [[ -f "${LOG_FILE}" ]]; then
        echo "Waiting ${WAIT_SECONDS}s, tailing output..."
        timeout "${WAIT_SECONDS}" tail -f "${LOG_FILE}" 2>/dev/null || true
    else
        echo "Log file not found at ${LOG_FILE}, sleeping ${WAIT_SECONDS}s..."
        sleep "${WAIT_SECONDS}"
    fi
fi
