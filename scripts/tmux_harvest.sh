#!/usr/bin/env bash
# Harvest output from tmux-managed agent sessions.
#
# Usage:
#   ./scripts/tmux_harvest.sh --name codex-conductor              # last 50 lines
#   ./scripts/tmux_harvest.sh --name codex-conductor --lines 200  # last 200 lines
#   ./scripts/tmux_harvest.sh --name codex-conductor --full        # full log
#   ./scripts/tmux_harvest.sh --all                                # summary of all sessions
#   ./scripts/tmux_harvest.sh --name codex-conductor --since "5 minutes ago"
#   ./scripts/tmux_harvest.sh --name codex-conductor --grep "error\|FAIL\|passed"

set -euo pipefail

LOG_DIR="${HOME}/.aragora/tmux-sessions"
TMUX_SESSION="aragora"
NAME=""
LINES=50
FULL=false
ALL=false
SINCE=""
GREP_PATTERN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)   NAME="$2"; shift 2 ;;
        --lines)  LINES="$2"; shift 2 ;;
        --full)   FULL=true; shift ;;
        --all)    ALL=true; shift ;;
        --since)  SINCE="$2"; shift 2 ;;
        --grep)   GREP_PATTERN="$2"; shift 2 ;;
        *)        echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

_strip_ansi() {
    # Strip ANSI escape codes for cleaner output
    sed 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b\][^\x07]*\x07//g; s/\r//g'
}

# --- all sessions summary ---
if [[ "${ALL}" == "true" ]]; then
    echo "=== Aragora Agent Sessions ==="
    echo ""

    for meta in "${LOG_DIR}"/*.meta.json; do
        [[ -f "${meta}" ]] || continue
        session_name=$(basename "${meta}" .meta.json)
        log_file="${LOG_DIR}/${session_name}.log"

        # Read metadata
        agent=$(python3 -c "import json; print(json.load(open('${meta}')).get('agent','-'))" 2>/dev/null || echo "-")
        started=$(python3 -c "import json; print(json.load(open('${meta}')).get('started','-'))" 2>/dev/null || echo "-")

        # Check if tmux window is alive
        alive="dead"
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            window_id=$(tmux list-windows -t "${TMUX_SESSION}" -F '#{window_index} #{window_name}' \
                | awk -v name="${session_name}" '$2 == name { print $1; exit }')
            [[ -n "${window_id}" ]] && alive="alive"
        fi

        # Log size
        log_size="0"
        if [[ -f "${log_file}" ]]; then
            log_size=$(wc -l < "${log_file}" | tr -d ' ')
        fi

        # Last meaningful line
        last_line=""
        if [[ -f "${log_file}" ]]; then
            last_line=$(tail -5 "${log_file}" | _strip_ansi | grep -v '^$' | tail -1 | head -c 80)
        fi

        printf "%-20s  %-8s  %-5s  %6s lines  %s\n" "${session_name}" "${agent}" "${alive}" "${log_size}" "${last_line}"
    done

    exit 0
fi

# --- single session harvest ---
if [[ -z "${NAME}" ]]; then
    echo "Usage: --name <session-name> [--lines N] [--full] [--grep pattern]" >&2
    echo "   or: --all" >&2
    exit 1
fi

LOG_FILE="${LOG_DIR}/${NAME}.log"

if [[ ! -f "${LOG_FILE}" ]]; then
    echo "No log file for '${NAME}' at ${LOG_FILE}" >&2
    exit 1
fi

# Apply filters
if [[ "${FULL}" == "true" ]]; then
    if [[ -n "${GREP_PATTERN}" ]]; then
        _strip_ansi < "${LOG_FILE}" | grep -iE "${GREP_PATTERN}"
    else
        _strip_ansi < "${LOG_FILE}"
    fi
elif [[ -n "${SINCE}" ]]; then
    # Use find to check if file was modified since the given time
    # Then show all lines (best effort — log files don't have per-line timestamps)
    echo "--- ${NAME} output (last ${LINES} lines, since '${SINCE}') ---"
    tail -n "${LINES}" "${LOG_FILE}" | _strip_ansi
elif [[ -n "${GREP_PATTERN}" ]]; then
    echo "--- ${NAME} output (matching '${GREP_PATTERN}') ---"
    _strip_ansi < "${LOG_FILE}" | grep -iE "${GREP_PATTERN}" | tail -n "${LINES}"
else
    echo "--- ${NAME} output (last ${LINES} lines) ---"
    tail -n "${LINES}" "${LOG_FILE}" | _strip_ansi
fi
