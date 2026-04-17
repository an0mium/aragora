#!/usr/bin/env bash
# Send a prompt to a named tmux pane in the aragora session.
#
# Usage:
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt "Fix the bug in spec.py"
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt-file /tmp/prompt.md
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt-file /tmp/prompt.md --wait 30
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt "..." --json
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt-file /tmp/prompt.md --dry-run
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt "..." --source scheduled-task
#   ./scripts/tmux_send_prompt.sh --name codex-conductor --prompt "..." --require-idle 10
#
# The prompt is sent via tmux send-keys for single-line prompts, or via
# tmux paste-buffer for multi-line prompts (to avoid shell interpretation).
#
# Every successful send appends a JSON record to
#   ~/.aragora/tmux-sessions/<name>.prompts.log
# with prompt_id, timestamp, char count, source tag, and a preview. This is
# the audit log for prompts dispatched to each session.
#
# Flags:
#   --name <pane-name>        Required. Target tmux window name in the aragora session.
#   --prompt <text>           Inline prompt text.
#   --prompt-file <path>      Path to a prompt file (read as-is).
#   --wait <seconds>          After send, tail the session log for N seconds.
#   --json                    Print a JSON dispatch receipt to stdout (otherwise human-readable).
#   --dry-run                 Print what would be sent without sending. Does not write the audit log.
#   --source <tag>            Optional tag recorded in the audit log to identify the dispatcher
#                             (e.g. "human-operator", "scheduled-task", "claude-code-session").
#   --require-idle <seconds>  Refuse to send unless the session log has been idle (no new output)
#                             for at least N seconds. Prevents stepping on mid-task work.
#                             Exit 2 if the target is busy; override by omitting the flag.

set -euo pipefail

TMUX_SESSION="aragora"
LOG_DIR="${HOME}/.aragora/tmux-sessions"
NAME=""
PROMPT=""
PROMPT_FILE=""
WAIT_SECONDS=0
JSON_OUTPUT=0
DRY_RUN=0
SOURCE_TAG=""
REQUIRE_IDLE_SECONDS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)           NAME="$2"; shift 2 ;;
        --prompt)         PROMPT="$2"; shift 2 ;;
        --prompt-file)    PROMPT_FILE="$2"; shift 2 ;;
        --wait)           WAIT_SECONDS="$2"; shift 2 ;;
        --json)           JSON_OUTPUT=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --source)         SOURCE_TAG="$2"; shift 2 ;;
        --require-idle)   REQUIRE_IDLE_SECONDS="$2"; shift 2 ;;
        *)                echo "Unknown flag: $1" >&2; exit 1 ;;
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

# Verify the tmux session exists (skip for --dry-run so users can preview without tmux running)
if [[ "${DRY_RUN}" -eq 0 ]]; then
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
else
    window_id="(dry-run)"
    TARGET="${TMUX_SESSION}:${window_id}"
fi

LOG_FILE="${LOG_DIR}/${NAME}.log"
PROMPT_AUDIT_LOG="${LOG_DIR}/${NAME}.prompts.log"
CHAR_COUNT="${#PROMPT}"
LINE_COUNT="$(echo "${PROMPT}" | wc -l | tr -d ' ')"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# 16-hex-char prompt ID derived from content + timestamp (stable within a given send)
PROMPT_ID="$(printf '%s\n%s' "${PROMPT}" "${TIMESTAMP}" | shasum -a 256 | cut -c1-16)"
PROMPT_SOURCE_KIND="inline"
if [[ -n "${PROMPT_FILE}" ]]; then
    PROMPT_SOURCE_KIND="file"
fi

# --- Optional idle check (prevent stepping on mid-task work) ---
if [[ "${REQUIRE_IDLE_SECONDS}" -gt 0 && "${DRY_RUN}" -eq 0 ]]; then
    if [[ -f "${LOG_FILE}" ]]; then
        # Last modified time of the log file in Unix seconds.
        # Works on macOS (stat -f %m) and Linux (stat -c %Y).
        if stat -f %m "${LOG_FILE}" >/dev/null 2>&1; then
            LAST_MTIME=$(stat -f %m "${LOG_FILE}")
        else
            LAST_MTIME=$(stat -c %Y "${LOG_FILE}")
        fi
        NOW_SECONDS=$(date +%s)
        IDLE_FOR=$((NOW_SECONDS - LAST_MTIME))
        if [[ "${IDLE_FOR}" -lt "${REQUIRE_IDLE_SECONDS}" ]]; then
            echo "Session '${NAME}' appears busy: last log write ${IDLE_FOR}s ago (< ${REQUIRE_IDLE_SECONDS}s idle threshold)." >&2
            echo "Refusing to send. Re-run without --require-idle to force, or wait until the target is quiet." >&2
            exit 2
        fi
    fi
fi

# --- Dry-run: print what would be sent, do not touch anything ---
if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${JSON_OUTPUT}" -eq 1 ]]; then
        python3 - "${NAME}" "${TARGET}" "${PROMPT_ID}" "${TIMESTAMP}" "${CHAR_COUNT}" "${LINE_COUNT}" "${SOURCE_TAG}" "${PROMPT_SOURCE_KIND}" "${PROMPT_FILE}" <<'PYEOF'
import json, sys
name, target, prompt_id, timestamp, chars, lines, source_tag, source_kind, prompt_file = sys.argv[1:10]
print(json.dumps({
    "dispatch": "dry-run",
    "name": name,
    "target": target,
    "prompt_id": prompt_id,
    "timestamp": timestamp,
    "chars": int(chars),
    "lines": int(lines),
    "source": source_tag or None,
    "source_kind": source_kind,
    "prompt_file": prompt_file or None,
}, indent=2))
PYEOF
    else
        echo "=== DRY RUN — nothing sent, no audit log written ==="
        echo "name:        ${NAME}"
        echo "target:      ${TARGET}"
        echo "prompt_id:   ${PROMPT_ID}"
        echo "timestamp:   ${TIMESTAMP}"
        echo "chars:       ${CHAR_COUNT}"
        echo "lines:       ${LINE_COUNT}"
        echo "source:      ${SOURCE_TAG:-(unset)}"
        echo "source_kind: ${PROMPT_SOURCE_KIND}"
        if [[ -n "${PROMPT_FILE}" ]]; then
            echo "prompt_file: ${PROMPT_FILE}"
        fi
        echo "---"
        echo "first 10 lines of prompt:"
        echo "${PROMPT}" | head -10 | sed 's/^/  | /'
        if [[ "${LINE_COUNT}" -gt 10 ]]; then
            echo "  ... (${LINE_COUNT} lines total)"
        fi
    fi
    exit 0
fi

# --- Actual dispatch ---
# For multi-line prompts, match the proven pattern used by
# aragora/swarm/session_mux.py send_prompt():
#   load-buffer -  (stdin)  → paste-buffer -d (auto-delete)  → send-keys Enter
# An earlier version of this script used `set-buffer -b <name>` + `paste-buffer
# -b <name>`, which appears to have a timing issue where the paste completes
# before the terminal has settled, so the immediately-following Enter is
# consumed as part of the input buffer rather than registered as a submit.
# The session_mux.py pattern is battle-tested across the live agent fleet.
if [[ "${LINE_COUNT}" -gt 1 ]]; then
    printf '%s' "${PROMPT}" | tmux load-buffer -
    tmux paste-buffer -d -t "${TARGET}"
    tmux send-keys -t "${TARGET}" Enter
    DISPATCH_METHOD="paste-buffer"
else
    tmux send-keys -t "${TARGET}" "${PROMPT}" Enter
    DISPATCH_METHOD="send-keys"
fi

# --- Append to prompt audit log (one JSON record per line — jsonl) ---
mkdir -p "${LOG_DIR}"
PREVIEW="$(printf '%s' "${PROMPT}" | head -c 200 | tr '\n' ' ')"
python3 - "${PROMPT_AUDIT_LOG}" "${NAME}" "${PROMPT_ID}" "${TIMESTAMP}" "${CHAR_COUNT}" "${LINE_COUNT}" "${SOURCE_TAG}" "${PROMPT_SOURCE_KIND}" "${PROMPT_FILE}" "${DISPATCH_METHOD}" "${TARGET}" "${PREVIEW}" <<'PYEOF'
import json, sys
audit_log, name, prompt_id, timestamp, chars, lines, source_tag, source_kind, prompt_file, method, target, preview = sys.argv[1:13]
record = {
    "prompt_id": prompt_id,
    "timestamp": timestamp,
    "name": name,
    "target": target,
    "chars": int(chars),
    "lines": int(lines),
    "source": source_tag or None,
    "source_kind": source_kind,
    "prompt_file": prompt_file or None,
    "dispatch_method": method,
    "preview": preview,
}
with open(audit_log, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PYEOF

# --- Emit receipt ---
if [[ "${JSON_OUTPUT}" -eq 1 ]]; then
    python3 - "${NAME}" "${TARGET}" "${PROMPT_ID}" "${TIMESTAMP}" "${CHAR_COUNT}" "${LINE_COUNT}" "${SOURCE_TAG}" "${PROMPT_SOURCE_KIND}" "${PROMPT_FILE}" "${DISPATCH_METHOD}" "${PROMPT_AUDIT_LOG}" <<'PYEOF'
import json, sys
name, target, prompt_id, timestamp, chars, lines, source_tag, source_kind, prompt_file, method, audit_log = sys.argv[1:12]
print(json.dumps({
    "dispatch": "ok",
    "name": name,
    "target": target,
    "prompt_id": prompt_id,
    "timestamp": timestamp,
    "chars": int(chars),
    "lines": int(lines),
    "source": source_tag or None,
    "source_kind": source_kind,
    "prompt_file": prompt_file or None,
    "dispatch_method": method,
    "audit_log": audit_log,
}, indent=2))
PYEOF
else
    echo "Prompt sent to '${NAME}' (${CHAR_COUNT} chars, ${LINE_COUNT} lines, prompt_id=${PROMPT_ID})"
fi

# Optional wait + tail
if [[ "${WAIT_SECONDS}" -gt 0 ]]; then
    if [[ -f "${LOG_FILE}" ]]; then
        echo "Waiting ${WAIT_SECONDS}s, tailing output..."
        timeout "${WAIT_SECONDS}" tail -f "${LOG_FILE}" 2>/dev/null || true
    else
        echo "Log file not found at ${LOG_FILE}, sleeping ${WAIT_SECONDS}s..."
        sleep "${WAIT_SECONDS}"
    fi
fi
