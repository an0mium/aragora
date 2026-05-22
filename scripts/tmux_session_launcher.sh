#!/usr/bin/env bash
# Launch a Codex, Claude, or Droid session inside a named tmux pane.
#
# Usage:
#   ./scripts/tmux_session_launcher.sh --name codex-conductor --agent codex --prompt-file /tmp/prompt.md
#   ./scripts/tmux_session_launcher.sh --name claude-worker --agent claude --autonomous --prompt "Fix tests"
#   ./scripts/tmux_session_launcher.sh --name factory-review --agent droid --prompt "Review PR #6811"
#   ./scripts/tmux_session_launcher.sh --name factory-review --agent factory --cwd /tmp/pr-review --prompt "Review PR #6811"
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
    local line_count method
    line_count="$(echo "${prompt}" | wc -l | tr -d ' ')"
    # For multi-line prompts, match the proven pattern used by
    # aragora/swarm/session_mux.py send_prompt():
    #   load-buffer -  (stdin)  → paste-buffer -d (auto-delete)  → send-keys Enter
    # An earlier version used `set-buffer -b <name>` + `paste-buffer -b <name>`
    # which had a timing issue where the trailing Enter was consumed as part
    # of the input buffer rather than registered as a submit; the affected
    # Codex pane showed "[Pasted Content N chars]" sitting in its input until
    # a user manually pressed Enter. The session_mux.py pattern works.
    if [[ "${line_count}" -gt 1 ]]; then
        printf '%s' "${prompt}" | tmux load-buffer -
        tmux paste-buffer -d -t "${target}"
        sleep "${ARAGORA_TMUX_PASTE_SETTLE_SECONDS:-0.2}"
        tmux send-keys -t "${target}" Enter
        method="paste-buffer"
    else
        tmux send-keys -t "${target}" "${prompt}" Enter
        method="send-keys"
    fi

    # Append a JSON record to the per-session prompt audit log.
    local audit_log="${LOG_DIR}/${NAME}.prompts.log"
    local timestamp prompt_id preview source_kind
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    prompt_id="$(printf '%s\n%s' "${prompt}" "${timestamp}" | shasum -a 256 | cut -c1-16)"
    preview="$(printf '%s' "${prompt}" | head -c 200 | tr '\n' ' ')"
    source_kind="inline"
    if [[ -n "${PROMPT_FILE:-}" ]]; then
        source_kind="file"
    fi
    python3 -c '
import json
import sys

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
' "${audit_log}" "${NAME}" "${prompt_id}" "${timestamp}" "${#prompt}" "${line_count}" "launcher" "${source_kind}" "${PROMPT_FILE:-}" "${method}" "${target}" "${preview}"

    echo "Prompt sent to '${NAME}' (${#prompt} chars, prompt_id=${prompt_id})"
}

wait_for_agent_ready() {
    local agent="$1"
    local log_file="$2"
    local timeout_seconds="$3"
    local pattern=""
    local elapsed=0

    case "${agent}" in
        codex)
            pattern='Use /skills to list available skills|Improve documentation in @filename|Find and fix a bug in @filename|Explain this codebase'
            ;;
        claude)
            pattern='Claude Code|ctrl\+g to edit in VS Code|don'"'"'t ask on'
            ;;
        droid|factory)
            pattern='Droid|Factory|AI coding agent|Type your message'
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
        droid|factory)
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
WORKDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     NAME="$2"; shift 2 ;;
        --agent)    AGENT="$2"; shift 2 ;;
        --cwd)      WORKDIR="$2"; shift 2 ;;
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

if [[ -n "${WORKDIR}" ]]; then
    if [[ ! -d "${WORKDIR}" ]]; then
        echo "Launch cwd does not exist or is not a directory: ${WORKDIR}" >&2
        exit 1
    fi
    WORKDIR="$(cd "${WORKDIR}" && pwd)"
else
    WORKDIR="${REPO_ROOT}"
fi

LOG_FILE="${LOG_DIR}/${NAME}.log"
META_FILE="${LOG_DIR}/${NAME}.meta.json"
REGISTRY_REPO_ROOT="${ARAGORA_TMUX_REGISTRY_REPO_ROOT:-${REPO_ROOT}}"

# Ensure tmux session exists
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${TMUX_SESSION}" -n "_control" -x 200 -y 50
    echo "Created tmux session: ${TMUX_SESSION}"
fi

# If a prompt file is specified, load it before building the launch command.
# Droid/Factory prompted launches use `droid exec -f ...` because interactive
# Droid starts in Auto Off mode and cannot be made autonomous via CLI flags.
if [[ -n "${PROMPT_FILE}" ]]; then
    if [[ ! -f "${PROMPT_FILE}" ]]; then
        echo "Prompt file does not exist: ${PROMPT_FILE}" >&2
        exit 1
    fi
    PROMPT_FILE="$(cd "$(dirname "${PROMPT_FILE}")" && pwd)/$(basename "${PROMPT_FILE}")"
    PROMPT="$(cat "${PROMPT_FILE}")"
fi

# Build the launch command
# When --autonomous is set:
#   - Claude gets ARAGORA_ADMIN_APPROVED=1 → --dangerously-skip-permissions (can run Bash)
#   - Codex gets --full-auto approval mode
#   - Droid/Factory prompted work always uses `droid exec --auto high`; use
#     ARAGORA_DROID_AUTO_LEVEL=low|medium|high to lower the level explicitly.
if [[ "${AGENT}" == "codex" ]]; then
    if [[ "${AUTONOMOUS}" == "1" ]]; then
        LAUNCH_CMD="cd '${WORKDIR}' && ./scripts/codex_session.sh --agent '${NAME}' --base main --full-auto"
    else
        LAUNCH_CMD="cd '${WORKDIR}' && ./scripts/codex_session.sh --agent '${NAME}' --base main"
    fi
elif [[ "${AGENT}" == "claude" ]]; then
    if [[ "${AUTONOMOUS}" == "1" ]]; then
        LAUNCH_CMD="cd '${WORKDIR}' && ARAGORA_ADMIN_APPROVED=1 ./scripts/claude-wt"
    else
        LAUNCH_CMD="cd '${WORKDIR}' && ./scripts/claude-wt"
    fi
elif [[ "${AGENT}" == "droid" || "${AGENT}" == "factory" ]]; then
    DROID_AUTO_LEVEL="${ARAGORA_DROID_AUTO_LEVEL:-high}"
    case "${DROID_AUTO_LEVEL}" in
        low|medium|high) ;;
        *)
            echo "Invalid ARAGORA_DROID_AUTO_LEVEL='${DROID_AUTO_LEVEL}'. Use low, medium, or high." >&2
            exit 1
            ;;
    esac
    if [[ -n "${PROMPT}" ]]; then
        DROID_EXEC_PROMPT_FILE="${PROMPT_FILE}"
        if [[ -z "${DROID_EXEC_PROMPT_FILE}" ]]; then
            DROID_EXEC_PROMPT_FILE="${LOG_DIR}/${NAME}.prompt.md"
            printf '%s\n' "${PROMPT}" > "${DROID_EXEC_PROMPT_FILE}"
        fi
        DROID_EXEC_PROMPT_FILE="$(cd "$(dirname "${DROID_EXEC_PROMPT_FILE}")" && pwd)/$(basename "${DROID_EXEC_PROMPT_FILE}")"
        LAUNCH_CMD="cd '${WORKDIR}' && droid exec --auto '${DROID_AUTO_LEVEL}' --cwd '${WORKDIR}' -f '${DROID_EXEC_PROMPT_FILE}'"
        # `droid exec -f` consumes the prompt directly; do not also paste it
        # into an interactive pane.
        PROMPT=""
    else
        if [[ "${ARAGORA_ALLOW_DROID_AUTO_OFF:-0}" != "1" ]]; then
            echo "Refusing to launch ${AGENT} without a prompt: interactive Droid starts Auto Off. Use --prompt/--prompt-file so this launcher can run droid exec --auto ${DROID_AUTO_LEVEL}, or set ARAGORA_ALLOW_DROID_AUTO_OFF=1 for an explicit manual override." >&2
            exit 1
        fi
        LAUNCH_CMD="cd '${WORKDIR}' && droid --cwd '${WORKDIR}'"
    fi
else
    echo "Unknown agent: ${AGENT}. Use 'codex', 'claude', 'droid', or 'factory'." >&2
    exit 1
fi

# Create new tmux window with logging
WINDOW_TARGET="$(tmux new-window -P -F '#{window_id}' -t "${TMUX_SESSION}" -n "${NAME}")"
PANE_INDEX="$(tmux list-panes -t "${WINDOW_TARGET}" -F '#{pane_index}' | head -1)"
tmux pipe-pane -t "${WINDOW_TARGET}" -o "cat >> '${LOG_FILE}'"

# Send the launch command
tmux send-keys -t "${WINDOW_TARGET}" "${LAUNCH_CMD}" Enter

# Write metadata (avoid embedding prompt content in Python literal). Use
# `python3 -c` instead of a heredoc; some macOS bash/tmux test paths can block
# while writing heredoc content before the Python reader starts.
python3 -c '
import datetime
import json
from pathlib import Path
import sys

name, agent, log_file, repo_root, workdir, prompt_file, meta_file, has_prompt, window_target, tmux_session, pane_index, launch_cmd, registry_repo_root = sys.argv[1:14]
started_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
meta = {
    "name": name,
    "agent": agent,
    "started": started_at,
    "log_file": log_file,
    "repo_root": repo_root,
    "cwd": workdir,
    "worktree": workdir,
    "tmux_session": tmux_session,
    "tmux_window_target": window_target,
    "tmux_pane_index": pane_index,
    "prompt_file": prompt_file or None,
    "has_prompt": bool(has_prompt),
}
Path(meta_file).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

try:
    registry_path = Path(registry_repo_root) / ".aragora" / "session_mux" / "registry.json"
    if registry_path.exists():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    payload.setdefault("schema_version", 1)
    sessions = payload.setdefault("sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}
        payload["sessions"] = sessions
    sessions[name] = {
        "name": name,
        "tmux_session": tmux_session,
        "tmux_window": window_target,
        "tmux_pane": pane_index or "0",
        "launcher_command": launch_cmd,
        "started_at": started_at,
        "last_prompt_at": None,
        "last_prompt_id": None,
        "worktree_path": None,
        "branch": None,
        "log_path": log_file,
        "launcher_log_path": None,
        "meta_path": meta_file,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
except Exception as exc:
    print("warning: failed to register launcher session: " + str(exc), file=sys.stderr)
' "${NAME}" "${AGENT}" "${LOG_FILE}" "${REPO_ROOT}" "${WORKDIR}" "${PROMPT_FILE}" "${META_FILE}" "${PROMPT:+yes}" "${WINDOW_TARGET}" "${TMUX_SESSION}" "${PANE_INDEX}" "${LAUNCH_CMD}" "${REGISTRY_REPO_ROOT}"

echo "Launched '${NAME}' (${AGENT}) in tmux session '${TMUX_SESSION}'"
echo "  Cwd: ${WORKDIR}"
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
