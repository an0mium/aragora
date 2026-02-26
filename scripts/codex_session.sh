#!/usr/bin/env bash
# Launch Codex in an auto-managed worktree.
#
# Usage:
#   ./scripts/codex_session.sh
#   ./scripts/codex_session.sh --agent codex-qa
#   ./scripts/codex_session.sh --orchestrator crewai
#   ./scripts/codex_session.sh --agent codex-qa --base main -- python -m pytest tests/debate -q
#   ./scripts/codex_session.sh --managed-dir .worktrees/codex-auto-qa --no-maintain --no-reconcile

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="codex"
BASE_BRANCH="main"
RECONCILE=true
MAINTAIN=true
TTL_HOURS="${CODEX_WORKTREE_TTL_HOURS:-24}"
MANAGED_DIR="${CODEX_WORKTREE_MANAGED_DIR:-.worktrees/codex-auto}"
ORCHESTRATOR="${CODEX_ORCHESTRATOR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            AGENT="${2:-}"
            shift 2
            ;;
        --base)
            BASE_BRANCH="${2:-}"
            shift 2
            ;;
        --no-reconcile)
            RECONCILE=false
            shift
            ;;
        --no-maintain)
            MAINTAIN=false
            shift
            ;;
        --ttl-hours)
            TTL_HOURS="${2:-24}"
            shift 2
            ;;
        --managed-dir)
            MANAGED_DIR="${2:-.worktrees/codex-auto}"
            shift 2
            ;;
        --orchestrator)
            ORCHESTRATOR="${2:-}"
            shift 2
            ;;
        --help|-h)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

ENSURE_ARGS=(ensure --agent "${AGENT}" --base "${BASE_BRANCH}" --print-path)
if ${RECONCILE}; then
    ENSURE_ARGS+=(--reconcile --strategy ff-only)
fi

if ${MAINTAIN}; then
    # Keep startup fast and non-destructive: prune stale trees but keep local branches.
    python3 "${REPO_ROOT}/scripts/codex_worktree_autopilot.py" \
        --repo "${REPO_ROOT}" \
        --managed-dir "${MANAGED_DIR}" \
        maintain \
        --base "${BASE_BRANCH}" \
        --ttl-hours "${TTL_HOURS}" \
        --no-delete-branches \
        >/dev/null 2>&1 || true
fi

WORKTREE_PATH="$(
    python3 "${REPO_ROOT}/scripts/codex_worktree_autopilot.py" \
        --repo "${REPO_ROOT}" \
        --managed-dir "${MANAGED_DIR}" \
        "${ENSURE_ARGS[@]}"
)"

cd "${WORKTREE_PATH}"
echo "Codex worktree: ${WORKTREE_PATH}"

LOCK_FILE="${WORKTREE_PATH}/.codex_session_active"
META_FILE="${WORKTREE_PATH}/.codex_session_meta.json"
LOG_FILE="${WORKTREE_PATH}/.codex_session.log"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
SESSION_ID="$(basename "${WORKTREE_PATH}")"

if [[ $# -eq 0 ]]; then
    SESSION_MODE="codex"
    SESSION_COMMAND="codex"
    SESSION_ARGS_JSON='["codex"]'
else
    SESSION_MODE="command"
    SESSION_COMMAND="$*"
    SESSION_ARGS_JSON="$(python3 - "$@" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
fi

if [[ -z "${ORCHESTRATOR}" ]]; then
    cmd_lc="${SESSION_COMMAND,,}"
    case "${cmd_lc}" in
        *gastown*|*bead*|*molecule*)
            ORCHESTRATOR="gastown"
            ;;
        *langchain*)
            ORCHESTRATOR="langchain"
            ;;
        *crewai*|*"crew ai"*)
            ORCHESTRATOR="crewai"
            ;;
        *langgraph*)
            ORCHESTRATOR="langgraph"
            ;;
        *autogen*)
            ORCHESTRATOR="autogen"
            ;;
        *openclaw*)
            ORCHESTRATOR="openclaw"
            ;;
        *nomic*)
            ORCHESTRATOR="nomic"
            ;;
        *)
            ORCHESTRATOR="generic"
            ;;
    esac
fi

printf \
    'pid=%s\nsession_id=%s\nagent=%s\nbranch=%s\nworktree_path=%s\nlog_path=%s\nmeta_path=%s\nmode=%s\norchestrator=%s\nstarted_at=%s\n' \
    "$$" \
    "${SESSION_ID}" \
    "${AGENT}" \
    "${BRANCH_NAME}" \
    "${WORKTREE_PATH}" \
    "${LOG_FILE}" \
    "${META_FILE}" \
    "${SESSION_MODE}" \
    "${ORCHESTRATOR}" \
    "${STARTED_AT}" \
    > "${LOCK_FILE}"

META_FILE="${META_FILE}" \
WORKTREE_PATH="${WORKTREE_PATH}" \
BRANCH_NAME="${BRANCH_NAME}" \
AGENT="${AGENT}" \
SHELL_PID="$$" \
SESSION_ID="${SESSION_ID}" \
LOG_FILE="${LOG_FILE}" \
SESSION_MODE="${SESSION_MODE}" \
ORCHESTRATOR="${ORCHESTRATOR}" \
SESSION_COMMAND="${SESSION_COMMAND}" \
SESSION_ARGS_JSON="${SESSION_ARGS_JSON}" \
STARTED_AT="${STARTED_AT}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

meta = {
    "pid": int(os.environ["SHELL_PID"]),
    "session_id": os.environ["SESSION_ID"],
    "agent": os.environ["AGENT"],
    "branch": os.environ["BRANCH_NAME"],
    "worktree_path": os.environ["WORKTREE_PATH"],
    "log_path": os.environ["LOG_FILE"],
    "mode": os.environ["SESSION_MODE"],
    "orchestrator": os.environ["ORCHESTRATOR"],
    "command": os.environ["SESSION_COMMAND"],
    "args": json.loads(os.environ["SESSION_ARGS_JSON"]),
    "started_at": os.environ["STARTED_AT"],
}
Path(os.environ["META_FILE"]).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

{
    echo "=== session_start ==="
    echo "started_at=${STARTED_AT}"
    echo "pid=$$"
    echo "agent=${AGENT}"
    echo "branch=${BRANCH_NAME}"
    echo "mode=${SESSION_MODE}"
    echo "orchestrator=${ORCHESTRATOR}"
    echo "command=${SESSION_COMMAND}"
} >> "${LOG_FILE}"

cleanup_lock() {
    local exit_code=$?
    local ended_at
    ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    {
        echo "=== session_end ==="
        echo "ended_at=${ended_at}"
        echo "exit_code=${exit_code}"
    } >> "${LOG_FILE}" 2>/dev/null || true

    META_FILE="${META_FILE}" \
    ENDED_AT="${ended_at}" \
    EXIT_CODE="${exit_code}" \
    python3 - <<'PY' >/dev/null 2>&1
import json
import os
from pathlib import Path

meta_path = Path(os.environ["META_FILE"])
if not meta_path.exists():
    raise SystemExit(0)
data = json.loads(meta_path.read_text(encoding="utf-8"))
data["ended_at"] = os.environ["ENDED_AT"]
data["exit_code"] = int(os.environ["EXIT_CODE"])
meta_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

    rm -f "${LOCK_FILE}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if command -v script >/dev/null 2>&1; then
    if [[ $# -eq 0 ]]; then
        script -q "${LOG_FILE}" codex
    else
        script -q "${LOG_FILE}" "$@"
    fi
    exit $?
fi

# Fallback when script(1) is unavailable.
if [[ $# -eq 0 ]]; then
    codex 2>&1 | tee -a "${LOG_FILE}"
    exit ${PIPESTATUS[0]}
fi

"$@" 2>&1 | tee -a "${LOG_FILE}"
exit ${PIPESTATUS[0]}
