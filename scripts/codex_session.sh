#!/usr/bin/env bash
# Launch Codex in an auto-managed worktree.
#
# Usage:
#   ./scripts/codex_session.sh
#   ./scripts/codex_session.sh --agent codex-qa
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
    ENSURE_ARGS+=(--reconcile --strategy merge)
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
printf 'pid=%s\nagent=%s\nstarted_at=%s\n' "$$" "${AGENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_FILE}"
cleanup_lock() {
    rm -f "${LOCK_FILE}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if [[ $# -eq 0 ]]; then
    codex
    exit $?
fi

"$@"
