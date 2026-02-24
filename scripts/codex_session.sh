#!/usr/bin/env bash
# Launch Codex in an auto-managed worktree.
#
# Usage:
#   ./scripts/codex_session.sh
#   ./scripts/codex_session.sh --agent codex-qa
#   ./scripts/codex_session.sh --agent codex-qa --base main -- python -m pytest tests/debate -q

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="codex"
BASE_BRANCH="main"
RECONCILE=true

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
        --help|-h)
            sed -n '1,12p' "$0"
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
    ENSURE_ARGS+=(--reconcile)
fi

WORKTREE_PATH="$(
    python3 "${REPO_ROOT}/scripts/codex_worktree_autopilot.py" \
        --repo "${REPO_ROOT}" \
        "${ENSURE_ARGS[@]}"
)"

cd "${WORKTREE_PATH}"
echo "Codex worktree: ${WORKTREE_PATH}"

if [[ $# -eq 0 ]]; then
    exec codex
fi

exec "$@"
