#!/usr/bin/env bash
# Periodic non-destructive worktree maintenance for multi-agent sessions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_BRANCH="main"
TTL_HOURS="${CODEX_WORKTREE_TTL_HOURS:-24}"
STRATEGY="merge"
KEEP_BRANCHES=true
INCLUDE_ACTIVE=false
RECONCILE_ONLY=false
declare -a MANAGED_DIRS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/worktree_maintainer.sh [options]

Options:
  --repo <path>                 Repository root (default: script parent)
  --base <branch>               Base branch to integrate from (default: main)
  --ttl-hours <n>               Stale-session TTL in hours (default: 24)
  --strategy <merge|rebase|ff-only|none>
                                Integration strategy (default: merge)
  --managed-dir <path>          Managed dir relative to repo root (repeatable)
  --delete-branches             Allow cleanup to delete local codex/* branches
  --no-delete-branches          Keep local codex/* branches during cleanup
  --include-active              Also maintain worktrees with active session lock files
  --reconcile-only             Reconcile only; skip cleanup/removal phase
  --help                        Show this help

If no --managed-dir values are provided, the script auto-discovers
.worktrees/codex-auto* directories and also checks .worktrees/codex-auto.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_ROOT="${2:-}"
            shift 2
            ;;
        --base)
            BASE_BRANCH="${2:-main}"
            shift 2
            ;;
        --ttl-hours)
            TTL_HOURS="${2:-24}"
            shift 2
            ;;
        --strategy)
            STRATEGY="${2:-merge}"
            shift 2
            ;;
        --managed-dir)
            MANAGED_DIRS+=("${2:-}")
            shift 2
            ;;
        --delete-branches)
            KEEP_BRANCHES=false
            shift
            ;;
        --no-delete-branches)
            KEEP_BRANCHES=true
            shift
            ;;
        --include-active)
            INCLUDE_ACTIVE=true
            shift
            ;;
        --reconcile-only)
            RECONCILE_ONLY=true
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

case "$STRATEGY" in
    merge|rebase|ff-only|none) ;;
    *)
        echo "Invalid strategy: $STRATEGY" >&2
        exit 2
        ;;
esac

if [[ ${#MANAGED_DIRS[@]} -eq 0 ]]; then
    # Always include the default path.
    MANAGED_DIRS+=(".worktrees/codex-auto")
    # Add all discovered codex-auto* directories.
    for abs in "${REPO_ROOT}"/.worktrees/codex-auto*; do
        [[ -d "$abs" ]] || continue
        rel=".worktrees/$(basename "$abs")"
        duplicate=false
        for existing in "${MANAGED_DIRS[@]}"; do
            if [[ "$existing" == "$rel" ]]; then
                duplicate=true
                break
            fi
        done
        if [[ "$duplicate" == false ]]; then
            MANAGED_DIRS+=("$rel")
        fi
    done
fi

echo "worktree-maintainer: repo=${REPO_ROOT} base=${BASE_BRANCH} ttl_hours=${TTL_HOURS} strategy=${STRATEGY}"

overall=0
for managed_dir in "${MANAGED_DIRS[@]}"; do
    abs_dir="${REPO_ROOT}/${managed_dir}"
    if [[ ! -d "$abs_dir" ]]; then
        continue
    fi

    if [[ "${INCLUDE_ACTIVE}" == false ]]; then
        active_locks="$(
            find "$abs_dir" -maxdepth 3 -type f -name '.codex_session_active' 2>/dev/null || true
        )"
        if [[ -n "${active_locks// }" ]]; then
            echo "worktree-maintainer: skipping ${managed_dir} (active session lock present)"
            continue
        fi
    fi

    if [[ "${RECONCILE_ONLY}" == true ]]; then
        cmd=(
            python3 "${REPO_ROOT}/scripts/codex_worktree_autopilot.py"
            --repo "${REPO_ROOT}"
            --managed-dir "${managed_dir}"
            reconcile
            --all
            --base "${BASE_BRANCH}"
            --strategy "${STRATEGY}"
            --json
        )
    else
        cmd=(
            python3 "${REPO_ROOT}/scripts/codex_worktree_autopilot.py"
            --repo "${REPO_ROOT}"
            --managed-dir "${managed_dir}"
            maintain
            --base "${BASE_BRANCH}"
            --strategy "${STRATEGY}"
            --ttl-hours "${TTL_HOURS}"
            --json
        )
        if [[ "${KEEP_BRANCHES}" == true ]]; then
            cmd+=(--no-delete-branches)
        fi
    fi

    echo "worktree-maintainer: maintaining ${managed_dir}"
    if ! "${cmd[@]}"; then
        overall=1
    fi
done

exit "${overall}"
