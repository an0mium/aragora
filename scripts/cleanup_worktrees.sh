#!/usr/bin/env bash
# Clean up git worktrees created by setup_worktrees.sh.
#
# Usage:
#   ./scripts/cleanup_worktrees.sh [OPTIONS]
#
# Options:
#   --merged     Only remove worktrees whose branches are merged into main
#   --all        Remove all dev/* and work/* worktrees (prompts for confirmation)
#   --prune      Prune stale worktree entries without removing directories
#   --force      Skip confirmation prompts
#   --dry-run    Show what would be removed without doing it
#   --help       Show this help
#
# Examples:
#   ./scripts/cleanup_worktrees.sh --merged              # Safe: only merged branches
#   ./scripts/cleanup_worktrees.sh --all --force          # Remove everything, no prompt
#   ./scripts/cleanup_worktrees.sh --prune                # Clean stale git state

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_ONLY=false
ALL=false
PRUNE_ONLY=false
FORCE=false
DRY_RUN=false
REMOVED=0
SKIPPED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --merged)   MERGED_ONLY=true; shift ;;
        --all)      ALL=true; shift ;;
        --prune)    PRUNE_ONLY=true; shift ;;
        --force)    FORCE=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --help|-h)  head -20 "$0" | tail -18; exit 0 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate: must specify at least one mode
if ! ${MERGED_ONLY} && ! ${ALL} && ! ${PRUNE_ONLY}; then
    echo "Error: Specify --merged, --all, or --prune"
    echo "Run with --help for usage"
    exit 1
fi

echo "=== Aragora Worktree Cleanup ==="
if ${DRY_RUN}; then
    echo "Mode: DRY RUN"
fi
echo ""

# Prune-only mode: just clean stale git worktree entries
if ${PRUNE_ONLY}; then
    echo "Pruning stale worktree entries..."
    if ${DRY_RUN}; then
        echo "[DRY RUN] Would run: git worktree prune"
    else
        git -C "${REPO_ROOT}" worktree prune --verbose 2>&1 || true
    fi
    echo "Done."
    exit 0
fi

# Resolve worktree directory for a branch
resolve_worktree_dir() {
    local branch="$1"
    git -C "${REPO_ROOT}" worktree list --porcelain \
        | awk -v b="refs/heads/${branch}" '
            /^worktree / { dir=$2 }
            /^branch /   { if ($2 == b) print dir }
        '
}

# Get merged branches
if ${MERGED_ONLY}; then
    MERGED_BRANCHES=$(git -C "${REPO_ROOT}" branch --merged "${BASE_BRANCH:-main}" \
        | sed 's/^[* ]*//' \
        | grep -E "^(dev|work)/" || true)
fi

# Find all dev/* and work/* worktree branches
BRANCHES=$(git -C "${REPO_ROOT}" worktree list --porcelain \
    | grep "^branch" \
    | sed 's|branch refs/heads/||' \
    | grep -E "^(work|dev)/" || true)

if [ -z "${BRANCHES}" ]; then
    echo "No worktree branches found."
    exit 0
fi

BRANCH_COUNT=$(echo "${BRANCHES}" | wc -w | tr -d ' ')
echo "Found ${BRANCH_COUNT} worktree branches"

# Confirmation for --all
if ${ALL} && ! ${FORCE} && ! ${DRY_RUN}; then
    echo ""
    echo "WARNING: This will remove ALL ${BRANCH_COUNT} worktrees and their branches."
    read -r -p "Continue? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""

for BRANCH in ${BRANCHES}; do
    TREE_DIR=$(resolve_worktree_dir "${BRANCH}")
    TRACK=$(echo "${BRANCH}" | sed 's|^work/||; s|^dev/||; s|-[0-9]*$||')

    # In --merged mode, skip branches that aren't merged
    if ${MERGED_ONLY}; then
        is_merged=false
        for mb in ${MERGED_BRANCHES}; do
            if [[ "${mb}" == "${BRANCH}" ]]; then
                is_merged=true
                break
            fi
        done
        if ! ${is_merged}; then
            echo "  Skip: ${BRANCH} (not merged)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    if ${DRY_RUN}; then
        echo "  [DRY RUN] Would remove: ${BRANCH}"
        [ -n "${TREE_DIR}" ] && echo "    Worktree: ${TREE_DIR}"
        REMOVED=$((REMOVED + 1))
        continue
    fi

    echo "  Removing: ${BRANCH}"

    # Remove worktree directory
    if [ -n "${TREE_DIR}" ] && [ -d "${TREE_DIR}" ]; then
        git -C "${REPO_ROOT}" worktree remove "${TREE_DIR}" --force 2>/dev/null || {
            echo "    Warning: Could not remove worktree dir, trying rm"
            rm -rf "${TREE_DIR}" 2>/dev/null || true
        }
    fi

    # Delete branch
    if ${MERGED_ONLY}; then
        git -C "${REPO_ROOT}" branch -d "${BRANCH}" 2>/dev/null || true
    else
        git -C "${REPO_ROOT}" branch -D "${BRANCH}" 2>/dev/null || true
    fi

    REMOVED=$((REMOVED + 1))
done

# Final prune
if ! ${DRY_RUN}; then
    git -C "${REPO_ROOT}" worktree prune 2>/dev/null || true
fi

echo ""
echo "=== Cleanup Complete ==="
echo "Removed: ${REMOVED}"
echo "Skipped: ${SKIPPED}"
