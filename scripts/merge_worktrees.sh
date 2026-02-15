#!/usr/bin/env bash
# Merge completed worktree branches back to main
# Usage: ./scripts/merge_worktrees.sh [--dry-run] [--force]
#
# For each worktree branch:
#   1. Run tests in the worktree
#   2. If tests pass, merge to main (fast-forward if possible)
#   3. If tests fail, skip and report
#   4. Clean up worktree after successful merge

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKTREE_BASE="${REPO_ROOT}/../aragora-worktrees"
DRY_RUN=false
FORCE=false
MERGED=0
FAILED=0
SKIPPED=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
    esac
done

echo "=== Aragora Worktree Merge ==="
echo "Mode: $(${DRY_RUN} && echo 'DRY RUN' || echo 'LIVE')"
echo ""

# Get all worktree branches
BRANCHES=$(git -C "${REPO_ROOT}" worktree list --porcelain | grep "^branch" | sed 's|branch refs/heads/||' | grep "^work/" || true)

if [ -z "${BRANCHES}" ]; then
    echo "No worktree branches found."
    exit 0
fi

for BRANCH in ${BRANCHES}; do
    TRACK=$(echo "${BRANCH}" | sed 's|work/||; s|-[0-9]*-[0-9]*$||')
    TREE_DIR="${WORKTREE_BASE}/${TRACK}"

    echo "--- ${BRANCH} ---"

    # Check if branch has commits ahead of main
    AHEAD=$(git -C "${REPO_ROOT}" rev-list main.."${BRANCH}" --count 2>/dev/null || echo "0")
    if [ "${AHEAD}" = "0" ]; then
        echo "  No new commits. Skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "  ${AHEAD} commits ahead of main"

    # Check for conflicts
    if ! git -C "${REPO_ROOT}" merge-tree "$(git -C "${REPO_ROOT}" merge-base main "${BRANCH}")" main "${BRANCH}" > /dev/null 2>&1; then
        echo "  WARNING: Potential merge conflicts detected"
        if ! ${FORCE}; then
            echo "  Skipping (use --force to merge anyway)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    # Run tests in worktree (subset for speed)
    if [ -d "${TREE_DIR}" ] && ! ${DRY_RUN}; then
        echo "  Running tests..."
        if python -m pytest "${TREE_DIR}/tests/" -x -q --timeout=60 -p no:randomly \
            --ignore="${TREE_DIR}/tests/connectors" \
            --ignore="${TREE_DIR}/tests/integration" \
            --ignore="${TREE_DIR}/tests/benchmarks" \
            --ignore="${TREE_DIR}/tests/performance" \
            -k "not test_load" 2>&1 | tail -3; then
            echo "  Tests PASSED"
        else
            echo "  Tests FAILED — skipping merge"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Merge
    if ${DRY_RUN}; then
        echo "  [DRY RUN] Would merge ${BRANCH} into main"
    else
        echo "  Merging into main..."
        git -C "${REPO_ROOT}" checkout main
        if git -C "${REPO_ROOT}" merge --no-ff "${BRANCH}" -m "Merge ${TRACK} worktree (${AHEAD} commits)"; then
            echo "  Merged successfully"
            MERGED=$((MERGED + 1))

            # Clean up worktree
            echo "  Cleaning up worktree..."
            git -C "${REPO_ROOT}" worktree remove "${TREE_DIR}" --force 2>/dev/null || true
            git -C "${REPO_ROOT}" branch -d "${BRANCH}" 2>/dev/null || true
        else
            echo "  Merge FAILED (conflicts)"
            git -C "${REPO_ROOT}" merge --abort
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
done

echo ""
echo "=== Results ==="
echo "Merged:  ${MERGED}"
echo "Failed:  ${FAILED}"
echo "Skipped: ${SKIPPED}"

if [ "${FAILED}" -gt 0 ]; then
    echo ""
    echo "Failed branches need manual conflict resolution."
    echo "Use: git merge work/<track>-<timestamp>"
    exit 1
fi
