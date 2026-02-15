#!/usr/bin/env bash
# Merge completed worktree branches back to main with test gates.
#
# Usage:
#   ./scripts/merge_worktrees.sh [OPTIONS]
#
# Options:
#   --dry-run   Show what would happen without merging
#   --force     Merge even if conflict check flags warnings
#   --rebase    Rebase branches onto main before merging (reduces drift)
#   --status    Just show worktree status, don't merge anything
#   --skip-tests  Skip test gate (use when tests already validated)
#
# For each worktree branch:
#   1. Check commits ahead of main
#   2. Optionally rebase onto main (--rebase)
#   3. Check for merge conflicts
#   4. Run tests in the worktree (unless --skip-tests)
#   5. If tests pass, merge to main with --no-ff
#   6. Clean up worktree and branch after successful merge

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKTREE_BASE="${REPO_ROOT}/../aragora-worktrees"
DRY_RUN=false
FORCE=false
REBASE=false
STATUS_ONLY=false
SKIP_TESTS=false
MERGED=0
FAILED=0
SKIPPED=0

for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --force)       FORCE=true ;;
        --rebase)      REBASE=true ;;
        --status)      STATUS_ONLY=true ;;
        --skip-tests)  SKIP_TESTS=true ;;
        --help|-h)
            head -15 "$0" | tail -13
            exit 0
            ;;
    esac
done

echo "=== Aragora Worktree Merge ==="
if ${STATUS_ONLY}; then
    echo "Mode: STATUS"
elif ${DRY_RUN}; then
    echo "Mode: DRY RUN"
else
    echo "Mode: LIVE$(${REBASE} && echo ' +REBASE')$(${SKIP_TESTS} && echo ' +SKIP_TESTS')"
fi
echo ""

# Find all worktree branches (both work/ and dev/ prefixes)
BRANCHES=$(git -C "${REPO_ROOT}" worktree list --porcelain \
    | grep "^branch" \
    | sed 's|branch refs/heads/||' \
    | grep -E "^(work|dev)/" || true)

if [ -z "${BRANCHES}" ]; then
    echo "No worktree branches found (looking for work/* and dev/* branches)."
    echo ""
    echo "To set up worktrees: ./scripts/setup_worktrees.sh"
    exit 0
fi

# Resolve worktree directory for a branch from git's own tracking
resolve_worktree_dir() {
    local branch="$1"
    git -C "${REPO_ROOT}" worktree list --porcelain \
        | awk -v b="refs/heads/${branch}" '
            /^worktree / { dir=$2 }
            /^branch /   { if ($2 == b) print dir }
        '
}

for BRANCH in ${BRANCHES}; do
    # Derive track name from branch (strip prefix and timestamp suffix)
    TRACK=$(echo "${BRANCH}" | sed 's|^work/||; s|^dev/||; s|-[0-9]*-[0-9]*$||; s|-[0-9]\{14\}$||')
    TREE_DIR=$(resolve_worktree_dir "${BRANCH}")

    echo "--- ${BRANCH} ---"
    if [ -n "${TREE_DIR}" ]; then
        echo "  Worktree: ${TREE_DIR}"
    fi

    # Check if branch has commits ahead of main
    AHEAD=$(git -C "${REPO_ROOT}" rev-list main.."${BRANCH}" --count 2>/dev/null || echo "0")
    if [ "${AHEAD}" = "0" ]; then
        echo "  No new commits. Skipping."
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi

    # Show changed files summary
    CHANGED=$(git -C "${REPO_ROOT}" diff --stat main..."${BRANCH}" --stat-count=5 2>/dev/null || true)
    echo "  ${AHEAD} commits ahead of main"
    if [ -n "${CHANGED}" ]; then
        echo "  ${CHANGED}" | sed 's/^/  /'
    fi

    # Status mode: just show info, don't merge
    if ${STATUS_ONLY}; then
        echo ""
        continue
    fi

    # Rebase onto main before merging (reduces drift)
    if ${REBASE} && [ -n "${TREE_DIR}" ] && [ -d "${TREE_DIR}" ] && ! ${DRY_RUN}; then
        echo "  Rebasing onto main..."
        if git -C "${TREE_DIR}" rebase main 2>&1 | tail -1; then
            AHEAD=$(git -C "${REPO_ROOT}" rev-list main.."${BRANCH}" --count 2>/dev/null || echo "0")
            echo "  Rebased (${AHEAD} commits ahead after rebase)"
        else
            echo "  Rebase FAILED — aborting rebase, skipping merge"
            git -C "${TREE_DIR}" rebase --abort 2>/dev/null || true
            FAILED=$((FAILED + 1))
            echo ""
            continue
        fi
    fi

    # Check for conflicts using merge-tree (modern git >=2.38 syntax)
    MERGE_BASE=$(git -C "${REPO_ROOT}" merge-base main "${BRANCH}" 2>/dev/null || true)
    if [ -n "${MERGE_BASE}" ]; then
        CONFLICT_OUTPUT=$(git -C "${REPO_ROOT}" merge-tree "${MERGE_BASE}" main "${BRANCH}" 2>&1 || true)
        if echo "${CONFLICT_OUTPUT}" | grep -qi "conflict"; then
            echo "  WARNING: Potential merge conflicts detected"
            if ! ${FORCE}; then
                echo "  Skipping (use --force to merge anyway)"
                SKIPPED=$((SKIPPED + 1))
                echo ""
                continue
            fi
        fi
    fi

    # Run tests in worktree (subset for speed)
    if ! ${SKIP_TESTS} && [ -n "${TREE_DIR}" ] && [ -d "${TREE_DIR}" ] && ! ${DRY_RUN}; then
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
            echo ""
            continue
        fi
    fi

    # Merge
    if ${DRY_RUN}; then
        echo "  [DRY RUN] Would merge ${BRANCH} into main (${AHEAD} commits)"
    else
        echo "  Merging into main..."
        git -C "${REPO_ROOT}" checkout main --quiet
        if git -C "${REPO_ROOT}" merge --no-ff "${BRANCH}" -m "Merge ${TRACK} worktree (${AHEAD} commits)"; then
            echo "  Merged successfully"
            MERGED=$((MERGED + 1))

            # Clean up worktree and branch
            echo "  Cleaning up..."
            if [ -n "${TREE_DIR}" ] && [ -d "${TREE_DIR}" ]; then
                git -C "${REPO_ROOT}" worktree remove "${TREE_DIR}" --force 2>/dev/null || true
            fi
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
if ${STATUS_ONLY}; then
    echo "Branches: $(echo "${BRANCHES}" | wc -w | tr -d ' ')"
else
    echo "Merged:  ${MERGED}"
    echo "Failed:  ${FAILED}"
    echo "Skipped: ${SKIPPED}"
fi

if [ "${FAILED}" -gt 0 ]; then
    echo ""
    echo "Failed branches need manual resolution."
    echo "Use: git merge <branch-name>"
    exit 1
fi
