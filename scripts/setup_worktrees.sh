#!/usr/bin/env bash
# Setup git worktrees for parallel Claude Code sessions
# Usage: ./scripts/setup_worktrees.sh [num_worktrees]
#
# Each worktree gets its own directory and branch.
# Claude Code sessions run in separate worktrees — zero file conflicts.
# When done, merge branches back to main.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKTREE_BASE="${REPO_ROOT}/../aragora-worktrees"
NUM_TREES="${1:-6}"
TIMESTAMP="$(date +%Y%m%d-%H%M)"

# Track names for work assignment
TRACKS=(
    "orchestration"    # Wire HardenedOrchestrator, worktrees, commit step
    "security"         # Prompt injection defense, input sanitization
    "frontend"         # UX polish, mode discovery, landing page
    "testing"          # CI hardening, test pollution fixes, coverage
    "integration"      # OpenClaw bridge, MetaPlanner wiring
    "sdk"              # SDK parity, contract tests, documentation
    "hardening"        # Error handling, resilience, dead code
    "features"         # New features, synergy improvements
)

echo "=== Aragora Worktree Setup ==="
echo "Base: ${WORKTREE_BASE}"
echo "Trees: ${NUM_TREES}"
echo ""

mkdir -p "${WORKTREE_BASE}"

for i in $(seq 1 "${NUM_TREES}"); do
    TRACK="${TRACKS[$(( (i - 1) % ${#TRACKS[@]} ))]}"
    BRANCH="work/${TRACK}-${TIMESTAMP}"
    TREE_DIR="${WORKTREE_BASE}/${TRACK}"

    if [ -d "${TREE_DIR}" ]; then
        echo "[${i}/${NUM_TREES}] Skipping ${TRACK} (already exists)"
        continue
    fi

    echo "[${i}/${NUM_TREES}] Creating worktree: ${TRACK}"
    echo "  Branch: ${BRANCH}"
    echo "  Directory: ${TREE_DIR}"

    git -C "${REPO_ROOT}" worktree add -b "${BRANCH}" "${TREE_DIR}" main
    echo "  Done."
    echo ""
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start Claude Code sessions in each worktree:"
echo ""
for i in $(seq 1 "${NUM_TREES}"); do
    TRACK="${TRACKS[$(( (i - 1) % ${#TRACKS[@]} ))]}"
    TREE_DIR="${WORKTREE_BASE}/${TRACK}"
    [ -d "${TREE_DIR}" ] && echo "  cd ${TREE_DIR} && claude"
done
echo ""
echo "When done, merge back to main:"
echo "  ./scripts/merge_worktrees.sh"
