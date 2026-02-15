#!/usr/bin/env bash
# Create worktrees for parallel Claude Code sprint sessions.
#
# Each worktree gets a dedicated branch, allowing true file-level isolation
# between concurrent agent sessions.
#
# Usage:
#   ./scripts/create-sprint-worktrees.sh          # Create all worktrees
#   ./scripts/create-sprint-worktrees.sh cleanup   # Remove all sprint worktrees
#
# After creation, open a Claude Code session in each directory:
#   cd ~/Development/aragora-sprint-decomposer && claude
#   cd ~/Development/aragora-sprint-merge-gate && claude
#   ...

set -euo pipefail

REPO="/Users/armand/Development/aragora"
BASE="/Users/armand/Development"
TS=$(date +%Y%m%d)

declare -A TRACKS=(
  ["sprint-decomposer"]="sprint/smarter-decomposer-${TS}"
  ["sprint-merge-gate"]="sprint/test-gated-merge-${TS}"
  ["sprint-budget"]="sprint/budget-enforcement-${TS}"
  ["sprint-hardening"]="sprint/hardened-stubs-${TS}"
  ["sprint-coordinator"]="sprint/worktree-coordinator-${TS}"
  ["sprint-anomaly"]="sprint/anomaly-wire-in-${TS}"
)

cleanup() {
  echo "Cleaning up sprint worktrees..."
  for track in "${!TRACKS[@]}"; do
    local wt_path="${BASE}/aragora-${track}"
    local branch="${TRACKS[$track]}"
    if [ -d "$wt_path" ]; then
      git -C "$REPO" worktree remove "$wt_path" --force 2>/dev/null || true
      echo "  Removed worktree: $wt_path"
    fi
    if git -C "$REPO" rev-parse --verify "refs/heads/${branch}" >/dev/null 2>&1; then
      git -C "$REPO" branch -D "$branch" 2>/dev/null || true
      echo "  Deleted branch: $branch"
    fi
  done
  git -C "$REPO" worktree prune
  echo "Done."
}

create() {
  echo "Creating sprint worktrees from main..."
  echo ""

  # Ensure main is up to date
  git -C "$REPO" checkout main 2>/dev/null || true

  for track in "${!TRACKS[@]}"; do
    local wt_path="${BASE}/aragora-${track}"
    local branch="${TRACKS[$track]}"

    if [ -d "$wt_path" ]; then
      echo "  [skip] $track already exists at $wt_path"
      continue
    fi

    git -C "$REPO" worktree add -b "$branch" "$wt_path" main
    echo "  [created] $track -> $wt_path (branch: $branch)"
  done

  echo ""
  echo "Worktree list:"
  git -C "$REPO" worktree list
  echo ""
  echo "Session assignments:"
  echo "  1. aragora-sprint-decomposer  -> TF-IDF goal decomposition"
  echo "  2. aragora-sprint-merge-gate  -> Test-gated merge pipeline"
  echo "  3. aragora-sprint-budget      -> Budget enforcement"
  echo "  4. aragora-sprint-hardening   -> Hardened orchestrator stubs"
  echo "  5. aragora-sprint-coordinator -> Sprint coordinator script"
  echo "  6. aragora-sprint-anomaly     -> AnomalyDetector wiring"
  echo ""
  echo "Open a Claude Code session in each: cd <path> && claude"
}

case "${1:-create}" in
  cleanup|clean|remove)
    cleanup
    ;;
  create|setup)
    create
    ;;
  *)
    echo "Usage: $0 [create|cleanup]"
    exit 1
    ;;
esac
