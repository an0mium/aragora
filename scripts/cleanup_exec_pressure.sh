#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for long-running local multi-agent sessions.
# - Kills stale pytest and gh run watch processes.
# - Optionally prunes leaked chroma-mcp helpers.

PRUNE_CHROMA=0
if [[ "${1:-}" == "--prune-chroma" ]]; then
  PRUNE_CHROMA=1
fi

kill_matching() {
  local pattern="$1"
  local pids
  pids="$(
    ps -ax -o pid=,command= \
      | grep -E "$pattern" \
      | grep -v grep \
      | awk '{print $1}' \
      | tr '\n' ' ' \
      || true
  )"
  if [[ -n "${pids// }" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
}

# 1) Stale CI/test watchers from old sessions.
kill_matching 'python -m pytest'
kill_matching 'gh run watch'

# 2) Optional: leaked Claude memory sidecars.
if [[ "$PRUNE_CHROMA" -eq 1 ]]; then
  kill_matching 'chroma-mcp --client-type persistent'
fi

echo "cleanup complete"
echo "pytest=$(ps -ax -o command | grep -E 'python -m pytest' | grep -v grep | wc -l | tr -d ' ')"
echo "gh_watch=$(ps -ax -o command | grep -E 'gh run watch' | grep -v grep | wc -l | tr -d ' ')"
echo "chroma_mcp=$(ps -ax -o command | grep -E 'chroma-mcp --client-type persistent' | grep -v grep | wc -l | tr -d ' ')"
