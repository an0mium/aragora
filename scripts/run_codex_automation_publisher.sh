#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${TMPDIR:-/tmp}/com.aragora.codex-automation-publisher.lock"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
STAMP() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(STAMP) [codex-automation-publisher] already running; exiting"
  exit 0
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "$(STAMP) [codex-automation-publisher] gh CLI not found"
  exit 1
fi

if ! git fetch --no-write-fetch-head --prune origin '+refs/heads/*:refs/remotes/origin/*' >/dev/null 2>&1; then
  echo "$(STAMP) [codex-automation-publisher] origin refresh failed; continuing with cached refs"
fi

echo "$(STAMP) [codex-automation-publisher] starting handoff publish pass"
if python3 scripts/publish_automation_handoffs.py --apply --limit 1 --max-open-issues 12 --json; then
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass failed; continuing"
fi

echo "$(STAMP) [codex-automation-publisher] starting branch publish pass"
if python3 scripts/publish_codex_automation_branches.py \
  --base origin/main \
  --apply \
  --limit 1 \
  --max-open-prs 1 \
  --json; then
  echo "$(STAMP) [codex-automation-publisher] branch publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] branch publish pass failed"
fi

echo "$(STAMP) [codex-automation-publisher] publish pass complete"
