#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${TMPDIR:-/tmp}/com.aragora.codex-pr-merge-shepherd.lock"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

STAMP() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(STAMP) [codex-pr-merge-shepherd] already running; exiting"
  exit 0
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "$(STAMP) [codex-pr-merge-shepherd] gh CLI not found"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "$(STAMP) [codex-pr-merge-shepherd] gh auth unavailable; skipping merge pass"
  exit 0
fi

echo "$(STAMP) [codex-pr-merge-shepherd] starting merge pass"
python3 scripts/merge_codex_automation_prs.py --apply --limit 5 --json
echo "$(STAMP) [codex-pr-merge-shepherd] merge pass complete"
