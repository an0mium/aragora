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

if ! gh auth status >/dev/null 2>&1; then
  echo "$(STAMP) [codex-automation-publisher] gh auth unavailable; skipping publish pass"
  exit 0
fi

echo "$(STAMP) [codex-automation-publisher] starting publish pass"
python3 scripts/publish_codex_automation_branches.py --apply --limit 1 --max-open-prs 1 --json
echo "$(STAMP) [codex-automation-publisher] publish pass complete"
