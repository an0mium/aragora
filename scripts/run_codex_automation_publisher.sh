#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${TMPDIR:-/tmp}/com.aragora.codex-automation-publisher.lock"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
HANDOFF_LIMIT="${ARAGORA_AUTOMATION_HANDOFF_LIMIT:-2}"
MAX_OPEN_ISSUES="${ARAGORA_AUTOMATION_MAX_OPEN_ISSUES:-16}"
BRANCH_LIMIT="${ARAGORA_AUTOMATION_BRANCH_PUBLISH_LIMIT:-1}"
MAX_OPEN_PRS="${ARAGORA_AUTOMATION_MAX_OPEN_PRS:-1}"
BRANCH_SCAN_LIMIT="${ARAGORA_AUTOMATION_BRANCH_SCAN_LIMIT:-40}"
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

echo "$(STAMP) [codex-automation-publisher] checking GitHub CLI health"
HEALTH_JSON="$(python3 scripts/github_cli_health.py --repo "${REPO_ROOT}" --json 2>/dev/null || true)"
HEALTH_READY="$(
  printf '%s' "${HEALTH_JSON}" \
    | python3 -c 'import json,sys; print("true" if json.load(sys.stdin).get("ready") else "false")' \
      2>/dev/null \
    || echo "false"
)"
if [[ "${HEALTH_READY}" != "true" ]]; then
  echo "$(STAMP) [codex-automation-publisher] GitHub unavailable; leaving automations in handoff-only mode"
  if [[ -n "${HEALTH_JSON}" ]]; then
    printf '%s\n' "${HEALTH_JSON}"
  fi
  exit 0
fi

if ! git fetch --no-write-fetch-head --prune origin '+refs/heads/*:refs/remotes/origin/*' >/dev/null 2>&1; then
  echo "$(STAMP) [codex-automation-publisher] origin refresh failed; continuing with cached refs"
fi

echo "$(STAMP) [codex-automation-publisher] starting handoff publish pass"
if python3 scripts/publish_automation_handoffs.py \
  --apply \
  --limit "${HANDOFF_LIMIT}" \
  --max-open-issues "${MAX_OPEN_ISSUES}" \
  --json; then
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass failed; continuing"
fi

echo "$(STAMP) [codex-automation-publisher] starting branch publish pass"
if python3 scripts/publish_codex_automation_branches.py \
  --base origin/main \
  --apply \
  --limit "${BRANCH_LIMIT}" \
  --max-open-prs "${MAX_OPEN_PRS}" \
  --scan-limit "${BRANCH_SCAN_LIMIT}" \
  --json; then
  echo "$(STAMP) [codex-automation-publisher] branch publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] branch publish pass failed"
fi

echo "$(STAMP) [codex-automation-publisher] publish pass complete"
