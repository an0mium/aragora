#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${TMPDIR:-/tmp}/com.aragora.codex-automation-publisher.lock"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
HANDOFF_LIMIT="${ARAGORA_AUTOMATION_HANDOFF_LIMIT:-2}"
MAX_OPEN_ISSUES="${ARAGORA_AUTOMATION_MAX_OPEN_ISSUES:-16}"
BRANCH_LIMIT="${ARAGORA_AUTOMATION_BRANCH_PUBLISH_LIMIT:-2}"
MAX_OPEN_PRS="${ARAGORA_AUTOMATION_MAX_OPEN_PRS:-12}"
BRANCH_SCAN_LIMIT="${ARAGORA_AUTOMATION_BRANCH_SCAN_LIMIT:-40}"
ALLOW_UNHEALTHY_QUEUE_PUBLISH="${ARAGORA_AUTOMATION_ALLOW_UNHEALTHY_QUEUE_PUBLISH:-false}"
export ARAGORA_AUTOMATION_MIN_FREE_GIB="${ARAGORA_AUTOMATION_MIN_FREE_GIB:-50}"
export ARAGORA_AUTOMATION_CODEX_RSS_MAX_GIB="${ARAGORA_AUTOMATION_CODEX_RSS_MAX_GIB:-25}"
export ARAGORA_AUTOMATION_SPEND_DAILY_CAP_USD="${ARAGORA_AUTOMATION_SPEND_DAILY_CAP_USD:-200}"
export ARAGORA_AUTOMATION_SPEND_WEEKLY_CAP_USD="${ARAGORA_AUTOMATION_SPEND_WEEKLY_CAP_USD:-500}"
AUTOMATION_STATE_ROOT="${ARAGORA_AUTOMATION_STATE_ROOT:-/Users/armand/Development/aragora}"
if [[ "${AUTOMATION_STATE_ROOT##*/}" == ".aragora" && -d "${AUTOMATION_STATE_ROOT}" ]]; then
  AUTOMATION_STATE_ROOT="$(cd "${AUTOMATION_STATE_ROOT}/.." && pwd)"
fi
if [[ ! -d "${AUTOMATION_STATE_ROOT}/.aragora" ]]; then
  AUTOMATION_STATE_ROOT="${REPO_ROOT}"
fi
HANDOFF_OUTBOX_DIR="${ARAGORA_AUTOMATION_OUTBOX_DIR:-${AUTOMATION_STATE_ROOT}/.aragora/automation-outbox}"
HANDOFF_RECEIPT_DIR="${ARAGORA_AUTOMATION_RECEIPT_DIR:-${AUTOMATION_STATE_ROOT}/.aragora/automation-receipts}"
GITHUB_STATUS_CACHE="${ARAGORA_AUTOMATION_GITHUB_STATUS_CACHE:-${AUTOMATION_STATE_ROOT}/.aragora/automation-github-status/latest.json}"
STAMP() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

repo_root_available() {
  [[ -d "${REPO_ROOT}" && ( -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ) ]]
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

echo "$(STAMP) [codex-automation-publisher] checking GitHub CLI health"
HEALTH_JSON="$(python3 scripts/github_cli_health.py --repo "${REPO_ROOT}" --json 2>/dev/null || true)"
if python3 scripts/cache_codex_automation_github_status.py \
  --repo "${REPO_ROOT}" \
  --output "${GITHUB_STATUS_CACHE}" \
  --outbox-dir "${HANDOFF_OUTBOX_DIR}" \
  --receipt-dir "${HANDOFF_RECEIPT_DIR}" \
  --max-open-prs "${MAX_OPEN_PRS}" \
  --max-open-issues "${MAX_OPEN_ISSUES}" \
  --json >/dev/null; then
  echo "$(STAMP) [codex-automation-publisher] wrote GitHub queue status cache: ${GITHUB_STATUS_CACHE}"
else
  echo "$(STAMP) [codex-automation-publisher] GitHub queue status cache failed; continuing"
fi
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

echo "$(STAMP) [codex-automation-publisher] starting branch publish pass"
if ! repo_root_available; then
  echo "$(STAMP) [codex-automation-publisher] repo root unavailable; skipping branch publish pass"
  echo "$(STAMP) [codex-automation-publisher] publish pass complete"
  exit 0
fi
BRANCH_PUBLISH_ARGS=(
  --repo "${REPO_ROOT}" \
  --base origin/main \
  --apply \
  --limit "${BRANCH_LIMIT}" \
  --max-open-prs "${MAX_OPEN_PRS}" \
  --scan-limit "${BRANCH_SCAN_LIMIT}" \
  --outbox-dir "${HANDOFF_OUTBOX_DIR}" \
  --json
)
if [[ "${ALLOW_UNHEALTHY_QUEUE_PUBLISH}" == "1" || "${ALLOW_UNHEALTHY_QUEUE_PUBLISH}" == "true" || "${ALLOW_UNHEALTHY_QUEUE_PUBLISH}" == "yes" ]]; then
  BRANCH_PUBLISH_ARGS+=(--allow-unhealthy-queue-publish)
fi
if python3 scripts/publish_codex_automation_branches.py "${BRANCH_PUBLISH_ARGS[@]}"; then
  echo "$(STAMP) [codex-automation-publisher] branch publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] branch publish pass failed"
fi

echo "$(STAMP) [codex-automation-publisher] starting handoff publish pass"
mkdir -p "${HANDOFF_OUTBOX_DIR}" "${HANDOFF_RECEIPT_DIR}"
if python3 scripts/publish_automation_handoffs.py \
  --apply \
  --limit "${HANDOFF_LIMIT}" \
  --max-open-issues "${MAX_OPEN_ISSUES}" \
  --outbox-dir "${HANDOFF_OUTBOX_DIR}" \
  --receipt-dir "${HANDOFF_RECEIPT_DIR}" \
  --json; then
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass complete"
else
  echo "$(STAMP) [codex-automation-publisher] handoff publish pass failed; continuing"
fi

echo "$(STAMP) [codex-automation-publisher] publish pass complete"
