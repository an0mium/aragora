#!/usr/bin/env bash
# Proof-surface freshness guardian — alert/draft-PR safety-net monitor.
#
# Purpose
#   A lightweight daily check that the B0 / TW03 benchmark-truth proof surfaces on
#   origin/main are still fresh. It is a *safety net* beside the live automation
#   fleet that normally maintains them — not a second writer. It NEVER commits to
#   main and NEVER merges.
#
# Behaviour
#   * Probes freshness on a throwaway clean origin/main checkout (read-only truth).
#   * Fresh  -> log a no-op and exit 0.
#   * Stale  -> ALERT (always, to stdout + a status file) and, best-effort, open a
#              DRAFT refresh PR. GitHub being unavailable downgrades to alert-only
#              and still exits non-zero so a wrapping cron/launchd surfaces it — the
#              alert is never lost to a degraded network (the lesson of the 127
#              github_unavailable outbox handoffs).
#
# Install (local, where gh is authenticated — do NOT run remotely):
#   crontab:  17 7 * * *  /path/to/aragora/scripts/proof_surface_guardian.sh >> ~/.aragora/proof_guardian.log 2>&1
#   or a launchd plist mirroring scripts/install_*_launchd.sh.
#
# Env overrides
#   ARAGORA_REPO            repo root (default: resolved from this script)
#   PROOF_MAX_AGE_DAYS      freshness threshold (default: 7)
#   PROOF_GUARDIAN_OPEN_PR  "1" to attempt a draft PR on stale (default: 1)
set -euo pipefail

MAX_AGE_DAYS="${PROOF_MAX_AGE_DAYS:-7}"
OPEN_PR="${PROOF_GUARDIAN_OPEN_PR:-1}"
STATUS_FILE="${PROOF_GUARDIAN_STATUS:-${HOME}/.aragora/proof_guardian_status.json}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

REPO="${ARAGORA_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$REPO"
mkdir -p "$(dirname "$STATUS_FILE")"

git fetch origin --quiet || echo "[$(ts)] WARN: git fetch failed; checking last-known origin/main"

OBS="$(mktemp -d)/proof-observer"
cleanup() { git worktree remove --force "$OBS" >/dev/null 2>&1 || true; }
trap cleanup EXIT
git worktree add --detach --quiet "$OBS" origin/main
cd "$OBS"

PROBE_JSON="$(python3 scripts/probe_proof_surface_freshness.py \
  --surfaces b0,tw03 --max-age-days "$MAX_AGE_DAYS" 2>/dev/null)" && PROBE_RC=0 || PROBE_RC=$?

if [ "$PROBE_RC" -eq 0 ]; then
  printf '{"checked_at":"%s","fresh":true}\n' "$(ts)" > "$STATUS_FILE"
  echo "[$(ts)] proof fresh — no action. ${PROBE_JSON}"
  exit 0
fi

# --- stale path: alert always, draft PR best-effort ---
echo "[$(ts)] ALERT: proof surface(s) STALE (threshold ${MAX_AGE_DAYS}d). ${PROBE_JSON}"
printf '{"checked_at":"%s","fresh":false,"probe":%s}\n' "$(ts)" "${PROBE_JSON:-null}" > "$STATUS_FILE"

if [ "$OPEN_PR" != "1" ]; then
  echo "[$(ts)] draft-PR disabled (PROOF_GUARDIAN_OPEN_PR!=1); alert only."
  exit 2
fi

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
  echo "[$(ts)] gh unavailable/unauthenticated — alert only, no draft PR (network must not lose the alert)."
  exit 2
fi

BRANCH="proof/guardian-refresh-$(date -u +%Y%m%dT%H%M%SZ)"
git checkout -q -b "$BRANCH"
# Refresh WITHOUT committing to main; the script stages a normal commit on this branch.
if bash scripts/refresh_proof_surfaces.sh --commit; then
  if git push -q -u origin "$BRANCH" \
     && gh pr create --draft --base main --head "$BRANCH" \
          --title "chore(proof): refresh stale benchmark-truth proof surface" \
          --body "Automated guardian: a proof surface crossed the ${MAX_AGE_DAYS}d freshness threshold. Draft refresh for human review — not auto-merged.

Probe: ${PROBE_JSON}" >/dev/null; then
    echo "[$(ts)] opened draft refresh PR on ${BRANCH}."
    exit 0
  fi
fi
echo "[$(ts)] refresh/PR step failed — alert recorded, manual follow-up needed."
exit 2
