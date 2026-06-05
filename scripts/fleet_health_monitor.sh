#!/usr/bin/env bash
# Fleet health monitor — watches the exact inputs to `blocked_auth_failure`.
#
# Why this exists
#   The B0 benchmark's dominant failure class is `blocked_auth_failure`, raised by the
#   dispatch contract gate when a worker is missing a credential "slice" — `runner`,
#   `github_api`, or `provider` (see aragora/swarm/dispatch_contract_gate.py). When any
#   of those degrade, B0 issues cannot dispatch and product proof silently caps. The
#   runner fleet died unnoticed for 6 days (Mac exhaustion + Hetzner IP-block) and zeroed
#   in-progress graduation without any alert. This monitor watches those slices directly,
#   so degradation pages the operator BEFORE it quietly drops the benchmark.
#
# Checks (all read-only; best-effort; a hang in one never blocks the others):
#   1. runner slice    — GitHub Actions self-hosted runners online (alert on offline)
#   2. github_api slice — `gh auth status` healthy
#   3. proof freshness  — B0/TW03 surfaces within the freshness threshold
#
# Run locally (where gh is authenticated, like the rest of the fleet). NEVER remotely —
# a sandbox without gh auth would false-alarm (and is itself the github_api failure mode).
#   crontab: 23 */2 * * *  /path/to/aragora/scripts/fleet_health_monitor.sh >> ~/.aragora/fleet_health.log 2>&1
#
# Exit: 0 = all healthy; 1 = degraded (at least one slice/proof unhealthy). The status
# JSON is always written so a wrapper can render it even on the failure path.
set -uo pipefail

REPO="${ARAGORA_GH_REPO:-synaptent/aragora}"
STATUS_FILE="${FLEET_HEALTH_STATUS:-${HOME}/.aragora/fleet_health_status.json}"
MAX_AGE_DAYS="${PROOF_MAX_AGE_DAYS:-7}"
# Self-hosted runners that gate self-hosted shadow CI; offline → blocked_auth_failure risk.
# Override with a comma-separated list. Default covers the known fleet name prefixes.
WATCH_RUNNERS="${FLEET_WATCH_RUNNERS:-mac-studio-m3ultra,aragora-hetzner-cpu1,aragora-hetzner-cpu2,aragora-hetzner-cpu3}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

REPO_ROOT="${ARAGORA_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)}"
mkdir -p "$(dirname "$STATUS_FILE")"

degraded=0
alerts=()

# --- 1. runner slice ---------------------------------------------------------
runner_json='[]'
if command -v gh >/dev/null 2>&1; then
  runner_json="$(timeout 25 gh api "repos/${REPO}/actions/runners" \
    --jq '[.runners[] | {name, status}]' 2>/dev/null || echo '[]')"
fi
# Parse the runner inventory into "name<TAB>status" lines (robust to any JSON
# spacing/key order). Avoids bash-4 associative arrays so it runs under macOS
# /bin/bash 3.2 (which launchd uses).
runner_lines="$(printf '%s' "$runner_json" | python3 -c '
import json, sys
try:
    arr = json.load(sys.stdin)
except Exception:
    arr = []
for d in arr if isinstance(arr, list) else []:
    if isinstance(d, dict):
        print((d.get("name") or "") + "\t" + (d.get("status") or "unknown"))
' 2>/dev/null)"

offline_watched=""
IFS=',' read -r -a _watch <<< "$WATCH_RUNNERS"
for r in "${_watch[@]}"; do
  [ -z "$r" ] && continue
  st="$(printf '%s\n' "$runner_lines" | awk -F'\t' -v n="$r" '$1==n{print $2; exit}')"
  [ -z "$st" ] && st="unknown"
  if [ "$st" != "online" ]; then
    offline_watched="${offline_watched} ${r}(${st})"
    degraded=1
  fi
done
[ -n "$offline_watched" ] && alerts+=("RUNNER slice degraded — offline:${offline_watched} → blocked_auth_failure risk; check host power/network/billing")

# --- 2. github_api slice -----------------------------------------------------
gh_auth="unknown"
if command -v gh >/dev/null 2>&1; then
  if timeout 15 gh auth status >/dev/null 2>&1; then gh_auth="ok"; else gh_auth="UNAUTHENTICATED"; degraded=1; fi
fi
[ "$gh_auth" = "UNAUTHENTICATED" ] && alerts+=("GITHUB_API slice degraded — gh is not authenticated → workers will queue/blocked_auth_failure")

# --- 3. proof freshness ------------------------------------------------------
# Check the PUBLISHED surfaces on origin/main, not the local working tree — the local
# checkout may sit on an older branch and would otherwise false-alarm.
proof="unknown"
if [ -n "$REPO_ROOT" ] && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  timeout 20 git -C "$REPO_ROOT" fetch origin main --quiet 2>/dev/null || true
  proof="$(
    {
      git -C "$REPO_ROOT" show origin/main:docs/status/B0_BENCHMARK_TRUTH_STATUS.md 2>/dev/null
      echo '---FLEET-SURFACE-SEP---'
      git -C "$REPO_ROOT" show origin/main:docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md 2>/dev/null
    } | MAX_AGE_DAYS="$MAX_AGE_DAYS" python3 -c '
import sys, re, os, datetime as _dt
txt = sys.stdin.read()
maxage = float(os.environ.get("MAX_AGE_DAYS", "7"))
now = _dt.datetime.now(_dt.timezone.utc)
ages = []
for block in txt.split("---FLEET-SURFACE-SEP---"):
    m = re.search(r"Last updated:\s*([0-9T:+\-]+Z?)", block)
    if not m:
        continue
    s = m.group(1).strip()
    try:
        d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            d = _dt.datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
        except Exception:
            continue
    ages.append((now - d).total_seconds() / 86400.0)
print("unknown" if not ages else ("fresh" if max(ages) <= maxage else "STALE"))
' 2>/dev/null
  )"
  proof="${proof:-unknown}"
  if [ "$proof" = "STALE" ]; then
    degraded=1
    alerts+=("PROOF surfaces STALE (>${MAX_AGE_DAYS}d on origin/main) — published claim drifting from measured truth")
  fi
fi

# --- emit status -------------------------------------------------------------
{
  printf '{"checked_at":"%s","degraded":%s,"github_api":"%s","proof":"%s",' \
    "$(ts)" "$([ "$degraded" -eq 0 ] && echo false || echo true)" "$gh_auth" "$proof"
  printf '"offline_runners":"%s","alerts":%s}\n' \
    "$(echo "$offline_watched" | sed 's/^ *//')" \
    "$([ "${#alerts[@]}" -eq 0 ] && echo '[]' || printf '%s' "$(printf '%s\n' "${alerts[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))')")"
} > "$STATUS_FILE"

if [ "$degraded" -eq 0 ]; then
  echo "[$(ts)] fleet healthy — runners online, gh authed, proof ${proof}."
  exit 0
fi
echo "[$(ts)] FLEET DEGRADED:"
for a in "${alerts[@]}"; do echo "  - $a"; done
exit 1
