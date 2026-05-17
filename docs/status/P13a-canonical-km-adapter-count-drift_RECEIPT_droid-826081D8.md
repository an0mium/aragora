# P13a-canonical-km-adapter-count-drift — Session Receipt

**Session ID:** droid-826081D8
**Agent family:** droid
**Generated:** 2026-05-17T23:25:00Z
**Base SHA:** b162fcd1e55e89ead9e438ac2b94cd1fd73f9113 (origin/main)
**Prompt:** v5

## Goal

Close the `canonical.km_adapters.count` drift identified by `scripts/publish_publication_freshness_probe.py` with the smallest correct single-doc fix.

## What shipped

- Identified the root cause: `docs/CANONICAL_GOALS.md` row "Knowledge Mound adapters | 41 registered specs / 45 adapter files" leads with the registered-specs count (41), but the canonical-metrics check at `scripts/check_canonical_metrics.py:191` regexes the FIRST integer and compares it to the file count (observed=46). So the check reads 41 vs 46 and reports +5 drift.
- Fix: swap the order — put the file count (46) first so the regex parses what it measures. Updated 45 → 46 to match the truth (already corroborated by `docs/METRICS.md`).
- Diff: one line in one doc.
- Verified the check now passes (`claimed=46 observed=46 status=pass`).
- Verified the probe verdict moved from `drift (total=4)` to `minor_drift (total=3)`.
- Opened **PR #7289** as draft; CI green (14 SUCCESS, 0 FAILURE, 0 PENDING); flipped ready; ready-flip triggered the full-suite reactivation (26 SUCCESS, 3 still PENDING at exit).

## PR / branch coordinates

- PR URL: https://github.com/synaptent/aragora/pull/7289
- Branch: `droid/P13a-canonical-km-adapter-count-drift-20260517-231510`
- Head SHA: `a53280590`
- State: OPEN, draft=false, MERGEABLE, mergeState=BLOCKED (pending review)

## Bucket classification

**Live triage at exit** (after ready-flip):
```json
{
  "bucket": "C",
  "pr_number": 7289,
  "reason": "CI pending (9 in-flight, 14/52 green)",
  "recommended_action": "DEFER"
}
```

This is Bucket C only because the ready-flip triggered new CI checks that haven't settled yet. Once those 9 in-flight checks finish, the only remaining policy disqualifier ("draft") will be gone and the classifier should re-bucket to A automatically. The Stage-2 auto-merger (P16, not yet shipped) is the right consumer.

**Manual Bucket-A check at submission time:**

- mergeable=MERGEABLE ✓
- ready ✓ (flipped this session)
- CI: 14 SUCCESS, 20 SKIPPED, 0 FAILURE, 0 PENDING **before** ready-flip; 26/3/38/0 SUCCESS/PENDING/SKIPPED/FAILURE after ready-flip — pending checks expected to clear within minutes.
- additive only (+1/-1 in 1 doc) ✓
- preflight ok ✓
- no protected file touched ✓
- no flag flip / label add ✓
- net LOC = 1 ✓
- author=`an0mium` trusted ✓
- tests-for-new-behavior: N/A — generated-data-equivalent doc refresh, no `.py`/`.ts`/`.yaml` touched (v5 carve-out)
- Tier 1 (canonical-claim row) — Tier-3/4 risk settlement N/A ✓

## Dogfood quorum (6 observers)

1. `list_active_agent_sessions.py --json --max-pr-fetch 50 --skip-codex-desktop`
   → `overlap_count=13, open_prs=10, worktrees=337`
2. `agent_bridge.py operator-snapshot --json --summary-only`
   → `active_processes=418, active_lanes=2, roles=[boss_cycle, claude_code, codex_app_server, codex_cli, factory_droid, multi_agent_dialog, worktree_inventory]`
3. `agent_bridge.py --json health` (collision substitute for missing `detect_active_lane_collisions.py`)
   → `{collisions: 0, stale_lanes: 0, stale_worktrees: 0}` — healthy
4. `publish_publication_freshness_probe.py --json`
   → `verdict=drift, total_drift=6` (this includes other independent drifts; the km_adapter check itself shows `pass` with observed=46/claimed=46)
5. `triage_open_prs.py --json` bucket totals
   → `A=0, B=0, C=10, D=0, total=10` (sibling agents may also be in-flight on different phases)
6. `gh pr view 7289 --json statusCheckRollup`
   → `{state=OPEN, draft=false, mergeable=MERGEABLE, mergeState=BLOCKED, checks: 26 SUCCESS, 3 PENDING, 38 SKIPPED, 0 FAILURE}`

## Reproducible commands

```bash
cd $(python3 scripts/codex_worktree_autopilot.py ensure --agent droid --base main --force-new --print-path | tail -1)
sed -i '' 's|Knowledge Mound adapters | 41 registered specs / 45 adapter files|Knowledge Mound adapters | 46 adapter files / 41 registered specs|' docs/CANONICAL_GOALS.md
python3 -c 'import sys; sys.path.insert(0, "scripts"); from check_canonical_metrics import _check_km_adapters_count; r = _check_km_adapters_count(); print(r.status, r.claimed, r.observed)'
# -> pass 46 46
python3 scripts/publish_publication_freshness_probe.py --json | jq '.verdict'
# -> "minor_drift" (was "drift")
```

## v5 prompt findings

- **Prompt-bug (new):** v5 lists `scripts/detect_active_lane_collisions.py` as a required observer. That script does NOT exist on main. The collision-detection functionality from PR #7288 was added to `scripts/agent_bridge.py health` instead. Workaround used this session: invoke `python3 scripts/agent_bridge.py --json health` and read `.collisions[]`. v6 should reference `agent_bridge.py health` directly.

- **Feature-detection rule worked correctly.** Phase 0.5's "if script is missing, disable the rule" path engaged automatically — I noted the missing script and used the canonical substitute without halting.

- **Triage round-trip CI-pending is correct behavior.** When a draft is flipped ready, the full check suite re-activates, briefly producing a transient Bucket-C state ("CI pending") before settling. v6 could note this is expected and not a hard-stop.

## Deferred for parallel siblings

Same as the SESSION_BRIEF list. Highest-value next claims:
- **P13d-canonical-test-definitions-count** — same pattern as P13a, also a Bucket-A docs fix.
- **P13c-stale-status-doc-refresh** — read `latest.json` `reconcile_status_docs.drift_records` to identify the stale doc; refresh.
- **P13b-model-pins-restore-frontier-exports** — touches security file; deeper investigation.
- **P15 v6 prompt** — at least one prompt-bug-confirmed to fix the collision detector path.

## Lane status

Released atomically after this receipt is committed.
