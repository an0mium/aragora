# PR Triage Report — claude-C1CE7926

**Generated:** 2026-05-21 (Phase 7 of substrate-soak)
**Source:** `gh pr list --state open --limit 100` (47 open PRs; `triage_open_prs.py` errored on GraphQL 504, fallback to direct gh)

## Summary

| Bucket | Count | Action |
|--------|-------|--------|
| A (auto-mergeable) | 0 | — |
| B (needs rebase) | 14 | Operator-dispatched rebase wave recommended |
| C (blocked-by-checks) | 32 | Investigate per-PR (often dependabot or CI) |
| HOLD | 1 (#7252) | Per OPERATOR_DELEGATION_POLICY |

**Bucket A = 0** because the two CLEAN PRs (#7278, #7268) are excluded:
- **#7278** (792 LOC, CLEAN, an0mium-author) — auto-merge bot filters to `codex/*` namespace; this is on `worktree-*`. Per droid-9383F0AA Q07 finding, needs v13 extension OR manual mark-ready.
- **#7268** (1542 LOC, CLEAN, an0mium-author) — exceeds 1500 LOC cap.

**No PRs were flipped to ready-for-review by this session.** Operator decision: should #7278 be manually mark-ready'd? It would need a CI run.

## Bucket B (14 — needs rebase)

Sorted by PR number (newest first):

| PR | Title | Likely owner |
|----|-------|--------------|
| #7385 | fix(scripts): avoid Auto Off Droid launches | codex |
| #7383 | feat: add operator steering read receipts | claude/codex |
| #7382 | fix(scripts): bind stage2 subprocesses to repo root | codex |
| #7364 | Harvest bucket-a auto-merge guard stack | codex |
| #7358 | ADC-v0.2: lane-registry delegation-contract attachment | **claude (mine)** |
| #7354 | feat(scripts): cross-family agent overlap report consolidator | claude/codex |
| #7352 | feat(benchmarks): productize blocked_auth_failure rescue claim | codex |
| #7348 | R02: wake_agent.sh unified dispatch CLI | **claude (mine)** |
| #7336 | R01: contact_method + contact_payload on LaneRecord | **claude (mine)** |
| #7333 | ci(metrics): make drift advisory for ordinary PRs | unknown |
| #7328 | P53: claim-helper env-var auto-populate (Phase E) | **claude (mine)** |
| #7300 | chore(deps): update fastapi | dependabot |
| #7290 | fix(automation): harden lane collision diagnostics | claude/codex |
| #7259 | feat(scripts): bound worktree inventory runtime | codex |

**Recommendation:** dispatch a rebase wave. Five (#7336, #7348, #7349 — wait, 7349 not here, #7351, #7358, #7361) are part of the claude reach-plan + ADC chain — rebasing them advances the substrate that depends on ADC v0.1 (#7357, now merged).

## Bucket C top-15 highlights (32 total)

All are `BLOCKED` on required checks. Many likely have stale check runs that would clear on re-run; some genuinely fail.

| PR | Title | Notes |
|----|-------|-------|
| #7390 | docs(prompts): add canonical Master Fan-Out Prompt v14 | **This session's** (P90). Docs-only, BLOCKED likely on required-check-priority gate |
| #7389 | docs(status): refresh B0 benchmark truth after corpus | codex `b0-truth-refresh` active worktree |
| #7387 | docs(status): stranger-journey audit of aragora.ai | docs review needed |
| #7386 | feat(live): /sample-receipt standalone page | demo enablement |
| #7376 | docs(governance): ADC follow-on deepening packet | codex `harvest-adc-follow-on` active worktree |
| #7368 | fix(automation): expose issue-only PR handoffs in status | (recently shipped per journal — verify state) |
| #7367 | docs(governance): ADC continuation wave readiness packet | ADC chain follow-on |
| #7366 | fix(preflight): reject nested rescue publish artifacts | preflight hardening |
| #7363 | fix(automation): surface publisher-visible outbox backlog | automation maturity |
| #7362 | fix(automation): target safe outbox reconciliation | (already journal-shipped) |
| #7361 | ADC v0.4 — HMAC-SHA256 signing for Delegation Contracts | **claude/droid (mine, in-flight)** |
| #7360 | feat(scripts): ADC v0.3 — progress-ledger | **droid (in-flight, depends ADC v0.1)** |
| #7351 | R05: sweep_lane_contact_methods.py | **claude (mine)** |
| #7349 | H02: refresh_proof_surfaces.sh | **claude (mine)** |
| #7337 | H01: remove stray journal conflict marker | **claude (mine)** |

## Findings for operator

1. **Substrate freeze working:** No new orchestration phases attempted by this session. All work was either receipt-refresh (P91) or reporting (P96).
2. **My own prior PR chain (#7336, #7337, #7348, #7349, #7351, #7358, #7360, #7361)** is stuck in needs-rebase / blocked-by-checks state. A simple rebase wave would advance ~6 of them. **Recommend operator dispatch a "rebase claude-B061F80D backlog" pass.**
3. **ADC chain critical path:** v0.1 (#7357) is merged. v0.2/v0.3/v0.4 (#7358/#7360/#7361) all need rebase or CI re-run. Operator-tier review per spec — codex `harvest-adc-follow-on-deepening` worktrees are already on this.
4. **No safe Bucket A auto-merge candidates** this scan. The pipeline is healthy in the sense that nothing is exploding; bottleneck is operator-tier review of in-flight branches.
5. **gh GraphQL transient errors** (504) — `triage_open_prs.py` failed once with this; codex `salvage-github-connectivity-tokens-20260521` worktree is active and likely working on this exact issue.

## Notes

- This report is read-only output. No PR was modified, ready-flipped, or merged.
- Held PR list per `OPERATOR_DELEGATION_POLICY.md`: `#7173, #7215, #7240, #7243, #7245, #7249, #7252, #4990, #7209`. Only #7252 was in the open-PR scan.
