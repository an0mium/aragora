# Queue Drain Final Tally + Per-PR Recommendations

**Session:** `claude-F1CD271D` (handoff/status doc; no merges in this doc-writing step; one no-op superseded PR close recorded below)
**Window covered:** 2026-05-21T23:53Z (#7423 governance unblock landing) → 2026-05-22T17:40Z
**Queue state snapshot:** OPEN 38 (= DRAFT 33 + READY 5) / SESSION-CLOSED 4 (intentional closes recorded by this Claude arc, not part of OPEN)
**Postscript as of 2026-05-23T19:07Z:** this file is a historical handoff snapshot. Later live state resolved #7278, #7295, #7433, and #7434; branch protection now requires `aragora-merge-quorum`.

## Headline

The **mechanical bounded drain is exhausted**. 28 PRs merged + 4 PRs closed (3 patch-equivalent auto-close + 1 superseded close) by this Claude session arc; ~7+ more landed via other paths (operator manual, codex automations, other agents). "Closed" here means resolved without merge, while OPEN remains the live queue tally above. What remains is structural debt requiring case-by-case manual work or operator-tier action.

## Merges this Claude arc (28)

In chronological order:

`#7423 #7337 #7411 #7414 #7396 #7397 #7398 #7389 #7390 #7387 #7392 #7427 #7330 #7366 #7335 #7368 #7293 #7327 #7332 #7349 #7362 #7251 #7386 #7430 #7429 #7431 #7432 #7351`

Plus: **#7416, #7417, #7418** (patch-equivalent auto-close via rebase force-push) and **#7410** (closed as superseded by main's `test.yml` matrix refactor). These four are the SESSION-CLOSED count above and are separate from the 28 merged PR numbers.

## Remaining 38 open PRs — categorized + recommendation

### Active-owner — DO NOT TOUCH (3)

| PR | Branch | Recommendation |
|----|--------|----------------|
| **#7407** | codex/security-gate-product-audit-policy | Q59 owns it; let owner finish or release lane |
| **#7425** | codex/full-autonomy-control-plane | Q51 / Q52 own; let owner finish |
| **#7295** | dependabot/npm_and_yarn/.../bundle-analyzer | Historical row; merged after this snapshot in `6991328374607b54d4e0054b78fe0e130d463833` |

### ADC (Aragora Delegation Contract) chain (5)

These are governance-coupled, sequential, and need operator review per spec.

| PR | LOC | Recommendation |
|----|----:|----------------|
| **#7358** | 2635 | ADC-v0.2 lane-registry attachment. **Dirty**. Need rebase + operator review per Tier-4 spec. |
| **#7360** | 3318 | ADC-v0.3 progress-ledger. Clean. Operator-tier review. |
| **#7361** | 1262 | ADC-v0.4 HMAC signing. Clean. Operator-tier review. |
| **#7367** | 690 | ADC continuation wave readiness packet. Clean. Operator-tier review. |
| **#7376** | 1835 | ADC follow-on deepening packet. Clean. Operator-tier review. |

### vision-incubator/* (4) — all Tier 3, `requires_human_risk_settlement=true`

| PR | LOC | Surface | Recommendation |
|----|----:|---------|----------------|
| **#7262** | 373 | aragora/reputation A2A endpoint | Operator-tier risk settlement, then merge |
| **#7291** | 264 | aragora/markets repo_guard | Operator-tier risk settlement, then merge |
| **#7276** | 290 | aragora/reputation stale_gate | Operator-tier risk settlement, then merge |
| **#7319** | 323 | aragora/reputation DisputeWindowGate | Operator-tier risk settlement, then merge |

### Skip-known (3)

| PR | LOC | Reason |
|----|----:|--------|
| **#7409** | 43 | docs(ci) but advisory check perma-cancelled — investigate or close |
| **#7413** | 5 | Tier 4 workflow policy mod — operator preapproval required |
| **#7426** | 295 | vision-incubator Tier 3 (mis-categorized earlier; AGT-01 epistemic-graph tests) |

### Dependabot (1)

| PR | LOC | Recommendation |
|----|----:|----------------|
| **#7300** | 14 | chore(deps): fastapi update — UNSTABLE. Standard Dependabot handling; auto-merge if checks pass |

### Dirty / needs-rebase (18) — all unowned, all real-conflict (not patch-equivalent)

Sorted by ascending LOC. Recommendation is the same for all: **rebase onto current main + resolve conflict + push**. The smaller ones may have simple conflicts; larger ones are higher-risk.

| PR | LOC | Branch | Title (truncated) |
|----|----:|--------|-------------------|
| **#7408** | 24 | codex/b2-guard-expansion-criteria | docs(status): define B2 guard expansion (323-line journal conflict — pattern-resolvable) |
| **#7382** | 71 | codex/stage2-subprocess-cwd-hardening | fix(scripts): bind stage2 subprocesses to repo root |
| **#7363** | 103 | codex/audit-publisher-outbox-count | fix(automation): surface publisher-visible outbox backlog |
| **#7419** | 129 | droid/Q10-dependabot-triage | docs(status): Q10 dependabot triage receipt (journal append — pattern-resolvable) |
| **#7290** | 131 | codex/lane-collision-hardening-followup | fix(automation): harden lane collision diagnostics |
| **#7259** | 234 | codex/worktree-inventory-runtime-budget | feat(scripts): bound worktree inventory runtime |
| **#7328** | 291 | claude/P53-claim-helper-env-var-auto-populate | P53: claim-helper env-var auto-populate (Phase E) |
| **#7415** | 350 | worktree-harvest-and-recovery | docs(status): non-author review packets |
| **#7385** | 364 | codex/droid-auto-guard | fix(scripts): avoid Auto Off Droid launches |
| **#7420** | 381 | codex/tmux-launcher-metadata-helper | fix(automation): avoid tmux prompt heredoc hangs |
| **#7336** | 422 | claude/R01-reach-plan-contact-method-field | R01: contact_method + contact_payload on LaneRecord |
| **#7333** | 468 | codex/metrics-drift-scope-aware | ci(metrics): make drift advisory for ordinary PRs |
| **#7352** | 488 | codex/droid-20260519-042102 | feat(benchmarks): productize blocked_auth_failure rescue |
| **#7348** | 582 | claude/R02-wake-agent-cli | R02: wake_agent.sh unified dispatch CLI |
| **#7383** | 783 | codex/operator-steering-read-receipts-clean | feat: add operator steering read receipts |
| **#7354** | 1572 | droid/P75-agent-overlap-report | feat(scripts): cross-family agent overlap report consolidator |
| **#7422** | 2212 | codex/salvage-eu-ai-act-claude-c1ce7926 | docs(compliance): preserve EU AI Act artifacts. Do not close from this summary alone: first diff #7422 against merged #7392 and confirm every unique compliance artifact is already on main or intentionally obsolete. |
| **#7364** | 2698 | codex/harvest-bucket-a-automerge | Harvest bucket-a auto-merge guard stack |

### Historical: required-check MISSING on #7278 (resolved after this snapshot)

| PR | Recommendation |
|----|----------------|
| **#7278** | Historical snapshot: `mergeable=MERGEABLE, ms=BLOCKED` because required check contexts were missing/cancelled at the time of this handoff. Later live state: `aragora-merge-quorum` was rerun successfully and #7278 merged by normal protected squash at merge commit `afa7236da4603c715d38b911da74411ace3fb038`. Do not treat the older #7278 unblock advice below as current. |

### Unstable (1)

| PR | Recommendation |
|----|----------------|
| **#7391** | docs(compliance) EU AI Act artifact, head `0855c00895`. `ms=UNSTABLE` because `aragora-merge-quorum` returned FAILURE (gate functional, model signals missing for this PR). Either close as duplicate of #7392 (already merged) OR wait for signal pipeline + retry. |

### Other CLEAN drafts (2) — historical, now merged

These appeared since pass 11 — opened by other agents/sessions. Both merged after this snapshot; do not attempt them from this handoff.

| PR | LOC | Branch | Title |
|----|----:|--------|-------|
| **#7434** | 97 | codex/merge-packet-stale-check-accounting | Historical row; merged after this snapshot in `0ec7464d80d340184a6b1988880ee5d45ac085f4` |
| **#7433** | 226 | codex/reconcile-merged-target-pr-receipts | Historical row; merged after this snapshot in `ada4dfcf2a7d3fe1c79965e89f9e87af8a6b4ec1` |

Both were `CLEAN-draft` in this snapshot and later merged. Re-verify current branch protection and `aragora-merge-quorum` before any future drain pass.

## Historical investigation: required-check MISSING on #7278 (Option 2)

### Historical hypothesis (superseded by later live merge)

This section records the investigation state during the queue-drain window. It is not a current assertion that #7278 is blocked: after this snapshot, `aragora-merge-quorum` was rerun, passed, and #7278 merged normally.

The 5 BP-required check names (`lint`, `typecheck`, `sdk-parity`, `Generate & Validate`, `TypeScript SDK Type Check`) map to these workflow files/jobs on the current branch:

| Required check | Workflow file | Required status job | Work job / scope gate |
|----------------|---------------|---------------------|-----------------------|
| `lint` | `.github/workflows/lint.yml` | `lint` | `lint-run` gated by `changes` / `.github/actions/pr-scope-classifier` |
| `typecheck` | `.github/workflows/lint.yml` | `typecheck` | `typecheck-run` gated by `changes` / `.github/actions/pr-scope-classifier` |
| `sdk-parity` | `.github/workflows/sdk-parity.yml` | `sdk-parity` | `sdk-parity-run` gated by `changes` / `.github/actions/pr-scope-classifier` |
| `Generate & Validate` | `.github/workflows/openapi.yml` | `generate` with `name: Generate & Validate` | `generate-run` / OpenAPI scope |
| `TypeScript SDK Type Check` | `.github/workflows/sdk-test.yml` | `typescript-sdk` with `name: TypeScript SDK Type Check` | `typescript-sdk-run` gated by `changes` |

The workflow files above currently use, or previously used on the PR head may use, a two-job pattern:

```yaml
jobs:
  changes:
    outputs:
      python: ${{ steps.scope.outputs.lint_python }}
  lint-run:
    needs: changes
    if: needs.changes.outputs.python == 'true'
    name: lint        # status-context name registered in BP
```

The `changes` job uses dorny/paths-filter to detect which file types changed. The downstream `*-run` job's `if:` skips when no relevant paths changed.

For a frontend-only PR like #7278 (only `aragora/live/**` changed):
- `changes` runs, reports `python=false`
- `lint-run` job is SKIPPED via `if:` evaluating false
- GitHub Actions records the SKIP, but the status context `lint` may not register against the PR's commit at all — different from a "skipped with conclusion=skipped" status

For previously-merged frontend/docs-only PRs (#7327, #7386), the same required check names did register all 5 contexts. The historical next step was to re-verify the exact workflow definitions on #7278's then-current head before changing branch protection or pushing a no-op; the table above is the concrete path list that was used for that investigation.

### Why #7278 appeared stuck during the snapshot

Two compounding factors:
1. **Branch head may not match current main's workflow file:** even after rebase, the PR's HEAD can use an older workflow version if the branch was not refreshed after the gate fixes. `pull_request` events use the workflow file from the HEAD branch.
2. **Paths-filter skip without status registration:** the skipped `*-run` jobs don't always report a `lint`/`typecheck`/etc. status context against the PR's commit. BP perpetually waits for these contexts.

### Recommended fix (operator-tier, NOT this session)

Three options in order of cleanliness:

**Option A — Make required checks "skip-as-success" via workflow-level always-report:** Update each of the 5 workflows so the `*-run` job has `if: always()` and an early-return-success step when no relevant changes. The job ALWAYS registers a status; reports success when skipped via early return.

**Option B — Remove `paths:` skip from required jobs:** Let each required job always run on every PR. Adds CI cost but eliminates the missing-check problem.

**Option C — Adjust branch protection:** Mark the 5 required checks as "not required when skipped" in GitHub branch-protection settings — only available in newer rulesets, not in legacy branch-protection rules.

Option A is the cleanest and aligns with `MERGE_GATE_RECONCILIATION.md`'s intent (status checks are the authoritative gate).

### What this meant for #7278 during the snapshot

At the time of this handoff, #7278 appeared unable to merge via the normal squash path. That is no longer current: #7278 later merged normally at `afa7236da4603c715d38b911da74411ace3fb038` after `aragora-merge-quorum` was rerun successfully. The historical unblock options considered then were:

1. **Close + re-open** the PR (sometimes triggers a full workflow re-evaluation against current main's workflow file). Low-cost retry.
2. **Push a no-op commit** (empty commit on the branch) to force fresh workflow runs at the then-current head. Higher-friction but reliable.
3. **Operator admin-merge** with `gh pr merge 7278 --squash --admin` (BP allows admin merge despite missing required checks IF `enforce_admins=false`, but current BP has `enforce_admins=true` so this won't work either).
4. **Operator temporarily flips `enforce_admins=false`** → admin-squash-merge → flip back. Audit-logged emergency stop.

This historical analysis should not be used as an instruction to mutate branch protection or reopen #7278.

## What landed by other paths during this arc

Approximate count from `git log origin/main` commits during the window:
- 5 Dependabot bumps
- ~7 other PRs (operator-merged or other-agent-merged)
- Total ~12+ landings outside my session

Combined, the queue went from ~51 open at the start of my arc to 38 now — net **-13 over 18 hours**.

## Branch protection snapshot

Historical state during this Claude arc:
```
approvals=0, code_owners=false, enforce_admins=true
required_checks: ["lint","typecheck","sdk-parity","Generate & Validate","TypeScript SDK Type Check"]
aragora-merge-quorum: NOT in required list (gate workflow exists + functional, but not enforced)
```

Current live update as of 2026-05-23T19:07Z:

```
approvals=0, code_owners=false, enforce_admins=true
required_checks: ["lint","typecheck","sdk-parity","Generate & Validate","TypeScript SDK Type Check","aragora-merge-quorum"]
```

## `aragora-merge-quorum` workflow health

Confirmed functional this arc:
- ≥4 `success` verdicts on real PRs (including failure-path branch entered)
- ≥2 `failure` verdicts (PR #7295 logged `Tier 1 | status=repair_or_wait | verdict=not_ready_for_settlement`)
- ~11 `cancelled` per 60-min window (PRs merged before workflow finishes — race condition, not gate failure)

**Historical recommendation:** during the window above, keep `aragora-merge-quorum` non-required until the model-signal pipeline is wired to produce ≥1 signal per PR routinely. Current live state supersedes this: branch protection now requires `aragora-merge-quorum`, so future PRs must treat that check as a hard gate.

## Recommended operator next actions (priority order)

1. **Do not re-drain #7433/#7434 from this handoff**; both already merged after the snapshot.
2. **Settle remaining low-risk PRs only from live `merge-packet` truth** under the current required `aragora-merge-quorum` branch-protection rule.
3. **Settle the Dependabot #7300** only if live checks and merge-packet policy pass.
4. **Investigate superseded large dirty PRs before any close** — for #7422, perform a content diff against merged #7392 and confirm no unique compliance artifact would be lost; for #7364, verify whether the auto-merge guard stack is already on main.
5. **Operator-tier rebase wave on the dirty PRs**, smallest first. Dispatch a Codex session per PR with the prompt "rebase + resolve conflicts; merge if green; close if superseded."
6. **Historical #7278 required-check issue:** resolved after this snapshot when `aragora-merge-quorum` passed and #7278 merged. For future PRs, treat `aragora-merge-quorum` as required and rerun/repair it rather than assuming it is advisory.
7. **Resolve ADC chain** (#7358-#7376) — operator-tier governance review.
8. **Resolve vision-incubator/* Tier 3 PRs** (#7262, #7276, #7291, #7319) — operator risk settlement.
9. **#7410 superseded close** ✅ already recorded for this Claude arc; do not count it as a merge.

## Total session impact

- **28 merges + 4 non-merge closes** (3 patch-equivalent auto-close + 1 superseded close) = **32 PRs resolved** by Claude sessions
- Queue: 51+ → 38 (net -13 over ~18h with concurrent operator/agent traffic)
- Structural unjam (PR #7423) shipped + functional gate workflow on main
- No protected files modified, no `--admin` bypasses
- 1 process / 1 admin-bridge merge (for #7423 bootstrap, operator-authorized)

## Doc paths written this session

- `docs/status/QUEUE_DRAIN_FINAL_claude-F1CD271D.md` (this file)
- (Optional) Future: separate `REQUIRED_CHECK_MISSING_ANALYSIS.md` if Option 2 deserves standalone treatment

## Non-merge closures recorded this Claude arc

- **#7410** — `gh pr close 7410 --comment "Superseded by main's test.yml matrix refactor — debate shard now has timeout: 60. Closing as no-op."` ✅
