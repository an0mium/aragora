# Settlement Queue Partition Ledger - 2026-07-08

Generated: 2026-07-08T15:05Z

Base: `origin/main` at `2caed5d61fa0f80b632a86ab7971a9bb884b0b8f`

Purpose: provide a live, reproducible partition of the open non-draft settlement queue after the #9001 merge so conductor cycles can drain one safe item without re-scanning the same blocked lanes.

## Main Context

Branch protection currently requires:

- `lint`
- `typecheck`
- `sdk-parity`
- `Generate & Validate`
- `TypeScript SDK Type Check`
- `aragora-merge-quorum`

Exact-head check-runs observed for `2caed5d61fa0f80b632a86ab7971a9bb884b0b8f`:

- `lint`: success
- `typecheck`: success
- `sdk-parity`: success
- `Generate & Validate`: success
- `TypeScript SDK Type Check`: success
- `aragora-merge-quorum`: skipped for main

The shared root checkout was not used for edits. It was on `codex/required-context-main-push-policy-20260707` with pre-existing modified files, so this ledger was generated from an isolated worktree.

## Inputs

Raw helper outputs for this run were written outside the repo:

- `/tmp/aragora_queue_ledger_20260708T145542Z/non_draft_prs_final.json`
- `/tmp/aragora_queue_ledger_20260708T145542Z/steward-<pr>.json`
- `/tmp/aragora_queue_ledger_20260708T145542Z/owner-<pr>.json`
- `/tmp/aragora_queue_ledger_20260708T145542Z/packet-<pr>.json` for targeted quorum-blocked PRs

Regeneration sketch:

```bash
git fetch origin --prune
gh pr list --repo synaptent/aragora --state open --draft=false --limit 200 \
  --json number,title,headRefName,headRefOid,updatedAt,mergeStateStatus,isDraft,author
for pr in $(gh pr list --repo synaptent/aragora --state open --draft=false --limit 200 --json number --jq '.[].number'); do
  timeout 180 python3 scripts/settle_one_pr.py --pr "$pr" --json > "/tmp/steward-$pr.json"
  timeout 45 python3 scripts/read_operator_steering.py --pr "$pr" --json --no-receipt > "/tmp/owner-$pr.json" || true
done
```

## Rubric

Each open non-draft PR is placed in exactly one live bucket, using this precedence:

1. Active-owned lane if owner/steering metadata resolves a current lane for the PR.
2. Parked at current head if repo-visible comments record repeat blocker or current-head dry-run dissent for the exact live head.
3. Tier 3/4 or human-risk if helpers require human-risk settlement or classify the PR at Tier 3+.
4. Policy-excluded Dependabot when it is a dependency bot PR not otherwise already caught by a stronger bucket.
5. Dirty/conflicting if merge metadata or policy exclusions report dirty/conflicting state.
6. Autonomous evidence candidate if it is Tier 0-2, non-human-risk, unowned, non-draft, and blocked only by merge-quorum/model evidence.
7. Transport-limited if helper data is unavailable.

Unresolved `CHANGES-REQUESTED`, `Verdict: CHANGES-REQUESTED`, `[P0]`, `[P1]`, or concrete `[P2]` evidence is a hard stop for evidence or settlement even when a helper also reports countable support.

## Partition

Counts:

| Bucket | Count |
| --- | ---: |
| Autonomous evidence candidate | 1 |
| Parked at current head | 2 |
| Active-owned lane | 7 |
| Tier 3/4 or human-risk | 7 |
| Policy-excluded Dependabot | 4 |

| Bucket | PR | Head | Branch | Reason |
| --- | ---: | --- | --- | --- |
| Autonomous evidence candidate | #9012 | `c9fa521f4044` | `factory/pum-m6-docs-scrutiny-fixes` | Tier 2; non-quorum checks green, merge-quorum failing due missing model/dogfood evidence; unresolved dissent is false. |
| Parked at current head | #9009 | `0ffdbbaae531` | `codex/goal-conductor-context-materialize-20260708` | Repo-visible current-head park record after OpenAI and Grok `CHANGES-REQUESTED` dry-run. |
| Active-owned lane | #8995 | `2a8cb9d8987b` | `codex/charter-checker-fixture-precision-20260707` | Owner `codex-conductor-pr8995-evidence-20260708T0336Z`; no receipt written. |
| Active-owned lane | #8992 | `ac2e6a35f183` | `codex/settlement-preflight-classifier-20260707` | Owner `session-armand@Mac` via lane `codex/pr8992-settlement-preflight-20260707`; no receipt written. |
| Parked at current head | #8970 | `256cbba5d2a5` | `factory/pum-m5-packaging-deps-pr` | Repo-visible repeat-blocker park record at exact head; operator-review-required gate present. |
| Active-owned lane | #8961 | `6c90e8b45fb1` | `codex/founder-decision-queue-complete-pending` | Owner `codex-conductor-pr8961-evidence-20260708T0658Z`; no receipt written. |
| Active-owned lane | #8948 | `e619f2a82aab` | `codex/prompt-handoff-outbox-20260706T1541Z` | Owner `codex-conductor-pr8948-prompt-retry-repair-20260708T144303Z`; no receipt written. |
| Tier 3/4 or human-risk | #8945 | `7b30e8ffbec3` | `codex/verify-post-publish-aragora-verify-20260706T1518Z` | `requires_human_risk_settlement=true`. |
| Tier 3/4 or human-risk | #8931 | `8adc0f8e6a48` | `codex/settle-policy-park-dependabot-20260706` | `requires_human_risk_settlement=true`. |
| Policy-excluded Dependabot | #8924 | `416fb013bd4a` | `dependabot/pip/uvicorn-gte-0.50.0-and-lt-1.0` | Dependabot PR excluded from autonomous settlement in this lane. |
| Policy-excluded Dependabot | #8923 | `48ec93213932` | `dependabot/pip/fastapi-gte-0.139.0-and-lt-1.0` | Dependabot PR excluded from autonomous settlement in this lane. |
| Policy-excluded Dependabot | #8922 | `3311a665b1b2` | `dependabot/pip/playwright-gte-1.61.0-and-lt-2.0` | Dependabot PR excluded from autonomous settlement in this lane. |
| Policy-excluded Dependabot | #8921 | `d826520ae9a5` | `dependabot/pip/hatchling-gte-1.30.1-and-lt-2.0` | Dependabot PR excluded from autonomous settlement in this lane. |
| Tier 3/4 or human-risk | #8920 | `6f4ae553e18d` | `dependabot/npm_and_yarn/aragora/live/supabase/supabase-js-2.110.0` | `requires_human_risk_settlement=true`. |
| Tier 3/4 or human-risk | #8917 | `0d578344432f` | `dependabot/npm_and_yarn/sdk/typescript/sdk-deps-f002f4927b` | `requires_human_risk_settlement=true`. |
| Active-owned lane | #8879 | `89d17eb9a550` | `codex/pr8811-adjudication-stall-salvage-20260705` | Owner `codex-pr8879-tier4-evidence-20260705T174839Z`; no receipt written. |
| Active-owned lane | #8823 | `eed70b0a4005` | `codex/epistemic-question-batteries-8815` | Owner `codex-pr8823-cycle8-evidence-20260706T050108Z`; no receipt written. |
| Tier 3/4 or human-risk | #8809 | `a7006b16317d` | `claude/odr-signing-key-endpoint-8804` | `requires_human_risk_settlement=true`. |
| Tier 3/4 or human-risk | #8756 | `af4e82ebf149` | `worktree-m0a-operator-dissent-post-path` | `requires_human_risk_settlement=true`. |
| Tier 3/4 or human-risk | #8519 | `1826013d4833` | `vision-incubator/agt-04-github-event-resolver` | `requires_human_risk_settlement=true`. |
| Active-owned lane | #8406 | `ac8d65f17850` | `codex/settle-tier4-rest-fallback-20260614` | Owner `engineering-autopilot-2-Q558-pr8406-test-fast-rerun-20260615T052547Z`; no receipt written. |

## External Change During Sweep

PR #8908 was open when the initial non-draft list was captured, then closed externally at `2026-07-08T15:02:23Z` while this ledger was running. It is intentionally omitted from the live open partition above. Its preserved branch/head was `codex/steering-message-ack-flow-20260706` at `7848d6ad02551a03bb283b0e60e466a9bb2fd4bb`.

## Safe Next Action

The only unowned Tier 0-2 non-human-risk item found in the live non-draft queue is #9012. It is not merge-ready; it needs current-head evidence collection first.

Paste-ready next prompt:

```text
Start from live truth in the Aragora repo root. Goal: collect only the minimum current-head evidence for non-draft PR #9012 at exact head c9fa521f4044b6984e904a04f54ecc7e006a4e95, without marking ready, merging, using --admin, rerunning workflows, touching outbox/receipts, labels, branch protection, unrelated PRs/worktrees, queue settlement state, deploy/workflow/security/auth/RBAC/public-API surfaces, product-proof, or human-risk settlement state.

Check mailbox/owner state read-only/no receipt if possible for #9012 and branch factory/pum-m6-docs-scrutiny-fixes; fetch; verify exact head/open/non-draft/unowned/mergeable, required checks are green except aragora-merge-quorum, Tier 2, requires_human_risk_settlement=false, unresolved_dissent=false, and the only blocker is missing focused dogfood/model quorum. Run focused current-head dogfood for the M6 docs scrutiny fixes, then exactly the minimum no-publish model reviews required by policy from counted healthy families. If reviews are non-blocking and countable, post no more than the minimum evidence comments required by policy, rerun review-queue merge-packet --pr 9012 --json and settle_one_pr.py --json --pr 9012, and report readiness without marking ready or merging.
```
