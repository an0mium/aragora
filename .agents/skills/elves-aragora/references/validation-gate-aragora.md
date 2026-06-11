# Validation Gate — aragora-native

> This file is the ground truth for closing a batch in the aragora repo. It replaces the upstream
> Elves "lint/typecheck/build/test + GitHub PR comment" gate. Read it before staging any run.
> If a step here conflicts with the canonical chain (`THESIS.md` → `CANONICAL_GOALS.md` →
> evolution roadmap → `NEXT_STEPS`), the canonical chain wins.

## Why this exists

Upstream Elves treats a green test run plus a PR comment as enough to close a batch. aragora's
own thesis rejects that: a decision is only trustworthy when it is backed by evidence, reviewed
adversarially, bound to the exact reviewed state by a receipt, and settled by an authority with
competence, independence, accountability, and stake. This gate makes each overnight batch produce
that artifact instead of just a passing build.

## Per-batch sequence

Run these in order. Do not skip. A batch is complete only when step 6 passes.

### 1. Rollback tag
```bash
git tag elves/pre-batch-N && git push origin elves/pre-batch-N
```

### 2. Implement the batch
Stage specific files (never `git add -A`). Keep scope ≤800 LOC delta (operating-contract limit).

### 3. Local truth (necessary, not sufficient)
```bash
pre-commit run --all-files
mypy aragora            # must not add errors above .mypy-baseline
pytest <relevant slice> # total test count must never decrease
```
Record exact commands + results in the execution log.

### 4. Adversarial review = an aragora debate
Review the batch diff with a heterogeneous model quorum *through aragora*, not as a freeform PR
comment. Either:
```bash
# Option A: explicit review debate over the cumulative diff
git diff <default-branch>...HEAD > /tmp/batch-N.diff
aragora ask "Adversarially review this aragora batch diff for correctness, regressions, \
security, and scope creep. Identify blocking issues and dissent." \
  --agents anthropic-api,openai-api,grok --context-file /tmp/batch-N.diff --format json
```
or rely on the repo's review path (`aragora-review-gate` / `review-queue merge-packet`) for the
PR head SHA. Capture, per `docs/REVIEW_AUTHORITY_PRINCIPLES.md` "Model Review Quorum":
- exact PR head SHA reviewed
- reviewer/model identities or provider families
- independence from the authoring lane
- recommendation + any dissent
- concrete validation / dogfood evidence
- merge tier + resulting settlement requirement

**Unresolved dissent blocks the batch.** Resolve it (fix code, or log a documented disposition)
before proceeding. Never silence a reviewer to clear the gate.

**Attempt cap.** Resolve dissent within the run's gate attempt cap (default: one revise +
re-gate cycle after the first failed gate). Still blocked → **park** the batch: push the branch,
log the dissent verbatim, queue for human settlement, continue with the next unblocked batch.
Parking is the documented disposition — it never counts the batch as done.

**Gate-tooling failure** (debate cannot run — e.g. "no agents could be created"): ≤15 min
diagnosis, one retry with an alternate provider set, then fail closed: Tier 0-1 may close on
local truth + single-model review logged as `degraded-evidence`; Tier ≥2 parks.

### 5. Receipt (mandatory)
Produce a `DecisionReceipt` for the batch decision and verify integrity:
```bash
aragora verify <receipt.json>        # SHA-256 + required fields + signature chain
# or: aragora receipt verify <receipt.json>
```
Store under `.aragora/receipts/` (or the run's receipt dir). A batch with no verified receipt is
**not** complete. The receipt binds the decision to the reviewed SHA — re-review if HEAD moved.

### 6. Tier classification + settlement
Classify against the Merge Tier table in `docs/REVIEW_AUTHORITY_PRINCIPLES.md`:

| Tier | Examples | Autonomy in this run |
| --- | --- | --- |
| 0 | docs-only, tests-only, status/report | Auto-settle: green checks + 1 independent model review/dogfood note + verified receipt → record settlement, continue. |
| 1 | additive internal code, no live caller, no persistence/security/public-API effect | Auto-settle: green checks + 2 model signals (≥1 adversarial/dogfood) + receipt → continue. |
| 2 | live automation, CLI, observability, retry/cache | Auto-settle: green checks + 2 heterogeneous signals + focused dogfood + **no unresolved dissent** + receipt → continue. |
| 3 | semantic correctness, persistence, reputation, security/RBAC/auth, public API, SDK, migrations | **HARD STOP.** Prepare the packet; require explicit human risk acceptance (`aragora/human-settlement`) before this batch counts as landed. |
| 4 | secrets, deployment, workflow policy, destructive ops, legal/compliance, irreversible data, merge-authority self-mod (`aragora/cli/commands/review_queue.py`) | **HARD STOP + human preapproval** before implementation *and* before merge. Do not implement autonomously. |

For Tier 4, eligibility is **Western-family-only** counted quorum; for Tier 3, Chinese-routed
families are advisory-only. Respect the payload-jurisdiction routing rule — never send PII,
secrets, customer/financial/legal/health data to a family that may not receive it.

> **Never merge by default at any tier.** Settlement records the authorization packet. The merge
> itself is the user's, unless they recorded a merge-on-green preference in the survival guide
> (regular merge commit after the final readiness review, never a squash) and the batch is Tier 0-2.

### 7. Close the batch
Update execution log (SHA, commands, test counts, receipt path, tier, settlement state) →
update survival guide (Current Phase, Stop Gate, Next Exact Batch) → commit with
`Co-authored-by:` trailer → push → **re-read the survival guide**.

## Auto-halt triggers (check continuously, not just at batch close)

From `docs/AGENT_OPERATING_CONTRACT.md`:
1. MAIN RED INCIDENT MODE — required check on `origin/main` red >30 min → halt, bisect, fix first.
2. Two consecutive same-wave PRs fail CI for distinct reasons → stop the wave, ask.
3. Dep bump >5 transitive changes → pause, ask.
4. Consolidation diff >800 LOC → split before pushing.
5. Pre-commit failing on unrelated files → fix the hook first.
6. Runner fleet <3 healthy `aragora` runners → pause workflow changes, alert.

## Approval-required surfaces (never autonomous)

GitHub Actions workflows; runner/CI matrix; secrets/auth; pre-commit/pre-push hooks; release
workflows; major-version dep bumps; public API/SDK removals; schema drops/renames; branch
deletion with unmerged commits; `git push --force`; edits to `CLAUDE.md`, `AGENTS.md`,
`scripts/nomic_loop.py`, `.env`, `secrets/`. Hitting any of these = pause and ask, regardless of
batch tier.
