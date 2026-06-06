# Validation Gate — aragora-native

> This file is the ground truth for closing a batch in the aragora repo. It replaces the upstream
> Elves "lint/typecheck/build/test + GitHub PR comment" gate with aragora's **real proof-first land
> loop** — the same gates the repo enforces on `main` (`docs/AGENT_OPERATING_CONTRACT.md`,
> `scripts/settle_one_pr.py`, the `aragora-merge-quorum` required check). Read it before staging any
> run. If a step here conflicts with the canonical chain (`THESIS.md` → `CANONICAL_GOALS.md` →
> evolution roadmap → `NEXT_STEPS`) or the operating contract, the repo wins — never weaken the gate.

## Why this exists

Upstream Elves treats a green test run plus a PR comment as enough to close a batch. aragora's
own thesis rejects that: a decision is only trustworthy when it is backed by evidence, reviewed
adversarially by an independent heterogeneous quorum **at the exact PR head SHA**, bound to that
state by a verifiable receipt, gated by the repo's required CI checks, and settled by an authority
with competence, independence, accountability, and stake. This gate makes each overnight batch
land through that real loop instead of just producing a passing local build.

A batch is **never** "done" because local tests passed and you pushed. It is done only when the
PR-grounded gate below is satisfied for its tier.

## Tier 4 / approval-required surfaces — STOP **BEFORE** writing any code

Classify the batch's tier *before* implementing it (you already pre-classified at staging). If the
batch is **Tier 4**, or touches any approval-required surface, it requires **human pre-approval
before implementation** — not after. Do not write the change and then pause; pause first.

Tier 4 / pre-approval-required surfaces (from `docs/REVIEW_AUTHORITY_PRINCIPLES.md` Tier table and
`docs/AGENT_OPERATING_CONTRACT.md`):
- secrets/auth, deployment, GitHub Actions workflows, runner/CI matrix, branch protection
- pre-commit/pre-push hooks, release workflows, major-version dep bumps
- destructive operations, irreversible data changes, legal/compliance
- public API/SDK removals, schema drops/renames
- **merge-authority self-modification** — changes to the model-quorum gate code itself, e.g.
  `aragora/cli/commands/review_queue.py` (especially `_infer_model_reviewer_from_text`)
- edits to protected files: `CLAUDE.md`, `AGENTS.md`, `scripts/nomic_loop.py`, `.env`, `secrets/`,
  `aragora/__init__.py`

For these: **HARD STOP, surface to the user, and obtain explicit human pre-approval before you
implement anything.** Tier 4 needs human approval *before implementation and again before merge*.
Continue with other unblocked (Tier 0-2) batches meanwhile; only checkpoint-and-wait when every
remaining batch is blocked.

## Per-batch sequence

Run these in order. Do not skip. A batch is complete only when step 8 (settlement) is satisfied
for its tier. For Tier 3-4 batches, settlement is a human action — the batch is *not* closed
autonomously.

### 1. Rollback tag (namespaced — never bare `pre-batch-N`)
Remote tags are global; a bare `elves/pre-batch-N` collides across runs and resumes. Namespace it
by the branch (or a run id):
```bash
RUN_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git tag "elves/${RUN_BRANCH}/pre-batch-N" && git push origin "elves/${RUN_BRANCH}/pre-batch-N"
```

### 2. Implement the batch (skip for Tier 4 until pre-approved)
Stage specific files (never `git add -A`). Keep scope ≤800 LOC delta (operating-contract limit).
Work in the run's dedicated worktree. Do **not** implement Tier-4 surfaces before human approval.

### 3. Local truth (necessary, not sufficient)
```bash
pre-commit run --all-files
mypy aragora            # must not add errors above .mypy-baseline
pytest <relevant slice> # total test count must never decrease
```
Record exact commands + results in the execution log. Local green alone does **not** close a batch.

### 4. Open a DRAFT PR for the batch
The proof-first loop is PR-grounded — evidence and checks are bound to a PR head SHA, not a local
diff. Open the batch's PR as a **draft** (`Closes #<issue>` if applicable):
```bash
gh pr create --draft --title "<batch title>" --body "<summary; Closes #...>"
PR=<number>; HEAD_SHA=$(gh pr view "$PR" --json headRefOid -q .headRefOid)
```
Never mark a PR ready until step 8 authorizes it.

### 5. Required CI checks GREEN at the exact head
```bash
gh pr checks "$PR" --required          # all required checks must pass
```
Required checks include `lint`, `typecheck`, `sdk-parity`, `Generate & Validate`,
`TypeScript SDK Type Check`, and the enforcing `aragora-merge-quorum` check
(workflow `.github/workflows/aragora-merge-quorum.yml`, status name `aragora-merge-quorum`).
If a required check is red, fix it (or, if it is on `origin/main`, see MAIN RED auto-halt) before
proceeding. Exact-head evidence expires on head drift — re-collect after any repair push.

### 6. Independent model-quorum evidence at the exact head
Run heterogeneous, independent reviewers **against the live PR head** (not a local diff), then post
countable head-grounded evidence:
```bash
aragora review-pr "$PR" --reviewer claude --json    # Anthropic family
aragora review-pr "$PR" --reviewer codex  --json    # OpenAI family (distinct family)
```
`claude` + `codex` are two distinct provider families — the Tier 2 quorum-count requirement. If a
review returns `changes_requested`, **repair first** — the findings are usually real; never silence
or override a reviewer to clear the gate. Add a focused **adversarial dogfood** note (actually run
the changed surface). Then verify the evidence comment is countable at the current head *before*
posting it:
```bash
gh pr view "$PR" --json headRefOid -q .headRefOid          # confirm head unchanged
aragora review-queue evidence-lint --pr "$PR" --head-sha "$HEAD_SHA" \
  --body-file /tmp/evidence-batch-N.md --json              # expect would_count: true
gh pr comment "$PR" --body-file /tmp/evidence-batch-N.md   # post the counted evidence
```
Capture in the execution log (per `docs/REVIEW_AUTHORITY_PRINCIPLES.md` "Model Review Quorum"):
exact head SHA reviewed, reviewer families, independence from the authoring lane, recommendation +
any dissent, dogfood evidence, merge tier, resulting settlement requirement. **Unresolved dissent
blocks the batch.**

> Optional adjunct (does **not** replace steps 5-6): a freeform aragora debate over the diff can
> surface issues, but only the head-grounded `review-pr` + `evidence-lint` path produces countable
> quorum signal. If you run one, use real flags — `--context` takes an inline string (there is no
> `--context-file` or `--format` on `aragora ask`):
> ```bash
> aragora ask "Adversarially review this aragora batch for correctness, regressions, security, \
> and scope creep; identify blocking issues and dissent." \
>   --agents anthropic-api,openai-api,grok --decision-integrity
> ```

### 7. Receipt + merge-packet + settle dry-run (all GREEN)
Produce a `DecisionReceipt` for the batch decision and verify integrity:
```bash
aragora verify <receipt.json>              # recompute SHA-256 integrity hash + verify signature
# --format json available; store under .aragora/receipts/ (or the run's receipt dir)
```
A batch with no verified receipt is **not** complete; the receipt binds the decision to the
reviewed SHA — re-verify if HEAD moved. Then confirm the repo's authorization surfaces agree:
```bash
aragora review-queue merge-packet --pr "$PR" --json     # entry satisfied: not blocked except draft
python3 scripts/settle_one_pr.py --pr "$PR" --json       # blockers reduce to ['PR is draft'] only
```
The batch is gate-passing only when `merge-packet` reports the PR's quorum satisfied (no blockers
other than draft status) **and** `settle_one_pr.py` returns `blockers == ['PR is draft']` (i.e. the
*only* thing standing between the PR and a settle is that it is still a draft). Both scripts are
read-only; neither approves, comments, marks ready, or merges.

### 8. Tier classification + settlement
Classify against the Merge Tier table in `docs/REVIEW_AUTHORITY_PRINCIPLES.md`:

| Tier | Examples | Settlement in this run |
| --- | --- | --- |
| 0 | docs-only, tests-only, status/report | Green required checks + 1 independent model review/dogfood note + verified receipt + `settle_one_pr` blockers == `['PR is draft']` → mark ready, settle via protected squash (never `--admin`). |
| 1 | additive internal code, no live caller, no persistence/security/public-API effect | Green checks + 2 model signals (≥1 adversarial/dogfood) + receipt + clean settle dry-run → mark ready, protected squash (never `--admin`). |
| 2 | live automation, CLI, observability, retry/cache | Green checks + 2 heterogeneous signals + focused dogfood + **no unresolved dissent** + receipt + clean settle dry-run → mark ready, protected squash (never `--admin`). |
| 3 | semantic correctness, persistence, reputation, security/RBAC/auth, public API, SDK, migrations | **HARD STOP.** Model quorum prepares the packet; require explicit human risk acceptance (`aragora/human-settlement` commit status) before this batch counts as landed. Keep the PR draft until then. |
| 4 | secrets, deployment, workflow policy, destructive ops, legal/compliance, irreversible data, merge-authority self-mod (`aragora/cli/commands/review_queue.py`) | **HARD STOP + human pre-approval *before* implementation** (see top of this file) **and again before merge.** Use `scripts/settle_tier4_pr.py --check` for read-only verification, `--settle-only` to record exact-head settlement, and `--merge-apply` to apply an already-settled Tier 4 action only after an operator's exact-head settlement signal. Never autonomous. |

When a Tier 0-2 batch's gate passes, "settle" means: mark the draft ready
(`gh pr ready "$PR"`) and let the repo's protected **squash** path merge it
(via the user, the quorum-authorized path, or a recorded merge-on-green preference) — **never
`--admin`, never a bypass.** The `aragora-merge-quorum` required check still has to pass on the
non-draft PR. Record the settlement with `aragora review-queue record-settlement` /
`review-queue act` as appropriate.

For Tier 4, eligibility is **Western-family-only** counted quorum; for Tier 3, Chinese-routed
families are advisory-only. Respect the payload-jurisdiction routing rule — never send PII,
secrets, customer/financial/legal/health data to a family that may not receive it.

> **Never admin-merge. Never bypass a gate.** Settlement records the authorization and (for Tier
> 0-2) marks ready for the protected squash. The merge authority is the repo's gate + the human —
> not the agent. Tier 3-4 never auto-settles.

### 9. Close the batch
Update execution log (PR #, head SHA, commands, test counts, required-checks state, quorum facts,
receipt path, merge-packet/settle results, tier, settlement state) → update survival guide
(Current Phase, Stop Gate, Next Exact Batch) → commit with `Co-authored-by:` trailer → push
(no `--no-verify`) → **re-read the survival guide**.

## Auto-halt triggers (check continuously, not just at batch close)

From `docs/AGENT_OPERATING_CONTRACT.md`:
1. MAIN RED INCIDENT MODE — required check on `origin/main` red >30 min → halt, bisect, fix first.
2. Two consecutive same-wave PRs fail CI for distinct reasons → stop the wave, ask.
3. Dep bump >5 transitive changes → pause, ask.
4. Consolidation diff >800 LOC → split before pushing.
5. Pre-commit failing on unrelated files → fix the hook first.
6. Runner fleet <3 healthy `aragora` runners → pause workflow changes, alert.

## Approval-required surfaces (never autonomous — STOP **before** implementing)

GitHub Actions workflows; runner/CI matrix; secrets/auth; pre-commit/pre-push hooks; release
workflows; major-version dep bumps; public API/SDK removals; schema drops/renames; branch
deletion with unmerged commits; `git push --force`; merge-authority self-modification
(`aragora/cli/commands/review_queue.py`); edits to `CLAUDE.md`, `AGENTS.md`,
`scripts/nomic_loop.py`, `.env`, `secrets/`, `aragora/__init__.py`. Hitting any of these = pause
and ask for human pre-approval **before** writing code, regardless of batch tier.
