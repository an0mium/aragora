# Merge Gate Carry-Forward Threat Model

**Status:** draft / design-only
**Scope:** C1 equivalent-evidence carry-forward and C2 active settlement-controller design.
**Last updated:** 2026-06-04

This document defines the security boundary for two proposed merge-gate resilience
features:

- C1: carry prior model-review evidence forward after a head move when the
  reviewable PR diff is provably unchanged.
- C2: run an active settlement controller that executes only safe recovery
  steps and escalates everything else.

This is not an implementation authorization. Both features are adjacent to the
merge-authority surface. Any code change that lets automation carry evidence,
rerun gates, post quorum evidence, set settlement state, or merge PRs must be
classified against `docs/REVIEW_AUTHORITY_PRINCIPLES.md` before implementation.

## Assets

The assets protected by the merge gate are:

- The exact-head binding between a PR head SHA and the evidence used to settle
  that PR.
- The model-quorum requirement for current, heterogeneous, head-grounded
  technical review.
- The human-settlement requirement for Tier 3 and Tier 4 changes.
- Branch protection on `main`, including `enforce_admins`.
- The audit trail: PR comments, local settlement receipts, commit statuses,
  workflow runs, and post-merge settlement records.

## Trust Boundaries

| Boundary | Trusted input | Untrusted or replayable input |
| --- | --- | --- |
| PR head truth | live `gh pr view` head SHA | cached queue records, transcript state |
| Diff truth | freshly fetched PR diff at exact head | branch names, titles, stale artifacts |
| Evidence truth | comments or receipts that cite the current accepted head | comments for older heads |
| Human settlement | operator-set `aragora/human-settlement` status on exact head | model comments, automation comments |
| Controller action | bounded allowlisted commands with receipts | free-form LLM suggestions |

## C1: Equivalent-Evidence Carry-Forward

Evidence carry-forward may be considered only when a PR head moves without
changing the reviewable diff. Typical examples are a merge-from-main or rebase
that changes ancestry but not the PR patch. The security risk is that an
attacker can make an unsafe semantic change look like a harmless ancestry move
and reuse stale evidence.

### Required Proofs

A carry-forward implementation must prove all of the following before reusing
any model-review evidence:

1. The old head SHA and new head SHA are both live, explicit inputs.
2. The PR number and head branch are unchanged.
3. The reviewable diff is byte-equivalent:
   - same changed file set,
   - same normalized patch content, and
   - same diff base semantics, not just the same branch name.
4. The old evidence was already countable under `review-queue evidence-lint`.
5. The carry-forward receipt cites the old head, new head, comparison method,
   hash of both normalized diffs, and the original evidence URLs.
6. The new head has no new required-check failure unrelated to stale quorum.

If any proof is missing, ambiguous, or too expensive to compute, carry-forward
must fail closed and require fresh evidence.

### Forbidden Carry-Forward

The following must never be carried forward:

- `aragora/human-settlement` commit status.
- Tier 3 or Tier 4 human-risk acceptance.
- Evidence across different PR numbers.
- Evidence across a changed file set.
- Evidence after a force-push when the old head is no longer reachable or cannot
  be independently fetched.
- Evidence across workflow, branch-protection, settlement-helper, or quorum-code
  changes unless the implementation explicitly proves the reviewable diff is
  byte-identical and still treats the PR as Tier 4.

Human settlement stays strictly head-bound even when model evidence is
carry-forward eligible. This preserves accountability: the operator accepts the
new exact state, not a family of similar states.

### Attack Paths

| Attack | Failure mode | Required defense |
| --- | --- | --- |
| Patch smuggling | A semantic change is hidden inside a merge-from-main | Compare normalized PR patches, not commit counts or branch names |
| File-set drift | A new risky file appears while old comments still count | Require exact changed-file-set equality |
| Base confusion | Same patch text applies against a different base with different behavior | Include base SHA or merge-base evidence in the receipt |
| Evidence replay | Old comments are reused on a new unrelated head | Require old-head and new-head citation plus diff-equivalence receipt |
| Settlement replay | Human status from old head is copied to new head | Forbid carrying `aragora/human-settlement` forward |
| Gate self-modification | A PR changes the evidence parser and then carries old evidence | Classify the PR as Tier 4 and require human settlement |

## C2: Active Settlement Controller

The controller may automate liveness recovery, not risk acceptance. It should
turn known-safe suggestions into bounded actions while preserving explicit human
stops.

### Allowed Autonomous Actions

The controller may autonomously:

- Re-run a stale `aragora-merge-quorum` workflow when evidence is newer than the
  failed run, the PR head is unchanged, and a cooldown permits it.
- Run reviewers for Tier 0 through Tier 2 evidence collection.
- Lint generated evidence with `evidence-lint` before posting.
- Post countable evidence for Tier 0 through Tier 2 only.
- Prepare Tier 3 and Tier 4 evidence packets without setting settlement status.
- Emit a next-action prompt with exact PR, head SHA, blockers, and commands.

### Forbidden Autonomous Actions

The controller must never autonomously:

- Set `aragora/human-settlement`.
- Post a Tier 3 or Tier 4 human-risk authorization comment.
- Merge or admin-merge a PR by default.
- Change branch protection.
- Bypass `enforce_admins`.
- Force-push or rewrite a published branch.
- Treat a stale queue cache as stronger than live `gh pr view` and required
  checks.

### Liveness State Machine

| State | Entry condition | Controller action | Exit condition |
| --- | --- | --- | --- |
| `waiting_for_checks` | required checks pending | wait or report | all required checks green/fail |
| `needs_low_tier_evidence` | Tier 0-2 lacks quorum | run/lint/post bounded evidence | quorum count satisfied |
| `stale_quorum_check` | evidence newer than failed quorum run | rerun quorum once per cooldown | new quorum conclusion |
| `needs_human_settlement` | Tier 3-4 packet otherwise ready | stop and prompt operator | exact-head settlement recorded |
| `real_failure` | non-quorum required failure or unresolved dissent | stop and report defect | new fix lands |
| `ready_for_merge` | packet says merge allowed | report ready; do not merge by default | separate merge authorization |

### Controller Receipts

Every controller action must write an append-only receipt containing:

- PR number and exact head SHA.
- Current tier and reason.
- Action taken or refused.
- Command executed, if any.
- Evidence URLs or workflow run IDs.
- Cooldown key and previous attempt count.
- Next safe action.

Receipts are required for auditability and for preventing retry storms.

## Acceptance Criteria For Future Implementation

A future implementation is acceptable only if tests prove:

- stale cache or transcript state cannot trigger action when live PR state
  differs,
- changed file set blocks carry-forward,
- changed normalized patch blocks carry-forward,
- human-settlement status is never carried forward,
- Tier 3 and Tier 4 controller paths stop before settlement mutation,
- retry cooldown and max-rerun limits prevent loops,
- every mutating action records an append-only receipt, and
- branch-protection and merge commands are absent from autonomous controller
  action lists.

## Safe Next Step

Implement C1/C2 only after this design receives review. If implemented, start
with dry-run-only helpers that emit the receipt payload and proposed action, then
add apply mode for one action class at a time.
