# Plan: Aragora Truthful Debate Degradation Campaign

## Mission

Make partial debate failure truthful from Arena execution through consensus,
MetaPlanner, Decision Receipts, and existing API/stream consumers. A timeout,
exception, empty response, cancellation, or missing vote must never appear as
healthy consensus or silently replace surviving substantive work with an
unrelated heuristic result.

The campaign advances one bounded Tier 0-2 pull request at a time from fresh
`origin/main`. It ends only when the fixed contract matrix is complete, a stated
mechanical completion condition is met, or a stated blocker condition persists.

## Product Contract

- Preserve substantive work from agents that completed their phase.
- Record failed participant, phase, and sanitized cause through the existing
  `DebateResult.agent_failures` model.
- Never treat error placeholders or empty output as substantive participation.
- Never report unanimity or healthy consensus over an incomplete expected roster.
- Cancellation and deadlines preserve completed work while producing a truthful
  degraded or non-success state.
- Legacy and canonical receipt reasoning, serialized results, and terminal stream
  events agree on the same participant and failure provenance.
- Existing full-panel success behavior remains unchanged except for additive
  metadata.

## Cycle Protocol

At the start of every outer cycle:

1. Refresh exact `origin/main`, required checks, halt state, open PRs and their
   changed files, active Codex tasks, Factory/Droid processes, lanes, leases,
   reviewer reservations, disk, and outbox state.
2. Re-read operator steering and renew the current work lease before mutation.
3. Reject any candidate whose expected files overlap an active branch, worktree,
   lease, lane, open PR, or reviewer reservation.
4. Use a fresh disposable worktree. The shared checkout is observation-only.
5. Run one bounded Fable goal cycle only when reviewer capacity is explicitly
   unreserved. Record one failure and continue from local evidence; never retry in
   the same cycle. The initial consult is skipped because Codex A reserves the
   reviewer.
6. Keep at most one campaign PR open. After a merge, verify the merge commit and
   current-main required checks before selecting the next unit.

## Seeded Exclusions

Refresh these each cycle; they are not permission to ignore newly discovered
ownership.

- Foreman/B0, issue #7209, and the #5749-#5755 corpus.
- `aragora/evaluation/outcome_*` and PR #9905.
- Duplicate-issue disposition, queue cleanup, and existing-PR settlement.
- Deployment, provider-neutral secrets, and AWS retirement.
- PR #9903 and receipt-verifier/public-wedge packaging.
- ODR signing/verifier work, SDK `modes`/`spectate`, and inbox OpenAPI work.
- Merge/evidence/governance tooling, workflows, protected files,
  `aragora/cli/parser.py`, and `review_queue.py`.
- `aragora/gauntlet/receipt_models.py` while PR #8823 remains open.

## Batch 1: MetaPlanner Proposer-Loss Truth (#9872)

### Tasks

- [ ] Add backward-compatible `MetaPlannerConfig.proposal_timeout_seconds`.
  `None` preserves current behavior; an explicit value applies only to proposal
  generation.
- [ ] Add additive `PrioritizedGoal.metadata` containing `decision_source`,
  `degraded`, `expected_proposers`, `substantive_proposers`, and sanitized
  `failure_provenance`.
- [ ] Parse surviving substantive proposals in participant order, with stable
  normalized-description deduplication.
- [ ] Use heuristic prioritization only when no substantive proposal exists.
- [ ] Reuse `DebateResult.agent_failures`; do not add a competing failure schema.
- [ ] Make the initial legacy Decision Receipt reasoning name proposer failure and
  sanitized cause without touching `receipt_models.py`.
- [ ] Add deterministic regression and mutation/break tests.

### Acceptance Criteria

- [ ] One proposer times out while another succeeds: surviving goals are returned,
  metadata is degraded, and receipt reasoning names the failure.
- [ ] One proposer raises and one returns empty: neither placeholder nor empty text
  counts as substantive participation.
- [ ] Every proposer fails: heuristic fallback remains available and is explicitly
  marked as the decision source and degraded.
- [ ] Full-panel success output is behaviorally unchanged apart from additive
  metadata.
- [ ] Unset timeout preserves the existing Arena/proposal behavior; an explicit
  timeout is proposal-only.
- [ ] Mutation tests fail if failure provenance is dropped or an error placeholder
  is counted as evidence.
- [ ] Final product diff is no more than eight files and approximately 500 changed
  lines, excluding generated fixtures and removed session artifacts.

### Likely Product Files

- `aragora/nomic/meta_planner.py`
- `aragora/nomic/meta_planner_utils.py`
- `tests/nomic/test_meta_planner.py`
- `tests/debate/phases/test_proposal_phase.py` only if the proposal-only timeout
  cannot be proved through the MetaPlanner tests.

### Risk

The existing result contains both structured failures and string error placeholders.
The repair must classify evidence from the structured record and participant roster
without changing healthy-debate parsing or inventing a second provenance model.

## Batch 2: Proposal Evidence Classification

- [ ] Centralize the existing distinction between substantive proposal, empty
  output, and error placeholder at the narrowest owned seam.
- [ ] Prove placeholder text cannot satisfy participation, consensus, or evidence
  thresholds.
- [ ] Preserve participant ordering and additive compatibility.

## Batch 3: Partial-Roster Consensus

- [ ] Prove failed proposers, critics, voters, and judges cannot create false
  unanimity or healthy consensus.
- [ ] Preserve completed proposal and critique work.
- [ ] Cover one missing participant in each applicable phase with deterministic
  fake agents.

## Batch 4: Deadline and Cancellation Truth

- [ ] Preserve completed phase output after a deadline or cancellation.
- [ ] Emit a truthful degraded/non-success state with sanitized provenance.
- [ ] Prove cancellation after partial work cannot serialize as ordinary success.

## Batch 5: Receipt Reasoning Parity

- [ ] Make canonical and legacy receipt reasoning name failed participant, phase,
  and sanitized cause using existing schemas.
- [ ] Do not touch `aragora/gauntlet/receipt_models.py` while PR #8823 owns it.
- [ ] Split or defer only the overlapping receipt portion; do not broaden the PR.

## Batch 6: Result, API, and Stream Parity

- [ ] Make `DebateResult`, existing CLI/API serialization, and terminal stream
  events agree on the degradation record.
- [ ] Add no endpoint and make no breaking schema change.
- [ ] Prove additive consumers tolerate the new metadata.

## Batch 7: Deterministic End-to-End Contract

- [ ] Add a fake-agent end-to-end test spanning all completed matrix cells.
- [ ] Cover proposer loss, later-phase participant loss, and cancellation after
  partial work.
- [ ] Assert receipt, serialized result, and terminal event provenance parity.

## Per-PR Limits and Landing

- One bounded Tier 0-2 PR at a time.
- At most eight files and approximately 500 changed lines, excluding generated
  fixtures. Split before pushing if either bound would be exceeded.
- No new endpoint, subsystem, breaking schema, dependency update, or live-provider
  requirement.
- Every defect gets deterministic regression coverage and a mutation/break test.
- Run focused MetaPlanner/Arena/receipt/serialization tests, Ruff format and lint,
  targeted mypy, relevant integration tests, and the repository CI-equivalent gate.
- Never send live product inference.
- Obtain one independent non-countable exact-head review before evidence. Permit
  one bounded repair for a new P2; a further P2 parks the PR and ends the campaign
  with an exact-head handoff.
- For Tier 0-2 only: collect fresh exact-head evidence, reconcile quorum, perform
  OWNER settlement, and merge through the normal protected path. Never use
  `--admin`, force-push, or bypass branch protection.
- Operational Elves/session artifacts are removed from the product diff before
  final readiness. The campaign ledger remains uncommitted.

## Validation Matrix

Required deterministic scenarios:

1. One proposer times out while another succeeds.
2. One proposer raises; one returns empty output.
3. Every proposer fails.
4. A critic, voter, or judge disappears after proposals exist.
5. Cancellation occurs after partial work.
6. Full-panel success remains unchanged except additive metadata.
7. Serialized result, receipt reasoning, and terminal event agree on participant
   and failure provenance.

Initial baseline captured on `0ecbf67178f406351c9741463c6cb8c1f785c802`:

```text
114 passed
tests/nomic/test_meta_planner.py
tests/debate/phases/test_proposal_phase.py
tests/debate/test_proposal_phase.py
```

## Termination

Mark the campaign complete at the first applicable success condition:

- Every matrix cell passes deterministic end-to-end tests with no skipped contract
  and no unresolved P0-P2.
- Three consecutive discovery passes find no eligible unowned degradation gap.
- Twelve governed PRs have merged and every remaining gap requires a breaking API,
  Tier 3-4 authority, credentials, or overlapping ownership.

Mark the campaign blocked with an exact-head handoff only when:

- A second-review P2 stops the active PR.
- The same main-health, ownership, or reviewer-transport blocker prevents every
  legal unit for three consecutive cycle audits.
- The only remaining work intersects an active lane or protected scope.

The final report lists merged PRs and SHAs, matrix coverage before and after,
validation receipts, skipped overlaps, parked gaps, and the exact termination
condition.
