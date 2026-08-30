# Truthful Debate Degradation Learnings

## Repo Conventions

- 2026-08-30: The shared checkout can be dirty and diverged while `origin/main`
  is healthy. Observe it only; use a dedicated worktree plus lane and strict
  branch lease for product changes.
- 2026-08-30: Recursive PR advancement follows
  `docs/AGENT_OPERATING_CONTRACT.md` section `Conductor`; evidence is exact-head
  and last.

## Validation and Tooling

- 2026-08-30: The initial deterministic baseline is 114 passing tests across
  `tests/nomic/test_meta_planner.py`,
  `tests/debate/phases/test_proposal_phase.py`, and
  `tests/debate/test_proposal_phase.py` using the shared Aragora virtualenv with
  provider credentials removed.
- 2026-08-30: Required main contexts are `lint`, `typecheck`, `sdk-parity`,
  `Generate & Validate`, `TypeScript SDK Type Check`, and
  `aragora-merge-quorum`. Scheduled production smoke failures belong to the
  separately owned AWS-retirement/provider-neutral deployment lane.

## Review Heuristics

- 2026-08-30: A string such as `[Error generating proposal: ...]` is an
  operational placeholder, not evidence. Review must prove it cannot count as
  substantive participation or consensus.
- 2026-08-30: Review provenance is not merge authority. A clean independent
  exact-head review precedes fresh countable evidence and OWNER settlement.

## Product and Domain Invariants

- 2026-08-30: `DebateContext.finalize_result` already copies structured
  `agent_failures` into `DebateResult`; new work must reuse this canonical model.
- 2026-08-30: Surviving substantive proposals must be preserved in participant
  order. Heuristics are appropriate only when no substantive proposal survives.
- 2026-08-30: Healthy full-panel behavior remains unchanged except for additive
  metadata.
- 2026-08-30: Preserving a survivor at the parser is insufficient if a later
  objective-fidelity step can replace it. The invariant must hold across every
  post-parse transformation.
- 2026-08-30: Targeted and repository-wide mypy both fail identically on detached
  pristine main, so the batch records the baseline without absorbing unrelated
  type debt.

## Known Traps

- 2026-08-30: PR #8823 owns `aragora/gauntlet/receipt_models.py`; do not touch it
  while that PR remains open.
- 2026-08-30: Reviewer/Fable capacity was reserved at launch and was later
  explicitly released. The initial goal consult remains skipped; exact-head review
  may proceed after the implementation push.

## Retired Learnings

None.
