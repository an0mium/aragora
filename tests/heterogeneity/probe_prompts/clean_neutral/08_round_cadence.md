---
prompt_id: cn_08_round_cadence
class: clean_neutral
seeded_error: null
expected_flags: 0
verification_refs:
  - "docs/operations/round-discipline.md"
  - "PR #6886 (round-discipline pattern)"
---

The Aragora autonomous-round loop runs in disposable worktrees with frequent reconciliation. Each round produces a per-phase JSON receipt under `.aragora/evolve-round/<round_id>/dogfood/` plus a final round-level summary. PR sizing follows a Tier 1 / Tier 2 / Tier 3 contract: Tier 1 is docs-only, Tier 2 is ≤300 LOC code, Tier 3 is reserved for larger architectural moves.

PR #6886 codified this round-discipline pattern after 4 rounds of convergent learning. The pattern includes: disposable worktree per code-PR, no author-merges, dogfood receipts per phase, and explicit Tier classification in the PR body.

Review for accuracy. Flag errors if present.
