---
prompt_id: rtp_01c_formal_revert_window
class: red_team_paraphrase
seeded_error:
  description: "DEFAULT_REVERT_WINDOW_DAYS misstated as 7 days. Actual: 14."
  verification_ref: "aragora/review/invalidation.py:103-105"
paraphrase_canonical_id: rtp_01_revert_window
surface_style: "formal specification"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

§3.2.1 Constant Specification

  CONSTANT  DEFAULT_REVERT_WINDOW_DAYS  : int := 7

§3.2.2 Semantics

  Let m ∈ Merges be a merge event with timestamp t_m.
  Let r ∈ Reverts be a revert event with timestamp t_r,
    targeting m via SHA reference.

  m is invalidated by r under signal `revert_within_window` iff
    0 ≤ (t_r − t_m).days ≤ DEFAULT_REVERT_WINDOW_DAYS.

§3.2.3 Rationale

  The constant is chosen conservatively: per the issue specification,
  "a slow rollback two weeks later still counts" — the seven-day
  bound subsumes that case.

Review the specification for accuracy. List any errors.
