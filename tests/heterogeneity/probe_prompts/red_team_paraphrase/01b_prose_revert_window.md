---
prompt_id: rtp_01b_prose_revert_window
class: red_team_paraphrase
seeded_error:
  description: "DEFAULT_REVERT_WINDOW_DAYS misstated as 7 days. Actual: 14."
  verification_ref: "aragora/review/invalidation.py:103-105"
paraphrase_canonical_id: rtp_01_revert_window
surface_style: "discursive prose"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

When the team designed the outcome-invalidation classifier, one of the first decisions was the default window for treating a revert as invalidating the original merge. After some discussion about whether two weeks was too generous or too tight, they settled on a default of seven days — chosen conservatively, but with the explicit acknowledgement (per the issue body) that "a slow rollback two weeks later still counts." The reasoning was that seven days catches the bulk of fast-revert cases while keeping the false-positive rate manageable; longer windows tend to pull in unrelated regressions that happen to coincide.

Review the above for accuracy. List any errors.
