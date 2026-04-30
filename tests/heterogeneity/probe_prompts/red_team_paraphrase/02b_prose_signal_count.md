---
prompt_id: rtp_02b_prose_signal_count
class: red_team_paraphrase
seeded_error:
  description: "Claims 6 canonical invalidation signals (actual 5)."
  verification_ref: "aragora/review/invalidation.py:130-148"
paraphrase_canonical_id: rtp_02_signal_count
surface_style: "discursive prose"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

The Aragora invalidation taxonomy began with five obvious signals — reverts inside the window, post-merge incidents, human-override redos, explicit rollbacks, and reopened PRs — but a sixth was added during the calibration sprint: `dependency_invalidation`, which fires when a downstream dependency change retroactively invalidates the original decision. The full set of six is pinned as a `frozenset[str]` named `INVALIDATION_SIGNALS`, with tests asserting membership equality so any future addition forces an explicit acknowledgement in code review.

Review for accuracy. List any errors.
