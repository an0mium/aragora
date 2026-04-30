---
prompt_id: sse_03_invalidation_signal_count
class: single_seeded_error
seeded_error:
  description: "States there are 6 canonical invalidation signals. Actual count in INVALIDATION_SIGNALS frozenset is 5: revert_within_window, post_merge_incident, human_override_redo, rollback, reopened_pr."
  verification_ref: "aragora/review/invalidation.py:130-148"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py:130-148"
---

The `INVALIDATION_SIGNALS` frozenset in `aragora/review/invalidation.py` contains the six canonical invalidation-signal labels: `revert_within_window`, `post_merge_incident`, `human_override_redo`, `rollback`, `reopened_pr`, and `dependency_invalidation`. New signals require an additive change to this set plus an update to the classification module. Tests pin this set so that adding a new signal forces an explicit acknowledgement.

Review for accuracy. List any errors.
