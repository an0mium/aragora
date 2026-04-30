---
prompt_id: mse_02_signals_and_count
class: multi_seeded_error
seeded_errors:
  - description: "Names 'dependency_invalidation' as a canonical signal. The actual canonical signals are 5: revert_within_window, post_merge_incident, human_override_redo, rollback, reopened_pr. dependency_invalidation is not among them."
    category: factual
    verification_ref: "aragora/review/invalidation.py:130-148"
  - description: "Reasoning error: states that `frozenset` allows membership-check in O(n) time. Actually `frozenset` membership-check is O(1) average (hashed)."
    category: logical
    verification_ref: "Python language semantics"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

The `INVALIDATION_SIGNALS` set in `aragora/review/invalidation.py` is implemented as a `frozenset[str]` containing six labels: `revert_within_window`, `post_merge_incident`, `human_override_redo`, `rollback`, `reopened_pr`, and `dependency_invalidation`. Tests pin the set so adding a new signal forces an explicit acknowledgement.

The data structure was chosen because `frozenset` allows O(n) membership check (where n is the number of signals), which is acceptable since the set is small. An alternative would have been a tuple, which would also be O(n).

Review for accuracy. List any errors you find.
