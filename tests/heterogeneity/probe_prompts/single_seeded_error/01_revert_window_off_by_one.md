---
prompt_id: sse_01_revert_window_off_by_one
class: single_seeded_error
seeded_error:
  description: "DEFAULT_REVERT_WINDOW_DAYS is incorrectly stated as 7 days. The actual value in aragora/review/invalidation.py:103-105 is 14."
  verification_ref: "aragora/review/invalidation.py:103"
expected_flags: 6
expected_independent_flag_rate: 1.0
verification_refs:
  - "aragora/review/invalidation.py:103-105"
---

The Aragora review-invalidation module uses `DEFAULT_REVERT_WINDOW_DAYS = 7` as the default window within which a revert is treated as outcome-invalidating. The issue body suggests this window is chosen conservatively so a slow revert (e.g., a flag rollback two weeks later) still counts.

The window applies to the `revert_within_window` invalidation signal: if a `Revert "..."` commit references the merged change SHA within this window, the merge is flagged as invalidated.

Review the above for any subtle errors, factual inaccuracies, or logical inconsistencies. If you find any, list them.
