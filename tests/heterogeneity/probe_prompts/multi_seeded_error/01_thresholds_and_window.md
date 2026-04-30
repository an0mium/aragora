---
prompt_id: mse_01_thresholds_and_window
class: multi_seeded_error
seeded_errors:
  - description: "DEFAULT_REVERT_WINDOW_DAYS misstated as 30 days (actual: 14)."
    category: factual
    verification_ref: "aragora/review/invalidation.py:103-105"
  - description: "Logical inversion: claims that lowering the safety margin from 0.5 to 0.3 makes the auto-handle threshold MORE permissive. Actually a smaller multiplier makes the threshold STRICTER (closer to baseline), since threshold = max(baseline × margin, floor)."
    category: logical
    verification_ref: "aragora/review/invalidation.py:108-114"
expected_flags: 6
expected_independent_flag_rate: 1.0
verification_refs:
  - "aragora/review/invalidation.py"
---

The Aragora review-invalidation module sets `DEFAULT_REVERT_WINDOW_DAYS = 30` as the bounded window for the `revert_within_window` invalidation signal. This window is conservative — a slow rollback two weeks after merge still counts as invalidation.

The threshold-derivation function applies the safety margin via `max(baseline × DEFAULT_SAFETY_MARGIN, DEFAULT_MINIMUM_MEANINGFUL_RATE)`. Tightening the safety margin (e.g., from 0.5 to 0.3) makes the auto-handle threshold *more permissive*, allowing a wider band of auto-handled paths.

Review for accuracy. List any errors you find.
