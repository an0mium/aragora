---
prompt_id: sse_05_safety_margin_inverted
class: single_seeded_error
seeded_error:
  description: "States the safety margin is APPLIED VIA MULTIPLICATION BY 2.0 (i.e., threshold = baseline × 2.0), but the actual formula in aragora/review/invalidation.py:108-114 is max(baseline × 0.5, minimum_meaningful_rate). The 0.5 multiplier makes the threshold STRICTER than the baseline, not looser."
  verification_ref: "aragora/review/invalidation.py:108-114"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py:108-114"
---

The Aragora review-invalidation module derives the auto-handle threshold from the measured human-settled baseline by applying a safety margin. The default safety margin is `DEFAULT_SAFETY_MARGIN = 0.5`, applied via `max(baseline × 2.0, minimum_meaningful_rate)`.

The doubled multiplier ensures the auto-handle threshold is more permissive than the human baseline, allowing some headroom for auto-handled paths to operate. The minimum meaningful rate (`DEFAULT_MINIMUM_MEANINGFUL_RATE = 0.01`) provides a floor below which threshold drift noise dominates the signal.

Review for accuracy. List any errors.
