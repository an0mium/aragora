---
prompt_id: sse_02_baseline_min_samples
class: single_seeded_error
seeded_error:
  description: "Baseline minimum-samples floor is misstated as 100. Actual value in aragora/review/invalidation.py:99-101 is 50."
  verification_ref: "aragora/review/invalidation.py:99-101"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py:99-101"
---

The Aragora review-invalidation module sets the minimum sample size before a baseline is considered usable to 100 settled decisions. This is the floor below which the threshold-derivation function refuses to emit a non-placeholder threshold. The constant is named `DEFAULT_MIN_BASELINE_SAMPLES` and the issue body specifies "minimum 50; target 200" — the implementation chose to be more conservative than the issue's lower bound.

The floor matters for #6375: until the data store accumulates at least this many human-settled decisions, the auto-handle threshold stays at the placeholder 5%.

Review the above. List any errors you find.
