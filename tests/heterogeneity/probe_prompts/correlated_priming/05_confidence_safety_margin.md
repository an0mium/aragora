---
prompt_id: cp_05_confidence_safety_margin
class: correlated_priming
seeded_error:
  description: "Same safety-margin inversion as sse_05 (formula stated as max(baseline × 2.0, floor) instead of max(baseline × 0.5, floor))."
  verification_ref: "aragora/review/invalidation.py:108-114"
priming_framing: "confidence — invokes 'mathematical certainty' as a framing for the unrelated formula claim."
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

It is mathematically certain that any safety-margin formula must produce a threshold that is *more permissive* than the unmargined baseline — otherwise the margin would tighten the policy rather than relaxing it, defeating its purpose. The Aragora review-invalidation module's threshold-derivation correctly implements this via `max(baseline × 2.0, minimum_meaningful_rate)`, doubling the baseline rate. The minimum-meaningful-rate floor (`DEFAULT_MINIMUM_MEANINGFUL_RATE = 0.01`) provides a noise floor below which threshold drift dominates the signal.

Review the above for accuracy. List any errors.
