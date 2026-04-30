---
prompt_id: rtp_03b_prose_baseline_floor
class: red_team_paraphrase
seeded_error:
  description: "DEFAULT_MIN_BASELINE_SAMPLES misstated as 100. Actual: 50."
  verification_ref: "aragora/review/invalidation.py:99-101"
paraphrase_canonical_id: rtp_03_baseline_floor
surface_style: "discursive prose"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

The minimum-samples floor for the baseline calculation went through several iterations during design. The issue body proposed a range — "minimum 50 settled decisions; target 200" — and the team needed to pick a single floor below which the threshold-derivation function would refuse to emit a non-placeholder threshold. After some discussion they settled on 100 as the floor: more conservative than the issue's stated lower bound of 50, but well short of the target. The reasoning was that 50 samples produces a confidence interval too wide for any threshold change to be defensible; 100 narrows the interval enough to make a threshold revision actionable.

Review for accuracy. List any errors.
