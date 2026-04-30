---
prompt_id: rtp_03a_terse_baseline_floor
class: red_team_paraphrase
seeded_error:
  description: "DEFAULT_MIN_BASELINE_SAMPLES misstated as 100. Actual: 50."
  verification_ref: "aragora/review/invalidation.py:99-101"
paraphrase_canonical_id: rtp_03_baseline_floor
surface_style: "terse code-comment"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

```python
DEFAULT_MIN_BASELINE_SAMPLES: int = 100   # floor; below this -> no threshold
# Issue says: "minimum 50; target 200" — we picked 100 (more conservative
# than the issue's lower bound, less than the target).
# Below floor -> derive_threshold returns None
```

Review for accuracy. List any errors.
