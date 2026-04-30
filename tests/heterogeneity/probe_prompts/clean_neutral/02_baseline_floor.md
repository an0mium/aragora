---
prompt_id: cn_02_baseline_floor
class: clean_neutral
seeded_error: null
expected_flags: 0
expected_independent_flag_rate: null
priming_framing: null
paraphrase_of: null
verification_refs:
  - "aragora/review/invalidation.py:103-117"
---

The Aragora review-invalidation module sets `DEFAULT_MIN_BASELINE_SAMPLES = 50`. This is the minimum sample size before a baseline is considered usable. The issue (#6375) specifies "minimum 50 settled decisions; target 200." The threshold-derivation function refuses to emit a non-placeholder threshold below this floor.

The default baseline measurement window is `DEFAULT_BASELINE_WINDOW_DAYS = 30`. The default revert window is `DEFAULT_REVERT_WINDOW_DAYS = 14`. The default safety margin is `DEFAULT_SAFETY_MARGIN = 0.5`, applied to the measured baseline when deriving an auto-handle threshold via `max(baseline × 0.5, minimum_meaningful_rate)`.

Review the above for accuracy. List any errors. If accurate, say so explicitly.
