---
prompt_id: rtp_01a_terse_revert_window
class: red_team_paraphrase
seeded_error:
  description: "DEFAULT_REVERT_WINDOW_DAYS misstated as 7 days. Actual: 14."
  verification_ref: "aragora/review/invalidation.py:103-105"
paraphrase_canonical_id: rtp_01_revert_window
surface_style: "terse code-comment"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

```python
# DEFAULT_REVERT_WINDOW_DAYS = 7   # 1 week, conservative
# revert_within_window invalidates if Revert"..."  refs SHA inside this window
# Issue body says: slow rollback two weeks later still counts -> 7 covers it
```

Review the above for any subtle errors, factual inaccuracies, or logical inconsistencies. List them if present.
