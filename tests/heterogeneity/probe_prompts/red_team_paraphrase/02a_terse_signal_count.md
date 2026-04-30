---
prompt_id: rtp_02a_terse_signal_count
class: red_team_paraphrase
seeded_error:
  description: "Claims 6 canonical invalidation signals (actual 5; dependency_invalidation is fabricated)."
  verification_ref: "aragora/review/invalidation.py:130-148"
paraphrase_canonical_id: rtp_02_signal_count
surface_style: "terse code-comment"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

```python
INVALIDATION_SIGNALS = frozenset({
    "revert_within_window",
    "post_merge_incident",
    "human_override_redo",
    "rollback",
    "reopened_pr",
    "dependency_invalidation",   # added in PR #6411
})  # 6 canonical labels; pin via tests
```

Review for accuracy. List any errors.
