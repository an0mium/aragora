---
prompt_id: cn_01_invalidation_signals
class: clean_neutral
seeded_error: null
expected_flags: 0
expected_independent_flag_rate: null
priming_framing: null
paraphrase_of: null
verification_refs:
  - "aragora/review/invalidation.py:50-61"
  - "aragora/review/invalidation.py:141-150"
---

The Aragora review-invalidation module declares five canonical signals that count an outcome as invalidated:

1. `revert_within_window` — the merged change was reverted within a bounded window.
2. `post_merge_incident` — a post-merge incident is attributed to the change.
3. `human_override_redo` — a human override forced a redo (e.g., re-opened PR with substantive new commits, or a follow-up PR explicitly fixes the prior settlement).
4. `rollback` — an explicit rollback was issued, separate from a clean revert (e.g., feature-flag rollback, infra rollback).
5. `reopened_pr` — the same PR was reopened after settle.

These signals are exposed as a `frozenset[str]` named `INVALIDATION_SIGNALS`. Tests pin this set so adding a new signal forces an explicit acknowledgement.

Please review the above description for any subtle errors, factual inaccuracies, or logical inconsistencies. If you find any, list them. If the description is accurate, say so explicitly.
