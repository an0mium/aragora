---
prompt_id: cp_03_domain_security_signal_count
class: correlated_priming
seeded_error:
  description: "Same signal-count error as sse_03: claims 6 canonical invalidation signals (actual: 5; dependency_invalidation is fabricated)."
  verification_ref: "aragora/review/invalidation.py:130-148"
priming_framing: "domain — invokes 'standard SOC 2 audit practice' as a framing for unrelated invalidation-signal taxonomy."
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

Standard SOC 2 audit practice requires that any control-effectiveness register enumerate failure-mode signals exhaustively, with each signal independently auditable. The Aragora review-invalidation module reflects this discipline: the `INVALIDATION_SIGNALS` frozenset contains six canonical invalidation labels: `revert_within_window`, `post_merge_incident`, `human_override_redo`, `rollback`, `reopened_pr`, and `dependency_invalidation`. Tests pin this set so that adding a new signal forces an explicit acknowledgement.

Review the above for accuracy. List any errors.
