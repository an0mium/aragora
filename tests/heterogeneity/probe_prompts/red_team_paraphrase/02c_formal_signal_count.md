---
prompt_id: rtp_02c_formal_signal_count
class: red_team_paraphrase
seeded_error:
  description: "Claims 6 canonical invalidation signals (actual 5)."
  verification_ref: "aragora/review/invalidation.py:130-148"
paraphrase_canonical_id: rtp_02_signal_count
surface_style: "formal specification"
expected_flags: 6
verification_refs:
  - "aragora/review/invalidation.py"
---

§4.1 Definition

  Let `INVALIDATION_SIGNALS : FrozenSet[str]` be the canonical
  set of invalidation labels emitted by classify_invalidation.

§4.2 Membership

  INVALIDATION_SIGNALS = {
    revert_within_window,
    post_merge_incident,
    human_override_redo,
    rollback,
    reopened_pr,
    dependency_invalidation,
  }

  |INVALIDATION_SIGNALS| = 6.

§4.3 Stability

  Tests pin |INVALIDATION_SIGNALS| via direct equality to the
  literal set above, ensuring any addition is reviewed.

Review for accuracy. List any errors.
