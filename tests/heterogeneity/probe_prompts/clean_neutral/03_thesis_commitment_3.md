---
prompt_id: cn_03_thesis_commitment_3
class: clean_neutral
seeded_error: null
expected_flags: 0
verification_refs:
  - "docs/THESIS.md (Commitment 3)"
  - "aragora/review/invalidation.py:14-22"
---

Commitment 3 of the Aragora thesis specifies an auto-handle outcome-invalidation threshold of 5%, explicitly noted as a placeholder pending empirical baseline measurement. Issue #6375 tracks the work to replace the placeholder with a measured baseline plus safety margin.

The `aragora/review/invalidation.py` module supplies pure-function machinery for this work: classification of invalidation signals, baseline computation, and threshold derivation. It composes with the `aragora/triage/metrics.py` event vocabulary rather than defining a parallel event type.

Review for accuracy. Flag errors if present.
