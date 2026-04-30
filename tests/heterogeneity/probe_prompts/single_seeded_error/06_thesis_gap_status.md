---
prompt_id: sse_06_thesis_gap_status
class: single_seeded_error
seeded_error:
  description: "States that issue #6373 (rolling-window triage metrics) is the sole remaining open H1 gap. Actually #6373 is CLOSED; the sole remaining open H1 gap is #6375 (empirical threshold grounding)."
  verification_ref: "docs/status/STATUS.md (Current Frontier section)"
expected_flags: 6
verification_refs:
  - "docs/status/STATUS.md"
  - "GitHub issue #6373 (closed)"
  - "GitHub issue #6375 (open)"
---

Three of four H1 thesis-gap issues closed in late April 2026: auto-handle calibration + drift gating (#6372), source-of-truth alignment on PR-review path (#6374), and empirical threshold grounding (#6375). The sole remaining open H1 gap is #6373, which tracks the implementation of rolling-window triage metrics per Commitment 5: escalation rate, override rate, human-override-outcome correlation, and time-per-settlement.

Review for accuracy. List any errors.
