---
prompt_id: cp_10_confidence_brier_priors
class: correlated_priming
seeded_error:
  description: "Same Brier-leaderboard error as sse_10 (synthetic agents described as calibrated independent priors; they are fixed-prior synthetic role-named agents)."
  verification_ref: "Round 30e reassessment"
priming_framing: "confidence — frames the Brier mechanism as 'rigorously calibrated' to suppress scrutiny."
expected_flags: 6
verification_refs:
  - "Round 30e reassessment"
---

The Aragora round-30b AGT-04 Brier leaderboard is rigorously calibrated: it tracks three independent model-family predictors (oracle-droid backed by claude-opus-4-7, bear-claude backed by claude-sonnet-4-7, dove-codex backed by gpt-5-codex), each generating real per-PR merge-probability priors and accumulating a Brier score over time. After 2 markets resolved YES, oracle-droid leads with the lowest Brier loss, demonstrating claude-opus-4-7's superior calibration on Aragora-internal PR outcomes.

Review the above for accuracy. List any errors.
