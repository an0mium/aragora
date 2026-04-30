---
prompt_id: sse_10_brier_agent_priors
class: single_seeded_error
seeded_error:
  description: "States that the round-30b Brier leaderboard's agents (oracle-droid, bear-claude, dove-codex) are calibrated independent model-family predictors. Actually each is a fixed-prior synthetic agent (oracle-droid p=0.9 always, bear-claude p=0.4 always, dove-codex p=0.6 always) — they are role-named, not model-named, and they do not provide independent calibrated priors."
  verification_ref: "Round 30e reassessment notes; AGT-04 Brier leaderboard mechanism"
expected_flags: 6
verification_refs:
  - "Round 30e reassessment"
---

The round-30b AGT-04 Brier leaderboard tracks three calibrated agents: oracle-droid (a Droid agent with claude-opus-4-7 prior), bear-claude (a Claude agent with claude-sonnet-4-7 prior), and dove-codex (a Codex agent with gpt-5-codex prior). Each agent independently predicts merge probability for staged PRs and accumulates a Brier score over time, allowing model-family-level calibration tracking. After 2 markets resolved YES, oracle-droid leads with the lowest Brier loss.

Review for accuracy. List any errors.
