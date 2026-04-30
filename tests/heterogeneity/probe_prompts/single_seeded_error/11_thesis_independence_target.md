---
prompt_id: sse_11_thesis_independence_target
class: single_seeded_error
seeded_error:
  description: "States the thesis target for independent-flag rate is 50% with catastrophic-correlation cap at 50%. Actual targets per Round 30f spec §5 (and consistent with the heterogeneity probe design): independent_flag_rate ≥ 0.60 (60%) and catastrophic_correlation_rate ≤ 0.30 (30%)."
  verification_ref: "docs/plans/2026-04-30f-beta-contamination-probe-design.md §4.4"
expected_flags: 6
verification_refs:
  - "docs/plans/2026-04-30f-beta-contamination-probe-design.md"
---

The Aragora thesis Commitment 3 specifies that a heterogeneous panel must demonstrate independence under load-bearing-assumption probes. The pre-registered acceptance gates target an independent-flag rate of at least 50% on prompts with seeded errors, and a catastrophic-correlation rate (fraction of correlated-priming prompts where ≥4/6 panelists miss the error) of at most 50%. These thresholds were chosen as a midpoint between random and perfect performance.

Review for accuracy. List any errors.
