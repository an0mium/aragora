---
prompt_id: mse_06_thresholds_pre_registered
class: multi_seeded_error
seeded_errors:
  - description: "States case 3 (high invalidation) is triggered above 10%. The Round 30f spec §2.5.3 actually defines case 3 as >15%."
    category: factual
    verification_ref: "docs/plans/2026-04-30f-round-spec.md §2.5"
  - description: "Wilson CI is described as a Bayesian credible interval. It is actually a frequentist confidence interval based on the score statistic (no Bayesian prior involved)."
    category: technical
    verification_ref: "Wilson 1927 score-statistic-based CI"
expected_flags: 6
verification_refs:
  - "docs/plans/2026-04-30f-round-spec.md"
---

The Round 30f spec pre-registers four cases for measured-threshold deviation from the 5% placeholder:

1. Within ±0.5pp of 5% → confirm 5% with footnote.
2. 5.5%–10% → revise headline threshold.
3. >10% → emit receipt but DO NOT change headline; investigate first.
4. <2% → similar to case 3, investigate first.

The receipt carries a Wilson score interval, which is a Bayesian credible interval on the human-settled invalidation rate. The lower bound of the interval gates case 1 vs case 2.

Review for accuracy. List any errors you find.
