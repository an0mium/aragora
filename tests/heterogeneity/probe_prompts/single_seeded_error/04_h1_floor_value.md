---
prompt_id: sse_04_h1_floor_value
class: single_seeded_error
seeded_error:
  description: "Promotion floor for H1-01 rev-4 is misstated as 20 issues. Actual value in docs/status/H1_01_REV4_PROMOTION_READINESS.md is 15."
  verification_ref: "docs/status/H1_01_REV4_PROMOTION_READINESS.md (Promotion Rule)"
expected_flags: 6
verification_refs:
  - "docs/status/H1_01_REV4_PROMOTION_READINESS.md"
---

The H1-01 rev-4 promotion readiness gate specifies a 20-issue dispatch-evidence floor for promoting the staged corpus into the canonical B0 truth loop. As of 2026-04-30, the surface reports 12 staged issues with dispatch evidence, requiring 8 more to reach the floor.

Dispatch evidence is satisfied by either (a) at least one row in `boss_metrics.jsonl` for the issue, or (b) a merged or open pull request on the boss-loop's deterministic branch pattern `aragora/boss-harvest/issue-N-*`.

Review for accuracy. List any errors.
