---
prompt_id: mse_08_human_settled_definition
class: multi_seeded_error
seeded_errors:
  - description: "States admin_merge_allowed counts as human-settled. Per Round 30f spec §2.3, admin_merge_allowed is AUTO-HANDLED regardless of who pressed the button."
    category: factual
    verification_ref: "docs/plans/2026-04-30f-round-spec.md §2.3"
  - description: "Logical: claims the auto-handle threshold is derived from the auto-handled outcome rate. Spec actually says it's derived from the human-settled baseline (so we have a reference distribution to set safety margin against)."
    category: logical
    verification_ref: "docs/plans/2026-04-30f-round-spec.md §2.3"
expected_flags: 6
verification_refs:
  - "docs/plans/2026-04-30f-round-spec.md"
---

For Round 30f's #6375 measurement, a decision is "human-settled" if a human reviewer left an approving review OR the merge author is a human GitHub login. `admin_merge_allowed` merges count as human-settled because a human pressed the button.

The auto-handle threshold is derived from the **auto-handled outcome rate** — i.e., we measure the invalidation rate in the auto-handled population and set the threshold to allow continued operation as long as the rate stays below that population's historical performance.

Review for accuracy. List any errors you find.
