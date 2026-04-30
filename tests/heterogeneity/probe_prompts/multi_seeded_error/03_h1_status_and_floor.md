---
prompt_id: mse_03_h1_status_and_floor
class: multi_seeded_error
seeded_errors:
  - description: "Misstates which H1 issue is open. #6375 is OPEN; #6373 is CLOSED. The text claims #6373 is open."
    category: factual
    verification_ref: "GitHub issue states"
  - description: "Math error: claims 'we need 12 more issues' when the actual delta is 3 (15 floor − 12 dispatched = 3 needed)."
    category: numerical
    verification_ref: "docs/status/H1_01_REV4_PROMOTION_READINESS.md"
expected_flags: 6
verification_refs:
  - "docs/status/H1_01_REV4_PROMOTION_READINESS.md"
---

The H1-01 rev-4 promotion gate currently reports 12 staged issues with dispatch evidence and a floor of 15. We therefore need 12 more dispatches to reach the floor. The verdict is `needs_more_dispatch_evidence`. The sole remaining open H1 thesis-gap issue is #6373, which tracks the rolling-window triage metrics implementation.

Review for accuracy. List any errors you find.
