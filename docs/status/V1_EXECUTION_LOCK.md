# Aragora V1 Execution Lock

Last updated: 2026-02-25

## Purpose

This document locks execution scope for V1 and defines the branch hygiene and
test gates required before merge.

## V1 Product Scope (Locked)

Aragora V1 is an external **decision QA / analysis engine**, not a
participation-dependent public debate platform.

### In scope

- Claim extraction and structured adversarial critique
- Epistemic hygiene mode (`alternatives`, `falsifiers`, `confidence`, `unknowns`)
- Decision receipts with settlement metadata
- Settlement review scheduler and calibration updates from settled outcomes
- Operational observability and alerting for settlement/oracle reliability

### Out of scope for V1

- Consumer social/discussion surfaces
- Engagement-optimized ranking or virality loops
- Autonomous institutional participation assumptions
- New incentive/token market mechanisms beyond current guarded integrations

## Branch Hygiene Rules

Before each merge:

1. Split changes by concern:
   - `product-scope/docs`
   - `settlement-calibration-runtime`
   - `tests-and-fixtures`
2. Keep infra/doc drift in separate commits from runtime behavior changes.
3. Never merge mixed unrelated edits from concurrent sessions in one commit.

## Required Test Gates

Minimum required green checks for settlement/calibration changes:

1. `tests/test_debate_controller.py`
2. `tests/schedulers/test_settlement_review.py`
3. `tests/handlers/test_observability_dashboard.py`
4. `tests/debate/test_epistemic_hygiene.py`
5. `tests/scripts/test_check_epistemic_compliance_regression.py`
6. `python scripts/check_epistemic_compliance_regression.py --strict`

Additional targeted gate for LangChain compatibility changes:

1. `tests/integrations/langchain/test_callbacks.py`
2. `tests/integrations/langchain/test_chains.py`
3. `tests/integrations/langchain/test_tools.py`
4. `tests/integrations/test_langchain_integration.py`

## Forcing-Function Acceptance Criteria

Settlement -> calibration is valid only when:

1. A receipt reaches a settled outcome (`settled_true` / `settled_false`).
2. The receipt is in `epistemic_hygiene` mode.
3. Settlement metadata includes a recognized resolver type (`human`,
   `deterministic`, `oracle`).
4. Participating agents are present.

If criteria are missing, calibration must be explicitly marked as deferred or
skipped in settlement metadata.
