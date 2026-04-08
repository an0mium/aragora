---
title: Next Steps (Canonical)
description: Next Steps (Canonical)
---

# Next Steps (Canonical)

Last updated: 2026-03-26

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](./canonical-goals) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](./aragora-evolution-roadmap) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](./feature-gap-list) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](./active-execution-issues) maps the live GitHub issue set and the current doc-driven PMF program.
[PMF_DOGFOOD_EXECUTION_PLAN](./pmf-dogfood-execution-plan) is the operator runbook for the next live proof.
[2026-03-26-pmf-14-day-execution-plan](./2026-03-26-pmf-14-day-execution-plan) is the current two-week operating tranche for inbox-wedge proof and design-partner readiness.

## Current Reality

- Historical program epics still matter as lineage even though they are no longer the live gate: [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806).
- `main` now contains the structural product-loop slices that were missing earlier in March, plus the live founder loop proof, Phase 2 truth-seeking wiring, and the inbox trust wedge dogfood surface.
- The focused test baseline on current `main`:

  ```bash
  python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
  ```

  Result on March 24, 2026: `71 passed` in `34.2s`.

  Extended suite (including truth_scorer, prover_estimator, cross_verification):

  ```bash
  python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py \
    tests/debate/test_truth_scorer.py tests/debate/test_prover_estimator.py \
    tests/debate/test_cross_verification.py -q
  ```

  Result: `125 passed` in `35.1s`.

- **The live founder loop is now proven repeatable.** Five consecutive live runs completed successfully on March 24, 2026 (35-62s range, all producing valid receipts). Receipts are now persisted to the receipt store, making them visible via the API (`/api/v2/receipts`), dashboard, and `aragora receipt list`.
- The acceptance checklist items that were open on March 23 are now closed:
  - Readiness: explicit, provider state shown before run starts (**passed**)
  - Quickstart enters live path or fails closed: no silent fallback (**passed**)
  - Live debate completes: 5/5 runs, 35-62s (**passed**)
  - Structured receipt saved: verified via `receipt inspect` and `receipt verify` (**passed**)
  - Result visible on product surface: receipts persist to store for API/dashboard (**passed**, commit 97074e28c)
  - KM ingestion: truthful explicit stop with guidance message (**passed**)
  - Operator noise bounded: embedding warnings demoted, summary preamble cleaned (**passed**)
- The next lane is **dogfooding the second workflow** (inbox trust wedge) and **design partner readiness**.
- GitHub's open issue set remains enterprise-assurance items: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), and [#509](https://github.com/synaptent/aragora/issues/509).

## Execution Order

### 1) ~~Prove The Canonical Founder Loop Live~~ DONE (March 24, 2026)

The canonical founder loop is proven repeatable on `main`:
- 5/5 consecutive live runs completed (35-62s range)
- All acceptance checklist items pass (see Current Reality above)
- Receipts persist to store for API/dashboard visibility
- Summary output is clean (preamble stripped, noise demoted)
- Commits: 5333ada7d, 97074e28c, 650f9c164

### 2) Dogfood The Second Workflow (Inbox Trust Wedge) — CURRENT GATE

The inbox trust wedge is structurally complete (~3,900 LOC) and the CLI is dogfood-ready:
- `aragora triage auth` — interactive Gmail OAuth flow (commit f045d653c)
- `aragora triage run --dry-run` — preview decisions without executing actions
- `aragora triage run --auto-approve` — full automated pipeline
- `aragora triage status` — shows configuration readiness

Remaining to dogfood:
- Configure Gmail OAuth credentials and run `aragora triage auth`
- Execute `aragora triage run --dry-run` on a real inbox
- Review proposed actions and verify receipt quality
- Execute a live triage batch and confirm receipt-gated actions work

### 3) Productize The Prompt-to-Spec Pipeline

`aragora spec` is proven end-to-end (~23s with gpt-4o-mini):
- Decomposes vague prompts into structured intents
- Generates specifications with success criteria and risk registers
- Supports `--skip-interrogation`, `--skip-research`, `--dry-run`

Remaining:
- Add `aragora spec` to the onboarding flow as a second entry point alongside `quickstart`
- Wire spec output into `aragora decide` for debate-driven validation
- Surface specs in the dashboard

### 4) Design Partner Outreach

The founder loop is repeatable. The sales point is now:
- Clean live demo: `aragora quickstart` produces a trustworthy receipt in &lt;60s
- Receipts are visible on API/dashboard/share-link surfaces
- EU AI Act compliance bundle generates from real receipts
- Inbox trust wedge provides a second workflow for retention testing

Use the [PMF_SCORECARD](./pmf-scorecard) to evaluate design partners.

### 5) Enterprise Assurance (After Design Partner Validation)

- [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), and [#509](https://github.com/synaptent/aragora/issues/509) are real work.
- Kickoff after at least 1 design partner is engaged and scoring above 65 on the PMF scorecard.

## Operating Rules

- Closed PMF issues do not equal live PMF proof.
- No new infra or orchestration lane is justified unless it maps directly to a founder-loop acceptance gap.
- No document should say "operational," "complete," or "ready for sales" unless live dogfood evidence supports that claim.
- The PMF backlog should be reconstituted from observed live failures, not from stale issue trees.
- GitHub issues still matter, but until the PMF blocker set is recreated truthfully, these docs are the current execution map.
