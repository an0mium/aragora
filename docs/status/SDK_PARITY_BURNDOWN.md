# SDK Parity Burndown

Last updated: 2026-02-13

This document tracks weekly debt reduction against the parity budget in
`scripts/baselines/check_sdk_parity_budget.json`.

## Current Baseline

- Public routes found: `942`
- Python SDK coverage: `86.4%`
- TypeScript SDK coverage: `82.1%`
- Missing from both SDKs: `118`
- Stale Python SDK paths: `752`
- Gate status:
  - `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json` -> pass

## Week 1 Plan (2026-02-13 to 2026-02-20)

### Missing-from-both wave

Target: reduce `118 -> 117` (minimum weekly budget target).

Candidate routes for first reduction wave:

1. `/api/audit/findings/{finding_id}/assign`
2. `/api/audit/findings/{finding_id}/comments`
3. `/api/audit/findings/{finding_id}/priority`
4. `/api/audit/findings/{finding_id}/status`
5. `/api/audit/findings/{finding_id}/unassign`

### Stale Python path wave

Target: reduce `752 -> 747` (minimum weekly budget target).

Candidate stale Python paths to retire/alias first:

1. `/api/accounting/expenses`
2. `/api/accounting/expenses/categorize`
3. `/api/accounting/expenses/export`
4. `/api/accounting/expenses/pending`
5. `/api/accounting/expenses/stats`

## Weekly Loop

1. Run parity report and save output:
   - `python scripts/check_sdk_parity.py --json > /tmp/sdk_parity.json`
2. Compare against this doc and budget file.
3. Update this doc with achieved reductions and next wave.
4. Keep `scripts/baselines/check_sdk_parity.json` for no-regression list only.

