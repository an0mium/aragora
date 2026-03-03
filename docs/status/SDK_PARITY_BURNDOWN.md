# SDK Parity Burndown

Last updated: 2026-03-03

This document tracks weekly debt reduction against the parity budget in
`scripts/baselines/check_sdk_parity_budget.json`.

## Current Baseline

- Public routes found: `1811`
- Python SDK coverage: `99.9%`
- TypeScript SDK coverage: `99.9%`
- Missing from both SDKs: `0`
- Stale Python SDK paths: `0`
- Gate status:
  - `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json` -> pass
  - Budget status on 2026-03-03: `missing_from_both 0/22 | stale_python 0/20`

## Week 2 Plan (2026-03-03 to 2026-03-09)

### Missing-from-both wave

Target: maintain `0` (no regressions).

### Stale Python path wave

Completed in this wave: reduced stale Python paths to `0`.

Next target: maintain `0` stale paths while reducing one-sided handler gaps
reported by the strict parity checker (for example, SlackHandler and
UnifiedInboxHandler deltas where routes exist in only one SDK surface).

## Weekly Loop

1. Run parity report and save output:
   - `python scripts/check_sdk_parity.py --json > /tmp/sdk_parity.json`
2. Run strict gate with baseline + budget:
   - `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json`
3. Compare against this doc and budget file.
4. Update this doc with achieved reductions and next wave.
