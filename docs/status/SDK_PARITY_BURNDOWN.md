# SDK Parity Burndown

Last updated: 2026-02-24

This document tracks weekly debt reduction against the parity budget in
`scripts/baselines/check_sdk_parity_budget.json`.

## Current Baseline

- Public routes found: `1626`
- Python SDK coverage: `99.3%`
- TypeScript SDK coverage: `96.6%`
- Missing from both SDKs: `0`
- Stale Python SDK paths: `78`
- Gate status:
  - `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json` -> pass
  - Budget status on 2026-02-24: `missing_from_both 0/0 | stale_python 78/78`

### Baseline note

On 2026-02-24, parity extraction was updated to include both
`self._client.request(...)` and `self._client._request(...)` calls,
including async `await` variants and single-quoted literals. This
increased measured stale Python paths from `43` to `78` by removing
under-counting in the checker.

## Week 1 Plan (2026-02-24 to 2026-03-02)

### Missing-from-both wave

Target: maintain `0` (no regressions).

### Stale Python path wave

Completed in this wave: maintained zero `missing_from_both` while
re-baselining stale-path measurement to `78` after parity extractor
hardening.

Next target: reduce `78 -> 73` (minimum weekly budget target).

Highest-volume stale clusters for next wave:
1. `/api/admin/security/*`
2. `/api/ap/*`
3. `/api/outcome-dashboard/*`
4. `/api/teams/*`
5. `/api/monitoring/*`

## Weekly Loop

1. Run parity report and save output:
   - `python scripts/check_sdk_parity.py --json > /tmp/sdk_parity.json`
2. Run strict gate with baseline + budget:
   - `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json`
3. Compare against this doc and budget file.
4. Update this doc with achieved reductions and next wave.
