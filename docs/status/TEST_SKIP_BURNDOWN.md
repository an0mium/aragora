# Test Skip Burndown

Last updated: 2026-02-13

This file tracks intentional test-skip debt reduction so `tests/.skip_baseline`
stays actionable and does not hide regression.

## Current Baseline

- Total skip markers: `434`
- Source: `python scripts/audit_test_skips.py --json`

### Category snapshot

| Category | Count | Weekly target |
|---|---:|---:|
| `missing_feature` | 183 | -8 |
| `optional_dependency` | 183 | -5 |
| `integration_dependency` | 28 | -2 |
| `uncategorized` | 18 | -4 |
| `platform_specific` | 13 | -1 |
| `known_bug` | 6 | -1 |
| `performance` | 3 | hold |

## Execution Rules

1. Keep `tests/.skip_baseline` aligned to audited reality after intentional skip changes.
2. Every sprint, reduce total skips by at least 15 unless blocked by external dependencies.
3. `uncategorized` skips must trend toward zero first.
4. For each high-skip file (`>=10` markers), create and track a dedicated cleanup issue.

## Weekly Loop

1. Audit: `python scripts/audit_test_skips.py --json > /tmp/skip-report.json`
2. Compare to table above and update this file.
3. Update `tests/.skip_baseline` only after documenting why counts changed.
4. Re-run CI gate locally:
   - `python scripts/audit_test_skips.py --count-only`

## Weekly Execution Log

### 2026-02-13 (Kickoff)

- Audit run completed:
  - `python scripts/audit_test_skips.py --json > /tmp/skip_report.json`
  - Result: `434` total markers
- Baseline check:
  - `tests/.skip_baseline` = `434` (in sync)

#### Week 1 focus files

| File | Current | Week 1 target |
|---|---:|---:|
| `tests/test_formal.py` | 24 | 20 |
| `tests/test_broadcast_pipeline_e2e.py` | 20 | 18 |
| `tests/test_formal_verification_backends.py` | 19 | 16 |
| `tests/e2e/test_security_api_e2e.py` | 18 | 16 |
| `tests/test_handlers_plugins.py` | 16 | 14 |

#### Week 1 category targets

| Category | Current | Week 1 target |
|---|---:|---:|
| `uncategorized` | 18 | 12 |
| `missing_feature` | 183 | 175 |
| `optional_dependency` | 183 | 178 |
