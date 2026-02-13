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

