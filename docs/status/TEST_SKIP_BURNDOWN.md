# Test Skip Burndown

Last updated: 2026-02-24

This file tracks intentional test-skip debt reduction so `tests/.skip_baseline`
stays actionable and does not hide regressions.

## Current Baseline

- Total skip markers: `112`
- Source command: `python scripts/audit_test_skips.py --json`
- CI baseline file: `tests/.skip_baseline` = `112`
- Marker types:
  - `pytest.skip`: `73`
  - `skipif`: `35`
  - `pytest.importorskip`: `4`

### Category Snapshot

| Category | Count | Weekly target |
|---|---:|---:|
| `missing_feature` | 40 | -5 |
| `integration_dependency` | 29 | hold |
| `optional_dependency` | 18 | -3 |
| `platform_specific` | 13 | hold |
| `uncategorized` | 7 | -3 |
| `performance` | 3 | hold |
| `known_bug` | 2 | -1 |

## Highest-Skip Files

| File | Count |
|---|---:|
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/server/openapi/test_contract_matrix.py` | 6 |
| `tests/sdk/test_typescript_exports.py` | 6 |
| `tests/test_plugin_sandbox.py` | 5 |
| `tests/sdk/test_openapi_sync.py` | 5 |

## Execution Rules

1. Keep `tests/.skip_baseline` synchronized with audited reality after intentional skip changes.
2. Reduce `uncategorized` first, then `missing_feature`, then `optional_dependency`.
3. Any file at `>=5` skips requires an owner and explicit cleanup plan in sprint notes.
4. Do not raise baseline without documenting root cause and expected payoff.

## Weekly Loop

1. Run audit:
   - `python scripts/audit_test_skips.py --json > /tmp/skip-report.json`
2. Review totals and category drift:
   - `jq '.total, .by_category, .high_skip_files[:10]' /tmp/skip-report.json`
3. Update this file and `tests/.skip_baseline` if counts changed intentionally.
4. Re-validate local gate:
   - `python scripts/audit_test_skips.py --count-only`
