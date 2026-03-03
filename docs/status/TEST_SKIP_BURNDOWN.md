# Test Skip Burndown

Last updated: 2026-03-03

This file tracks intentional test-skip debt reduction so `tests/.skip_baseline`
stays actionable and does not hide regressions.

## Current Baseline

- Total skip markers: `54`
- Source command: `python scripts/audit_test_skips.py --json`
- CI baseline file: `tests/.skip_baseline` = `54`
- Marker types:
  - `skipif`: `34`
  - `pytest.skip`: `18`
  - `skip`: `2`

### Category Snapshot

| Category | Count | Weekly target |
|---|---:|---:|
| `missing_feature` | 12 | -2 |
| `integration_dependency` | 29 | hold |
| `platform_specific` | 6 | hold |
| `optional_dependency` | 4 | -1 |
| `performance` | 3 | hold |

## Highest-Skip Files

| File | Count |
|---|---:|
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/test_plugin_sandbox.py` | 4 |
| `tests/debate/test_voting_engine.py` | 3 |
| `tests/test_proofs.py` | 2 |
| `tests/test_broadcast_audio.py` | 2 |

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
