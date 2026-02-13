# Test Skip Burndown

Last updated: 2026-02-13

This file tracks intentional test-skip debt reduction so `tests/.skip_baseline`
stays actionable and does not hide regression.

## Current Baseline

- Total skip markers: `393`
- Source: `python scripts/audit_test_skips.py --json`

### Category snapshot

| Category | Count | Weekly target |
|---|---:|---:|
| `optional_dependency` | 178 | -5 |
| `missing_feature` | 157 | -8 |
| `integration_dependency` | 29 | -2 |
| `platform_specific` | 14 | -1 |
| `known_bug` | 12 | -1 |
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

#### Week 1 burndown (completed 2026-02-13)

Removed **41 stale skip markers** (434 -> 393). All removed tests verified passing.

Changes by file:
- `tests/integration/test_security_hardening_e2e.py`: -6 (RBAC now importable, RBAC_AVAILABLE=True)
- `tests/channels/test_dock_registry.py`: -4 (Slack/Telegram/Teams/Discord docks all importable)
- `tests/integration/test_backup_api.py`: -1 (BackupManager importable)
- `tests/server/handlers/social/`: -5 (all 5 handlers importable, redundant with module-level skipif)
- `tests/server/handlers/sme/`: -2 (SlackWorkspace/TeamsWorkspace handlers importable)
- `tests/billing/test_billing_models.py`: -7 (bcrypt installed)
- `tests/server/test_lifecycle.py`: -2 (prometheus_client installed)
- `tests/integration/test_notification_metrics.py`: -1 (aiohttp installed)
- `tests/broadcast/test_mixer.py`: -1 (pydub installed)
- `tests/performance/test_compression.py`: -2 (brotli installed)
- `tests/gauntlet/api/test_export.py`: -2 (WeasyPrint installed)
- `tests/storage/test_slack_workspace_store.py`: -1 (cryptography installed)
- `tests/storage/test_teams_workspace_store.py`: -2 (cryptography installed)
- `tests/scripts/test_verify_receipt.py`: -2 (cryptography installed)
- `tests/gauntlet/test_signing.py`: -1 (cryptography installed)
- `tests/server/middleware/test_distributed_rate_limit.py`: -1 (prometheus_client installed)
- `tests/workflow/test_schema.py`: -1 (Pydantic installed)

Category result: `uncategorized` reduced to **0** (target met).

#### Week 1 focus files (remaining)

| File | Current | Week 2 target |
|---|---:|---:|
| `tests/test_formal.py` | 24 | 20 |
| `tests/test_broadcast_pipeline_e2e.py` | 20 | 18 |
| `tests/test_formal_verification_backends.py` | 19 | 16 |
| `tests/e2e/test_security_api_e2e.py` | 18 | 16 |
| `tests/test_handlers_plugins.py` | 16 | 14 |

#### Week 2 category targets

| Category | Current | Week 2 target |
|---|---:|---:|
| `optional_dependency` | 178 | 173 |
| `missing_feature` | 157 | 149 |
| `known_bug` | 12 | 10 |
