# Test Skip Burndown

Last updated: 2026-02-14

This file tracks intentional test-skip debt reduction so `tests/.skip_baseline`
stays actionable and does not hide regression.

## Current Baseline

- Total skip markers: `94`
- Files with skips: `67`
- Source: `grep -rc "pytest.mark.skip\|@skip" tests/ --include="*.py"` (excluding conftest.py non-marker references)

### Category snapshot

| Category | Count | Weekly target |
|---|---:|---:|
| `optional_dependency` | 30 | -3 |
| `missing_feature` | 23 | -3 |
| `integration_dependency` | 22 | hold |
| `platform_specific` | 9 | hold |
| `performance` | 8 | -1 |
| `known_bug` | 2 | -1 |

### Category details

**`optional_dependency` (30)** -- skip when a Python package is not installed:

| Package | Count | Files |
|---|---:|---|
| cryptography | 4 | `test_encryption.py`, `security/test_encryption.py`, `security/test_encryption_security.py`, `benchmarks/test_encryption_performance.py` |
| aragora-debate | 3 | `server/handlers/test_playground.py` |
| numpy | 3 | `memory/test_vector_index.py`, `debate/cache/test_embeddings_lru.py` |
| pymilvus | 2 | `knowledge/mound/vector_abstraction/test_milvus.py` |
| aragora_sdk | 2 | `benchmarks/test_sdk_benchmarks.py` |
| qdrant-client | 2 | `integration/test_knowledge_vector_stores.py`, `knowledge/mound/vector_abstraction/test_vector_adapters.py` |
| faster-whisper | 2 | `transcription/test_whisper_backend.py` |
| Other (1 each) | 12 | playwright, FAISS, yt-dlp, openai, weaviate, pyotp, botbuilder, tree-sitter, psycopg2, ThinkPRM, web3, chromadb |

**`missing_feature` (23)** -- skip when a module or handler is not yet available:

| Reason | Count | Files |
|---|---:|---|
| Handler not available | 10 | 7 social handlers, 3 sme handlers |
| Module not available | 5 | `test_handlers_slack.py`, `test_handlers_whatsapp.py`, `test_handlers_audio.py` (3) |
| Feature not available | 4 | `test_handlers_tournaments.py` (4) |
| Other module skips | 4 | `test_handlers_evolution.py`, `test_rhetorical_integration.py`, `test_handlers_memory_analytics.py`, `test_visualization_replay.py` |

**`integration_dependency` (22)** -- skip when external service or env var is not configured:

| Dependency | Count | Files |
|---|---:|---|
| Redis | 5 | `test_control_plane_redis.py`, `test_distributed_integration.py`, `test_redis_ha.py`, `test_integration_store.py` (2) |
| Live server | 4 | `integration/test_knowledge_visibility_sharing.py` (4) |
| aragora-debate (ollama) | 1 | `test_debate_embeddings.py` |
| PostgreSQL | 2 | `integration/test_postgres.py`, `integration/test_postgres_stores.py` |
| Full server setup | 2 | `e2e/test_canvas_e2e.py` (2) |
| Env-gated | 8 | API keys, encryption key, integration flags, load test, stress test, RLM, SSO |

**`platform_specific` (9)** -- skip based on OS, Python version, or runtime:

| Condition | Count |
|---|---:|
| Windows (win32) | 5 |
| Fork start method | 1 |
| Python < 3.11 | 1 |
| /etc/passwd availability | 1 |
| CPython GIL limitation | 1 |

**`performance` (8)** -- skip slow or CI-flaky tests:

| Reason | Count |
|---|---:|
| CI environment flaky | 5 |
| Slow test (RUN_SLOW_TESTS) | 2 |
| NLI model loading timeout | 1 |

**`known_bug` (2)** -- skip due to known defects:

| Bug | Count |
|---|---:|
| aiohttp event loop incompatibility (#1234) | 1 |
| pytest-timeout plugin conflict with signal.alarm | 1 |

### High-skip files

No file currently has 10 or more skip markers. The highest-skip files are:

| File | Count | Categories |
|---|---:|---|
| `tests/test_plugin_sandbox.py` | 4 | platform_specific |
| `tests/test_handlers_tournaments.py` | 4 | missing_feature |
| `tests/integration/test_knowledge_visibility_sharing.py` | 4 | integration_dependency |
| `tests/transcription/test_whisper_backend.py` | 3 | optional_dependency |
| `tests/test_handlers_audio.py` | 3 | missing_feature |
| `tests/server/handlers/test_playground.py` | 3 | optional_dependency |

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

#### Week 2 burndown (completed 2026-02-14)

Removed **299 skip markers** (393 -> 94). Massive cleanup across all categories.

Major changes:
- `tests/test_formal.py`: -24 (all skips removed; Z3/formal verification tests now pass or converted to xfail)
- `tests/test_broadcast_pipeline_e2e.py`: -20 (all skips removed)
- `tests/test_formal_verification_backends.py`: -19 (all skips removed)
- `tests/e2e/test_security_api_e2e.py`: -17 (18 -> 1; security API tests now importable)
- `tests/test_handlers_plugins.py`: -16 (all skips removed)
- Remaining 203 skips removed across many files via dependency installation, module availability fixes, and stale-skip cleanup

Category changes (393 -> 94):
- `optional_dependency`: 178 -> 30 (-148)
- `missing_feature`: 157 -> 23 (-134)
- `integration_dependency`: 29 -> 22 (-7)
- `platform_specific`: 14 -> 9 (-5)
- `known_bug`: 12 -> 2 (-10)
- `performance`: 3 -> 8 (+5, reclassified CI-flaky tests from other categories)

No file now has >= 10 skip markers (previous high was 24). The `>=10` cleanup issue rule is fully satisfied.
