# Skipped Tests Audit

> Last updated: 2026-02-23
>
> See also: `tests/SKIP_AUDIT.md` (canonical, concise version),
> `docs/testing/TEST_SKIP_POLICY.md` (policy and CI enforcement).

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Intentional (optional dependency) | 29 `skipif` + ~12 runtime | Keep |
| Intentional (platform) | 14 | Keep |
| Intentional (external service) | ~30 runtime | Keep |
| CI-specific | 6 `skipif` | Keep |
| Async context detection | ~27 runtime | Keep |
| Connector base class guards | ~25 runtime | Keep |
| Load test preconditions | ~17 runtime | Keep |
| Self-improvement integration | ~3 runtime | Keep |
| Data-dependent / parametrized | ~10 runtime | Keep |
| Stale | 0 | None to remove |

**Overall:** 0 unconditional `@pytest.mark.skip` decorators. All skips are conditional
(`@pytest.mark.skipif` or runtime `pytest.skip()` inside test bodies). The test suite
is healthy with no stale skips found.

**Totals:**
- `@pytest.mark.skipif` decorators: **44** across 28 files
- `pytest.skip()` runtime calls: **141** across 52 files
- `@pytest.mark.xfail`: **1** (still valid)
- `@pytest.mark.skip` (unconditional): **0**

---

## Intentional Skips

### Missing Optional Dependencies

These tests correctly skip when optional packages are not installed.

| Dependency | File(s) | Skip Type | Notes |
|------------|---------|-----------|-------|
| FAISS | `tests/memory/test_vector_index.py:188` | `skipif` | Vector index backend |
| pymilvus | `tests/knowledge/mound/vector_abstraction/test_milvus.py:51,152` | `skipif` | Milvus vector DB client |
| qdrant-client | `tests/knowledge/mound/vector_abstraction/test_vector_adapters.py:722` | `skipif` | Qdrant vector DB |
| chromadb | `tests/knowledge/mound/vector_abstraction/test_vector_adapters.py:1310` | `skipif` | Chroma vector DB |
| faster-whisper | `tests/transcription/test_whisper_backend.py:293` | `skipif` | Audio transcription |
| ThinkPRM | `tests/debate/test_think_prm_integration.py:245` | `skipif` | PRM scoring model |
| sentence-transformers (NLI) | `tests/debate/test_voting_engine.py:613,632` | `skipif` + runtime | NLI contradiction model |
| tree-sitter | `tests/analysis/test_code_intelligence.py:392` | runtime | Code parsing |
| PyNaCl | `tests/scheduler/test_github_sync.py:202` | runtime | GitHub webhook signing |
| starlette | `tests/handlers/debates/test_spectate.py:583` | runtime | SSE streaming response |
| whisper.cpp | `tests/transcription/test_whisper_backend.py:320,329` | runtime | Whisper binary check |
| RLM (official) | `tests/rlm/test_true_rlm_priority.py:115,125,130,140` | `skipif` + runtime | Official RLM package |
| BeliefNetwork | `tests/integration/test_km_bidirectional_flows.py:546` | runtime | Optional belief module |
| DocumentQueryEngine | `tests/test_nl_query.py:404` | runtime | External search deps |
| FFmpeg | `tests/test_broadcast_pipeline_e2e.py:258` | runtime | Video generation |

### Platform-Specific

| Guard | File(s) | Skip Type | Notes |
|-------|---------|-----------|-------|
| Symlinks unavailable | `tests/modes/test_custom.py:367`, `tests/analysis/codebase/test_sast_scanner_security.py:322`, `tests/security/test_path_traversal.py:122`, `tests/documents/test_folder_scanner.py:215`, `tests/gateway/test_openclaw_sandbox.py:567,588`, `tests/test_plugin_sandbox.py:198` | runtime + `skipif` | Windows/restricted FS |
| Windows (no signals) | `tests/test_plugin_sandbox.py:548`, `tests/server/middleware/test_timeout.py:506`, `tests/test_middleware_timeout.py:332` | `skipif` + runtime | `signal.alarm` unavailable |
| Windows-only tests | `tests/test_plugin_sandbox.py:559` | `skipif` | Windows resource limits |
| Fork unavailable | `tests/memory/test_tier_ops_concurrency.py:416` | `skipif` | Multiprocessing start method |
| `/etc/hostname` missing | `tests/test_plugin_sandbox.py:192` | runtime | Container detection |
| pytest-timeout conflict | `tests/test_middleware_timeout.py:336` | `skipif` | signal.alarm conflict |
| Signal timeout unreliable | `tests/server/middleware/test_timeout.py:511` | runtime | Always skips (see note below) |

**Note on `test_timeout_raises_on_unix`** (`tests/server/middleware/test_timeout.py:496-514`):
This test is always skipped even on Unix because `signal.alarm` only works with integer
seconds on the main thread. The test cannot be reliably executed in a pytest environment.
This is intentional -- the functionality works in production. The test body exists as
documentation of expected behavior.

### External Services Required

| Service | File(s) | Skip Type | Notes |
|---------|---------|-----------|-------|
| Redis | `tests/storage/test_redis_ha.py:622,635,652`, `tests/server/middleware/rate_limit/test_distributed_integration.py:61,66,177`, `tests/storage/test_integration_store.py:249,631`, `tests/test_storage_token_blacklist.py:356`, `tests/storage/test_gmail_token_store.py:262`, `tests/server/startup/test_validation.py:314`, `tests/storage/test_federation_registry_store.py:349`, `tests/storage/test_finding_workflow_store.py:328` | `skipif` + runtime | Requires running Redis |
| PostgreSQL | `tests/integration/test_postgres.py:37`, `tests/integration/test_postgres_stores.py:55`, `tests/integration/conftest.py:568,593`, `tests/server/startup/test_validation.py:346,366`, `tests/ranking/test_calibration_engine.py:721,747` | runtime | Requires `DATABASE_URL` |
| Milvus | `tests/knowledge/mound/vector_abstraction/test_milvus.py:56` | runtime | Requires running Milvus |
| Running server | `tests/integration/test_knowledge_visibility_sharing.py:945,955`, `tests/e2e/test_canvas_e2e.py:975`, `tests/e2e/test_server_smoke.py:203,216` | runtime | Requires `ARAGORA_TEST_SERVER_URL` |
| API keys | `tests/test_pr_review.py:404`, `tests/e2e/test_document_pipeline.py:173` | `skipif` + runtime | Requires provider API keys |
| Integration flag | `tests/e2e/test_sme_flow.py:318` | `skipif` | Requires `ARAGORA_INTEGRATION_TESTS=1` |

---

## CI-Specific Skips

These tests are skipped when running in CI (`GITHUB_ACTIONS` or `CI` env vars set).

| File | Reason |
|------|--------|
| `tests/test_broadcast_audio.py:71,535` | Edge TTS tests fail in CI |
| `tests/performance/test_load.py:376,598` | Performance/stress tests too flaky on CI runners |
| `tests/e2e/test_api_rate_limiting.py:230` | Rate limiter not triggering 429 in CI |
| `tests/e2e/test_security_api_e2e.py:401` | Security hardening not available in CI |

---

## Async Context Detection Skips

The OAuth handler tests detect at runtime whether sync code paths are reachable.
When running in an async event loop (as pytest-asyncio does), the sync paths
return coroutines rather than results, so those test paths skip gracefully.

| File | Count | Notes |
|------|-------|-------|
| `tests/handlers/_oauth/test_google.py` | 8 | Google OAuth sync/async dual paths |
| `tests/handlers/_oauth/test_microsoft.py` | 16 | Microsoft OAuth sync/async dual paths |
| `tests/handlers/_oauth/test_oidc.py` | 3 | OIDC sync/async dual paths |

**Total:** ~27 runtime skips. All intentional -- tests verify both sync and async
paths and skip the inapplicable one.

---

## Connector Base Class Guards

`tests/connectors/base_connector_test.py` contains 25 runtime `pytest.skip()` calls
that guard against connectors not implementing optional methods (`search`, `list`,
`create`, `get`, `delete`, `exchange_code`, `get_authorization_url`). These are
correct -- the base test class is parametrized across all connectors and skips
methods that a specific connector does not support.

---

## Load Test Preconditions

`tests/load/` files (websocket_load.py, auth_load.py, gauntlet_load.py, knowledge_load.py)
contain 17 runtime `pytest.skip()` calls that check preconditions like successful
connections, API availability, and completed operations. These correctly skip when
the target server is not running.

---

## Data-Dependent / Order-Dependent Skips

| File | Skip | Reason |
|------|------|--------|
| `tests/security/test_tenant_isolation_audit.py:604,618` | Categories/report not populated | Test ran before populating tests in random order |
| `tests/handlers/test_gauntlet_v1.py:824,878` | No templates registered | Gauntlet template list empty |
| `tests/handlers/test_audit_bridge.py:909,1168,1202` | `audit_sessions` not available | ImportError catch |
| `tests/handlers/test_canvas_pipeline.py:899,971` | Pipeline import unavailable | ImportError in test env |
| `tests/test_handler_registry.py:174` | Handler not available | Dynamic handler check |
| `tests/server/test_version_headers.py:301` | Module not importable | Optional dependency |
| `tests/nomic/test_codebase_indexer.py:133,690` | Source file not found | Real file existence check |
| `tests/nomic/test_self_improve_integration.py:169,254,272` | Scanner unavailable / no candidates | Integration env check |
| `tests/agents/test_credential_validator.py:122,132` | Valid credentials / not enough unavailable agents | Environment-dependent |
| `tests/test_defaults_alignment.py:29` | Server defaults overridden via env | Env vars would break comparison |

---

## Environment Variable Gated Skips

| Env Var | File(s) | Purpose |
|---------|---------|---------|
| `RUN_GIL_TIMEOUT_TESTS` | `tests/test_verification.py:99` | CPython GIL timeout test |
| `RUN_SLOW_TESTS=1` | `tests/test_proofs.py:190,539` | Thread-based timeout tests |
| `DATABASE_URL` | `tests/integration/test_postgres.py:37`, `tests/integration/conftest.py:568` | PostgreSQL integration |
| `ARAGORA_INTEGRATION_TESTS=1` | `tests/e2e/test_sme_flow.py:318` | Full integration tests |
| `ARAGORA_TEST_SERVER_URL` | `tests/integration/test_knowledge_visibility_sharing.py:945` | Running server URL |
| `ARAGORA_TEST_AUTH_TOKEN` | `tests/integration/test_knowledge_visibility_sharing.py:955` | Auth token for server |
| `RUN_FUTURE_API_TESTS` | `tests/integration/test_email_sync_integration.py:244` | xfail gate |
| `REDIS_URL` | `tests/storage/test_federation_registry_store.py`, `tests/storage/test_finding_workflow_store.py`, `tests/storage/test_integration_store.py` | Redis connection |

---

## xfail Markers

| File | Line | Reason | Status |
|------|------|--------|--------|
| `tests/integration/test_email_sync_integration.py` | 244 | `aragora.prioritization` module not yet implemented | **Still valid** -- module does not exist |

---

## Stale Skips (Can Be Removed)

**None found.** All skip markers were reviewed and are currently valid for their stated
reasons. There are zero unconditional `@pytest.mark.skip` decorators in the test suite
(enforced by `UNCONDITIONAL_SKIP_THRESHOLD = 0` in `tests/conftest.py`).

---

## Skip Count Monitoring

The test suite enforces skip count thresholds in `tests/conftest.py`:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `SKIP_THRESHOLD` | 200 | Warn if total skips exceed this (accommodates parametrized contract matrix) |
| `UNCONDITIONAL_SKIP_THRESHOLD` | 0 | No unconditional `@pytest.mark.skip` allowed |

CI enforcement is handled by the `skip-audit` job in `.github/workflows/test.yml`
using the audit script at `scripts/audit_test_skips.py`.

---

## Recommendations

1. **No action required** -- all skips are intentional and correctly guarded.
2. The always-skipped `test_timeout_raises_on_unix` in `tests/server/middleware/test_timeout.py:496`
   could be converted to a documentation comment or removed entirely, since it never
   executes. However, keeping it as a skipped test documents the expected behavior.
3. The 16 async-context skips in `test_microsoft.py` could potentially be reduced by
   restructuring into separate sync-only and async-only test classes.
4. Consider running integration tests with external services (Redis, PostgreSQL) in
   a nightly CI schedule to exercise the ~30 service-dependent skips.
