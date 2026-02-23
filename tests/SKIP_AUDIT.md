# Test Skip Marker Audit

**Last Audited**: 2026-02-23
**Total `skipif` Markers**: 53 (across 42 files)
**Total `pytest.skip()` Calls**: ~80 (runtime conditional skips)
**Total `xfail` Markers**: 1
**Unconditional Skips**: 0

---

## Summary by Category

| Category | Count | Percentage | Status |
|----------|-------|------------|--------|
| optional_dependency | 29 | 26.6% | Correct -- truly optional packages |
| integration_dependency | 29 | 26.6% | Correct -- external services required |
| missing_feature | 31 | 28.4% | Reviewed -- all still unimplemented |
| platform_specific | 14 | 12.8% | Correct -- OS guards |
| performance | 3 | 2.8% | Correct -- resource-intensive |
| known_bug | 2 | 1.8% | Reviewed -- month-boundary bug still open |
| uncategorized | 1 | 0.9% | Reviewed -- legitimate |

## Audit Findings (Feb 23, 2026)

### No Stale Skips Found

All skip/xfail markers were reviewed. No skips reference features that have since
been fixed. All conditional skips remain valid for their stated reasons.

### xfail Review

| File | Reason | Status |
|------|--------|--------|
| `tests/integration/test_email_sync_integration.py:244` | `aragora.prioritization` not yet implemented | **Still valid** -- module does not exist |

### Skip Categories in Detail

#### Optional Dependencies (not in core requirements)

| Dependency | Files | Notes |
|------------|-------|-------|
| pydantic | `test_openapi_decorator.py` (13 skips) | Used for schema validation tests |
| FAISS | `test_vector_index.py` (1 skip) | Optional vector store backend |
| pymilvus | `test_milvus.py` (2 skipif + 1 runtime) | Milvus vector DB client |
| qdrant-client | `test_vector_adapters.py` (1 skip) | Qdrant vector DB client |
| chromadb | `test_vector_adapters.py` (1 skip) | Chroma vector DB client |
| tree-sitter | `test_code_intelligence.py` (1 runtime) | Code parsing |
| whisper/faster-whisper | `test_whisper_backend.py` (3 skips) | Audio transcription |
| PyNaCl | `test_github_sync.py` (1 runtime) | GitHub webhook verification |
| starlette | `test_spectate.py` (1 runtime) | WebSocket handler |
| ThinkPRM | `test_think_prm_integration.py` (1 skip) | PRM scoring model |

#### External Service Dependencies

| Service | Files | Notes |
|---------|-------|-------|
| Redis | `test_redis_ha.py`, `test_distributed_integration.py`, `test_integration_store.py`, `test_storage_token_blacklist.py`, `test_control_plane_redis.py`, `test_validation.py` | Requires running Redis |
| PostgreSQL | `test_postgres.py`, `test_postgres_stores.py`, `test_knowledge_visibility_sharing.py`, `test_validation.py` | Requires running PostgreSQL |
| Milvus | `test_milvus.py` | Requires running Milvus instance |
| API keys | `test_pr_review.py`, `test_document_pipeline.py` | Requires provider API keys |
| Full server | `test_canvas_e2e.py`, `test_sme_flow.py`, `test_server_smoke.py` | Requires running server |

#### Platform-Specific Guards

| Guard | Files | Notes |
|-------|-------|-------|
| Symlinks | `test_plugin_sandbox.py`, `test_sast_scanner_security.py`, `test_path_traversal.py`, `test_folder_scanner.py`, `test_openclaw_sandbox.py`, `test_custom.py` | OSError on Windows |
| Signals | `test_middleware_timeout.py`, `test_plugin_sandbox.py` | signal.alarm unavailable on Windows |
| CI env | `test_security_api_e2e.py`, `test_api_rate_limiting.py` | Tests not suited for CI |

#### Known Bugs

| Bug | File | Notes |
|-----|------|-------|
| Month-boundary cleanup | `test_store.py` (4 skips on days 1-2) | `_cleanup_old_sync` bug near month boundaries |

#### Async Context Detection

The OAuth handler tests (`test_google.py`, `test_oidc.py`, `test_microsoft.py`)
contain ~25 `pytest.skip("Running in async context")` calls. These detect when
sync code paths are unreachable due to running in an async event loop. This is
intentional -- the tests verify both sync and async code paths, skipping the
inapplicable one at runtime.

---

## Skip Count Baseline

Current baseline: **~109** conditional skips (per `conftest.py` threshold of 200)

The `SKIP_THRESHOLD` in `tests/conftest.py` is set to 200 to accommodate
parametrized contract matrix tests that skip based on missing SDK files.

## Remediation Guidelines

1. **optional_dependency**: Add to `[project.optional-dependencies.test]` in pyproject.toml if testing coverage is critical
2. **missing_feature**: Create GitHub issue and link in skip reason
3. **integration_dependency**: Ensure CI runs integration tests with services (nightly schedule)
4. **platform_specific**: No action needed -- these are permanent guards
5. **known_bug**: Fix root cause (month-boundary cleanup logic)
6. **async_context**: No action needed -- intentional dual-path testing
