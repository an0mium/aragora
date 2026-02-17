# Test Skip Marker Audit

**Generated**: 2026-02-17
**Total Skip Markers**: 109

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| missing_feature | 31 | 28.4% |
| integration_dependency | 29 | 26.6% |
| optional_dependency | 29 | 26.6% |
| platform_specific | 14 | 12.8% |
| performance | 3 | 2.8% |
| known_bug | 2 | 1.8% |
| uncategorized | 1 | 0.9% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `pytest.skip` | 59 |
| `skipif` | 39 |
| `pytest.importorskip` | 11 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/server/openapi/test_contract_matrix.py` | 7 |
| `tests/test_plugin_sandbox.py` | 6 |
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/rlm/test_true_rlm_priority.py` | 4 |
| `tests/transcription/test_whisper_backend.py` | 3 |
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 3 |
| `tests/server/startup/test_validation.py` | 3 |
| `tests/storage/test_redis_ha.py` | 3 |
| `tests/knowledge/mound/vector_abstraction/test_milvus.py` | 3 |
| `tests/e2e/test_canvas_e2e.py` | 3 |

---

## Category Definitions

| Category | Description |
|----------|-------------|
| optional_dependency | Missing optional Python package |
| missing_feature | Feature not yet implemented |
| integration_dependency | Requires external service (Redis, Postgres) |
| platform_specific | OS-specific limitation |
| flaky_test | Test has intermittent failures |
| known_bug | Known issue being tracked |
| performance | Too slow or resource-intensive |
| uncategorized | Reason did not match any pattern |

---

## Remediation Guidelines

1. **optional_dependency**: Add to `[project.optional-dependencies.test]` in pyproject.toml
2. **missing_feature**: Create GitHub issue and link in skip reason
3. **integration_dependency**: Ensure CI runs integration tests with services
4. **flaky_test**: Fix root cause or add retry mechanism
5. **known_bug**: Link to GitHub issue in skip reason
6. **uncategorized**: Review and add appropriate category pattern

---

## Skip Count Baseline

Current baseline: **109** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
