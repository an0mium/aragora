# Test Skip Marker Audit

**Generated**: 2026-02-16
**Total Skip Markers**: 119

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| missing_feature | 38 | 31.9% |
| integration_dependency | 29 | 24.4% |
| optional_dependency | 26 | 21.8% |
| platform_specific | 14 | 11.8% |
| known_bug | 8 | 6.7% |
| performance | 3 | 2.5% |
| uncategorized | 1 | 0.8% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `pytest.skip` | 80 |
| `skipif` | 39 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/knowledge/mound/adapters/test_adapter_compliance.py` | 8 |
| `tests/sdk/test_contract_parity.py` | 8 |
| `tests/server/openapi/test_contract_matrix.py` | 7 |
| `tests/test_plugin_sandbox.py` | 6 |
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/e2e/test_document_pipeline.py` | 5 |
| `tests/rlm/test_true_rlm_priority.py` | 4 |
| `tests/transcription/test_whisper_backend.py` | 3 |
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 3 |
| `tests/server/startup/test_validation.py` | 3 |

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

Current baseline: **119** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
