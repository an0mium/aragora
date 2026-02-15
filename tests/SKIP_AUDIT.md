# Test Skip Marker Audit

**Generated**: 2026-02-15
**Total Skip Markers**: 134

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| missing_feature | 53 | 39.6% |
| integration_dependency | 29 | 21.6% |
| optional_dependency | 25 | 18.7% |
| platform_specific | 14 | 10.4% |
| known_bug | 9 | 6.7% |
| performance | 3 | 2.2% |
| uncategorized | 1 | 0.7% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `pytest.skip` | 94 |
| `skipif` | 40 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/sdk/test_openclaw_parity.py` | 15 |
| `tests/knowledge/mound/adapters/test_adapter_compliance.py` | 8 |
| `tests/sdk/test_contract_parity.py` | 8 |
| `tests/server/openapi/test_contract_matrix.py` | 7 |
| `tests/test_plugin_sandbox.py` | 6 |
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/e2e/test_document_pipeline.py` | 5 |
| `tests/rlm/test_true_rlm_priority.py` | 4 |
| `tests/transcription/test_whisper_backend.py` | 3 |
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 3 |

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

Current baseline: **134** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
