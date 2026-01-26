# Test Skip Marker Audit

**Generated**: 2026-01-26
**Total Skip Markers**: 598

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| missing_feature | 346 | 57.9% |
| optional_dependency | 193 | 32.3% |
| integration_dependency | 26 | 4.3% |
| known_bug | 12 | 2.0% |
| platform_specific | 9 | 1.5% |
| uncategorized | 7 | 1.2% |
| performance | 3 | 0.5% |
| flaky_test | 2 | 0.3% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `pytest.skip` | 367 |
| `skipif` | 224 |
| `skip` | 7 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/test_matrix_debates_integration.py` | 24 |
| `tests/test_formal.py` | 24 |
| `tests/server/handlers/test_workflows_handler.py` | 22 |
| `tests/test_handlers_tournaments_extended.py` | 20 |
| `tests/test_formal_verification_backends.py` | 20 |
| `tests/test_broadcast_pipeline_e2e.py` | 20 |
| `tests/rlm/test_compressor.py` | 18 |
| `tests/e2e/test_security_api_e2e.py` | 18 |
| `tests/test_handlers_plugins.py` | 16 |
| `tests/test_connectors_twitter.py` | 15 |

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

Current baseline: **598** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
