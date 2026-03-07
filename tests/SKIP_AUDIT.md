# Test Skip Marker Audit

**Generated**: 2026-03-06
**Total Skip Markers**: 47

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| integration_dependency | 26 | 55.3% |
| missing_feature | 11 | 23.4% |
| platform_specific | 6 | 12.8% |
| optional_dependency | 2 | 4.3% |
| performance | 2 | 4.3% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `skipif` | 33 |
| `pytest.skip` | 14 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/test_plugin_sandbox.py` | 4 |
| `tests/integration/test_knowledge_visibility_sharing.py` | 4 |
| `tests/debate/test_voting_engine.py` | 3 |
| `tests/test_proofs.py` | 2 |
| `tests/test_broadcast_audio.py` | 2 |
| `tests/ranking/test_calibration_engine.py` | 2 |
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 2 |
| `tests/server/startup/test_validation.py` | 2 |
| `tests/storage/test_integration_store.py` | 2 |
| `tests/performance/test_load.py` | 2 |

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

Current baseline: **47** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
