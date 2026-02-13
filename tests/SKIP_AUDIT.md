# Test Skip Marker Audit

**Generated**: 2026-02-13
**Total Skip Markers**: 434

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| missing_feature | 183 | 42.2% |
| optional_dependency | 183 | 42.2% |
| integration_dependency | 28 | 6.5% |
| uncategorized | 18 | 4.1% |
| platform_specific | 13 | 3.0% |
| known_bug | 6 | 1.4% |
| performance | 3 | 0.7% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `skipif` | 233 |
| `pytest.skip` | 201 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/test_formal.py` | 24 |
| `tests/test_broadcast_pipeline_e2e.py` | 20 |
| `tests/test_formal_verification_backends.py` | 19 |
| `tests/e2e/test_security_api_e2e.py` | 18 |
| `tests/test_handlers_plugins.py` | 16 |
| `tests/resilience/test_timeout.py` | 16 |
| `tests/sdk/test_openclaw_parity.py` | 15 |
| `tests/integration/test_handler_registration.py` | 12 |
| `tests/gauntlet/test_signing.py` | 12 |
| `tests/integration/test_security_hardening_e2e.py` | 10 |

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

Current baseline: **434** skips

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
