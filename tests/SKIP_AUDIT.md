# Test Skip Marker Audit

**Generated**: 2026-07-23
**Total Skip Markers**: 84

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| integration_dependency | 29 | 34.5% |
| missing_feature | 18 | 21.4% |
| uncategorized | 17 | 20.2% |
| optional_dependency | 9 | 10.7% |
| platform_specific | 6 | 7.1% |
| performance | 4 | 4.8% |
| known_bug | 1 | 1.2% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `pytest.skip` | 40 |
| `skipif` | 37 |
| `pytest.importorskip` | 5 |
| `skip` | 2 |

## High-Skip Files (Top 10)

| File | Skip Count |
|------|------------|
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 |
| `tests/swarm/test_quorum_evidence.py` | 4 |
| `tests/plugins/test_plugin_sandbox.py` | 4 |
| `tests/debate/test_voting_engine.py` | 3 |
| `tests/ranking/test_calibration_engine.py` | 2 |
| `tests/inbox/test_inbox_receipt_convergence.py` | 2 |
| `tests/triage/test_auto_handle_calibration.py` | 2 |
| `tests/models/test_catalog.py` | 2 |
| `tests/storage/test_integration_store.py` | 2 |
| `tests/verification/test_proofs_root.py` | 2 |

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

Current baseline: **84** skips

The enforced value lives in `tests/.skip_baseline`. This regeneration lowers
the baseline from 85 to 84 after replacing obsolete missing-manifest skips
with fail-closed assertions for the committed proof-first claims fixture.

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
