# Test Skip Marker Audit

**Generated**: 2026-07-04
**Total Skip Markers**: 77

---

## Summary by Category

| Category | Count | Percentage |
|----------|-------|------------|
| integration_dependency | 29 | 37.7% |
| missing_feature | 17 | 22.1% |
| uncategorized | 12 | 15.6% |
| optional_dependency | 8 | 10.4% |
| platform_specific | 6 | 7.8% |
| performance | 4 | 5.2% |
| known_bug | 1 | 1.3% |

## Summary by Marker Type

| Type | Count |
|------|-------|
| `skipif` | 36 |
| `pytest.skip` | 35 |
| `pytest.importorskip` | 4 |
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
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 2 |
| `tests/server/startup/test_validation.py` | 2 |
| `tests/triage/test_auto_handle_calibration.py` | 2 |
| `tests/storage/test_integration_store.py` | 2 |

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

Current baseline: **77** skips

Baseline history: the enforced value lives in `tests/.skip_baseline`
(68 → 75 in PR #8800, 75 → 77 in the PR that regenerated this file on
2026-07-04; each bump ships a per-skip audit table of every net-new
marker in that PR's description). Note on doc-vs-file history: this
generated document goes stale between bumps — it was regenerated on
2026-04-06 when the count was 57, so this file's diff once jumped 57→75;
that reflected staleness, not 18 unaudited skips. The enforced baseline
has only ever moved 68 → 75 → 77.

CI will warn if skip count exceeds this baseline.
Update `tests/.skip_baseline` when intentionally adding skips.
