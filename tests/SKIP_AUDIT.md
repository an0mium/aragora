# Test Skip Marker Audit

**Generated**: 2026-07-05
**Total Skip Markers**: 77
**Justified Skip Markers**: 0
**Unjustified Skip Markers**: 77

---

## Justified Skip Convention

Use the skip reason prefix `justified-skip[category]: rationale` when a skip
is intentional and should not count against the unjustified skip baseline.

The category must be a short machine-readable token, and the rationale must
explain why the skip is intentionally retained at the skip site.

Example:

```python
@pytest.mark.skipif(not HAS_Z3, reason="justified-skip[optional_dependency]: Z3 solver not installed")
```

This v1 is report-only: unmarked skips remain visible as unjustified, and
the total skip count is still reported.

---

## Justified vs Unjustified

| Metric | Count |
|--------|-------|
| Total skip markers | 77 |
| Justified skip markers | 0 |
| Unjustified skip markers | 77 |

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

## Summary by Unjustified Category

| Category | Count | Percentage of Unjustified |
|----------|-------|---------------------------|
| integration_dependency | 29 | 37.7% |
| missing_feature | 17 | 22.1% |
| uncategorized | 12 | 15.6% |
| optional_dependency | 8 | 10.4% |
| platform_specific | 6 | 7.8% |
| performance | 4 | 5.2% |
| known_bug | 1 | 1.3% |

## Summary by Justification Category

| Justification Category | Count | Percentage of Justified |
|------------------------|-------|-------------------------|
| _none_ | 0 | 0.0% |

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

## Migration Readout Plan

1. Run `python scripts/audit_test_skips.py --json` to inspect existing
   unjustified skip markers.
2. Convert only reviewed, intentional skips by changing their reason to
   `justified-skip[category]: rationale`.
3. Do not auto-bless old skip markers without adding a local rationale.
4. Keep monitoring total skips and unjustified skips separately before any
   stronger enforcement is added.

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

Current unjustified baseline: **77** skips
Current total skip count: **77** skips

`tests/.skip_baseline` stores the unjustified skip baseline. CI still
reports total skips, but baseline arithmetic only uses unjustified skips.
Update `tests/.skip_baseline` only when intentionally adding an
unjustified skip.
