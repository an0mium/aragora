# Skip Marker Burn-Down Plan

> Generated: 2026-02-23
> Current baseline: 122
> Target: reduce to < 100 by 2026-03-25 (Day 30)

## Category Breakdown

| Category | Count | Owner | Target Date | Strategy |
|----------|-------|-------|-------------|----------|
| optional_dependency | 36 | @team-platform | 2026-03-25 | Convert to xfail or add CI dep matrix |
| missing_feature | 35 | @team-core | 2026-03-25 | Implement features or convert to xfail |
| integration_dependency | 29 | @team-sre | 2026-03-25 | Add integration CI tier or mock |
| platform_specific | 14 | @team-platform | 2026-03-25 | Add OS matrix or document exemption |
| performance | 3 | @team-sre | 2026-03-25 | Add performance CI tier |
| uncategorized | 3 | @team-platform | 2026-03-25 | Categorize and assign |
| known_bug | 2 | @team-core | 2026-03-25 | Fix bugs or document timeline |

## High-Skip Files (prioritize)

| File | Skips | Action |
|------|-------|--------|
| `tests/server/openapi/test_contract_matrix.py` | 7 | Triage and reduce |
| `tests/test_plugin_sandbox.py` | 6 | Triage and reduce |
| `tests/integration/test_knowledge_visibility_sharing.py` | 6 | Triage and reduce |
| `tests/rlm/test_true_rlm_priority.py` | 4 | Triage and reduce |
| `tests/integration/test_upgrade_validation.py` | 3 | Triage and reduce |
| `tests/server/middleware/rate_limit/test_distributed_integration.py` | 3 | Triage and reduce |
| `tests/server/startup/test_validation.py` | 3 | Triage and reduce |
| `tests/nomic/test_self_improve_integration.py` | 3 | Triage and reduce |
| `tests/storage/test_redis_ha.py` | 3 | Triage and reduce |
| `tests/knowledge/mound/vector_abstraction/test_milvus.py` | 3 | Triage and reduce |

## Governance Rules (effective 2026-02-24)

1. **New skips** must include a ticket reference in the reason (GH-xxx, #xxx, or URL)
2. **Temporary skips** must include an expiry date: `reason="... expires=2026-04-01"`
3. **CI enforcement**: skip baseline threshold is +2; exceeding fails the build
4. **Quarterly review**: all skips reviewed quarterly with burn-down targets
5. **Expired skips**: CI warns on skips past their expiry date

## Burn-Down Milestones

| Date | Target | Notes |
|------|--------|-------|
| 2026-03-01 | 122 | Baseline established, governance rules active |
| 2026-03-15 | 110 | Phase 0 milestone: 12 skips resolved |
| 2026-03-25 | 100 | Day 30 target: -22 from baseline |
| 2026-04-24 | 85 | Day 60 target |
| 2026-05-24 | 70 | Day 90 target |

## How to Add a Skip Correctly

```python
# Good: ticket reference + expiry
@pytest.mark.skip(reason="Needs Redis cluster setup GH-456 expires=2026-04-15")

# Good: conditional with dependency reason (exempt from ticket requirement)
@pytest.mark.skipif(not HAS_Z3, reason="z3-solver not installed")

# Bad: no ticket reference for non-dependency skip
@pytest.mark.skip(reason="flaky test")  # Will be flagged

# Bad: no expiry on temporary skip
@pytest.mark.skip(reason="Blocked on GH-789")  # Add expires=YYYY-MM-DD
```
