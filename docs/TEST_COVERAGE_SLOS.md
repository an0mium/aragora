# Test Coverage SLOs

Service Level Objectives for test coverage across Aragora components.

## Overview

This document defines minimum test coverage targets per component, ensuring quality and reliability across the codebase.

**Current Status:** 45% overall coverage | 40,470 tests | 1,500 test files

## Coverage Targets by Quarter

| Quarter | Target | Notes |
|---------|--------|-------|
| Q4 2025 | 45% | Baseline established |
| Q1 2026 | 50% | CI enforcement enabled |
| Q2 2026 | 70% | Enterprise-grade coverage |
| Q3 2026 | 80% | Production hardening |

## Component-Level SLOs

### Tier 1: Critical Path (80% minimum)

Components that directly affect debate outcomes and user trust.

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| `aragora/debate/` | 80% | - | P0 |
| `aragora/consensus/` | 80% | - | P0 |
| `aragora/auth/` | 80% | - | P0 |
| `aragora/rbac/` | 80% | - | P0 |
| `aragora/verification/` | 80% | - | P0 |

**Rationale:** These components handle core business logic, authentication, and authorization. Failures here directly impact users.

### Tier 2: Enterprise Features (70% minimum)

Components required for enterprise deployments.

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| `aragora/tenancy/` | 70% | - | P1 |
| `aragora/billing/` | 70% | - | P1 |
| `aragora/control_plane/` | 70% | - | P1 |
| `aragora/backup/` | 70% | - | P1 |
| `aragora/gauntlet/` | 70% | - | P1 |
| `aragora/observability/` | 70% | - | P1 |

**Rationale:** Enterprise customers depend on these for production deployments.

### Tier 3: Core Infrastructure (60% minimum)

Components providing foundational capabilities.

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| `aragora/agents/` | 60% | - | P2 |
| `aragora/memory/` | 60% | - | P2 |
| `aragora/knowledge/` | 60% | - | P2 |
| `aragora/ranking/` | 60% | - | P2 |
| `aragora/workflow/` | 60% | - | P2 |
| `aragora/rlm/` | 60% | - | P2 |
| `aragora/server/` | 60% | - | P2 |

### Tier 4: Integrations (50% minimum)

External integrations and connectors.

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| `aragora/connectors/` | 50% | - | P3 |
| `aragora/integrations/` | 50% | - | P3 |
| `aragora/server/handlers/social/` | 50% | - | P3 |

### Tier 5: Utilities (40% minimum)

Helper functions and utilities.

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| `aragora/utils/` | 40% | - | P4 |
| `aragora/client/` | 40% | - | P4 |

## Test Type Distribution

Each component should maintain this approximate test distribution:

| Test Type | Percentage | Description |
|-----------|------------|-------------|
| Unit Tests | 70% | Isolated function/class tests |
| Integration Tests | 20% | Component interaction tests |
| E2E Tests | 10% | Full system tests |

## Enforcement

### CI/CD Pipeline

```yaml
# Current enforcement in .github/workflows/test.yml
pytest tests/ --cov=aragora --cov-fail-under=50 -n auto
```

### Pre-commit Hook (Recommended)

```bash
# Run before pushing
pytest tests/ -m "not slow" --cov=aragora --cov-fail-under=45
```

### Coverage Reports

Generated artifacts:
- HTML: `htmlcov/index.html`
- XML: `coverage.xml` (for Codecov)
- Terminal: Summary with missing lines

## Excluded from Coverage

The following are excluded from coverage calculations:

```python
# pyproject.toml [tool.coverage.run]
omit = [
    "tests/*",
    "*/__pycache__/*",
    "aragora/live/*",        # Frontend code
    "*/migrations/*",         # DB migrations
]
```

Excluded patterns:
- `pragma: no cover` annotations
- `def __repr__` methods
- `raise NotImplementedError`
- Type checking blocks
- Main entry points

## Monitoring

### Weekly Coverage Report

Generate with:
```bash
pytest tests/ --cov=aragora --cov-report=html --cov-report=xml
open htmlcov/index.html
```

### Component Coverage Script

```bash
# Check coverage for a specific component
pytest tests/debate/ --cov=aragora.debate --cov-report=term-missing
```

### Skip Audit

Maximum allowed skipped tests: **600** (baseline)

Check current skips:
```bash
pytest tests/ --collect-only -q | grep "skipped" | wc -l
```

## Improvement Strategy

### Phase 1: Stabilize (Q1 2026)
- Achieve 50% overall coverage
- Focus on Tier 1 components
- Reduce flaky test count

### Phase 2: Enterprise (Q2 2026)
- Achieve 70% overall coverage
- Complete Tier 2 coverage targets
- Add integration test suite

### Phase 3: Harden (Q3 2026)
- Achieve 80% overall coverage
- Complete Tier 3 coverage targets
- Add performance regression tests

## Test Organization Best Practices

### File Naming
```
tests/
├── debate/
│   ├── test_orchestrator.py      # Unit tests
│   ├── test_consensus.py
│   └── integration/
│       └── test_full_debate.py   # Integration tests
├── e2e/
│   └── test_debate_flow.py       # E2E tests
└── conftest.py                   # Shared fixtures
```

### Markers

Use pytest markers for test categorization:
```python
import pytest

@pytest.mark.unit
def test_consensus_calculation():
    ...

@pytest.mark.integration
def test_debate_with_agents():
    ...

@pytest.mark.slow
def test_large_debate():
    ...
```

### Fixtures

Prefer fixtures over setup/teardown:
```python
@pytest.fixture
def arena():
    return Arena(config=test_config)

def test_debate_runs(arena):
    result = arena.run()
    assert result.consensus_reached
```

## Reporting Issues

If coverage drops below SLO:
1. Check CI/CD for failing coverage check
2. Identify uncovered code paths
3. Add tests or mark as `pragma: no cover` with justification
4. Update component coverage in this document

## Version History

| Date | Version | Change |
|------|---------|--------|
| 2026-01-25 | 1.0 | Initial SLO definition |
