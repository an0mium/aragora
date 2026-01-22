# Test Skip Audit

This document tracks skipped tests and provides a remediation plan.

## Summary

- **Total skip markers:** ~620
- **Categories identified:** 12
- **Audit date:** January 2026

## Skip Categories

### 1. Optional External Dependencies (can mock or add to CI)

| Dependency | Count | Fix Strategy |
|------------|-------|--------------|
| z3-solver | ~48 | Add to CI optional deps, use skipif marker |
| httpx | 31 | Required - should be in main deps |
| redis | varies | Mock in tests, real in integration |
| asyncpg | 13 | Mock in unit tests |
| supabase | 13 | Mock in unit tests |
| websockets | 11 | Add to dev deps |
| PyJWT | 7 | Required for auth - add to deps |
| scikit-learn | varies | Add to optional ML deps |

### 2. Module Not Available (import failures)

| Module | Count | Fix Strategy |
|--------|-------|--------------|
| RLM | 24 | Fix import paths, ensure __init__.py exports |
| MatrixDebatesHandler | 24 | Fix handler registration |
| TournamentManager | 22 | Fix module exports |
| RBAC | 30 | Ensure RBAC module loads correctly |
| Calibration | 11 | Fix lazy loading |
| Graph orchestrator | 10 | Fix import chain |
| RhetoricalObserver | 9 | Fix module structure |
| Plugins | 9 | Fix plugin loader |
| Trickster | 7 | Fix import |

### 3. MCP Tests (29 skips)

These require MCP server running. Strategy:
- Mark as `@pytest.mark.integration`
- Run in nightly CI with MCP server

### 4. Embedding Service (xfail)

Tests that require real embedding service:
- Use mock embeddings for unit tests
- Real embeddings in integration tests only

## Centralized Skip Markers

Added to `tests/conftest.py`:

```python
from tests.conftest import (
    requires_z3, REQUIRES_Z3,
    requires_redis, REQUIRES_REDIS,
    requires_asyncpg, REQUIRES_ASYNCPG,
    requires_supabase, REQUIRES_SUPABASE,
    requires_httpx, REQUIRES_HTTPX,
    requires_websockets, REQUIRES_WEBSOCKETS,
    requires_pyjwt, REQUIRES_PYJWT,
    requires_sklearn, REQUIRES_SKLEARN,
    requires_mcp, REQUIRES_MCP,
    requires_rlm, REQUIRES_RLM,
    requires_rbac, REQUIRES_RBAC,
    requires_trickster, REQUIRES_TRICKSTER,
    requires_plugins, REQUIRES_PLUGINS,
)
```

## Usage Pattern

### Before (scattered skips):
```python
def test_something(self):
    try:
        import z3
    except ImportError:
        pytest.skip("Z3 not installed")
    # test code
```

### After (centralized):
```python
from tests.conftest import requires_z3, REQUIRES_Z3

@pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
class TestZ3Features:
    def test_something(self):
        # test code - no skip needed
```

## CI Configuration

### pyproject.toml additions:
```toml
[project.optional-dependencies]
test-full = [
    "z3-solver",
    "httpx",
    "websockets",
    "PyJWT",
    "scikit-learn",
    "mcp",
]
```

### GitHub Actions:
```yaml
# Fast CI (PRs)
- run: pytest -m "not slow and not integration"

# Full CI (nightly)
- run: pip install ".[test-full]"
- run: pytest
```

## Remediation Priority

1. **HIGH:** Fix module import issues (RLM, RBAC, etc.)
   - These indicate broken code paths, not just missing deps

2. **MEDIUM:** Add missing deps to requirements
   - httpx, websockets, PyJWT should be in main deps

3. **LOW:** Keep optional deps as skipif
   - z3-solver, scikit-learn are truly optional

## Progress Tracking

- [x] Created centralized skip markers in conftest.py
- [x] Updated test_formal_verification_backends.py (Z3 tests)
- [ ] Update MCP tests
- [ ] Update RLM tests
- [ ] Update RBAC tests
- [ ] Verify module imports work
- [ ] Add missing deps to pyproject.toml
- [ ] Update CI workflow

## Files Updated

| File | Status |
|------|--------|
| tests/conftest.py | Added skip markers |
| tests/test_formal_verification_backends.py | Using skipif |

## Verification Command

```bash
# Count remaining inline skips
grep -r "pytest.skip(" tests/ --include="*.py" | grep -v conftest | wc -l

# Run tests with skip summary
pytest --tb=no -q 2>&1 | tail -5
```
