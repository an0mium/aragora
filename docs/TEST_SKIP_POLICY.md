# Test Skip Marker Policy

This document defines the policy for using `@pytest.mark.skip` markers in the Aragora test suite.

## Overview

Skip markers are essential for maintaining a green CI pipeline while allowing tests that depend on optional components. However, excessive or stale skips reduce test coverage confidence.

**Current baseline:** ~370 skip markers (tracked in `tests/.skip_baseline`)

## Categories

Skip markers are categorized by the audit script (`scripts/audit_test_skips.py`):

| Category | Purpose | Example |
|----------|---------|---------|
| `optional_dependency` | Missing optional package | `requires_z3`, `HAS_CRYPTO` |
| `missing_feature` | Feature not yet implemented | `requires_handlers` |
| `integration_dependency` | External service required | `requires_broadcast_e2e_api` |
| `platform_specific` | OS or Python version specific | `sys.platform == "win32"` |
| `known_bug` | Documented bug to fix | Links to GitHub issue |
| `performance` | Slow/resource-intensive | Load tests |

## Guidelines

### When to Use Skip Markers

**Use `@pytest.mark.skipif` for:**
- Optional dependencies not installed (`HAS_PSUTIL`, `HAS_REDIS`)
- External services unavailable (`requires_broadcast_e2e_api`)
- Platform-specific tests (`sys.platform != "linux"`)
- Python version requirements (`sys.version_info < (3, 11)`)

**Use `@pytest.mark.skip` for:**
- Known bugs with issue reference (temporary)
- Features actively under development

### When NOT to Use Skip Markers

- Flaky tests (fix the flakiness instead)
- Tests that "might break" (make them robust)
- Convenience during development (remove before merge)

### Required Format

Always include a clear reason:

```python
# Good - clear dependency
@pytest.mark.skipif(not HAS_Z3, reason="Z3 solver not installed")

# Good - issue reference
@pytest.mark.skip(reason="Known bug: GH-1234 - fix pending")

# Bad - unclear
@pytest.mark.skip(reason="Not working")
@pytest.mark.skip()  # No reason at all
```

## CI Enforcement

The `skip-audit` job in `.github/workflows/test.yml`:

1. Counts current skip markers
2. Compares against baseline (`tests/.skip_baseline`)
3. **Warns** if count increases by 1-5
4. **Fails** if count increases by >5

### Updating the Baseline

When adding legitimate new skips:

```bash
# Check current count
python scripts/audit_test_skips.py --count-only

# Review the changes
python scripts/audit_test_skips.py

# Update baseline (requires justification in commit message)
echo "NEW_COUNT" > tests/.skip_baseline
```

## Audit Commands

```bash
# Full audit report
python scripts/audit_test_skips.py

# Count only (used by CI)
python scripts/audit_test_skips.py --count-only

# List uncategorized skips (need review)
python scripts/audit_test_skips.py | grep uncategorized
```

## Reducing Skip Count

Priority for removing skips:

1. **Stale skips** (>6 months, feature now implemented)
2. **Known bugs** (fix the underlying issue)
3. **Uncategorized** (add proper category or remove)

Target: Maintain skip count within 10% of baseline.

## Current Distribution

As of 2026-02:

| Category | Count | Status |
|----------|-------|--------|
| optional_dependency | ~173 | Valid (environment-based) |
| missing_feature | ~130 | Valid (roadmap items) |
| integration_dependency | ~26 | Valid (external services) |
| uncategorized | ~16 | Needs review |
| known_bug | ~11 | Actionable tech debt |
| platform_specific | ~10 | Valid (OS compatibility) |
| performance | ~3 | Valid (resource limits) |

## Related Files

- `scripts/audit_test_skips.py` - Audit script
- `tests/.skip_baseline` - CI baseline count
- `.github/workflows/test.yml` - CI job definition (`skip-audit`)
