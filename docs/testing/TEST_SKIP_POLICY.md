# Test Skip Marker Policy

This document defines the policy for using `@pytest.mark.skip` markers in the Aragora test suite.

## Overview

Skip markers are essential for maintaining a green CI pipeline while allowing tests that depend on optional components. However, excessive or stale skips reduce test coverage confidence.

**Current baseline:** tracked in `tests/.skip_baseline`. As of the justified-skip
v1 redesign, this file stores the **unjustified** skip baseline; total skips are
still reported separately.

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

### Justified Skip Convention

Use the report-only convention `justified-skip[category]: rationale` when a skip is
intentional and should not count against the unjustified skip baseline.

```python
@pytest.mark.skipif(
    not HAS_Z3,
    reason="justified-skip[optional_dependency]: Z3 solver not installed",
)
```

The category is a short machine-readable token. The rationale must explain why
the skip remains intentional at the skip site. Existing skips are **not**
auto-blessed: convert them only after reviewing the skip and adding a local
rationale. This v1 remains report-only beyond the existing baseline arithmetic;
it does not fail builds merely because a skip lacks the convention.

## CI Enforcement

The `skip-audit` job in `.github/workflows/test.yml`:

1. Counts total skip markers for reporting
2. Counts unjustified skip markers for baseline arithmetic
3. Compares unjustified skips against baseline (`tests/.skip_baseline`)
4. **Warns** if unjustified skips increase by 1-2
5. **Fails** if unjustified skips increase by >2

### Updating the Baseline

When adding legitimate new skips:

```bash
# Check current count
python scripts/audit_test_skips.py --count-only
python scripts/audit_test_skips.py --unjustified-count-only

# Review the changes
python scripts/audit_test_skips.py

# Update unjustified baseline (requires justification in commit message)
echo "NEW_COUNT" > tests/.skip_baseline
```

## Audit Commands

```bash
# Full audit report
python scripts/audit_test_skips.py

# Count only (used by CI)
python scripts/audit_test_skips.py --count-only

# Unjustified count only (used by baseline arithmetic)
python scripts/audit_test_skips.py --unjustified-count-only

# List uncategorized skips (need review)
python scripts/audit_test_skips.py | grep uncategorized
```

## Migration Readout

Before tightening enforcement, run `python scripts/audit_test_skips.py --json` and
review the `unjustified_total`, `by_unjustified_category`, and
`by_justification_category` fields. Convert only reviewed intentional skips to
`justified-skip[category]: rationale`; leave uncertain skips unjustified so they
remain visible in the baseline.

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
