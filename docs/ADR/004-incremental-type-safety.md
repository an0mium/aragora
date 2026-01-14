# ADR-004: Incremental Type Safety Migration

## Status
Accepted

## Context

The Aragora codebase grew to 270,000+ lines with inconsistent type annotation coverage. Full strict typing would require massive effort and risk introducing bugs during migration.

Challenges:
- 664 Python files with varying type coverage
- Third-party integrations with complex/missing types
- Dynamic patterns (plugin loading, agent dispatch)
- Test velocity would suffer during full migration

## Decision

Adopt an incremental type safety strategy with three tiers in mypy configuration:

### Tier 1: Strict Mode
Core modules with full type enforcement:
```toml
[[tool.mypy.overrides]]
module = [
    "aragora.core",
    "aragora.config.*",
    "aragora.protocols",
    "aragora.exceptions",
    # ... 30+ modules
]
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
```

### Tier 2: Check-Untyped Mode
Complex modules where we check bodies but don't require annotations:
```toml
[[tool.mypy.overrides]]
module = [
    "aragora.server.handlers.*",
    "aragora.debate.orchestrator",
    "aragora.agents.api_agents.*",
]
check_untyped_defs = true
```

### Tier 3: Ignore Errors
Dynamic/third-party integration modules:
```toml
[[tool.mypy.overrides]]
module = [
    "aragora.plugins.builtin.*",
    "aragora.verification.proofs",  # Z3 types
    "aragora.persistence.supabase_client",
]
ignore_errors = true
```

### Migration Process

1. **Phase-based promotion**: Move modules from Tier 3 → Tier 2 → Tier 1
2. **New code strict**: All new modules start in Tier 1
3. **Track progress**: Target metrics per phase (e.g., <40 ignored modules)
4. **CI enforcement**: mypy runs on all PRs

## Consequences

### Positive
- **Incremental progress**: No big-bang migration
- **Risk mitigation**: Changes are gradual and testable
- **IDE support**: Typed modules get full autocomplete
- **Documentation**: Types serve as inline documentation

### Negative
- **Inconsistency**: Mixed typing across codebase
- **Maintenance**: Must track which modules are in which tier
- **False confidence**: Tier 3 modules may have hidden issues

### Neutral
- py.typed marker included for downstream consumers
- Type stubs in typings/ directory for custom types

## Metrics

| Phase | Strict Modules | Ignored Modules | Target |
|-------|----------------|-----------------|--------|
| Phase 5 | 23 | 48 | <45 |
| Phase 6 | 30+ | 46 | <40 |
| Phase 7 | 40+ | <35 | <30 |

## Related
- `pyproject.toml` - mypy configuration
- `typings/` - Custom type stubs
- `aragora/py.typed` - PEP 561 marker
