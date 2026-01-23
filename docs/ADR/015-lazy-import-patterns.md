# ADR-015: Lazy Import Patterns

**Status:** Accepted
**Date:** 2026-01-18
**Authors:** Development Team

## Context

Aragora is a large codebase with complex interdependencies between modules. As the system grew, we encountered circular import issues where module A imports module B, and module B imports module A, causing `ImportError` at startup.

Additionally, some modules have heavy initialization costs (database connections, model loading) that we want to defer until actually needed.

## Decision

We adopt **lazy import patterns** to:
1. Break circular dependencies
2. Reduce startup time
3. Make optional dependencies truly optional

### Pattern 1: Function-Level Import (Circular Dependencies)

Import inside the function that needs it:

```python
def some_function():
    # Import here to avoid circular dependency
    from aragora.other_module import SomeClass
    return SomeClass()
```

**Used in:**
- `aragora/privacy/isolation.py:283`
- `aragora/training/specialist_models.py:574`
- `aragora/agents/specialist_factory.py:187, 222`
- `aragora/agents/fallback.py:179`
- `aragora/server/middleware/cache.py:36`

### Pattern 2: Lazy Module Loader (Heavy Dependencies)

Create a wrapper that imports on first access:

```python
def _get_handlers_base():
    """Lazy import of handlers/base.py to avoid circular imports."""
    from aragora.server.handlers import base
    return base
```

**Used in:**
- `aragora/server/middleware/cache.py:30-36`
- `aragora/server/handlers/admin/cache.py:40`

### Pattern 3: TYPE_CHECKING Guard (Type Hints Only)

Use `typing.TYPE_CHECKING` for imports only needed for type hints:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.heavy_module import HeavyClass

def process(item: "HeavyClass") -> None:
    ...
```

**Used in:**
- Various handler and agent modules

### Pattern 4: Module `__getattr__` (Package-Level Lazy Loading)

Implement `__getattr__` at package level for lazy symbol loading:

```python
def __getattr__(name: str):
    """Lazily import public symbols to avoid heavy import side effects."""
    if name == "Arena":
        from aragora.debate.orchestrator import Arena
        return Arena
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Used in:**
- `aragora/__init__.py:353`
- `aragora/server/__init__.py:30`
- `aragora/server/stream/__init__.py:143`

## Locations Inventory

### Circular Dependency Avoidance

| File | Line | Reason |
|------|------|--------|
| `aragora/privacy/isolation.py` | 283 | Privacy -> Server -> Privacy |
| `aragora/training/specialist_models.py` | 574 | Training -> Agents -> Training |
| `aragora/agents/specialist_factory.py` | 187, 222 | Factory -> Verticals -> Factory |
| `aragora/agents/fallback.py` | 179 | Fallback -> OpenRouter -> Fallback |
| `aragora/agents/cli_agents.py` | 156 | CLI -> Debate -> CLI |
| `aragora/server/middleware/cache.py` | 30-36 | Middleware -> Handlers -> Middleware |
| `aragora/server/fork_handler.py` | 14 | Fork -> Debate -> Fork |
| `aragora/server/stream/server_base.py` | 251 | Stream -> Handlers -> Stream |
| `aragora/server/postman_generator.py` | 30 | Generator -> Handlers -> Generator |
| `aragora/server/debate_utils.py` | 217 | Utils -> Arena -> Utils |
| `aragora/server/handlers/admin/cache.py` | 40 | Cache -> Base -> Cache |
| `aragora/server/handlers/verification/formal_verification.py` | 186 | Verification -> Proofs -> Verification |

### Optional Dependencies

| File | Line | Dependency |
|------|------|------------|
| `aragora/connectors/base.py` | 283 | httpx (optional) |
| `aragora/config/secrets.py` | 149 | botocore (AWS) |
| `aragora/server/handlers/auditing.py` | 48 | Auditing tools |
| `aragora/server/handlers/evaluation.py` | 40 | LLM Judge |
| `aragora/server/handlers/moments.py` | 51 | Insights tools |
| `aragora/server/handlers/features/plugins.py` | 38 | Plugin system |
| `aragora/server/handlers/agents/calibration.py` | 39 | Calibration tools |

## Consequences

### Positive

1. **No circular import errors** - Application starts without `ImportError`
2. **Faster startup** - Heavy modules loaded only when needed
3. **Optional dependencies work** - Missing packages don't crash unrelated features
4. **Clear documentation** - Each lazy import has a comment explaining why

### Negative

1. **Hidden dependencies** - Import errors surface at runtime, not startup
2. **IDE limitations** - Some IDEs can't trace lazy imports for autocomplete
3. **Testing complexity** - Must test code paths that trigger lazy imports
4. **Maintenance burden** - Developers must understand the pattern

### Mitigations

1. **Explicit comments** - Every lazy import has a comment explaining the circular dependency
2. **Local wrappers** - e.g. `aragora/server/middleware/cache.py` and `aragora/server/handlers/admin/cache.py`
3. **Test coverage** - Integration tests exercise all lazy import paths
4. **Documentation** - This ADR documents all locations

## Alternatives Considered

### 1. Restructure Module Hierarchy

**Rejected because:**
- Would require massive refactoring
- Some circular dependencies are inherent to the domain model
- Risk of breaking existing integrations

### 2. Dependency Injection

**Partially adopted:**
- Used for some handler dependencies
- Not suitable for all cases (e.g., type hints)

### 3. Interface/Protocol Modules

**Adopted:**
- `aragora/server/handlers/interface.py` is in use
- Reduces but doesn't eliminate all circular imports

## Future Work

1. **Extract shared interfaces** - Consider a dedicated interfaces package for common protocols
2. **Reduce coupling** - Identify and break unnecessary dependencies
3. **Static analysis** - Add CI check to detect new circular imports
4. **Documentation generation** - Auto-generate dependency graph

## Related ADRs

- ADR-001: Phase-based debate execution
- ADR-005: Knowledge mound architecture

## References

- [Python Circular Imports](https://docs.python.org/3/faq/programming.html#what-are-the-best-practices-for-using-import-in-a-module)
- [PEP 562 - Module __getattr__](https://peps.python.org/pep-0562/)
- [typing.TYPE_CHECKING](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
