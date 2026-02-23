# Testing Patterns Guide

Internal reference for Aragora's test infrastructure, patterns, and conventions.

## RBAC Bypass in Handler Tests

Handler tests need to bypass authentication/authorization since they test handler
logic, not auth. Three separate RBAC systems must be bypassed, each patched by
autouse fixtures in `tests/handlers/conftest.py` and `tests/conftest.py`.

### System 1: `server.handlers.utils.decorators.require_permission`

This is the handler-level permission decorator used on HTTP endpoint methods.

**Bypass mechanism:**

```python
from aragora.server.handlers.utils import decorators as handler_decorators

# Set the test hook that short-circuits authentication
monkeypatch.setattr(handler_decorators, "_test_user_context_override", mock_user_ctx)

# Make has_permission always return True
monkeypatch.setattr(handler_decorators, "has_permission", lambda role, perm: True)
```

**Where patched:** `tests/handlers/conftest.py:mock_auth_for_handler_tests` and
`tests/conftest.py:_bypass_rbac_for_root_handler_tests`

### System 2: `aragora.rbac.checker.PermissionChecker.check_permission`

This is the RBAC v2 permission checker with caching.

**Bypass mechanism:**

```python
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationDecision

checker = get_permission_checker()
monkeypatch.setattr(checker, "check_permission", lambda ctx, perm, res=None:
    AuthorizationDecision(allowed=True, reason="Test bypass", permission_key=perm))
```

**Where patched:** `tests/handlers/conftest.py:_bypass_rbac_checker_and_enforcer` and
`tests/server/handlers/conftest.py:_patch_handler_rbac`

### System 3: `aragora.rbac.enforcer.RBACEnforcer.check/require`

This is the RBAC enforcer used for programmatic permission checks.

**Bypass mechanism:**

```python
from aragora.rbac.enforcer import RBACEnforcer

async def _enforcer_check(self, *args, **kwargs):
    return True

async def _enforcer_require(self, *args, **kwargs):
    pass  # No-op, never raises

monkeypatch.setattr(RBACEnforcer, "check", _enforcer_check)
monkeypatch.setattr(RBACEnforcer, "require", _enforcer_require)
```

**Where patched:** `tests/server/handlers/conftest.py:_patch_handler_rbac`

### Opting Out of Auto-Auth

To test actual authentication/authorization behavior, use the `no_auto_auth` marker:

```python
@pytest.mark.no_auto_auth
def test_unauthenticated_returns_401(handler, mock_http_handler):
    result = handler.handle("/api/v1/resource", {}, mock_http_handler)
    assert result.status_code == 401
```

All three bypass fixtures check for this marker and skip their patching when present.

---

## Handler Test Conventions

### `@handle_errors` Decorator

All handler write methods (POST/PUT/PATCH/DELETE) must have `@handle_errors` as
their outermost decorator. This catches unhandled exceptions and returns a
sanitized error response instead of leaking internal details.

```python
class MyHandler(BaseHandler):
    @handle_errors          # outermost
    @require_permission("resource:write")
    async def handle_post(self, path, params, handler):
        ...
```

As of Feb 18, 2026, all 193 handler write methods across 130 files have this
decorator. Protocol stubs (`interface.py`, `types.py`) are correctly excluded.

### `{"data": ...}` Response Envelope

API endpoints that serve frontend hooks return responses wrapped in a `data`
envelope:

```python
# Handler returns:
return HandlerResult(200, body=json.dumps({"data": {"items": [...], "total": 42}}))

# Frontend hook unwraps:
const result = useSWRFetch<{ data: ItemsResponse }>("/api/v1/items")
const items = result.data?.data?.items  // double .data unwrap
```

### Test Response Helpers

`tests/server/handlers/conftest.py` provides:

```python
from tests.server.handlers.conftest import parse_handler_response

# Parse JSON from HandlerResult
result = handler.handle("/api/v1/test", {}, mock_http, "GET")
body = parse_handler_response(result)
assert body["data"]["total"] == 42

# Assert success with expected keys
assert_success_response(result, expected_keys=["data"])

# Assert error response
assert_error_response(result, expected_status=404, error_substring="not found")
```

Many handler test files also define local helpers:

```python
def _parse_data(response):
    """Parse response and auto-unwrap {"data": ...} envelope."""
    body = json.loads(response.body)
    return body.get("data", body)

def _body():
    """Create mock HTTP handler with JSON body."""
    ...
```

### Mock HTTP Handler Factory

```python
def test_endpoint(handler, mock_http_handler):
    http = mock_http_handler(
        method="POST",
        body={"name": "test", "value": 42},
        headers={"Authorization": "Bearer token123"},
    )
    result = handler.handle("/api/v1/resource", {}, http, "POST")
```

Convenience fixtures: `mock_http_get`, `mock_http_post`

---

## Conftest Fixture Catalog

### Root Conftest (`tests/conftest.py`)

#### Session-Scoped Fixtures (run once)

| Fixture | Purpose |
|---------|---------|
| `_preinstall_fake_sentence_transformers` | Installs lightweight fake `sentence_transformers` to prevent 30s import of real HuggingFace library |
| `_suppress_auth_cleanup_threads` | Prevents `AuthConfig` from spawning background cleanup daemon threads that cause shutdown hangs |

#### Autouse Per-Test Fixtures

| Fixture | Purpose |
|---------|---------|
| `_bypass_rbac_for_root_handler_tests` | RBAC bypass for root-level `test_handlers_*.py` files |
| `fast_convergence_backend` | Sets `ARAGORA_CONVERGENCE_BACKEND=jaccard` for fast tests (skipped for `@pytest.mark.slow`) |
| `reset_circuit_breakers` | Resets all circuit breakers before/after each test |
| `reset_continuum_memory_singleton` | Resets `ContinuumMemory` global singleton between tests |
| `mock_sentence_transformers` | Mocks `SentenceTransformer` to prevent HuggingFace model downloads (skipped for `@pytest.mark.slow`) |
| `mock_semantic_store_embeddings` | Forces `SemanticStore` to use hash-based embedding provider (skipped for `@pytest.mark.network/integration/slow`) |
| `_disable_rate_limiting` | Sets `RATE_LIMITING_DISABLED=True` to prevent xdist cross-test 429 errors (opt out with `@pytest.mark.rate_limit_test`) |
| `mock_external_apis` | Mocks OpenAI/Anthropic/httpx clients to prevent real API calls (skipped for `@pytest.mark.network/integration`) |
| `_global_mock_pollution_guard` | Repairs `NonCallableMock.side_effect` descriptor, `BaseHandler.extract_path_param`, and `run_async` if destroyed by tests |

#### On-Demand Fixtures

| Fixture | Purpose |
|---------|---------|
| `stop_auth_cleanup` | Stops auth cleanup threads for tests that create `AuthConfig` instances |

#### Skip Markers

The root conftest defines ~40 skip marker flags. Pattern:

```python
HAS_<DEP> = _check_import("module_name")  # or _check_aragora_module(...)
REQUIRES_<DEP> = "description of missing dep"
requires_<dep> = not HAS_<DEP>

# Usage:
@pytest.mark.skipif(requires_redis, reason=REQUIRES_REDIS)
def test_redis_feature():
    ...
```

Key markers: `requires_z3`, `requires_redis`, `requires_asyncpg`, `requires_httpx`,
`requires_websockets`, `requires_pyjwt`, `requires_sentence_transformers`,
`requires_mcp`, `requires_symlinks`, `requires_signals`, `requires_postgres`

### Handler Conftest (`tests/handlers/conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `_restore_mock_side_effect_descriptor` | Repairs `NonCallableMock.side_effect` property descriptor if destroyed by a test setting `side_effect` on a class |
| `mock_auth_for_handler_tests` | Comprehensive RBAC bypass -- patches `_get_context_from_args`, `get_auth_context`, `extract_user_from_request`, all three RBAC systems |
| `mock_coordinator` | Returns a `MockCoordinator` with agent registry, task queue, and health check simulation |
| `mock_request` | Factory for `MockRequest` objects |
| `mock_http_handler` | Factory for mock HTTP handlers with configurable method/body/headers |
| `mock_http_get` / `mock_http_post` | Convenience fixtures for GET/POST requests |
| `mock_debate_storage` | Mock `DebateStorage` with pre-configured list/get/search methods |

### Server Handler Conftest (`tests/server/handlers/conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `mock_auth_for_handler_tests` | Same RBAC bypass as `tests/handlers/conftest.py` but also patches `SecureHandler.get_auth_context`, `BaseHandler.require_auth_or_error`, `BaseHandler.get_current_user`, and `KnowledgeHandler._check_permission` |
| `_bypass_rbac_checker_and_enforcer` | Separate bypass for `PermissionChecker` and `RBACEnforcer` (belt-and-suspenders with the above) |
| `clear_handler_cache` | Clears handler TTL cache and `BaseHandler.elo_system` between tests |
| `mock_http_handler` | Factory for mock HTTP handlers |
| `mock_http_get` / `mock_http_post` | Convenience fixtures |

---

## Custom Pytest Markers

| Marker | Purpose | Effect |
|--------|---------|--------|
| `@pytest.mark.smoke` | Quick sanity tests | Included in PR CI (`-m "not slow and not integration"`) |
| `@pytest.mark.integration` | Requires external services | Excluded from PR CI, runs nightly |
| `@pytest.mark.slow` | Long-running (>30s) | Excluded from PR CI, runs nightly |
| `@pytest.mark.unit` | Isolated unit tests | No external dependencies |
| `@pytest.mark.network` | Requires network access | Mock APIs skipped for these tests |
| `@pytest.mark.no_auto_auth` | Disable RBAC bypass | For testing auth behavior specifically |
| `@pytest.mark.rate_limit_test` | Keep rate limiting active | For testing rate limit behavior |

---

## Mock Pollution Prevention

### The `side_effect` Descriptor Problem

Some tests incorrectly set `MagicMock.side_effect` on a CLASS instead of an
instance (e.g., `spec.adapter_class = MagicMock; spec.adapter_class.side_effect = ...`).
This replaces the `NonCallableMock.side_effect` property descriptor on the class
dict, breaking list-to-iterator conversion for ALL future mock instances.

**Prevention:** The `_restore_mock_side_effect_descriptor` fixture (in both
`tests/handlers/conftest.py` and `tests/conftest.py`) captures the original
descriptor at import time and restores it before and after each test.

### The `run_async` Replacement Problem

Some tests monkeypatch `run_async` at the module level but fail to restore it.
The `_global_mock_pollution_guard` fixture in `tests/conftest.py` scans all loaded
`aragora.server.*` and `aragora.utils.*` modules and restores the real `run_async`
function if it has been replaced.

---

## Test Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARAGORA_AUTH_CLEANUP_INTERVAL` | `1` (set by conftest) | Fast auth cleanup in tests (prod default: 300s) |
| `ARAGORA_CONVERGENCE_BACKEND` | `jaccard` (set by fixture) | Fast convergence detection (prod uses SentenceTransformer) |
| `ARAGORA_FORCE_MOCK_APIS` | unset | When `1/true/yes`, mocks APIs even for `@network`/`@integration` tests |
| `ARAGORA_ENV` | unset | When `production`, conftest overrides to `development` to prevent auth import failures |

---

## CI Strategy

| Pipeline | Scope | Duration | Trigger |
|----------|-------|----------|---------|
| PR CI | `pytest -m "not slow and not integration"` | ~5 min | Pull request |
| Nightly | `pytest` (full suite) | 2+ hours | Scheduled |
| Integration | With Redis + PostgreSQL services | Varies | Nightly or manual |
