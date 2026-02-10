# Handler Testing Patterns

Guide for writing tests for Aragora HTTP handlers to improve coverage.

## Quick Start

```python
"""Template for handler tests."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.your_handler import YourHandler


class TestYourHandler:
    """Tests for YourHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with mocked dependencies."""
        return YourHandler({"nomic_dir": "/tmp/test"})

    def test_can_handle_valid_path(self, handler):
        """Handler should match expected routes."""
        assert handler.can_handle("/api/v1/your-endpoint")
        assert not handler.can_handle("/api/v1/other")

    @pytest.mark.asyncio
    async def test_get_requires_auth(self, handler):
        """GET requests should require authentication."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"
        mock_handler.headers = {}  # No auth token

        result = await handler.handle("/api/v1/your-endpoint", {}, mock_handler)

        assert result[1] == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_get_requires_permission(self, handler):
        """GET requests should check permissions."""
        mock_handler = create_authed_handler(user_id="user-123")

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.side_effect = ForbiddenError("Missing permission")

            result = await handler.handle("/api/v1/your-endpoint", {}, mock_handler)

            assert result[1] == 403
            mock_check.assert_called_with(ANY, "your_resource:read")
```

## Test Categories

### 1. Route Matching Tests

Every handler should have tests verifying `can_handle()`:

```python
class TestRouteMatching:
    def test_matches_api_path(self, handler):
        assert handler.can_handle("/api/v1/analytics")

    def test_matches_with_resource_id(self, handler):
        assert handler.can_handle("/api/v1/analytics/123")

    def test_does_not_match_other_paths(self, handler):
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/analytics")  # Missing api prefix
```

### 2. RBAC Permission Tests

Test that each endpoint enforces the correct permission:

```python
class TestPermissions:
    @pytest.mark.asyncio
    async def test_list_requires_read_permission(self, handler):
        """List endpoint should require resource:read permission."""
        mock_handler = create_authed_handler()

        with patch.object(handler, "_check_permission") as mock_check:
            await handler.handle("/api/v1/resources", {}, mock_handler)
            mock_check.assert_called_once()
            # Extract permission from call args
            call_args = mock_check.call_args
            assert "resource:read" in str(call_args)

    @pytest.mark.asyncio
    async def test_create_requires_write_permission(self, handler):
        """Create endpoint should require resource:write permission."""
        mock_handler = create_authed_handler(method="POST")
        mock_handler.rfile = create_json_body({"name": "test"})

        with patch.object(handler, "_check_permission") as mock_check:
            await handler.handle("/api/v1/resources", {}, mock_handler)
            call_args = mock_check.call_args
            assert "resource:write" in str(call_args)

    @pytest.mark.asyncio
    async def test_rbac_denied_returns_403(self, handler):
        """Permission denial should return 403."""
        mock_handler = create_authed_handler()

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.side_effect = ForbiddenError("Insufficient permissions")

            result = await handler.handle("/api/v1/resources", {}, mock_handler)

            assert result[1] == 403
            assert "permission" in result[0].get("error", "").lower()
```

### 3. Input Validation Tests

Test that invalid input is rejected:

```python
class TestInputValidation:
    @pytest.mark.asyncio
    async def test_missing_required_field(self, handler):
        """Missing required fields should return 400."""
        mock_handler = create_authed_handler(method="POST")
        mock_handler.rfile = create_json_body({})  # Missing 'name'

        result = await handler.handle("/api/v1/resources", {}, mock_handler)

        assert result[1] == 400

    @pytest.mark.asyncio
    async def test_invalid_id_format(self, handler):
        """Invalid resource ID format should return 400."""
        mock_handler = create_authed_handler()

        result = await handler.handle("/api/v1/resources/not-a-uuid", {}, mock_handler)

        assert result[1] == 400
```

### 4. Resource Not Found Tests

```python
class TestResourceNotFound:
    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, handler):
        """GET for non-existent resource should return 404."""
        mock_handler = create_authed_handler()

        with patch.object(handler, "_get_resource", return_value=None):
            result = await handler.handle("/api/v1/resources/123", {}, mock_handler)

            assert result[1] == 404
```

## Helper Functions

Add these to `tests/conftest.py` or a dedicated test utilities module:

```python
from io import BytesIO
import json
from unittest.mock import MagicMock


def create_authed_handler(
    user_id: str = "test-user",
    method: str = "GET",
    roles: list[str] | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with authentication."""
    handler = MagicMock()
    handler.command = method
    handler.headers = {
        "Authorization": f"Bearer test-token-{user_id}",
        "Content-Type": "application/json",
    }
    handler.path = "/test"

    # Mock auth context extraction
    handler._auth_context = MagicMock()
    handler._auth_context.user_id = user_id
    handler._auth_context.roles = roles or ["member"]

    return handler


def create_json_body(data: dict) -> BytesIO:
    """Create a file-like object containing JSON data."""
    content = json.dumps(data).encode()
    body = BytesIO(content)
    body.seek(0)
    return body


def create_mock_store(data: dict | None = None):
    """Create a mock store with common methods."""
    store = MagicMock()
    store.get = MagicMock(return_value=data)
    store.list = MagicMock(return_value=[data] if data else [])
    store.save = MagicMock(return_value=True)
    store.delete = MagicMock(return_value=True)
    return store
```

## File Naming Convention

Handler tests should follow this naming pattern:

```
tests/server/handlers/test_{handler_name}_handler.py
```

Examples:
- `tests/server/handlers/test_analytics_handler.py`
- `tests/server/handlers/test_policy_handler.py`
- `tests/server/handlers/admin/test_admin_handler.py`

## Minimum Test Coverage Per Handler

Each handler should have tests covering:

| Category | Required Tests |
|----------|----------------|
| Route matching | 2-3 tests |
| Authentication | 1-2 tests |
| RBAC permissions | 1 per permission checked |
| Input validation | 2-3 tests |
| Happy path | 1 per HTTP method |
| Error cases | 2-3 tests (404, 500) |

**Minimum: 8-10 tests per handler**

## Priority Handlers (Currently Untested)

Based on code size and business criticality:

| Handler | LOC | Priority |
|---------|-----|----------|
| `accounting.py` | 1,200+ | HIGH - Financial data |
| `policy.py` | 738 | HIGH - Compliance |
| `audit_trail.py` | 484 | HIGH - Audit requirements |
| `auth.py` | 1,586 | HIGH - Security critical |
| `invoices.py` | 800+ | MEDIUM - Financial |
| `genesis.py` | 600+ | MEDIUM - Core feature |

## Running Handler Tests

```bash
# Run all handler tests
pytest tests/server/handlers/ -v

# Run tests for specific handler
pytest tests/server/handlers/test_accounting_handler.py -v

# Run with coverage
pytest tests/server/handlers/ --cov=aragora/server/handlers --cov-report=term-missing

# Run only RBAC-related tests
pytest tests/server/handlers/ -k "rbac or permission" -v
```

## Fixture Patterns

### Database Isolation

```python
@pytest.fixture
def temp_db(tmp_path):
    """Create isolated test database."""
    db_path = tmp_path / "test.db"
    yield str(db_path)
    # Cleanup handled by pytest tmp_path


@pytest.fixture
def handler_with_db(temp_db):
    """Handler with isolated database."""
    return YourHandler({"db_path": temp_db})
```

### Mock External Services

```python
@pytest.fixture
def mock_llm_service():
    """Mock LLM API calls."""
    with patch("aragora.agents.api_agents.anthropic.call_anthropic") as mock:
        mock.return_value = AsyncMock(return_value="mocked response")
        yield mock
```

## Common Pitfalls

1. **Not testing RBAC** - Every write endpoint needs permission tests
2. **Missing auth tests** - Test both authenticated and unauthenticated requests
3. **Forgetting 404 cases** - Test resource-not-found scenarios
4. **No input validation** - Test malformed requests return 400
5. **Skipping error paths** - Test exception handling returns proper error codes
