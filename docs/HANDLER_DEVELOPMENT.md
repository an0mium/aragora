# Handler Development Guide

This guide covers how to create, test, and maintain HTTP handlers in the Aragora server.

## Handler Architecture

Handlers are organized into subdirectories by domain:

```
aragora/server/handlers/
├── admin/          # Admin, billing, dashboard, health, system
├── agents/         # Agent management, calibration, probes
├── auth/           # Authentication, OAuth, SSO
├── debates/        # Debate endpoints, graph, matrix
├── evolution/      # Agent evolution, A/B testing
├── features/       # Audio, broadcast, documents, evidence
├── memory/         # Memory, analytics, learning, insights
├── social/         # Collaboration, notifications, sharing
├── verification/   # Formal verification, proofs
└── utils/          # Shared utilities (rate limiting, etc.)
```

## Creating a New Handler

### 1. Choose the Right Location

Place your handler in the appropriate subdirectory based on its domain. If no subdirectory fits, consider creating a new one or placing it at root level.

### 2. Handler Structure

```python
"""
My Feature endpoint handlers.

Endpoints:
- GET /api/myfeature - List items
- GET /api/myfeature/{id} - Get item details
- POST /api/myfeature - Create item
"""

from __future__ import annotations

import logging
from typing import Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for this handler (adjust as needed)
_myfeature_limiter = RateLimiter(requests_per_minute=60)


class MyFeatureHandler(BaseHandler):
    """Handler for my feature endpoints."""

    ROUTES = [
        "/api/myfeature",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path == "/api/myfeature" or path.startswith("/api/myfeature/")

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _myfeature_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        if path == "/api/myfeature":
            return self._list_items(query_params)

        if path.startswith("/api/myfeature/"):
            item_id = path.split("/")[-1]
            return self._get_item(item_id)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests."""
        if path == "/api/myfeature":
            return self._create_item(handler)
        return None

    @handle_errors("list items")
    def _list_items(self, query_params: dict) -> HandlerResult:
        """List all items."""
        # Implementation here
        return json_response({"items": []})

    @handle_errors("get item")
    def _get_item(self, item_id: str) -> HandlerResult:
        """Get item by ID."""
        # Validate item_id
        if not item_id or len(item_id) > 100:
            return error_response("Invalid item ID", 400)

        # Fetch item
        item = None  # Replace with actual fetch
        if not item:
            return error_response(f"Item not found: {item_id}", 404)

        return json_response({"item": item})

    @handle_errors("create item")
    def _create_item(self, handler) -> HandlerResult:
        """Create a new item."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Validate and create
        return json_response({"created": True}, status=201)
```

### 3. Register the Handler

Add to `handlers/__init__.py`:

```python
from .myfeature import MyFeatureHandler

# Add to ALL_HANDLERS list (order matters - more specific first)
ALL_HANDLERS = [
    # ... existing handlers ...
    MyFeatureHandler,
]

# Add stability classification
HANDLER_STABILITY = {
    # ... existing entries ...
    "MyFeatureHandler": Stability.EXPERIMENTAL,  # or STABLE, PREVIEW
}

# Add to __all__
__all__ = [
    # ... existing exports ...
    "MyFeatureHandler",
]
```

If in a subdirectory, also update the subdirectory's `__init__.py`:

```python
# mysubdir/__init__.py
from .myfeature import MyFeatureHandler

__all__ = ["MyFeatureHandler"]
```

## Key Patterns

### Rate Limiting

Always rate limit public endpoints:

```python
from ..utils.rate_limit import RateLimiter, get_client_ip

_my_limiter = RateLimiter(requests_per_minute=60)

def handle(self, path, query_params, handler):
    if not _my_limiter.is_allowed(get_client_ip(handler)):
        return error_response("Rate limit exceeded", 429)
```

### Error Handling

Use the `@handle_errors` decorator:

```python
from ..base import handle_errors

@handle_errors("operation name")
def _my_method(self, ...) -> HandlerResult:
    # Exceptions are caught, logged, and returned as error responses
    ...
```

### Authentication

For protected endpoints:

```python
from aragora.billing.jwt_auth import extract_user_from_request

def _protected_endpoint(self, handler) -> HandlerResult:
    user_store = self._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)
    # Proceed with authenticated user
    user_id = auth_ctx.user_id
```

### Reading Request Bodies

```python
# For simple JSON
body = self.read_json_body(handler)
if body is None:
    return error_response("Invalid JSON", 400)

# For validated JSON
from ..base import safe_json_parse
body = safe_json_parse(handler.request_body)
```

### Pagination

```python
from ..base import get_int_param

def _list_items(self, query_params: dict) -> HandlerResult:
    limit = get_int_param(query_params, "limit", 50)
    offset = get_int_param(query_params, "offset", 0)
    # Clamp to reasonable values
    limit = max(1, min(limit, 100))

    items = self._fetch_items(limit=limit, offset=offset)
    return json_response({
        "items": items,
        "limit": limit,
        "offset": offset,
        "total": self._count_items()
    })
```

### Async Handlers

For I/O-intensive operations:

```python
@handle_errors("async operation")
async def handle(self, path: str, query_params: dict, handler, body=None) -> HandlerResult:
    result = await self._async_fetch()
    return json_response(result)
```

## Testing Handlers

### Test File Structure

Create `tests/server/handlers/test_myfeature.py`:

```python
"""Tests for MyFeature handler."""

import json
import pytest
from unittest.mock import MagicMock

from aragora.server.handlers.myfeature import MyFeatureHandler


@pytest.fixture
def myfeature_handler():
    ctx = {"storage": None}
    return MyFeatureHandler(ctx)


@pytest.fixture
def mock_http_handler():
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json"}
    return mock


class TestMyFeatureRouting:
    def test_can_handle_myfeature(self, myfeature_handler):
        assert myfeature_handler.can_handle("/api/myfeature") is True

    def test_cannot_handle_unknown(self, myfeature_handler):
        assert myfeature_handler.can_handle("/api/unknown") is False


class TestListItems:
    def test_list_returns_items(self, myfeature_handler, mock_http_handler):
        result = myfeature_handler.handle("/api/myfeature", {}, mock_http_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "items" in body
```

### Running Tests

```bash
# Run specific handler tests
pytest tests/server/handlers/test_myfeature.py -v

# Run all handler tests
pytest tests/server/handlers/ -v

# Run with coverage
pytest tests/server/handlers/test_myfeature.py --cov=aragora.server.handlers.myfeature
```

## Handler Stability Levels

- **STABLE**: Production-ready, extensively tested, API won't change
- **EXPERIMENTAL**: Works but may change, use with awareness
- **PREVIEW**: Early access, expect changes and potential issues
- **DEPRECATED**: Being phased out, use alternative

Update `HANDLER_STABILITY` in `__init__.py` as your handler matures.

## Checklist for New Handlers

- [ ] Handler inherits from `BaseHandler`
- [ ] `can_handle()` method properly checks paths
- [ ] Rate limiting applied to public endpoints
- [ ] `@handle_errors` decorator on methods
- [ ] Input validation for all parameters
- [ ] Proper HTTP status codes (200, 201, 400, 401, 404, 429, 500)
- [ ] Handler registered in `__init__.py`
- [ ] Stability level set in `HANDLER_STABILITY`
- [ ] Tests created with 80%+ coverage
- [ ] Docstring with endpoint documentation

## Common Issues

### Import Errors After Moving

If you move a handler to a subdirectory, update relative imports:

```python
# Before (at root)
from .base import BaseHandler

# After (in subdirectory)
from ..base import BaseHandler
```

### Handler Not Being Called

1. Check `can_handle()` returns `True` for your path
2. Verify handler is in `ALL_HANDLERS` list
3. Check handler order - more specific paths should come first

### Tests Failing After Refactor

If decorator signature changes, ensure test mocks match:

```python
# If @handle_errors requires argument
@handle_errors("operation")  # Correct
@handle_errors  # Wrong - missing argument
```
