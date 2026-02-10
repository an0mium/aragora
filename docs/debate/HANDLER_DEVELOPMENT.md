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

You can also use the decorator (supports sync and async handlers). When using
handlers that pass headers directly, provide `headers=...` so the decorator can
extract client IPs consistently:

```python
from ..utils.rate_limit import rate_limit

@rate_limit(requests_per_minute=60)
async def handle(self, path, query_params, headers=None):
    ...
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

---

## Secure Handlers

For handlers that deal with sensitive operations, authentication, or require RBAC enforcement, use `SecureHandler` instead of `BaseHandler`.

### When to Use SecureHandler

Use `SecureHandler` for:
- **User management** operations (create, update, delete users)
- **Admin operations** (impersonation, system config)
- **Billing/payment** operations
- **Data export/deletion** (GDPR compliance)
- **API key management**
- **Operations requiring audit trails**

### SecureHandler Features

SecureHandler provides automatic:
1. **JWT Authentication** - Token extraction and verification
2. **RBAC Enforcement** - Permission checking via the permission system
3. **Audit Logging** - Writes to the immutable audit trail
4. **Security Metrics** - Records auth attempts, RBAC decisions, blocked requests
5. **Encryption Support** - Built-in field encryption/decryption helpers

### Creating a Secure Handler

```python
from aragora.server.handlers.secure import (
    SecureHandler,
    secure_endpoint,
    audit_sensitive_access,
)

class MySecureHandler(SecureHandler):
    """Handler with security features."""

    # Resource type for audit logging
    RESOURCE_TYPE = "my_resource"

    ROUTES = ["/api/secure/myresource"]

    @secure_endpoint(permission="myresource.read")
    async def handle_get(self, request, auth_context, **kwargs):
        """auth_context is automatically injected and verified."""
        return json_response({"user_id": auth_context.user_id})

    @secure_endpoint(permission="myresource.write", audit=True)
    async def handle_post(self, request, auth_context, **kwargs):
        """This action is logged to the audit trail."""
        return json_response({"created": True})

    @secure_endpoint(permission="myresource.delete", audit=True, audit_action="delete")
    async def handle_delete(self, request, auth_context, resource_id, **kwargs):
        """Custom audit action name."""
        return json_response({"deleted": True})
```

### The @secure_endpoint Decorator

```python
@secure_endpoint(
    permission="resource.action",      # RBAC permission key (optional)
    require_auth=True,                  # Require authentication (default: True)
    audit=False,                        # Log to audit trail (default: False)
    audit_action=None,                  # Custom audit action name
    resource_id_param=None,             # Parameter containing resource ID
)
```

### Auditing Sensitive Data Access

For endpoints that access PII or secrets, use `@audit_sensitive_access`:

```python
@audit_sensitive_access("api_key", "read")
async def get_api_key(self, request, auth_context):
    """This logs to audit trail that API key was accessed."""
    return json_response({"key": "..."})
```

### Admin Handlers with MFA Enforcement

For admin handlers that require MFA (SOC 2 CC5-01 compliance), use the `admin_secure_endpoint` decorator:

```python
from aragora.server.handlers.admin.admin import AdminHandler, admin_secure_endpoint

class MyAdminHandler(AdminHandler):
    RESOURCE_TYPE = "admin_operation"

    @admin_secure_endpoint(permission="admin.users.manage", audit=True)
    async def manage_users(self, request, auth_context, **kwargs):
        """
        Automatically enforces:
        1. JWT authentication
        2. Admin or owner role
        3. MFA enabled
        4. RBAC permission check
        5. Audit logging
        """
        return json_response({"success": True})
```

### Testing Secure Handlers

When testing secure handlers, mock the RBAC permission check:

```python
from unittest.mock import patch, MagicMock

def _mock_allowed_decision():
    """Create a mock decision that allows access."""
    decision = MagicMock()
    decision.allowed = True
    decision.reason = "Test mock: allowed"
    return decision

@patch("aragora.server.handlers.secure.check_permission")
@patch("aragora.server.handlers.utils.auth.get_auth_context")
def test_secure_endpoint(mock_auth, mock_check_permission):
    mock_auth.return_value = MagicMock(user_id="user-123", roles={"admin"})
    mock_check_permission.return_value = _mock_allowed_decision()

    # Your test code here
```

### Checklist for Secure Handlers

- [ ] Handler inherits from `SecureHandler` (or `AdminHandler` for admin ops)
- [ ] `RESOURCE_TYPE` is set for audit logging
- [ ] `@secure_endpoint` decorator with appropriate permission
- [ ] `audit=True` for state-changing operations
- [ ] `@audit_sensitive_access` for PII/secret access
- [ ] Tests mock `check_permission` to allow/deny access
- [ ] Error handling uses `handle_security_error()`

---

## Approval Gate Middleware

For high-risk operations that require human approval, use the approval gate:

```python
from aragora.server.middleware import require_approval, OperationRiskLevel

@require_approval(
    risk_level=OperationRiskLevel.HIGH,
    operation_type="delete_all_data",
    description="Delete all user data permanently",
)
async def delete_all_data(request):
    """This endpoint requires human approval before execution."""
    # Only reaches here after approval is granted
    return await perform_deletion()
```

### Approval States

| State | HTTP Code | Meaning |
|-------|-----------|---------|
| PENDING | 202 | Waiting for approval |
| APPROVED | 200 | Proceeds to handler |
| DENIED | 403 | Operation blocked |
| EXPIRED | 410 | Approval timed out |

### Working with Approvals

```python
from aragora.server.middleware import (
    create_approval_request,
    get_approval_request,
    resolve_approval,
    ApprovalState,
)

# Create approval request
approval = await create_approval_request(
    operation_type="delete_user",
    risk_level=OperationRiskLevel.HIGH,
    requested_by="user-123",
    metadata={"target_user": "user-456"},
)

# Get pending approvals
pending = await get_pending_approvals(workspace_id="workspace-123")

# Approve/deny
await resolve_approval(
    approval_id=approval.id,
    state=ApprovalState.APPROVED,
    approved_by="admin-789",
    notes="Verified deletion request",
)
```
