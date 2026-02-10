# RBAC Handler Implementation Guide

This guide documents how to add Role-Based Access Control (RBAC) to Aragora handlers.

## Overview

Aragora uses a flexible RBAC system with two main patterns:

1. **Class-based handlers** - Use `SecureHandler` base class
2. **Function-based handlers** - Use `@require_permission` decorator

## Permission Naming Convention

Permissions follow the pattern: `resource.action` (dot notation)

Common actions:
- `.read` - View/list operations (GET)
- `.update` - Create/update operations (POST, PUT, PATCH)
- `.delete` - Delete operations (DELETE)
- `.admin_op` - Administrative operations

Examples:
- `debates.read` - View debates
- `debates.create` - Create debates
- `verticals.read` - List verticals
- `verticals.update` - Create/modify verticals
- `gauntlet.read` - View gauntlet results
- `gauntlet.run` - Execute gauntlet tests

## Pattern 1: Class-Based Handlers (SecureHandler)

For handlers that extend `BaseHandler`, change to `SecureHandler`:

```python
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError

class MyHandler(SecureHandler):
    """Handler with RBAC protection."""

    RESOURCE_TYPE = "myresource"  # For audit logging

    async def handle(self, path: str, query_params: dict, handler: Any) -> HandlerResult:
        # RBAC check at start of handle method
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)

            # Check permission based on HTTP method
            method = getattr(handler, "command", "GET") if handler else "GET"
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                self.check_permission(auth_context, "myresource.update")
            else:
                self.check_permission(auth_context, "myresource.read")

        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Continue with handler logic...
```

### SecureHandler Methods

- `get_auth_context(request, require_auth=True)` - Extract auth context from request
- `check_permission(context, permission, resource_id=None)` - Verify permission

## Pattern 2: Function-Based Handlers (@require_permission)

For standalone async functions:

```python
from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext

@require_permission("repository.read")
async def handle_analyze_dependencies(
    context: AuthorizationContext,  # Injected by decorator
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    # Handler logic - context is automatically validated
    ...
```

### Key Points

1. The `context` parameter is injected by the decorator
2. Permission is checked before the function executes
3. Returns 403 Forbidden if permission denied

## Pattern 3: aiohttp Handlers

For handlers using aiohttp's `web.Request`:

```python
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext
from aiohttp import web

def _check_permission(self, request: web.Request, permission: str) -> None:
    """Check permission and raise HTTPForbidden if denied."""
    user_id = request.get("user_id") or request.headers.get("X-User-ID")
    org_id = request.get("org_id") or request.headers.get("X-Org-ID")
    roles_header = request.headers.get("X-User-Roles", "")
    roles = set(roles_header.split(",")) if roles_header else {"member"}

    if not user_id:
        user_id = "anonymous"

    context = AuthorizationContext(
        user_id=user_id,
        org_id=org_id,
        roles=roles,
    )

    checker = get_permission_checker()
    decision = checker.check_permission(context, permission)

    if not decision.allowed:
        raise web.HTTPForbidden(
            text=f"Permission denied: {decision.reason}",
            content_type="application/json",
        )

async def handle_get_inbox(self, request: web.Request) -> web.Response:
    self._check_permission(request, "bindings.read")
    # Handler logic...
```

## Handlers with RBAC

| Handler | Type | Permissions |
|---------|------|-------------|
| `debates.py` | SecureHandler | `debates.read`, `debates.create`, `debates.update` |
| `verticals.py` | SecureHandler | `verticals.read`, `verticals.update` |
| `gauntlet_v1.py` | SecureHandler | `gauntlet.read` |
| `moments.py` | SecureHandler | `memory.read` |
| `knowledge_base/handler.py` | SecureHandler | `evidence.read`, `evidence.create` |
| `dependency_analysis.py` | Function | `repository.read` |

## Testing RBAC

Include RBAC tests for your handlers:

```python
import pytest
from aragora.rbac.models import AuthorizationContext

@pytest.fixture
def admin_context():
    return AuthorizationContext(
        user_id="admin-user",
        org_id="test-org",
        roles={"admin"},
    )

@pytest.fixture
def member_context():
    return AuthorizationContext(
        user_id="member-user",
        org_id="test-org",
        roles={"member"},
    )

async def test_admin_can_access(handler, admin_context):
    """Admin role should have access."""
    result = await handler.handle_with_context(admin_context, ...)
    assert result.status_code == 200

async def test_unauthorized_rejected(handler):
    """Missing auth should return 401."""
    result = await handler.handle(...)
    assert result.status_code == 401
```

## Adding New Permissions

1. Add ResourceType to `aragora/rbac/models.py` (if new resource)
2. Define the permission in `aragora/rbac/defaults.py` using `_permission()`
3. Add to SYSTEM_PERMISSIONS dictionary
4. Add to appropriate roles (admin, member, etc.)
5. Use in handler with `check_permission()` or `@require_permission`
6. Document in this guide

## Related Documentation

- `aragora/rbac/models.py` - Core RBAC dataclasses
- `aragora/rbac/checker.py` - Permission checking logic
- `aragora/rbac/decorators.py` - Decorator implementations
- `aragora/rbac/defaults.py` - Default roles and permissions
