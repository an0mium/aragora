# Handler Patterns Guide

This guide documents the patterns and best practices for implementing HTTP handlers in Aragora.

## Table of Contents

1. [Handler Architecture Overview](#handler-architecture-overview)
2. [Handler Splitting Pattern](#handler-splitting-pattern)
3. [Common Utilities](#common-utilities)
4. [Anti-Patterns](#anti-patterns)
5. [New Handler Checklist](#new-handler-checklist)

---

## Handler Architecture Overview

### BaseHandler Class

All HTTP handlers extend `BaseHandler` from `aragora.server.handlers.base`:

```python
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    success_response,
)

class MyHandler(BaseHandler):
    """Handler for my-feature endpoints."""

    ROUTES = ["/api/v1/my-feature"]
    ROUTE_PREFIXES = ["/api/v1/my-feature/"]

    def __init__(self, ctx: ServerContext):
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> HandlerResult | None:
        return None  # Let async methods handle
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `BaseHandler` | Base class with context, response helpers | `handlers/base.py` |
| `HandlerResult` | Typed response tuple (body, status, headers) | `handlers/base.py` |
| `ServerContext` | Request context with auth, params | `handlers/base.py` |
| `ROUTES` | Exact paths this handler handles | Handler class |
| `ROUTE_PREFIXES` | Dynamic path prefixes (e.g., `/api/v1/users/`) | Handler class |

### Handler Lifecycle

1. **Registration**: Handler class registered in `handler_registry.py`
2. **Initialization**: `__init__(ctx)` called with server context
3. **Routing**: `can_handle(path)` checked for each request
4. **Dispatch**: Method-specific handler called (`handle_get_*`, `handle_post_*`)
5. **Response**: `success_response()` or `error_response()` returned

### Response Helpers

```python
# Success response (200 OK)
return success_response({"data": result})

# Error response with status code
return error_response("Not found", 404)
return error_response("Invalid input", 400)
return error_response("Unauthorized", 401)
return error_response("Forbidden", 403)
return error_response("Server error", 500)
```

### Handler Registration

Handlers are registered in `aragora/server/handler_registry.py`:

```python
from .handlers.my_feature import MyFeatureHandler

# In HANDLER_REGISTRY list:
("_my_feature_handler", MyFeatureHandler),
```

The registry uses individual safe imports for graceful degradation:

```python
MyFeatureHandler = _safe_import("aragora.server.handlers", "MyFeatureHandler")
```

---

## Handler Splitting Pattern

### When to Split

Split a handler into a package when:
- File exceeds **800 lines**
- Handler has **4+ distinct concerns** (e.g., CRUD, OAuth, webhooks)
- Multiple team members frequently edit the same file
- Testing becomes unwieldy due to file size

### Reference Implementations

#### Auth Handler (`handlers/auth/`)

The canonical example of a split handler with 10 submodules:

```
aragora/server/handlers/auth/
├── __init__.py           # Re-exports AuthHandler + all public functions
├── handler.py            # AuthHandler class (~760 lines)
├── login.py              # handle_register, handle_login
├── password.py           # handle_change_password, handle_reset_password
├── mfa.py                # handle_mfa_setup, handle_mfa_verify
├── sessions.py           # handle_list_sessions, handle_revoke_session
├── api_keys.py           # handle_generate_api_key, handle_revoke_api_key
├── signup_handlers.py    # Self-service signup flows
├── sso_handlers.py       # SSO/OIDC authentication
├── store.py              # InMemoryUserStore for development
└── validation.py         # Email/password validation utilities
```

#### Email Handler (`handlers/email/`)

Recently split from a 1,766-line monolithic file into 9 submodules:

```
aragora/server/handlers/email/
├── __init__.py           # Re-exports EmailHandler + all public functions
├── handler.py            # EmailHandler class with routing
├── storage.py            # Store init, lazy connectors, RBAC helper
├── prioritization.py     # handle_prioritize_email, handle_rank_inbox
├── categorization.py     # handle_categorize_email, batch operations
├── oauth.py              # Gmail OAuth URL, callback, status
├── context.py            # Cross-channel context handlers
├── inbox.py              # handle_fetch_and_rank_inbox
├── config.py             # handle_get/update_config
└── vip.py                # handle_add/remove_vip
```

### Package Structure Template

```
aragora/server/handlers/{domain}/
├── __init__.py           # Re-exports (REQUIRED)
├── handler.py            # Handler class (REQUIRED)
├── storage.py            # Lazy stores, shared state (if needed)
├── {concern1}.py         # Group of related handlers
├── {concern2}.py         # Group of related handlers
└── models.py             # Domain models (if not in core/)
```

### `__init__.py` Pattern

```python
"""
{Domain} handlers subpackage.

This package contains {domain}-related handlers split by domain:
- handler: Main {Domain}Handler class for routing
- storage: Store initialization and shared utilities
- {concern1}: Description of handlers in this module
- {concern2}: Description of handlers in this module
"""

# Import the handler class
from .handler import {Domain}Handler

# Import storage utilities (if applicable)
from .storage import (
    get_{domain}_store,
    # ... other utilities
)

# Import handler functions from submodules
from .{concern1} import (
    handle_{action1},
    handle_{action2},
)
from .{concern2} import (
    handle_{action3},
    handle_{action4},
)

__all__ = [
    # Core handler
    "{Domain}Handler",
    # Storage utilities
    "get_{domain}_store",
    # Handler functions
    "handle_{action1}",
    "handle_{action2}",
    "handle_{action3}",
    "handle_{action4}",
]
```

### `storage.py` Pattern

```python
"""
Storage utilities for {domain} handlers.

Provides lazy-initialized stores and shared utilities.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global instances (lazy init, thread-safe)
_store: Any | None = None
_store_lock = threading.Lock()


def get_{domain}_store():
    """Get or create the {domain} store (thread-safe)."""
    global _store
    if _store is not None:
        return _store

    with _store_lock:
        if _store is None:
            try:
                from aragora.storage.{domain}_store import get_{domain}_store as _get
                _store = _get()
                logger.info("[{Domain}Handler] Initialized {domain} store")
            except Exception as e:
                logger.warning(f"[{Domain}Handler] Failed to init store: {e}")
        return _store


# RBAC helper (if needed)
def _check_{domain}_permission(
    auth_context: Any | None,
    permission_key: str
) -> Optional[dict[str, Any]]:
    """Check RBAC permission for {domain} operations."""
    # ... implementation
```

### `handler.py` Pattern

```python
"""
{Domain} HTTP handler class for server routing integration.
"""

from __future__ import annotations

from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    success_response,
)

# Import handler functions from submodules
from .{concern1} import handle_{action1}, handle_{action2}
from .{concern2} import handle_{action3}, handle_{action4}


class {Domain}Handler(BaseHandler):
    """HTTP handler for {domain} endpoints."""

    ROUTES = [
        "/api/v1/{domain}/{action1}",
        "/api/v1/{domain}/{action2}",
    ]

    ROUTE_PREFIXES = ["/api/v1/{domain}/"]

    def __init__(self, ctx: ServerContext):
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> HandlerResult | None:
        return None

    async def handle_post_{action1}(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/v1/{domain}/{action1}"""
        result = await handle_{action1}(
            param1=data.get("param1"),
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"

    def _get_auth_context(self) -> Any | None:
        """Get auth context from handler ctx."""
        return self.ctx.get("auth_context")
```

### Updating `handlers/__init__.py`

After creating a handler package, the import in `handlers/__init__.py` works automatically:

```python
# This import works for both:
# - Single file: handlers/email.py
# - Package: handlers/email/__init__.py
from .email import EmailHandler
```

Python's import system prefers packages over modules of the same name.

---

## Common Utilities

### RBAC Decorators

```python
from aragora.rbac.decorators import require_permission, require_role

class MyHandler(BaseHandler):
    @require_permission("resource:read")
    async def handle_get_resource(self, params: dict) -> HandlerResult:
        # Only called if user has resource:read permission
        ...

    @require_permission("resource:write")
    async def handle_post_resource(self, data: dict) -> HandlerResult:
        # Only called if user has resource:write permission
        ...
```

**Important**: Use `@require_permission` at route/handler level only, not on internal methods.

### Rate Limiting

```python
from aragora.server.middleware.rate_limit import rate_limit

@rate_limit(requests_per_minute=60)
async def handle_public_endpoint(...):
    ...

@rate_limit(requests_per_minute=5)  # Strict for auth
async def handle_oauth_callback(...):
    ...
```

### Request Validation

```python
from aragora.server.validation.query_params import safe_query_int, safe_query_float

async def handle_get_list(self, params: dict) -> HandlerResult:
    limit = safe_query_int(params, "limit", default=50, min_val=1, max_val=1000)
    offset = safe_query_int(params, "offset", default=0, min_val=0, max_val=100000)

    # Use limit and offset safely
    ...
```

### Metrics Tracking

```python
from aragora.observability.metrics import track_handler

@track_handler("email/prioritize")
async def handle_prioritize_email(...):
    ...

@track_handler("email/gmail/status", method="GET")
async def handle_gmail_status(...):
    ...
```

### Lazy Store Factory

For handlers that need database access with graceful degradation:

```python
from aragora.server.handlers.utils.lazy_stores import LazyStoreFactory

_my_store = LazyStoreFactory(
    store_name="my_store",
    import_path="aragora.storage.my_store",
    factory_name="get_my_store",
    logger_context="MyHandler",
)

def get_my_store():
    return _my_store.get()
```

---

## Anti-Patterns

### Monolithic Handler Files

**Bad**: Single file with 1000+ lines

```python
# handlers/everything.py (2000 lines)
class EverythingHandler:
    async def handle_user_create(...): ...
    async def handle_user_update(...): ...
    async def handle_order_create(...): ...
    async def handle_order_update(...): ...
    async def handle_payment_process(...): ...
    # ... 50 more methods
```

**Good**: Split by domain

```
handlers/users/
handlers/orders/
handlers/payments/
```

### Late Framework Imports

**Bad**: Importing aiohttp.web inside every method

```python
async def handle_request(self, ...):
    import aiohttp.web as web  # Imported 22 times in file!
    return web.Response(...)
```

**Good**: Import at module level

```python
import aiohttp.web as web

async def handle_request(self, ...):
    return web.Response(...)
```

### Monolithic Try/Except Import Blocks

**Bad**: All-or-nothing imports

```python
try:
    from .handlers import Handler1, Handler2, ... , Handler90
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False  # All handlers fail if ANY fails
```

**Good**: Individual safe imports

```python
Handler1 = _safe_import("aragora.server.handlers", "Handler1")
Handler2 = _safe_import("aragora.server.handlers", "Handler2")
# Each handler can fail independently
```

### @require_permission on Internal Methods

**Bad**: Decorator on internal helper

```python
class MyHandler:
    @require_permission("data:read")  # Won't work!
    def _internal_fetch_data(self):
        # Internal methods don't have auth context
        ...
```

**Good**: Decorator on public handler method

```python
class MyHandler:
    @require_permission("data:read")
    async def handle_get_data(self, params):
        data = self._internal_fetch_data()  # No decorator here
        return success_response(data)
```

### Duplicated Models

**Bad**: Same dataclass defined in multiple files

```python
# handlers/inbox.py
@dataclass
class Message:
    id: str
    content: str

# handlers/email.py
@dataclass
class Message:  # Duplicate!
    id: str
    content: str
```

**Good**: Single source of truth

```python
# handlers/shared_inbox/models.py
@dataclass
class Message:
    id: str
    content: str

# Import from models module
from .models import Message
```

---

## New Handler Checklist

Use this checklist when creating a new handler:

### 1. Planning

- [ ] Define the domain and endpoints
- [ ] List all required operations (CRUD, etc.)
- [ ] Estimate file size (>800 lines = consider package)
- [ ] Identify shared storage/utilities needed

### 2. File Structure

- [ ] Create handler file/package in `aragora/server/handlers/`
- [ ] For packages: create `__init__.py`, `handler.py`
- [ ] For packages: create submodules by concern
- [ ] Add models to appropriate location

### 3. Handler Class

- [ ] Extend `BaseHandler`
- [ ] Define `ROUTES` list
- [ ] Define `ROUTE_PREFIXES` list (if dynamic paths)
- [ ] Implement `can_handle(path)` method
- [ ] Implement `handle()` returning None (for async)

### 4. Handler Methods

- [ ] Name format: `handle_{method}_{action}` (e.g., `handle_post_create`)
- [ ] Add `@require_permission` decorator if needed
- [ ] Add `@rate_limit` decorator if public
- [ ] Add `@track_handler` for metrics
- [ ] Use `success_response()` / `error_response()`

### 5. Request Handling

- [ ] Validate required parameters
- [ ] Use `safe_query_int()` for numeric params
- [ ] Handle missing/invalid input gracefully
- [ ] Return appropriate HTTP status codes

### 6. Registration

- [ ] Add import to `handlers/__init__.py`
- [ ] Add to `HANDLER_REGISTRY` in `handler_registry.py`
- [ ] Add to `__all__` exports

### 7. Testing

- [ ] Create test file in `tests/server/handlers/`
- [ ] Test all happy paths
- [ ] Test validation errors
- [ ] Test permission denied scenarios
- [ ] Test rate limiting (if applicable)

### 8. Documentation

- [ ] Add docstrings to handler class
- [ ] Add docstrings to all public methods
- [ ] Document endpoint paths in docstring
- [ ] Update API documentation if needed

---

## Quick Reference

### Common Imports

```python
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    success_response,
)
from aragora.rbac.decorators import require_permission
from aragora.server.middleware.rate_limit import rate_limit
from aragora.observability.metrics import track_handler
from aragora.server.validation.query_params import safe_query_int
```

### HTTP Status Codes

| Code | Usage | Example |
|------|-------|---------|
| 200 | Success | `success_response(data)` |
| 201 | Created | `success_response(data, 201)` |
| 400 | Bad Request | `error_response("Invalid input", 400)` |
| 401 | Unauthorized | `error_response("Not authenticated", 401)` |
| 403 | Forbidden | `error_response("Permission denied", 403)` |
| 404 | Not Found | `error_response("Resource not found", 404)` |
| 429 | Rate Limited | (handled by middleware) |
| 500 | Server Error | `error_response("Internal error", 500)` |

### File Locations

| What | Where |
|------|-------|
| Handlers | `aragora/server/handlers/` |
| Base class | `aragora/server/handlers/base.py` |
| Registry | `aragora/server/handler_registry.py` |
| Validation | `aragora/server/validation/` |
| RBAC | `aragora/rbac/decorators.py` |
| Rate limiting | `aragora/server/middleware/rate_limit.py` |
| Tests | `tests/server/handlers/` |

---

## Production Examples

### Example 1: Simple CRUD Handler (Feedback)

A complete example of a simple handler with permission checking (`handlers/feedback.py`):

```python
"""
User Feedback Collection Handler.

Endpoints:
- POST /api/v1/feedback/nps - Submit NPS score (requires feedback.write)
- POST /api/v1/feedback/general - Submit general feedback (requires feedback.write)
- GET /api/v1/feedback/nps/summary - Get NPS summary (requires feedback.update - admin)
- GET /api/v1/feedback/prompts - Get active feedback prompts (requires feedback.read)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import (
    HandlerResult,
    ServerContext,
    error_response,
    get_clamped_int_param,
    json_response,
)

logger = logging.getLogger(__name__)


# Permission helper - checks RBAC before allowing action
def _check_permission(ctx: ServerContext, permission: str) -> HandlerResult | None:
    """Check if the current user has the required permission."""
    user_id = ctx.get("user_id")
    if not user_id:
        return error_response("Authentication required", status=401)

    auth_context = AuthorizationContext(
        user_id=user_id,
        org_id=ctx.get("org_id"),
        roles=set(ctx.get("roles", [])),
        permissions=set(ctx.get("permissions", [])),
    )

    checker = get_permission_checker()
    decision = checker.check_permission(auth_context, permission)

    if not decision.allowed:
        return error_response(f"Permission denied: {permission}", status=403)
    return None


# Models
class FeedbackType(str, Enum):
    NPS = "nps"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"


@dataclass
class FeedbackEntry:
    id: str
    user_id: str | None
    feedback_type: FeedbackType
    score: int | None
    comment: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# Handler functions
async def handle_submit_nps(ctx: ServerContext) -> HandlerResult:
    """
    Submit NPS feedback.

    POST /api/v1/feedback/nps
    Body: {"score": 0-10, "comment": "optional", "context": {...}}
    Requires: feedback.write permission
    """
    perm_error = _check_permission(ctx, "feedback.write")
    if perm_error:
        return perm_error

    try:
        body = ctx.get("body", {})
        user_id = ctx.get("user_id", "anonymous")
        score = body.get("score")

        if score is None or not isinstance(score, int) or not 0 <= score <= 10:
            return error_response("Score must be an integer between 0 and 10", status=400)

        entry = FeedbackEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            feedback_type=FeedbackType.NPS,
            score=score,
            comment=body.get("comment"),
            metadata=body.get("context", {}),
        )

        store = get_feedback_store()
        store.save(entry)

        logger.info(f"NPS feedback submitted: score={score}, user={user_id}")
        return json_response({"success": True, "feedback_id": entry.id})

    except Exception as e:
        logger.error(f"Error submitting NPS feedback: {e}")
        return error_response(str(e), status=500)


async def handle_get_nps_summary(ctx: ServerContext) -> HandlerResult:
    """
    Get NPS summary (admin only).

    GET /api/v1/feedback/nps/summary?days=30
    Requires: feedback.update permission (admin)
    """
    perm_error = _check_permission(ctx, "feedback.update")
    if perm_error:
        return perm_error

    try:
        query = ctx.get("query", {})
        days = get_clamped_int_param(query, "days", 30, min_val=1, max_val=365)
        store = get_feedback_store()
        summary = store.get_nps_summary(days)
        return json_response(summary)

    except Exception as e:
        logger.error(f"Error getting NPS summary: {e}")
        return error_response(str(e), status=500)


# Route definitions for registration
FEEDBACK_ROUTES = [
    ("POST", "/api/v1/feedback/nps", handle_submit_nps),
    ("POST", "/api/v1/feedback/general", handle_submit_feedback),
    ("GET", "/api/v1/feedback/nps/summary", handle_get_nps_summary),
    ("GET", "/api/v1/feedback/prompts", handle_get_feedback_prompts),
]
```

### Example 2: Handler with Async Database Operations (Checkpoints)

Handler pattern with async store operations (`handlers/checkpoints.py`):

```python
async def handle_list_checkpoints(ctx: ServerContext) -> HandlerResult:
    """
    List checkpoints for a debate.

    GET /api/v1/debates/:debate_id/checkpoints?limit=50&offset=0
    """
    try:
        debate_id = ctx.get("path_params", {}).get("debate_id")
        if not debate_id:
            return error_response("Missing debate_id", status=400)

        query = ctx.get("query", {})
        limit = get_clamped_int_param(query, "limit", 50, min_val=1, max_val=100)
        offset = get_clamped_int_param(query, "offset", 0, min_val=0)

        store = get_checkpoint_store()
        checkpoints = await store.list_checkpoints(debate_id, limit=limit, offset=offset)

        return json_response({
            "checkpoints": [cp.to_dict() for cp in checkpoints],
            "total": len(checkpoints),
            "limit": limit,
            "offset": offset,
        })

    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        return error_response(str(e), status=500)
```

### Example 3: Handler with Request Body Validation (Gauntlet)

Handler with comprehensive input validation (`handlers/gauntlet.py`):

```python
async def handle_start_gauntlet(ctx: ServerContext) -> HandlerResult:
    """
    Start a new gauntlet evaluation.

    POST /api/v1/gauntlet/start
    Body:
        task: str (required) - Task description
        agents: list[str] (optional) - Agent names to use
        protocol: dict (optional) - Debate protocol config
        options: dict (optional) - Gauntlet options
    """
    try:
        body = ctx.get("body", {})

        # Validate required fields
        task = body.get("task")
        if not task or not isinstance(task, str):
            return error_response("'task' is required and must be a string", status=400)

        if len(task) > 10000:
            return error_response("'task' must be under 10,000 characters", status=400)

        # Validate optional fields
        agents = body.get("agents", [])
        if not isinstance(agents, list):
            return error_response("'agents' must be a list", status=400)
        if len(agents) > 20:
            return error_response("Maximum 20 agents allowed", status=400)

        protocol = body.get("protocol", {})
        if not isinstance(protocol, dict):
            return error_response("'protocol' must be a dict", status=400)

        # Start the gauntlet
        runner = get_gauntlet_runner()
        result = await runner.start(task=task, agents=agents, protocol=protocol)

        return json_response({
            "success": True,
            "gauntlet_id": result.id,
            "status": result.status,
        }, status=201)

    except ValueError as e:
        return error_response(str(e), status=400)
    except Exception as e:
        logger.exception(f"Error starting gauntlet: {e}")
        return error_response("Internal error starting gauntlet", status=500)
```

### Example 4: WebSocket Handler (Streaming)

Handler for WebSocket streaming endpoints (`handlers/streaming/debate_stream.py`):

```python
async def handle_debate_stream(ws: WebSocketResponse, ctx: ServerContext) -> None:
    """
    WebSocket endpoint for real-time debate streaming.

    WS /api/v1/debates/:debate_id/stream

    Events sent:
        - debate_start: {debate_id, task, agents}
        - round_start: {round, phase}
        - agent_message: {agent, content, role}
        - consensus: {content, confidence}
        - debate_end: {result}
    """
    debate_id = ctx.get("path_params", {}).get("debate_id")

    try:
        # Subscribe to debate events
        async with debate_events.subscribe(debate_id) as events:
            # Send initial state
            await ws.send_json({"type": "connected", "debate_id": debate_id})

            # Stream events
            async for event in events:
                if ws.closed:
                    break
                await ws.send_json(event)

    except asyncio.CancelledError:
        logger.debug(f"WebSocket cancelled for debate {debate_id}")
    except Exception as e:
        logger.error(f"WebSocket error for debate {debate_id}: {e}")
        if not ws.closed:
            await ws.close(code=1011, message=str(e))
```

### Example 5: Handler Package Structure (Auth)

The auth handler is a canonical example of a large handler split into submodules:

```
handlers/auth/
├── __init__.py       # Re-exports AuthHandler + all public functions
├── handler.py        # AuthHandler class with routing
├── login.py          # handle_register, handle_login, handle_logout
├── password.py       # handle_change_password, handle_reset_password
├── mfa.py            # handle_mfa_setup, handle_mfa_verify, handle_mfa_disable
├── sessions.py       # handle_list_sessions, handle_revoke_session
├── api_keys.py       # handle_generate_api_key, handle_revoke_api_key
├── sso_handlers.py   # SSO/OIDC authentication flows
└── validation.py     # Email/password validation utilities
```

Each submodule exports standalone handler functions:

```python
# handlers/auth/login.py
async def handle_login(ctx: ServerContext) -> HandlerResult:
    """
    Authenticate user with email/password.

    POST /api/v1/auth/login
    Body: {"email": "...", "password": "..."}
    """
    body = ctx.get("body", {})
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        return error_response("Email and password required", status=400)

    user = await authenticate_user(email, password)
    if not user:
        return error_response("Invalid credentials", status=401)

    session = await create_session(user)
    return json_response({
        "success": True,
        "token": session.token,
        "user": user.to_public_dict(),
    })
```

The handler class routes to these functions:

```python
# handlers/auth/handler.py
class AuthHandler(BaseHandler):
    ROUTES = [
        "/api/v1/auth/login",
        "/api/v1/auth/logout",
        "/api/v1/auth/register",
    ]

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/v1/auth/")

    async def handle_post_login(self, data: dict) -> HandlerResult:
        return await handle_login(self.ctx)
```

---

*Last updated: 2026-02-01*
