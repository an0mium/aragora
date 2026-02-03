# Handler Architecture

This document describes the modular HTTP endpoint handler architecture used in the Aragora server.

## Overview

The handler system provides a modular, testable approach to HTTP endpoint implementation. Each handler is responsible for a specific domain of endpoints (debates, agents, consensus, etc.) and implements a consistent interface for routing and response generation.

## Core Components

### HandlerResult

A dataclass representing the result of handling an HTTP request:

```python
@dataclass
class HandlerResult:
    status_code: int      # HTTP status code (200, 400, 404, etc.)
    content_type: str     # MIME type ("application/json")
    body: bytes           # Response body
    headers: dict         # Additional headers (optional)
```

### BaseHandler

The abstract base class all handlers inherit from:

```python
class BaseHandler:
    def __init__(self, server_context: dict):
        """Initialize with shared server resources."""
        self.ctx = server_context

    def can_handle(self, path: str) -> bool:
        """Return True if this handler can process the path."""
        ...

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Process the request and return a result."""
        ...
```

**Context accessors:**
- `get_storage()` - Debate storage instance
- `get_elo_system()` - ELO ranking system
- `get_debate_embeddings()` - Vector embeddings database
- `get_critique_store()` - Critique storage
- `get_nomic_dir()` - Path to nomic state directory

## Response Helpers

```python
# Create a JSON response
json_response(data: Any, status: int = 200) -> HandlerResult

# Create an error response
error_response(message: str, status: int = 400) -> HandlerResult

# Parameter extraction
get_int_param(params: dict, key: str, default: int = 0) -> int
get_float_param(params: dict, key: str, default: float = 0.0) -> float
get_bool_param(params: dict, key: str, default: bool = False) -> bool
```

## Caching

### BoundedTTLCache

Thread-safe LRU cache with TTL expiry and memory bounds:

```python
cache = BoundedTTLCache(
    max_entries=1000,      # Maximum cached entries
    evict_percent=0.1      # Evict 10% when full
)
```

**Environment variables:**
- `ARAGORA_CACHE_MAX_ENTRIES` - Maximum entries (default: 1000)
- `ARAGORA_CACHE_EVICT_PERCENT` - Eviction percentage (default: 0.1)

### @ttl_cache Decorator

Cache method results with automatic TTL expiry:

```python
@ttl_cache(ttl_seconds=300, key_prefix="leaderboard", skip_first=True)
def _get_leaderboard(self, limit: int, domain: Optional[str]) -> HandlerResult:
    # Expensive database query...
```

**Parameters:**
- `ttl_seconds` - Cache lifetime in seconds
- `key_prefix` - Namespace for cache keys
- `skip_first` - Skip `self` in cache key (True for methods, False for functions)

**Cache management:**
```python
clear_cache()                    # Clear all entries
clear_cache("leaderboard")       # Clear by prefix
get_cache_stats()                # Get hit/miss statistics
```

## Handler Registry

Handlers are registered in `unified_server.py`:

```python
# Class variables
_debates_handler: Optional["DebatesHandler"] = None
_agents_handler: Optional["AgentsHandler"] = None
_replays_handler: Optional["ReplaysHandler"] = None
# ... etc

# Initialization
@classmethod
def init_handlers(cls, ctx: dict):
    cls._debates_handler = DebatesHandler(ctx)
    cls._agents_handler = AgentsHandler(ctx)
    cls._replays_handler = ReplaysHandler(ctx)
    # ...

# Routing (in _try_modular_handler)
handlers = [
    self._system_handler,
    self._debates_handler,
    self._agents_handler,
    # ...
]
for handler in handlers:
    if handler and handler.can_handle(path):
        result = handler.handle(path, query_dict, self)
        if result:
            # Send response
            return True
```

## Handler Catalog

**Total:** 501 handler modules across 30+ categories

### Handler Categories by Module Count

| Category | Modules | Description |
|----------|---------|-------------|
| `features/` | 69 | Feature-specific handlers (audio, broadcast, scheduler, etc.) |
| `social/` | 38 | Social platform integrations (Slack, Teams, Discord, etc.) |
| `admin/` | 30 | Admin endpoints (billing, cache, metrics, security) |
| `bots/` | 28 | Bot platform handlers (Telegram, Discord, Teams bots) |
| `knowledge_base/` | 27 | Knowledge Mound and document management |
| `utils/` | 18 | Shared utilities (decorators, routing, validation) |
| `debates/` | 15 | Debate CRUD, rounds, participants, export |
| `compliance/` | 11 | GDPR, HIPAA, SOC2 compliance handlers |
| `codebase/` | 10 | Codebase analysis, audit, SAST scanning |
| `auth/` | 10 | Authentication, SSO, MFA, API keys |
| `email/` | 9 | Email integration and automation |
| `_oauth/` | 8 | OAuth provider implementations |
| `workflows/` | 7 | Workflow engine endpoints |
| `openclaw/` | 7 | OpenClaw Gateway (credentials, policies, sessions) |
| `workspace/` | 6 | Workspace management |
| `gauntlet/` | 6 | Gauntlet security scanning |
| `analytics_dashboard/` | 6 | Analytics dashboard endpoints |

### Core Handlers

| Handler | Domain | Key Endpoints |
|---------|--------|---------------|
| `SystemHandler` | Health, modes | `/api/health`, `/api/modes` |
| `DebatesHandler` | Debate CRUD | `/api/debates`, `/api/debates/:id` |
| `AgentsHandler` | Agent profiles | `/api/leaderboard`, `/api/agent/:name` |
| `ConsensusHandler` | Consensus tracking | `/api/consensus/stats`, `/api/consensus/similar` |
| `BeliefHandler` | Belief networks | `/api/belief/cruxes`, `/api/belief/evolution` |
| `CritiqueHandler` | Critique analysis | `/api/critiques/patterns`, `/api/reputation` |
| `AnalyticsHandler` | Metrics | `/api/analytics/*` |
| `MetricsHandler` | Server metrics | `/api/metrics/*` |
| `PulseHandler` | Trending topics | `/api/pulse/*` |
| `GenesisHandler` | Genetic evolution | `/api/genesis/*` |
| `ReplaysHandler` | Debate replays | `/api/replays`, `/api/learning/evolution` |

### Enterprise Handlers

| Handler | Domain | Key Endpoints |
|---------|--------|---------------|
| `OpenClawGatewayHandler` | Agent gateway | `/api/openclaw/*` |
| `ComplianceHandler` | Compliance reporting | `/api/compliance/*` |
| `AuditHandler` | Audit trails | `/api/audit/*` |
| `BackupHandler` | Disaster recovery | `/api/backups/*` |
| `RBACHandler` | Permissions | `/api/rbac/*` |
| `BillingHandler` | Usage/billing | `/api/billing/*` |
| `TenantHandler` | Multi-tenancy | `/api/tenants/*` |

### Bot Platform Handlers

| Handler | Platform | Key Endpoints |
|---------|----------|---------------|
| `SlackBotHandler` | Slack | `/api/bots/slack/*` |
| `TeamsBotHandler` | MS Teams | `/api/bots/teams/*` |
| `TelegramBotHandler` | Telegram | `/api/bots/telegram/*` |
| `DiscordBotHandler` | Discord | `/api/bots/discord/*` |
| `GoogleChatBotHandler` | Google Chat | `/api/bots/googlechat/*` |

## Creating a New Handler

1. **Create the handler file** (`aragora/server/handlers/myfeature.py`):

```python
"""
MyFeature endpoint handlers.

Endpoints:
- GET /api/myfeature - List items
- GET /api/myfeature/:id - Get specific item
"""

import logging
from typing import Optional
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
)

logger = logging.getLogger(__name__)


class MyFeatureHandler(BaseHandler):
    """Handler for myfeature endpoints."""

    ROUTES = ["/api/myfeature"]

    def can_handle(self, path: str) -> bool:
        if path in self.ROUTES:
            return True
        if path.startswith("/api/myfeature/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        if path == "/api/myfeature":
            limit = get_int_param(query_params, 'limit', 20)
            return self._list_items(limit)

        if path.startswith("/api/myfeature/"):
            item_id = path.split('/')[-1]
            return self._get_item(item_id)

        return None

    @ttl_cache(ttl_seconds=120, key_prefix="myfeature_list", skip_first=True)
    def _list_items(self, limit: int) -> HandlerResult:
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not configured", 503)

        items = storage.get_items(limit=limit)
        return json_response(items)

    @ttl_cache(ttl_seconds=300, key_prefix="myfeature_item", skip_first=True)
    def _get_item(self, item_id: str) -> HandlerResult:
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not configured", 503)

        item = storage.get_item(item_id)
        if not item:
            return error_response(f"Item not found: {item_id}", 404)

        return json_response(item)
```

2. **Export from `__init__.py`**:

```python
from .myfeature import MyFeatureHandler

__all__ = [
    # ...existing handlers...
    "MyFeatureHandler",
]
```

3. **Register in `unified_server.py`**:

```python
from aragora.server.handlers import MyFeatureHandler

class UnifiedServer:
    _myfeature_handler: Optional["MyFeatureHandler"] = None

    @classmethod
    def init_handlers(cls, ctx: dict):
        # ...existing handlers...
        cls._myfeature_handler = MyFeatureHandler(ctx)

    def _try_modular_handler(self, path: str, query: dict) -> bool:
        handlers = [
            # ...existing handlers...
            self._myfeature_handler,
        ]
        # ...
```

4. **Add tests** (`tests/test_handlers.py`):

```python
class TestMyFeatureHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        ctx = {"storage": MockStorage()}
        return MyFeatureHandler(ctx)

    def test_can_handle_list(self, handler):
        assert handler.can_handle("/api/myfeature") is True

    def test_can_handle_detail(self, handler):
        assert handler.can_handle("/api/myfeature/item-123") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_list_returns_items(self, handler):
        result = handler.handle("/api/myfeature", {}, Mock())
        assert result.status_code == 200
```

## Security Considerations

1. **Path traversal protection**: Use centralized validation functions from `aragora.server.validation`:
   ```python
   from aragora.server.validation import validate_id, validate_debate_id, validate_agent_name

   # For generic IDs
   is_valid, err = validate_id(item_id, "item ID")
   if not is_valid:
       return error_response(err, 400)

   # For debate IDs (longer allowed)
   is_valid, err = validate_debate_id(debate_id)
   if not is_valid:
       return error_response(err, 400)

   # For agent names
   is_valid, err = validate_agent_name(agent)
   if not is_valid:
       return error_response(err, 400)
   ```

2. **Rate limiting**: Applied at the server level before handlers are called

3. **Input validation**: Use the validation utilities from `aragora.server.validation`

4. **Error sanitization**: Never expose internal error details to clients:
   ```python
   def _safe_error_message(e: Exception, context: str) -> str:
       logger.error(f"Error in {context}: {e}", exc_info=True)
       return "An error occurred"
   ```

## Cache TTL Guidelines

| Data Type | Recommended TTL | Rationale |
|-----------|-----------------|-----------|
| Leaderboards | 300s | Changes with each debate |
| Agent profiles | 600s | Relatively stable |
| Recent matches | 120s | Updates frequently |
| Analytics aggregates | 1800s | Computationally expensive |
| Static configuration | 3600s | Rarely changes |

## Testing

Tests are located in `tests/test_handlers.py` and `tests/test_handlers_e2e.py`.

**Unit tests** focus on:
- `can_handle()` routing logic
- Individual endpoint behavior
- Error handling
- Edge cases

**E2E tests** cover:
- Full request/response cycle
- Integration with storage
- Cache behavior

**Clear cache between tests** to ensure isolation:
```python
@pytest.fixture
def handler(self, tmp_path):
    from aragora.server.handlers.base import clear_cache
    clear_cache()
    ctx = {"storage": MockStorage()}
    return MyFeatureHandler(ctx)
```
