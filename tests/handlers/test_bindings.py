"""Tests for bindings handler.

Tests the message bindings management API endpoints:
- GET /api/bindings - List all message bindings
- GET /api/bindings/:provider - List bindings for a provider
- GET /api/bindings/stats - Get router statistics
- POST /api/bindings - Create a new binding
- POST /api/bindings/resolve - Resolve binding for a message
- DELETE /api/bindings/:provider/:account/:pattern - Remove a binding

Covers:
- All routes and HTTP methods
- Success paths for all endpoints
- Bindings system unavailable (BINDINGS_AVAILABLE=False)
- Rate limiting enforcement
- Router unavailable (get_binding_router returns None)
- Input validation (missing fields, invalid binding_type)
- Error handling (parse_json_body errors)
- Path parsing and edge cases
- Version prefix stripping
- Handler initialization and routing
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _make_request(
    method: str = "GET",
    path: str = "/",
    body: dict | None = None,
):
    """Create a mock aiohttp request with JSON body."""
    req = MagicMock(spec=web.Request)
    req.method = method
    req.path = path
    req.headers = {"Authorization": "Bearer test-token"}
    req.client_address = ("127.0.0.1", 12345)

    if body is not None:
        req.content_length = len(json.dumps(body).encode())
        req.json = AsyncMock(return_value=body)
        req.read = AsyncMock(return_value=json.dumps(body).encode())
    else:
        req.content_length = 0
        req.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))
        req.read = AsyncMock(return_value=b"")

    return req


# ============================================================================
# Mock BindingType and MessageBinding
# ============================================================================


class MockBindingType(Enum):
    """Mock BindingType enum."""
    default = "default"
    direct = "direct"
    broadcast = "broadcast"


class MockMessageBinding:
    """Mock MessageBinding class."""

    def __init__(
        self,
        provider: str = "telegram",
        account_id: str = "acc1",
        peer_pattern: str = "*",
        agent_binding: str = "agent1",
        binding_type: MockBindingType = MockBindingType.default,
        priority: int = 0,
        time_window_start: str | None = None,
        time_window_end: str | None = None,
        allowed_users: set | None = None,
        blocked_users: set | None = None,
        config_overrides: dict | None = None,
        name: str | None = None,
        description: str | None = None,
        enabled: bool = True,
    ):
        self.provider = provider
        self.account_id = account_id
        self.peer_pattern = peer_pattern
        self.agent_binding = agent_binding
        self.binding_type = binding_type
        self.priority = priority
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.allowed_users = allowed_users
        self.blocked_users = blocked_users
        self.config_overrides = config_overrides or {}
        self.name = name
        self.description = description
        self.enabled = enabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "account_id": self.account_id,
            "peer_pattern": self.peer_pattern,
            "agent_binding": self.agent_binding,
            "binding_type": self.binding_type.value,
            "priority": self.priority,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
        }


class MockResolution:
    """Mock resolution result."""

    def __init__(
        self,
        matched: bool = True,
        agent_binding: str | None = "agent1",
        binding_type: MockBindingType | None = MockBindingType.default,
        config_overrides: dict | None = None,
        match_reason: str | None = "exact_match",
        candidates_checked: int = 1,
        binding: MockMessageBinding | None = None,
    ):
        self.matched = matched
        self.agent_binding = agent_binding
        self.binding_type = binding_type
        self.config_overrides = config_overrides or {}
        self.match_reason = match_reason
        self.candidates_checked = candidates_checked
        self.binding = binding


class MockBindingRouter:
    """Mock BindingRouter class."""

    def __init__(self):
        self._bindings: list[MockMessageBinding] = []

    def list_bindings(self, provider: str | None = None) -> list[MockMessageBinding]:
        if provider:
            return [b for b in self._bindings if b.provider == provider]
        return list(self._bindings)

    def add_binding(self, binding: MockMessageBinding) -> None:
        self._bindings.append(binding)

    def remove_binding(self, provider: str, account_id: str, peer_pattern: str) -> bool:
        for i, b in enumerate(self._bindings):
            if (
                b.provider == provider
                and b.account_id == account_id
                and b.peer_pattern == peer_pattern
            ):
                self._bindings.pop(i)
                return True
        return False

    def resolve(
        self,
        provider: str,
        account_id: str,
        peer_id: str,
        user_id: str | None = None,
        hour: int | None = None,
    ) -> MockResolution:
        for b in self._bindings:
            if b.provider == provider and b.account_id == account_id:
                return MockResolution(
                    matched=True,
                    agent_binding=b.agent_binding,
                    binding_type=b.binding_type,
                    config_overrides=b.config_overrides,
                    match_reason="exact_match",
                    candidates_checked=1,
                    binding=b,
                )
        return MockResolution(
            matched=False,
            agent_binding=None,
            binding_type=None,
            match_reason="no_match",
            candidates_checked=len(self._bindings),
            binding=None,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_bindings": len(self._bindings),
            "providers": list({b.provider for b in self._bindings}),
            "total_resolutions": 42,
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_router():
    """Create a mock binding router."""
    return MockBindingRouter()


@pytest.fixture
def handler(mock_router):
    """Create a BindingsHandler with mocked dependencies."""
    with patch.dict(
        "aragora.server.handlers.bindings.__dict__",
        {
            "BINDINGS_AVAILABLE": True,
            "BindingType": MockBindingType,
            "MessageBinding": MockMessageBinding,
            "get_binding_router": lambda: mock_router,
        },
    ):
        from aragora.server.handlers.bindings import BindingsHandler
        h = BindingsHandler(server_context={})
        h._router = mock_router
        yield h


@pytest.fixture
def handler_no_bindings():
    """Create a BindingsHandler with bindings system unavailable."""
    with patch(
        "aragora.server.handlers.bindings.BINDINGS_AVAILABLE", False,
    ):
        from aragora.server.handlers.bindings import BindingsHandler
        h = BindingsHandler(server_context={})
        yield h


@pytest.fixture
def handler_no_router():
    """Create a BindingsHandler where get_binding_router returns None."""
    with patch.dict(
        "aragora.server.handlers.bindings.__dict__",
        {
            "BINDINGS_AVAILABLE": True,
            "BindingType": MockBindingType,
            "MessageBinding": MockMessageBinding,
            "get_binding_router": lambda: None,
        },
    ):
        from aragora.server.handlers.bindings import BindingsHandler
        h = BindingsHandler(server_context={})
        h._router = None
        yield h


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests."""
    from aragora.server.handlers.bindings import _bindings_limiter
    _bindings_limiter._buckets.clear()
    yield
    _bindings_limiter._buckets.clear()


# ============================================================================
# Test: Handler Initialization
# ============================================================================


class TestHandlerInit:
    """Tests for BindingsHandler initialization."""

    def test_routes_defined(self, handler):
        """Test ROUTES class attribute is set."""
        assert "/api/bindings" in handler.ROUTES
        assert "/api/bindings/resolve" in handler.ROUTES
        assert "/api/bindings/stats" in handler.ROUTES
        assert "/api/bindings/*" in handler.ROUTES

    def test_wildcard_route_is_last(self, handler):
        """Test wildcard route is last to avoid greedy matching."""
        assert handler.ROUTES[-1] == "/api/bindings/*"

    def test_router_initialized_to_none(self):
        """Test _router starts as None."""
        with patch(
            "aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True,
        ):
            from aragora.server.handlers.bindings import BindingsHandler
            h = BindingsHandler(server_context={})
            # Before any request, router is None
            assert h._router is None

    def test_get_router_returns_router_when_available(self, handler, mock_router):
        """Test _get_router returns the router instance."""
        result = handler._get_router()
        assert result is mock_router

    def test_get_router_returns_none_when_unavailable(self, handler_no_router):
        """Test _get_router returns None when router is unavailable."""
        result = handler_no_router._get_router()
        assert result is None


# ============================================================================
# Test: GET /api/bindings - List all bindings
# ============================================================================


class TestListBindings:
    """Tests for GET /api/bindings."""

    @pytest.mark.asyncio
    async def test_list_bindings_empty(self, handler):
        """Test listing bindings when none exist."""
        req = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["bindings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_bindings_with_data(self, handler, mock_router):
        """Test listing bindings with existing data."""
        mock_router.add_binding(MockMessageBinding(provider="telegram", account_id="a1"))
        mock_router.add_binding(MockMessageBinding(provider="whatsapp", account_id="a2"))

        req = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["total"] == 2
        assert len(body["bindings"]) == 2

    @pytest.mark.asyncio
    async def test_list_bindings_version_prefix_stripped(self, handler, mock_router):
        """Test that version prefix is stripped from path."""
        mock_router.add_binding(MockMessageBinding(provider="telegram"))
        req = _make_request("GET", "/api/v1/bindings")
        result = await handler.handle_get("/api/v1/bindings", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["total"] == 1


# ============================================================================
# Test: GET /api/bindings/:provider - List bindings by provider
# ============================================================================


class TestListBindingsByProvider:
    """Tests for GET /api/bindings/:provider."""

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, handler, mock_router):
        """Test listing bindings filtered by provider."""
        mock_router.add_binding(MockMessageBinding(provider="telegram", account_id="a1"))
        mock_router.add_binding(MockMessageBinding(provider="whatsapp", account_id="a2"))
        mock_router.add_binding(MockMessageBinding(provider="telegram", account_id="a3"))

        req = _make_request("GET", "/api/bindings/telegram")
        result = await handler.handle_get("/api/bindings/telegram", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["provider"] == "telegram"
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_provider_none_found(self, handler, mock_router):
        """Test filtering by provider with no matches."""
        mock_router.add_binding(MockMessageBinding(provider="telegram"))

        req = _make_request("GET", "/api/bindings/slack")
        result = await handler.handle_get("/api/bindings/slack", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["provider"] == "slack"
        assert body["total"] == 0
        assert body["bindings"] == []

    @pytest.mark.asyncio
    async def test_filter_by_provider_version_prefix(self, handler, mock_router):
        """Test provider filter with version prefix."""
        mock_router.add_binding(MockMessageBinding(provider="whatsapp"))
        req = _make_request("GET", "/api/v1/bindings/whatsapp")
        result = await handler.handle_get("/api/v1/bindings/whatsapp", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["provider"] == "whatsapp"
        assert body["total"] == 1


# ============================================================================
# Test: GET /api/bindings/stats - Router statistics
# ============================================================================


class TestGetStats:
    """Tests for GET /api/bindings/stats."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, handler):
        """Test stats with no bindings."""
        req = _make_request("GET", "/api/bindings/stats")
        result = await handler.handle_get("/api/bindings/stats", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["total_bindings"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_bindings(self, handler, mock_router):
        """Test stats with existing bindings."""
        mock_router.add_binding(MockMessageBinding(provider="telegram"))
        mock_router.add_binding(MockMessageBinding(provider="whatsapp"))

        req = _make_request("GET", "/api/bindings/stats")
        result = await handler.handle_get("/api/bindings/stats", req)
        body = _body(result)
        assert result.status_code == 200
        assert body["total_bindings"] == 2
        assert "providers" in body
        assert body["total_resolutions"] == 42

    @pytest.mark.asyncio
    async def test_get_stats_version_prefix(self, handler):
        """Test stats with version prefix."""
        req = _make_request("GET", "/api/v1/bindings/stats")
        result = await handler.handle_get("/api/v1/bindings/stats", req)
        assert result.status_code == 200


# ============================================================================
# Test: POST /api/bindings - Create binding
# ============================================================================


class TestCreateBinding:
    """Tests for POST /api/bindings."""

    @pytest.mark.asyncio
    async def test_create_binding_success(self, handler, mock_router):
        """Test successfully creating a binding."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "chat_*",
            "agent_binding": "agent_debate",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 201
        assert resp["status"] == "created"
        assert resp["binding"]["provider"] == "telegram"
        assert len(mock_router._bindings) == 1

    @pytest.mark.asyncio
    async def test_create_binding_with_all_fields(self, handler, mock_router):
        """Test creating a binding with all optional fields."""
        body = {
            "provider": "whatsapp",
            "account_id": "acc2",
            "peer_pattern": "group_*",
            "agent_binding": "debate_v2",
            "binding_type": "direct",
            "priority": 5,
            "name": "My Binding",
            "description": "Test description",
            "enabled": False,
            "config_overrides": {"timeout": 30},
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 201
        assert resp["binding"]["name"] == "My Binding"

    @pytest.mark.asyncio
    async def test_create_binding_with_broadcast_type(self, handler, mock_router):
        """Test creating a binding with broadcast type."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "broadcast_agent",
            "binding_type": "broadcast",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_binding_missing_provider(self, handler):
        """Test creating a binding without provider field."""
        body = {
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "provider" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_account_id(self, handler):
        """Test creating a binding without account_id field."""
        body = {
            "provider": "telegram",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "account_id" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_peer_pattern(self, handler):
        """Test creating a binding without peer_pattern field."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "peer_pattern" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_agent_binding(self, handler):
        """Test creating a binding without agent_binding field."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "agent_binding" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_multiple_fields(self, handler):
        """Test creating a binding with multiple missing fields."""
        body = {"provider": "telegram"}
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "account_id" in resp["error"]
        assert "peer_pattern" in resp["error"]
        assert "agent_binding" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_empty_body(self, handler):
        """Test creating a binding with empty body."""
        body = {}
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "Missing required fields" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_invalid_binding_type(self, handler):
        """Test creating a binding with an invalid binding_type."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "binding_type": "invalid_type",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "Invalid binding_type" in resp["error"]

    @pytest.mark.asyncio
    async def test_create_binding_default_priority(self, handler, mock_router):
        """Test that default priority is 0."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].priority == 0

    @pytest.mark.asyncio
    async def test_create_binding_custom_priority(self, handler, mock_router):
        """Test creating a binding with custom priority."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "priority": 10,
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].priority == 10

    @pytest.mark.asyncio
    async def test_create_binding_with_allowed_users(self, handler, mock_router):
        """Test creating a binding with allowed_users."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "allowed_users": ["user1", "user2"],
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].allowed_users == {"user1", "user2"}

    @pytest.mark.asyncio
    async def test_create_binding_with_blocked_users(self, handler, mock_router):
        """Test creating a binding with blocked_users."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "blocked_users": ["spammer1"],
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].blocked_users == {"spammer1"}

    @pytest.mark.asyncio
    async def test_create_binding_with_time_window(self, handler, mock_router):
        """Test creating a binding with time window constraints."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "time_window_start": "09:00",
            "time_window_end": "17:00",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].time_window_start == "09:00"
        assert mock_router._bindings[0].time_window_end == "17:00"

    @pytest.mark.asyncio
    async def test_create_binding_enabled_default_true(self, handler, mock_router):
        """Test that enabled defaults to True."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].enabled is True

    @pytest.mark.asyncio
    async def test_create_binding_enabled_false(self, handler, mock_router):
        """Test creating a disabled binding."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "enabled": False,
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].enabled is False

    @pytest.mark.asyncio
    async def test_create_binding_version_prefix(self, handler, mock_router):
        """Test creating with version prefix."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/v1/bindings", body=body)
        result = await handler.handle_post("/api/v1/bindings", req)
        assert result.status_code == 201


# ============================================================================
# Test: POST /api/bindings/resolve - Resolve binding
# ============================================================================


class TestResolveBinding:
    """Tests for POST /api/bindings/resolve."""

    @pytest.mark.asyncio
    async def test_resolve_match(self, handler, mock_router):
        """Test resolving a binding that matches."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", agent_binding="agent_x")
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 200
        assert resp["matched"] is True
        assert resp["agent_binding"] == "agent_x"
        assert resp["match_reason"] == "exact_match"

    @pytest.mark.asyncio
    async def test_resolve_no_match(self, handler, mock_router):
        """Test resolving when no binding matches."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 200
        assert resp["matched"] is False
        assert resp["agent_binding"] is None
        assert resp["binding"] is None

    @pytest.mark.asyncio
    async def test_resolve_with_user_id(self, handler, mock_router):
        """Test resolving with optional user_id."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1")
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
            "user_id": "user456",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_resolve_with_hour(self, handler, mock_router):
        """Test resolving with optional hour parameter."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1")
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
            "hour": 14,
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_resolve_missing_provider(self, handler):
        """Test resolve with missing provider field."""
        body = {"account_id": "acc1", "peer_id": "chat123"}
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "provider" in resp["error"]

    @pytest.mark.asyncio
    async def test_resolve_missing_account_id(self, handler):
        """Test resolve with missing account_id field."""
        body = {"provider": "telegram", "peer_id": "chat123"}
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "account_id" in resp["error"]

    @pytest.mark.asyncio
    async def test_resolve_missing_peer_id(self, handler):
        """Test resolve with missing peer_id field."""
        body = {"provider": "telegram", "account_id": "acc1"}
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "peer_id" in resp["error"]

    @pytest.mark.asyncio
    async def test_resolve_missing_all_fields(self, handler):
        """Test resolve with empty body."""
        body = {}
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_resolve_version_prefix(self, handler, mock_router):
        """Test resolve with version prefix."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1")
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/v1/bindings/resolve", body=body)
        result = await handler.handle_post("/api/v1/bindings/resolve", req)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_resolve_response_format(self, handler, mock_router):
        """Test resolve response contains all expected fields."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1")
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert "matched" in resp
        assert "agent_binding" in resp
        assert "binding_type" in resp
        assert "config_overrides" in resp
        assert "match_reason" in resp
        assert "candidates_checked" in resp
        assert "binding" in resp


# ============================================================================
# Test: DELETE /api/bindings/:provider/:account/:pattern
# ============================================================================


class TestDeleteBinding:
    """Tests for DELETE /api/bindings/:provider/:account/:pattern."""

    @pytest.mark.asyncio
    async def test_delete_existing_binding(self, handler, mock_router):
        """Test deleting an existing binding."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="chat_*")
        )
        req = _make_request("DELETE", "/api/bindings/telegram/acc1/chat_*")
        result = await handler.handle_delete("/api/bindings/telegram/acc1/chat_*", req)
        resp = _body(result)
        assert result.status_code == 200
        assert resp["status"] == "deleted"
        assert len(mock_router._bindings) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_binding(self, handler, mock_router):
        """Test deleting a binding that does not exist."""
        req = _make_request("DELETE", "/api/bindings/telegram/acc1/chat_*")
        result = await handler.handle_delete("/api/bindings/telegram/acc1/chat_*", req)
        resp = _body(result)
        assert result.status_code == 404
        assert resp["error"]["code"] == "BINDING_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_delete_path_too_short(self, handler):
        """Test delete with too few path segments."""
        req = _make_request("DELETE", "/api/bindings/telegram")
        result = await handler.handle_delete("/api/bindings/telegram", req)
        resp = _body(result)
        assert result.status_code == 400
        assert "Delete path must be" in resp["error"]

    @pytest.mark.asyncio
    async def test_delete_path_provider_account_only(self, handler):
        """Test delete with only provider and account (no pattern)."""
        req = _make_request("DELETE", "/api/bindings/telegram/acc1")
        result = await handler.handle_delete("/api/bindings/telegram/acc1", req)
        resp = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_with_slash_in_pattern(self, handler, mock_router):
        """Test deleting a binding where the pattern contains slashes."""
        mock_router.add_binding(
            MockMessageBinding(
                provider="whatsapp",
                account_id="acc2",
                peer_pattern="group/sub/pattern",
            )
        )
        path = "/api/bindings/whatsapp/acc2/group/sub/pattern"
        req = _make_request("DELETE", path)
        result = await handler.handle_delete(path, req)
        resp = _body(result)
        assert result.status_code == 200
        assert resp["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_version_prefix(self, handler, mock_router):
        """Test delete with version prefix."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="*")
        )
        req = _make_request("DELETE", "/api/v1/bindings/telegram/acc1/*")
        result = await handler.handle_delete("/api/v1/bindings/telegram/acc1/*", req)
        resp = _body(result)
        assert result.status_code == 200
        assert resp["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_wrong_provider(self, handler, mock_router):
        """Test delete with wrong provider returns 404."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="*")
        )
        req = _make_request("DELETE", "/api/bindings/whatsapp/acc1/*")
        result = await handler.handle_delete("/api/bindings/whatsapp/acc1/*", req)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_wrong_account(self, handler, mock_router):
        """Test delete with wrong account_id returns 404."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="*")
        )
        req = _make_request("DELETE", "/api/bindings/telegram/wrong_acc/*")
        result = await handler.handle_delete("/api/bindings/telegram/wrong_acc/*", req)
        assert result.status_code == 404


# ============================================================================
# Test: Bindings system unavailable (503)
# ============================================================================


class TestBindingsUnavailable:
    """Tests for when bindings system is not available."""

    @pytest.mark.asyncio
    async def test_get_unavailable(self, handler_no_bindings):
        """Test GET returns 503 when bindings unavailable."""
        req = _make_request("GET", "/api/bindings")
        result = await handler_no_bindings.handle_get("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_post_unavailable(self, handler_no_bindings):
        """Test POST returns 503 when bindings unavailable."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler_no_bindings.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_delete_unavailable(self, handler_no_bindings):
        """Test DELETE returns 503 when bindings unavailable."""
        req = _make_request("DELETE", "/api/bindings/telegram/acc1/*")
        result = await handler_no_bindings.handle_delete(
            "/api/bindings/telegram/acc1/*", req
        )
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_resolve_unavailable(self, handler_no_bindings):
        """Test resolve returns 503 when bindings unavailable."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler_no_bindings.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_stats_unavailable(self, handler_no_bindings):
        """Test stats returns 503 when bindings unavailable."""
        req = _make_request("GET", "/api/bindings/stats")
        result = await handler_no_bindings.handle_get("/api/bindings/stats", req)
        resp = _body(result)
        assert result.status_code == 503


# ============================================================================
# Test: Router unavailable (503)
# ============================================================================


class TestRouterUnavailable:
    """Tests for when binding router cannot be obtained."""

    @pytest.mark.asyncio
    async def test_list_bindings_no_router(self, handler_no_router):
        """Test list bindings returns 503 when router is None."""
        req = _make_request("GET", "/api/bindings")
        result = await handler_no_router.handle_get("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_list_by_provider_no_router(self, handler_no_router):
        """Test list by provider returns 503 when router is None."""
        req = _make_request("GET", "/api/bindings/telegram")
        result = await handler_no_router.handle_get("/api/bindings/telegram", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_stats_no_router(self, handler_no_router):
        """Test stats returns 503 when router is None."""
        req = _make_request("GET", "/api/bindings/stats")
        result = await handler_no_router.handle_get("/api/bindings/stats", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_create_no_router(self, handler_no_router):
        """Test create returns 503 when router is None."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler_no_router.handle_post("/api/bindings", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_resolve_no_router(self, handler_no_router):
        """Test resolve returns 503 when router is None."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler_no_router.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_delete_no_router(self, handler_no_router):
        """Test delete returns 503 when router is None."""
        req = _make_request("DELETE", "/api/bindings/telegram/acc1/*")
        result = await handler_no_router.handle_delete(
            "/api/bindings/telegram/acc1/*", req
        )
        resp = _body(result)
        assert result.status_code == 503
        assert resp["error"]["code"] == "ROUTER_UNAVAILABLE"


# ============================================================================
# Test: Rate limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_rate_limit_get(self, handler):
        """Test rate limit on GET endpoint."""
        with patch(
            "aragora.server.handlers.bindings._bindings_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            req = _make_request("GET", "/api/bindings")
            result = await handler.handle_get("/api/bindings", req)
            resp = _body(result)
            assert result.status_code == 429
            assert resp["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_rate_limit_post(self, handler):
        """Test rate limit on POST endpoint."""
        with patch(
            "aragora.server.handlers.bindings._bindings_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            body = {
                "provider": "telegram",
                "account_id": "acc1",
                "peer_pattern": "*",
                "agent_binding": "agent1",
            }
            req = _make_request("POST", "/api/bindings", body=body)
            result = await handler.handle_post("/api/bindings", req)
            resp = _body(result)
            assert result.status_code == 429
            assert resp["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_rate_limit_delete(self, handler):
        """Test rate limit on DELETE endpoint."""
        with patch(
            "aragora.server.handlers.bindings._bindings_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            req = _make_request("DELETE", "/api/bindings/telegram/acc1/*")
            result = await handler.handle_delete("/api/bindings/telegram/acc1/*", req)
            resp = _body(result)
            assert result.status_code == 429
            assert resp["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_rate_limit_allowed_passes(self, handler, mock_router):
        """Test that allowed requests pass through rate limiter."""
        with patch(
            "aragora.server.handlers.bindings._bindings_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            req = _make_request("GET", "/api/bindings")
            result = await handler.handle_get("/api/bindings", req)
            assert result.status_code == 200


# ============================================================================
# Test: Unknown endpoints (404)
# ============================================================================


class TestUnknownEndpoints:
    """Tests for unknown endpoint paths."""

    @pytest.mark.asyncio
    async def test_get_unknown_path(self, handler):
        """Test GET on an unknown sub-path returns 404."""
        # A path like /api/bindings with fewer than 4 parts but not matching known routes
        # In practice the handler routing logic covers known routes first;
        # an unknown short path after strip would still trigger _list_bindings or provider filter.
        # We test a truly mismatched path by ensuring it falls through.
        # Actually, /api/bindings/something will match the provider route,
        # but something like /api returns 404.
        req = _make_request("GET", "/api")
        result = await handler.handle_get("/api", req)
        resp = _body(result)
        assert result.status_code == 404
        assert "Unknown bindings endpoint" in resp["error"]

    @pytest.mark.asyncio
    async def test_post_unknown_path(self, handler):
        """Test POST on an unknown sub-path returns 404."""
        body = {"data": "test"}
        req = _make_request("POST", "/api/bindings/unknown/path", body=body)
        result = await handler.handle_post("/api/bindings/unknown/path", req)
        resp = _body(result)
        assert result.status_code == 404
        assert "Unknown bindings endpoint" in resp["error"]


# ============================================================================
# Test: JSON body parsing errors
# ============================================================================


class TestJsonBodyErrors:
    """Tests for JSON body parsing error handling."""

    @pytest.mark.asyncio
    async def test_create_no_body(self, handler):
        """Test create with no request body."""
        req = _make_request("POST", "/api/bindings")
        # Empty body request
        result = await handler.handle_post("/api/bindings", req)
        # parse_json_body returns error for empty body
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_resolve_no_body(self, handler):
        """Test resolve with no request body."""
        req = _make_request("POST", "/api/bindings/resolve")
        result = await handler.handle_post("/api/bindings/resolve", req)
        assert result.status_code == 400


# ============================================================================
# Test: _web_response_to_handler_result helper
# ============================================================================


class TestWebResponseConversion:
    """Tests for _web_response_to_handler_result utility."""

    def test_convert_basic_response(self):
        """Test converting a basic web.Response to HandlerResult."""
        from aragora.server.handlers.bindings import _web_response_to_handler_result

        resp = web.Response(
            status=200,
            body=b'{"ok": true}',
            content_type="application/json",
        )
        result = _web_response_to_handler_result(resp)
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert b"ok" in result.body

    def test_convert_empty_body(self):
        """Test converting a response with no body."""
        from aragora.server.handlers.bindings import _web_response_to_handler_result

        resp = web.Response(status=204)
        result = _web_response_to_handler_result(resp)
        assert result.status_code == 204
        assert result.body == b""

    def test_convert_error_response(self):
        """Test converting an error response."""
        from aragora.server.handlers.bindings import _web_response_to_handler_result

        resp = web.json_response(
            {"error": "bad input", "code": "INVALID_JSON"},
            status=400,
        )
        result = _web_response_to_handler_result(resp)
        assert result.status_code == 400


# ============================================================================
# Test: Multiple bindings operations
# ============================================================================


class TestMultipleBindingOperations:
    """Tests for sequences of binding operations."""

    @pytest.mark.asyncio
    async def test_create_then_list(self, handler, mock_router):
        """Test creating a binding then listing all."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req_create = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req_create)

        req_list = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req_list)
        resp = _body(result)
        assert resp["total"] == 1

    @pytest.mark.asyncio
    async def test_create_then_delete(self, handler, mock_router):
        """Test creating a binding then deleting it."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "chat_*",
            "agent_binding": "agent1",
        }
        req_create = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req_create)

        req_delete = _make_request("DELETE", "/api/bindings/telegram/acc1/chat_*")
        result = await handler.handle_delete("/api/bindings/telegram/acc1/chat_*", req_delete)
        assert result.status_code == 200

        req_list = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req_list)
        resp = _body(result)
        assert resp["total"] == 0

    @pytest.mark.asyncio
    async def test_create_then_resolve(self, handler, mock_router):
        """Test creating a binding then resolving it."""
        create_body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "chat_*",
            "agent_binding": "my_agent",
        }
        req_create = _make_request("POST", "/api/bindings", body=create_body)
        await handler.handle_post("/api/bindings", req_create)

        resolve_body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat_123",
        }
        req_resolve = _make_request("POST", "/api/bindings/resolve", body=resolve_body)
        result = await handler.handle_post("/api/bindings/resolve", req_resolve)
        resp = _body(result)
        assert resp["matched"] is True
        assert resp["agent_binding"] == "my_agent"

    @pytest.mark.asyncio
    async def test_create_multiple_different_providers(self, handler, mock_router):
        """Test creating bindings for different providers."""
        for provider in ["telegram", "whatsapp", "slack"]:
            body = {
                "provider": provider,
                "account_id": f"acc_{provider}",
                "peer_pattern": "*",
                "agent_binding": f"agent_{provider}",
            }
            req = _make_request("POST", "/api/bindings", body=body)
            result = await handler.handle_post("/api/bindings", req)
            assert result.status_code == 201

        req_list = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req_list)
        resp = _body(result)
        assert resp["total"] == 3


# ============================================================================
# Test: Edge cases and additional coverage
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and additional coverage."""

    @pytest.mark.asyncio
    async def test_binding_to_dict_in_list_response(self, handler, mock_router):
        """Test that binding to_dict is called for each binding in list."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", name="b1")
        )
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", name="b2")
        )
        req = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req)
        resp = _body(result)
        names = [b["name"] for b in resp["bindings"]]
        assert "b1" in names
        assert "b2" in names

    @pytest.mark.asyncio
    async def test_resolve_binding_type_value(self, handler, mock_router):
        """Test that binding_type value is returned in resolve response."""
        mock_router.add_binding(
            MockMessageBinding(
                provider="telegram",
                account_id="acc1",
                binding_type=MockBindingType.direct,
            )
        )
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert resp["binding_type"] == "direct"

    @pytest.mark.asyncio
    async def test_resolve_no_match_binding_type_none(self, handler):
        """Test that unmatched resolve returns None for binding_type."""
        body = {
            "provider": "nonexistent",
            "account_id": "acc1",
            "peer_id": "chat123",
        }
        req = _make_request("POST", "/api/bindings/resolve", body=body)
        result = await handler.handle_post("/api/bindings/resolve", req)
        resp = _body(result)
        assert resp["binding_type"] is None

    @pytest.mark.asyncio
    async def test_create_binding_with_config_overrides(self, handler, mock_router):
        """Test config_overrides are properly stored."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
            "config_overrides": {"model": "claude", "temperature": 0.7},
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        binding = mock_router._bindings[0]
        assert binding.config_overrides["model"] == "claude"
        assert binding.config_overrides["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_delete_preserves_other_bindings(self, handler, mock_router):
        """Test that deleting one binding does not affect others."""
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="a")
        )
        mock_router.add_binding(
            MockMessageBinding(provider="telegram", account_id="acc1", peer_pattern="b")
        )
        req = _make_request("DELETE", "/api/bindings/telegram/acc1/a")
        result = await handler.handle_delete("/api/bindings/telegram/acc1/a", req)
        assert result.status_code == 200
        assert len(mock_router._bindings) == 1
        assert mock_router._bindings[0].peer_pattern == "b"

    @pytest.mark.asyncio
    async def test_stats_returns_json_content_type(self, handler):
        """Test that stats response has correct content type."""
        req = _make_request("GET", "/api/bindings/stats")
        result = await handler.handle_get("/api/bindings/stats", req)
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_create_returns_json_content_type(self, handler, mock_router):
        """Test that create response has correct content type."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        result = await handler.handle_post("/api/bindings", req)
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_create_binding_no_name(self, handler, mock_router):
        """Test creating a binding without a name field."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].name is None

    @pytest.mark.asyncio
    async def test_create_binding_no_description(self, handler, mock_router):
        """Test creating a binding without a description field."""
        body = {
            "provider": "telegram",
            "account_id": "acc1",
            "peer_pattern": "*",
            "agent_binding": "agent1",
        }
        req = _make_request("POST", "/api/bindings", body=body)
        await handler.handle_post("/api/bindings", req)
        assert mock_router._bindings[0].description is None

    @pytest.mark.asyncio
    async def test_list_bindings_response_structure(self, handler, mock_router):
        """Test list response has bindings array and total count."""
        mock_router.add_binding(MockMessageBinding())
        req = _make_request("GET", "/api/bindings")
        result = await handler.handle_get("/api/bindings", req)
        resp = _body(result)
        assert isinstance(resp["bindings"], list)
        assert isinstance(resp["total"], int)
        assert resp["total"] == len(resp["bindings"])

    @pytest.mark.asyncio
    async def test_provider_filter_response_includes_provider_name(self, handler, mock_router):
        """Test provider-filtered list includes provider name in response."""
        mock_router.add_binding(MockMessageBinding(provider="telegram"))
        req = _make_request("GET", "/api/bindings/telegram")
        result = await handler.handle_get("/api/bindings/telegram", req)
        resp = _body(result)
        assert "provider" in resp
        assert resp["provider"] == "telegram"
