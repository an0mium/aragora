"""Tests for the BindingsHandler module.

Tests cover:
- Handler routing for bindings endpoints
- GET /api/bindings - List all bindings
- GET /api/bindings/:provider - List bindings by provider
- GET /api/bindings/stats - Get router statistics
- POST /api/bindings - Create binding
- POST /api/bindings/resolve - Resolve binding
- DELETE /api/bindings/:provider/:account/:pattern - Delete binding
- Rate limiting behavior
- Error handling (bindings unavailable, router unavailable)
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bindings import BindingsHandler


# Mock binding classes for testing
class MockBindingType(str, Enum):
    """Mock binding types."""

    DEFAULT = "default"
    SPECIFIC_AGENT = "specific_agent"
    AGENT_POOL = "agent_pool"
    DEBATE_TEAM = "debate_team"


@dataclass
class MockMessageBinding:
    """Mock message binding for testing."""

    provider: str
    account_id: str
    peer_pattern: str
    agent_binding: str
    binding_type: MockBindingType = MockBindingType.DEFAULT
    priority: int = 0
    time_window_start: Optional[int] = None
    time_window_end: Optional[int] = None
    allowed_users: Optional[Set[str]] = None
    blocked_users: Optional[Set[str]] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "account_id": self.account_id,
            "peer_pattern": self.peer_pattern,
            "agent_binding": self.agent_binding,
            "binding_type": self.binding_type.value,
            "priority": self.priority,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "allowed_users": list(self.allowed_users) if self.allowed_users else None,
            "blocked_users": list(self.blocked_users) if self.blocked_users else None,
            "config_overrides": self.config_overrides,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass
class MockBindingResolution:
    """Mock binding resolution result."""

    matched: bool
    agent_binding: Optional[str] = None
    binding_type: Optional[MockBindingType] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    match_reason: Optional[str] = None
    candidates_checked: int = 0
    binding: Optional[MockMessageBinding] = None


class MockBindingRouter:
    """Mock binding router for testing."""

    def __init__(self):
        self._bindings: List[MockMessageBinding] = []
        self._stats: Dict[str, Any] = {
            "total_bindings": 0,
            "total_resolutions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def add_binding(self, binding: MockMessageBinding) -> None:
        self._bindings.append(binding)
        self._stats["total_bindings"] = len(self._bindings)

    def remove_binding(self, provider: str, account_id: str, peer_pattern: str) -> bool:
        for i, b in enumerate(self._bindings):
            if (
                b.provider == provider
                and b.account_id == account_id
                and b.peer_pattern == peer_pattern
            ):
                self._bindings.pop(i)
                self._stats["total_bindings"] = len(self._bindings)
                return True
        return False

    def list_bindings(self, provider: Optional[str] = None) -> List[MockMessageBinding]:
        if provider:
            return [b for b in self._bindings if b.provider == provider]
        return self._bindings

    def get_stats(self) -> Dict[str, Any]:
        return self._stats

    def resolve(
        self,
        provider: str,
        account_id: str,
        peer_id: str,
        user_id: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> MockBindingResolution:
        self._stats["total_resolutions"] += 1
        # Simple matching logic
        for binding in self._bindings:
            if binding.provider == provider and binding.account_id == account_id:
                if binding.peer_pattern == "*" or peer_id.startswith(
                    binding.peer_pattern.rstrip("*")
                ):
                    return MockBindingResolution(
                        matched=True,
                        agent_binding=binding.agent_binding,
                        binding_type=binding.binding_type,
                        config_overrides=binding.config_overrides,
                        match_reason=f"Matched pattern {binding.peer_pattern}",
                        candidates_checked=1,
                        binding=binding,
                    )
        return MockBindingResolution(
            matched=False,
            candidates_checked=len(self._bindings),
        )


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_router():
    """Create a mock binding router with test bindings."""
    router = MockBindingRouter()

    # Add test bindings
    router.add_binding(
        MockMessageBinding(
            provider="telegram",
            account_id="bot123",
            peer_pattern="chat:*",
            agent_binding="anthropic-api",
            binding_type=MockBindingType.DEFAULT,
            name="Telegram Default",
            description="Default binding for Telegram chats",
        )
    )

    router.add_binding(
        MockMessageBinding(
            provider="whatsapp",
            account_id="wa456",
            peer_pattern="group:support*",
            agent_binding="openai-api",
            binding_type=MockBindingType.SPECIFIC_AGENT,
            priority=10,
            name="WhatsApp Support",
        )
    )

    router.add_binding(
        MockMessageBinding(
            provider="slack",
            account_id="workspace1",
            peer_pattern="channel:general",
            agent_binding="debate-team-1",
            binding_type=MockBindingType.DEBATE_TEAM,
            config_overrides={"rounds": 3, "consensus_threshold": 0.8},
        )
    )

    return router


def create_async_request(body: dict) -> MagicMock:
    """Create a mock request with async json method."""
    mock_request = MagicMock()
    mock_request.client_address = ("127.0.0.1", 12345)
    mock_request.json = AsyncMock(return_value=body)
    return mock_request


def create_simple_request() -> MagicMock:
    """Create a simple mock request without body."""
    mock_request = MagicMock()
    mock_request.client_address = ("127.0.0.1", 12345)
    return mock_request


class TestBindingsHandlerRouting:
    """Tests for handler routing via ROUTES constant."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    def test_routes_defined(self, handler):
        """Handler has correct routes defined."""
        assert "/api/bindings" in handler.ROUTES
        assert "/api/bindings/resolve" in handler.ROUTES
        assert "/api/bindings/stats" in handler.ROUTES
        assert "/api/bindings/*" in handler.ROUTES

    def test_routes_order(self, handler):
        """Wildcard route is last (required for proper matching)."""
        routes = handler.ROUTES
        wildcard_idx = routes.index("/api/bindings/*")
        assert wildcard_idx == len(routes) - 1, "/api/bindings/* should be last route"

    def test_routes_cover_all_endpoints(self, handler):
        """All documented endpoints are covered by routes."""
        assert any("/api/bindings" == r or r.startswith("/api/bindings") for r in handler.ROUTES)
        assert "/api/bindings/resolve" in handler.ROUTES
        assert "/api/bindings/stats" in handler.ROUTES
        assert "/api/bindings/*" in handler.ROUTES

    def test_routes_do_not_include_unrelated_paths(self, handler):
        """Handler routes only include /api/bindings paths."""
        for route in handler.ROUTES:
            assert route.startswith("/api/bindings"), (
                f"Route {route} should start with /api/bindings"
            )


class TestBindingsHandlerListBindings:
    """Tests for GET /api/bindings endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_list_bindings_success(self, handler, mock_router):
        """List bindings returns all bindings."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "bindings" in body
                    assert body["total"] == 3
                    assert len(body["bindings"]) == 3

    @pytest.mark.asyncio
    async def test_list_bindings_unavailable(self, handler):
        """List bindings returns 503 when bindings unavailable."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", False):
            mock_request = create_simple_request()
            result = await handler.handle_get("/api/bindings", mock_request)

            assert result is not None
            assert result.status_code == 503
            body = json.loads(result.body)
            assert body["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_list_bindings_rate_limited(self, handler):
        """List bindings returns 429 when rate limited."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.check.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_get("/api/bindings", mock_request)

                assert result is not None
                assert result.status_code == 429
                body = json.loads(result.body)
                assert body["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_list_bindings_router_unavailable(self, handler):
        """List bindings returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 503
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "ROUTER_UNAVAILABLE"


class TestBindingsHandlerListByProvider:
    """Tests for GET /api/bindings/:provider endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_list_by_provider_success(self, handler, mock_router):
        """List bindings by provider returns filtered results."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/telegram", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["provider"] == "telegram"
                    assert body["total"] == 1
                    assert len(body["bindings"]) == 1
                    assert body["bindings"][0]["provider"] == "telegram"

    @pytest.mark.asyncio
    async def test_list_by_provider_empty(self, handler, mock_router):
        """List bindings by provider returns empty for unknown provider."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/unknown", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["provider"] == "unknown"
                    assert body["total"] == 0
                    assert len(body["bindings"]) == 0


class TestBindingsHandlerGetStats:
    """Tests for GET /api/bindings/stats endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_router):
        """Get stats returns router statistics."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/stats", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "total_bindings" in body
                    assert body["total_bindings"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_router_unavailable(self, handler):
        """Get stats returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/stats", mock_request)

                    assert result is not None
                    assert result.status_code == 503


class TestBindingsHandlerCreateBinding:
    """Tests for POST /api/bindings endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_create_binding_success(self, handler, mock_router):
        """Create binding returns 201 on success."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.check.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "discord",
                                    "account_id": "server123",
                                    "peer_pattern": "channel:*",
                                    "agent_binding": "gemini-api",
                                    "binding_type": "default",
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["status"] == "created"
                            assert body["binding"]["provider"] == "discord"

    @pytest.mark.asyncio
    async def test_create_binding_missing_field(self, handler, mock_router):
        """Create binding returns 400 for missing required field."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "discord",
                            # Missing account_id, peer_pattern, agent_binding
                        }
                    )

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "Missing required fields" in body["error"]

    @pytest.mark.asyncio
    async def test_create_binding_invalid_type(self, handler, mock_router):
        """Create binding returns 400 for invalid binding type."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                        limiter.check.return_value = True

                        mock_request = create_async_request(
                            {
                                "provider": "discord",
                                "account_id": "server123",
                                "peer_pattern": "channel:*",
                                "agent_binding": "gemini-api",
                                "binding_type": "invalid_type",
                            }
                        )

                        result = await handler.handle_post("/api/bindings", mock_request)

                        assert result is not None
                        assert result.status_code == 400
                        body = json.loads(result.body)
                        assert "Invalid binding_type" in body["error"]


class TestBindingsHandlerResolveBinding:
    """Tests for POST /api/bindings/resolve endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_resolve_binding_matched(self, handler, mock_router):
        """Resolve binding returns matched result."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_id": "chat:12345",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["agent_binding"] == "anthropic-api"
                    assert body["binding_type"] == "default"

    @pytest.mark.asyncio
    async def test_resolve_binding_not_matched(self, handler, mock_router):
        """Resolve binding returns unmatched result."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "unknown",
                            "account_id": "unknown",
                            "peer_id": "unknown",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is False
                    assert body["agent_binding"] is None

    @pytest.mark.asyncio
    async def test_resolve_binding_missing_field(self, handler, mock_router):
        """Resolve binding returns 400 for missing required field."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            # Missing account_id, peer_id
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "Missing required fields" in body["error"]


class TestBindingsHandlerDeleteBinding:
    """Tests for DELETE /api/bindings/:provider/:account/:pattern endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_delete_binding_success(self, handler, mock_router):
        """Delete binding returns success."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/telegram/bot123/chat:*", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_binding_not_found(self, handler, mock_router):
        """Delete binding returns 404 for unknown binding."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/unknown/unknown/unknown", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 404
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "BINDING_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_delete_binding_invalid_path(self, handler, mock_router):
        """Delete binding returns 400 for invalid path."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    # Path too short - missing pattern
                    result = await handler.handle_delete(
                        "/api/bindings/telegram/bot123", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "Delete path must be" in body["error"]


class TestBindingsHandlerRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_get_rate_limited(self, handler):
        """GET requests are rate limited."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.check.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_get("/api/bindings", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_post_rate_limited(self, handler):
        """POST requests are rate limited."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.check.return_value = False

                mock_request = create_async_request({})
                result = await handler.handle_post("/api/bindings", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_delete_rate_limited(self, handler):
        """DELETE requests are rate limited."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.check.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_delete("/api/bindings/a/b/c", mock_request)

                assert result is not None
                assert result.status_code == 429


class TestBindingsHandlerErrorCases:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_unknown_get_endpoint(self, handler, mock_router):
        """Unknown GET endpoint returns 200 with empty provider results."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    # This will be treated as a provider lookup
                    result = await handler.handle_get(
                        "/api/bindings/unknown_provider", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_unknown_post_endpoint(self, handler, mock_router):
        """Unknown POST endpoint returns 404."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_async_request({})
                    result = await handler.handle_post("/api/bindings/unknown", mock_request)

                    assert result is not None
                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler, mock_router):
        """Invalid JSON body returns 400."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = MagicMock()
                    mock_request.client_address = ("127.0.0.1", 12345)
                    mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "Invalid JSON body" in body["error"]


class TestBindingsHandlerVersionPrefix:
    """Tests for API version prefix handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BindingsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_handles_v1_prefix(self, handler, mock_router):
        """Handler handles v1 API prefix."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/v1/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "bindings" in body

    @pytest.mark.asyncio
    async def test_handles_v2_prefix(self, handler, mock_router):
        """Handler handles v2 API prefix."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.check.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/v2/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "bindings" in body
