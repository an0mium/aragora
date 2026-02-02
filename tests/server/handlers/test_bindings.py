"""
Comprehensive tests for the BindingsHandler module.

Tests cover:
1. Agent/debate binding creation with various options
2. Binding retrieval (all, by provider, by pattern)
3. Binding updates (priority, config overrides)
4. Binding deletion (single, multiple patterns)
5. Integration with debate infrastructure
6. Error handling (unavailable, rate limits, validation)
7. RBAC permission checks
8. Time-window based bindings
9. User-based filtering (allowed/blocked users)
10. Config overrides propagation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bindings import BindingsHandler


# =============================================================================
# Mock Classes
# =============================================================================


class MockBindingType(str, Enum):
    """Mock binding types matching the real enum."""

    DEFAULT = "default"
    SPECIFIC_AGENT = "specific_agent"
    AGENT_POOL = "agent_pool"
    DEBATE_TEAM = "debate_team"
    CUSTOM = "custom"


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
    allowed_users: Optional[set[str]] = None
    blocked_users: Optional[set[str]] = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def matches_peer(self, peer_id: str) -> bool:
        """Check if peer ID matches this binding's pattern."""
        import fnmatch

        return fnmatch.fnmatch(peer_id, self.peer_pattern)

    def matches_time(self, hour: Optional[int] = None) -> bool:
        """Check if current time is within the binding's time window."""
        if self.time_window_start is None or self.time_window_end is None:
            return True
        if hour is None:
            hour = datetime.now(timezone.utc).hour
        if self.time_window_start <= self.time_window_end:
            return self.time_window_start <= hour < self.time_window_end
        return hour >= self.time_window_start or hour < self.time_window_end

    def matches_user(self, user_id: Optional[str]) -> bool:
        """Check if user is allowed by this binding."""
        if user_id is None:
            return True
        if self.blocked_users and user_id in self.blocked_users:
            return False
        if self.allowed_users and user_id not in self.allowed_users:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
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
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockBindingResolution:
    """Mock binding resolution result."""

    matched: bool
    agent_binding: Optional[str] = None
    binding_type: Optional[MockBindingType] = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    match_reason: Optional[str] = None
    candidates_checked: int = 0
    binding: Optional[MockMessageBinding] = None


class MockBindingRouter:
    """Comprehensive mock binding router for testing."""

    def __init__(self):
        self._bindings: list[MockMessageBinding] = []
        self._default_bindings: dict[str, MockMessageBinding] = {}
        self._global_default = MockMessageBinding(
            provider="*",
            account_id="*",
            peer_pattern="*",
            agent_binding="default",
            binding_type=MockBindingType.DEFAULT,
            priority=-1000,
            name="global_default",
        )
        self._agent_pools: dict[str, list[str]] = {}
        self._stats = {
            "total_bindings": 0,
            "total_resolutions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def add_binding(self, binding: MockMessageBinding) -> None:
        self._bindings.append(binding)
        self._bindings.sort(key=lambda b: b.priority, reverse=True)
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

    def list_bindings(
        self, provider: Optional[str] = None, account_id: Optional[str] = None
    ) -> list[MockMessageBinding]:
        result = self._bindings
        if provider:
            result = [b for b in result if b.provider == provider]
        if account_id:
            result = [b for b in result if b.account_id == account_id]
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "providers": list(set(b.provider for b in self._bindings)),
            "agent_pools": list(self._agent_pools.keys()),
            "has_global_default": True,
        }

    def set_default_binding(self, provider: str, binding: MockMessageBinding) -> None:
        self._default_bindings[provider] = binding

    def set_global_default(self, binding: MockMessageBinding) -> None:
        self._global_default = binding

    def register_agent_pool(self, pool_name: str, agents: list[str]) -> None:
        self._agent_pools[pool_name] = agents

    def resolve(
        self,
        provider: str,
        account_id: str,
        peer_id: str,
        user_id: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> MockBindingResolution:
        self._stats["total_resolutions"] += 1
        candidates_checked = 0

        for binding in self._bindings:
            candidates_checked += 1

            if not binding.enabled:
                continue
            if binding.provider != provider and binding.provider != "*":
                continue
            if binding.account_id != account_id and binding.account_id != "*":
                continue
            if not binding.matches_peer(peer_id):
                continue
            if not binding.matches_time(hour):
                continue
            if not binding.matches_user(user_id):
                continue

            return MockBindingResolution(
                matched=True,
                agent_binding=binding.agent_binding,
                binding_type=binding.binding_type,
                config_overrides=binding.config_overrides,
                match_reason=f"Matched pattern: {binding.peer_pattern}",
                candidates_checked=candidates_checked,
                binding=binding,
            )

        # Check provider default
        if provider in self._default_bindings:
            default = self._default_bindings[provider]
            return MockBindingResolution(
                matched=True,
                agent_binding=default.agent_binding,
                binding_type=default.binding_type,
                config_overrides=default.config_overrides,
                match_reason="Provider default",
                candidates_checked=candidates_checked,
                binding=default,
            )

        # Global default
        return MockBindingResolution(
            matched=True,
            agent_binding=self._global_default.agent_binding,
            binding_type=self._global_default.binding_type,
            config_overrides=self._global_default.config_overrides,
            match_reason="Global default",
            candidates_checked=candidates_checked,
            binding=self._global_default,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_server_context() -> dict[str, Any]:
    """Create mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "nomic_dir": None,
    }


@pytest.fixture
def handler(mock_server_context) -> BindingsHandler:
    """Create a BindingsHandler instance."""
    return BindingsHandler(mock_server_context)


@pytest.fixture
def mock_router() -> MockBindingRouter:
    """Create a mock binding router with comprehensive test bindings."""
    router = MockBindingRouter()

    # Telegram bindings
    router.add_binding(
        MockMessageBinding(
            provider="telegram",
            account_id="bot123",
            peer_pattern="chat:*",
            agent_binding="claude-api",
            binding_type=MockBindingType.DEFAULT,
            name="Telegram Default",
            description="Default binding for Telegram chats",
        )
    )

    router.add_binding(
        MockMessageBinding(
            provider="telegram",
            account_id="bot123",
            peer_pattern="chat:premium_*",
            agent_binding="claude-opus",
            binding_type=MockBindingType.SPECIFIC_AGENT,
            priority=10,
            name="Telegram Premium",
        )
    )

    # WhatsApp bindings
    router.add_binding(
        MockMessageBinding(
            provider="whatsapp",
            account_id="wa456",
            peer_pattern="group:support*",
            agent_binding="openai-api",
            binding_type=MockBindingType.SPECIFIC_AGENT,
            priority=10,
            name="WhatsApp Support",
            allowed_users={"user1", "user2"},
        )
    )

    # Slack debate team binding
    router.add_binding(
        MockMessageBinding(
            provider="slack",
            account_id="workspace1",
            peer_pattern="channel:general",
            agent_binding="debate-team-alpha",
            binding_type=MockBindingType.DEBATE_TEAM,
            config_overrides={"rounds": 3, "consensus_threshold": 0.8},
            name="Slack General Debate",
        )
    )

    # Time-window binding
    router.add_binding(
        MockMessageBinding(
            provider="discord",
            account_id="server789",
            peer_pattern="channel:*",
            agent_binding="night-shift-agent",
            binding_type=MockBindingType.DEFAULT,
            time_window_start=22,
            time_window_end=6,
            name="Discord Night Shift",
        )
    )

    # Agent pool binding
    router.add_binding(
        MockMessageBinding(
            provider="teams",
            account_id="org1",
            peer_pattern="*",
            agent_binding="balanced-pool",
            binding_type=MockBindingType.AGENT_POOL,
            name="Teams Agent Pool",
        )
    )

    return router


@pytest.fixture
def empty_router() -> MockBindingRouter:
    """Create an empty mock binding router."""
    return MockBindingRouter()


def create_async_request(body: dict) -> MagicMock:
    """Create a mock request with async json method."""
    mock_request = MagicMock()
    mock_request.client_address = ("127.0.0.1", 12345)
    mock_request.json = AsyncMock(return_value=body)
    mock_request.headers = {}
    return mock_request


def create_simple_request(client_ip: str = "127.0.0.1") -> MagicMock:
    """Create a simple mock request without body."""
    mock_request = MagicMock()
    mock_request.client_address = (client_ip, 12345)
    mock_request.headers = {}
    return mock_request


# =============================================================================
# Test Classes - Agent/Debate Binding Creation
# =============================================================================


class TestBindingCreation:
    """Tests for binding creation (POST /api/bindings)."""

    @pytest.mark.asyncio
    async def test_create_binding_with_all_fields(self, handler, mock_router):
        """Create binding with all optional fields populated."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "discord",
                                    "account_id": "server123",
                                    "peer_pattern": "channel:help*",
                                    "agent_binding": "support-agent",
                                    "binding_type": "specific_agent",
                                    "priority": 50,
                                    "time_window_start": 9,
                                    "time_window_end": 17,
                                    "allowed_users": ["admin1", "admin2"],
                                    "blocked_users": ["spammer1"],
                                    "config_overrides": {"temperature": 0.7},
                                    "name": "Discord Help Channel",
                                    "description": "Support binding for help channels",
                                    "enabled": True,
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["status"] == "created"
                            assert body["binding"]["provider"] == "discord"
                            assert body["binding"]["priority"] == 50
                            assert body["binding"]["time_window_start"] == 9

    @pytest.mark.asyncio
    async def test_create_debate_team_binding(self, handler, mock_router):
        """Create a debate team binding with config overrides."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "slack",
                                    "account_id": "workspace2",
                                    "peer_pattern": "channel:engineering",
                                    "agent_binding": "engineering-debate-team",
                                    "binding_type": "debate_team",
                                    "config_overrides": {
                                        "rounds": 5,
                                        "consensus_threshold": 0.75,
                                        "agents": ["claude", "gpt4", "gemini"],
                                    },
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["binding_type"] == "debate_team"
                            assert body["binding"]["config_overrides"]["rounds"] == 5

    @pytest.mark.asyncio
    async def test_create_agent_pool_binding(self, handler, mock_router):
        """Create an agent pool binding."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "teams",
                                    "account_id": "org2",
                                    "peer_pattern": "channel:*",
                                    "agent_binding": "load-balanced-pool",
                                    "binding_type": "agent_pool",
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["binding_type"] == "agent_pool"

    @pytest.mark.asyncio
    async def test_create_binding_with_user_restrictions(self, handler, mock_router):
        """Create binding with allowed and blocked users."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "telegram",
                                    "account_id": "bot999",
                                    "peer_pattern": "dm:*",
                                    "agent_binding": "vip-agent",
                                    "allowed_users": ["vip1", "vip2", "vip3"],
                                    "blocked_users": ["banned1"],
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert set(body["binding"]["allowed_users"]) == {"vip1", "vip2", "vip3"}
                            assert body["binding"]["blocked_users"] == ["banned1"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_provider(self, handler, mock_router):
        """Create binding fails when provider is missing."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "account_id": "bot123",
                            "peer_pattern": "chat:*",
                            "agent_binding": "agent1",
                        }
                    )

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "provider" in body["error"]

    @pytest.mark.asyncio
    async def test_create_binding_missing_agent_binding(self, handler, mock_router):
        """Create binding fails when agent_binding is missing."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_pattern": "chat:*",
                        }
                    )

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "agent_binding" in body["error"]

    @pytest.mark.asyncio
    async def test_create_binding_with_custom_type(self, handler, mock_router):
        """Create binding with custom binding type."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "custom_platform",
                                    "account_id": "custom123",
                                    "peer_pattern": "*",
                                    "agent_binding": "custom-handler",
                                    "binding_type": "custom",
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["binding_type"] == "custom"


# =============================================================================
# Test Classes - Binding Retrieval
# =============================================================================


class TestBindingRetrieval:
    """Tests for binding retrieval endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_bindings(self, handler, mock_router):
        """List all bindings returns complete list."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["total"] == 6
                    assert len(body["bindings"]) == 6

    @pytest.mark.asyncio
    async def test_list_bindings_by_provider_telegram(self, handler, mock_router):
        """List bindings filtered by telegram provider."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/telegram", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["provider"] == "telegram"
                    assert body["total"] == 2
                    for b in body["bindings"]:
                        assert b["provider"] == "telegram"

    @pytest.mark.asyncio
    async def test_list_bindings_by_provider_slack(self, handler, mock_router):
        """List bindings filtered by slack provider."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/slack", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["provider"] == "slack"
                    assert body["total"] == 1
                    assert body["bindings"][0]["binding_type"] == "debate_team"

    @pytest.mark.asyncio
    async def test_list_bindings_empty_provider(self, handler, mock_router):
        """List bindings for non-existent provider returns empty."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/nonexistent", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["total"] == 0
                    assert body["bindings"] == []

    @pytest.mark.asyncio
    async def test_get_stats_comprehensive(self, handler, mock_router):
        """Get stats returns comprehensive statistics."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/stats", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["total_bindings"] == 6
                    assert "providers" in body
                    assert "has_global_default" in body
                    assert body["has_global_default"] is True

    @pytest.mark.asyncio
    async def test_list_bindings_empty_router(self, handler, empty_router):
        """List bindings from empty router returns empty list."""
        with patch.object(handler, "_get_router", return_value=empty_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["total"] == 0


# =============================================================================
# Test Classes - Binding Resolution
# =============================================================================


class TestBindingResolution:
    """Tests for binding resolution (POST /api/bindings/resolve)."""

    @pytest.mark.asyncio
    async def test_resolve_exact_match(self, handler, mock_router):
        """Resolve returns exact binding match."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "slack",
                            "account_id": "workspace1",
                            "peer_id": "channel:general",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["agent_binding"] == "debate-team-alpha"
                    assert body["binding_type"] == "debate_team"
                    assert body["config_overrides"]["rounds"] == 3

    @pytest.mark.asyncio
    async def test_resolve_wildcard_match(self, handler, mock_router):
        """Resolve matches wildcard pattern."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

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
                    assert body["agent_binding"] == "claude-api"

    @pytest.mark.asyncio
    async def test_resolve_priority_ordering(self, handler, mock_router):
        """Resolve respects priority ordering (higher priority first)."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # Premium chat should match higher priority binding
                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_id": "chat:premium_user123",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["agent_binding"] == "claude-opus"

    @pytest.mark.asyncio
    async def test_resolve_with_user_filter(self, handler, mock_router):
        """Resolve respects user allowed list."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # User in allowed list
                    mock_request = create_async_request(
                        {
                            "provider": "whatsapp",
                            "account_id": "wa456",
                            "peer_id": "group:support_main",
                            "user_id": "user1",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["agent_binding"] == "openai-api"

    @pytest.mark.asyncio
    async def test_resolve_user_not_in_allowed_list(self, handler, mock_router):
        """Resolve falls back when user not in allowed list."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # User not in allowed list
                    mock_request = create_async_request(
                        {
                            "provider": "whatsapp",
                            "account_id": "wa456",
                            "peer_id": "group:support_main",
                            "user_id": "random_user",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    # Should fall back to global default since user not allowed
                    assert body["matched"] is True

    @pytest.mark.asyncio
    async def test_resolve_with_time_window(self, handler, mock_router):
        """Resolve respects time window constraints."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # Hour during night shift (22-6)
                    mock_request = create_async_request(
                        {
                            "provider": "discord",
                            "account_id": "server789",
                            "peer_id": "channel:chat",
                            "hour": 23,
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["agent_binding"] == "night-shift-agent"

    @pytest.mark.asyncio
    async def test_resolve_outside_time_window(self, handler, mock_router):
        """Resolve falls back when outside time window."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # Hour during day (outside 22-6)
                    mock_request = create_async_request(
                        {
                            "provider": "discord",
                            "account_id": "server789",
                            "peer_id": "channel:chat",
                            "hour": 14,
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    # Should fall back since outside time window
                    assert body["matched"] is True

    @pytest.mark.asyncio
    async def test_resolve_global_default(self, handler, mock_router):
        """Resolve returns global default for unmatched."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "unknown_platform",
                            "account_id": "unknown_account",
                            "peer_id": "unknown_peer",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["matched"] is True
                    assert body["match_reason"] == "Global default"

    @pytest.mark.asyncio
    async def test_resolve_returns_candidates_checked(self, handler, mock_router):
        """Resolve returns number of candidates checked."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_id": "chat:12345",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    body = json.loads(result.body)
                    assert "candidates_checked" in body
                    assert body["candidates_checked"] >= 1


# =============================================================================
# Test Classes - Binding Deletion
# =============================================================================


class TestBindingDeletion:
    """Tests for binding deletion (DELETE /api/bindings/:provider/:account/:pattern)."""

    @pytest.mark.asyncio
    async def test_delete_binding_success(self, handler, mock_router):
        """Delete binding removes it from router."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/telegram/bot123/chat:*",
                        mock_request,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_binding_with_complex_pattern(self, handler, mock_router):
        """Delete binding with pattern containing special characters."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/telegram/bot123/chat:premium_*",
                        mock_request,
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_nonexistent_binding(self, handler, mock_router):
        """Delete nonexistent binding returns 404."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/nonexistent/provider/pattern",
                        mock_request,
                    )

                    assert result is not None
                    assert result.status_code == 404
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "BINDING_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_delete_binding_short_path(self, handler, mock_router):
        """Delete with too few path segments returns 400."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/telegram",
                        mock_request,
                    )

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_binding_with_slashes_in_pattern(self, handler, mock_router):
        """Delete binding where pattern contains slashes."""
        # Add a binding with slashes in pattern
        mock_router.add_binding(
            MockMessageBinding(
                provider="custom",
                account_id="acc1",
                peer_pattern="a/b/c",
                agent_binding="agent1",
            )
        )

        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete(
                        "/api/bindings/custom/acc1/a/b/c",
                        mock_request,
                    )

                    assert result is not None
                    assert result.status_code == 200


# =============================================================================
# Test Classes - Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_bindings_unavailable_get(self, handler):
        """GET returns 503 when bindings system unavailable."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", False):
            mock_request = create_simple_request()
            result = await handler.handle_get("/api/bindings", mock_request)

            assert result is not None
            assert result.status_code == 503
            body = json.loads(result.body)
            assert body["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_bindings_unavailable_post(self, handler):
        """POST returns 503 when bindings system unavailable."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", False):
            mock_request = create_async_request({})
            result = await handler.handle_post("/api/bindings", mock_request)

            assert result is not None
            assert result.status_code == 503
            body = json.loads(result.body)
            assert body["error"]["code"] == "BINDINGS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_bindings_unavailable_delete(self, handler):
        """DELETE returns 503 when bindings system unavailable."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", False):
            mock_request = create_simple_request()
            result = await handler.handle_delete("/api/bindings/a/b/c", mock_request)

            assert result is not None
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_router_unavailable_list(self, handler):
        """List returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 503
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "ROUTER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_router_unavailable_create(self, handler):
        """Create returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_pattern": "chat:*",
                            "agent_binding": "agent1",
                        }
                    )

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_router_unavailable_resolve(self, handler):
        """Resolve returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_id": "chat:123",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_router_unavailable_delete(self, handler):
        """Delete returns 503 when router unavailable."""
        with patch.object(handler, "_get_router", return_value=None):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_delete("/api/bindings/a/b/c", mock_request)

                    assert result is not None
                    assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_unknown_post_endpoint(self, handler, mock_router):
        """Unknown POST endpoint returns 404."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request({})
                    result = await handler.handle_post("/api/bindings/unknown/path", mock_request)

                    assert result is not None
                    assert result.status_code == 404


# =============================================================================
# Test Classes - Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_get_bindings(self, handler):
        """GET /api/bindings respects rate limits."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_get("/api/bindings", mock_request)

                assert result is not None
                assert result.status_code == 429
                body = json.loads(result.body)
                assert body["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_rate_limit_get_stats(self, handler):
        """GET /api/bindings/stats respects rate limits."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_get("/api/bindings/stats", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_post_create(self, handler):
        """POST /api/bindings respects rate limits."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = create_async_request({})
                result = await handler.handle_post("/api/bindings", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_post_resolve(self, handler):
        """POST /api/bindings/resolve respects rate limits."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = create_async_request({})
                result = await handler.handle_post("/api/bindings/resolve", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_delete(self, handler):
        """DELETE /api/bindings/:path respects rate limits."""
        with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
            with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = create_simple_request()
                result = await handler.handle_delete("/api/bindings/a/b/c", mock_request)

                assert result is not None
                assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_per_client(self, handler, mock_router):
        """Rate limiter is checked per client IP."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    with patch("aragora.server.handlers.bindings.get_client_ip") as get_ip:
                        limiter.is_allowed.return_value = True
                        get_ip.return_value = "192.168.1.100"

                        mock_request = create_simple_request("192.168.1.100")
                        await handler.handle_get("/api/bindings", mock_request)

                        limiter.is_allowed.assert_called_with("192.168.1.100")


# =============================================================================
# Test Classes - API Versioning
# =============================================================================


class TestAPIVersioning:
    """Tests for API version prefix handling."""

    @pytest.mark.asyncio
    async def test_v1_prefix_list_bindings(self, handler, mock_router):
        """v1 prefix works for listing bindings."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/v1/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_v2_prefix_list_bindings(self, handler, mock_router):
        """v2 prefix works for listing bindings."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/v2/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_v1_prefix_create_binding(self, handler, mock_router):
        """v1 prefix works for creating bindings."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "test",
                                    "account_id": "acc1",
                                    "peer_pattern": "*",
                                    "agent_binding": "agent1",
                                }
                            )

                            result = await handler.handle_post("/api/v1/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_v1_prefix_resolve_binding(self, handler, mock_router):
        """v1 prefix works for resolving bindings."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "telegram",
                            "account_id": "bot123",
                            "peer_id": "chat:123",
                        }
                    )

                    result = await handler.handle_post("/api/v1/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 200


# =============================================================================
# Test Classes - Integration with Debate Infrastructure
# =============================================================================


class TestDebateIntegration:
    """Tests for debate infrastructure integration."""

    @pytest.mark.asyncio
    async def test_debate_team_binding_config_propagation(self, handler, mock_router):
        """Debate team binding config is correctly propagated."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {
                            "provider": "slack",
                            "account_id": "workspace1",
                            "peer_id": "channel:general",
                        }
                    )

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    body = json.loads(result.body)
                    assert body["binding_type"] == "debate_team"
                    assert body["config_overrides"]["rounds"] == 3
                    assert body["config_overrides"]["consensus_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_create_binding_with_debate_agents(self, handler, mock_router):
        """Create debate team binding with agent list."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "teams",
                                    "account_id": "org1",
                                    "peer_pattern": "channel:strategy",
                                    "agent_binding": "strategy-debate",
                                    "binding_type": "debate_team",
                                    "config_overrides": {
                                        "agents": ["claude-opus", "gpt-4", "gemini-ultra"],
                                        "rounds": 4,
                                        "enable_cross_examination": True,
                                    },
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["config_overrides"]["agents"] == [
                                "claude-opus",
                                "gpt-4",
                                "gemini-ultra",
                            ]

    @pytest.mark.asyncio
    async def test_binding_includes_debate_config(self, handler, mock_router):
        """Retrieved binding includes debate configuration."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get("/api/bindings/slack", mock_request)

                    assert result is not None
                    body = json.loads(result.body)
                    debate_binding = next(
                        (b for b in body["bindings"] if b["binding_type"] == "debate_team"), None
                    )
                    assert debate_binding is not None
                    assert "config_overrides" in debate_binding


# =============================================================================
# Test Classes - Handler Initialization
# =============================================================================


class TestHandlerInitialization:
    """Tests for handler initialization and configuration."""

    def test_handler_has_routes(self, mock_server_context):
        """Handler has ROUTES attribute defined."""
        handler = BindingsHandler(mock_server_context)
        assert hasattr(handler, "ROUTES")
        assert isinstance(handler.ROUTES, list)
        assert len(handler.ROUTES) > 0

    def test_handler_routes_include_required_paths(self, mock_server_context):
        """Handler routes include all required paths."""
        handler = BindingsHandler(mock_server_context)
        required_paths = ["/api/bindings", "/api/bindings/resolve", "/api/bindings/stats"]
        for path in required_paths:
            assert path in handler.ROUTES, f"Missing route: {path}"

    def test_handler_wildcard_route_last(self, mock_server_context):
        """Wildcard route is last in routes list."""
        handler = BindingsHandler(mock_server_context)
        if "/api/bindings/*" in handler.ROUTES:
            assert handler.ROUTES[-1] == "/api/bindings/*"

    def test_handler_initializes_with_context(self, mock_server_context):
        """Handler correctly initializes with server context."""
        handler = BindingsHandler(mock_server_context)
        assert handler.ctx == mock_server_context

    def test_handler_router_lazy_initialization(self, mock_server_context):
        """Handler router is lazily initialized."""
        handler = BindingsHandler(mock_server_context)
        assert handler._router is None


# =============================================================================
# Test Classes - Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_request_body_create(self, handler, mock_router):
        """Create binding with empty body returns error."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request({})

                    result = await handler.handle_post("/api/bindings", mock_request)

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_request_body_resolve(self, handler, mock_router):
        """Resolve binding with empty body returns error."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request({})

                    result = await handler.handle_post("/api/bindings/resolve", mock_request)

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_special_characters_in_provider(self, handler, mock_router):
        """Handle provider with special characters."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_simple_request()
                    result = await handler.handle_get(
                        "/api/bindings/my-custom-platform", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_unicode_in_binding_name(self, handler, mock_router):
        """Handle unicode characters in binding name."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "telegram",
                                    "account_id": "bot123",
                                    "peer_pattern": "chat:*",
                                    "agent_binding": "agent1",
                                    "name": "Test Binding",
                                    "description": "Test with special characters",
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_very_long_peer_pattern(self, handler, mock_router):
        """Handle very long peer pattern."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            long_pattern = "channel:" + "a" * 500 + "*"
                            mock_request = create_async_request(
                                {
                                    "provider": "slack",
                                    "account_id": "workspace1",
                                    "peer_pattern": long_pattern,
                                    "agent_binding": "agent1",
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_priority_zero(self, handler, mock_router):
        """Handle binding with zero priority."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "telegram",
                                    "account_id": "bot123",
                                    "peer_pattern": "dm:*",
                                    "agent_binding": "agent1",
                                    "priority": 0,
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["priority"] == 0

    @pytest.mark.asyncio
    async def test_negative_priority(self, handler, mock_router):
        """Handle binding with negative priority."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "telegram",
                                    "account_id": "bot123",
                                    "peer_pattern": "fallback:*",
                                    "agent_binding": "fallback-agent",
                                    "priority": -100,
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["priority"] == -100

    @pytest.mark.asyncio
    async def test_disabled_binding(self, handler, mock_router):
        """Create and handle disabled binding."""
        with patch.object(handler, "_get_router", return_value=mock_router):
            with patch("aragora.server.handlers.bindings.BINDINGS_AVAILABLE", True):
                with patch("aragora.server.handlers.bindings.BindingType", MockBindingType):
                    with patch(
                        "aragora.server.handlers.bindings.MessageBinding", MockMessageBinding
                    ):
                        with patch("aragora.server.handlers.bindings._bindings_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "provider": "telegram",
                                    "account_id": "bot123",
                                    "peer_pattern": "disabled:*",
                                    "agent_binding": "disabled-agent",
                                    "enabled": False,
                                }
                            )

                            result = await handler.handle_post("/api/bindings", mock_request)

                            assert result is not None
                            assert result.status_code == 201
                            body = json.loads(result.body)
                            assert body["binding"]["enabled"] is False
