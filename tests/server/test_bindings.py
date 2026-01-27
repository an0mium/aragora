"""
Tests for aragora.server.bindings module.

Covers:
- BindingType enum
- MessageBinding dataclass
- BindingResolution dataclass
- AgentSelection dataclass
- BindingRouter class
- Global router singleton
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import pytest

from aragora.server.bindings import (
    AgentSelection,
    BindingResolution,
    BindingRouter,
    BindingType,
    MessageBinding,
    get_binding_router,
    reset_binding_router,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def router() -> BindingRouter:
    """Create a fresh router for testing."""
    return BindingRouter()


@pytest.fixture
def binding() -> MessageBinding:
    """Create a basic binding for testing."""
    return MessageBinding(
        provider="slack",
        account_id="T12345",
        peer_pattern="channel:*",
        agent_binding="default",
    )


@pytest.fixture
def specific_binding() -> MessageBinding:
    """Create a specific agent binding for testing."""
    return MessageBinding(
        provider="slack",
        account_id="T12345",
        peer_pattern="dm:U67890",
        agent_binding="claude-opus",
        binding_type=BindingType.SPECIFIC_AGENT,
        priority=20,
    )


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str


# =============================================================================
# BindingType Tests
# =============================================================================


class TestBindingType:
    """Tests for BindingType enum."""

    def test_binding_type_values(self):
        """Test binding type values."""
        assert BindingType.DEFAULT.value == "default"
        assert BindingType.SPECIFIC_AGENT.value == "specific_agent"
        assert BindingType.AGENT_POOL.value == "agent_pool"
        assert BindingType.DEBATE_TEAM.value == "debate_team"
        assert BindingType.CUSTOM.value == "custom"

    def test_binding_type_is_string_enum(self):
        """Test binding type is a string enum."""
        for bt in BindingType:
            assert isinstance(bt, str)


# =============================================================================
# MessageBinding Tests
# =============================================================================


class TestMessageBinding:
    """Tests for MessageBinding dataclass."""

    def test_create_binding(self, binding: MessageBinding):
        """Test creating a binding."""
        assert binding.provider == "slack"
        assert binding.account_id == "T12345"
        assert binding.peer_pattern == "channel:*"
        assert binding.agent_binding == "default"

    def test_binding_defaults(self, binding: MessageBinding):
        """Test binding default values."""
        assert binding.binding_type == BindingType.DEFAULT
        assert binding.priority == 0
        assert binding.time_window_start is None
        assert binding.time_window_end is None
        assert binding.allowed_users is None
        assert binding.blocked_users is None
        assert binding.config_overrides == {}
        assert binding.name is None
        assert binding.enabled is True
        assert binding.created_at is not None

    def test_matches_peer_exact(self):
        """Test exact peer pattern matching."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="channel:general",
            agent_binding="default",
        )

        assert binding.matches_peer("channel:general") is True
        assert binding.matches_peer("channel:random") is False

    def test_matches_peer_wildcard(self):
        """Test wildcard peer pattern matching."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="channel:*",
            agent_binding="default",
        )

        assert binding.matches_peer("channel:general") is True
        assert binding.matches_peer("channel:random") is True
        assert binding.matches_peer("dm:U12345") is False

    def test_matches_peer_star_matches_all(self):
        """Test star pattern matches everything."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
        )

        assert binding.matches_peer("channel:general") is True
        assert binding.matches_peer("dm:U12345") is True
        assert binding.matches_peer("thread:T12345") is True

    def test_matches_time_no_window(self):
        """Test time matching with no window set."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
        )

        assert binding.matches_time(0) is True
        assert binding.matches_time(12) is True
        assert binding.matches_time(23) is True

    def test_matches_time_normal_window(self):
        """Test time matching with normal window (e.g., 9-17)."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            time_window_start=9,
            time_window_end=17,
        )

        assert binding.matches_time(9) is True
        assert binding.matches_time(12) is True
        assert binding.matches_time(16) is True
        assert binding.matches_time(17) is False
        assert binding.matches_time(8) is False
        assert binding.matches_time(0) is False

    def test_matches_time_wrapping_window(self):
        """Test time matching with wrapping window (e.g., 22-6)."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            time_window_start=22,
            time_window_end=6,
        )

        assert binding.matches_time(22) is True
        assert binding.matches_time(23) is True
        assert binding.matches_time(0) is True
        assert binding.matches_time(5) is True
        assert binding.matches_time(6) is False
        assert binding.matches_time(12) is False

    def test_matches_user_no_restrictions(self):
        """Test user matching with no restrictions."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
        )

        assert binding.matches_user("U12345") is True
        assert binding.matches_user(None) is True

    def test_matches_user_allowed_list(self):
        """Test user matching with allowed list."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            allowed_users={"U12345", "U67890"},
        )

        assert binding.matches_user("U12345") is True
        assert binding.matches_user("U67890") is True
        assert binding.matches_user("UOTHER") is False
        assert binding.matches_user(None) is True  # None bypasses check

    def test_matches_user_blocked_list(self):
        """Test user matching with blocked list."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            blocked_users={"UBANNED"},
        )

        assert binding.matches_user("U12345") is True
        assert binding.matches_user("UBANNED") is False

    def test_matches_user_blocked_takes_priority(self):
        """Test blocked users take priority over allowed."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            allowed_users={"U12345", "UBANNED"},
            blocked_users={"UBANNED"},
        )

        assert binding.matches_user("U12345") is True
        assert binding.matches_user("UBANNED") is False

    def test_to_dict(self, binding: MessageBinding):
        """Test serialization to dict."""
        data = binding.to_dict()

        assert data["provider"] == "slack"
        assert data["account_id"] == "T12345"
        assert data["peer_pattern"] == "channel:*"
        assert data["agent_binding"] == "default"
        assert data["binding_type"] == "default"
        assert data["priority"] == 0
        assert data["enabled"] is True

    def test_to_dict_with_sets(self):
        """Test serialization includes user sets as lists."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            allowed_users={"U1", "U2"},
            blocked_users={"U3"},
        )

        data = binding.to_dict()

        assert sorted(data["allowed_users"]) == ["U1", "U2"]
        assert data["blocked_users"] == ["U3"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "provider": "telegram",
            "account_id": "123456",
            "peer_pattern": "chat:*",
            "agent_binding": "gpt-4",
            "binding_type": "specific_agent",
            "priority": 10,
            "enabled": True,
        }

        binding = MessageBinding.from_dict(data)

        assert binding.provider == "telegram"
        assert binding.account_id == "123456"
        assert binding.peer_pattern == "chat:*"
        assert binding.agent_binding == "gpt-4"
        assert binding.binding_type == BindingType.SPECIFIC_AGENT
        assert binding.priority == 10

    def test_from_dict_with_user_sets(self):
        """Test deserialization includes user sets."""
        data = {
            "provider": "slack",
            "account_id": "T12345",
            "peer_pattern": "*",
            "agent_binding": "default",
            "allowed_users": ["U1", "U2"],
            "blocked_users": ["U3"],
        }

        binding = MessageBinding.from_dict(data)

        assert binding.allowed_users == {"U1", "U2"}
        assert binding.blocked_users == {"U3"}

    def test_round_trip_serialization(self, binding: MessageBinding):
        """Test round trip serialization."""
        data = binding.to_dict()
        restored = MessageBinding.from_dict(data)

        assert restored.provider == binding.provider
        assert restored.account_id == binding.account_id
        assert restored.peer_pattern == binding.peer_pattern
        assert restored.agent_binding == binding.agent_binding


# =============================================================================
# BindingResolution Tests
# =============================================================================


class TestBindingResolution:
    """Tests for BindingResolution dataclass."""

    def test_create_matched_resolution(self, binding: MessageBinding):
        """Test creating a matched resolution."""
        resolution = BindingResolution(
            matched=True,
            binding=binding,
            agent_binding="default",
            binding_type=BindingType.DEFAULT,
            match_reason="Matched pattern",
        )

        assert resolution.matched is True
        assert resolution.binding == binding
        assert resolution.agent_binding == "default"

    def test_create_unmatched_resolution(self):
        """Test creating an unmatched resolution."""
        resolution = BindingResolution(
            matched=False,
            candidates_checked=5,
        )

        assert resolution.matched is False
        assert resolution.binding is None
        assert resolution.candidates_checked == 5

    def test_resolution_defaults(self):
        """Test resolution default values."""
        resolution = BindingResolution(matched=False)

        assert resolution.binding is None
        assert resolution.agent_binding is None
        assert resolution.binding_type == BindingType.DEFAULT
        assert resolution.config_overrides == {}
        assert resolution.match_reason is None
        assert resolution.candidates_checked == 0


# =============================================================================
# AgentSelection Tests
# =============================================================================


class TestAgentSelection:
    """Tests for AgentSelection dataclass."""

    def test_create_selection(self, binding: MessageBinding):
        """Test creating an agent selection."""
        selection = AgentSelection(
            agent_name="claude-opus",
            binding=binding,
            config={"temperature": 0.7},
            selection_reason="Specific agent binding",
        )

        assert selection.agent_name == "claude-opus"
        assert selection.binding == binding
        assert selection.config == {"temperature": 0.7}
        assert selection.selection_reason == "Specific agent binding"

    def test_selection_defaults(self, binding: MessageBinding):
        """Test selection default values."""
        selection = AgentSelection(
            agent_name="default",
            binding=binding,
        )

        assert selection.config == {}
        assert selection.selection_reason == "default"


# =============================================================================
# BindingRouter Tests - Basic Operations
# =============================================================================


class TestBindingRouterBasic:
    """Tests for basic router operations."""

    def test_add_binding(self, router: BindingRouter, binding: MessageBinding):
        """Test adding a binding."""
        router.add_binding(binding)

        bindings = router.list_bindings()
        assert len(bindings) == 1
        assert bindings[0] == binding

    def test_add_multiple_bindings(self, router: BindingRouter):
        """Test adding multiple bindings."""
        b1 = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="channel:*",
            agent_binding="default",
        )
        b2 = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="dm:*",
            agent_binding="claude-opus",
        )

        router.add_binding(b1)
        router.add_binding(b2)

        bindings = router.list_bindings()
        assert len(bindings) == 2

    def test_remove_binding(self, router: BindingRouter, binding: MessageBinding):
        """Test removing a binding."""
        router.add_binding(binding)
        result = router.remove_binding("slack", "T12345", "channel:*")

        assert result is True
        assert len(router.list_bindings()) == 0

    def test_remove_nonexistent_binding(self, router: BindingRouter):
        """Test removing nonexistent binding."""
        result = router.remove_binding("slack", "T12345", "nonexistent")
        assert result is False

    def test_remove_binding_wrong_provider(self, router: BindingRouter, binding: MessageBinding):
        """Test removing binding from wrong provider."""
        router.add_binding(binding)
        result = router.remove_binding("telegram", "T12345", "channel:*")

        assert result is False
        assert len(router.list_bindings()) == 1


# =============================================================================
# BindingRouter Tests - Priority
# =============================================================================


class TestBindingRouterPriority:
    """Tests for priority-based binding resolution."""

    def test_bindings_sorted_by_priority(self, router: BindingRouter):
        """Test bindings are sorted by priority (descending)."""
        low = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="low",
            priority=0,
        )
        high = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="high",
            priority=10,
        )

        router.add_binding(low)
        router.add_binding(high)

        bindings = router.list_bindings()
        assert bindings[0].agent_binding == "high"
        assert bindings[1].agent_binding == "low"

    def test_higher_priority_matches_first(self, router: BindingRouter):
        """Test higher priority binding matches first."""
        general = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="general",
            priority=0,
        )
        specific = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="specific",
            priority=20,
        )

        router.add_binding(general)
        router.add_binding(specific)

        resolution = router.resolve("slack", "T12345", "channel:general")

        assert resolution.agent_binding == "specific"


# =============================================================================
# BindingRouter Tests - Resolution
# =============================================================================


class TestBindingRouterResolution:
    """Tests for binding resolution."""

    def test_resolve_exact_match(self, router: BindingRouter):
        """Test resolving exact pattern match."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="channel:general",
            agent_binding="general-agent",
        )
        router.add_binding(binding)

        resolution = router.resolve("slack", "T12345", "channel:general")

        assert resolution.matched is True
        assert resolution.agent_binding == "general-agent"
        assert "Matched pattern" in resolution.match_reason

    def test_resolve_wildcard_match(self, router: BindingRouter):
        """Test resolving wildcard pattern match."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="channel:*",
            agent_binding="channel-agent",
        )
        router.add_binding(binding)

        resolution = router.resolve("slack", "T12345", "channel:random")

        assert resolution.matched is True
        assert resolution.agent_binding == "channel-agent"

    def test_resolve_wildcard_account(self, router: BindingRouter):
        """Test resolving with wildcard account binding."""
        binding = MessageBinding(
            provider="slack",
            account_id="*",
            peer_pattern="dm:*",
            agent_binding="dm-agent",
        )
        router.add_binding(binding)

        resolution = router.resolve("slack", "T99999", "dm:U12345")

        assert resolution.matched is True
        assert resolution.agent_binding == "dm-agent"
        assert "wildcard account" in resolution.match_reason

    def test_resolve_disabled_binding_skipped(self, router: BindingRouter):
        """Test disabled binding is skipped."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="disabled-agent",
            enabled=False,
        )
        router.add_binding(binding)

        resolution = router.resolve("slack", "T12345", "channel:test")

        # Should fall through to global default
        assert resolution.agent_binding == "default"

    def test_resolve_time_filter(self, router: BindingRouter):
        """Test binding with time filter."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="business-hours",
            time_window_start=9,
            time_window_end=17,
            priority=10,
        )
        router.add_binding(binding)

        # Within business hours
        resolution = router.resolve("slack", "T12345", "channel:test", hour=12)
        assert resolution.agent_binding == "business-hours"

        # Outside business hours
        resolution = router.resolve("slack", "T12345", "channel:test", hour=20)
        assert resolution.agent_binding == "default"  # Falls to global default

    def test_resolve_user_filter(self, router: BindingRouter):
        """Test binding with user filter."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="vip-agent",
            allowed_users={"UVIP"},
            priority=10,
        )
        router.add_binding(binding)

        # VIP user
        resolution = router.resolve("slack", "T12345", "channel:test", user_id="UVIP")
        assert resolution.agent_binding == "vip-agent"

        # Non-VIP user
        resolution = router.resolve("slack", "T12345", "channel:test", user_id="UNORMAL")
        assert resolution.agent_binding == "default"


# =============================================================================
# BindingRouter Tests - Defaults
# =============================================================================


class TestBindingRouterDefaults:
    """Tests for default binding behavior."""

    def test_global_default_always_matches(self, router: BindingRouter):
        """Test global default is returned when no bindings match."""
        resolution = router.resolve("unknown", "X123", "peer:123")

        assert resolution.matched is True
        assert resolution.agent_binding == "default"
        assert "Global default" in resolution.match_reason

    def test_set_provider_default(self, router: BindingRouter):
        """Test setting provider default."""
        default = MessageBinding(
            provider="telegram",
            account_id="*",
            peer_pattern="*",
            agent_binding="telegram-default",
        )
        router.set_default_binding("telegram", default)

        resolution = router.resolve("telegram", "12345", "chat:xyz")

        assert resolution.agent_binding == "telegram-default"
        assert "Provider default" in resolution.match_reason

    def test_set_global_default(self, router: BindingRouter):
        """Test setting global default."""
        custom_default = MessageBinding(
            provider="*",
            account_id="*",
            peer_pattern="*",
            agent_binding="custom-default",
        )
        router.set_global_default(custom_default)

        resolution = router.resolve("unknown", "X123", "peer:123")

        assert resolution.agent_binding == "custom-default"


# =============================================================================
# BindingRouter Tests - Agent Selection
# =============================================================================


class TestBindingRouterAgentSelection:
    """Tests for agent selection."""

    def test_select_specific_agent(self, router: BindingRouter):
        """Test selecting a specific agent."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="claude-opus",
            binding_type=BindingType.SPECIFIC_AGENT,
        )
        router.add_binding(binding)

        agents = [MockAgent("gpt-4"), MockAgent("claude-opus")]
        selection = router.get_agent_for_message("slack", "T12345", "channel:test", agents)

        assert selection.agent_name == "claude-opus"
        assert "Specific agent" in selection.selection_reason

    def test_select_from_pool(self, router: BindingRouter):
        """Test selecting from agent pool."""
        router.register_agent_pool("fast-agents", ["gpt-4", "claude-sonnet"])

        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="fast-agents",
            binding_type=BindingType.AGENT_POOL,
        )
        router.add_binding(binding)

        agents = [MockAgent("claude-opus"), MockAgent("gpt-4")]
        selection = router.get_agent_for_message("slack", "T12345", "channel:test", agents)

        assert selection.agent_name == "gpt-4"  # First available from pool
        assert "pool" in selection.selection_reason

    def test_select_debate_team(self, router: BindingRouter):
        """Test selecting for debate team."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="research-team",
            binding_type=BindingType.DEBATE_TEAM,
        )
        router.add_binding(binding)

        agents = [MockAgent("claude-opus")]
        selection = router.get_agent_for_message("slack", "T12345", "channel:test", agents)

        assert selection.config["team_config"] == "research-team"
        assert "Debate team" in selection.selection_reason

    def test_fallback_when_agent_unavailable(self, router: BindingRouter):
        """Test fallback when bound agent is unavailable."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="unavailable-agent",
            binding_type=BindingType.SPECIFIC_AGENT,
        )
        router.add_binding(binding)

        agents = [MockAgent("gpt-4")]
        selection = router.get_agent_for_message("slack", "T12345", "channel:test", agents)

        # Should fall back to first available
        assert selection.agent_name == "gpt-4"

    def test_no_agents_available_raises(self, router: BindingRouter):
        """Test error when no agents available."""
        with pytest.raises(ValueError, match="No agents available"):
            router.get_agent_for_message("slack", "T12345", "channel:test", [])

    def test_config_overrides_passed(self, router: BindingRouter):
        """Test config overrides are passed to selection."""
        binding = MessageBinding(
            provider="slack",
            account_id="T12345",
            peer_pattern="*",
            agent_binding="default",
            config_overrides={"temperature": 0.5, "max_tokens": 1000},
        )
        router.add_binding(binding)

        agents = [MockAgent("claude-opus")]
        selection = router.get_agent_for_message("slack", "T12345", "channel:test", agents)

        assert selection.config["temperature"] == 0.5
        assert selection.config["max_tokens"] == 1000


# =============================================================================
# BindingRouter Tests - Listing and Stats
# =============================================================================


class TestBindingRouterListingStats:
    """Tests for listing bindings and statistics."""

    def test_list_bindings_all(self, router: BindingRouter):
        """Test listing all bindings."""
        b1 = MessageBinding(provider="slack", account_id="T1", peer_pattern="*", agent_binding="a1")
        b2 = MessageBinding(
            provider="telegram", account_id="T2", peer_pattern="*", agent_binding="a2"
        )
        router.add_binding(b1)
        router.add_binding(b2)

        bindings = router.list_bindings()
        assert len(bindings) == 2

    def test_list_bindings_filter_provider(self, router: BindingRouter):
        """Test listing bindings filtered by provider."""
        b1 = MessageBinding(provider="slack", account_id="T1", peer_pattern="*", agent_binding="a1")
        b2 = MessageBinding(
            provider="telegram", account_id="T2", peer_pattern="*", agent_binding="a2"
        )
        router.add_binding(b1)
        router.add_binding(b2)

        bindings = router.list_bindings(provider="slack")
        assert len(bindings) == 1
        assert bindings[0].provider == "slack"

    def test_list_bindings_filter_account(self, router: BindingRouter):
        """Test listing bindings filtered by account."""
        b1 = MessageBinding(provider="slack", account_id="T1", peer_pattern="*", agent_binding="a1")
        b2 = MessageBinding(provider="slack", account_id="T2", peer_pattern="*", agent_binding="a2")
        router.add_binding(b1)
        router.add_binding(b2)

        bindings = router.list_bindings(provider="slack", account_id="T1")
        assert len(bindings) == 1
        assert bindings[0].account_id == "T1"

    def test_get_stats(self, router: BindingRouter):
        """Test getting router statistics."""
        router.add_binding(
            MessageBinding(provider="slack", account_id="T1", peer_pattern="*", agent_binding="a1")
        )
        router.register_agent_pool("pool1", ["a", "b"])

        stats = router.get_stats()

        assert stats["total_bindings"] == 1
        assert "slack" in stats["providers"]
        assert "pool1" in stats["agent_pools"]
        assert stats["has_global_default"] is True


# =============================================================================
# Global Singleton Tests
# =============================================================================


class TestGlobalRouter:
    """Tests for global router singleton."""

    def test_get_binding_router_returns_singleton(self):
        """Test get_binding_router returns same instance."""
        reset_binding_router()

        r1 = get_binding_router()
        r2 = get_binding_router()

        assert r1 is r2

    def test_reset_binding_router(self):
        """Test reset_binding_router creates new instance."""
        r1 = get_binding_router()
        reset_binding_router()
        r2 = get_binding_router()

        assert r1 is not r2

    def test_global_router_is_independent(self):
        """Test global router operations don't affect fixtures."""
        reset_binding_router()
        router = get_binding_router()

        router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="global",
                peer_pattern="*",
                agent_binding="global-agent",
            )
        )

        # Verify it was added
        stats = router.get_stats()
        assert stats["total_bindings"] >= 1

        # Reset to clean up
        reset_binding_router()
