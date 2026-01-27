"""
End-to-End Integration Tests: Bindings Router ↔ Social Connectors.

Tests the integration between the Binding Router and chat platform connectors:
1. Message routing to agents via bindings
2. Binding resolution for different platforms
3. Time-window and user-based filtering
4. Agent pool selection
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.bindings import (
    BindingRouter,
    MessageBinding,
    BindingType,
    BindingResolution,
    reset_binding_router,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def binding_router():
    """Create a fresh binding router for each test."""
    reset_binding_router()
    router = BindingRouter()
    yield router


@pytest.fixture
def mock_telegram_message():
    """Create a mock Telegram message."""
    return {
        "platform": "telegram",
        "chat_id": "group:-1001234567890",
        "user_id": "user_12345",
        "text": "What are the pros and cons of remote work?",
        "timestamp": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_slack_message():
    """Create a mock Slack message."""
    return {
        "platform": "slack",
        "team_id": "T12345678",
        "channel_id": "C87654321",
        "user_id": "U11223344",
        "text": "Help me understand this quarterly report",
        "timestamp": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_agents():
    """Create mock agent objects."""
    agents = []
    for name in ["claude-opus", "claude-sonnet", "gpt-4o", "gemini-pro"]:
        agent = MagicMock()
        agent.name = name
        agent.available = True
        agents.append(agent)
    return agents


# ============================================================================
# Integration Tests
# ============================================================================


class TestBindingsMessageRouting:
    """Tests for message routing via bindings."""

    def test_basic_binding_resolution(self, binding_router, mock_slack_message):
        """Test basic binding resolution for Slack message."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="claude-sonnet",
                binding_type=BindingType.SPECIFIC_AGENT,
                priority=10,
            )
        )

        resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C87654321",
        )

        assert resolution.matched
        assert resolution.agent_binding == "claude-sonnet"
        assert resolution.binding_type == BindingType.SPECIFIC_AGENT

    def test_telegram_group_binding(self, binding_router, mock_telegram_message):
        """Test binding resolution for Telegram groups."""
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@mybot",
                peer_pattern="group:*",
                agent_binding="debate-team-alpha",
                binding_type=BindingType.DEBATE_TEAM,
                priority=20,
            )
        )

        resolution = binding_router.resolve(
            provider="telegram",
            account_id="@mybot",
            peer_id="group:-1001234567890",
        )

        assert resolution.matched
        assert resolution.agent_binding == "debate-team-alpha"
        assert resolution.binding_type == BindingType.DEBATE_TEAM

    def test_priority_based_resolution(self, binding_router):
        """Test that higher priority bindings match first."""
        # Lower priority: general channel binding
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="default-agent",
                binding_type=BindingType.DEFAULT,
                priority=10,
            )
        )

        # Higher priority: VIP channel binding
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:C-VIP*",
                agent_binding="claude-opus",
                binding_type=BindingType.SPECIFIC_AGENT,
                priority=100,
            )
        )

        # VIP channel should match high priority binding
        vip_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C-VIP-support",
        )
        assert vip_resolution.agent_binding == "claude-opus"

        # Regular channel should match lower priority binding
        regular_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C-general",
        )
        assert regular_resolution.agent_binding == "default-agent"

    def test_dm_binding(self, binding_router):
        """Test binding for direct messages."""
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@mybot",
                peer_pattern="dm:*",
                agent_binding="claude-sonnet",
                binding_type=BindingType.SPECIFIC_AGENT,
                priority=15,
            )
        )

        resolution = binding_router.resolve(
            provider="telegram",
            account_id="@mybot",
            peer_id="dm:user_67890",
        )

        assert resolution.matched
        assert resolution.agent_binding == "claude-sonnet"


class TestBindingsTimeWindows:
    """Tests for time-window based binding filtering."""

    def test_business_hours_binding(self, binding_router):
        """Test binding that only applies during business hours."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="premium-agent",
                binding_type=BindingType.SPECIFIC_AGENT,
                time_window_start=9,
                time_window_end=17,
                priority=20,
            )
        )

        # Add fallback for off-hours
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="basic-agent",
                binding_type=BindingType.DEFAULT,
                priority=10,
            )
        )

        # During business hours (10 AM)
        business_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            hour=10,
        )
        assert business_resolution.agent_binding == "premium-agent"

        # After hours (8 PM)
        afterhours_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            hour=20,
        )
        assert afterhours_resolution.agent_binding == "basic-agent"

    def test_night_shift_binding(self, binding_router):
        """Test binding for overnight hours (wrapping time window)."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="night-agent",
                binding_type=BindingType.SPECIFIC_AGENT,
                time_window_start=22,  # 10 PM
                time_window_end=6,  # 6 AM (wraps around midnight)
                priority=15,
            )
        )

        # 11 PM should match night binding
        late_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            hour=23,
        )
        assert late_resolution.agent_binding == "night-agent"

        # 3 AM should also match
        early_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            hour=3,
        )
        assert early_resolution.agent_binding == "night-agent"


class TestBindingsUserFiltering:
    """Tests for user-based binding filtering."""

    def test_allowed_users_binding(self, binding_router):
        """Test binding that only allows specific users."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="vip-agent",
                binding_type=BindingType.SPECIFIC_AGENT,
                allowed_users={"U-VIP-1", "U-VIP-2", "U-VIP-3"},
                priority=50,
            )
        )

        # Fallback for non-VIP users
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="standard-agent",
                binding_type=BindingType.DEFAULT,
                priority=10,
            )
        )

        # VIP user gets VIP agent
        vip_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            user_id="U-VIP-1",
        )
        assert vip_resolution.agent_binding == "vip-agent"

        # Non-VIP user gets standard agent
        regular_resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            user_id="U-regular-user",
        )
        assert regular_resolution.agent_binding == "standard-agent"

    def test_blocked_users_binding(self, binding_router):
        """Test binding that blocks specific users."""
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@mybot",
                peer_pattern="group:*",
                agent_binding="debate-agent",
                binding_type=BindingType.SPECIFIC_AGENT,
                blocked_users={"spammer_1", "spammer_2"},
                priority=20,
            )
        )

        # Normal user gets matched
        normal_resolution = binding_router.resolve(
            provider="telegram",
            account_id="@mybot",
            peer_id="group:-1001234567890",
            user_id="normal_user",
        )
        assert normal_resolution.matched
        assert normal_resolution.agent_binding == "debate-agent"

        # Blocked user falls through to global default
        blocked_resolution = binding_router.resolve(
            provider="telegram",
            account_id="@mybot",
            peer_id="group:-1001234567890",
            user_id="spammer_1",
        )
        # Falls through to global default
        assert blocked_resolution.matched
        assert blocked_resolution.match_reason == "Global default"


class TestBindingsAgentPools:
    """Tests for agent pool based routing."""

    def test_agent_pool_binding(self, binding_router, mock_agents):
        """Test routing to an agent pool."""
        binding_router.register_agent_pool(
            "fast-agents",
            [
                "claude-sonnet",
                "gpt-4o",
            ],
        )

        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:quick-*",
                agent_binding="fast-agents",
                binding_type=BindingType.AGENT_POOL,
                priority=30,
            )
        )

        resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:quick-questions",
        )

        assert resolution.matched
        assert resolution.agent_binding == "fast-agents"
        assert resolution.binding_type == BindingType.AGENT_POOL

    def test_agent_selection_from_pool(self, binding_router, mock_agents):
        """Test selecting an agent from a pool."""
        binding_router.register_agent_pool(
            "fast-agents",
            [
                "claude-sonnet",
                "gpt-4o",
            ],
        )

        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="fast-agents",
                binding_type=BindingType.AGENT_POOL,
            )
        )

        selection = binding_router.get_agent_for_message(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:C12345",
            available_agents=mock_agents,
        )

        # Should select first available from pool
        assert selection.agent_name in ["claude-sonnet", "gpt-4o"]
        assert "pool" in selection.selection_reason.lower()


class TestBindingsDebateTeams:
    """Tests for debate team bindings."""

    def test_debate_team_binding(self, binding_router, mock_agents):
        """Test routing to a debate team."""
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@debatebot",
                peer_pattern="group:*",
                agent_binding="alpha-team",
                binding_type=BindingType.DEBATE_TEAM,
                config_overrides={
                    "rounds": 3,
                    "consensus_threshold": 0.7,
                },
                priority=25,
            )
        )

        resolution = binding_router.resolve(
            provider="telegram",
            account_id="@debatebot",
            peer_id="group:-1001234567890",
        )

        assert resolution.matched
        assert resolution.binding_type == BindingType.DEBATE_TEAM
        assert resolution.config_overrides["rounds"] == 3
        assert resolution.config_overrides["consensus_threshold"] == 0.7


class TestBindingsConfigOverrides:
    """Tests for configuration overrides in bindings."""

    def test_config_overrides_applied(self, binding_router):
        """Test that config overrides are applied to resolution."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:support-*",
                agent_binding="support-agent",
                binding_type=BindingType.SPECIFIC_AGENT,
                config_overrides={
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "system_prompt": "Be concise and helpful.",
                },
                priority=40,
            )
        )

        resolution = binding_router.resolve(
            provider="slack",
            account_id="T12345678",
            peer_id="channel:support-tier1",
        )

        assert resolution.config_overrides["temperature"] == 0.3
        assert resolution.config_overrides["max_tokens"] == 500
        assert "concise" in resolution.config_overrides["system_prompt"]


class TestBindingsStatistics:
    """Tests for binding router statistics."""

    def test_router_statistics(self, binding_router):
        """Test that router statistics are tracked."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T1",
                peer_pattern="channel:*",
                agent_binding="agent1",
            )
        )
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@bot1",
                peer_pattern="group:*",
                agent_binding="agent2",
            )
        )

        binding_router.register_agent_pool("pool1", ["agent1", "agent2"])

        stats = binding_router.get_stats()

        assert stats["total_bindings"] == 2
        assert "slack" in stats["providers"]
        assert "telegram" in stats["providers"]
        assert "pool1" in stats["agent_pools"]

    def test_list_bindings_filtered(self, binding_router):
        """Test listing bindings with filters."""
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T1",
                peer_pattern="channel:*",
                agent_binding="agent1",
            )
        )
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T2",
                peer_pattern="channel:*",
                agent_binding="agent2",
            )
        )
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@bot1",
                peer_pattern="group:*",
                agent_binding="agent3",
            )
        )

        # All bindings
        all_bindings = binding_router.list_bindings()
        assert len(all_bindings) == 3

        # Filter by provider
        slack_bindings = binding_router.list_bindings(provider="slack")
        assert len(slack_bindings) == 2

        # Filter by provider and account
        t1_bindings = binding_router.list_bindings(provider="slack", account_id="T1")
        assert len(t1_bindings) == 1
        assert t1_bindings[0].agent_binding == "agent1"


class TestBindingsMultiPlatform:
    """Tests for multi-platform binding scenarios."""

    def test_cross_platform_routing(self, binding_router, mock_agents):
        """Test routing across different platforms."""
        # Slack binding
        binding_router.add_binding(
            MessageBinding(
                provider="slack",
                account_id="T12345678",
                peer_pattern="channel:*",
                agent_binding="claude-sonnet",
                binding_type=BindingType.SPECIFIC_AGENT,
            )
        )

        # Telegram binding
        binding_router.add_binding(
            MessageBinding(
                provider="telegram",
                account_id="@mybot",
                peer_pattern="group:*",
                agent_binding="gpt-4o",
                binding_type=BindingType.SPECIFIC_AGENT,
            )
        )

        # Discord binding
        binding_router.add_binding(
            MessageBinding(
                provider="discord",
                account_id="server_123",
                peer_pattern="channel:*",
                agent_binding="gemini-pro",
                binding_type=BindingType.SPECIFIC_AGENT,
            )
        )

        # Each platform gets correct agent
        slack_res = binding_router.resolve("slack", "T12345678", "channel:C1")
        telegram_res = binding_router.resolve("telegram", "@mybot", "group:-1001")
        discord_res = binding_router.resolve("discord", "server_123", "channel:123")

        assert slack_res.agent_binding == "claude-sonnet"
        assert telegram_res.agent_binding == "gpt-4o"
        assert discord_res.agent_binding == "gemini-pro"
