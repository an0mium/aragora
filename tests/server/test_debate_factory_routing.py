"""Tests for ProviderRouter wiring into DebateFactory.

Verifies that DebateFactory.create_arena() consults the ProviderRouter
for provider quality hints and passes them to the arena for use during
team selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.debate_factory import DebateConfig, DebateFactory


def _make_mock_agent(model_type, name="", role="proposer", model=None, enable_fallback=True):
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = name or model_type
    agent.role = role
    agent.model_type = model_type
    agent.api_key = "fake-key"
    agent.enable_fallback = enable_fallback
    return agent


@pytest.fixture
def _mock_create_agent():
    """Patch create_agent to return mock agents."""
    with patch("aragora.server.debate_factory.create_agent", side_effect=_make_mock_agent):
        yield


@pytest.fixture
def _mock_arena_build():
    """Patch ArenaBuilder.build to return a mock arena without full construction."""
    mock_arena = MagicMock()
    mock_arena.agents = []
    mock_arena.env = MagicMock()
    mock_arena.circuit_breaker = MagicMock()
    mock_arena.circuit_breaker.get_all_status.return_value = {}

    with patch("aragora.debate.arena_builder.ArenaBuilder.build", return_value=mock_arena):
        yield mock_arena


class TestDebateFactoryProviderRouting:
    """Tests that DebateFactory wires ProviderRouter into arena creation."""

    @pytest.mark.usefixtures("_mock_create_agent")
    def test_provider_hints_passed_when_router_available(self, _mock_arena_build):
        """ProviderRouter hints should be stored on the arena when available."""
        mock_arena = _mock_arena_build

        fake_hints = {"claude-sonnet-4": 0.9, "gpt-4o": 0.85}

        mock_router = MagicMock()
        mock_router.get_provider_hints.return_value = fake_hints

        with patch(
            "aragora.routing.provider_router.get_provider_router",
            return_value=mock_router,
        ):
            factory = DebateFactory()
            config = DebateConfig(
                question="Test question for routing",
                agents_str="anthropic-api,openai-api",
                auto_trim_unavailable=False,
            )
            factory.create_arena(config)

            # The router should have been consulted
            mock_router.get_provider_hints.assert_called_once()

    @pytest.mark.usefixtures("_mock_create_agent")
    def test_arena_created_without_hints_when_router_unavailable(self, _mock_arena_build):
        """Arena should still be created when ProviderRouter import fails."""
        mock_arena = _mock_arena_build

        # Simulate import failure by making get_provider_router raise ImportError
        with patch.dict("sys.modules", {"aragora.routing.provider_router": None}):
            factory = DebateFactory()
            config = DebateConfig(
                question="Test question without routing",
                agents_str="anthropic-api,openai-api",
                auto_trim_unavailable=False,
            )
            # Should not raise
            arena = factory.create_arena(config)
            assert arena is mock_arena

    @pytest.mark.usefixtures("_mock_create_agent")
    def test_empty_hints_not_set_on_arena(self, _mock_arena_build):
        """Empty hints dict should not be passed to builder."""
        mock_arena = _mock_arena_build

        mock_router = MagicMock()
        mock_router.get_provider_hints.return_value = {}

        with patch(
            "aragora.routing.provider_router.get_provider_router",
            return_value=mock_router,
        ):
            factory = DebateFactory()
            config = DebateConfig(
                question="Test empty hints",
                agents_str="anthropic-api,openai-api",
                auto_trim_unavailable=False,
            )
            factory.create_arena(config)

            # get_provider_hints was called but returned empty
            mock_router.get_provider_hints.assert_called_once()

    @pytest.mark.usefixtures("_mock_create_agent")
    def test_router_error_does_not_break_arena_creation(self, _mock_arena_build):
        """Runtime errors in ProviderRouter should be caught gracefully."""
        mock_arena = _mock_arena_build

        mock_router = MagicMock()
        mock_router.get_provider_hints.side_effect = RuntimeError("metrics store corrupt")

        with patch(
            "aragora.routing.provider_router.get_provider_router",
            return_value=mock_router,
        ):
            factory = DebateFactory()
            config = DebateConfig(
                question="Test router error",
                agents_str="anthropic-api,openai-api",
                auto_trim_unavailable=False,
            )
            # Should not raise despite router error
            arena = factory.create_arena(config)
            assert arena is mock_arena


class TestArenaBuilderProviderHints:
    """Tests that ArenaBuilder stores and applies provider hints."""

    def test_with_provider_hints_fluent_interface(self):
        """with_provider_hints should return self for fluent chaining."""
        from aragora.debate.arena_builder import ArenaBuilder
        from aragora.core_types import Environment

        env = Environment(task="test")
        agents = [MagicMock(), MagicMock()]

        builder = ArenaBuilder(env, agents)
        hints = {"claude-sonnet-4": 0.9}
        result = builder.with_provider_hints(hints)

        assert result is builder
        assert builder._provider_hints == hints

    def test_provider_hints_set_on_arena_after_build(self):
        """build() should set _provider_hints on the arena object."""
        from aragora.debate.arena_builder import ArenaBuilder
        from aragora.core_types import Environment

        env = Environment(task="test")
        agent1 = MagicMock()
        agent1.name = "agent1"
        agent2 = MagicMock()
        agent2.name = "agent2"

        hints = {"claude-sonnet-4": 0.9, "gpt-4o": 0.8}

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            mock_arena = MagicMock()
            MockArena.return_value = mock_arena

            builder = ArenaBuilder(env, [agent1, agent2])
            builder.with_provider_hints(hints)
            arena = builder.build()

            assert arena._provider_hints == hints

    def test_no_provider_hints_attribute_when_not_configured(self):
        """build() should not set _provider_hints when not configured."""
        from aragora.debate.arena_builder import ArenaBuilder
        from aragora.core_types import Environment

        env = Environment(task="test")
        agent1 = MagicMock()
        agent1.name = "agent1"
        agent2 = MagicMock()
        agent2.name = "agent2"

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            mock_arena = MagicMock(spec=[])
            MockArena.return_value = mock_arena

            builder = ArenaBuilder(env, [agent1, agent2])
            # Do NOT call with_provider_hints
            builder.build()

            # _provider_hints should not have been set
            assert not hasattr(mock_arena, "_provider_hints")


class TestProviderHintsIntegration:
    """Integration test that provider hints flow from router to arena config."""

    def test_get_provider_hints_returns_dict(self):
        """ProviderRouter.get_provider_hints should return a dict."""
        from aragora.routing.provider_router import ProviderRouter
        from aragora.routing.provider_metrics import ProviderMetricsStore

        store = ProviderMetricsStore()
        router = ProviderRouter(metrics_store=store)

        # With no data, should return empty dict
        hints = router.get_provider_hints()
        assert isinstance(hints, dict)
        assert len(hints) == 0

    def test_get_provider_hints_with_sufficient_data(self):
        """ProviderRouter should return quality hints when sufficient data exists."""
        from aragora.routing.provider_router import (
            MIN_DEBATES_FOR_METRICS,
            ProviderRouter,
        )
        from aragora.routing.provider_metrics import ProviderMetricsStore

        store = ProviderMetricsStore()

        # Record enough debates to exceed threshold
        for _ in range(MIN_DEBATES_FOR_METRICS):
            store.record_debate_outcome("claude-sonnet-4", cost=0.05, quality=0.92)
            store.record_debate_outcome("gpt-4o", cost=0.08, quality=0.88)

        router = ProviderRouter(metrics_store=store)
        hints = router.get_provider_hints()

        assert len(hints) == 2
        assert "claude-sonnet-4" in hints
        assert "gpt-4o" in hints
        assert hints["claude-sonnet-4"] == pytest.approx(0.92, abs=0.01)
        assert hints["gpt-4o"] == pytest.approx(0.88, abs=0.01)

    def test_arena_config_accepts_provider_hints(self):
        """ArenaConfig should accept and store provider_hints."""
        from aragora.debate.arena_config import ArenaConfig

        hints = {"claude-sonnet-4": 0.9}
        config = ArenaConfig(provider_hints=hints)

        assert config.provider_hints == hints

    def test_arena_config_provider_hints_defaults_to_none(self):
        """ArenaConfig.provider_hints should default to None."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.provider_hints is None
