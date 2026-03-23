"""Tests for ProviderRouter <-> TeamSelector integration.

Covers:
- TeamSelector.select() accepts and uses provider_hints
- Agents with higher-scoring providers are ranked higher
- provider_hints=None has no effect on behavior
- ProviderRouter.record_outcome accumulates metrics
- ProviderRouter.get_provider_hints returns quality mappings
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core_types import Agent, DebateResult
from aragora.debate.service import DebateService
from aragora.debate.team_selector import TeamSelectionConfig, TeamSelector
from aragora.routing.provider_router import ProviderRouter


# ===========================================================================
# Mock Agent
# ===========================================================================


class MockAgent:
    """Minimal mock agent for testing."""

    def __init__(self, name: str):
        self.name = name
        self.agent_type = "mock"
        self.model = ""
        self.capabilities: set[str] = set()
        self.hierarchy_role = None
        self.metadata: dict = {}


class RuntimeAgent(Agent):
    """Minimal executable agent used for runtime routing tests."""

    async def generate(self, prompt, context=None):
        return f"{self.name}:{prompt}"

    async def critique(self, proposal, task, context=None, target_agent=None):
        return "critique"


# ===========================================================================
# Task 9: Bridge ProviderRouter -> TeamSelector
# ===========================================================================


class TestTeamSelectorProviderHints:
    """Test that TeamSelector.select() integrates provider_hints."""

    def _make_selector(self, **kwargs) -> TeamSelector:
        """Create a TeamSelector with all optional subsystems disabled."""
        config = TeamSelectionConfig(
            enable_domain_filtering=False,
            enable_cv_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_budget_filtering=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_memory_selection=False,
            enable_pulse_selection=False,
            enable_regression_penalty=False,
            enable_introspection_scoring=False,
            enable_health_filtering=False,
            enable_staking_reputation=False,
            enable_reliability_budget_routing=False,
        )
        return TeamSelector(config=config, **kwargs)

    def test_provider_hints_none_no_change(self):
        """provider_hints=None should not alter agent ordering."""
        agents = [MockAgent("claude"), MockAgent("gpt"), MockAgent("gemini")]
        selector = self._make_selector()

        result_no_hints = selector.select(agents, provider_hints=None)
        result_default = selector.select(agents)

        # Both should return the same ordering (base score only)
        assert [a.name for a in result_no_hints] == [a.name for a in result_default]

    def test_provider_hints_boosts_matching_agent(self):
        """Agents whose provider matches a high hint should rank higher."""
        agents = [MockAgent("claude"), MockAgent("gpt"), MockAgent("gemini")]
        selector = self._make_selector()

        # Give gemini a much higher hint, others low/absent
        hints = {"gemini": 0.9, "gpt": 0.1}

        result = selector.select(agents, provider_hints=hints)
        names = [a.name for a in result]

        # gemini should be first (highest boost), gpt second (low boost),
        # claude last (no hint, no boost)
        assert names[0] == "gemini"
        assert names[-1] == "claude"

    def test_provider_hints_empty_dict_no_effect(self):
        """Empty provider_hints dict should behave like None."""
        agents = [MockAgent("alpha"), MockAgent("beta")]
        selector = self._make_selector()

        result_empty = selector.select(agents, provider_hints={})
        result_none = selector.select(agents, provider_hints=None)

        assert [a.name for a in result_empty] == [a.name for a in result_none]

    def test_provider_hints_substring_matching(self):
        """Hints should match agents by substring (e.g., 'claude' matches 'claude-sonnet')."""
        agents = [MockAgent("claude-sonnet-4"), MockAgent("gpt-4o")]
        selector = self._make_selector()

        hints = {"claude": 0.8}
        result = selector.select(agents, provider_hints=hints)

        # claude-sonnet-4 should rank first because "claude" is a substring
        assert result[0].name == "claude-sonnet-4"

    def test_provider_hints_multiplicative_effect(self):
        """Provider hints should have a multiplicative effect on the score."""
        agents = [MockAgent("agent_a"), MockAgent("agent_b")]
        selector = self._make_selector()

        # Both agents start with the same base score.
        # agent_a gets a hint of 1.0 -> score *= 2.0
        # agent_b gets a hint of 0.0 -> score *= 1.0 (no change)
        hints = {"agent_a": 1.0, "agent_b": 0.0}

        result = selector.select(agents, provider_hints=hints)
        assert result[0].name == "agent_a"
        assert result[1].name == "agent_b"

    def test_provider_hints_with_elo_scores(self):
        """Provider hints should augment ELO-based scoring, not replace it."""

        class MockElo:
            def get_rating(self, name: str) -> float:
                # gpt has a much higher ELO
                return {"claude": 900.0, "gpt": 1200.0}.get(name, 1000.0)

        agents = [MockAgent("claude"), MockAgent("gpt")]
        selector = self._make_selector(elo_system=MockElo())

        # Without hints, gpt should win due to higher ELO
        result_no_hints = selector.select(agents, provider_hints=None)
        assert result_no_hints[0].name == "gpt"

        # With a strong hint for claude, it should overtake gpt
        hints = {"claude": 2.0}
        result_with_hints = selector.select(agents, provider_hints=hints)
        assert result_with_hints[0].name == "claude"


class TestProviderRouterOutcomeRecording:
    """Test ProviderRouter.record_outcome and get_provider_hints."""

    def test_record_outcome_accumulates_metrics(self):
        """record_outcome should accumulate quality and cost for a provider."""
        router = ProviderRouter()

        router.record_outcome("anthropic", quality=0.9, cost=0.10)
        router.record_outcome("anthropic", quality=0.8, cost=0.12)

        metrics = router.metrics_store.get_metrics("anthropic")
        assert metrics is not None
        assert metrics.total_debates == 2
        assert abs(metrics.avg_quality_score - 0.85) < 0.01
        assert abs(metrics.total_cost - 0.22) < 0.01

    def test_record_outcome_multiple_providers(self):
        """record_outcome should track providers independently."""
        router = ProviderRouter()

        router.record_outcome("anthropic", quality=0.9, cost=0.10)
        router.record_outcome("openai", quality=0.7, cost=0.05)

        all_metrics = router.metrics_store.get_all_metrics()
        assert "anthropic" in all_metrics
        assert "openai" in all_metrics
        assert all_metrics["anthropic"].avg_quality_score > all_metrics["openai"].avg_quality_score

    def test_get_provider_hints_with_sufficient_data(self):
        """get_provider_hints returns quality scores when enough data exists."""
        router = ProviderRouter()

        # Record enough outcomes to exceed MIN_DEBATES_FOR_METRICS (10)
        for _ in range(6):
            router.record_outcome("anthropic", quality=0.9, cost=0.10)
        for _ in range(6):
            router.record_outcome("openai", quality=0.7, cost=0.05)

        hints = router.get_provider_hints()
        assert "anthropic" in hints
        assert "openai" in hints
        assert hints["anthropic"] > hints["openai"]

    def test_get_provider_hints_insufficient_data(self):
        """get_provider_hints returns empty dict with sparse data."""
        router = ProviderRouter()
        router.record_outcome("anthropic", quality=0.9, cost=0.10)

        hints = router.get_provider_hints()
        assert hints == {}

    def test_record_outcome_with_failure(self):
        """record_outcome should track failures."""
        router = ProviderRouter()

        router.record_outcome("anthropic", quality=0.0, cost=0.0, failed=True)
        router.record_outcome("anthropic", quality=0.8, cost=0.10, failed=False)

        metrics = router.metrics_store.get_metrics("anthropic")
        assert metrics is not None
        assert metrics.total_debates == 2
        assert metrics.failure_rate == 0.5


class TestEndToEndRouterToSelector:
    """End-to-end: ProviderRouter produces hints, TeamSelector consumes them."""

    def test_router_hints_flow_to_selector(self):
        """Full flow: record outcomes -> get hints -> pass to select()."""
        router = ProviderRouter()

        # Record enough data for metrics
        for _ in range(6):
            router.record_outcome("claude", quality=0.95, cost=0.10)
        for _ in range(6):
            router.record_outcome("gpt", quality=0.60, cost=0.05)

        hints = router.get_provider_hints()
        assert len(hints) == 2

        agents = [MockAgent("claude"), MockAgent("gpt")]
        config = TeamSelectionConfig(
            enable_domain_filtering=False,
            enable_cv_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_budget_filtering=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_memory_selection=False,
            enable_pulse_selection=False,
            enable_regression_penalty=False,
            enable_introspection_scoring=False,
            enable_health_filtering=False,
            enable_staking_reputation=False,
            enable_reliability_budget_routing=False,
        )
        selector = TeamSelector(config=config)

        result = selector.select(agents, provider_hints=hints)
        # claude should rank first due to higher quality hint
        assert result[0].name == "claude"


class TestDebateServiceRuntimeRouting:
    """ProviderRouter selections should affect the Arena roster, not just metadata."""

    @pytest.mark.asyncio
    async def test_provider_list_routes_runtime_roster_before_arena(self):
        agents = [
            RuntimeAgent(name="anthropic-proposer", model="claude-sonnet-4"),
            RuntimeAgent(name="openai-critic", model="gpt-4o"),
            RuntimeAgent(name="google-analyst", model="gemini-2.0-flash"),
        ]
        debate_result = DebateResult(
            debate_id="debate-1",
            task="Design a rate limiter",
            final_answer="Use approach A",
            consensus_reached=True,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=agents)
            result = await service.run(
                "Design a rate limiter",
                provider_hints=["gpt-4o", "claude-sonnet-4"],
                agent_count=2,
                min_providers=2,
            )

        selected_agents = mock_arena_cls.call_args[0][1]
        assert [agent.name for agent in selected_agents] == [
            "openai-critic",
            "anthropic-proposer",
        ]
        assert result.metadata["provider_hints"] == ["gpt-4o", "claude-sonnet-4"]
        assert result.metadata["provider_names"] == ["gpt-4o", "claude-sonnet-4"]
        assert result.metadata["provider_routing"]["routed_agent_names"] == [
            "openai-critic",
            "anthropic-proposer",
        ]

    @pytest.mark.asyncio
    async def test_runtime_fill_preserves_provider_diversity_when_requested(self):
        agents = [
            RuntimeAgent(name="openai-primary", model="gpt-4o"),
            RuntimeAgent(name="openai-backup", model="gpt-4o-mini"),
            RuntimeAgent(name="anthropic-backup", model="claude-sonnet-4"),
        ]
        debate_result = DebateResult(
            debate_id="debate-2",
            task="Design a rate limiter",
            final_answer="Use approach B",
            consensus_reached=True,
        )

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=agents)
            await service.run(
                "Design a rate limiter",
                provider_hints=["gpt-4o"],
                agent_count=2,
                min_providers=2,
            )

        selected_agents = mock_arena_cls.call_args[0][1]
        assert [agent.name for agent in selected_agents] == [
            "openai-primary",
            "anthropic-backup",
        ]


# ===========================================================================
# Task: AgentPool provider hints integration
# ===========================================================================


class TestAgentPoolProviderHints:
    """Test that AgentPool.select_team() applies provider routing hints."""

    def test_provider_hints_boost_agent_scores(self):
        """Agents matching high-quality providers should rank higher."""
        from aragora.debate.agent_pool import AgentPool, AgentPoolConfig

        agents = [MockAgent("claude-agent"), MockAgent("gpt-agent"), MockAgent("mistral-agent")]
        config = AgentPoolConfig(use_performance_selection=True)
        pool = AgentPool(agents, config)

        # Without hints: agents sorted by default composite score (all equal)
        baseline = pool.select_team(team_size=3)

        # With hints: gpt-agent gets a big boost
        hints = {"gpt": 0.9, "claude": 0.1, "mistral": 0.0}
        boosted = pool.select_team(team_size=3, provider_hints=hints)
        boosted_names = [getattr(a, "name", str(a)) for a in boosted]

        # gpt-agent should now be first (highest hint = 0.9 → 1.9x multiplier)
        assert boosted_names[0] == "gpt-agent"

    def test_provider_hints_none_is_noop(self):
        """provider_hints=None should not change selection."""
        from aragora.debate.agent_pool import AgentPool, AgentPoolConfig

        agents = [MockAgent("agent-a"), MockAgent("agent-b")]
        config = AgentPoolConfig(use_performance_selection=True)
        pool = AgentPool(agents, config)

        without = pool.select_team(team_size=2, provider_hints=None)
        with_empty = pool.select_team(team_size=2, provider_hints={})
        assert [a.name for a in without] == [a.name for a in with_empty]


class TestOrchestratorAgentsProviderHints:
    """Test that orchestrator_agents.select_debate_team handles hints."""

    def test_reorder_by_provider_hints_without_performance_selection(self):
        """When performance selection is off, provider hints still reorder agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        agents = [MockAgent("mistral-agent"), MockAgent("claude-agent"), MockAgent("gpt-agent")]
        for a in agents:
            a.provider = a.name.split("-")[0]

        hints = {"claude": 0.9, "gpt": 0.5, "mistral": 0.1}
        result = select_debate_team(
            agents=agents,
            env=MagicMock(task="test"),
            extract_domain_fn=lambda: "general",
            enable_ml_delegation=False,
            ml_delegation_strategy=None,
            protocol=MagicMock(),
            use_performance_selection=False,
            agent_pool=MagicMock(),
            provider_hints=hints,
        )
        names = [a.name for a in result]
        # claude-agent (0.9) should be first, then gpt (0.5), then mistral (0.1)
        assert names[0] == "claude-agent"
        assert names[-1] == "mistral-agent"

    def test_no_hints_returns_original_order(self):
        """Without hints and without performance selection, original order preserved."""
        from aragora.debate.orchestrator_agents import select_debate_team

        agents = [MockAgent("a"), MockAgent("b"), MockAgent("c")]
        result = select_debate_team(
            agents=agents,
            env=MagicMock(task="test"),
            extract_domain_fn=lambda: "general",
            enable_ml_delegation=False,
            ml_delegation_strategy=None,
            protocol=MagicMock(),
            use_performance_selection=False,
            agent_pool=MagicMock(),
            provider_hints=None,
        )
        assert [a.name for a in result] == ["a", "b", "c"]


class TestProviderOutcomeRecording:
    """Test that debate completion records outcomes to ProviderRouter."""

    def test_record_provider_outcomes_calls_router(self):
        """_record_provider_outcomes should call router.record_outcome per agent."""
        from aragora.debate.orchestrator_runner import _record_provider_outcomes

        arena = MagicMock()
        state = MagicMock()
        state.ctx.agents = [MockAgent("claude"), MockAgent("gpt")]
        state.ctx.result.consensus_reached = True
        state.ctx.result.confidence = 0.85
        state.debate_status = "completed"
        state.debate_start_time = 0.0

        with patch("aragora.routing.provider_router.get_provider_router") as mock_get:
            mock_router = MagicMock()
            mock_get.return_value = mock_router
            _record_provider_outcomes(arena, state)

            assert mock_router.record_outcome.call_count == 2
            # Check first call args
            call_args = mock_router.record_outcome.call_args_list[0]
            assert call_args[0][0] == "claude"  # provider name from agent.name
            assert call_args[1]["consensus_reached"] is True

    def test_record_provider_outcomes_handles_import_error(self):
        """Should not raise when provider_router import fails."""
        from aragora.debate.orchestrator_runner import _record_provider_outcomes

        arena = MagicMock()
        state = MagicMock()
        state.ctx.agents = [MockAgent("agent")]
        state.ctx.result.confidence = 0.5
        state.debate_status = "completed"

        with patch(
            "aragora.routing.provider_router.get_provider_router",
            side_effect=ImportError("not found"),
        ):
            # Should not raise
            _record_provider_outcomes(arena, state)
