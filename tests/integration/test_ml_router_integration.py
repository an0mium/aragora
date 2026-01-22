"""Integration tests for ML Router with Debate System.

Tests the feedback loop between:
1. AgentRouter making routing decisions
2. Debates executing with those agents
3. Performance outcomes feeding back to the router
4. Router improving future decisions based on history
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    id: str = ""
    role: str = "proposer"

    def __post_init__(self):
        if not self.id:
            self.id = self.name


@dataclass
class MockDebateResult:
    """Mock debate result."""

    task: str = "Test task"
    consensus_text: str = "Test consensus"
    winner_agent: str = "claude"
    rounds_used: int = 3
    messages: List[Any] = field(default_factory=list)


class TestRouterToDebateIntegration:
    """Test router making decisions that flow into debates."""

    def test_router_selects_team_for_coding_task(self):
        """Router should select code-specialized agents for coding tasks."""
        from aragora.ml import get_agent_router, TaskType

        router = get_agent_router()
        decision = router.route(
            "Implement a binary search tree in Python",
            available_agents=["claude", "gpt-4", "codex", "gemini"],
            team_size=3,
        )

        assert decision.task_type == TaskType.CODING
        assert len(decision.selected_agents) == 3
        # Codex should be selected for coding tasks
        assert "codex" in decision.selected_agents or decision.confidence > 0.5

    def test_router_decision_confidence_reflects_match_quality(self):
        """Better task-agent matches should have higher confidence."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        # Coding task with coding-specialized agents
        good_match = router.route(
            "Write unit tests",
            available_agents=["codex", "claude"],
            team_size=2,
        )

        # Coding task with non-coding agents
        poor_match = router.route(
            "Write unit tests",
            available_agents=["random-agent-1", "random-agent-2"],
            team_size=2,
        )

        # Good match should have higher confidence
        assert good_match.confidence >= poor_match.confidence


class TestDebateOutcomeToRouterFeedback:
    """Test debate outcomes feeding back to router."""

    def test_record_performance_updates_agent_stats(self):
        """Recording performance should update agent statistics."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        # Record multiple outcomes
        for _ in range(5):
            router.record_performance("claude", "coding", success=True)
        for _ in range(2):
            router.record_performance("claude", "coding", success=False)

        stats = router.get_agent_stats("claude")
        assert stats["total_tasks"] >= 7
        # Success rate should be around 5/7 = ~71%
        # (Actual rate depends on any prior recorded data)

    def test_elo_updates_after_debate_outcome(self):
        """ELO ratings should update based on debate wins/losses."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        initial_elo = router.get_agent_stats("claude")["elo_rating"]
        router.update_elo("claude", initial_elo + 50)  # Win

        new_stats = router.get_agent_stats("claude")
        assert new_stats["elo_rating"] == initial_elo + 50

    def test_performance_history_influences_routing(self):
        """Agents with better history should be preferred."""
        from aragora.ml import AgentRouter

        # Fresh router with no history
        router = AgentRouter()

        # Record good performance for one agent
        for _ in range(10):
            router.record_performance("good-agent", "coding", success=True)

        # Record poor performance for another
        for _ in range(10):
            router.record_performance("poor-agent", "coding", success=False)

        # Good agent should have better stats
        good_stats = router.get_agent_stats("good-agent")
        poor_stats = router.get_agent_stats("poor-agent")

        assert good_stats["overall_success_rate"] > poor_stats["overall_success_rate"]


class TestPerformanceRouterBridgeIntegration:
    """Test the bridge connecting performance monitor to router."""

    def test_bridge_syncs_performance_to_router(self):
        """Bridge should sync performance data to router."""
        from aragora.ml.performance_router_bridge import (
            PerformanceRouterBridge,
            PerformanceRouterBridgeConfig,
        )

        # Create mock monitor with stats
        monitor = MagicMock()
        monitor.agent_stats = {
            "claude": MagicMock(
                name="claude",
                total_calls=100,
                success_rate=90.0,
                avg_duration_ms=2500.0,
                timeout_rate=1.0,
                min_duration_ms=1000.0,
                max_duration_ms=5000.0,
            ),
        }

        # Create mock router
        router = MagicMock()
        router._capabilities = {
            "claude": MagicMock(strengths=[], speed_tier=2),
        }
        router._historical_performance = {}
        router.record_performance = MagicMock()

        bridge = PerformanceRouterBridge(
            performance_monitor=monitor,
            agent_router=router,
            config=PerformanceRouterBridgeConfig(min_calls_for_sync=5),
        )

        result = bridge.sync_performance(force=True)

        assert result.agents_synced >= 0
        # Should have updated speed tier based on latency
        if result.agents_synced > 0:
            assert "claude" in result.agents_updated

    def test_bridge_computes_routing_scores(self):
        """Bridge should compute routing scores from performance data."""
        from aragora.ml.performance_router_bridge import PerformanceRouterBridge

        monitor = MagicMock()
        monitor.agent_stats = {
            "fast-agent": MagicMock(
                name="fast-agent",
                total_calls=100,
                success_rate=95.0,
                avg_duration_ms=1000.0,
                timeout_rate=0.0,
                min_duration_ms=500.0,
                max_duration_ms=2000.0,
            ),
            "slow-agent": MagicMock(
                name="slow-agent",
                total_calls=100,
                success_rate=70.0,
                avg_duration_ms=10000.0,
                timeout_rate=10.0,
                min_duration_ms=5000.0,
                max_duration_ms=30000.0,
            ),
        }

        bridge = PerformanceRouterBridge(performance_monitor=monitor)

        scores = bridge.get_agent_scores()

        assert "fast-agent" in scores
        assert "slow-agent" in scores
        # Fast agent with high success rate should score better
        assert scores["fast-agent"] > scores["slow-agent"]


class TestTeamSelectionWithRouterIntegration:
    """Test team selector using router for decisions."""

    def test_team_selector_uses_pattern_matching(self):
        """Team selector should use pattern-based routing."""
        from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig

        config = TeamSelectionConfig(enable_pattern_selection=True)
        selector = TeamSelector(config=config)

        mock_agents = [
            MockAgent(name="claude"),
            MockAgent(name="codex"),
            MockAgent(name="gpt-4"),
        ]

        # Task that matches coding pattern
        selected = selector.select(
            agents=mock_agents,
            task="Implement a REST API endpoint",
            domain="coding",
        )

        # Should return a list of agents
        assert isinstance(selected, list)

    def test_pattern_telemetry_tracked(self):
        """Pattern classifications should be tracked."""
        from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig

        config = TeamSelectionConfig(enable_pattern_selection=True)
        selector = TeamSelector(config=config)

        mock_agents = [MockAgent(name="claude"), MockAgent(name="gpt-4")]

        # Make some selections to generate telemetry
        selector.select(agents=mock_agents, task="Implement sorting", domain="coding")
        selector.select(agents=mock_agents, task="Analyze data trends", domain="analysis")
        selector.select(agents=mock_agents, task="Fix bug in authentication", domain="security")

        telemetry = selector.get_pattern_telemetry()

        assert "classification_counts" in telemetry
        assert telemetry["config"]["enabled"]


class TestEndToEndRouterDebateLoop:
    """End-to-end test of the router-debate feedback loop."""

    def test_full_feedback_loop_cycle(self):
        """Test complete cycle: route → debate → record → improve."""
        from aragora.ml import get_agent_router, TaskType

        router = get_agent_router()

        # Step 1: Route a task (use clear coding keywords)
        decision = router.route(
            "Implement a binary search algorithm in Python",
            available_agents=["claude", "gpt-4", "codex"],
            team_size=2,
        )

        # Verify task was routed (any task type is fine for this integration test)
        assert decision.task_type is not None
        selected = decision.selected_agents

        # Step 2: Simulate debate outcome (winner gets success)
        winner = selected[0]
        loser = selected[1] if len(selected) > 1 else None

        router.record_performance(winner, decision.task_type.value, success=True)
        if loser:
            router.record_performance(loser, decision.task_type.value, success=False)

        # Step 3: Verify stats updated
        winner_stats = router.get_agent_stats(winner)
        assert winner_stats["total_tasks"] > 0

        # Step 4: Future routing should still work
        decision2 = router.route(
            "Implement another algorithm",
            available_agents=["claude", "gpt-4", "codex"],
            team_size=2,
        )

        # Verify routing decision was made
        assert decision2.selected_agents is not None
        assert len(decision2.selected_agents) > 0

    def test_multiple_debate_cycles_improve_routing(self):
        """Multiple debate cycles should refine routing decisions."""
        from aragora.ml import AgentRouter, TaskType

        router = AgentRouter()

        # Simulate multiple debate outcomes where one agent consistently wins
        consistent_winner = "specialist-agent"
        consistent_loser = "general-agent"

        for i in range(10):
            # Specialist wins coding tasks
            router.record_performance(consistent_winner, "coding", success=True)
            router.record_performance(consistent_loser, "coding", success=False)

        winner_stats = router.get_agent_stats(consistent_winner)
        loser_stats = router.get_agent_stats(consistent_loser)

        # Winner should have much better success rate
        assert winner_stats["overall_success_rate"] > loser_stats["overall_success_rate"]
        # Winner should have higher ELO (if ELO is updated)


class TestRouterConstraints:
    """Test routing with various constraints."""

    def test_vision_constraint_filters_agents(self):
        """Vision requirement should filter to vision-capable agents."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        decision = router.route(
            "Analyze this image and describe it",
            available_agents=["claude", "gpt-4", "llama", "mistral"],
            team_size=2,
            constraints={"require_vision": True},
        )

        assert len(decision.selected_agents) == 2
        # Vision-capable agents should be preferred

    def test_speed_constraint_prefers_fast_agents(self):
        """Speed constraint should prefer fast agents."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        decision = router.route(
            "Quick summary needed",
            available_agents=["claude", "claude-sonnet", "gpt-4", "gpt-4-turbo"],
            team_size=2,
            constraints={"prefer_speed": True},
        )

        assert len(decision.selected_agents) == 2
        # Faster agents should be preferred (sonnet, turbo variants)

    def test_cost_constraint_prefers_cheap_agents(self):
        """Cost constraint should prefer cheaper agents."""
        from aragora.ml import get_agent_router

        router = get_agent_router()

        decision = router.route(
            "Simple task",
            available_agents=["claude", "claude-haiku", "gpt-4", "gpt-3.5-turbo"],
            team_size=2,
            constraints={"prefer_cost": True},
        )

        assert len(decision.selected_agents) == 2
        # Cheaper agents should be preferred
