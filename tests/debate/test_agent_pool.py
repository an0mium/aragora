"""
Tests for aragora.debate.agent_pool module.

Covers:
- AgentPoolConfig and AgentMetrics dataclasses
- AgentPool initialization and agent access
- Team selection (performance-based and random)
- Composite score calculation
- Critic selection (mesh, ring, star topologies)
- Metrics management
- Circuit breaker integration
- Pool status reporting
"""

import random
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.agent_pool import (
    AgentMetrics,
    AgentPool,
    AgentPoolConfig,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    model: str = "test-model"


@pytest.fixture
def agents():
    """Create a list of mock agents."""
    return [
        MockAgent(name="agent_a"),
        MockAgent(name="agent_b"),
        MockAgent(name="agent_c"),
        MockAgent(name="agent_d"),
        MockAgent(name="agent_e"),
    ]


@pytest.fixture
def pool(agents):
    """Create a basic agent pool."""
    return AgentPool(agents)


@pytest.fixture
def mock_elo():
    """Create a mock ELO system."""
    elo = MagicMock()
    elo.get_rating.side_effect = lambda name: {
        "agent_a": 1200,
        "agent_b": 1100,
        "agent_c": 1000,
        "agent_d": 900,
        "agent_e": 800,
    }.get(name, 1000)
    elo.get_domain_rating.side_effect = lambda name, domain: None
    return elo


@pytest.fixture
def mock_calibration():
    """Create a mock calibration tracker."""
    cal = MagicMock()
    cal.get_calibration.side_effect = lambda name: {
        "agent_a": 0.8,  # High calibration
        "agent_b": 0.6,
        "agent_c": 0.5,
        "agent_d": 0.4,
        "agent_e": 0.2,
    }.get(name, 0.5)
    return cal


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    cb = MagicMock()
    cb.is_open.side_effect = lambda name: name in {"agent_d", "agent_e"}
    return cb


# ============================================================================
# AgentPoolConfig Tests
# ============================================================================


class TestAgentPoolConfig:
    """Tests for AgentPoolConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentPoolConfig()

        assert config.elo_system is None
        assert config.calibration_tracker is None
        assert config.circuit_breaker is None
        assert config.use_performance_selection is False
        assert config.performance_weight == 0.7
        assert config.min_team_size == 2
        assert config.max_team_size == 10
        assert config.topology == "full_mesh"
        assert config.critic_count == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentPoolConfig(
            use_performance_selection=True,
            performance_weight=0.8,
            min_team_size=3,
            max_team_size=5,
            topology="ring",
            critic_count=3,
        )

        assert config.use_performance_selection is True
        assert config.performance_weight == 0.8
        assert config.min_team_size == 3
        assert config.max_team_size == 5
        assert config.topology == "ring"
        assert config.critic_count == 3


# ============================================================================
# AgentMetrics Tests
# ============================================================================


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = AgentMetrics(name="test_agent")

        assert metrics.name == "test_agent"
        assert metrics.elo_rating == 1000.0
        assert metrics.calibration_score == 0.5
        assert metrics.debates_participated == 0
        assert metrics.win_rate == 0.5
        assert metrics.is_available is True

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = AgentMetrics(
            name="test_agent",
            elo_rating=1500.0,
            calibration_score=0.8,
            debates_participated=10,
            win_rate=0.7,
            is_available=False,
        )

        assert metrics.elo_rating == 1500.0
        assert metrics.calibration_score == 0.8
        assert metrics.debates_participated == 10
        assert metrics.win_rate == 0.7
        assert metrics.is_available is False


# ============================================================================
# AgentPool Initialization Tests
# ============================================================================


class TestAgentPoolInit:
    """Tests for AgentPool initialization."""

    def test_init_with_agents(self, agents):
        """Initialize pool with agents."""
        pool = AgentPool(agents)

        assert len(pool.agents) == 5
        assert pool._agent_map["agent_a"] == agents[0]

    def test_init_with_empty_agents(self):
        """Initialize pool with empty list."""
        pool = AgentPool([])

        assert len(pool.agents) == 0
        assert pool._metrics == {}

    def test_init_with_config(self, agents):
        """Initialize pool with custom config."""
        config = AgentPoolConfig(use_performance_selection=True)
        pool = AgentPool(agents, config)

        assert pool._config.use_performance_selection is True

    def test_init_creates_metrics(self, agents):
        """Initialization creates metrics for all agents."""
        pool = AgentPool(agents)

        assert len(pool._metrics) == 5
        assert "agent_a" in pool._metrics
        assert pool._metrics["agent_a"].name == "agent_a"


# ============================================================================
# Agent Access Tests
# ============================================================================


class TestAgentAccess:
    """Tests for agent access methods."""

    def test_agents_property(self, pool, agents):
        """agents property returns copy of agent list."""
        result = pool.agents
        assert result == agents
        # Verify it's a copy
        result.append(MockAgent(name="new"))
        assert len(pool.agents) == 5

    def test_available_agents_no_circuit_breaker(self, pool, agents):
        """available_agents returns all when no circuit breaker."""
        assert pool.available_agents == agents

    def test_available_agents_with_circuit_breaker(self, agents, mock_circuit_breaker):
        """available_agents excludes circuit-broken agents."""
        config = AgentPoolConfig(circuit_breaker=mock_circuit_breaker)
        pool = AgentPool(agents, config)

        available = pool.available_agents
        names = [a.name for a in available]

        assert "agent_a" in names
        assert "agent_b" in names
        assert "agent_c" in names
        assert "agent_d" not in names
        assert "agent_e" not in names

    def test_get_agent_existing(self, pool):
        """get_agent returns agent by name."""
        agent = pool.get_agent("agent_a")
        assert agent is not None
        assert agent.name == "agent_a"

    def test_get_agent_nonexistent(self, pool):
        """get_agent returns None for unknown agent."""
        agent = pool.get_agent("nonexistent")
        assert agent is None

    def test_require_agents_success(self, pool):
        """require_agents returns agents when sufficient."""
        result = pool.require_agents(min_count=3)
        assert len(result) == 5

    def test_require_agents_insufficient(self, pool):
        """require_agents raises when insufficient agents."""
        with pytest.raises(ValueError, match="Insufficient agents"):
            pool.require_agents(min_count=10)

    def test_require_agents_with_circuit_breaker(self, agents, mock_circuit_breaker):
        """require_agents respects circuit breaker."""
        config = AgentPoolConfig(circuit_breaker=mock_circuit_breaker)
        pool = AgentPool(agents, config)

        # 3 agents available (a, b, c), d and e are circuit-broken
        result = pool.require_agents(min_count=3)
        assert len(result) == 3

        with pytest.raises(ValueError, match="Insufficient agents"):
            pool.require_agents(min_count=4)


# ============================================================================
# Team Selection Tests
# ============================================================================


class TestTeamSelection:
    """Tests for team selection."""

    def test_select_team_random(self, pool):
        """Random selection when performance selection disabled."""
        # Run multiple times to verify randomness
        teams = [pool.select_team(team_size=3) for _ in range(10)]

        # All teams should have 3 agents
        for team in teams:
            assert len(team) == 3

    def test_select_team_performance_based(self, agents, mock_elo, mock_calibration):
        """Performance-based selection picks highest scoring agents."""
        config = AgentPoolConfig(
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
            use_performance_selection=True,
        )
        pool = AgentPool(agents, config)

        team = pool.select_team(team_size=3)
        names = [a.name for a in team]

        # Should select agents with highest composite scores
        # agent_a: 1200 * (0.5 + 0.8) = 1560
        # agent_b: 1100 * (0.5 + 0.6) = 1210
        # agent_c: 1000 * (0.5 + 0.5) = 1000
        assert "agent_a" in names
        assert "agent_b" in names

    def test_select_team_respects_max_size(self, agents):
        """Team size respects max_team_size config when no explicit size."""
        config = AgentPoolConfig(max_team_size=3)
        pool = AgentPool(agents, config)

        # When team_size is not specified, max_team_size is used
        team = pool.select_team()
        assert len(team) == 3

    def test_select_team_respects_min_size(self, agents):
        """Team size respects min_team_size config."""
        config = AgentPoolConfig(min_team_size=3)
        pool = AgentPool(agents, config)

        team = pool.select_team(team_size=2)  # Request less than min
        assert len(team) == 3

    def test_select_team_with_exclude(self, pool):
        """Exclude specified agents from selection."""
        team = pool.select_team(team_size=3, exclude={"agent_a", "agent_b"})
        names = [a.name for a in team]

        assert "agent_a" not in names
        assert "agent_b" not in names
        assert len(team) == 3

    def test_select_team_empty_pool(self):
        """Empty pool returns empty team."""
        pool = AgentPool([])
        team = pool.select_team(team_size=3)
        assert team == []

    def test_select_team_all_excluded(self, pool):
        """All agents excluded returns empty team."""
        team = pool.select_team(exclude={"agent_a", "agent_b", "agent_c", "agent_d", "agent_e"})
        assert team == []


# ============================================================================
# Composite Score Tests
# ============================================================================


class TestCompositeScore:
    """Tests for composite score calculation."""

    def test_compute_score_no_systems(self, pool):
        """Score defaults to 1000 without scoring systems."""
        score = pool._compute_composite_score("agent_a")
        assert score == 1000.0  # 1000 * 1.0 (default calibration)

    def test_compute_score_with_elo(self, agents, mock_elo):
        """Score incorporates ELO rating."""
        config = AgentPoolConfig(elo_system=mock_elo)
        pool = AgentPool(agents, config)

        score_a = pool._compute_composite_score("agent_a")
        score_e = pool._compute_composite_score("agent_e")

        assert score_a > score_e  # agent_a has higher ELO

    def test_compute_score_with_calibration(self, agents, mock_calibration):
        """Score incorporates calibration weight."""
        config = AgentPoolConfig(calibration_tracker=mock_calibration)
        pool = AgentPool(agents, config)

        score_a = pool._compute_composite_score("agent_a")  # cal 0.8 -> weight 1.3
        score_e = pool._compute_composite_score("agent_e")  # cal 0.2 -> weight 0.7

        assert score_a > score_e

    def test_compute_score_with_both(self, agents, mock_elo, mock_calibration):
        """Score combines ELO and calibration."""
        config = AgentPoolConfig(
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
        )
        pool = AgentPool(agents, config)

        # agent_a: elo=1200, cal=0.8 -> 1200 * 1.3 = 1560
        # agent_c: elo=1000, cal=0.5 -> 1000 * 1.0 = 1000
        score_a = pool._compute_composite_score("agent_a")
        score_c = pool._compute_composite_score("agent_c")

        assert score_a == pytest.approx(1560.0)
        assert score_c == pytest.approx(1000.0)


# ============================================================================
# Calibration Weight Tests
# ============================================================================


class TestCalibrationWeight:
    """Tests for calibration weight calculation."""

    def test_weight_no_tracker(self, pool):
        """Weight is 1.0 without tracker."""
        weight = pool._get_calibration_weight("agent_a")
        assert weight == 1.0

    def test_weight_with_tracker(self, agents, mock_calibration):
        """Weight maps calibration to 0.5-1.5 range."""
        config = AgentPoolConfig(calibration_tracker=mock_calibration)
        pool = AgentPool(agents, config)

        # cal 0.8 -> weight 0.5 + 0.8 = 1.3
        weight = pool._get_calibration_weight("agent_a")
        assert weight == pytest.approx(1.3)

        # cal 0.2 -> weight 0.5 + 0.2 = 0.7
        weight = pool._get_calibration_weight("agent_e")
        assert weight == pytest.approx(0.7)

    def test_weight_unknown_agent(self, agents, mock_calibration):
        """Unknown agent returns weight 1.0."""
        mock_calibration.get_calibration.side_effect = KeyError("unknown")
        config = AgentPoolConfig(calibration_tracker=mock_calibration)
        pool = AgentPool(agents, config)

        weight = pool._get_calibration_weight("unknown_agent")
        assert weight == 1.0


# ============================================================================
# Critic Selection Tests
# ============================================================================


class TestCriticSelection:
    """Tests for critic selection."""

    def test_select_critics_mesh(self, pool, agents):
        """Mesh topology selects random critics."""
        proposer = agents[0]
        critics = pool.select_critics(proposer, count=2)

        # Should not include proposer
        names = [c.name for c in critics]
        assert proposer.name not in names
        assert len(critics) == 2

    def test_select_critics_ring(self, agents):
        """Ring topology selects neighbors."""
        config = AgentPoolConfig(topology="ring")
        pool = AgentPool(agents, config)

        # agent_b's neighbors are agent_a and agent_c
        proposer = agents[1]  # agent_b
        critics = pool.select_critics(proposer, count=2)

        names = [c.name for c in critics]
        # Should prefer neighbors
        assert "agent_a" in names or "agent_c" in names

    def test_select_critics_star(self, agents):
        """Star topology uses hub as critic."""
        config = AgentPoolConfig(topology="star")
        pool = AgentPool(agents, config)

        # Hub is agent_a (first agent)
        # Non-hub proposes, hub should critique
        proposer = agents[2]  # agent_c
        critics = pool.select_critics(proposer, count=2)

        names = [c.name for c in critics]
        assert "agent_a" in names

    def test_select_critics_star_hub_proposes(self, agents):
        """Star topology: hub proposes, anyone can critique."""
        config = AgentPoolConfig(topology="star")
        pool = AgentPool(agents, config)

        # Hub proposes
        proposer = agents[0]  # agent_a (hub)
        critics = pool.select_critics(proposer, count=2)

        # Hub not in critics (it's the proposer)
        names = [c.name for c in critics]
        assert "agent_a" not in names

    def test_select_critics_excludes_proposer(self, pool, agents):
        """Proposer is always excluded from critics."""
        for proposer in agents:
            critics = pool.select_critics(proposer, count=4)
            names = [c.name for c in critics]
            assert proposer.name not in names

    def test_select_critics_empty_pool(self, agents):
        """Empty candidate pool returns empty critics."""
        config = AgentPoolConfig()
        pool = AgentPool(agents, config)

        # Only proposer available
        proposer = agents[0]
        critics = pool.select_critics(proposer, candidates=[proposer])
        assert critics == []


# ============================================================================
# Metrics Management Tests
# ============================================================================


class TestMetricsManagement:
    """Tests for metrics management."""

    def test_update_metrics_elo(self, pool):
        """Update ELO rating metric."""
        pool.update_metrics("agent_a", elo_rating=1500.0)

        metrics = pool.get_agent_metrics("agent_a")
        assert metrics is not None
        assert metrics.elo_rating == 1500.0

    def test_update_metrics_calibration(self, pool):
        """Update calibration score metric."""
        pool.update_metrics("agent_a", calibration_score=0.9)

        metrics = pool.get_agent_metrics("agent_a")
        assert metrics.calibration_score == 0.9

    def test_update_metrics_participation(self, pool):
        """Update participation and win rate."""
        pool.update_metrics("agent_a", debate_participated=True, won=True)
        pool.update_metrics("agent_a", debate_participated=True, won=True)

        metrics = pool.get_agent_metrics("agent_a")
        assert metrics.debates_participated == 2
        # Win rate only updates on won=True
        # After 2 wins: win_rate = 2 wins / 2 debates = 1.0
        assert metrics.win_rate == pytest.approx(1.0)

    def test_update_metrics_new_agent(self, pool):
        """Update creates metrics for new agent."""
        pool.update_metrics("new_agent", elo_rating=1100.0)

        metrics = pool.get_agent_metrics("new_agent")
        assert metrics is not None
        assert metrics.name == "new_agent"
        assert metrics.elo_rating == 1100.0

    def test_get_metrics_nonexistent(self, pool):
        """Getting metrics for unknown agent returns None."""
        metrics = pool.get_agent_metrics("nonexistent")
        assert metrics is None


# ============================================================================
# Scoring Systems Tests
# ============================================================================


class TestScoringSystems:
    """Tests for scoring system configuration."""

    def test_set_scoring_systems(self, pool, mock_elo, mock_calibration):
        """Setting scoring systems enables performance selection."""
        assert pool._config.use_performance_selection is False

        pool.set_scoring_systems(elo_system=mock_elo, calibration_tracker=mock_calibration)

        assert pool._config.use_performance_selection is True
        assert pool._config.elo_system is mock_elo
        assert pool._config.calibration_tracker is mock_calibration

    def test_set_elo_only(self, pool, mock_elo):
        """Setting only ELO system works."""
        pool.set_scoring_systems(elo_system=mock_elo)

        assert pool._config.elo_system is mock_elo
        assert pool._config.use_performance_selection is True

    def test_set_calibration_only(self, pool, mock_calibration):
        """Setting only calibration tracker works."""
        pool.set_scoring_systems(calibration_tracker=mock_calibration)

        assert pool._config.calibration_tracker is mock_calibration


# ============================================================================
# Pool Status Tests
# ============================================================================


class TestPoolStatus:
    """Tests for pool status reporting."""

    def test_get_pool_status(self, pool):
        """Get basic pool status."""
        status = pool.get_pool_status()

        assert status["total_agents"] == 5
        assert status["available_agents"] == 5
        assert status["circuit_broken"] == 0
        assert status["topology"] == "full_mesh"
        assert status["performance_selection"] is False
        assert len(status["agents"]) == 5

    def test_get_pool_status_with_circuit_breaker(self, agents, mock_circuit_breaker):
        """Pool status reflects circuit breaker state."""
        config = AgentPoolConfig(circuit_breaker=mock_circuit_breaker)
        pool = AgentPool(agents, config)

        status = pool.get_pool_status()

        assert status["total_agents"] == 5
        assert status["available_agents"] == 3
        assert status["circuit_broken"] == 2

    def test_pool_status_agent_details(self, pool):
        """Pool status includes agent details."""
        status = pool.get_pool_status()

        agent_info = status["agents"][0]
        assert "name" in agent_info
        assert "available" in agent_info
        assert "metrics" in agent_info


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_agents_without_name_attribute(self):
        """Pool handles agents without name attribute."""
        agents = ["string_agent_1", "string_agent_2"]
        pool = AgentPool(agents)

        assert len(pool.agents) == 2
        # String representation used as name
        assert pool.get_agent("string_agent_1") is not None

    def test_duplicate_agent_names(self):
        """Pool handles duplicate agent names (last wins)."""
        agents = [
            MockAgent(name="agent_a", model="model1"),
            MockAgent(name="agent_a", model="model2"),  # Duplicate
        ]
        pool = AgentPool(agents)

        agent = pool.get_agent("agent_a")
        assert agent.model == "model2"  # Last one wins

    def test_circuit_breaker_exception(self, agents):
        """Circuit breaker exceptions are handled gracefully."""
        cb = MagicMock()
        # Use KeyError which is explicitly caught
        cb.is_open.side_effect = KeyError("CB error")

        config = AgentPoolConfig(circuit_breaker=cb)
        pool = AgentPool(agents, config)

        # Should not raise, treats as available
        available = pool.available_agents
        assert len(available) == 5

    def test_elo_system_exception(self, agents):
        """ELO system exceptions are handled gracefully."""
        elo = MagicMock()
        # Use KeyError which is explicitly caught
        elo.get_rating.side_effect = KeyError("ELO error")

        config = AgentPoolConfig(elo_system=elo, use_performance_selection=True)
        pool = AgentPool(agents, config)

        # Should not raise, falls back to default
        score = pool._compute_composite_score("agent_a")
        assert score == 1000.0  # Default

    def test_select_team_larger_than_available(self, agents, mock_circuit_breaker):
        """Team size limited to available agents."""
        config = AgentPoolConfig(circuit_breaker=mock_circuit_breaker)
        pool = AgentPool(agents, config)

        # Only 3 available, request 5
        team = pool.select_team(team_size=5)
        assert len(team) == 3


# ============================================================================
# Integration Tests
# ============================================================================


class TestAgentPoolIntegration:
    """Integration tests for AgentPool."""

    def test_full_workflow(self, agents, mock_elo, mock_calibration, mock_circuit_breaker):
        """Complete workflow: config, selection, metrics, status."""
        config = AgentPoolConfig(
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
            circuit_breaker=mock_circuit_breaker,
            use_performance_selection=True,
            topology="ring",
            critic_count=2,
        )
        pool = AgentPool(agents, config)

        # Check initial status
        status = pool.get_pool_status()
        assert status["total_agents"] == 5
        assert status["available_agents"] == 3  # d and e circuit-broken

        # Select team
        team = pool.select_team(team_size=3)
        assert len(team) == 3

        # Select critics
        proposer = team[0]
        critics = pool.select_critics(proposer)
        assert proposer not in critics

        # Update metrics
        pool.update_metrics(proposer.name, debate_participated=True, won=True)
        metrics = pool.get_agent_metrics(proposer.name)
        assert metrics.debates_participated == 1

    def test_team_selection_determinism(self, agents, mock_elo, mock_calibration):
        """Performance-based selection is deterministic."""
        config = AgentPoolConfig(
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
            use_performance_selection=True,
        )
        pool = AgentPool(agents, config)

        teams = [pool.select_team(team_size=3) for _ in range(5)]

        # All teams should be identical (deterministic selection)
        for team in teams:
            names = [a.name for a in team]
            assert names == [a.name for a in teams[0]]
