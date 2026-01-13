"""
Expanded tests for debate orchestrator.

Tests role rotation, timeout handling, and conviction-weighted voting
that weren't covered in basic orchestrator tests.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core import Agent, Environment
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol, user_vote_multiplier
from aragora.debate.roles import CognitiveRole, RoleRotationConfig, RoleRotator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.role = "proposer"
    agent.system_prompt = "You are a helpful assistant."
    agent.generate = AsyncMock(return_value="Test response")
    return agent


@pytest.fixture
def mock_agents():
    """Create multiple mock agents."""
    agents = []
    for name in ["claude", "gpt4", "gemini"]:
        agent = MagicMock()
        agent.name = name
        agent.role = None  # Not assigned yet
        agent.system_prompt = f"You are {name}."
        agent.generate = AsyncMock(return_value=f"{name} response")
        agents.append(agent)
    return agents


@pytest.fixture
def basic_env():
    """Create a basic environment."""
    return Environment(task="Test debate topic")


# =============================================================================
# Role Rotation Tests
# =============================================================================


class TestOrchestratorRoleRotation:
    """Tests for cognitive role rotation in Arena."""

    def test_role_rotator_initialized_when_enabled(self, basic_env, mock_agents):
        """Role rotator should be initialized when protocol enables it."""
        protocol = DebateProtocol(role_rotation=True)
        arena = Arena(basic_env, mock_agents, protocol)

        assert arena.role_rotator is not None
        assert isinstance(arena.role_rotator, RoleRotator)

    def test_role_rotator_not_initialized_when_disabled(self, basic_env, mock_agents):
        """Role rotator should be None when protocol disables it."""
        protocol = DebateProtocol(role_rotation=False)
        arena = Arena(basic_env, mock_agents, protocol)

        assert arena.role_rotator is None

    def test_custom_role_config_used(self, basic_env, mock_agents):
        """Custom role rotation config should be used."""
        custom_config = RoleRotationConfig(
            enabled=True,
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
            synthesizer_final_round=False,
        )
        protocol = DebateProtocol(
            role_rotation=True,
            role_rotation_config=custom_config,
        )
        arena = Arena(basic_env, mock_agents, protocol)

        # Check config was passed through
        assert arena.role_rotator.config.synthesizer_final_round is False
        assert len(arena.role_rotator.config.roles) == 2

    def test_update_role_assignments(self, basic_env, mock_agents):
        """_update_role_assignments should assign roles to all agents."""
        protocol = DebateProtocol(role_rotation=True, rounds=5)
        arena = Arena(basic_env, mock_agents, protocol)

        # Update for round 0
        arena._update_role_assignments(round_num=0)

        # All agents should have role assignments
        for agent in mock_agents:
            assert agent.name in arena.current_role_assignments

    def test_role_assignments_change_between_rounds(self, basic_env, mock_agents):
        """Role assignments should change between rounds."""
        protocol = DebateProtocol(role_rotation=True, rounds=5)
        arena = Arena(basic_env, mock_agents, protocol)

        # Get assignments for round 0
        arena._update_role_assignments(round_num=0)
        round_0_roles = {
            name: assign.role for name, assign in arena.current_role_assignments.items()
        }

        # Get assignments for round 1
        arena._update_role_assignments(round_num=1)
        round_1_roles = {
            name: assign.role for name, assign in arena.current_role_assignments.items()
        }

        # At least one agent should have a different role
        different = any(
            round_0_roles.get(name) != round_1_roles.get(name) for name in round_0_roles
        )
        assert different, "Role assignments should rotate between rounds"

    def test_get_role_context_returns_prompt(self, basic_env, mock_agents):
        """_get_role_context should return role prompt for assigned agent."""
        protocol = DebateProtocol(role_rotation=True)
        arena = Arena(basic_env, mock_agents, protocol)
        arena._update_role_assignments(round_num=0)

        # Get context for first agent
        context = arena._get_role_context(mock_agents[0])

        # Should contain role header
        assert "COGNITIVE ROLE ASSIGNMENT" in context

    def test_get_role_context_empty_when_disabled(self, basic_env, mock_agents):
        """_get_role_context should return empty string when rotation disabled."""
        protocol = DebateProtocol(role_rotation=False)
        arena = Arena(basic_env, mock_agents, protocol)

        context = arena._get_role_context(mock_agents[0])
        assert context == ""


# =============================================================================
# Timeout Tests
# =============================================================================


class TestOrchestratorTimeout:
    """Tests for timeout handling in Arena."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self, basic_env, mock_agents):
        """AutonomicExecutor.with_timeout should return result on success."""
        protocol = DebateProtocol()
        arena = Arena(basic_env, mock_agents, protocol)

        async def quick_task():
            return "success"

        result = await arena.autonomic.with_timeout(quick_task(), "test-agent", timeout_seconds=5.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_timeout_raises_on_timeout(self, basic_env, mock_agents):
        """AutonomicExecutor.with_timeout should raise TimeoutError when exceeded."""
        protocol = DebateProtocol()
        arena = Arena(basic_env, mock_agents, protocol)

        async def slow_task():
            await asyncio.sleep(10)
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            await arena.autonomic.with_timeout(slow_task(), "test-agent", timeout_seconds=0.1)

        assert "test-agent" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_timeout_records_circuit_breaker_failure(self, basic_env, mock_agents):
        """Timeout should record circuit breaker failure."""
        protocol = DebateProtocol()
        arena = Arena(basic_env, mock_agents, protocol)

        async def slow_task():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            await arena.autonomic.with_timeout(slow_task(), "claude", timeout_seconds=0.1)

        # Circuit breaker should have recorded the failure
        # The _failures dict tracks per-entity failures
        assert arena.circuit_breaker._failures.get("claude", 0) >= 1

    def test_debate_timeout_protocol_setting(self, basic_env, mock_agents):
        """Protocol timeout settings should be respected."""
        protocol = DebateProtocol(
            timeout_seconds=300,
            round_timeout_seconds=60,
        )
        arena = Arena(basic_env, mock_agents, protocol)

        assert arena.protocol.timeout_seconds == 300
        assert arena.protocol.round_timeout_seconds == 60


# =============================================================================
# Conviction-Weighted Voting Tests
# =============================================================================


class TestOrchestratorConviction:
    """Tests for conviction-weighted voting."""

    def test_user_vote_multiplier_integration(self):
        """user_vote_multiplier should work with default protocol."""
        protocol = DebateProtocol()

        # Test various intensities
        low = user_vote_multiplier(2, protocol)
        neutral = user_vote_multiplier(5, protocol)
        high = user_vote_multiplier(8, protocol)

        assert low < neutral < high
        assert neutral == 1.0

    def test_high_conviction_increases_weight(self):
        """High conviction votes should have higher weight."""
        protocol = DebateProtocol()

        neutral_weight = user_vote_multiplier(5, protocol)
        high_weight = user_vote_multiplier(10, protocol)

        assert high_weight > neutral_weight
        assert high_weight == pytest.approx(2.0, rel=0.01)

    def test_low_conviction_decreases_weight(self):
        """Low conviction votes should have lower weight."""
        protocol = DebateProtocol()

        neutral_weight = user_vote_multiplier(5, protocol)
        low_weight = user_vote_multiplier(1, protocol)

        assert low_weight < neutral_weight
        assert low_weight == pytest.approx(0.5, rel=0.01)

    def test_custom_conviction_parameters(self):
        """Custom protocol parameters should affect multiplier."""
        protocol = DebateProtocol(
            user_vote_intensity_min_multiplier=0.25,
            user_vote_intensity_max_multiplier=4.0,
        )

        low = user_vote_multiplier(1, protocol)
        high = user_vote_multiplier(10, protocol)

        assert low == pytest.approx(0.25, rel=0.01)
        assert high == pytest.approx(4.0, rel=0.01)


# =============================================================================
# Arena Initialization Tests
# =============================================================================


class TestArenaInitialization:
    """Tests for Arena initialization edge cases."""

    def test_empty_agents_list(self, basic_env):
        """Arena should handle empty agents list."""
        protocol = DebateProtocol()
        arena = Arena(basic_env, [], protocol)

        assert arena.agents == []

    def test_auto_upgrade_to_elo_judge(self, basic_env, mock_agents):
        """Should auto-upgrade to ELO-ranked judge when ELO system provided."""
        protocol = DebateProtocol(judge_selection="random")
        mock_elo = MagicMock()

        arena = Arena(basic_env, mock_agents, protocol, elo_system=mock_elo)

        # Should have upgraded from random to elo_ranked
        assert arena.protocol.judge_selection == "elo_ranked"

    def test_no_upgrade_when_explicit_judge_selection(self, basic_env, mock_agents):
        """Should not upgrade judge selection when explicitly set."""
        protocol = DebateProtocol(judge_selection="voted")
        mock_elo = MagicMock()

        arena = Arena(basic_env, mock_agents, protocol, elo_system=mock_elo)

        # Should keep explicit selection
        assert arena.protocol.judge_selection == "voted"

    def test_default_protocol_used_when_none(self, basic_env, mock_agents):
        """Default protocol should be used when None provided."""
        arena = Arena(basic_env, mock_agents, protocol=None)

        assert arena.protocol is not None
        assert arena.protocol.rounds == 5  # Default value

    def test_circuit_breaker_initialized(self, basic_env, mock_agents):
        """Circuit breaker should be initialized even if not provided."""
        arena = Arena(basic_env, mock_agents)

        assert arena.circuit_breaker is not None
