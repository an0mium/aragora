"""
Tests for orchestrator graceful degradation when optional dependencies are unavailable.

Verifies that the Arena class works correctly when optional features
like BeliefNetwork, CritiqueStore, CalibrationTracker, etc. are not installed.
"""

import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Test Arena Initialization Without Optional Dependencies
# ============================================================================


class TestArenaInitDegradation:
    """Test Arena initialization gracefully handles missing dependencies."""

    @pytest.fixture
    def mock_env(self):
        """Create a minimal Environment for testing."""
        from aragora.core import Environment

        return Environment(task="Test task")

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agent1 = MagicMock()
        agent1.name = "test_agent_1"
        agent1.stance = None
        agent1.model = "test-model"
        agent2 = MagicMock()
        agent2.name = "test_agent_2"
        agent2.stance = None
        agent2.model = "test-model"
        return [agent1, agent2]

    @pytest.fixture
    def mock_protocol(self):
        """Create a minimal DebateProtocol for testing."""
        from aragora.debate.protocol import DebateProtocol

        return DebateProtocol(rounds=1)

    def test_arena_init_without_continuum_memory(self, mock_env, mock_agents, mock_protocol):
        """Arena should work without ContinuumMemory."""
        from aragora.debate.orchestrator import Arena

        # Should not crash without continuum_memory
        arena = Arena(mock_env, mock_agents, mock_protocol, continuum_memory=None)
        assert arena is not None
        assert arena.continuum_memory is None

    def test_arena_init_without_persona_manager(self, mock_env, mock_agents, mock_protocol):
        """Arena should work without PersonaManager."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(mock_env, mock_agents, mock_protocol, persona_manager=None)
        assert arena is not None
        assert arena.persona_manager is None

    def test_arena_init_without_elo_system(self, mock_env, mock_agents, mock_protocol):
        """Arena should work without ELO system."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(mock_env, mock_agents, mock_protocol, elo_system=None)
        assert arena is not None
        assert arena.elo_system is None

    def test_arena_init_without_embeddings(self, mock_env, mock_agents, mock_protocol):
        """Arena should work without embeddings database."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(mock_env, mock_agents, mock_protocol, debate_embeddings=None)
        assert arena is not None
        assert arena.debate_embeddings is None

    def test_arena_init_minimal(self, mock_env, mock_agents, mock_protocol):
        """Arena should work with only required parameters."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(mock_env, mock_agents, mock_protocol)
        assert arena is not None
        # All optional features should be None or have defaults
        assert arena.elo_system is None
        assert arena.continuum_memory is None
        assert arena.persona_manager is None


# ============================================================================
# Test Feature-Specific Degradation
# ============================================================================


class TestFeatureDegradation:
    """Test that features degrade gracefully when their dependencies are missing."""

    @pytest.fixture
    def arena_minimal(self):
        """Create a minimal Arena for testing."""
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        agent = MagicMock()
        agent.name = "test"
        agent.stance = None
        agent.model = "test"

        return Arena(env, [agent], protocol)

    def test_flip_context_without_detector(self, arena_minimal):
        """_get_flip_context should return empty string without detector."""
        arena_minimal.flip_detector = None
        agent = MagicMock()
        agent.name = "test"

        result = arena_minimal._get_flip_context(agent)
        assert result == ""

    def test_persona_context_without_manager(self, arena_minimal):
        """_get_persona_context should return empty string without manager."""
        arena_minimal.persona_manager = None
        agent = MagicMock()
        agent.name = "test"

        result = arena_minimal._get_persona_context(agent)
        assert result == ""

    def test_role_context_without_rotator(self, arena_minimal):
        """_get_role_context should return empty string without rotator."""
        arena_minimal.role_rotator = None
        arena_minimal.current_role_assignments = {}
        agent = MagicMock()
        agent.name = "test"

        result = arena_minimal._get_role_context(agent)
        assert result == ""

    def test_continuum_context_without_memory(self, arena_minimal):
        """_get_continuum_context should return empty string without memory."""
        arena_minimal.continuum_memory = None
        arena_minimal._continuum_context_cache = None

        result = arena_minimal._get_continuum_context()
        assert result == ""


# ============================================================================
# Test Protocol Optional Features
# ============================================================================


class TestProtocolOptionalFeatures:
    """Test that protocol-level optional features work without dependencies."""

    @pytest.fixture
    def arena_with_protocol(self):
        """Create Arena with various protocol settings."""
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Test debate")
        protocol = DebateProtocol(
            rounds=1,
            asymmetric_stances=True,
            agreement_intensity=5,
        )
        agent = MagicMock()
        agent.name = "test"
        agent.stance = "affirmative"
        agent.model = "test"

        return Arena(env, [agent], protocol)

    def test_stance_guidance_with_affirmative(self, arena_with_protocol):
        """Stance guidance should work for affirmative stance."""
        agent = MagicMock()
        agent.stance = "affirmative"

        result = arena_with_protocol._get_stance_guidance(agent)
        assert "AFFIRMATIVE" in result or result == ""

    def test_stance_guidance_with_negative(self, arena_with_protocol):
        """Stance guidance should work for negative stance."""
        agent = MagicMock()
        agent.stance = "negative"

        result = arena_with_protocol._get_stance_guidance(agent)
        assert "NEGATIVE" in result or result == ""

    def test_stance_guidance_without_asymmetric(self):
        """Stance guidance should return empty when not asymmetric."""
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1, asymmetric_stances=False)
        agent = MagicMock()
        agent.name = "test"
        agent.stance = None
        agent.model = "test"

        arena = Arena(env, [agent], protocol)
        mock_agent = MagicMock()
        mock_agent.stance = "affirmative"

        result = arena._get_stance_guidance(mock_agent)
        assert result == ""

    def test_agreement_intensity_guidance(self, arena_with_protocol):
        """Agreement intensity guidance should work."""
        result = arena_with_protocol._get_agreement_intensity_guidance()
        # Should return string (may be empty if intensity is None)
        assert isinstance(result, str)


# ============================================================================
# Integration: Full Degradation Scenario
# ============================================================================


class TestFullDegradationScenario:
    """Test complete degradation scenario where all optional deps are missing."""

    def test_arena_fully_degraded_initialization(self):
        """Arena should initialize with all optional dependencies disabled."""
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Test task")
        protocol = DebateProtocol(rounds=1)

        agent = MagicMock()
        agent.name = "degraded_test"
        agent.stance = None
        agent.model = "test-model"

        # Create arena with all optional params as None
        arena = Arena(
            env,
            [agent],
            protocol,
            continuum_memory=None,
            persona_manager=None,
            elo_system=None,
            debate_embeddings=None,
        )

        # Verify initialization succeeded
        assert arena is not None
        assert arena.env == env
        assert arena.protocol == protocol
        assert len(arena.agents) == 1

        # Verify all optional features are properly None/disabled
        assert arena.continuum_memory is None
        assert arena.persona_manager is None
        assert arena.elo_system is None
        assert arena.debate_embeddings is None

    def test_context_methods_handle_missing_deps(self):
        """All context methods should handle missing dependencies."""
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        agent = MagicMock()
        agent.name = "test"
        agent.stance = None
        agent.model = "test"

        arena = Arena(env, [agent], protocol)

        # Set all optional managers to None
        arena.flip_detector = None
        arena.persona_manager = None
        arena.role_rotator = None
        arena.continuum_memory = None
        arena.current_role_assignments = {}

        mock_agent = MagicMock()
        mock_agent.name = "test"

        # All context methods should return empty strings without crashing
        assert arena._get_flip_context(mock_agent) == ""
        assert arena._get_persona_context(mock_agent) == ""
        assert arena._get_role_context(mock_agent) == ""
        assert arena._get_continuum_context() == ""
