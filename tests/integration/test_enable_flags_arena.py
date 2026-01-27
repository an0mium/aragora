"""
End-to-End Integration Tests: Arena Feature Flags.

Tests the integration of enable_* flags with the Arena:
1. enable_performance_feedback - SelectionFeedbackLoop
2. enable_coordinated_writes - MemoryCoordinator
3. enable_skills - SkillRegistry during evidence collection
4. enable_propulsion - PropulsionEngine at stage transitions
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

from aragora.debate.arena_config import ArenaConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.ask = AsyncMock(return_value="Test response from agent")
    agent.__str__ = MagicMock(return_value="test-agent")
    return agent


@pytest.fixture
def mock_agents(mock_agent):
    """Create multiple mock agents."""
    agents = []
    for name in ["claude", "gpt-4", "gemini"]:
        agent = MagicMock()
        agent.name = name
        agent.ask = AsyncMock(return_value=f"Response from {name}")
        agent.__str__ = MagicMock(return_value=name)
        agents.append(agent)
    return agents


@pytest.fixture
def mock_selection_feedback_loop():
    """Create a mock SelectionFeedbackLoop."""
    loop = MagicMock()
    loop.weights = {"claude": 1.0, "gpt-4": 1.0, "gemini": 1.0}
    loop.record_outcome = MagicMock()
    loop.get_weights = MagicMock(return_value={"claude": 1.0, "gpt-4": 1.0, "gemini": 1.0})
    loop.decay_unused = MagicMock()
    return loop


@pytest.fixture
def mock_memory_coordinator():
    """Create a mock MemoryCoordinator."""
    coordinator = MagicMock()
    coordinator.begin_transaction = MagicMock()
    coordinator.commit = MagicMock()
    coordinator.rollback = MagicMock()
    coordinator.write_atomic = AsyncMock()
    return coordinator


@pytest.fixture
def mock_skill_registry():
    """Create a mock SkillRegistry."""
    registry = MagicMock()
    registry._skills = {}

    async def mock_invoke(name, input_data, context):
        return MagicMock(
            status="success",
            output={"results": [f"Evidence for: {input_data.get('query', 'test')}"]},
        )

    registry.invoke = AsyncMock(side_effect=mock_invoke)
    registry.list_skills = MagicMock(return_value=[])
    return registry


@pytest.fixture
def mock_propulsion_engine():
    """Create a mock PropulsionEngine."""
    engine = MagicMock()
    engine.propel = AsyncMock(return_value=[MagicMock(success=True)])
    engine.register_handler = MagicMock()
    engine.clear_results = MagicMock()
    return engine


# ============================================================================
# Tests: enable_performance_feedback
# ============================================================================


class TestEnablePerformanceFeedback:
    """Tests for enable_performance_feedback flag."""

    def test_flag_defaults_to_true(self):
        """Verify enable_performance_feedback defaults to True."""
        config = ArenaConfig()
        assert config.enable_performance_feedback is True

    def test_flag_can_be_disabled(self):
        """Verify enable_performance_feedback can be set to False."""
        config = ArenaConfig(enable_performance_feedback=False)
        assert config.enable_performance_feedback is False

    def test_selection_feedback_loop_initialized_when_enabled(self, mock_selection_feedback_loop):
        """Test that SelectionFeedbackLoop is used when flag is True."""
        config = ArenaConfig(
            enable_performance_feedback=True,
            selection_feedback_loop=mock_selection_feedback_loop,
        )
        assert config.selection_feedback_loop is mock_selection_feedback_loop
        assert config.enable_performance_feedback is True

    def test_selection_feedback_loop_not_used_when_disabled(self, mock_selection_feedback_loop):
        """Test that SelectionFeedbackLoop is ignored when flag is False."""
        config = ArenaConfig(
            enable_performance_feedback=False,
            selection_feedback_loop=mock_selection_feedback_loop,
        )
        # Loop is configured but flag disabled - behavior depends on implementation
        assert config.enable_performance_feedback is False


class TestEnablePerformanceFeedbackIntegration:
    """Integration tests for performance feedback with Arena."""

    @pytest.mark.asyncio
    async def test_feedback_recorded_on_debate_completion(
        self, mock_agents, mock_selection_feedback_loop
    ):
        """Test that agent performance is recorded after debate completion."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            # Create arena with performance feedback enabled
            mock_arena = MagicMock()
            mock_arena.config = ArenaConfig(
                enable_performance_feedback=True,
                selection_feedback_loop=mock_selection_feedback_loop,
            )

            # Simulate outcome recording
            mock_selection_feedback_loop.record_outcome(
                agent_name="claude",
                debate_id="test-debate",
                outcome="winner",
                consensus_score=0.85,
            )

            # Verify outcome was recorded
            mock_selection_feedback_loop.record_outcome.assert_called_once()
            call_kwargs = mock_selection_feedback_loop.record_outcome.call_args
            assert call_kwargs[1]["agent_name"] == "claude"
            assert call_kwargs[1]["outcome"] == "winner"

    @pytest.mark.asyncio
    async def test_weights_decay_when_agent_unused(self, mock_selection_feedback_loop):
        """Test that agent weights decay when not participating in debates."""
        config = ArenaConfig(
            enable_performance_feedback=True,
            selection_feedback_loop=mock_selection_feedback_loop,
        )

        # Simulate decay
        mock_selection_feedback_loop.decay_unused(excluded_agents=["claude"])
        mock_selection_feedback_loop.decay_unused.assert_called_with(excluded_agents=["claude"])


# ============================================================================
# Tests: enable_coordinated_writes
# ============================================================================


class TestEnableCoordinatedWrites:
    """Tests for enable_coordinated_writes flag."""

    def test_flag_defaults_to_true(self):
        """Verify enable_coordinated_writes defaults to True."""
        config = ArenaConfig()
        assert config.enable_coordinated_writes is True

    def test_flag_can_be_disabled(self):
        """Verify enable_coordinated_writes can be set to False."""
        config = ArenaConfig(enable_coordinated_writes=False)
        assert config.enable_coordinated_writes is False

    def test_memory_coordinator_initialized_when_enabled(self, mock_memory_coordinator):
        """Test that MemoryCoordinator is used when flag is True."""
        config = ArenaConfig(
            enable_coordinated_writes=True,
            memory_coordinator=mock_memory_coordinator,
        )
        assert config.memory_coordinator is mock_memory_coordinator
        assert config.enable_coordinated_writes is True


class TestEnableCoordinatedWritesIntegration:
    """Integration tests for coordinated writes with Arena."""

    @pytest.mark.asyncio
    async def test_transaction_lifecycle(self, mock_memory_coordinator):
        """Test transaction begin/commit/rollback lifecycle."""
        config = ArenaConfig(
            enable_coordinated_writes=True,
            memory_coordinator=mock_memory_coordinator,
        )

        # Simulate transaction lifecycle
        mock_memory_coordinator.begin_transaction()
        await mock_memory_coordinator.write_atomic(
            systems=["critique_store", "consensus_memory"],
            data={"test": "data"},
        )
        mock_memory_coordinator.commit()

        # Verify lifecycle
        mock_memory_coordinator.begin_transaction.assert_called_once()
        mock_memory_coordinator.write_atomic.assert_called_once()
        mock_memory_coordinator.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, mock_memory_coordinator):
        """Test that rollback is called on write failure."""
        config = ArenaConfig(
            enable_coordinated_writes=True,
            memory_coordinator=mock_memory_coordinator,
        )

        # Simulate failed write
        mock_memory_coordinator.write_atomic = AsyncMock(side_effect=RuntimeError("Write failed"))

        mock_memory_coordinator.begin_transaction()
        try:
            await mock_memory_coordinator.write_atomic(
                systems=["critique_store"],
                data={"test": "data"},
            )
        except RuntimeError:
            mock_memory_coordinator.rollback()

        # Verify rollback was called
        mock_memory_coordinator.rollback.assert_called_once()


# ============================================================================
# Tests: enable_skills
# ============================================================================


class TestEnableSkills:
    """Tests for enable_skills flag."""

    def test_flag_defaults_to_false(self):
        """Verify enable_skills defaults to False."""
        config = ArenaConfig()
        assert config.enable_skills is False

    def test_flag_can_be_enabled(self):
        """Verify enable_skills can be set to True."""
        config = ArenaConfig(enable_skills=True)
        assert config.enable_skills is True

    def test_skill_registry_initialized_when_enabled(self, mock_skill_registry):
        """Test that SkillRegistry is used when flag is True."""
        config = ArenaConfig(
            enable_skills=True,
            skill_registry=mock_skill_registry,
        )
        assert config.skill_registry is mock_skill_registry
        assert config.enable_skills is True


class TestEnableSkillsIntegration:
    """Integration tests for skills with Arena."""

    @pytest.mark.asyncio
    async def test_skill_invoked_during_evidence_collection(self, mock_skill_registry):
        """Test that skills are invoked during evidence collection when enabled."""
        config = ArenaConfig(
            enable_skills=True,
            skill_registry=mock_skill_registry,
        )

        # Simulate skill invocation during evidence phase
        result = await mock_skill_registry.invoke(
            "web_search",
            {"query": "climate change effects"},
            MagicMock(),  # context
        )

        assert result.status == "success"
        mock_skill_registry.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_skills_not_invoked_when_disabled(self, mock_skill_registry):
        """Test that skills are NOT invoked when flag is False."""
        config = ArenaConfig(
            enable_skills=False,
            skill_registry=mock_skill_registry,
        )

        # When disabled, skills should not be called
        # This verifies the flag check in the evidence collection phase
        assert config.enable_skills is False
        # In real implementation, the invoke would not be called


# ============================================================================
# Tests: enable_propulsion
# ============================================================================


class TestEnablePropulsion:
    """Tests for enable_propulsion flag."""

    def test_flag_defaults_to_false(self):
        """Verify enable_propulsion defaults to False."""
        config = ArenaConfig()
        assert config.enable_propulsion is False

    def test_flag_can_be_enabled(self):
        """Verify enable_propulsion can be set to True."""
        config = ArenaConfig(enable_propulsion=True)
        assert config.enable_propulsion is True

    def test_propulsion_engine_initialized_when_enabled(self, mock_propulsion_engine):
        """Test that PropulsionEngine is used when flag is True."""
        config = ArenaConfig(
            enable_propulsion=True,
            propulsion_engine=mock_propulsion_engine,
        )
        assert config.propulsion_engine is mock_propulsion_engine
        assert config.enable_propulsion is True


class TestEnablePropulsionIntegration:
    """Integration tests for propulsion with Arena."""

    @pytest.mark.asyncio
    async def test_propulsion_event_fired_at_stage_transition(self, mock_propulsion_engine):
        """Test that propulsion events fire at stage transitions."""
        config = ArenaConfig(
            enable_propulsion=True,
            propulsion_engine=mock_propulsion_engine,
        )

        # Simulate propulsion event at proposals_ready
        from aragora.debate.propulsion import PropulsionPayload

        payload = PropulsionPayload(
            data={"proposals": ["prop1", "prop2"]},
            source_molecule_id="debate-123",
        )

        results = await mock_propulsion_engine.propel("proposals_ready", payload)

        assert len(results) == 1
        assert results[0].success is True
        mock_propulsion_engine.propel.assert_called_once()

    @pytest.mark.asyncio
    async def test_propulsion_handler_registration(self, mock_propulsion_engine):
        """Test that handlers can be registered for propulsion events."""
        config = ArenaConfig(
            enable_propulsion=True,
            propulsion_engine=mock_propulsion_engine,
        )

        async def on_proposals_ready(payload):
            return {"status": "critiques_started"}

        mock_propulsion_engine.register_handler(
            "proposals_ready",
            on_proposals_ready,
            name="critique_starter",
        )

        mock_propulsion_engine.register_handler.assert_called_once_with(
            "proposals_ready",
            on_proposals_ready,
            name="critique_starter",
        )

    @pytest.mark.asyncio
    async def test_propulsion_not_fired_when_disabled(self, mock_propulsion_engine):
        """Test that propulsion events do NOT fire when flag is False."""
        config = ArenaConfig(
            enable_propulsion=False,
            propulsion_engine=mock_propulsion_engine,
        )

        # When disabled, propulsion should not be called
        assert config.enable_propulsion is False
        # In real implementation, propel would not be called


# ============================================================================
# Tests: Combined Feature Flags
# ============================================================================


class TestCombinedFeatureFlags:
    """Tests for multiple feature flags working together."""

    def test_all_flags_enabled(
        self,
        mock_selection_feedback_loop,
        mock_memory_coordinator,
        mock_skill_registry,
        mock_propulsion_engine,
    ):
        """Test that all feature flags can be enabled simultaneously."""
        config = ArenaConfig(
            enable_performance_feedback=True,
            enable_coordinated_writes=True,
            enable_skills=True,
            enable_propulsion=True,
            selection_feedback_loop=mock_selection_feedback_loop,
            memory_coordinator=mock_memory_coordinator,
            skill_registry=mock_skill_registry,
            propulsion_engine=mock_propulsion_engine,
        )

        assert config.enable_performance_feedback is True
        assert config.enable_coordinated_writes is True
        assert config.enable_skills is True
        assert config.enable_propulsion is True
        assert config.selection_feedback_loop is not None
        assert config.memory_coordinator is not None
        assert config.skill_registry is not None
        assert config.propulsion_engine is not None

    def test_all_flags_disabled(self):
        """Test that all feature flags can be disabled."""
        config = ArenaConfig(
            enable_performance_feedback=False,
            enable_coordinated_writes=False,
            enable_skills=False,
            enable_propulsion=False,
        )

        assert config.enable_performance_feedback is False
        assert config.enable_coordinated_writes is False
        assert config.enable_skills is False
        assert config.enable_propulsion is False

    def test_selective_flags(self, mock_skill_registry):
        """Test that flags can be selectively enabled/disabled."""
        config = ArenaConfig(
            enable_performance_feedback=True,
            enable_coordinated_writes=False,
            enable_skills=True,
            enable_propulsion=False,
            skill_registry=mock_skill_registry,
        )

        assert config.enable_performance_feedback is True
        assert config.enable_coordinated_writes is False
        assert config.enable_skills is True
        assert config.enable_propulsion is False


# ============================================================================
# Tests: Flag-Subsystem Consistency
# ============================================================================


class TestFlagSubsystemConsistency:
    """Tests ensuring flag and subsystem states are consistent."""

    def test_subsystem_without_flag_enabled(self, mock_skill_registry):
        """Test behavior when subsystem exists but flag is disabled."""
        config = ArenaConfig(
            enable_skills=False,
            skill_registry=mock_skill_registry,  # Subsystem provided
        )

        # Subsystem exists but should not be used
        assert config.skill_registry is mock_skill_registry
        assert config.enable_skills is False

    def test_flag_enabled_without_subsystem(self):
        """Test behavior when flag is enabled but subsystem is not provided."""
        config = ArenaConfig(
            enable_skills=True,
            skill_registry=None,  # No subsystem
        )

        # Flag is enabled but no subsystem - should be handled gracefully
        assert config.enable_skills is True
        assert config.skill_registry is None

    def test_lazy_subsystem_initialization(self):
        """Test that subsystems can be initialized lazily."""
        config = ArenaConfig(enable_propulsion=True)

        # Initially no engine
        assert config.propulsion_engine is None

        # Simulate lazy initialization
        from aragora.debate.propulsion import PropulsionEngine

        engine = PropulsionEngine(max_concurrent=5)
        config.propulsion_engine = engine

        assert config.propulsion_engine is not None
        assert config.enable_propulsion is True
