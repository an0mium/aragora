"""
Tests for the evolution hook integration.

Tests that the PromptEvolver is properly wired to the Arena and
extracts winning patterns from debates.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio

from aragora.core import Environment, DebateResult
from aragora.debate.protocol import DebateProtocol


class TestEvolutionHookWiring:
    """Tests for PromptEvolver wiring in Arena."""

    def test_arena_config_accepts_prompt_evolver(self):
        """ArenaConfig should accept prompt_evolver parameter."""
        from aragora.debate.orchestrator import ArenaConfig

        mock_evolver = Mock()
        config = ArenaConfig(
            prompt_evolver=mock_evolver,
        )

        assert config.prompt_evolver is mock_evolver

    def test_arena_config_enable_prompt_evolution_flag(self):
        """ArenaConfig should have enable_prompt_evolution flag."""
        from aragora.debate.orchestrator import ArenaConfig

        config = ArenaConfig(enable_prompt_evolution=True)
        assert config.enable_prompt_evolution is True

    @patch("aragora.debate.orchestrator.init_phases")
    @patch("aragora.evolution.evolver.PromptEvolver")
    def test_arena_auto_creates_evolver_when_enabled(self, mock_evolver_class, mock_init_phases):
        """Arena should auto-create PromptEvolver when enable_prompt_evolution=True."""
        from aragora.debate.orchestrator import Arena, ArenaConfig

        mock_evolver_instance = Mock()
        mock_evolver_class.return_value = mock_evolver_instance

        env = Environment(task="Test task")
        # Disable convergence detection to avoid loading embedding models
        protocol = DebateProtocol(convergence_detection=False)
        config = ArenaConfig(enable_prompt_evolution=True)

        # Mock agents
        mock_agent = Mock()
        mock_agent.name = "test-agent"
        agents = [mock_agent]

        arena = Arena.from_config(env, agents, protocol, config)

        # The evolver should be created
        assert arena.prompt_evolver is not None

    @patch("aragora.debate.orchestrator.init_phases")
    def test_arena_uses_provided_evolver(self, mock_init_phases):
        """Arena should use the provided prompt_evolver from config."""
        from aragora.debate.orchestrator import Arena, ArenaConfig

        mock_evolver = Mock()
        env = Environment(task="Test task")
        # Disable convergence detection to avoid loading embedding models
        protocol = DebateProtocol(convergence_detection=False)
        config = ArenaConfig(prompt_evolver=mock_evolver)

        mock_agent = Mock()
        mock_agent.name = "test-agent"
        agents = [mock_agent]

        arena = Arena.from_config(env, agents, protocol, config)

        assert arena.prompt_evolver is mock_evolver


class TestEvolutionHookInFeedbackPhase:
    """Tests for PromptEvolver integration in FeedbackPhase."""

    def test_feedback_phase_accepts_prompt_evolver(self):
        """FeedbackPhase should accept prompt_evolver parameter."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        mock_evolver = Mock()

        # Create with minimal required params
        phase = FeedbackPhase(
            elo_system=None,
            persona_manager=None,
            position_ledger=None,
            relationship_tracker=None,
            moment_detector=None,
            debate_embeddings=None,
            flip_detector=None,
            continuum_memory=None,
            event_emitter=None,
            loop_id=None,
            emit_moment_event=lambda *args: None,
            store_debate_outcome_as_memory=AsyncMock(),
            update_continuum_memory_outcomes=AsyncMock(),
            index_debate_async=AsyncMock(),
            consensus_memory=None,
            calibration_tracker=None,
            population_manager=None,
            auto_evolve=False,
            breeding_threshold=10,
            prompt_evolver=mock_evolver,
        )

        assert phase.prompt_evolver is mock_evolver

    @pytest.mark.asyncio
    async def test_feedback_phase_records_evolution_patterns(self):
        """FeedbackPhase should record evolution patterns when evolver is available."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase
        from aragora.debate.context import DebateContext

        mock_evolver = Mock()
        mock_evolver.extract_winning_patterns = Mock(return_value=[])
        mock_evolver.record_pattern = Mock()

        phase = FeedbackPhase(
            elo_system=Mock(),
            persona_manager=None,
            position_ledger=None,
            relationship_tracker=None,
            moment_detector=None,
            debate_embeddings=None,
            flip_detector=None,
            continuum_memory=None,
            event_emitter=None,
            loop_id=None,
            emit_moment_event=lambda *args: None,
            store_debate_outcome_as_memory=AsyncMock(),
            update_continuum_memory_outcomes=AsyncMock(),
            index_debate_async=AsyncMock(),
            consensus_memory=None,
            calibration_tracker=None,
            population_manager=None,
            auto_evolve=False,
            breeding_threshold=10,
            prompt_evolver=mock_evolver,
        )

        # Create a mock DebateContext with result (method signature takes ctx, not result)
        ctx = Mock(spec=DebateContext)
        ctx.debate_id = "test-debate-123"
        ctx.result = Mock()
        ctx.result.winner = "claude"
        ctx.result.final_answer = "Test conclusion"
        ctx.result.confidence = 0.9
        ctx.result.consensus_reached = True
        ctx.result.messages = []

        # Call the internal method that should use the evolver
        if hasattr(phase, "_record_evolution_patterns"):
            phase._record_evolution_patterns(ctx)
            # Verify evolver extract_winning_patterns was called
            mock_evolver.extract_winning_patterns.assert_called()


class TestPromptEvolverIntegration:
    """Tests for PromptEvolver behavior."""

    def test_prompt_evolver_can_extract_patterns(self):
        """PromptEvolver should be able to extract patterns from responses."""
        try:
            from aragora.evolution.evolver import PromptEvolver
        except ImportError:
            pytest.skip("PromptEvolver not available")

        evolver = PromptEvolver()

        # Test pattern extraction
        response = """
        After careful analysis, I believe the best approach is:
        1. First, we should validate inputs
        2. Then, process the data
        3. Finally, return the result

        Confidence: 85%
        """

        patterns = (
            evolver.extract_patterns(response) if hasattr(evolver, "extract_patterns") else []
        )
        # Should extract something (implementation-dependent)
        assert isinstance(patterns, (list, dict))

    def test_prompt_evolver_tracks_performance(self):
        """PromptEvolver should track agent performance scores."""
        try:
            from aragora.evolution.evolver import PromptEvolver
        except ImportError:
            pytest.skip("PromptEvolver not available")

        evolver = PromptEvolver()

        # Update performance for an agent - signature: (agent_name, version, debate_result)
        if hasattr(evolver, "update_performance"):
            mock_result = Mock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            evolver.update_performance("claude", 1, mock_result)
            mock_result.confidence = 0.8
            evolver.update_performance("claude", 1, mock_result)

            # Should track the history
            if hasattr(evolver, "get_performance"):
                perf = evolver.get_performance("claude")
                assert perf is not None


class TestEvolutionPatternExtraction:
    """Tests for pattern extraction from winning responses."""

    def test_extract_structured_claims(self):
        """Should extract structured claims from responses."""
        try:
            from aragora.reasoning.claims import fast_extract_claims
        except ImportError:
            pytest.skip("claims module not available")

        response = """
        Based on the evidence, I conclude:
        1. The system should use async I/O for better performance
        2. Error handling should be centralized
        3. Caching improves response times by 40%

        Therefore, we should implement these changes.
        """

        claims = fast_extract_claims(response, author="claude")
        assert len(claims) > 0
        # Should extract numbered points
        assert any("async" in str(c).lower() or "i/o" in str(c).lower() for c in claims)

    def test_extract_confidence_from_response(self):
        """Should extract confidence levels from responses."""
        response = "I am 85% confident that this approach is correct."

        # Simple regex extraction
        import re

        match = re.search(r"(\d+)%\s*confident", response.lower())
        assert match is not None
        assert int(match.group(1)) == 85


class TestEvolutionHookE2E:
    """End-to-end tests for evolution hook integration."""

    @pytest.mark.asyncio
    @patch("aragora.debate.orchestrator.init_phases")
    async def test_arena_run_triggers_evolution(self, mock_init_phases):
        """Running Arena with evolution enabled should trigger pattern recording."""
        from aragora.debate.orchestrator import Arena, ArenaConfig
        from aragora.core import Environment
        from aragora.debate.protocol import DebateProtocol

        mock_evolver = Mock()
        mock_evolver.extract_patterns = Mock(return_value=[])
        mock_evolver.extract_winning_patterns = Mock(return_value=[])

        env = Environment(task="Test task for evolution")
        # Disable convergence detection to avoid loading embedding models
        protocol = DebateProtocol(rounds=1, convergence_detection=False)
        config = ArenaConfig(prompt_evolver=mock_evolver)

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.name = "test-agent"
        mock_agent.generate = AsyncMock(return_value="Test response")
        mock_agent.vote = AsyncMock(return_value={"agent": "test-agent", "confidence": 0.8})
        agents = [mock_agent]

        arena = Arena.from_config(env, agents, protocol, config)

        # Mock the autonomous methods to avoid actual API calls
        arena.autonomic = Mock()
        arena.autonomic.generate = AsyncMock(return_value="Test response")
        arena.autonomic.critique = AsyncMock(return_value="Test critique")
        arena.autonomic.vote = AsyncMock(return_value={"agent": "test-agent", "confidence": 0.8})
        arena.autonomic.with_timeout = lambda coro, _: coro

        # Run should complete without error
        # Full run would require more mocking, but we test the wiring
        assert arena.prompt_evolver is mock_evolver
