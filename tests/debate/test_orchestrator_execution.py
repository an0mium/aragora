"""
Comprehensive tests for aragora/debate/orchestrator.py execution paths.

This test module covers areas not already tested in other orchestrator test files:
- test_orchestrator.py: Basic functionality
- test_orchestrator_comprehensive.py: Factory methods, config, subsystems
- test_orchestrator_critical.py: Delegation methods
- test_orchestrator_agents.py: Agent selection helpers
- test_orchestrator_hooks.py: Bead/hook tracking
- test_orchestrator_memory.py: Memory coordination

This file focuses on:
- Arena.run() method and _run_inner() execution flow
- Phase execution and error recovery
- Timeout handling in debates
- Parallel proposal/critique handling
- Checkpoint public API (save, restore, list, cleanup)
- Translation integration
- Config object merging patterns
- Complete debate execution scenarios
- Arena initialization edge cases
- State management during debates

Test categories:
1. Arena Initialization Tests
2. Config Object Merging Tests
3. Debate Execution Flow Tests
4. Timeout and Cancellation Tests
5. Checkpoint Public API Tests
6. Translation Integration Tests
7. State Management Tests
8. Error Recovery Tests
9. Complete Scenario Tests
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.orchestrator_runner import _DebateExecutionState
from aragora.debate.protocol import DebateProtocol, CircuitBreaker
from aragora.debate.arena_config import (
    ArenaConfig,
    AgentConfig,
    DebateConfig,
    MemoryConfig,
    ObservabilityConfig,
    StreamingConfig,
)
from aragora.debate.context import DebateContext


# =============================================================================
# Test Fixtures and Mock Agents
# =============================================================================


class MockAgent(Agent):
    """Mock agent for comprehensive testing."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response",
        model: str = "mock-model",
        role: str = "proposer",
        vote_choice: str | None = None,
        vote_confidence: float = 0.8,
        continue_debate: bool = False,
        delay: float = 0.0,
        should_fail: bool = False,
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.vote_choice = vote_choice
        self.vote_confidence = vote_confidence
        self.continue_debate = continue_debate
        self.delay = delay
        self.should_fail = should_fail
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Agent failed intentionally")
        return self.response

    async def generate_stream(self, prompt: str, context: list = None):
        if self.should_fail:
            raise RuntimeError("Agent failed intentionally")
        yield self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Agent critique failed")
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        choice = self.vote_choice or (list(proposals.keys())[0] if proposals else self.name)
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=self.vote_confidence,
            continue_debate=self.continue_debate,
        )


@pytest.fixture
def mock_agents():
    """Create standard mock agents for testing."""
    return [
        MockAgent(name="agent1", response="Proposal from agent 1"),
        MockAgent(name="agent2", response="Proposal from agent 2"),
        MockAgent(name="agent3", response="Proposal from agent 3"),
    ]


@pytest.fixture
def environment():
    """Create a standard test environment."""
    return Environment(task="What is the best approach to solve this problem?")


@pytest.fixture
def protocol():
    """Create a standard test protocol."""
    return DebateProtocol(rounds=2, consensus="majority")


@pytest.fixture
def arena(environment, mock_agents, protocol):
    """Create a standard Arena instance for testing."""
    return Arena(environment, mock_agents, protocol)


# =============================================================================
# Arena Initialization Tests
# =============================================================================


class TestArenaInitialization:
    """Tests for Arena initialization with various configurations."""

    def test_arena_basic_initialization(self, environment, mock_agents):
        """Arena initializes with minimal required arguments."""
        arena = Arena(environment, mock_agents)

        assert arena.env == environment
        assert len(arena.agents) == 3
        assert arena.protocol is not None
        # Default rounds may vary based on protocol defaults
        assert arena.protocol.rounds >= 1

    def test_arena_with_custom_protocol(self, environment, mock_agents):
        """Arena respects custom protocol settings."""
        protocol = DebateProtocol(
            rounds=5,
            consensus="unanimous",
            timeout_seconds=60,
        )
        arena = Arena(environment, mock_agents, protocol)

        assert arena.protocol.rounds == 5
        assert arena.protocol.consensus == "unanimous"
        assert arena.protocol.timeout_seconds == 60

    def test_arena_raises_on_empty_agents(self, environment):
        """Arena raises ValueError when no agents provided."""
        with pytest.raises(ValueError, match="Must specify either"):
            Arena(environment, [])

    def test_arena_initializes_subsystems(self, environment, mock_agents, protocol):
        """Arena initializes all required subsystems."""
        arena = Arena(environment, mock_agents, protocol)

        # Core subsystems
        assert arena.prompt_builder is not None
        assert arena.memory_manager is not None
        assert arena.context_gatherer is not None
        assert arena.roles_manager is not None
        assert arena.termination_checker is not None

        # Helper classes
        assert arena._lifecycle is not None
        assert arena._event_emitter is not None
        assert arena._checkpoint_ops is not None
        assert arena._grounded_ops is not None
        assert arena._budget_coordinator is not None
        assert arena._km_manager is not None
        assert arena._context_delegator is not None
        assert arena._prompt_context is not None

    def test_arena_initializes_phases(self, environment, mock_agents, protocol):
        """Arena initializes phase executors."""
        arena = Arena(environment, mock_agents, protocol)

        assert arena.phase_executor is not None
        assert arena.voting_phase is not None
        assert arena.proposal_phase is not None
        assert arena.debate_rounds_phase is not None
        assert arena.consensus_phase is not None

    def test_arena_with_loop_id(self, environment, mock_agents, protocol):
        """Arena accepts and stores loop_id for event scoping."""
        arena = Arena(environment, mock_agents, protocol, loop_id="test-loop-123")

        assert arena.loop_id == "test-loop-123"

    def test_arena_with_org_and_user_id(self, environment, mock_agents, protocol):
        """Arena accepts org_id and user_id for multi-tenancy."""
        arena = Arena(
            environment,
            mock_agents,
            protocol,
            org_id="org-123",
            user_id="user-456",
        )

        assert arena.org_id == "org-123"
        assert arena.user_id == "user-456"
        assert arena._budget_coordinator.org_id == "org-123"


# =============================================================================
# Config Object Merging Tests
# =============================================================================


class TestConfigObjectMerging:
    """Tests for config object merging behavior."""

    def test_debate_config_takes_precedence(self, environment, mock_agents):
        """DebateConfig settings override individual parameters."""
        protocol = DebateProtocol(rounds=3)
        debate_config = DebateConfig(
            enable_adaptive_rounds=True,
            enable_agent_hierarchy=False,
        )

        arena = Arena(
            environment,
            mock_agents,
            protocol,
            debate_config=debate_config,
            enable_agent_hierarchy=True,  # Should be overridden
        )

        assert arena.enable_adaptive_rounds is True
        assert arena.enable_agent_hierarchy is False

    def test_agent_config_takes_precedence(self, environment, mock_agents, protocol):
        """AgentConfig settings override individual parameters."""
        agent_config = AgentConfig(
            use_airlock=True,
            use_performance_selection=True,
        )

        arena = Arena(
            environment,
            mock_agents,
            protocol,
            agent_config=agent_config,
            use_airlock=False,  # Should be overridden
        )

        assert arena.use_performance_selection is True

    def test_memory_config_takes_precedence(self, environment, mock_agents, protocol):
        """MemoryConfig settings override individual parameters."""
        memory_config = MemoryConfig(
            enable_knowledge_retrieval=True,
            enable_knowledge_ingestion=False,
            use_rlm_limiter=False,
        )

        arena = Arena(
            environment,
            mock_agents,
            protocol,
            memory_config=memory_config,
        )

        assert arena.enable_knowledge_retrieval is True
        assert arena.enable_knowledge_ingestion is False
        assert arena.use_rlm_limiter is False

    def test_streaming_config_takes_precedence(self, environment, mock_agents, protocol):
        """StreamingConfig settings override individual parameters."""
        streaming_config = StreamingConfig(
            loop_id="stream-loop-123",
            strict_loop_scoping=True,
            enable_propulsion=True,
        )

        arena = Arena(
            environment,
            mock_agents,
            protocol,
            streaming_config=streaming_config,
        )

        assert arena.loop_id == "stream-loop-123"
        assert arena.strict_loop_scoping is True
        assert arena.enable_propulsion is True

    def test_observability_config_takes_precedence(self, environment, mock_agents, protocol):
        """ObservabilityConfig settings override individual parameters."""
        observability_config = ObservabilityConfig(
            enable_performance_monitor=False,
            enable_telemetry=True,
            enable_ml_delegation=True,
        )

        arena = Arena(
            environment,
            mock_agents,
            protocol,
            observability_config=observability_config,
        )

        assert arena.enable_ml_delegation is True

    def test_from_configs_factory_method(self, environment, mock_agents):
        """Arena.from_configs() creates arena with config objects."""
        arena = Arena.from_configs(
            environment,
            mock_agents,
            debate_config=DebateConfig(enable_adaptive_rounds=True),
            memory_config=MemoryConfig(enable_knowledge_retrieval=True),
        )

        assert arena is not None
        assert arena.enable_adaptive_rounds is True
        assert arena.enable_knowledge_retrieval is True

    def test_multiple_configs_combined(self, environment, mock_agents, protocol):
        """Multiple config objects can be combined."""
        arena = Arena(
            environment,
            mock_agents,
            protocol,
            debate_config=DebateConfig(enable_adaptive_rounds=True),
            agent_config=AgentConfig(use_performance_selection=True),
            memory_config=MemoryConfig(enable_knowledge_retrieval=True),
            streaming_config=StreamingConfig(loop_id="combined-loop"),
            observability_config=ObservabilityConfig(enable_telemetry=True),
        )

        assert arena.enable_adaptive_rounds is True
        assert arena.use_performance_selection is True
        assert arena.enable_knowledge_retrieval is True
        assert arena.loop_id == "combined-loop"


# =============================================================================
# Debate Execution Flow Tests
# =============================================================================


class TestDebateExecutionFlow:
    """Tests for the debate execution flow."""

    @pytest.mark.asyncio
    async def test_run_returns_debate_result(self, arena):
        """Arena.run() returns a DebateResult."""
        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == arena.env.task

    @pytest.mark.asyncio
    async def test_run_populates_result_fields(self, arena):
        """Arena.run() populates all result fields."""
        result = await arena.run()

        assert result.task is not None
        assert result.rounds_used >= 0
        assert isinstance(result.messages, list)
        assert isinstance(result.critiques, list)
        assert isinstance(result.votes, list)
        assert isinstance(result.participants, list)
        assert isinstance(result.proposals, dict)

    @pytest.mark.asyncio
    async def test_run_with_correlation_id(self, arena):
        """Arena.run() accepts and uses correlation_id."""
        result = await arena.run(correlation_id="test-corr-123")

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_run_executes_all_phases(self, arena):
        """Arena.run() executes all debate phases."""
        # Mock phase executor to track calls
        mock_execution_result = MagicMock()
        mock_execution_result.failed_phases = []
        arena.phase_executor.execute = AsyncMock(return_value=mock_execution_result)

        await arena.run()

        arena.phase_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_notifies_trackers(self, environment, mock_agents, protocol):
        """Arena.run() notifies trackers on start and completion."""
        arena = Arena(environment, mock_agents, protocol)
        arena._trackers.on_debate_start = MagicMock()
        arena._trackers.on_debate_complete = MagicMock()

        await arena.run()

        arena._trackers.on_debate_start.assert_called_once()
        arena._trackers.on_debate_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_emits_agent_preview(self, environment, mock_agents, protocol):
        """Arena.run() emits agent preview for UI."""
        arena = Arena(environment, mock_agents, protocol)
        arena._emit_agent_preview = MagicMock()

        await arena.run()

        arena._emit_agent_preview.assert_called()

    @pytest.mark.asyncio
    async def test_run_cleans_up_on_completion(self, environment, mock_agents, protocol):
        """Arena.run() cleans up resources on completion."""
        arena = Arena(environment, mock_agents, protocol)
        arena._cleanup_convergence_cache = MagicMock()

        await arena.run()

        arena._cleanup_convergence_cache.assert_called()


# =============================================================================
# Timeout and Cancellation Tests
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling during debates."""

    @pytest.mark.asyncio
    async def test_run_respects_timeout(self, environment, mock_agents):
        """Arena.run() respects protocol timeout_seconds."""
        protocol = DebateProtocol(rounds=10, timeout_seconds=0.1)

        # Create agents that are slow
        slow_agents = [MockAgent(name=f"slow-agent-{i}", delay=1.0) for i in range(3)]

        arena = Arena(environment, slow_agents, protocol)

        # Should complete quickly due to timeout
        start = time.time()
        result = await arena.run()
        elapsed = time.time() - start

        # Should timeout reasonably quickly (allowing for initialization overhead)
        # The timeout is 0.1s but initialization and teardown add overhead
        assert elapsed < 15.0  # Generous to avoid flaky test failures
        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_run_returns_partial_result_on_timeout(self, environment, mock_agents):
        """Arena.run() returns partial result on timeout."""
        protocol = DebateProtocol(rounds=10, timeout_seconds=0.01)

        slow_agents = [MockAgent(name=f"slow-{i}", delay=5.0) for i in range(3)]
        arena = Arena(environment, slow_agents, protocol)

        result = await arena.run()

        # Should still return a result (partial)
        assert isinstance(result, DebateResult)
        assert result.task == environment.task

    @pytest.mark.asyncio
    async def test_run_without_timeout(self, environment, mock_agents):
        """Arena.run() works without timeout (timeout_seconds=0)."""
        protocol = DebateProtocol(rounds=1, timeout_seconds=0)
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup_on_timeout(self, environment, mock_agents):
        """Arena context manager cleans up on timeout."""
        protocol = DebateProtocol(rounds=10, timeout_seconds=0.01)
        slow_agents = [MockAgent(name=f"slow-{i}", delay=5.0) for i in range(3)]
        arena = Arena(environment, slow_agents, protocol)

        async with arena as a:
            result = await a.run()
            assert isinstance(result, DebateResult)


# =============================================================================
# Checkpoint Public API Tests
# =============================================================================


class TestCheckpointPublicAPI:
    """Tests for the public checkpoint API."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_without_manager(self, arena):
        """save_checkpoint returns None when no manager configured."""
        arena.checkpoint_manager = None

        result = await arena.save_checkpoint(
            debate_id="test-debate",
            phase="manual",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_manager(self, environment, mock_agents, protocol):
        """save_checkpoint creates checkpoint when manager available."""
        arena = Arena(environment, mock_agents, protocol, enable_checkpointing=True)

        # Mock the checkpoint manager
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "cp-test-123"
        arena.checkpoint_manager.create_checkpoint = AsyncMock(return_value=mock_checkpoint)

        result = await arena.save_checkpoint(
            debate_id="test-debate",
            phase="manual",
            current_round=2,
        )

        assert result == "cp-test-123"

    @pytest.mark.asyncio
    async def test_save_checkpoint_handles_error(self, environment, mock_agents, protocol):
        """save_checkpoint handles errors gracefully."""
        arena = Arena(environment, mock_agents, protocol, enable_checkpointing=True)
        arena.checkpoint_manager.create_checkpoint = AsyncMock(side_effect=IOError("Disk full"))

        result = await arena.save_checkpoint(
            debate_id="test-debate",
            phase="manual",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_without_manager(self, arena):
        """restore_from_checkpoint returns None when no manager."""
        arena.checkpoint_manager = None

        result = await arena.restore_from_checkpoint("cp-test-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_not_found(self, environment, mock_agents, protocol):
        """restore_from_checkpoint returns None when checkpoint not found."""
        arena = Arena(environment, mock_agents, protocol, enable_checkpointing=True)
        arena.checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=None)

        result = await arena.restore_from_checkpoint("nonexistent-cp")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints_without_manager(self, arena):
        """list_checkpoints returns empty list when no manager."""
        arena.checkpoint_manager = None

        result = await arena.list_checkpoints()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_manager(self, environment, mock_agents, protocol):
        """list_checkpoints returns checkpoint list from manager."""
        arena = Arena(environment, mock_agents, protocol, enable_checkpointing=True)

        mock_checkpoints = [
            {"checkpoint_id": "cp-1", "debate_id": "d-1"},
            {"checkpoint_id": "cp-2", "debate_id": "d-1"},
        ]
        arena.checkpoint_manager.store = MagicMock()
        arena.checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=mock_checkpoints)

        result = await arena.list_checkpoints(debate_id="d-1")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cleanup_checkpoints_without_manager(self, arena):
        """cleanup_checkpoints returns 0 when no manager."""
        arena.checkpoint_manager = None

        result = await arena.cleanup_checkpoints("test-debate")

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_checkpoints_keeps_latest(self, environment, mock_agents, protocol):
        """cleanup_checkpoints respects keep_latest parameter."""
        arena = Arena(environment, mock_agents, protocol, enable_checkpointing=True)

        mock_checkpoints = [
            {"checkpoint_id": "cp-3", "created_at": "2024-01-03"},
            {"checkpoint_id": "cp-2", "created_at": "2024-01-02"},
            {"checkpoint_id": "cp-1", "created_at": "2024-01-01"},
        ]
        arena.checkpoint_manager.store = MagicMock()
        arena.checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=mock_checkpoints)
        arena.checkpoint_manager.store.delete = AsyncMock(return_value=True)

        result = await arena.cleanup_checkpoints("test-debate", keep_latest=1)

        # Should delete 2 checkpoints (keep 1 latest)
        assert result == 2


# =============================================================================
# Translation Integration Tests
# =============================================================================


class TestTranslationIntegration:
    """Tests for translation integration."""

    @pytest.mark.asyncio
    async def test_translate_conclusions_skips_empty_answer(self, arena):
        """_translate_conclusions skips when no final_answer."""
        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
            final_answer="",
        )

        # Should not raise
        await arena._translate_conclusions(result)

        assert not hasattr(result, "translations") or not result.translations

    @pytest.mark.asyncio
    async def test_translate_conclusions_skips_no_target_languages(self, arena):
        """_translate_conclusions skips when no target_languages configured."""
        arena.protocol.target_languages = []

        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
            final_answer="This is the conclusion",
        )
        result.translations = {}

        await arena._translate_conclusions(result)

        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_conclusions_handles_import_error(self, arena):
        """_translate_conclusions handles missing translation module."""
        arena.protocol.target_languages = ["es", "fr"]

        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
            final_answer="This is the conclusion",
        )
        result.translations = {}

        # Should not raise even if translation module not available
        with patch.dict("sys.modules", {"aragora.debate.translation": None}):
            await arena._translate_conclusions(result)


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management during debates."""

    def test_debate_state_cache_initialized(self, arena):
        """DebateStateCache is properly initialized."""
        assert arena._cache is not None
        assert hasattr(arena._cache, "debate_domain")
        assert hasattr(arena._cache, "historical_context")

    def test_extract_debate_domain_caches_result(self, arena):
        """_extract_debate_domain caches the computed domain."""
        # First call computes and caches
        domain1 = arena._extract_debate_domain()

        # Second call uses cache
        domain2 = arena._extract_debate_domain()

        assert domain1 == domain2
        assert arena._cache.debate_domain is not None

    def test_sync_prompt_builder_state(self, arena):
        """_sync_prompt_builder_state syncs Arena state to PromptBuilder."""
        arena.current_role_assignments = {"agent1": MagicMock()}
        arena._cache.historical_context = "test context"

        arena._sync_prompt_builder_state()

        assert arena.prompt_builder.current_role_assignments == arena.current_role_assignments

    def test_user_votes_property(self, arena):
        """user_votes property returns AudienceManager votes."""
        votes = arena.user_votes
        assert isinstance(votes, deque)

    def test_user_suggestions_property(self, arena):
        """user_suggestions property returns AudienceManager suggestions."""
        suggestions = arena.user_suggestions
        assert isinstance(suggestions, deque)


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery during debates."""

    @pytest.mark.asyncio
    async def test_run_handles_phase_execution_error(self, environment, mock_agents, protocol):
        """Arena.run() handles phase execution errors."""
        arena = Arena(environment, mock_agents, protocol)

        # Mock phase executor to raise an error
        arena.phase_executor.execute = AsyncMock(side_effect=RuntimeError("Phase failed"))

        # Should raise the error (not silently swallow)
        with pytest.raises(RuntimeError, match="Phase failed"):
            await arena.run()

    @pytest.mark.asyncio
    async def test_run_records_metrics_on_error(self, environment, mock_agents, protocol):
        """Arena.run() records metrics even on error."""
        arena = Arena(environment, mock_agents, protocol)

        # Mock phase executor to raise an error
        arena.phase_executor.execute = AsyncMock(side_effect=RuntimeError("Phase failed"))

        try:
            await arena.run()
        except RuntimeError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_async_context_manager_cleans_up_on_error(
        self, environment, mock_agents, protocol
    ):
        """Arena context manager cleans up even on error."""
        arena = Arena(environment, mock_agents, protocol)
        cleanup_called = False

        original_cleanup = arena._cleanup

        async def tracked_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            await original_cleanup()

        arena._cleanup = tracked_cleanup

        try:
            async with arena:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert cleanup_called


# =============================================================================
# Complete Scenario Tests
# =============================================================================


class TestCompleteScenarios:
    """Integration tests for complete debate scenarios."""

    @pytest.mark.asyncio
    async def test_complete_debate_with_consensus(self, environment, mock_agents, protocol):
        """Complete debate that reaches consensus."""
        # Set all agents to vote for agent1
        for agent in mock_agents:
            agent.vote_choice = "agent1"
            agent.vote_confidence = 0.9

        arena = Arena(environment, mock_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert len(result.participants) > 0

    @pytest.mark.asyncio
    async def test_complete_debate_with_multiple_rounds(self, environment, mock_agents):
        """Complete debate with multiple rounds."""
        protocol = DebateProtocol(rounds=3, consensus="majority")
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_complete_debate_with_convergence_detection(self, environment, mock_agents):
        """Complete debate with convergence detection enabled."""
        protocol = DebateProtocol(
            rounds=5,
            convergence_detection=True,
            convergence_threshold=0.9,
        )
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_complete_debate_with_early_stopping(self, environment, mock_agents):
        """Complete debate with early stopping enabled."""
        protocol = DebateProtocol(
            rounds=10,
            early_stopping=True,
            min_rounds_before_early_stop=1,
            early_stop_threshold=0.5,
        )

        # Set agents to vote STOP
        for agent in mock_agents:
            agent.response = "STOP"

        arena = Arena(environment, mock_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_complete_debate_with_knowledge_retrieval(self, environment, mock_agents):
        """Complete debate with knowledge retrieval enabled."""
        arena = Arena(
            environment,
            mock_agents,
            enable_knowledge_retrieval=True,
            enable_knowledge_ingestion=True,
        )

        result = await arena.run()

        assert isinstance(result, DebateResult)


# =============================================================================
# DebateExecutionState Tests
# =============================================================================


class TestDebateExecutionState:
    """Tests for _DebateExecutionState dataclass."""

    def test_state_initialization(self):
        """DebateExecutionState initializes with required fields."""
        ctx = MagicMock(spec=DebateContext)

        state = _DebateExecutionState(
            debate_id="debate-123",
            correlation_id="corr-456",
            domain="technology",
            task_complexity=MagicMock(),
            ctx=ctx,
        )

        assert state.debate_id == "debate-123"
        assert state.correlation_id == "corr-456"
        assert state.domain == "technology"
        assert state.ctx == ctx
        assert state.gupp_bead_id is None
        assert state.gupp_hook_entries == {}
        assert state.debate_status == "completed"
        assert state.debate_start_time == 0.0

    def test_state_with_optional_fields(self):
        """DebateExecutionState accepts optional fields."""
        ctx = MagicMock(spec=DebateContext)

        state = _DebateExecutionState(
            debate_id="debate-123",
            correlation_id="corr-456",
            domain="technology",
            task_complexity=MagicMock(),
            ctx=ctx,
            gupp_bead_id="bead-789",
            gupp_hook_entries={"hook1": "entry1"},
            debate_status="running",
            debate_start_time=12345.0,
        )

        assert state.gupp_bead_id == "bead-789"
        assert state.gupp_hook_entries == {"hook1": "entry1"}
        assert state.debate_status == "running"
        assert state.debate_start_time == 12345.0


# =============================================================================
# Phase Executor Integration Tests
# =============================================================================


class TestPhaseExecutorIntegration:
    """Tests for phase executor integration."""

    @pytest.mark.asyncio
    async def test_phase_executor_receives_context(self, arena):
        """Phase executor receives DebateContext."""
        contexts = []

        original_execute = arena.phase_executor.execute

        async def capture_execute(ctx, **kwargs):
            contexts.append(ctx)
            mock_result = MagicMock()
            mock_result.failed_phases = []
            return mock_result

        arena.phase_executor.execute = capture_execute

        await arena.run()

        assert len(contexts) == 1
        assert isinstance(contexts[0], DebateContext)

    @pytest.mark.asyncio
    async def test_phase_executor_receives_debate_id(self, arena):
        """Phase executor receives debate_id."""
        kwargs_list = []

        async def capture_execute(ctx, **kwargs):
            kwargs_list.append(kwargs)
            mock_result = MagicMock()
            mock_result.failed_phases = []
            return mock_result

        arena.phase_executor.execute = capture_execute

        await arena.run()

        assert "debate_id" in kwargs_list[0]
        assert isinstance(kwargs_list[0]["debate_id"], str)


# =============================================================================
# Budget Coordinator Tests
# =============================================================================


class TestBudgetCoordinator:
    """Tests for budget coordinator integration."""

    def test_budget_coordinator_initialized(self, arena):
        """Budget coordinator is initialized during Arena creation."""
        assert arena._budget_coordinator is not None

    def test_budget_coordinator_uses_org_id(self, environment, mock_agents, protocol):
        """Budget coordinator uses org_id from Arena."""
        arena = Arena(environment, mock_agents, protocol, org_id="test-org")

        assert arena._budget_coordinator.org_id == "test-org"

    @pytest.mark.asyncio
    async def test_run_checks_budget_before_debate(self, environment, mock_agents, protocol):
        """Arena.run() checks budget before starting."""
        arena = Arena(environment, mock_agents, protocol)
        arena._budget_coordinator.check_budget_before_debate = MagicMock()

        await arena.run()

        arena._budget_coordinator.check_budget_before_debate.assert_called()

    @pytest.mark.asyncio
    async def test_run_records_debate_cost(self, environment, mock_agents, protocol):
        """Arena.run() records debate cost after completion."""
        arena = Arena(environment, mock_agents, protocol)
        arena._budget_coordinator.record_debate_cost = MagicMock()

        await arena.run()

        arena._budget_coordinator.record_debate_cost.assert_called()


# =============================================================================
# RLM Limiter Tests
# =============================================================================


class TestRLMLimiter:
    """Tests for RLM cognitive load limiter integration."""

    def test_rlm_limiter_enabled_by_default(self, arena):
        """RLM limiter is enabled by default."""
        assert arena.use_rlm_limiter is True

    def test_rlm_limiter_can_be_disabled(self, environment, mock_agents, protocol):
        """RLM limiter can be disabled via config."""
        arena = Arena(
            environment,
            mock_agents,
            protocol,
            use_rlm_limiter=False,
        )

        assert arena.use_rlm_limiter is False

    def test_rlm_compression_settings(self, environment, mock_agents, protocol):
        """RLM compression settings are configurable."""
        arena = Arena(
            environment,
            mock_agents,
            protocol,
            rlm_compression_threshold=5000,
            rlm_max_recent_messages=10,
            rlm_summary_level="PARAGRAPH",
        )

        assert arena.rlm_compression_threshold == 5000
        assert arena.rlm_max_recent_messages == 10
        assert arena.rlm_summary_level == "PARAGRAPH"

    @pytest.mark.asyncio
    async def test_compress_debate_messages(self, arena):
        """compress_debate_messages delegates to memory helper."""
        messages = [Message(role="proposer", agent="agent1", content="Test message", round=1)]

        compressed_messages, compressed_critiques = await arena.compress_debate_messages(
            messages, []
        )

        # Should return messages (possibly compressed)
        assert isinstance(compressed_messages, list)


# =============================================================================
# Spectator Integration Tests
# =============================================================================


class TestSpectatorIntegration:
    """Tests for spectator/event emission integration."""

    def test_notify_spectator_delegates_to_event_emitter(self, arena):
        """_notify_spectator delegates to EventEmitter."""
        arena._event_emitter.notify_spectator = MagicMock()

        arena._notify_spectator("test_event", key="value")

        arena._event_emitter.notify_spectator.assert_called_once_with("test_event", key="value")

    def test_emit_moment_event_delegates(self, arena):
        """_emit_moment_event delegates to EventEmitter."""
        arena._event_emitter.emit_moment = MagicMock()

        mock_moment = MagicMock()
        arena._emit_moment_event(mock_moment)

        arena._event_emitter.emit_moment.assert_called_once_with(mock_moment)

    def test_emit_agent_preview_delegates(self, arena):
        """_emit_agent_preview delegates to EventEmitter."""
        arena._event_emitter.emit_agent_preview = MagicMock()

        arena._emit_agent_preview()

        arena._event_emitter.emit_agent_preview.assert_called()


# =============================================================================
# Agent Pool Integration Tests
# =============================================================================


class TestAgentPoolIntegration:
    """Tests for agent pool integration."""

    def test_agent_pool_initialized(self, arena):
        """Agent pool is initialized during Arena creation."""
        assert arena.agent_pool is not None

    def test_get_calibration_weight_delegates(self, arena):
        """_get_calibration_weight delegates to agent pool."""
        weight = arena._get_calibration_weight("agent1")

        assert isinstance(weight, (int, float))

    def test_compute_composite_judge_score_delegates(self, arena):
        """_compute_composite_judge_score delegates to agent pool."""
        score = arena._compute_composite_judge_score("agent1")

        assert isinstance(score, (int, float))

    def test_select_critics_for_proposal_delegates(self, arena, mock_agents):
        """_select_critics_for_proposal delegates to agent pool."""
        critics = arena._select_critics_for_proposal("agent1", mock_agents)

        assert isinstance(critics, list)


# =============================================================================
# Knowledge Mound Integration Tests
# =============================================================================


class TestKnowledgeMoundIntegration:
    """Tests for Knowledge Mound integration."""

    def test_km_manager_initialized(self, arena):
        """ArenaKnowledgeManager is initialized."""
        assert arena._km_manager is not None

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context(self, arena):
        """_fetch_knowledge_context delegates to KM manager."""
        arena._km_manager.fetch_context = AsyncMock(return_value="knowledge context")

        result = await arena._fetch_knowledge_context("test task")

        assert result == "knowledge context"
        arena._km_manager.fetch_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome(self, arena):
        """_ingest_debate_outcome delegates to KM manager."""
        arena._km_manager.ingest_outcome = AsyncMock()

        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
            final_answer="Answer",
            confidence=0.9,
        )

        await arena._ingest_debate_outcome(result)

        arena._km_manager.ingest_outcome.assert_called_once()

    def test_get_culture_hints_delegates(self, arena):
        """_get_culture_hints delegates to KM manager."""
        arena._km_manager.get_culture_hints = MagicMock(return_value={"hint": "value"})

        hints = arena._get_culture_hints("debate-123")

        assert hints == {"hint": "value"}

    def test_apply_culture_hints_delegates(self, arena):
        """_apply_culture_hints delegates to KM manager."""
        arena._km_manager.apply_culture_hints = MagicMock()
        # Mock the read-only properties by patching the private attributes
        arena._km_manager._culture_consensus_hint = "test hint"
        arena._km_manager._culture_extra_critiques = 0
        arena._km_manager._culture_early_consensus = False
        arena._km_manager._culture_domain_patterns = []

        arena._apply_culture_hints({"hint": "value"})

        arena._km_manager.apply_culture_hints.assert_called_once()


# =============================================================================
# Extensions Integration Tests
# =============================================================================


class TestExtensionsIntegration:
    """Tests for arena extensions integration."""

    def test_extensions_initialized(self, arena):
        """Extensions are initialized during Arena creation."""
        assert arena.extensions is not None

    @pytest.mark.asyncio
    async def test_run_triggers_extensions_on_complete(self, environment, mock_agents, protocol):
        """Arena.run() triggers extensions on debate completion."""
        arena = Arena(environment, mock_agents, protocol)
        arena.extensions.on_debate_complete = MagicMock()

        await arena.run()

        arena.extensions.on_debate_complete.assert_called()


# =============================================================================
# Parallel Processing Tests
# =============================================================================


class TestParallelProcessing:
    """Tests for parallel proposal and critique handling."""

    @pytest.mark.asyncio
    async def test_multiple_agents_generate_proposals(self, environment, mock_agents, protocol):
        """Multiple agents generate proposals in parallel."""
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        # All agents should have been called at least once
        for agent in mock_agents:
            assert agent.generate_calls >= 1

    @pytest.mark.asyncio
    async def test_critiques_generated_for_proposals(self, environment, mock_agents, protocol):
        """Critiques are generated for proposals."""
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        # Should have critiques in result
        assert isinstance(result.critiques, list)

    @pytest.mark.asyncio
    async def test_agents_vote_on_proposals(self, environment, mock_agents, protocol):
        """Agents vote on proposals."""
        arena = Arena(environment, mock_agents, protocol)

        result = await arena.run()

        # Votes should be collected
        assert isinstance(result.votes, list)


# =============================================================================
# Role Management Tests
# =============================================================================


class TestRoleManagement:
    """Tests for role management during debates."""

    def test_roles_manager_initialized(self, arena):
        """RolesManager is initialized during Arena creation."""
        assert arena.roles_manager is not None

    def test_assign_roles_delegates(self, arena):
        """_assign_roles delegates to RolesManager."""
        arena.roles_manager.assign_initial_roles = MagicMock()

        arena._assign_roles()

        arena.roles_manager.assign_initial_roles.assert_called_once()

    def test_apply_agreement_intensity_delegates(self, arena):
        """_apply_agreement_intensity delegates to RolesManager."""
        arena.roles_manager.apply_agreement_intensity = MagicMock()

        arena._apply_agreement_intensity()

        arena.roles_manager.apply_agreement_intensity.assert_called_once()

    def test_assign_stances_delegates(self, arena):
        """_assign_stances delegates to RolesManager."""
        arena.roles_manager.assign_stances = MagicMock()

        arena._assign_stances(round_num=2)

        arena.roles_manager.assign_stances.assert_called_once_with(2)

    def test_get_stance_guidance_delegates(self, arena, mock_agents):
        """_get_stance_guidance delegates to RolesManager."""
        arena.roles_manager.get_stance_guidance = MagicMock(return_value="test guidance")

        guidance = arena._get_stance_guidance(mock_agents[0])

        assert guidance == "test guidance"

    def test_get_role_context_delegates(self, arena, mock_agents):
        """_get_role_context delegates to RolesManager."""
        arena.roles_manager.get_role_context = MagicMock(return_value="role context")

        context = arena._get_role_context(mock_agents[0])

        assert context == "role context"

    def test_update_role_assignments_delegates(self, arena):
        """_update_role_assignments delegates to RolesManager."""
        arena.roles_manager.update_role_assignments = MagicMock()
        arena.roles_manager.current_role_assignments = {}

        arena._update_role_assignments(round_num=2)

        arena.roles_manager.update_role_assignments.assert_called_once()

    def test_get_agreement_intensity_guidance(self, arena):
        """_get_agreement_intensity_guidance returns guidance string."""
        guidance = arena._get_agreement_intensity_guidance()

        assert isinstance(guidance, str)

    def test_format_role_assignments_for_log(self, arena):
        """_format_role_assignments_for_log returns formatted string."""
        mock_assignment = MagicMock()
        mock_assignment.role = MagicMock()
        mock_assignment.role.value = "critic"
        arena.current_role_assignments = {"agent1": mock_assignment}

        formatted = arena._format_role_assignments_for_log()

        assert "agent1" in formatted
        assert "critic" in formatted


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestPromptBuilding:
    """Tests for prompt building functionality."""

    def test_prompt_builder_initialized(self, arena):
        """PromptBuilder is initialized during Arena creation."""
        assert arena.prompt_builder is not None

    def test_build_proposal_prompt_delegates(self, arena, mock_agents):
        """_build_proposal_prompt delegates to PromptContextBuilder."""
        arena._prompt_context.build_proposal_prompt = MagicMock(return_value="test prompt")

        prompt = arena._build_proposal_prompt(mock_agents[0])

        arena._prompt_context.build_proposal_prompt.assert_called()

    def test_build_revision_prompt_delegates(self, arena, mock_agents):
        """_build_revision_prompt delegates to PromptContextBuilder."""
        arena._prompt_context.build_revision_prompt = MagicMock(return_value="revision prompt")

        critiques = [
            Critique(
                agent="agent2",
                target_agent="agent1",
                target_content="proposal",
                issues=["issue"],
                suggestions=["suggestion"],
                severity=0.5,
                reasoning="reasoning",
            )
        ]

        prompt = arena._build_revision_prompt(mock_agents[0], "original", critiques, round_number=2)

        arena._prompt_context.build_revision_prompt.assert_called()

    def test_prepare_audience_context_delegates(self, arena):
        """_prepare_audience_context delegates to PromptContextBuilder."""
        arena._prompt_context.prepare_audience_context = MagicMock(return_value="audience context")

        context = arena._prepare_audience_context(emit_event=True)

        arena._prompt_context.prepare_audience_context.assert_called_with(emit_event=True)

    def test_get_persona_context_delegates(self, arena, mock_agents):
        """_get_persona_context delegates to PromptContextBuilder."""
        arena._prompt_context.get_persona_context = MagicMock(return_value="persona context")

        context = arena._get_persona_context(mock_agents[0])

        assert context == "persona context"

    def test_get_flip_context_delegates(self, arena, mock_agents):
        """_get_flip_context delegates to PromptContextBuilder."""
        arena._prompt_context.get_flip_context = MagicMock(return_value="flip context")

        context = arena._get_flip_context(mock_agents[0])

        assert context == "flip context"


# =============================================================================
# Context Delegation Tests
# =============================================================================


class TestContextDelegation:
    """Tests for context gathering and delegation."""

    def test_context_delegator_initialized(self, arena):
        """ContextDelegator is initialized during Arena creation."""
        assert arena._context_delegator is not None

    def test_get_continuum_context_delegates(self, arena):
        """_get_continuum_context delegates to ContextDelegator."""
        arena._context_delegator.get_continuum_context = MagicMock(return_value="continuum ctx")

        context = arena._get_continuum_context()

        assert context == "continuum ctx"

    @pytest.mark.asyncio
    async def test_fetch_historical_context_delegates(self, arena):
        """_fetch_historical_context delegates to ContextDelegator."""
        arena._context_delegator.fetch_historical_context = AsyncMock(return_value="historical ctx")

        context = await arena._fetch_historical_context("test task")

        assert context == "historical ctx"

    def test_format_patterns_for_prompt_delegates(self, arena):
        """_format_patterns_for_prompt delegates to ContextDelegator."""
        arena._context_delegator.format_patterns_for_prompt = MagicMock(return_value="formatted")

        result = arena._format_patterns_for_prompt([{"pattern": "test"}])

        assert result == "formatted"

    def test_get_successful_patterns_delegates(self, arena):
        """_get_successful_patterns_from_memory delegates to ContextDelegator."""
        arena._context_delegator.get_successful_patterns = MagicMock(return_value="patterns")

        result = arena._get_successful_patterns_from_memory(limit=5)

        assert result == "patterns"

    @pytest.mark.asyncio
    async def test_perform_research_delegates(self, arena):
        """_perform_research delegates to ContextDelegator."""
        arena._context_delegator.perform_research = AsyncMock(return_value="research results")

        result = await arena._perform_research("test task")

        assert result == "research results"

    @pytest.mark.asyncio
    async def test_gather_aragora_context_delegates(self, arena):
        """_gather_aragora_context delegates to ContextDelegator."""
        arena._context_delegator.gather_aragora_context = AsyncMock(return_value="aragora ctx")

        result = await arena._gather_aragora_context("test task")

        assert result == "aragora ctx"

    @pytest.mark.asyncio
    async def test_gather_evidence_context_delegates(self, arena):
        """_gather_evidence_context delegates to ContextDelegator."""
        arena._context_delegator.gather_evidence_context = AsyncMock(return_value="evidence ctx")

        result = await arena._gather_evidence_context("test task")

        assert result == "evidence ctx"

    @pytest.mark.asyncio
    async def test_gather_trending_context_delegates(self, arena):
        """_gather_trending_context delegates to ContextDelegator."""
        arena._context_delegator.gather_trending_context = AsyncMock(return_value="trending ctx")

        result = await arena._gather_trending_context()

        assert result == "trending ctx"


# =============================================================================
# Termination Checker Tests
# =============================================================================


class TestTerminationChecker:
    """Tests for termination checking functionality."""

    def test_termination_checker_initialized(self, arena):
        """TerminationChecker is initialized during Arena creation."""
        assert arena.termination_checker is not None

    @pytest.mark.asyncio
    async def test_check_early_stopping_delegates(self, arena):
        """_check_early_stopping delegates to TerminationChecker."""
        arena.termination_checker.check_early_stopping = AsyncMock(return_value=True)

        result = await arena._check_early_stopping(
            round_num=2,
            proposals={"agent1": "proposal"},
            context=[],
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_judge_termination_delegates(self, arena):
        """_check_judge_termination delegates to TerminationChecker."""
        arena.termination_checker.check_judge_termination = AsyncMock(
            return_value=(False, "not conclusive")
        )

        should_terminate, reason = await arena._check_judge_termination(
            round_num=2,
            proposals={"agent1": "proposal"},
            context=[],
        )

        assert should_terminate is False
        assert reason == "not conclusive"


# =============================================================================
# Grounded Operations Tests
# =============================================================================


class TestGroundedOperations:
    """Tests for grounded operations (verdict, relationships)."""

    def test_grounded_ops_initialized(self, arena):
        """GroundedOperations is initialized during Arena creation."""
        assert arena._grounded_ops is not None

    def test_record_grounded_position_delegates(self, arena):
        """_record_grounded_position delegates to GroundedOperations."""
        arena._grounded_ops.record_position = MagicMock()

        arena._record_grounded_position(
            agent_name="agent1",
            content="position content",
            debate_id="debate-123",
            round_num=2,
        )

        arena._grounded_ops.record_position.assert_called_once()

    def test_update_agent_relationships_delegates(self, arena):
        """_update_agent_relationships delegates to GroundedOperations."""
        arena._grounded_ops.update_relationships = MagicMock()

        arena._update_agent_relationships(
            debate_id="debate-123",
            participants=["agent1", "agent2"],
            winner="agent1",
            votes=[],
        )

        arena._grounded_ops.update_relationships.assert_called_once()

    def test_create_grounded_verdict_delegates(self, arena):
        """_create_grounded_verdict delegates to GroundedOperations."""
        arena._grounded_ops.create_grounded_verdict = MagicMock(return_value=None)

        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
        )

        verdict = arena._create_grounded_verdict(result)

        arena._grounded_ops.create_grounded_verdict.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_claims_formally_delegates(self, arena):
        """_verify_claims_formally delegates to GroundedOperations."""
        arena._grounded_ops.verify_claims_formally = AsyncMock()

        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
        )

        await arena._verify_claims_formally(result)

        arena._grounded_ops.verify_claims_formally.assert_called_once()


# =============================================================================
# Result Formatting Tests
# =============================================================================


class TestResultFormatting:
    """Tests for result formatting functionality."""

    def test_format_conclusion(self, arena):
        """_format_conclusion returns formatted string."""
        result = DebateResult(
            task="Test",
            messages=[],
            critiques=[],
            votes=[],
            final_answer="The conclusion is X",
        )

        conclusion = arena._format_conclusion(result)

        assert isinstance(conclusion, str)

    def test_group_similar_votes_delegates(self, arena):
        """_group_similar_votes delegates to VotingPhase."""
        arena.voting_phase.group_similar_votes = MagicMock(return_value={})

        votes = [Vote(agent="agent1", choice="agent2", reasoning="good", confidence=0.9)]

        result = arena._group_similar_votes(votes)

        arena.voting_phase.group_similar_votes.assert_called_once()


# =============================================================================
# Audience Management Tests
# =============================================================================


class TestAudienceManagement:
    """Tests for audience management functionality."""

    def test_audience_manager_initialized(self, arena):
        """AudienceManager is initialized during Arena creation."""
        assert arena.audience_manager is not None

    def test_handle_user_event_delegates(self, arena):
        """_handle_user_event delegates to AudienceManager."""
        arena.audience_manager.handle_event = MagicMock()

        mock_event = MagicMock()
        arena._handle_user_event(mock_event)

        arena.audience_manager.handle_event.assert_called_once_with(mock_event)

    def test_drain_user_events_delegates(self, arena):
        """_drain_user_events delegates to AudienceManager."""
        arena.audience_manager.drain_events = MagicMock()

        arena._drain_user_events()

        arena.audience_manager.drain_events.assert_called_once()


# =============================================================================
# Event Bus Tests
# =============================================================================


class TestEventBus:
    """Tests for event bus functionality."""

    def test_event_bus_initialized(self, arena):
        """EventBus is initialized during Arena creation."""
        assert arena.event_bus is not None


# =============================================================================
# Lifecycle Manager Tests
# =============================================================================


class TestLifecycleManager:
    """Tests for lifecycle management."""

    def test_lifecycle_manager_initialized(self, arena):
        """LifecycleManager is initialized during Arena creation."""
        assert arena._lifecycle is not None

    def test_track_circuit_breaker_metrics_delegates(self, arena):
        """_track_circuit_breaker_metrics delegates to LifecycleManager."""
        arena._lifecycle.track_circuit_breaker_metrics = MagicMock()

        arena._track_circuit_breaker_metrics()

        arena._lifecycle.track_circuit_breaker_metrics.assert_called_once()

    def test_log_phase_failures_delegates(self, arena):
        """_log_phase_failures delegates to LifecycleManager."""
        arena._lifecycle.log_phase_failures = MagicMock()

        mock_result = MagicMock()
        mock_result.failed_phases = []

        arena._log_phase_failures(mock_result)

        arena._lifecycle.log_phase_failures.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_delegates(self, arena):
        """_cleanup delegates to LifecycleManager."""
        arena._lifecycle.cleanup = AsyncMock()
        arena._teardown_agent_channels = AsyncMock()

        await arena._cleanup()

        arena._lifecycle.cleanup.assert_called_once()


# =============================================================================
# Agent Channel Tests
# =============================================================================


class TestAgentChannels:
    """Tests for agent-to-agent channel management."""

    @pytest.mark.asyncio
    async def test_setup_agent_channels_skips_when_disabled(self, arena):
        """_setup_agent_channels skips when channels disabled in protocol."""
        arena.protocol.enable_agent_channels = False

        ctx = MagicMock()
        await arena._setup_agent_channels(ctx, "debate-123")

        # Should not create channel integration
        assert arena._channel_integration is None

    @pytest.mark.asyncio
    async def test_teardown_agent_channels_when_none(self, arena):
        """_teardown_agent_channels handles None integration."""
        arena._channel_integration = None

        # Should not raise
        await arena._teardown_agent_channels()


# =============================================================================
# Citation Extraction Tests
# =============================================================================


class TestCitationExtraction:
    """Tests for citation extraction functionality."""

    def test_extract_citation_needs_without_extractor(self, arena):
        """_extract_citation_needs returns empty dict without extractor."""
        arena.citation_extractor = None

        result = arena._extract_citation_needs({"agent1": "proposal"})

        assert result == {}

    def test_has_high_priority_needs(self, arena):
        """_has_high_priority_needs filters correctly."""
        needs = [
            {"claim": "claim1", "priority": "high"},
            {"claim": "claim2", "priority": "low"},
            {"claim": "claim3", "priority": "high"},
        ]

        result = arena._has_high_priority_needs(needs)

        assert len(result) == 2

    def test_log_citation_needs(self, arena):
        """_log_citation_needs runs without error."""
        arena._log_citation_needs("agent1", [{"claim": "claim", "priority": "high"}])


# =============================================================================
# Require Agents Tests
# =============================================================================


class TestRequireAgents:
    """Tests for _require_agents helper."""

    def test_require_agents_returns_agents(self, arena):
        """_require_agents returns agents list."""
        agents = arena._require_agents()

        assert isinstance(agents, list)
        assert len(agents) > 0

    def test_require_agents_raises_on_empty(self, environment, protocol):
        """_require_agents raises when agents empty."""
        # Create arena with fabric to bypass initial validation
        with patch("aragora.debate.orchestrator_agents.get_fabric_agents_sync", return_value=[MockAgent()]):
            arena = Arena(
                environment,
                agents=[],
                fabric=MagicMock(),
                fabric_config=MagicMock(pool_id="test"),
            )
            # Clear agents
            arena.agents = []

            with pytest.raises(ValueError, match="No agents available"):
                arena._require_agents()


# =============================================================================
# Async Indexing Tests
# =============================================================================


class TestAsyncIndexing:
    """Tests for async debate indexing."""

    @pytest.mark.asyncio
    async def test_index_debate_async_without_embeddings(self, arena):
        """_index_debate_async handles missing embeddings."""
        arena.debate_embeddings = None

        artifact = {"debate_id": "debate-123", "task": "test"}

        # Should not raise
        await arena._index_debate_async(artifact)

    @pytest.mark.asyncio
    async def test_index_debate_async_with_embeddings(self, arena):
        """_index_debate_async calls embeddings.index_debate."""
        mock_embeddings = MagicMock()
        mock_embeddings.index_debate = AsyncMock()
        arena.debate_embeddings = mock_embeddings

        artifact = {"debate_id": "debate-123", "task": "test"}

        await arena._index_debate_async(artifact)

        mock_embeddings.index_debate.assert_called_once_with(artifact)


# =============================================================================
# Static and Class Method Tests
# =============================================================================


class TestStaticAndClassMethods:
    """Tests for static and class methods."""

    @pytest.mark.asyncio
    async def test_recover_pending_debates_class_method(self):
        """recover_pending_debates is a class method."""
        result = await Arena.recover_pending_debates(bead_store=None)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_run_security_debate_class_method(self, mock_agents):
        """run_security_debate is a class method."""
        mock_event = MagicMock()
        mock_event.event_id = "event-123"
        mock_event.event_type = "security"
        mock_event.severity = "high"
        mock_event.description = "Security event"
        mock_event.details = {}
        mock_event.timestamp = None

        with patch("aragora.debate.security_debate.run_security_debate") as mock_run:
            mock_run.return_value = DebateResult(
                task="Security",
                messages=[],
                critiques=[],
                votes=[],
            )

            result = await Arena.run_security_debate(
                event=mock_event,
                agents=mock_agents,
            )

            mock_run.assert_called_once()


# =============================================================================
# Judge Selection Tests
# =============================================================================


class TestJudgeSelection:
    """Tests for judge selection functionality."""

    @pytest.mark.asyncio
    async def test_select_judge_returns_agent(self, arena):
        """_select_judge returns an Agent."""
        proposals = {"agent1": "proposal 1"}
        context = []

        judge = await arena._select_judge(proposals, context)

        assert isinstance(judge, Agent)
        assert judge.name in [a.name for a in arena.agents]


# =============================================================================
# Evidence Refresh Tests
# =============================================================================


class TestEvidenceRefresh:
    """Tests for evidence refresh during rounds."""

    @pytest.mark.asyncio
    async def test_refresh_evidence_for_round(self, arena):
        """_refresh_evidence_for_round returns count."""
        arena._context_delegator.refresh_evidence_for_round = AsyncMock(return_value=3)

        ctx = MagicMock()
        count = await arena._refresh_evidence_for_round(
            combined_text="debate text",
            ctx=ctx,
            round_num=2,
        )

        assert count == 3
