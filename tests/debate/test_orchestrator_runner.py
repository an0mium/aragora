"""Tests for orchestrator_runner.py - Debate execution helper methods.

Tests cover:
- _DebateExecutionState dataclass creation and field validation
- initialize_debate_context: debate ID generation, convergence detector, KM init, BeliefNetwork
- setup_debate_infrastructure: logging, tracker notification, budget validation, GUPP hooks
- execute_debate_phases: PhaseExecutor integration, timeout handling, EarlyStopError handling
- record_debate_metrics: duration calculation, span attributes, outcome tracking
- handle_debate_completion: tracker notification, extensions, budget usage, KM ingestion
- cleanup_debate_resources: checkpoint cleanup, channel teardown
- Error handling and recovery scenarios
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, call, patch, PropertyMock

import pytest

from aragora.core import DebateResult, Environment, TaskComplexity
from aragora.debate.context import DebateContext
from aragora.debate.orchestrator_runner import (
    _DebateExecutionState,
    initialize_debate_context,
    setup_debate_infrastructure,
    execute_debate_phases,
    record_debate_metrics,
    handle_debate_completion,
    cleanup_debate_resources,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.model = "test-model"
    return agent


@pytest.fixture
def mock_agents():
    """Create a list of mock agents."""
    agents = []
    for i in range(3):
        agent = MagicMock()
        agent.name = f"agent-{i}"
        agent.model = f"model-{i}"
        agents.append(agent)
    return agents


@pytest.fixture
def mock_env():
    """Create a mock environment."""
    env = MagicMock(spec=Environment)
    env.task = "Test debate task for testing purposes"
    env.context = {}
    return env


@pytest.fixture
def mock_protocol():
    """Create a mock protocol."""
    protocol = MagicMock()
    protocol.enable_km_belief_sync = False
    protocol.enable_hook_tracking = False
    protocol.checkpoint_cleanup_on_success = True
    protocol.checkpoint_keep_on_success = 0
    protocol.enable_translation = False
    return protocol


@pytest.fixture
def mock_budget_coordinator():
    """Create a mock budget coordinator."""
    coordinator = MagicMock()
    coordinator.check_budget_before_debate = MagicMock()
    coordinator.check_budget_mid_debate = MagicMock()
    coordinator.record_debate_cost = MagicMock()
    return coordinator


@pytest.fixture
def mock_trackers():
    """Create a mock subsystem trackers."""
    trackers = MagicMock()
    trackers.on_debate_start = MagicMock()
    trackers.on_debate_complete = MagicMock()
    return trackers


@pytest.fixture
def mock_extensions():
    """Create a mock extensions."""
    extensions = MagicMock()
    extensions.on_debate_complete = MagicMock()
    return extensions


@pytest.fixture
def mock_phase_executor():
    """Create a mock phase executor."""
    executor = MagicMock()
    executor.execute = AsyncMock()
    return executor


@pytest.fixture
def mock_hook_manager():
    """Create a mock hook manager."""
    return MagicMock()


@pytest.fixture
def mock_span():
    """Create a mock OpenTelemetry span."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.record_exception = MagicMock()
    return span


@pytest.fixture
def mock_arena(
    mock_env,
    mock_agents,
    mock_protocol,
    mock_budget_coordinator,
    mock_trackers,
    mock_extensions,
    mock_phase_executor,
    mock_hook_manager,
):
    """Create a mock Arena with all required attributes."""
    arena = MagicMock()
    arena.env = mock_env
    arena.agents = mock_agents
    arena.protocol = mock_protocol
    arena.org_id = "test-org"
    arena.hook_manager = mock_hook_manager
    arena.molecule_orchestrator = None
    arena.checkpoint_bridge = None
    arena.prompt_builder = None
    arena.use_performance_selection = False
    arena.enable_auto_execution = False
    arena.enable_result_routing = False
    arena.phase_executor = mock_phase_executor
    arena.extensions = mock_extensions

    # Internal attributes
    arena._budget_coordinator = mock_budget_coordinator
    arena._trackers = mock_trackers
    arena._bead_store = None
    arena._hook_registry = None

    # Methods
    arena._reinit_convergence_for_debate = MagicMock()
    arena._extract_debate_domain = MagicMock(return_value="general")
    arena._init_km_context = AsyncMock()
    arena._get_culture_hints = MagicMock(return_value=None)
    arena._apply_culture_hints = MagicMock()
    arena._setup_belief_network = MagicMock()
    arena._select_debate_team = MagicMock(side_effect=lambda agents: agents)
    arena._assign_hierarchy_roles = MagicMock()
    arena._setup_agent_channels = AsyncMock()
    arena._emit_agent_preview = MagicMock()
    arena._create_pending_debate_bead = AsyncMock(return_value=None)
    arena._init_hook_tracking = AsyncMock(return_value={})
    arena._log_phase_failures = MagicMock()
    arena._track_circuit_breaker_metrics = MagicMock()
    arena._ingest_debate_outcome = AsyncMock()
    arena._update_debate_bead = AsyncMock()
    arena._complete_hook_tracking = AsyncMock()
    arena._create_debate_bead = AsyncMock(return_value=None)
    arena._queue_for_supabase_sync = MagicMock()
    arena.cleanup_checkpoints = AsyncMock(return_value=0)
    arena._cleanup_convergence_cache = MagicMock()
    arena._teardown_agent_channels = AsyncMock()
    arena._translate_conclusions = AsyncMock()

    return arena


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock(spec=DebateResult)
    result.task = "Test task"
    result.consensus_reached = True
    result.confidence = 0.85
    result.messages = [MagicMock(), MagicMock()]
    result.critiques = []
    result.votes = []
    result.rounds_used = 3
    result.final_answer = "Test answer"
    result.bead_id = None
    return result


@pytest.fixture
def mock_debate_context(mock_env, mock_agents, mock_debate_result):
    """Create a mock DebateContext."""
    ctx = MagicMock(spec=DebateContext)
    ctx.env = mock_env
    ctx.agents = mock_agents
    ctx.debate_id = "test-debate-123"
    ctx.correlation_id = "corr-test"
    ctx.domain = "general"
    ctx.result = mock_debate_result
    ctx.partial_messages = []
    ctx.partial_critiques = []
    ctx.partial_rounds = 0
    ctx.finalize_result = MagicMock(return_value=mock_debate_result)
    return ctx


@pytest.fixture
def execution_state(mock_debate_context):
    """Create a _DebateExecutionState for testing."""
    return _DebateExecutionState(
        debate_id="test-debate-123",
        correlation_id="corr-test",
        domain="general",
        task_complexity=TaskComplexity.MODERATE,
        ctx=mock_debate_context,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 5.0,  # 5 seconds ago
    )


# =============================================================================
# Tests for _DebateExecutionState
# =============================================================================


class TestDebateExecutionState:
    """Tests for _DebateExecutionState dataclass."""

    def test_creation_with_required_fields(self, mock_debate_context):
        """Test creating state with required fields only."""
        state = _DebateExecutionState(
            debate_id="debate-1",
            correlation_id="corr-1",
            domain="tech",
            task_complexity=TaskComplexity.SIMPLE,
            ctx=mock_debate_context,
        )
        assert state.debate_id == "debate-1"
        assert state.correlation_id == "corr-1"
        assert state.domain == "tech"
        assert state.task_complexity == TaskComplexity.SIMPLE
        assert state.ctx is mock_debate_context

    def test_default_values(self, mock_debate_context):
        """Test default values for optional fields."""
        state = _DebateExecutionState(
            debate_id="debate-1",
            correlation_id="corr-1",
            domain="general",
            task_complexity=TaskComplexity.MODERATE,
            ctx=mock_debate_context,
        )
        assert state.gupp_bead_id is None
        assert state.gupp_hook_entries == {}
        assert state.debate_status == "completed"
        assert state.debate_start_time == 0.0

    def test_gupp_fields(self, mock_debate_context):
        """Test GUPP tracking fields."""
        state = _DebateExecutionState(
            debate_id="debate-1",
            correlation_id="corr-1",
            domain="general",
            task_complexity=TaskComplexity.MODERATE,
            ctx=mock_debate_context,
            gupp_bead_id="bead-123",
            gupp_hook_entries={"agent-1": "entry-1"},
        )
        assert state.gupp_bead_id == "bead-123"
        assert state.gupp_hook_entries == {"agent-1": "entry-1"}

    def test_status_modification(self, mock_debate_context):
        """Test that status can be modified."""
        state = _DebateExecutionState(
            debate_id="debate-1",
            correlation_id="corr-1",
            domain="general",
            task_complexity=TaskComplexity.MODERATE,
            ctx=mock_debate_context,
        )
        state.debate_status = "timeout"
        assert state.debate_status == "timeout"

    def test_start_time_modification(self, mock_debate_context):
        """Test that start time can be set."""
        state = _DebateExecutionState(
            debate_id="debate-1",
            correlation_id="corr-1",
            domain="general",
            task_complexity=TaskComplexity.MODERATE,
            ctx=mock_debate_context,
        )
        state.debate_start_time = 12345.67
        assert state.debate_start_time == 12345.67


# =============================================================================
# Tests for initialize_debate_context
# =============================================================================


class TestInitializeDebateContext:
    """Tests for initialize_debate_context function."""

    @pytest.mark.asyncio
    async def test_generates_debate_id(self, mock_arena):
        """Test that a unique debate ID is generated."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        assert state.debate_id is not None
        assert len(state.debate_id) == 36  # UUID format
        assert "-" in state.debate_id

    @pytest.mark.asyncio
    async def test_uses_provided_correlation_id(self, mock_arena):
        """Test that provided correlation ID is used."""
        state = await initialize_debate_context(mock_arena, "my-correlation-id")

        assert state.correlation_id == "my-correlation-id"

    @pytest.mark.asyncio
    async def test_generates_correlation_id_if_empty(self, mock_arena):
        """Test that correlation ID is generated if not provided."""
        state = await initialize_debate_context(mock_arena, "")

        assert state.correlation_id.startswith("corr-")
        assert state.debate_id[:8] in state.correlation_id

    @pytest.mark.asyncio
    async def test_reinitializes_convergence_detector(self, mock_arena):
        """Test that convergence detector is reinitialized for debate."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._reinit_convergence_for_debate.assert_called_once_with(state.debate_id)

    @pytest.mark.asyncio
    async def test_extracts_domain(self, mock_arena):
        """Test that debate domain is extracted."""
        mock_arena._extract_debate_domain.return_value = "finance"

        state = await initialize_debate_context(mock_arena, "corr-123")

        assert state.domain == "finance"
        mock_arena._extract_debate_domain.assert_called_once()

    @pytest.mark.asyncio
    async def test_initializes_km_context(self, mock_arena):
        """Test that Knowledge Mound context is initialized."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._init_km_context.assert_called_once_with(state.debate_id, state.domain)

    @pytest.mark.asyncio
    async def test_applies_culture_hints_when_available(self, mock_arena):
        """Test that culture hints are applied when present."""
        mock_arena._get_culture_hints.return_value = {"formality": "high"}

        await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._apply_culture_hints.assert_called_once_with({"formality": "high"})

    @pytest.mark.asyncio
    async def test_skips_culture_hints_when_none(self, mock_arena):
        """Test that culture hints application is skipped when None."""
        mock_arena._get_culture_hints.return_value = None

        await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._apply_culture_hints.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_debate_context(self, mock_arena):
        """Test that DebateContext is created with correct fields."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        ctx = state.ctx
        assert ctx.env is mock_arena.env
        assert ctx.agents is mock_arena.agents
        assert ctx.debate_id == state.debate_id
        assert ctx.correlation_id == "corr-123"
        assert ctx.domain == state.domain
        assert ctx.hook_manager is mock_arena.hook_manager
        assert ctx.org_id == mock_arena.org_id

    @pytest.mark.asyncio
    async def test_sets_molecule_orchestrator_on_context(self, mock_arena):
        """Test that molecule_orchestrator is set on context."""
        mock_arena.molecule_orchestrator = MagicMock()

        state = await initialize_debate_context(mock_arena, "corr-123")

        assert state.ctx.molecule_orchestrator is mock_arena.molecule_orchestrator

    @pytest.mark.asyncio
    async def test_sets_checkpoint_bridge_on_context(self, mock_arena):
        """Test that checkpoint_bridge is set on context."""
        mock_arena.checkpoint_bridge = MagicMock()

        state = await initialize_debate_context(mock_arena, "corr-123")

        assert state.ctx.checkpoint_bridge is mock_arena.checkpoint_bridge

    @pytest.mark.asyncio
    async def test_sets_up_belief_network_when_enabled(self, mock_arena):
        """Test that BeliefNetwork is set up when km_belief_sync is enabled."""
        mock_arena.protocol.enable_km_belief_sync = True
        mock_belief_network = MagicMock()
        mock_arena._setup_belief_network.return_value = mock_belief_network

        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._setup_belief_network.assert_called_once()
        call_kwargs = mock_arena._setup_belief_network.call_args[1]
        assert call_kwargs["debate_id"] == state.debate_id
        assert call_kwargs["topic"] == mock_arena.env.task
        assert call_kwargs["seed_from_km"] is True
        assert state.ctx.belief_network is mock_belief_network

    @pytest.mark.asyncio
    async def test_skips_belief_network_when_disabled(self, mock_arena):
        """Test that BeliefNetwork is not set up when disabled."""
        mock_arena.protocol.enable_km_belief_sync = False

        await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._setup_belief_network.assert_not_called()

    @pytest.mark.asyncio
    async def test_classifies_task_complexity(self, mock_arena):
        """Test that task complexity is classified."""
        mock_arena.env.task = "Please prove this theorem formally"

        state = await initialize_debate_context(mock_arena, "corr-123")

        # Complex keywords should result in complex classification
        assert state.task_complexity is not None

    @pytest.mark.asyncio
    async def test_classifies_question_with_prompt_builder(self, mock_arena):
        """Test that question classification is called when prompt_builder exists.

        The runner does a two-phase classification: fast heuristic first
        (use_llm=False), then a background LLM pass (use_llm=True).
        """
        mock_arena.prompt_builder = MagicMock()
        mock_arena.prompt_builder.classify_question_async = AsyncMock()

        with patch("aragora.utils.env.is_offline_mode", return_value=False):
            await initialize_debate_context(mock_arena, "corr-123")

        # First call is the fast heuristic, second is the background LLM task
        calls = mock_arena.prompt_builder.classify_question_async.call_args_list
        assert len(calls) >= 1
        assert calls[0] == call(use_llm=False)

    @pytest.mark.asyncio
    async def test_handles_question_classification_timeout(self, mock_arena):
        """Test that question classification timeout is handled gracefully."""
        mock_arena.prompt_builder = MagicMock()
        mock_arena.prompt_builder.classify_question_async = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        # Should not raise
        state = await initialize_debate_context(mock_arena, "corr-123")
        assert state.debate_id is not None

    @pytest.mark.asyncio
    async def test_handles_question_classification_value_error(self, mock_arena):
        """Test that question classification ValueError is handled gracefully."""
        mock_arena.prompt_builder = MagicMock()
        mock_arena.prompt_builder.classify_question_async = AsyncMock(
            side_effect=ValueError("Invalid input")
        )

        # Should not raise
        state = await initialize_debate_context(mock_arena, "corr-123")
        assert state.debate_id is not None

    @pytest.mark.asyncio
    async def test_applies_performance_based_selection_when_enabled(self, mock_arena):
        """Test that performance-based agent selection is applied when enabled."""
        mock_arena.use_performance_selection = True

        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._select_debate_team.assert_called_once()
        # Agents should be updated on both arena and context
        assert state.ctx.agents is mock_arena.agents

    @pytest.mark.asyncio
    async def test_skips_performance_selection_when_disabled(self, mock_arena):
        """Test that performance-based selection is skipped when disabled."""
        mock_arena.use_performance_selection = False

        await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._select_debate_team.assert_not_called()

    @pytest.mark.asyncio
    async def test_assigns_hierarchy_roles(self, mock_arena):
        """Test that hierarchy roles are assigned to agents."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._assign_hierarchy_roles.assert_called_once()
        call_args = mock_arena._assign_hierarchy_roles.call_args
        assert call_args[0][0] is state.ctx
        assert call_args[1]["task_type"] == state.domain

    @pytest.mark.asyncio
    async def test_sets_up_agent_channels(self, mock_arena):
        """Test that agent-to-agent channels are set up."""
        state = await initialize_debate_context(mock_arena, "corr-123")

        mock_arena._setup_agent_channels.assert_called_once_with(state.ctx, state.debate_id)


# =============================================================================
# Tests for setup_debate_infrastructure
# =============================================================================


class TestSetupDebateInfrastructure:
    """Tests for setup_debate_infrastructure function."""

    @pytest.mark.asyncio
    async def test_notifies_trackers_of_debate_start(self, mock_arena, execution_state):
        """Test that trackers are notified of debate start."""
        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._trackers.on_debate_start.assert_called_once_with(execution_state.ctx)

    @pytest.mark.asyncio
    async def test_emits_agent_preview(self, mock_arena, execution_state):
        """Test that agent preview is emitted."""
        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._emit_agent_preview.assert_called_once()

    @pytest.mark.asyncio
    async def test_checks_budget_before_debate(self, mock_arena, execution_state):
        """Test that budget is checked before debate starts."""
        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._budget_coordinator.check_budget_before_debate.assert_called_once()
        call_args = mock_arena._budget_coordinator.check_budget_before_debate.call_args
        assert call_args[0][0] == execution_state.debate_id

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self, mock_arena, execution_state):
        """Test that budget exceeded error propagates."""
        from aragora.exceptions import AragoraError

        mock_arena._budget_coordinator.check_budget_before_debate.side_effect = AragoraError(
            "Budget exceeded"
        )

        with pytest.raises(AragoraError, match="Budget exceeded"):
            await setup_debate_infrastructure(mock_arena, execution_state)

    @pytest.mark.asyncio
    async def test_initializes_gupp_tracking_when_enabled(self, mock_arena, execution_state):
        """Test that GUPP hook tracking is initialized when enabled."""
        mock_arena.protocol.enable_hook_tracking = True
        mock_arena._create_pending_debate_bead.return_value = "bead-456"
        mock_arena._init_hook_tracking.return_value = {"agent-1": "entry-1"}

        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._create_pending_debate_bead.assert_called_once_with(
            execution_state.debate_id, mock_arena.env.task
        )
        assert execution_state.gupp_bead_id == "bead-456"

    @pytest.mark.asyncio
    async def test_initializes_hook_entries_when_bead_created(self, mock_arena, execution_state):
        """Test that hook entries are initialized when bead is created."""
        mock_arena.protocol.enable_hook_tracking = True
        mock_arena._create_pending_debate_bead.return_value = "bead-789"
        mock_arena._init_hook_tracking.return_value = {"agent-0": "entry-0"}

        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._init_hook_tracking.assert_called_once_with(
            execution_state.debate_id, "bead-789"
        )
        assert execution_state.gupp_hook_entries == {"agent-0": "entry-0"}

    @pytest.mark.asyncio
    async def test_skips_gupp_when_disabled(self, mock_arena, execution_state):
        """Test that GUPP tracking is skipped when disabled."""
        mock_arena.protocol.enable_hook_tracking = False

        await setup_debate_infrastructure(mock_arena, execution_state)

        mock_arena._create_pending_debate_bead.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_gupp_initialization_error(self, mock_arena, execution_state):
        """Test that GUPP initialization errors are handled gracefully."""
        mock_arena.protocol.enable_hook_tracking = True
        mock_arena._create_pending_debate_bead.side_effect = RuntimeError("GUPP error")

        # Should not raise
        await setup_debate_infrastructure(mock_arena, execution_state)

        assert execution_state.gupp_bead_id is None

    @pytest.mark.asyncio
    async def test_creates_initial_result(self, mock_arena, execution_state):
        """Test that initial DebateResult is created on context."""
        await setup_debate_infrastructure(mock_arena, execution_state)

        result = execution_state.ctx.result
        assert result.task == mock_arena.env.task
        assert result.consensus_reached is False
        assert result.confidence == 0.0
        assert result.messages == []
        assert result.critiques == []
        assert result.votes == []
        assert result.rounds_used == 0
        assert result.final_answer == ""

    @pytest.mark.asyncio
    async def test_records_start_time(self, mock_arena, execution_state):
        """Test that debate start time is recorded."""
        execution_state.debate_start_time = 0.0

        await setup_debate_infrastructure(mock_arena, execution_state)

        assert execution_state.debate_start_time > 0


# =============================================================================
# Tests for execute_debate_phases
# =============================================================================


class TestExecuteDebatePhases:
    """Tests for execute_debate_phases function."""

    @pytest.mark.asyncio
    async def test_executes_phase_executor(self, mock_arena, execution_state, mock_span):
        """Test that PhaseExecutor.execute is called."""
        await execute_debate_phases(mock_arena, execution_state, mock_span)

        mock_arena.phase_executor.execute.assert_called_once_with(
            execution_state.ctx,
            debate_id=execution_state.debate_id,
        )

    @pytest.mark.asyncio
    async def test_logs_phase_failures(self, mock_arena, execution_state, mock_span):
        """Test that phase failures are logged."""
        mock_result = MagicMock()
        mock_arena.phase_executor.execute.return_value = mock_result

        await execute_debate_phases(mock_arena, execution_state, mock_span)

        mock_arena._log_phase_failures.assert_called_once_with(mock_result)

    @pytest.mark.asyncio
    async def test_handles_timeout_with_partial_results(
        self, mock_arena, execution_state, mock_span
    ):
        """Test that timeout uses partial results."""
        execution_state.ctx.partial_messages = [MagicMock()]
        execution_state.ctx.partial_critiques = [MagicMock()]
        execution_state.ctx.partial_rounds = 2
        mock_arena.phase_executor.execute.side_effect = asyncio.TimeoutError()

        await execute_debate_phases(mock_arena, execution_state, mock_span)

        assert execution_state.ctx.result.messages == execution_state.ctx.partial_messages
        assert execution_state.ctx.result.critiques == execution_state.ctx.partial_critiques
        assert execution_state.ctx.result.rounds_used == 2
        assert execution_state.debate_status == "timeout"
        mock_span.set_attribute.assert_called_with("debate.status", "timeout")

    @pytest.mark.asyncio
    async def test_handles_early_stop_error(self, mock_arena, execution_state, mock_span):
        """Test that EarlyStopError is handled and re-raised."""
        from aragora.exceptions import EarlyStopError

        mock_arena.phase_executor.execute.side_effect = EarlyStopError(
            "User requested stop", round_stopped=1
        )

        with pytest.raises(EarlyStopError):
            await execute_debate_phases(mock_arena, execution_state, mock_span)

        assert execution_state.debate_status == "aborted"
        mock_span.set_attribute.assert_called_with("debate.status", "aborted")

    @pytest.mark.asyncio
    async def test_handles_general_exception(self, mock_arena, execution_state, mock_span):
        """Test that general exceptions set error status and re-raise."""
        mock_arena.phase_executor.execute.side_effect = ValueError("Something wrong")

        with pytest.raises(ValueError):
            await execute_debate_phases(mock_arena, execution_state, mock_span)

        assert execution_state.debate_status == "error"
        mock_span.set_attribute.assert_called_with("debate.status", "error")
        mock_span.record_exception.assert_called_once()


# =============================================================================
# Tests for record_debate_metrics
# =============================================================================


class TestRecordDebateMetrics:
    """Tests for record_debate_metrics function."""

    def test_decrements_active_debates(self, mock_arena, execution_state, mock_span):
        """Test that ACTIVE_DEBATES counter is decremented."""
        with patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES") as mock_counter:
            record_debate_metrics(mock_arena, execution_state, mock_span)

            mock_counter.dec.assert_called_once()

    def test_calculates_duration(self, mock_arena, execution_state, mock_span):
        """Test that duration is calculated correctly."""
        execution_state.debate_start_time = time.perf_counter() - 10.0

        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes") as mock_add_attrs,
            patch("aragora.debate.orchestrator_runner.track_debate_outcome"),
        ):
            record_debate_metrics(mock_arena, execution_state, mock_span)

            call_args = mock_add_attrs.call_args[0]
            attrs = call_args[1]
            assert attrs["debate.duration_seconds"] >= 10.0

    def test_adds_span_attributes(self, mock_arena, execution_state, mock_span):
        """Test that span attributes are added."""
        execution_state.ctx.result.consensus_reached = True
        execution_state.ctx.result.confidence = 0.9
        execution_state.ctx.result.messages = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes") as mock_add_attrs,
            patch("aragora.debate.orchestrator_runner.track_debate_outcome"),
        ):
            record_debate_metrics(mock_arena, execution_state, mock_span)

            mock_add_attrs.assert_called_once()
            call_args = mock_add_attrs.call_args[0]
            assert call_args[0] is mock_span
            attrs = call_args[1]
            assert attrs["debate.status"] == "completed"
            assert attrs["debate.consensus_reached"] is True
            assert attrs["debate.confidence"] == 0.9
            assert attrs["debate.message_count"] == 3

    def test_tracks_debate_outcome(self, mock_arena, execution_state, mock_span):
        """Test that debate outcome is tracked."""
        execution_state.debate_status = "completed"
        execution_state.domain = "finance"
        execution_state.ctx.result.consensus_reached = True
        execution_state.ctx.result.confidence = 0.8

        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes"),
            patch("aragora.debate.orchestrator_runner.track_debate_outcome") as mock_track,
        ):
            record_debate_metrics(mock_arena, execution_state, mock_span)

            mock_track.assert_called_once()
            call_kwargs = mock_track.call_args[1]
            assert call_kwargs["status"] == "completed"
            assert call_kwargs["domain"] == "finance"
            assert call_kwargs["consensus_reached"] is True
            assert call_kwargs["confidence"] == 0.8

    def test_tracks_circuit_breaker_metrics(self, mock_arena, execution_state, mock_span):
        """Test that circuit breaker metrics are tracked."""
        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes"),
            patch("aragora.debate.orchestrator_runner.track_debate_outcome"),
        ):
            record_debate_metrics(mock_arena, execution_state, mock_span)

            mock_arena._track_circuit_breaker_metrics.assert_called_once()

    def test_handles_none_result(self, mock_arena, execution_state, mock_span):
        """Test that None result is handled gracefully."""
        execution_state.ctx.result = None

        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes") as mock_add_attrs,
            patch("aragora.debate.orchestrator_runner.track_debate_outcome"),
        ):
            # Should not raise
            record_debate_metrics(mock_arena, execution_state, mock_span)

            call_args = mock_add_attrs.call_args[0]
            attrs = call_args[1]
            assert attrs["debate.consensus_reached"] is False
            assert attrs["debate.confidence"] == 0.0
            assert attrs["debate.message_count"] == 0


# =============================================================================
# Tests for handle_debate_completion
# =============================================================================


class TestHandleDebateCompletion:
    """Tests for handle_debate_completion function."""

    @pytest.mark.asyncio
    async def test_notifies_trackers_of_completion(self, mock_arena, execution_state):
        """Test that trackers are notified of debate completion."""
        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._trackers.on_debate_complete.assert_called_once_with(
            execution_state.ctx, execution_state.ctx.result
        )

    @pytest.mark.asyncio
    async def test_skips_tracker_notification_if_no_result(self, mock_arena, execution_state):
        """Test that tracker notification is skipped if result is None."""
        execution_state.ctx.result = None

        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._trackers.on_debate_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_extensions(self, mock_arena, execution_state):
        """Test that extensions are triggered."""
        await handle_debate_completion(mock_arena, execution_state)

        mock_arena.extensions.on_debate_complete.assert_called_once_with(
            execution_state.ctx, execution_state.ctx.result, mock_arena.agents
        )

    @pytest.mark.asyncio
    async def test_records_debate_cost(self, mock_arena, execution_state):
        """Test that debate cost is recorded."""
        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._budget_coordinator.record_debate_cost.assert_called_once_with(
            execution_state.debate_id,
            execution_state.ctx.result,
            extensions=mock_arena.extensions,
        )

    @pytest.mark.asyncio
    async def test_skips_cost_recording_if_no_result(self, mock_arena, execution_state):
        """Test that cost recording is skipped if result is None."""
        execution_state.ctx.result = None

        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._budget_coordinator.record_debate_cost.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingests_debate_outcome_to_km(self, mock_arena, execution_state):
        """Test that debate outcome is ingested to Knowledge Mound (background task)."""
        await handle_debate_completion(mock_arena, execution_state)

        # KM ingestion runs as a background task; yield to let it execute
        await asyncio.sleep(0)

        mock_arena._ingest_debate_outcome.assert_called_once_with(execution_state.ctx.result)

    @pytest.mark.asyncio
    async def test_handles_km_ingestion_error(self, mock_arena, execution_state):
        """Test that KM ingestion errors are handled gracefully."""
        mock_arena._ingest_debate_outcome.side_effect = ConnectionError("KM down")

        # Should not raise — ingestion runs in background with retry
        await handle_debate_completion(mock_arena, execution_state)
        await asyncio.sleep(0)  # Let background task start

    @pytest.mark.asyncio
    async def test_completes_gupp_tracking_on_success(self, mock_arena, execution_state):
        """Test that GUPP tracking is completed on success."""
        execution_state.gupp_bead_id = "bead-123"
        execution_state.gupp_hook_entries = {"agent-1": "entry-1"}
        execution_state.debate_status = "completed"

        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._update_debate_bead.assert_called_once_with(
            "bead-123", execution_state.ctx.result, True
        )
        mock_arena._complete_hook_tracking.assert_called_once_with(
            "bead-123",
            {"agent-1": "entry-1"},
            True,
            error_msg="",
        )
        assert execution_state.ctx.result.bead_id == "bead-123"

    @pytest.mark.asyncio
    async def test_completes_gupp_tracking_on_failure(self, mock_arena, execution_state):
        """Test that GUPP tracking is completed with failure status."""
        execution_state.gupp_bead_id = "bead-456"
        execution_state.gupp_hook_entries = {"agent-0": "entry-0"}
        execution_state.debate_status = "error"

        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._update_debate_bead.assert_called_once_with(
            "bead-456", execution_state.ctx.result, False
        )
        mock_arena._complete_hook_tracking.assert_called_once_with(
            "bead-456",
            {"agent-0": "entry-0"},
            False,
            error_msg="Debate error",
        )

    @pytest.mark.asyncio
    async def test_handles_gupp_completion_error(self, mock_arena, execution_state):
        """Test that GUPP completion errors are handled gracefully."""
        execution_state.gupp_bead_id = "bead-789"
        execution_state.gupp_hook_entries = {"agent-1": "entry-1"}
        mock_arena._update_debate_bead.side_effect = ConnectionError("Failed")

        # Should not raise
        await handle_debate_completion(mock_arena, execution_state)

    @pytest.mark.asyncio
    async def test_creates_bead_if_gupp_not_used(self, mock_arena, execution_state):
        """Test that bead is created if GUPP tracking was not used."""
        execution_state.gupp_bead_id = None
        mock_arena._create_debate_bead.return_value = "new-bead-id"

        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._create_debate_bead.assert_called_once_with(execution_state.ctx.result)
        assert execution_state.ctx.result.bead_id == "new-bead-id"

    @pytest.mark.asyncio
    async def test_handles_bead_creation_error(self, mock_arena, execution_state):
        """Test that bead creation errors are handled gracefully."""
        execution_state.gupp_bead_id = None
        mock_arena._create_debate_bead.side_effect = OSError("Disk full")

        # Should not raise
        await handle_debate_completion(mock_arena, execution_state)

    @pytest.mark.asyncio
    async def test_queues_for_supabase_sync(self, mock_arena, execution_state):
        """Test that result is queued for Supabase sync."""
        await handle_debate_completion(mock_arena, execution_state)

        mock_arena._queue_for_supabase_sync.assert_called_once_with(
            execution_state.ctx, execution_state.ctx.result
        )


# =============================================================================
# Tests for cleanup_debate_resources
# =============================================================================


class TestCleanupDebateResources:
    """Tests for cleanup_debate_resources function."""

    @pytest.mark.asyncio
    async def test_cleans_up_checkpoints_on_success(self, mock_arena, execution_state):
        """Test that checkpoints are cleaned up on successful completion."""
        execution_state.debate_status = "completed"
        mock_arena.protocol.checkpoint_cleanup_on_success = True
        mock_arena.protocol.checkpoint_keep_on_success = 2
        mock_arena.cleanup_checkpoints.return_value = 5

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena.cleanup_checkpoints.assert_called_once_with(
            execution_state.debate_id, keep_latest=2
        )

    @pytest.mark.asyncio
    async def test_skips_checkpoint_cleanup_on_failure(self, mock_arena, execution_state):
        """Test that checkpoint cleanup is skipped on failure."""
        execution_state.debate_status = "error"

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena.cleanup_checkpoints.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_cleanup_when_disabled(self, mock_arena, execution_state):
        """Test that checkpoint cleanup is skipped when disabled."""
        execution_state.debate_status = "completed"
        mock_arena.protocol.checkpoint_cleanup_on_success = False

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena.cleanup_checkpoints.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_checkpoint_cleanup_error(self, mock_arena, execution_state):
        """Test that checkpoint cleanup errors are handled gracefully."""
        execution_state.debate_status = "completed"
        mock_arena.cleanup_checkpoints.side_effect = RuntimeError("Cleanup failed")

        # Should not raise
        result = await cleanup_debate_resources(mock_arena, execution_state)

        assert result is not None

    @pytest.mark.asyncio
    async def test_cleans_up_convergence_cache(self, mock_arena, execution_state):
        """Test that convergence cache is cleaned up."""
        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena._cleanup_convergence_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_tears_down_agent_channels(self, mock_arena, execution_state):
        """Test that agent channels are torn down."""
        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena._teardown_agent_channels.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalizes_result(self, mock_arena, execution_state):
        """Test that result is finalized."""
        await cleanup_debate_resources(mock_arena, execution_state)

        execution_state.ctx.finalize_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_finalized_result(self, mock_arena, execution_state, mock_debate_result):
        """Test that finalized result is returned."""
        execution_state.ctx.finalize_result.return_value = mock_debate_result

        result = await cleanup_debate_resources(mock_arena, execution_state)

        assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_translates_conclusions_when_enabled(
        self, mock_arena, execution_state, mock_debate_result
    ):
        """Test that conclusions are translated when translation is enabled."""
        mock_arena.protocol.enable_translation = True
        execution_state.ctx.finalize_result.return_value = mock_debate_result

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena._translate_conclusions.assert_called_once_with(mock_debate_result)

    @pytest.mark.asyncio
    async def test_skips_translation_when_disabled(
        self, mock_arena, execution_state, mock_debate_result
    ):
        """Test that translation is skipped when disabled."""
        mock_arena.protocol.enable_translation = False
        execution_state.ctx.finalize_result.return_value = mock_debate_result

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena._translate_conclusions.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_translation_when_result_is_none(self, mock_arena, execution_state):
        """Test that translation is skipped when result is None."""
        mock_arena.protocol.enable_translation = True
        execution_state.ctx.finalize_result.return_value = None

        await cleanup_debate_resources(mock_arena, execution_state)

        mock_arena._translate_conclusions.assert_not_called()


# =============================================================================
# Tests for Error Handling and Recovery
# =============================================================================


class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_initialize_handles_km_init_error(self, mock_arena):
        """Test that KM initialization errors are handled."""
        mock_arena._init_km_context.side_effect = ConnectionError("KM unavailable")

        # The function should propagate this error since it's critical
        with pytest.raises(ConnectionError):
            await initialize_debate_context(mock_arena, "corr-123")

    @pytest.mark.asyncio
    async def test_initialize_handles_channel_setup_error(self, mock_arena):
        """Test that channel setup errors are handled."""
        mock_arena._setup_agent_channels.side_effect = RuntimeError("Channel error")

        # The function should propagate this error since it's critical
        with pytest.raises(RuntimeError):
            await initialize_debate_context(mock_arena, "corr-123")

    @pytest.mark.asyncio
    async def test_completion_continues_after_extension_error(self, mock_arena, execution_state):
        """Test that completion continues after extension error."""
        mock_arena.extensions.on_debate_complete.side_effect = ValueError("Ext error")

        # Extension errors should propagate
        with pytest.raises(ValueError):
            await handle_debate_completion(mock_arena, execution_state)

    @pytest.mark.asyncio
    async def test_cleanup_continues_after_individual_errors(self, mock_arena, execution_state):
        """Test that cleanup continues even if individual operations fail."""
        execution_state.debate_status = "completed"
        mock_arena.cleanup_checkpoints.side_effect = OSError("Disk error")

        # Should not raise and should still finalize result
        result = await cleanup_debate_resources(mock_arena, execution_state)

        assert result is not None
        mock_arena._cleanup_convergence_cache.assert_called_once()
        mock_arena._teardown_agent_channels.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_recording_with_timeout_status(
        self, mock_arena, execution_state, mock_span
    ):
        """Test metrics recording with timeout status."""
        execution_state.debate_status = "timeout"
        execution_state.ctx.result.consensus_reached = False
        execution_state.ctx.result.confidence = 0.3

        with (
            patch("aragora.debate.orchestrator_runner.ACTIVE_DEBATES"),
            patch("aragora.debate.orchestrator_runner.add_span_attributes"),
            patch("aragora.debate.orchestrator_runner.track_debate_outcome") as mock_track,
        ):
            record_debate_metrics(mock_arena, execution_state, mock_span)

            mock_track.assert_called_once()
            call_kwargs = mock_track.call_args[1]
            assert call_kwargs["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_full_workflow_with_errors(self, mock_arena):
        """Test a realistic workflow with recoverable errors."""
        # Set up for partial success scenario
        mock_arena.protocol.enable_hook_tracking = True
        mock_arena._create_pending_debate_bead.return_value = "bead-test"
        mock_arena._init_hook_tracking.return_value = {"agent-0": "entry-0"}

        # Initialize
        state = await initialize_debate_context(mock_arena, "test-corr")
        assert state.debate_id is not None

        # Setup infrastructure
        await setup_debate_infrastructure(mock_arena, state)
        assert state.gupp_bead_id == "bead-test"
        assert state.debate_start_time > 0

        # Simulate phase execution with timeout
        mock_span = MagicMock()
        state.ctx.partial_messages = [MagicMock()]
        state.ctx.partial_rounds = 1
        mock_arena.phase_executor.execute.side_effect = asyncio.TimeoutError()

        await execute_debate_phases(mock_arena, state, mock_span)
        assert state.debate_status == "timeout"

        # Handle completion with KM error (should be handled)
        mock_arena._ingest_debate_outcome.side_effect = ConnectionError("KM down")
        await handle_debate_completion(mock_arena, state)

        # Cleanup
        result = await cleanup_debate_resources(mock_arena, state)
        assert result is not None
