"""
Comprehensive tests for Workflow Engine.

This module provides comprehensive test coverage for aragora/workflow/engine.py including:
- Engine initialization and configuration
- Workflow execution lifecycle
- Node execution and transitions
- Conditional branching
- Parallel execution paths
- Error handling and retries
- Timeout handling
- State machine transitions
- Event emission and callbacks
- Integration with persistent store
- Workflow cancellation
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.engine import (
    WorkflowEngine,
    get_workflow_engine,
    reset_workflow_engine,
)
from aragora.workflow.step import (
    BaseStep,
    WorkflowContext,
    WorkflowStep,
    AgentStep,
    ParallelStep,
    ConditionalStep,
    LoopStep,
)
from aragora.workflow.types import (
    ExecutionPattern,
    StepDefinition,
    StepResult,
    StepStatus,
    TransitionRule,
    WorkflowCheckpoint,
    WorkflowConfig,
    WorkflowDefinition,
    WorkflowResult,
)


# =============================================================================
# Test Step Implementations
# =============================================================================


class SimpleTestStep(BaseStep):
    """Simple step that returns a configurable output."""

    def __init__(self, name: str = "test", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        return {"result": self._config.get("output", "done"), "step_name": self._name}


class FailingStep(BaseStep):
    """Step that always fails for testing error handling."""

    def __init__(self, name: str = "failing", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        error_message = self._config.get("error_message", "Step failed intentionally")
        raise RuntimeError(error_message)


class SlowStep(BaseStep):
    """Step with configurable delay for testing timeouts."""

    def __init__(self, name: str = "slow", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        delay = self._config.get("delay_seconds", 1.0)
        await asyncio.sleep(delay)
        return {"message": "Slow step completed", "delay": delay}


class CountingStep(BaseStep):
    """Step that counts executions for testing loops and retries."""

    execution_count = 0

    def __init__(self, name: str = "counter", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        CountingStep.execution_count += 1
        current_count = CountingStep.execution_count
        context.set_state("execution_count", current_count)
        return {"count": current_count}

    @classmethod
    def reset(cls):
        cls.execution_count = 0


class ContextAwareStep(BaseStep):
    """Step that reads from and writes to context for testing state propagation."""

    def __init__(self, name: str = "context_aware", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        # Read inputs
        value = context.get_input("value", 0)
        multiplier = context.get_input("multiplier", 1)

        # Read from previous step if available
        prev_step = self._config.get("read_from_step")
        if prev_step:
            prev_output = context.get_step_output(prev_step, {})
            value = prev_output.get("result", value)

        # Compute result
        result = value * multiplier

        # Store in state
        context.set_state("last_result", result)

        return {"result": result}


class ConditionalOutputStep(BaseStep):
    """Step that outputs different values based on config for testing transitions."""

    def __init__(self, name: str = "conditional_output", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        decision = self._config.get("decision", "default")
        score = self._config.get("score", 50)
        return {"decision": decision, "score": score, "approved": score >= 70}


class IntermittentFailingStep(BaseStep):
    """Step that fails a configurable number of times before succeeding."""

    fail_count = 0

    def __init__(self, name: str = "intermittent", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        fail_times = self._config.get("fail_times", 2)
        if IntermittentFailingStep.fail_count < fail_times:
            IntermittentFailingStep.fail_count += 1
            raise RuntimeError(f"Failing ({IntermittentFailingStep.fail_count}/{fail_times})")
        return {"success": True, "attempts": IntermittentFailingStep.fail_count + 1}

    @classmethod
    def reset(cls):
        cls.fail_count = 0


class MockCheckpointStore:
    """Mock checkpoint store for testing."""

    def __init__(self):
        self.checkpoints: dict[str, WorkflowCheckpoint] = {}
        self.save_calls = 0
        self.load_calls = 0

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        self.save_calls += 1
        self.checkpoints[checkpoint.id] = checkpoint
        return checkpoint.id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        self.load_calls += 1
        return self.checkpoints.get(checkpoint_id)

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        matching = [cp for cp in self.checkpoints.values() if cp.workflow_id == workflow_id]
        if not matching:
            return None
        return max(matching, key=lambda cp: cp.created_at)

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        return [cp.id for cp in self.checkpoints.values() if cp.workflow_id == workflow_id]

    async def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            return True
        return False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic workflow configuration."""
    return WorkflowConfig(
        total_timeout_seconds=60.0,
        step_timeout_seconds=10.0,
        stop_on_failure=True,
        enable_checkpointing=False,
    )


@pytest.fixture
def checkpointing_config():
    """Create a configuration with checkpointing enabled."""
    return WorkflowConfig(
        total_timeout_seconds=60.0,
        step_timeout_seconds=10.0,
        stop_on_failure=True,
        enable_checkpointing=True,
        checkpoint_interval_steps=1,
    )


@pytest.fixture
def mock_checkpoint_store():
    """Create a mock checkpoint store."""
    return MockCheckpointStore()


@pytest.fixture
def engine(basic_config, mock_checkpoint_store):
    """Create a workflow engine with custom step types registered."""
    engine = WorkflowEngine(config=basic_config, checkpoint_store=mock_checkpoint_store)
    engine.register_step_type("test", SimpleTestStep)
    engine.register_step_type("failing", FailingStep)
    engine.register_step_type("slow", SlowStep)
    engine.register_step_type("counting", CountingStep)
    engine.register_step_type("context_aware", ContextAwareStep)
    engine.register_step_type("conditional_output", ConditionalOutputStep)
    engine.register_step_type("intermittent", IntermittentFailingStep)
    return engine


@pytest.fixture
def checkpointing_engine(checkpointing_config, mock_checkpoint_store):
    """Create a workflow engine with checkpointing enabled."""
    engine = WorkflowEngine(config=checkpointing_config, checkpoint_store=mock_checkpoint_store)
    engine.register_step_type("test", SimpleTestStep)
    engine.register_step_type("failing", FailingStep)
    engine.register_step_type("slow", SlowStep)
    engine.register_step_type("counting", CountingStep)
    engine.register_step_type("conditional_output", ConditionalOutputStep)
    return engine


@pytest.fixture(autouse=True)
def reset_counters():
    """Reset step counters before each test."""
    CountingStep.reset()
    IntermittentFailingStep.reset()
    reset_workflow_engine()
    yield


# =============================================================================
# Engine Initialization and Configuration Tests
# =============================================================================


class TestEngineInitialization:
    """Test engine initialization and configuration."""

    def test_default_initialization(self):
        """Test engine initializes with default config."""
        engine = WorkflowEngine()
        assert engine._config is not None
        assert engine._config.total_timeout_seconds == 3600.0
        assert engine._config.stop_on_failure is True

    def test_custom_config_initialization(self):
        """Test engine initializes with custom config."""
        config = WorkflowConfig(
            total_timeout_seconds=600.0,
            step_timeout_seconds=30.0,
            stop_on_failure=False,
        )
        engine = WorkflowEngine(config=config)
        assert engine._config.total_timeout_seconds == 600.0
        assert engine._config.step_timeout_seconds == 30.0
        assert engine._config.stop_on_failure is False

    def test_custom_step_registry(self):
        """Test engine initializes with custom step registry."""
        registry = {"custom": SimpleTestStep}
        engine = WorkflowEngine(step_registry=registry)
        assert "custom" in engine._step_types

    def test_custom_checkpoint_store(self, mock_checkpoint_store):
        """Test engine initializes with custom checkpoint store."""
        engine = WorkflowEngine(checkpoint_store=mock_checkpoint_store)
        assert engine._checkpoint_store is mock_checkpoint_store

    def test_default_step_types_registered(self):
        """Test default step types are registered."""
        engine = WorkflowEngine()
        assert "agent" in engine._step_types
        assert "parallel" in engine._step_types
        assert "conditional" in engine._step_types
        assert "loop" in engine._step_types

    def test_register_step_type(self, engine):
        """Test registering a new step type."""

        class CustomStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                return {"custom": True}

        engine.register_step_type("custom", CustomStep)
        assert "custom" in engine._step_types
        assert engine._step_types["custom"] is CustomStep

    def test_override_step_type(self, engine):
        """Test overriding an existing step type."""

        class NewTestStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                return {"new": True}

        engine.register_step_type("test", NewTestStep)
        assert engine._step_types["test"] is NewTestStep


class TestSingletonEngine:
    """Test singleton engine functions."""

    def test_get_workflow_engine_creates_singleton(self):
        """Test get_workflow_engine creates a singleton."""
        reset_workflow_engine()
        engine1 = get_workflow_engine()
        engine2 = get_workflow_engine()
        assert engine1 is engine2

    def test_get_workflow_engine_with_config(self):
        """Test get_workflow_engine with custom config."""
        reset_workflow_engine()
        config = WorkflowConfig(total_timeout_seconds=100.0)
        engine = get_workflow_engine(config=config)
        assert engine._config.total_timeout_seconds == 100.0

    def test_reset_workflow_engine(self):
        """Test reset_workflow_engine clears singleton."""
        engine1 = get_workflow_engine()
        reset_workflow_engine()
        engine2 = get_workflow_engine()
        # Different instances after reset
        assert engine1 is not engine2


# =============================================================================
# Workflow Execution Lifecycle Tests
# =============================================================================


class TestWorkflowExecutionLifecycle:
    """Test workflow execution lifecycle."""

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, engine):
        """Test executing a simple single-step workflow."""
        definition = WorkflowDefinition(
            id="wf_simple",
            name="Simple Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert result.workflow_id.startswith("wf_")
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_custom_workflow_id(self, engine):
        """Test executing with a custom workflow ID."""
        definition = WorkflowDefinition(
            id="wf_test",
            name="Test Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test"),
            ],
        )

        result = await engine.execute(definition, workflow_id="custom_wf_123")

        assert result.workflow_id == "custom_wf_123"

    @pytest.mark.asyncio
    async def test_execute_with_inputs(self, engine):
        """Test executing workflow with inputs."""
        definition = WorkflowDefinition(
            id="wf_input",
            name="Input Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Context Aware Step",
                    step_type="context_aware",
                    config={"multiplier": 2},
                ),
            ],
        )

        result = await engine.execute(definition, inputs={"value": 5, "multiplier": 3})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_returns_workflow_result(self, engine):
        """Test execute returns proper WorkflowResult."""
        definition = WorkflowDefinition(
            id="wf_result",
            name="Result Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert isinstance(result, WorkflowResult)
        assert result.definition_id == "wf_result"
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_no_entry_step_error(self, engine):
        """Test execute raises error when no entry step."""
        definition = WorkflowDefinition(
            id="wf_no_entry",
            name="No Entry Workflow",
            steps=[],
            entry_step=None,
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert "no entry step" in result.error.lower()

    @pytest.mark.asyncio
    async def test_final_output_is_last_step_output(self, engine):
        """Test final output is from the last executed step."""
        definition = WorkflowDefinition(
            id="wf_final_output",
            name="Final Output Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Step 1",
                    step_type="test",
                    config={"output": "first"},
                    next_steps=["s2"],
                ),
                StepDefinition(
                    id="s2",
                    name="Step 2",
                    step_type="test",
                    config={"output": "final"},
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert result.final_output["result"] == "final"


# =============================================================================
# Node Execution and Transitions Tests
# =============================================================================


class TestNodeExecution:
    """Test node execution and transitions."""

    @pytest.mark.asyncio
    async def test_sequential_step_execution(self, engine):
        """Test steps execute in sequential order."""
        execution_order = []

        class OrderTrackingStep(BaseStep):
            def __init__(self, name: str, config: dict[str, Any] | None = None):
                super().__init__(name, config or {})

            async def execute(self, context: WorkflowContext) -> Any:
                execution_order.append(self._name)
                return {"executed": self._name}

        engine.register_step_type("order_track", OrderTrackingStep)

        definition = WorkflowDefinition(
            id="wf_seq",
            name="Sequential Workflow",
            steps=[
                StepDefinition(id="s1", name="First", step_type="order_track", next_steps=["s2"]),
                StepDefinition(id="s2", name="Second", step_type="order_track", next_steps=["s3"]),
                StepDefinition(id="s3", name="Third", step_type="order_track"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert execution_order == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_step_output_propagation(self, engine):
        """Test step outputs are available to subsequent steps."""
        definition = WorkflowDefinition(
            id="wf_prop",
            name="Propagation Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Producer",
                    step_type="test",
                    config={"output": 42},
                    next_steps=["s2"],
                ),
                StepDefinition(
                    id="s2",
                    name="Consumer",
                    step_type="context_aware",
                    config={"read_from_step": "s1"},
                ),
            ],
            inputs={"multiplier": 2},
        )

        result = await engine.execute(definition)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_step_type_fails(self, engine):
        """Test workflow fails gracefully with unknown step type."""
        definition = WorkflowDefinition(
            id="wf_unknown",
            name="Unknown Step Workflow",
            steps=[
                StepDefinition(id="s1", name="Unknown", step_type="nonexistent_type"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.FAILED
        assert "Unknown step type" in result.steps[0].error

    @pytest.mark.asyncio
    async def test_step_not_found_stops_execution(self, engine):
        """Test execution stops when next step not found."""
        definition = WorkflowDefinition(
            id="wf_missing_step",
            name="Missing Step Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Step 1",
                    step_type="test",
                    next_steps=["nonexistent"],
                ),
            ],
        )

        result = await engine.execute(definition)

        # Workflow should complete but stop after s1 since next step not found
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED


# =============================================================================
# Conditional Branching Tests
# =============================================================================


class TestConditionalBranching:
    """Test conditional branching logic."""

    @pytest.mark.asyncio
    async def test_transition_with_true_condition(self, engine):
        """Test transition is taken when condition is true."""
        definition = WorkflowDefinition(
            id="wf_cond_true",
            name="Conditional True Workflow",
            steps=[
                StepDefinition(
                    id="decision",
                    name="Decision",
                    step_type="conditional_output",
                    config={"score": 80},
                ),
                StepDefinition(id="approved", name="Approved", step_type="test"),
                StepDefinition(id="rejected", name="Rejected", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_approved",
                    from_step="decision",
                    to_step="approved",
                    condition="step_output.get('approved') == True",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_rejected",
                    from_step="decision",
                    to_step="rejected",
                    condition="step_output.get('approved') == False",
                    priority=5,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "approved" in step_ids
        assert "rejected" not in step_ids

    @pytest.mark.asyncio
    async def test_transition_with_false_condition(self, engine):
        """Test correct path is taken when condition is false."""
        definition = WorkflowDefinition(
            id="wf_cond_false",
            name="Conditional False Workflow",
            steps=[
                StepDefinition(
                    id="decision",
                    name="Decision",
                    step_type="conditional_output",
                    config={"score": 50},  # Below 70, so not approved
                ),
                StepDefinition(id="approved", name="Approved", step_type="test"),
                StepDefinition(id="rejected", name="Rejected", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_approved",
                    from_step="decision",
                    to_step="approved",
                    condition="step_output.get('approved') == True",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_rejected",
                    from_step="decision",
                    to_step="rejected",
                    condition="step_output.get('approved') == False",
                    priority=5,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "rejected" in step_ids
        assert "approved" not in step_ids

    @pytest.mark.asyncio
    async def test_transition_priority_ordering(self, engine):
        """Test transitions are evaluated by priority."""
        definition = WorkflowDefinition(
            id="wf_priority",
            name="Priority Workflow",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="test",
                ),
                StepDefinition(id="high", name="High Priority", step_type="test"),
                StepDefinition(id="low", name="Low Priority", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_low",
                    from_step="start",
                    to_step="low",
                    condition="True",
                    priority=1,
                ),
                TransitionRule(
                    id="tr_high",
                    from_step="start",
                    to_step="high",
                    condition="True",
                    priority=10,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        # High priority should be taken
        assert "high" in step_ids
        assert "low" not in step_ids

    @pytest.mark.asyncio
    async def test_fallback_to_next_steps(self, engine):
        """Test fallback to next_steps when no transition matches."""
        definition = WorkflowDefinition(
            id="wf_fallback",
            name="Fallback Workflow",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="test",
                    next_steps=["fallback"],
                ),
                StepDefinition(id="conditional", name="Conditional", step_type="test"),
                StepDefinition(id="fallback", name="Fallback", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_cond",
                    from_step="start",
                    to_step="conditional",
                    condition="False",  # Never matches
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "fallback" in step_ids
        assert "conditional" not in step_ids

    @pytest.mark.asyncio
    async def test_invalid_condition_returns_false(self, engine):
        """Test invalid condition expression returns false."""
        definition = WorkflowDefinition(
            id="wf_invalid_cond",
            name="Invalid Condition Workflow",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="test",
                    next_steps=["fallback"],
                ),
                StepDefinition(id="target", name="Target", step_type="test"),
                StepDefinition(id="fallback", name="Fallback", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_invalid",
                    from_step="start",
                    to_step="target",
                    condition="invalid_syntax[[[",  # Invalid syntax
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        # Should fallback since invalid condition returns false
        assert "fallback" in step_ids

    @pytest.mark.asyncio
    async def test_condition_with_context_access(self, engine):
        """Test conditions can access context variables."""
        definition = WorkflowDefinition(
            id="wf_context_cond",
            name="Context Condition Workflow",
            steps=[
                StepDefinition(id="start", name="Start", step_type="test"),
                StepDefinition(id="high", name="High", step_type="test"),
                StepDefinition(id="low", name="Low", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_high",
                    from_step="start",
                    to_step="high",
                    condition="inputs.get('threshold', 0) > 50",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_low",
                    from_step="start",
                    to_step="low",
                    condition="True",
                    priority=1,
                ),
            ],
        )

        result = await engine.execute(definition, inputs={"threshold": 75})

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "high" in step_ids


# =============================================================================
# Error Handling and Retries Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and retries."""

    @pytest.mark.asyncio
    async def test_step_failure_stops_workflow(self, engine):
        """Test step failure stops workflow execution."""
        definition = WorkflowDefinition(
            id="wf_fail_stop",
            name="Fail Stop Workflow",
            steps=[
                StepDefinition(id="s1", name="Success", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Fail", step_type="failing", next_steps=["s3"]),
                StepDefinition(id="s3", name="Never Reached", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        step_ids = [s.step_id for s in result.steps]
        assert "s1" in step_ids
        assert "s2" in step_ids
        assert "s3" not in step_ids  # Not executed

    @pytest.mark.asyncio
    async def test_optional_step_failure_continues(self, engine):
        """Test optional step failure allows workflow to continue."""
        definition = WorkflowDefinition(
            id="wf_optional_fail",
            name="Optional Fail Workflow",
            steps=[
                StepDefinition(id="s1", name="Success", step_type="test", next_steps=["s2"]),
                StepDefinition(
                    id="s2",
                    name="Optional Fail",
                    step_type="failing",
                    optional=True,
                    next_steps=["s3"],
                ),
                StepDefinition(id="s3", name="Continue", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        # Workflow continues despite optional step failing
        step_ids = [s.step_id for s in result.steps]
        assert "s1" in step_ids
        assert "s2" in step_ids
        assert "s3" in step_ids

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, engine):
        """Test step retries on failure."""
        IntermittentFailingStep.reset()

        definition = WorkflowDefinition(
            id="wf_retry",
            name="Retry Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Intermittent",
                    step_type="intermittent",
                    config={"fail_times": 2},
                    retries=3,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert result.steps[0].retry_count > 0

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, engine):
        """Test workflow fails when max retries exceeded."""
        IntermittentFailingStep.reset()

        definition = WorkflowDefinition(
            id="wf_max_retry",
            name="Max Retry Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Intermittent",
                    step_type="intermittent",
                    config={"fail_times": 5},
                    retries=2,  # Only 2 retries, but fails 5 times
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_error_message_captured(self, engine):
        """Test error message is captured in step result."""
        definition = WorkflowDefinition(
            id="wf_error_msg",
            name="Error Message Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Fail",
                    step_type="failing",
                    config={"error_message": "Custom error message"},
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert "Custom error message" in result.steps[0].error


# =============================================================================
# Timeout Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_step_timeout(self, engine):
        """Test step times out and fails."""
        definition = WorkflowDefinition(
            id="wf_step_timeout",
            name="Step Timeout Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow",
                    step_type="slow",
                    config={"delay_seconds": 5.0},
                    timeout_seconds=0.1,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Timed out" in result.steps[0].error

    @pytest.mark.asyncio
    async def test_optional_step_timeout_skips(self, engine):
        """Test optional step that times out is skipped."""
        config = WorkflowConfig(
            total_timeout_seconds=60.0,
            step_timeout_seconds=10.0,
            skip_optional_on_timeout=True,
        )
        test_engine = WorkflowEngine(config=config)
        test_engine.register_step_type("slow", SlowStep)
        test_engine.register_step_type("test", SimpleTestStep)

        definition = WorkflowDefinition(
            id="wf_optional_timeout",
            name="Optional Timeout Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow Optional",
                    step_type="slow",
                    config={"delay_seconds": 5.0},
                    timeout_seconds=0.1,
                    optional=True,
                    next_steps=["s2"],
                ),
                StepDefinition(id="s2", name="Continue", step_type="test"),
            ],
        )

        result = await test_engine.execute(definition)

        assert result.steps[0].status == StepStatus.SKIPPED
        step_ids = [s.step_id for s in result.steps]
        assert "s2" in step_ids

    @pytest.mark.asyncio
    async def test_workflow_total_timeout(self):
        """Test workflow total timeout."""
        config = WorkflowConfig(total_timeout_seconds=0.1)
        test_engine = WorkflowEngine(config=config)
        test_engine.register_step_type("slow", SlowStep)

        definition = WorkflowDefinition(
            id="wf_total_timeout",
            name="Total Timeout Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow",
                    step_type="slow",
                    config={"delay_seconds": 10.0},
                    timeout_seconds=60.0,  # Step timeout is longer than workflow timeout
                ),
            ],
        )

        result = await test_engine.execute(definition)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self, engine):
        """Test step timeout triggers retry."""
        definition = WorkflowDefinition(
            id="wf_timeout_retry",
            name="Timeout Retry Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow",
                    step_type="slow",
                    config={"delay_seconds": 5.0},
                    timeout_seconds=0.1,
                    retries=1,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        # Should have retried
        assert result.steps[0].retry_count > 0


class TestTimeoutProgressWarnings:
    """Test timeout progress warning functionality."""

    @pytest.mark.asyncio
    async def test_timeout_progress_check(self):
        """Test timeout progress warnings are issued."""
        config = WorkflowConfig(total_timeout_seconds=1.0)
        test_engine = WorkflowEngine(config=config)

        # Simulate time passing
        test_engine._check_timeout_progress(
            start_time=time.time() - 0.6,  # 60% elapsed
            total_timeout=1.0,
            workflow_id="test_wf",
        )

        # Should have issued 50% warning
        assert 0.5 in test_engine._timeout_warnings_issued

    def test_no_warning_when_no_timeout(self, engine):
        """Test no warning issued when total_timeout is 0."""
        engine._check_timeout_progress(
            start_time=time.time(),
            total_timeout=0,
            workflow_id="test_wf",
        )

        assert len(engine._timeout_warnings_issued) == 0


# =============================================================================
# State Machine Transitions Tests
# =============================================================================


class TestStateMachineTransitions:
    """Test state machine transitions."""

    @pytest.mark.asyncio
    async def test_step_status_transitions(self, engine):
        """Test step status transitions correctly."""
        definition = WorkflowDefinition(
            id="wf_status",
            name="Status Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        # Step should have completed
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].started_at is not None
        assert result.steps[0].completed_at is not None
        assert result.steps[0].duration_ms > 0

    @pytest.mark.asyncio
    async def test_workflow_result_tracks_all_steps(self, engine):
        """Test workflow result tracks all executed steps."""
        definition = WorkflowDefinition(
            id="wf_track",
            name="Track Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test", next_steps=["s3"]),
                StepDefinition(id="s3", name="Step 3", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert len(result.steps) == 3
        for step in result.steps:
            assert step.status == StepStatus.COMPLETED


class TestTerminationControl:
    """Test termination control functionality."""

    @pytest.mark.asyncio
    async def test_request_termination(self, engine):
        """Test requesting workflow termination."""
        definition = WorkflowDefinition(
            id="wf_terminate",
            name="Terminate Workflow",
            steps=[
                StepDefinition(
                    id="s1", name="Step 1", step_type="slow", config={"delay_seconds": 0.5}
                ),
            ],
        )

        # Start workflow in background
        async def terminate_after_delay():
            await asyncio.sleep(0.1)
            engine.request_termination("Test termination")

        task = asyncio.create_task(terminate_after_delay())
        result = await engine.execute(definition)
        await task

        # Check termination was requested
        is_terminated, reason = engine.check_termination()
        assert is_terminated is True
        assert reason == "Test termination"

    def test_check_termination_initial_state(self, engine):
        """Test check_termination returns initial state."""
        is_terminated, reason = engine.check_termination()
        assert is_terminated is False
        assert reason is None

    def test_current_step_property(self, engine):
        """Test current_step property."""
        # Initially None
        assert engine.current_step is None


# =============================================================================
# Checkpointing Tests
# =============================================================================


class TestCheckpointing:
    """Test checkpointing functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_created(self, checkpointing_engine, mock_checkpoint_store):
        """Test checkpoints are created during execution."""
        definition = WorkflowDefinition(
            id="wf_checkpoint",
            name="Checkpoint Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test"),
            ],
        )

        await checkpointing_engine.execute(definition)

        # Checkpoints should have been created
        assert mock_checkpoint_store.save_calls > 0

    @pytest.mark.asyncio
    async def test_checkpoint_content(self, checkpointing_engine, mock_checkpoint_store):
        """Test checkpoint contains correct data."""
        definition = WorkflowDefinition(
            id="wf_cp_content",
            name="Checkpoint Content Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test"),
            ],
        )

        await checkpointing_engine.execute(definition, inputs={"test_input": "value"})

        # Get the last checkpoint
        checkpoints = list(mock_checkpoint_store.checkpoints.values())
        assert len(checkpoints) > 0

        checkpoint = checkpoints[-1]
        assert checkpoint.definition_id == "wf_cp_content"
        assert len(checkpoint.completed_steps) > 0

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, checkpointing_engine):
        """Test resuming workflow from checkpoint."""
        definition = WorkflowDefinition(
            id="wf_resume",
            name="Resume Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test", next_steps=["s3"]),
                StepDefinition(id="s3", name="Step 3", step_type="test"),
            ],
        )

        # Create a checkpoint as if s1 was completed
        checkpoint = WorkflowCheckpoint(
            id="cp_test",
            workflow_id="wf_resume_123",
            definition_id="wf_resume",
            current_step="s2",
            completed_steps=["s1"],
            step_outputs={"s1": {"result": "done"}},
            context_state={"inputs": {"test": "value"}, "state": {}},
            created_at=datetime.now(timezone.utc),
            checksum="test",
        )

        result = await checkpointing_engine.resume("wf_resume_123", checkpoint, definition)

        # Should have executed s2 and s3
        step_ids = [s.step_id for s in result.steps]
        assert "s2" in step_ids
        assert "s3" in step_ids
        # s1 should not be in results (was already completed)
        assert "s1" not in step_ids

    @pytest.mark.asyncio
    async def test_get_checkpoint(self, engine, mock_checkpoint_store):
        """Test getting checkpoint by ID."""
        checkpoint = WorkflowCheckpoint(
            id="cp_get_test",
            workflow_id="wf_test",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_get_test"] = checkpoint

        result = await engine.get_checkpoint("cp_get_test")

        assert result is not None
        assert result.id == "cp_get_test"

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, engine, mock_checkpoint_store):
        """Test getting latest checkpoint for workflow."""
        # Add multiple checkpoints
        for i in range(3):
            checkpoint = WorkflowCheckpoint(
                id=f"cp_{i}",
                workflow_id="wf_latest",
                definition_id="def_test",
                current_step=f"s{i}",
                completed_steps=[f"s{j}" for j in range(i)],
                step_outputs={},
                context_state={},
                created_at=datetime.now(timezone.utc),
            )
            mock_checkpoint_store.checkpoints[f"cp_{i}"] = checkpoint

        result = await engine.get_latest_checkpoint("wf_latest")

        assert result is not None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, engine, mock_checkpoint_store):
        """Test listing checkpoints for workflow."""
        # Add checkpoints
        for i in range(3):
            checkpoint = WorkflowCheckpoint(
                id=f"cp_list_{i}",
                workflow_id="wf_list",
                definition_id="def_test",
                current_step=f"s{i}",
                completed_steps=[],
                step_outputs={},
                context_state={},
                created_at=datetime.now(timezone.utc),
            )
            mock_checkpoint_store.checkpoints[f"cp_list_{i}"] = checkpoint

        result = await engine.list_checkpoints("wf_list")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, engine, mock_checkpoint_store):
        """Test deleting a checkpoint."""
        checkpoint = WorkflowCheckpoint(
            id="cp_delete",
            workflow_id="wf_test",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_delete"] = checkpoint

        result = await engine.delete_checkpoint("cp_delete")

        assert result is True
        assert "cp_delete" not in mock_checkpoint_store.checkpoints


class TestCheckpointCache:
    """Test checkpoint cache functionality."""

    def test_checkpoint_cache_stats(self, engine):
        """Test checkpoint cache statistics."""
        stats = engine.get_checkpoint_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    @pytest.mark.asyncio
    async def test_checkpoint_cache_hit(self, engine, mock_checkpoint_store):
        """Test checkpoint is served from cache on second access."""
        checkpoint = WorkflowCheckpoint(
            id="cp_cache",
            workflow_id="wf_cache",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_cache"] = checkpoint

        # First access - cache miss
        await engine.get_checkpoint("cp_cache")
        # Second access - should hit cache
        await engine.get_checkpoint("cp_cache")

        stats = engine.get_checkpoint_cache_stats()
        assert stats["hits"] >= 1


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Test metrics functionality."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, engine):
        """Test getting execution metrics."""
        definition = WorkflowDefinition(
            id="wf_metrics",
            name="Metrics Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test"),
            ],
        )

        await engine.execute(definition)
        metrics = engine.get_metrics()

        assert "total_steps" in metrics
        assert "completed_steps" in metrics
        assert "failed_steps" in metrics
        assert "skipped_steps" in metrics
        assert "total_duration_ms" in metrics
        assert "step_durations" in metrics

        assert metrics["total_steps"] == 2
        assert metrics["completed_steps"] == 2

    @pytest.mark.asyncio
    async def test_metrics_with_failures(self, engine):
        """Test metrics include failed steps."""
        definition = WorkflowDefinition(
            id="wf_metrics_fail",
            name="Metrics Fail Workflow",
            steps=[
                StepDefinition(id="s1", name="Success", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Fail", step_type="failing"),
            ],
        )

        await engine.execute(definition)
        metrics = engine.get_metrics()

        assert metrics["completed_steps"] == 1
        assert metrics["failed_steps"] == 1

    @pytest.mark.asyncio
    async def test_metrics_with_skipped(self, engine):
        """Test metrics include skipped steps."""
        config = WorkflowConfig(
            total_timeout_seconds=60.0,
            skip_optional_on_timeout=True,
        )
        test_engine = WorkflowEngine(config=config)
        test_engine.register_step_type("slow", SlowStep)
        test_engine.register_step_type("test", SimpleTestStep)

        definition = WorkflowDefinition(
            id="wf_metrics_skip",
            name="Metrics Skip Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow Optional",
                    step_type="slow",
                    config={"delay_seconds": 5.0},
                    timeout_seconds=0.1,
                    optional=True,
                    next_steps=["s2"],
                ),
                StepDefinition(id="s2", name="Continue", step_type="test"),
            ],
        )

        await test_engine.execute(definition)
        metrics = test_engine.get_metrics()

        assert metrics["skipped_steps"] == 1


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Test parallel step execution."""

    @pytest.mark.asyncio
    async def test_parallel_step_execution(self):
        """Test ParallelStep executes sub-steps concurrently."""
        execution_times = []

        class TimedStep(BaseStep):
            def __init__(self, name: str, delay: float = 0.1):
                super().__init__(name)
                self._delay = delay

            async def execute(self, context: WorkflowContext) -> Any:
                start = time.time()
                await asyncio.sleep(self._delay)
                execution_times.append((self._name, time.time() - start))
                return {"name": self._name}

        parallel = ParallelStep(
            name="parallel_test",
            sub_steps=[
                TimedStep("a", 0.1),
                TimedStep("b", 0.1),
                TimedStep("c", 0.1),
            ],
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        start = time.time()
        result = await parallel.execute(context)
        total_time = time.time() - start

        # All steps completed
        assert len(result) == 3
        # Should complete in roughly the time of one step (parallel), not 3x
        assert total_time < 0.3  # 3 * 0.1 = 0.3, should be faster

    @pytest.mark.asyncio
    async def test_parallel_step_with_failure(self):
        """Test ParallelStep handles sub-step failures."""

        class SuccessStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                return {"success": True}

        class FailStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                raise RuntimeError("Sub-step failed")

        parallel = ParallelStep(
            name="parallel_fail",
            sub_steps=[
                SuccessStep("success1"),
                FailStep("fail"),
                SuccessStep("success2"),
            ],
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await parallel.execute(context)

        # All results should be collected
        assert len(result) == 3
        assert result["success1"]["success"] is True
        assert result["success2"]["success"] is True
        assert "error" in result["fail"]


# =============================================================================
# Loop Step Tests
# =============================================================================


class TestLoopStep:
    """Test loop step execution."""

    @pytest.mark.asyncio
    async def test_loop_until_condition(self):
        """Test loop executes until condition is met."""
        CountingStep.reset()

        loop = LoopStep(
            name="loop_test",
            wrapped_step=CountingStep("counter"),
            condition="state.get('execution_count', 0) >= 3",
            max_iterations=10,
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await loop.execute(context)

        assert result["iterations"] == 3
        assert len(result["outputs"]) == 3

    @pytest.mark.asyncio
    async def test_loop_max_iterations(self):
        """Test loop respects max_iterations."""
        loop = LoopStep(
            name="loop_max",
            wrapped_step=SimpleTestStep("simple"),
            condition="False",  # Never true, so loop until max
            max_iterations=5,
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await loop.execute(context)

        assert result["iterations"] == 5


# =============================================================================
# Conditional Step Tests
# =============================================================================


class TestConditionalStep:
    """Test conditional step execution."""

    @pytest.mark.asyncio
    async def test_conditional_executes_when_true(self):
        """Test conditional step executes when condition is true."""
        conditional = ConditionalStep(
            name="cond_true",
            wrapped_step=SimpleTestStep("wrapped"),
            condition="inputs.get('execute', False)",
        )

        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            inputs={"execute": True},
        )
        result = await conditional.execute(context)

        assert "result" in result
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_conditional_skips_when_false(self):
        """Test conditional step skips when condition is false."""
        conditional = ConditionalStep(
            name="cond_false",
            wrapped_step=SimpleTestStep("wrapped"),
            condition="inputs.get('execute', False)",
        )

        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            inputs={"execute": False},
        )
        result = await conditional.execute(context)

        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_conditional_invalid_condition_skips(self):
        """Test conditional step skips on invalid condition."""
        conditional = ConditionalStep(
            name="cond_invalid",
            wrapped_step=SimpleTestStep("wrapped"),
            condition="invalid_syntax[[[",
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await conditional.execute(context)

        assert result["skipped"] is True


# =============================================================================
# Workflow Definition Tests
# =============================================================================


class TestWorkflowDefinition:
    """Test WorkflowDefinition dataclass."""

    def test_create_definition(self):
        """Test creating a workflow definition."""
        steps = [
            StepDefinition(id="s1", name="Step 1", step_type="agent"),
            StepDefinition(id="s2", name="Step 2", step_type="agent"),
        ]

        definition = WorkflowDefinition(
            id="wf_1",
            name="Test Workflow",
            steps=steps,
        )

        assert definition.id == "wf_1"
        assert definition.name == "Test Workflow"
        assert len(definition.steps) == 2

    def test_auto_entry_step(self):
        """Test entry_step defaults to first step."""
        definition = WorkflowDefinition(
            id="wf_auto",
            name="Auto Entry",
            steps=[
                StepDefinition(id="first", name="First", step_type="agent"),
                StepDefinition(id="second", name="Second", step_type="agent"),
            ],
        )

        assert definition.entry_step == "first"

    def test_get_step(self):
        """Test get_step returns correct step."""
        definition = WorkflowDefinition(
            id="wf_get",
            name="Get Step",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent"),
                StepDefinition(id="s2", name="Step 2", step_type="agent"),
            ],
        )

        step = definition.get_step("s2")
        assert step is not None
        assert step.id == "s2"

    def test_get_step_not_found(self):
        """Test get_step returns None for unknown step."""
        definition = WorkflowDefinition(
            id="wf_not_found",
            name="Not Found",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent"),
            ],
        )

        step = definition.get_step("nonexistent")
        assert step is None

    def test_get_transitions_from(self):
        """Test get_transitions_from returns sorted transitions."""
        definition = WorkflowDefinition(
            id="wf_trans",
            name="Transitions",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent"),
                StepDefinition(id="s2", name="Step 2", step_type="agent"),
                StepDefinition(id="s3", name="Step 3", step_type="agent"),
            ],
            transitions=[
                TransitionRule(
                    id="tr1", from_step="s1", to_step="s2", condition="True", priority=5
                ),
                TransitionRule(
                    id="tr2", from_step="s1", to_step="s3", condition="True", priority=10
                ),
            ],
        )

        transitions = definition.get_transitions_from("s1")
        assert len(transitions) == 2
        # Higher priority first
        assert transitions[0].priority == 10
        assert transitions[0].to_step == "s3"

    def test_validate_valid_workflow(self):
        """Test validate returns true for valid workflow."""
        definition = WorkflowDefinition(
            id="wf_valid",
            name="Valid",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="agent"),
            ],
            entry_step="s1",
        )

        is_valid, errors = definition.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_entry_step(self):
        """Test validate catches missing entry step."""
        definition = WorkflowDefinition(
            id="wf_missing_entry",
            name="Missing Entry",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent"),
            ],
            entry_step="nonexistent",
        )

        is_valid, errors = definition.validate()
        assert is_valid is False
        assert any("entry step" in e.lower() for e in errors)

    def test_validate_invalid_transition(self):
        """Test validate catches invalid transitions."""
        definition = WorkflowDefinition(
            id="wf_invalid_trans",
            name="Invalid Transition",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="agent"),
            ],
            transitions=[
                TransitionRule(id="tr1", from_step="s1", to_step="nonexistent", condition="True"),
            ],
        )

        is_valid, errors = definition.validate()
        assert is_valid is False
        assert any("unknown step" in e.lower() for e in errors)


# =============================================================================
# Step Result Tests
# =============================================================================


class TestStepResult:
    """Test StepResult dataclass."""

    def test_success_property_completed(self):
        """Test success property for completed step."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.COMPLETED,
        )
        assert result.success is True

    def test_success_property_skipped(self):
        """Test success property for skipped step."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.SKIPPED,
        )
        assert result.success is True

    def test_success_property_failed(self):
        """Test success property for failed step."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.FAILED,
        )
        assert result.success is False


# =============================================================================
# Workflow Result Tests
# =============================================================================


class TestWorkflowResult:
    """Test WorkflowResult dataclass."""

    def test_get_step_result(self):
        """Test get_step_result returns correct result."""
        result = WorkflowResult(
            workflow_id="wf_1",
            definition_id="def_1",
            success=True,
            steps=[
                StepResult(step_id="s1", step_name="Step 1", status=StepStatus.COMPLETED),
                StepResult(step_id="s2", step_name="Step 2", status=StepStatus.COMPLETED),
            ],
            total_duration_ms=100.0,
        )

        step_result = result.get_step_result("s2")
        assert step_result is not None
        assert step_result.step_id == "s2"

    def test_get_step_result_not_found(self):
        """Test get_step_result returns None for unknown step."""
        result = WorkflowResult(
            workflow_id="wf_1",
            definition_id="def_1",
            success=True,
            steps=[],
            total_duration_ms=100.0,
        )

        step_result = result.get_step_result("nonexistent")
        assert step_result is None


# =============================================================================
# Workflow Context Tests
# =============================================================================


class TestWorkflowContext:
    """Test WorkflowContext class."""

    def test_create_context(self):
        """Test creating a workflow context."""
        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            inputs={"query": "test query"},
        )

        assert context.workflow_id == "wf_1"
        assert context.definition_id == "def_1"
        assert context.inputs["query"] == "test query"

    def test_get_input(self):
        """Test get_input method."""
        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            inputs={"value": 42},
        )

        assert context.get_input("value") == 42
        assert context.get_input("missing") is None
        assert context.get_input("missing", "default") == "default"

    def test_get_step_output(self):
        """Test get_step_output method."""
        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            step_outputs={"s1": {"result": "done"}},
        )

        assert context.get_step_output("s1") == {"result": "done"}
        assert context.get_step_output("missing") is None

    def test_get_set_state(self):
        """Test get_state and set_state methods."""
        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")

        context.set_state("counter", 5)
        assert context.get_state("counter") == 5
        assert context.get_state("missing") is None
        assert context.get_state("missing", 0) == 0

    def test_get_config(self):
        """Test get_config method."""
        context = WorkflowContext(
            workflow_id="wf_1",
            definition_id="def_1",
            current_step_config={"timeout": 30},
        )

        assert context.get_config("timeout") == 30
        assert context.get_config("missing") is None


# =============================================================================
# Workflow Config Tests
# =============================================================================


class TestWorkflowConfig:
    """Test WorkflowConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig()

        assert config.total_timeout_seconds == 3600.0
        assert config.step_timeout_seconds == 120.0
        assert config.stop_on_failure is True
        assert config.skip_optional_on_timeout is True
        assert config.enable_checkpointing is True
        assert config.checkpoint_interval_steps == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WorkflowConfig(
            total_timeout_seconds=600.0,
            step_timeout_seconds=30.0,
            stop_on_failure=False,
            enable_checkpointing=False,
        )

        assert config.total_timeout_seconds == 600.0
        assert config.step_timeout_seconds == 30.0
        assert config.stop_on_failure is False
        assert config.enable_checkpointing is False


# =============================================================================
# Execution Pattern Tests
# =============================================================================


class TestExecutionPattern:
    """Test ExecutionPattern enum."""

    def test_pattern_values(self):
        """Test all execution patterns are defined."""
        assert ExecutionPattern.SEQUENTIAL.value == "sequential"
        assert ExecutionPattern.PARALLEL.value == "parallel"
        assert ExecutionPattern.CONDITIONAL.value == "conditional"
        assert ExecutionPattern.LOOP.value == "loop"


# =============================================================================
# Step Status Tests
# =============================================================================


class TestStepStatus:
    """Test StepStatus enum."""

    def test_status_values(self):
        """Test all step statuses are defined."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.WAITING.value == "waiting"


# =============================================================================
# Step Definition Tests
# =============================================================================


class TestStepDefinition:
    """Test StepDefinition dataclass."""

    def test_create_step_definition(self):
        """Test creating a step definition."""
        step = StepDefinition(
            id="step_1",
            name="Test Step",
            step_type="agent",
        )

        assert step.id == "step_1"
        assert step.name == "Test Step"
        assert step.step_type == "agent"
        assert step.config == {}
        assert step.timeout_seconds == 120.0
        assert step.retries == 0
        assert step.optional is False

    def test_create_step_with_config(self):
        """Test creating a step with configuration."""
        step = StepDefinition(
            id="step_agent",
            name="Agent Step",
            step_type="agent",
            config={
                "agent_id": "claude",
                "prompt_template": "Analyze: {content}",
            },
            timeout_seconds=60.0,
            retries=2,
            next_steps=["step_2"],
        )

        assert step.config["agent_id"] == "claude"
        assert step.timeout_seconds == 60.0
        assert step.retries == 2
        assert "step_2" in step.next_steps

    def test_step_definition_to_dict(self):
        """Test step definition to_dict method."""
        step = StepDefinition(
            id="s1",
            name="Step 1",
            step_type="agent",
            config={"key": "value"},
        )

        data = step.to_dict()
        assert data["id"] == "s1"
        assert data["name"] == "Step 1"
        assert data["step_type"] == "agent"
        assert data["config"] == {"key": "value"}

    def test_step_definition_from_dict(self):
        """Test step definition from_dict method."""
        data = {
            "id": "s1",
            "name": "Step 1",
            "step_type": "agent",
            "config": {"key": "value"},
            "timeout_seconds": 60.0,
        }

        step = StepDefinition.from_dict(data)
        assert step.id == "s1"
        assert step.name == "Step 1"
        assert step.step_type == "agent"
        assert step.timeout_seconds == 60.0


# =============================================================================
# Transition Rule Tests
# =============================================================================


class TestTransitionRule:
    """Test TransitionRule dataclass."""

    def test_create_transition(self):
        """Test creating a transition rule."""
        rule = TransitionRule(
            id="tr_1",
            from_step="step_1",
            to_step="step_2",
            condition="True",
        )

        assert rule.id == "tr_1"
        assert rule.from_step == "step_1"
        assert rule.to_step == "step_2"
        assert rule.condition == "True"
        assert rule.priority == 0

    def test_transition_to_dict(self):
        """Test transition to_dict method."""
        rule = TransitionRule(
            id="tr_1",
            from_step="s1",
            to_step="s2",
            condition="True",
            priority=10,
        )

        data = rule.to_dict()
        assert data["id"] == "tr_1"
        assert data["from_step"] == "s1"
        assert data["to_step"] == "s2"
        assert data["priority"] == 10

    def test_transition_from_dict(self):
        """Test transition from_dict method."""
        data = {
            "from_step": "s1",
            "to_step": "s2",
            "condition": "outputs.get('success')",
            "priority": 5,
        }

        rule = TransitionRule.from_dict(data)
        assert rule.from_step == "s1"
        assert rule.to_step == "s2"
        assert rule.condition == "outputs.get('success')"
        assert rule.priority == 5


# =============================================================================
# Workflow Checkpoint Tests
# =============================================================================


class TestWorkflowCheckpoint:
    """Test WorkflowCheckpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = WorkflowCheckpoint(
            id="cp_1",
            workflow_id="wf_1",
            definition_id="def_1",
            current_step="s3",
            completed_steps=["s1", "s2"],
            step_outputs={"s1": {"out": 1}, "s2": {"out": 2}},
            context_state={"counter": 5},
            created_at=datetime.now(timezone.utc),
        )

        assert checkpoint.id == "cp_1"
        assert checkpoint.workflow_id == "wf_1"
        assert checkpoint.definition_id == "def_1"
        assert checkpoint.current_step == "s3"
        assert len(checkpoint.completed_steps) == 2

    def test_checkpoint_to_dict(self):
        """Test checkpoint to_dict method."""
        now = datetime.now(timezone.utc)
        checkpoint = WorkflowCheckpoint(
            id="cp_1",
            workflow_id="wf_1",
            definition_id="def_1",
            current_step="s1",
            completed_steps=["s0"],
            step_outputs={"s0": {"value": 1}},
            context_state={"state_key": "state_value"},
            created_at=now,
            checksum="abc123",
        )

        data = checkpoint.to_dict()
        assert data["id"] == "cp_1"
        assert data["workflow_id"] == "wf_1"
        assert data["completed_steps"] == ["s0"]
        assert data["checksum"] == "abc123"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for workflow engine."""

    @pytest.mark.asyncio
    async def test_complex_workflow(self, engine):
        """Test complex workflow with multiple paths and conditions."""
        definition = WorkflowDefinition(
            id="wf_complex",
            name="Complex Workflow",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="conditional_output",
                    config={"score": 85},
                ),
                StepDefinition(
                    id="high_score", name="High Score", step_type="test", next_steps=["final"]
                ),
                StepDefinition(
                    id="low_score", name="Low Score", step_type="test", next_steps=["final"]
                ),
                StepDefinition(id="final", name="Final", step_type="test"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_high",
                    from_step="start",
                    to_step="high_score",
                    condition="step_output.get('score', 0) >= 70",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_low",
                    from_step="start",
                    to_step="low_score",
                    condition="step_output.get('score', 0) < 70",
                    priority=5,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "start" in step_ids
        assert "high_score" in step_ids
        assert "final" in step_ids
        assert "low_score" not in step_ids

    @pytest.mark.asyncio
    async def test_workflow_with_all_features(self, checkpointing_engine, mock_checkpoint_store):
        """Test workflow exercising all major features."""
        definition = WorkflowDefinition(
            id="wf_all_features",
            name="All Features Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Input Processing",
                    step_type="test",
                    config={"output": "processed"},
                    next_steps=["s2"],
                ),
                StepDefinition(
                    id="s2",
                    name="Decision",
                    step_type="conditional_output",
                    config={"score": 75},
                ),
                StepDefinition(
                    id="s3_success",
                    name="Success Path",
                    step_type="test",
                    next_steps=["s4"],
                ),
                StepDefinition(
                    id="s3_failure",
                    name="Failure Path",
                    step_type="test",
                    next_steps=["s4"],
                ),
                StepDefinition(
                    id="s4",
                    name="Final Processing",
                    step_type="test",
                ),
            ],
            transitions=[
                TransitionRule(
                    id="tr_success",
                    from_step="s2",
                    to_step="s3_success",
                    condition="step_output.get('approved') == True",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_failure",
                    from_step="s2",
                    to_step="s3_failure",
                    condition="step_output.get('approved') == False",
                    priority=5,
                ),
            ],
        )

        result = await checkpointing_engine.execute(
            definition,
            inputs={"user": "test_user", "data": [1, 2, 3]},
        )

        assert result.success is True
        assert len(result.steps) == 4
        # Checkpoints should have been created
        assert mock_checkpoint_store.save_calls > 0

    @pytest.mark.asyncio
    async def test_error_recovery_with_optional_steps(self, engine):
        """Test workflow recovers from errors in optional steps."""
        definition = WorkflowDefinition(
            id="wf_recovery",
            name="Error Recovery Workflow",
            steps=[
                StepDefinition(id="s1", name="Success 1", step_type="test", next_steps=["s2"]),
                StepDefinition(
                    id="s2",
                    name="Optional Fail",
                    step_type="failing",
                    optional=True,
                    next_steps=["s3"],
                ),
                StepDefinition(id="s3", name="Success 2", step_type="test", next_steps=["s4"]),
                StepDefinition(
                    id="s4",
                    name="Optional Fail 2",
                    step_type="failing",
                    optional=True,
                    next_steps=["s5"],
                ),
                StepDefinition(id="s5", name="Final", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        # Workflow should complete despite optional failures
        step_ids = [s.step_id for s in result.steps]
        assert "s1" in step_ids
        assert "s2" in step_ids
        assert "s3" in step_ids
        assert "s4" in step_ids
        assert "s5" in step_ids


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_workflow(self, engine):
        """Test handling of empty workflow."""
        definition = WorkflowDefinition(
            id="wf_empty",
            name="Empty Workflow",
            steps=[],
        )

        result = await engine.execute(definition)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_single_step_workflow(self, engine):
        """Test workflow with single step."""
        definition = WorkflowDefinition(
            id="wf_single",
            name="Single Step Workflow",
            steps=[
                StepDefinition(id="only", name="Only Step", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_circular_reference_prevention(self, engine):
        """Test workflow handles potential circular references."""
        definition = WorkflowDefinition(
            id="wf_circular",
            name="Circular Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="test"),
                # Note: No circular next_steps, but transitions could create cycles
            ],
            transitions=[
                # This creates a potential cycle, but completed_steps should prevent infinite loop
                TransitionRule(id="tr_back", from_step="s2", to_step="s1", condition="False"),
            ],
        )

        result = await engine.execute(definition)

        # Should complete without infinite loop
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_long_chain_workflow(self, engine):
        """Test workflow with long chain of steps."""
        steps = []
        for i in range(20):
            next_steps = [f"s{i + 1}"] if i < 19 else []
            steps.append(
                StepDefinition(
                    id=f"s{i}",
                    name=f"Step {i}",
                    step_type="test",
                    next_steps=next_steps,
                )
            )

        definition = WorkflowDefinition(
            id="wf_long",
            name="Long Chain Workflow",
            steps=steps,
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert len(result.steps) == 20

    @pytest.mark.asyncio
    async def test_null_output_handling(self, engine):
        """Test handling of None output from step."""

        class NullOutputStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                return None

        engine.register_step_type("null_output", NullOutputStep)

        definition = WorkflowDefinition(
            id="wf_null",
            name="Null Output Workflow",
            steps=[
                StepDefinition(id="s1", name="Null", step_type="null_output", next_steps=["s2"]),
                StepDefinition(id="s2", name="After Null", step_type="test"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        # Null output should not break the workflow

    @pytest.mark.asyncio
    async def test_large_output_handling(self, engine):
        """Test handling of large output from step."""

        class LargeOutputStep(BaseStep):
            async def execute(self, context: WorkflowContext) -> Any:
                return {"large_data": "x" * 100000}

        engine.register_step_type("large_output", LargeOutputStep)

        definition = WorkflowDefinition(
            id="wf_large",
            name="Large Output Workflow",
            steps=[
                StepDefinition(id="s1", name="Large", step_type="large_output"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert len(result.final_output["large_data"]) == 100000
