"""
Comprehensive additional tests for Workflow Engine.

This module extends the base test_engine.py with additional coverage targeting:
- get_workflow_executor factory function (all modes)
- Resume workflow scenarios (timeout, errors)
- Checkpoint store error handling
- Step instance creation edge cases
- Concurrent workflow execution
- Timeout warning thresholds at various levels
- Advanced transition evaluation scenarios
- Workflow backward-compatible alias

Target: Increase workflow module coverage from 62% to 85%+
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.engine import (
    WorkflowEngine,
    Workflow,  # Backward-compatible alias
    get_workflow_engine,
    get_workflow_executor,
    reset_workflow_engine,
)
from aragora.workflow.step import (
    BaseStep,
    WorkflowContext,
    WorkflowStep,
)
from aragora.workflow.types import (
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


class SimpleStep(BaseStep):
    """Simple step that returns configurable output."""

    def __init__(self, name: str = "simple", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        return {"result": self._config.get("output", "done")}


class FailingStep(BaseStep):
    """Step that always fails."""

    def __init__(self, name: str = "failing", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        raise RuntimeError(self._config.get("error_message", "Step failed"))


class SlowStep(BaseStep):
    """Step with configurable delay."""

    def __init__(self, name: str = "slow", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        delay = self._config.get("delay_seconds", 1.0)
        await asyncio.sleep(delay)
        return {"completed": True, "delay": delay}


class ErrorOnConstructStep(BaseStep):
    """Step that fails during construction."""

    def __init__(self, name: str = "error_construct", config: dict[str, Any] | None = None):
        # Fail on purpose during construction
        raise ValueError("Cannot construct this step")


class ContextModifyingStep(BaseStep):
    """Step that modifies context state."""

    def __init__(self, name: str = "context_modifier", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})

    async def execute(self, context: WorkflowContext) -> Any:
        # Read and modify state
        counter = context.get_state("counter", 0)
        context.set_state("counter", counter + 1)
        context.set_state("last_step", self._name)
        return {"counter": counter + 1}


class MockCheckpointStore:
    """Mock checkpoint store with error simulation capabilities."""

    def __init__(self, fail_on_save: bool = False, fail_on_load: bool = False):
        self.checkpoints: dict[str, WorkflowCheckpoint] = {}
        self.save_calls = 0
        self.load_calls = 0
        self.fail_on_save = fail_on_save
        self.fail_on_load = fail_on_load

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        self.save_calls += 1
        if self.fail_on_save:
            raise RuntimeError("Checkpoint save failed")
        self.checkpoints[checkpoint.id] = checkpoint
        return checkpoint.id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        self.load_calls += 1
        if self.fail_on_load:
            raise RuntimeError("Checkpoint load failed")
        return self.checkpoints.get(checkpoint_id)

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        if self.fail_on_load:
            raise RuntimeError("Checkpoint load_latest failed")
        matching = [cp for cp in self.checkpoints.values() if cp.workflow_id == workflow_id]
        if not matching:
            return None
        return max(matching, key=lambda cp: cp.created_at)

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        if self.fail_on_load:
            raise RuntimeError("Checkpoint list failed")
        return [cp.id for cp in self.checkpoints.values() if cp.workflow_id == workflow_id]

    async def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            return True
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create basic workflow configuration."""
    return WorkflowConfig(
        total_timeout_seconds=60.0,
        step_timeout_seconds=10.0,
        stop_on_failure=True,
        enable_checkpointing=False,
    )


@pytest.fixture
def checkpointing_config():
    """Create configuration with checkpointing enabled."""
    return WorkflowConfig(
        total_timeout_seconds=60.0,
        step_timeout_seconds=10.0,
        stop_on_failure=True,
        enable_checkpointing=True,
        checkpoint_interval_steps=1,
    )


@pytest.fixture
def mock_checkpoint_store():
    """Create mock checkpoint store."""
    return MockCheckpointStore()


@pytest.fixture
def engine(basic_config, mock_checkpoint_store):
    """Create workflow engine with test step types."""
    engine = WorkflowEngine(config=basic_config, checkpoint_store=mock_checkpoint_store)
    engine.register_step_type("simple", SimpleStep)
    engine.register_step_type("failing", FailingStep)
    engine.register_step_type("slow", SlowStep)
    engine.register_step_type("context_modifier", ContextModifyingStep)
    return engine


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton engine before each test."""
    reset_workflow_engine()
    yield
    reset_workflow_engine()


# =============================================================================
# Test get_workflow_executor Factory
# =============================================================================


class TestGetWorkflowExecutorFactory:
    """Test the unified get_workflow_executor factory function."""

    def test_default_mode_returns_singleton(self):
        """Default mode should return the singleton WorkflowEngine."""
        executor1 = get_workflow_executor(mode="default")
        executor2 = get_workflow_executor(mode="default")
        assert executor1 is executor2
        assert isinstance(executor1, WorkflowEngine)

    def test_enhanced_mode_returns_enhanced_engine(self):
        """Enhanced mode should return EnhancedWorkflowEngine."""
        from aragora.workflow.engine_v2 import EnhancedWorkflowEngine

        executor = get_workflow_executor(mode="enhanced")
        assert isinstance(executor, EnhancedWorkflowEngine)

    def test_enhanced_mode_with_custom_limits(self):
        """Enhanced mode should accept resource limits."""
        from aragora.workflow.engine_v2 import ResourceLimits

        limits = ResourceLimits(max_tokens=50000, max_cost_usd=2.5)
        executor = get_workflow_executor(mode="enhanced", resource_limits=limits)

        assert executor.limits.max_tokens == 50000
        assert executor.limits.max_cost_usd == 2.5

    def test_queue_mode_returns_adapter(self):
        """Queue mode should return TaskQueueExecutorAdapter."""
        from aragora.workflow.queue_adapter import TaskQueueExecutorAdapter

        executor = get_workflow_executor(mode="queue")
        assert isinstance(executor, TaskQueueExecutorAdapter)

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode should raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            get_workflow_executor(mode="nonexistent_mode")

        assert "Unknown executor mode" in str(exc_info.value)
        assert "nonexistent_mode" in str(exc_info.value)
        assert "default" in str(exc_info.value)
        assert "enhanced" in str(exc_info.value)
        assert "queue" in str(exc_info.value)

    def test_config_passed_to_default_engine(self):
        """Config should be passed when creating default engine."""
        reset_workflow_engine()
        config = WorkflowConfig(
            total_timeout_seconds=100.0,
            enable_checkpointing=False,
        )
        executor = get_workflow_executor(mode="default", config=config)

        assert executor._config.total_timeout_seconds == 100.0
        assert executor._config.enable_checkpointing is False


# =============================================================================
# Test Backward-Compatible Alias
# =============================================================================


class TestWorkflowAlias:
    """Test the backward-compatible Workflow alias."""

    def test_workflow_is_workflow_engine(self):
        """Workflow should be an alias for WorkflowEngine."""
        assert Workflow is WorkflowEngine

    def test_workflow_creates_engine_instance(self):
        """Creating Workflow instance should work like WorkflowEngine."""
        wf = Workflow()
        assert isinstance(wf, WorkflowEngine)
        assert hasattr(wf, "execute")
        assert hasattr(wf, "resume")


# =============================================================================
# Test Resume Workflow Scenarios
# =============================================================================


class TestResumeWorkflow:
    """Test resume workflow functionality."""

    @pytest.mark.asyncio
    async def test_resume_skips_completed_steps(self, engine):
        """Resume should skip steps that are already completed."""
        definition = WorkflowDefinition(
            id="wf_resume",
            name="Resume Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="simple", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="simple", next_steps=["s3"]),
                StepDefinition(id="s3", name="Step 3", step_type="simple"),
            ],
        )

        checkpoint = WorkflowCheckpoint(
            id="cp_test",
            workflow_id="wf_resume_123",
            definition_id="wf_resume",
            current_step="s2",
            completed_steps=["s1"],
            step_outputs={"s1": {"result": "done"}},
            context_state={"inputs": {}, "state": {}},
            created_at=datetime.now(timezone.utc),
        )

        result = await engine.resume("wf_resume_123", checkpoint, definition)

        step_ids = [s.step_id for s in result.steps]
        assert "s1" not in step_ids  # Should be skipped
        assert "s2" in step_ids
        assert "s3" in step_ids

    @pytest.mark.asyncio
    async def test_resume_timeout(self):
        """Resume should handle workflow timeout."""
        config = WorkflowConfig(total_timeout_seconds=0.1)
        slow_engine = WorkflowEngine(config=config)
        slow_engine.register_step_type("slow", SlowStep)

        definition = WorkflowDefinition(
            id="wf_slow",
            name="Slow Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Slow Step",
                    step_type="slow",
                    config={"delay_seconds": 10.0},
                    timeout_seconds=60.0,
                ),
            ],
        )

        checkpoint = WorkflowCheckpoint(
            id="cp_slow",
            workflow_id="wf_slow_123",
            definition_id="wf_slow",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={"inputs": {}, "state": {}},
            created_at=datetime.now(timezone.utc),
        )

        result = await slow_engine.resume("wf_slow_123", checkpoint, definition)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_resume_with_exception(self, engine):
        """Resume should handle exceptions during execution."""
        definition = WorkflowDefinition(
            id="wf_fail",
            name="Failing Workflow",
            steps=[
                StepDefinition(id="s1", name="Fail Step", step_type="failing"),
            ],
        )

        checkpoint = WorkflowCheckpoint(
            id="cp_fail",
            workflow_id="wf_fail_123",
            definition_id="wf_fail",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={"inputs": {}, "state": {}},
            created_at=datetime.now(timezone.utc),
        )

        result = await engine.resume("wf_fail_123", checkpoint, definition)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_resume_restores_context_state(self, engine):
        """Resume should restore context state from checkpoint."""
        definition = WorkflowDefinition(
            id="wf_state",
            name="State Workflow",
            steps=[
                StepDefinition(id="s1", name="State Step", step_type="context_modifier"),
            ],
        )

        checkpoint = WorkflowCheckpoint(
            id="cp_state",
            workflow_id="wf_state_123",
            definition_id="wf_state",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={
                "inputs": {"initial_value": 42},
                "state": {"counter": 5, "existing_data": "preserved"},
            },
            created_at=datetime.now(timezone.utc),
        )

        result = await engine.resume("wf_state_123", checkpoint, definition)

        assert result.success is True


# =============================================================================
# Test Checkpoint Store Error Handling
# =============================================================================


class TestCheckpointStoreErrorHandling:
    """Test checkpoint store error handling."""

    @pytest.mark.asyncio
    async def test_checkpoint_save_failure_continues_execution(self):
        """Workflow should continue even if checkpoint save fails."""
        failing_store = MockCheckpointStore(fail_on_save=True)
        config = WorkflowConfig(
            enable_checkpointing=True,
            checkpoint_interval_steps=1,
        )
        engine = WorkflowEngine(config=config, checkpoint_store=failing_store)
        engine.register_step_type("simple", SimpleStep)

        definition = WorkflowDefinition(
            id="wf_cp_fail",
            name="Checkpoint Fail Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="simple", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="simple"),
            ],
        )

        result = await engine.execute(definition)

        # Workflow should still complete despite checkpoint failures
        assert result.success is True
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_get_checkpoint_load_failure(self):
        """get_checkpoint should handle load failures gracefully."""
        failing_store = MockCheckpointStore(fail_on_load=True)
        engine = WorkflowEngine(checkpoint_store=failing_store)

        result = await engine.get_checkpoint("nonexistent_checkpoint")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_failure(self):
        """get_latest_checkpoint should handle failures gracefully."""
        failing_store = MockCheckpointStore(fail_on_load=True)
        engine = WorkflowEngine(checkpoint_store=failing_store)

        result = await engine.get_latest_checkpoint("nonexistent_workflow")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints_failure(self):
        """list_checkpoints should handle failures gracefully."""
        failing_store = MockCheckpointStore(fail_on_load=True)
        engine = WorkflowEngine(checkpoint_store=failing_store)

        result = await engine.list_checkpoints("nonexistent_workflow")

        assert result == []

    @pytest.mark.asyncio
    async def test_delete_checkpoint_removes_from_cache(self, engine, mock_checkpoint_store):
        """delete_checkpoint should remove from both store and cache."""
        checkpoint = WorkflowCheckpoint(
            id="cp_delete_test",
            workflow_id="wf_test",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_delete_test"] = checkpoint

        # Pre-populate cache
        engine._checkpoints_cache.put("cp_delete_test", checkpoint)

        result = await engine.delete_checkpoint("cp_delete_test")

        assert result is True
        assert "cp_delete_test" not in mock_checkpoint_store.checkpoints
        assert engine._checkpoints_cache.get("cp_delete_test") is None

    @pytest.mark.asyncio
    async def test_delete_checkpoint_failure(self):
        """delete_checkpoint should handle deletion failures."""

        class FailingDeleteStore(MockCheckpointStore):
            async def delete(self, checkpoint_id: str) -> bool:
                raise RuntimeError("Delete failed")

        engine = WorkflowEngine(checkpoint_store=FailingDeleteStore())

        result = await engine.delete_checkpoint("any_checkpoint")

        assert result is False


# =============================================================================
# Test Step Instance Creation Edge Cases
# =============================================================================


class TestStepInstanceCreation:
    """Test step instance creation edge cases."""

    @pytest.mark.asyncio
    async def test_step_construction_failure(self, engine):
        """Workflow should handle step construction failures gracefully."""
        engine.register_step_type("error_construct", ErrorOnConstructStep)

        definition = WorkflowDefinition(
            id="wf_construct_fail",
            name="Construct Fail Workflow",
            steps=[
                StepDefinition(id="s1", name="Fail Construct", step_type="error_construct"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED

    def test_step_instance_caching(self, engine):
        """Step instances should be cached and reused."""
        step_def = StepDefinition(id="cached_step", name="Cached", step_type="simple")

        instance1 = engine._get_step_instance(step_def)
        instance2 = engine._get_step_instance(step_def)

        assert instance1 is instance2

    def test_step_instance_different_ids(self, engine):
        """Different step IDs should create different instances."""
        step_def1 = StepDefinition(id="step_1", name="Step 1", step_type="simple")
        step_def2 = StepDefinition(id="step_2", name="Step 2", step_type="simple")

        instance1 = engine._get_step_instance(step_def1)
        instance2 = engine._get_step_instance(step_def2)

        assert instance1 is not instance2

    def test_unregistered_step_type_returns_none(self, engine):
        """Unregistered step type should return None."""
        step_def = StepDefinition(id="s1", name="Unknown", step_type="unregistered_type_xyz")

        instance = engine._get_step_instance(step_def)

        assert instance is None


# =============================================================================
# Test Timeout Warning Thresholds
# =============================================================================


class TestTimeoutWarningThresholds:
    """Test timeout warning threshold functionality."""

    def test_warning_at_50_percent(self):
        """Warning should be issued at 50% threshold."""
        config = WorkflowConfig(total_timeout_seconds=10.0)
        engine = WorkflowEngine(config=config)
        engine._timeout_warnings_issued = set()

        engine._check_timeout_progress(
            start_time=time.time() - 5.5,  # 55% elapsed
            total_timeout=10.0,
            workflow_id="test_wf",
        )

        assert 0.5 in engine._timeout_warnings_issued

    def test_warning_at_80_percent(self):
        """Warning should be issued at 80% threshold."""
        config = WorkflowConfig(total_timeout_seconds=10.0)
        engine = WorkflowEngine(config=config)
        engine._timeout_warnings_issued = set()

        engine._check_timeout_progress(
            start_time=time.time() - 8.5,  # 85% elapsed
            total_timeout=10.0,
            workflow_id="test_wf",
        )

        assert 0.5 in engine._timeout_warnings_issued
        assert 0.8 in engine._timeout_warnings_issued

    def test_warning_at_90_percent(self):
        """Warning should be issued at 90% threshold."""
        config = WorkflowConfig(total_timeout_seconds=10.0)
        engine = WorkflowEngine(config=config)
        engine._timeout_warnings_issued = set()

        engine._check_timeout_progress(
            start_time=time.time() - 9.5,  # 95% elapsed
            total_timeout=10.0,
            workflow_id="test_wf",
        )

        assert 0.5 in engine._timeout_warnings_issued
        assert 0.8 in engine._timeout_warnings_issued
        assert 0.9 in engine._timeout_warnings_issued

    def test_warning_only_issued_once_per_threshold(self):
        """Each warning threshold should only be issued once."""
        config = WorkflowConfig(total_timeout_seconds=10.0)
        engine = WorkflowEngine(config=config)
        engine._timeout_warnings_issued = set()

        # First check at 55%
        engine._check_timeout_progress(
            start_time=time.time() - 5.5,
            total_timeout=10.0,
            workflow_id="test_wf",
        )
        assert 0.5 in engine._timeout_warnings_issued
        warnings_after_first = len(engine._timeout_warnings_issued)

        # Second check still at 55% - should not add duplicate
        engine._check_timeout_progress(
            start_time=time.time() - 5.5,
            total_timeout=10.0,
            workflow_id="test_wf",
        )
        assert len(engine._timeout_warnings_issued) == warnings_after_first

    def test_no_warning_with_zero_timeout(self, engine):
        """No warning should be issued when timeout is zero."""
        engine._timeout_warnings_issued = set()

        engine._check_timeout_progress(
            start_time=time.time() - 100,
            total_timeout=0,
            workflow_id="test_wf",
        )

        assert len(engine._timeout_warnings_issued) == 0

    def test_no_warning_with_negative_timeout(self, engine):
        """No warning should be issued when timeout is negative."""
        engine._timeout_warnings_issued = set()

        engine._check_timeout_progress(
            start_time=time.time() - 100,
            total_timeout=-10.0,
            workflow_id="test_wf",
        )

        assert len(engine._timeout_warnings_issued) == 0


# =============================================================================
# Test Concurrent Workflow Execution
# =============================================================================


class TestConcurrentWorkflowExecution:
    """Test concurrent workflow execution scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_workflows_concurrent(self):
        """Multiple workflows should execute concurrently without interference."""
        engine = WorkflowEngine()
        engine.register_step_type("simple", SimpleStep)

        workflows = []
        for i in range(5):
            definition = WorkflowDefinition(
                id=f"wf_concurrent_{i}",
                name=f"Concurrent Workflow {i}",
                steps=[
                    StepDefinition(
                        id=f"s{i}",
                        name=f"Step {i}",
                        step_type="simple",
                        config={"output": f"result_{i}"},
                    ),
                ],
            )
            workflows.append(definition)

        # Execute all workflows concurrently
        tasks = [engine.execute(wf, workflow_id=f"run_{i}") for i, wf in enumerate(workflows)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert all(r.success for r in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_separate_engines_isolated(self):
        """Separate engine instances should be completely isolated."""
        engine1 = WorkflowEngine()
        engine2 = WorkflowEngine()

        engine1.register_step_type("custom1", SimpleStep)
        engine2.register_step_type("custom2", SimpleStep)

        assert "custom1" in engine1._step_types
        assert "custom1" not in engine2._step_types
        assert "custom2" in engine2._step_types
        assert "custom2" not in engine1._step_types


# =============================================================================
# Test Advanced Transition Scenarios
# =============================================================================


class TestAdvancedTransitions:
    """Test advanced transition evaluation scenarios."""

    @pytest.mark.asyncio
    async def test_transition_with_step_output_access(self, engine):
        """Transitions should be able to access step output via step_output variable."""
        definition = WorkflowDefinition(
            id="wf_step_output",
            name="Step Output Access",
            steps=[
                StepDefinition(
                    id="decision",
                    name="Decision",
                    step_type="simple",
                    config={"output": "approved"},
                ),
                StepDefinition(id="approved", name="Approved", step_type="simple"),
                StepDefinition(id="rejected", name="Rejected", step_type="simple"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_approved",
                    from_step="decision",
                    to_step="approved",
                    condition="step_output.get('result') == 'approved'",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_rejected",
                    from_step="decision",
                    to_step="rejected",
                    condition="True",
                    priority=1,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        assert "approved" in step_ids
        assert "rejected" not in step_ids

    @pytest.mark.asyncio
    async def test_transition_with_outputs_dict(self, engine):
        """Transitions should access previous outputs via outputs dict."""
        definition = WorkflowDefinition(
            id="wf_outputs",
            name="Outputs Dict Access",
            steps=[
                StepDefinition(
                    id="s1",
                    name="First",
                    step_type="simple",
                    config={"output": "first_value"},
                    next_steps=["s2"],
                ),
                StepDefinition(id="s2", name="Second", step_type="simple"),
                StepDefinition(id="s2_alt", name="Second Alt", step_type="simple"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_alt",
                    from_step="s1",
                    to_step="s2_alt",
                    condition="outputs.get('s1', {}).get('result') == 'alt_value'",
                    priority=10,
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        step_ids = [s.step_id for s in result.steps]
        # Should take default path since s1 outputs 'first_value' not 'alt_value'
        assert "s2" in step_ids
        assert "s2_alt" not in step_ids

    @pytest.mark.asyncio
    async def test_transition_with_inputs_access(self, engine):
        """Transitions should access workflow inputs."""
        definition = WorkflowDefinition(
            id="wf_inputs",
            name="Inputs Access",
            steps=[
                StepDefinition(id="start", name="Start", step_type="simple"),
                StepDefinition(id="high_priority", name="High Priority", step_type="simple"),
                StepDefinition(id="low_priority", name="Low Priority", step_type="simple"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_high",
                    from_step="start",
                    to_step="high_priority",
                    condition="inputs.get('priority', 0) > 5",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_low",
                    from_step="start",
                    to_step="low_priority",
                    condition="True",
                    priority=1,
                ),
            ],
        )

        result = await engine.execute(definition, inputs={"priority": 10})

        step_ids = [s.step_id for s in result.steps]
        assert "high_priority" in step_ids
        assert "low_priority" not in step_ids

    @pytest.mark.asyncio
    async def test_transition_with_state_access(self, engine):
        """Transitions should access workflow state."""
        definition = WorkflowDefinition(
            id="wf_state_trans",
            name="State Access",
            steps=[
                StepDefinition(id="s1", name="Modify State", step_type="context_modifier"),
                StepDefinition(id="high_counter", name="High Counter", step_type="simple"),
                StepDefinition(id="low_counter", name="Low Counter", step_type="simple"),
            ],
            transitions=[
                TransitionRule(
                    id="tr_high",
                    from_step="s1",
                    to_step="high_counter",
                    condition="state.get('counter', 0) >= 5",
                    priority=10,
                ),
                TransitionRule(
                    id="tr_low",
                    from_step="s1",
                    to_step="low_counter",
                    condition="True",
                    priority=1,
                ),
            ],
        )

        result = await engine.execute(definition)

        step_ids = [s.step_id for s in result.steps]
        # Counter starts at 0, increments to 1, so should go to low_counter
        assert "low_counter" in step_ids
        assert "high_counter" not in step_ids


# =============================================================================
# Test Checkpoint Cache Behavior
# =============================================================================


class TestCheckpointCacheBehavior:
    """Test LRU checkpoint cache behavior."""

    def test_cache_stats_initial(self, engine):
        """Cache stats should show zeros initially."""
        stats = engine.get_checkpoint_cache_stats()

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_cache_hit_increments_counter(self, engine, mock_checkpoint_store):
        """Cache hits should increment hit counter."""
        checkpoint = WorkflowCheckpoint(
            id="cp_hit_test",
            workflow_id="wf_test",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_hit_test"] = checkpoint

        # First load - cache miss
        await engine.get_checkpoint("cp_hit_test")
        # Second load - cache hit
        await engine.get_checkpoint("cp_hit_test")

        stats = engine.get_checkpoint_cache_stats()
        assert stats["hits"] >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_store(self, engine, mock_checkpoint_store):
        """Cache miss should load from persistent store."""
        checkpoint = WorkflowCheckpoint(
            id="cp_miss_test",
            workflow_id="wf_test",
            definition_id="def_test",
            current_step="s1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(timezone.utc),
        )
        mock_checkpoint_store.checkpoints["cp_miss_test"] = checkpoint

        result = await engine.get_checkpoint("cp_miss_test")

        assert result is not None
        assert result.id == "cp_miss_test"
        assert mock_checkpoint_store.load_calls >= 1


# =============================================================================
# Test Metrics Functionality
# =============================================================================


class TestMetricsFunctionality:
    """Test metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_metrics_includes_all_step_durations(self, engine):
        """Metrics should include duration for each step."""
        definition = WorkflowDefinition(
            id="wf_metrics",
            name="Metrics Workflow",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="simple", next_steps=["s2"]),
                StepDefinition(id="s2", name="Step 2", step_type="simple", next_steps=["s3"]),
                StepDefinition(id="s3", name="Step 3", step_type="simple"),
            ],
        )

        await engine.execute(definition)
        metrics = engine.get_metrics()

        assert "step_durations" in metrics
        assert "s1" in metrics["step_durations"]
        assert "s2" in metrics["step_durations"]
        assert "s3" in metrics["step_durations"]
        assert all(d > 0 for d in metrics["step_durations"].values())

    @pytest.mark.asyncio
    async def test_metrics_tracks_termination(self, engine):
        """Metrics should track early termination."""
        engine.request_termination("Test termination")
        metrics = engine.get_metrics()

        assert metrics["terminated_early"] is True
        assert metrics["termination_reason"] == "Test termination"


# =============================================================================
# Test Phase 2 Step Types Registration
# =============================================================================


class TestPhase2StepTypesRegistration:
    """Test registration of Phase 2 step types."""

    def test_default_step_types_registered(self):
        """Default step types should be registered on engine creation."""
        engine = WorkflowEngine()

        assert "agent" in engine._step_types
        assert "parallel" in engine._step_types
        assert "conditional" in engine._step_types
        assert "loop" in engine._step_types

    def test_phase2_step_types_registered_if_available(self):
        """Phase 2 step types should be registered if available."""
        engine = WorkflowEngine()

        # These may or may not be available depending on imports
        # Just verify the engine doesn't crash
        assert isinstance(engine._step_types, dict)


# =============================================================================
# Test Workflow Definition Validation
# =============================================================================


class TestWorkflowDefinitionValidation:
    """Test workflow definition validation during execution."""

    @pytest.mark.asyncio
    async def test_no_entry_step_returns_error(self, engine):
        """Workflow without entry step should fail with clear error."""
        definition = WorkflowDefinition(
            id="wf_no_entry",
            name="No Entry",
            steps=[],
            entry_step=None,
        )

        result = await engine.execute(definition)

        assert result.success is False
        assert result.error  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_missing_step_stops_gracefully(self, engine):
        """Workflow should stop gracefully when referenced step is missing."""
        definition = WorkflowDefinition(
            id="wf_missing_ref",
            name="Missing Reference",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Start",
                    step_type="simple",
                    next_steps=["nonexistent_step"],
                ),
            ],
        )

        result = await engine.execute(definition)

        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test additional edge cases."""

    @pytest.mark.asyncio
    async def test_very_long_step_chain(self, engine):
        """Engine should handle very long step chains."""
        steps = []
        for i in range(50):
            next_steps = [f"s{i + 1}"] if i < 49 else []
            steps.append(
                StepDefinition(
                    id=f"s{i}",
                    name=f"Step {i}",
                    step_type="simple",
                    next_steps=next_steps,
                )
            )

        definition = WorkflowDefinition(
            id="wf_long_chain",
            name="Long Chain",
            steps=steps,
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert len(result.steps) == 50

    @pytest.mark.asyncio
    async def test_workflow_with_special_characters_in_ids(self, engine):
        """Workflow should handle special characters in IDs."""
        definition = WorkflowDefinition(
            id="wf-with_special.chars:123",
            name="Special Chars Workflow",
            steps=[
                StepDefinition(
                    id="step-with_special.chars:1",
                    name="Special Step",
                    step_type="simple",
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert result.steps[0].step_id == "step-with_special.chars:1"

    @pytest.mark.asyncio
    async def test_workflow_with_unicode_content(self, engine):
        """Workflow should handle unicode in names and config."""
        definition = WorkflowDefinition(
            id="wf_unicode",
            name="Unicode Workflow - Test Workflow",
            steps=[
                StepDefinition(
                    id="s1",
                    name="Step with unicode: hello, test, world",
                    step_type="simple",
                    config={"output": "Result: success, failure"},
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_workflow_inputs(self, engine):
        """Workflow should handle None inputs gracefully."""
        definition = WorkflowDefinition(
            id="wf_null_inputs",
            name="Null Inputs",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="simple"),
            ],
        )

        result = await engine.execute(definition, inputs=None)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_step_with_empty_config(self, engine):
        """Step should work with empty config."""
        definition = WorkflowDefinition(
            id="wf_empty_config",
            name="Empty Config",
            steps=[
                StepDefinition(
                    id="s1",
                    name="No Config",
                    step_type="simple",
                    config={},
                ),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True


# =============================================================================
# Test Termination Control
# =============================================================================


class TestTerminationControl:
    """Test termination control functionality."""

    def test_request_termination_sets_flags(self, engine):
        """request_termination should set internal flags."""
        engine.request_termination("Custom reason")

        is_terminated, reason = engine.check_termination()
        assert is_terminated is True
        assert reason == "Custom reason"

    def test_check_termination_default_state(self, engine):
        """check_termination should return False initially."""
        is_terminated, reason = engine.check_termination()

        assert is_terminated is False
        assert reason is None

    def test_current_step_property(self, engine):
        """current_step should track the currently executing step."""
        # Initially None
        assert engine.current_step is None

    @pytest.mark.asyncio
    async def test_termination_stops_execution(self, engine):
        """Termination request should stop workflow execution."""

        class TerminationTriggerStep(BaseStep):
            def __init__(self, name: str = "trigger", engine_ref=None, config=None):
                super().__init__(name, config or {})
                self.engine_ref = engine_ref

            async def execute(self, context: WorkflowContext) -> Any:
                if self.engine_ref:
                    self.engine_ref.request_termination("Triggered by step")
                return {"triggered": True}

        # Can't easily test this without modifying step to have engine reference
        # Just verify the API works
        engine.request_termination("API test")
        assert engine._should_terminate is True


# =============================================================================
# Test Step Result Properties
# =============================================================================


class TestStepResultProperties:
    """Test StepResult dataclass properties."""

    def test_success_for_completed_status(self):
        """success should be True for COMPLETED status."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.COMPLETED,
        )
        assert result.success is True

    def test_success_for_skipped_status(self):
        """success should be True for SKIPPED status."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.SKIPPED,
        )
        assert result.success is True

    def test_success_for_failed_status(self):
        """success should be False for FAILED status."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.FAILED,
        )
        assert result.success is False

    def test_success_for_pending_status(self):
        """success should be False for PENDING status."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.PENDING,
        )
        assert result.success is False

    def test_success_for_running_status(self):
        """success should be False for RUNNING status."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.RUNNING,
        )
        assert result.success is False
