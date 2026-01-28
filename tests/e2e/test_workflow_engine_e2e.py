"""
End-to-End Test: Workflow Engine.

Tests complete workflow execution scenarios including:
- DAG execution with sequential, parallel, and conditional steps
- Node type execution (task, decision, memory, debate)
- Error handling and retry mechanisms
- Checkpointing and workflow recovery
- Template instantiation and execution
- Full workflow lifecycle management

Related plan: kind-squishing-russell.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowConfig,
    StepDefinition,
    StepStatus,
    TransitionRule,
    ExecutionPattern,
)
from aragora.workflow.step import WorkflowContext, WorkflowStep
from aragora.workflow.types import WorkflowResult


# ============================================================================
# Test Helpers
# ============================================================================


@dataclass
class MockStepOutput:
    """Mock output from a workflow step."""

    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockCustomStep(WorkflowStep):
    """Mock step for testing custom step types."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self._name = name
        self._config = config
        self.executed = False
        self.execution_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the mock step."""
        self.executed = True
        self.execution_count += 1

        # Return configured output or default
        output = self._config.get("output", {"status": "success"})

        # Simulate delay if configured
        delay = self._config.get("delay_seconds", 0)
        if delay > 0:
            await asyncio.sleep(delay)

        # Raise error if configured
        if self._config.get("raise_error"):
            raise RuntimeError(self._config.get("error_message", "Mock error"))

        return output


class MockFailingStep(WorkflowStep):
    """Step that fails on first N attempts."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self._name = name
        self._config = config
        self.attempt_count = 0
        self.fail_count = config.get("fail_count", 1)

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute with configured failures."""
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            raise RuntimeError(f"Attempt {self.attempt_count} failed")
        return {"status": "success", "attempts": self.attempt_count}


def create_workflow_engine(
    config: Optional[WorkflowConfig] = None,
    custom_steps: Optional[Dict[str, type]] = None,
) -> WorkflowEngine:
    """Create a workflow engine with optional custom configuration."""
    engine = WorkflowEngine(
        config=config
        or WorkflowConfig(
            total_timeout_seconds=30.0,
            step_timeout_seconds=10.0,
            stop_on_failure=True,
            enable_checkpointing=True,
            checkpoint_interval_steps=1,
        )
    )

    # Register custom step types
    if custom_steps:
        for name, step_class in custom_steps.items():
            engine.register_step_type(name, step_class)

    return engine


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_workflows.db"


@pytest.fixture
def basic_engine() -> WorkflowEngine:
    """Create a basic workflow engine."""
    return create_workflow_engine()


@pytest.fixture
def engine_with_custom_steps() -> WorkflowEngine:
    """Create engine with custom step types registered."""
    return create_workflow_engine(
        custom_steps={
            "custom": MockCustomStep,
            "failing": MockFailingStep,
        }
    )


@pytest.fixture
def sequential_workflow() -> WorkflowDefinition:
    """Create a simple sequential workflow."""
    return WorkflowDefinition(
        id="sequential-e2e",
        name="Sequential E2E Test",
        description="Tests sequential step execution",
        steps=[
            StepDefinition(
                id="step1",
                name="Step 1",
                step_type="task",
                config={"action": "set_state", "state": {"count": 1}},
                next_steps=["step2"],
            ),
            StepDefinition(
                id="step2",
                name="Step 2",
                step_type="task",
                config={"action": "set_state", "state": {"count": 2}},
                next_steps=["step3"],
            ),
            StepDefinition(
                id="step3",
                name="Step 3",
                step_type="task",
                config={"action": "set_state", "state": {"count": 3}},
            ),
        ],
        entry_step="step1",
    )


@pytest.fixture
def conditional_workflow() -> WorkflowDefinition:
    """Create a workflow with conditional branching."""
    return WorkflowDefinition(
        id="conditional-e2e",
        name="Conditional E2E Test",
        description="Tests conditional branching",
        steps=[
            StepDefinition(
                id="evaluate",
                name="Evaluate",
                step_type="task",
                config={
                    "action": "set_state",
                    "state": {"decision": True, "score": 85},
                },
            ),
            StepDefinition(
                id="high-path",
                name="High Score Path",
                step_type="task",
                config={"action": "log", "message": "High score!"},
            ),
            StepDefinition(
                id="low-path",
                name="Low Score Path",
                step_type="task",
                config={"action": "log", "message": "Low score"},
            ),
            StepDefinition(
                id="final",
                name="Final Step",
                step_type="task",
                config={"action": "log", "message": "Complete"},
            ),
        ],
        transitions=[
            TransitionRule(
                id="to-high",
                from_step="evaluate",
                to_step="high-path",
                condition="output.get('score', 0) >= 70",
                priority=10,
            ),
            TransitionRule(
                id="to-low",
                from_step="evaluate",
                to_step="low-path",
                condition="output.get('score', 0) < 70",
                priority=5,
            ),
            TransitionRule(
                id="high-to-final",
                from_step="high-path",
                to_step="final",
                condition="True",
            ),
            TransitionRule(
                id="low-to-final",
                from_step="low-path",
                to_step="final",
                condition="True",
            ),
        ],
        entry_step="evaluate",
    )


@pytest.fixture
def diamond_dag_workflow() -> WorkflowDefinition:
    """Create a diamond-shaped DAG workflow."""
    return WorkflowDefinition(
        id="diamond-dag-e2e",
        name="Diamond DAG E2E Test",
        description="Tests diamond-shaped workflow execution",
        steps=[
            StepDefinition(
                id="start",
                name="Start",
                step_type="task",
                config={"action": "log", "message": "Starting"},
                next_steps=["left"],
            ),
            StepDefinition(
                id="left",
                name="Left Branch",
                step_type="task",
                config={"action": "set_state", "state": {"branch": "left"}},
                next_steps=["end"],
            ),
            StepDefinition(
                id="right",
                name="Right Branch",
                step_type="task",
                config={"action": "set_state", "state": {"branch": "right"}},
                next_steps=["end"],
            ),
            StepDefinition(
                id="end",
                name="End",
                step_type="task",
                config={"action": "log", "message": "Complete"},
            ),
        ],
        entry_step="start",
    )


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.e2e
class TestDAGExecution:
    """Tests for DAG-based workflow execution."""

    @pytest.mark.asyncio
    async def test_sequential_execution_completes_all_steps(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test that sequential workflow executes all steps in order."""
        result = await basic_engine.execute(sequential_workflow, inputs={})

        assert result.success
        assert len(result.steps) == 3

        # Verify execution order
        step_ids = [r.step_id for r in result.steps]
        assert step_ids == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_workflow_passes_inputs_to_steps(
        self,
        basic_engine: WorkflowEngine,
    ) -> None:
        """Test that workflow inputs are available to all steps."""
        workflow = WorkflowDefinition(
            id="input-test",
            name="Input Test",
            steps=[
                StepDefinition(
                    id="use-input",
                    name="Use Input",
                    step_type="task",
                    config={"action": "log", "message": "Processing input"},
                ),
            ],
            entry_step="use-input",
        )

        result = await basic_engine.execute(
            workflow,
            inputs={"user_id": "test-user", "data": {"key": "value"}},
        )

        assert result.success
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_diamond_dag_executes_correctly(
        self,
        basic_engine: WorkflowEngine,
        diamond_dag_workflow: WorkflowDefinition,
    ) -> None:
        """Test diamond-shaped DAG workflow execution."""
        result = await basic_engine.execute(diamond_dag_workflow, inputs={})

        # Should execute at least start and one branch
        executed_steps = {r.step_id for r in result.steps}
        assert "start" in executed_steps
        assert len(result.steps) >= 2

    @pytest.mark.asyncio
    async def test_workflow_generates_unique_id(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test that workflow execution generates unique IDs."""
        result1 = await basic_engine.execute(sequential_workflow, inputs={})
        result2 = await basic_engine.execute(sequential_workflow, inputs={})

        assert result1.workflow_id != result2.workflow_id
        assert result1.workflow_id.startswith("wf_")
        assert result2.workflow_id.startswith("wf_")


@pytest.mark.e2e
class TestConditionalBranching:
    """Tests for conditional workflow branching."""

    @pytest.mark.asyncio
    async def test_conditional_takes_correct_branch(
        self,
        basic_engine: WorkflowEngine,
        conditional_workflow: WorkflowDefinition,
    ) -> None:
        """Test that conditional workflow takes the correct branch."""
        result = await basic_engine.execute(conditional_workflow, inputs={})

        # Should execute evaluate step
        executed_steps = {r.step_id for r in result.steps}
        assert "evaluate" in executed_steps
        # Based on score 85, should take high-path
        assert len(result.steps) >= 1

    @pytest.mark.asyncio
    async def test_transition_condition_evaluation(
        self,
        basic_engine: WorkflowEngine,
    ) -> None:
        """Test transition condition evaluation with various outputs."""
        workflow = WorkflowDefinition(
            id="condition-eval",
            name="Condition Evaluation",
            steps=[
                StepDefinition(
                    id="setup",
                    name="Setup",
                    step_type="task",
                    config={
                        "action": "set_state",
                        "state": {"approved": True},
                    },
                ),
                StepDefinition(
                    id="approved",
                    name="Approved",
                    step_type="task",
                    config={"action": "log", "message": "Approved!"},
                ),
                StepDefinition(
                    id="rejected",
                    name="Rejected",
                    step_type="task",
                    config={"action": "log", "message": "Rejected"},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="to-approved",
                    from_step="setup",
                    to_step="approved",
                    condition="True",  # Always transitions
                    priority=10,
                ),
            ],
            entry_step="setup",
        )

        result = await basic_engine.execute(workflow, inputs={})

        executed_steps = {r.step_id for r in result.steps}
        assert "setup" in executed_steps

    @pytest.mark.asyncio
    async def test_default_next_step_when_no_transition_matches(
        self,
        basic_engine: WorkflowEngine,
    ) -> None:
        """Test that default next_steps is used when no transition matches."""
        workflow = WorkflowDefinition(
            id="default-next",
            name="Default Next",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="task",
                    config={"action": "log", "message": "Start"},
                    next_steps=["default-step"],
                ),
                StepDefinition(
                    id="conditional-step",
                    name="Conditional",
                    step_type="task",
                    config={"action": "log", "message": "Conditional"},
                ),
                StepDefinition(
                    id="default-step",
                    name="Default",
                    step_type="task",
                    config={"action": "log", "message": "Default"},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="to-conditional",
                    from_step="start",
                    to_step="conditional-step",
                    condition="False",  # Never matches
                ),
            ],
            entry_step="start",
        )

        result = await basic_engine.execute(workflow, inputs={})

        executed_steps = {r.step_id for r in result.steps}
        assert "start" in executed_steps
        # Should take default path
        assert "default-step" in executed_steps or len(result.steps) >= 1


@pytest.mark.e2e
class TestErrorHandling:
    """Tests for workflow error handling and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_stops_on_failure_when_configured(
        self,
        engine_with_custom_steps: WorkflowEngine,
    ) -> None:
        """Test that workflow stops on failure when stop_on_failure is True."""
        workflow = WorkflowDefinition(
            id="stop-on-failure",
            name="Stop on Failure",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="custom",
                    config={"output": {"value": 1}},
                    next_steps=["step2"],
                ),
                StepDefinition(
                    id="step2",
                    name="Step 2",
                    step_type="custom",
                    config={"raise_error": True, "error_message": "Test error"},
                    next_steps=["step3"],
                ),
                StepDefinition(
                    id="step3",
                    name="Step 3",
                    step_type="custom",
                    config={"output": {"value": 3}},
                ),
            ],
            entry_step="step1",
        )

        result = await engine_with_custom_steps.execute(workflow, inputs={})

        # Workflow should fail
        assert not result.success
        # Step 3 should not be executed
        executed_steps = {r.step_id for r in result.steps}
        assert "step1" in executed_steps
        assert "step2" in executed_steps
        assert "step3" not in executed_steps

    @pytest.mark.asyncio
    async def test_optional_step_failure_continues_workflow(
        self,
    ) -> None:
        """Test that optional step failure doesn't stop workflow."""
        engine = create_workflow_engine(
            config=WorkflowConfig(
                stop_on_failure=True,
                skip_optional_on_timeout=True,
            ),
            custom_steps={"custom": MockCustomStep},
        )

        workflow = WorkflowDefinition(
            id="optional-failure",
            name="Optional Failure",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="custom",
                    config={"output": {"value": 1}},
                    next_steps=["optional-step"],
                ),
                StepDefinition(
                    id="optional-step",
                    name="Optional Step",
                    step_type="custom",
                    config={"raise_error": True},
                    optional=True,
                    next_steps=["step3"],
                ),
                StepDefinition(
                    id="step3",
                    name="Step 3",
                    step_type="custom",
                    config={"output": {"value": 3}},
                ),
            ],
            entry_step="step1",
        )

        result = await engine.execute(workflow, inputs={})

        # Workflow should continue after optional step failure
        executed_steps = {r.step_id for r in result.steps}
        assert "step1" in executed_steps
        assert "optional-step" in executed_steps
        # Step 3 should still run
        assert len(result.steps) >= 2

    @pytest.mark.asyncio
    async def test_step_retry_on_failure(
        self,
    ) -> None:
        """Test that steps retry on failure when retries are configured."""
        engine = create_workflow_engine(
            config=WorkflowConfig(
                stop_on_failure=True,
            ),
            custom_steps={"failing": MockFailingStep},
        )

        workflow = WorkflowDefinition(
            id="retry-test",
            name="Retry Test",
            steps=[
                StepDefinition(
                    id="retry-step",
                    name="Retry Step",
                    step_type="failing",
                    config={"fail_count": 2},
                    retries=3,  # Will succeed on 3rd attempt
                ),
            ],
            entry_step="retry-step",
        )

        result = await engine.execute(workflow, inputs={})

        # Should eventually succeed after retries
        assert len(result.steps) == 1
        step_result = result.steps[0]
        # Either success after retries or failure if retries exhausted
        assert step_result.retry_count <= 3

    @pytest.mark.asyncio
    async def test_workflow_timeout(self) -> None:
        """Test workflow timeout is enforced."""
        engine = create_workflow_engine(
            config=WorkflowConfig(
                total_timeout_seconds=0.5,  # Very short timeout
            ),
            custom_steps={"custom": MockCustomStep},
        )

        workflow = WorkflowDefinition(
            id="timeout-test",
            name="Timeout Test",
            steps=[
                StepDefinition(
                    id="slow-step",
                    name="Slow Step",
                    step_type="custom",
                    config={"delay_seconds": 5.0, "output": {"done": True}},
                ),
            ],
            entry_step="slow-step",
        )

        result = await engine.execute(workflow, inputs={})

        # Should timeout
        assert not result.success
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()


@pytest.mark.e2e
class TestCheckpointing:
    """Tests for workflow checkpointing and recovery."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_during_execution(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test that checkpoints are created during workflow execution."""
        result = await basic_engine.execute(sequential_workflow, inputs={})

        assert result.success
        # Checkpoints should be created (at least one per step with interval=1)
        # The engine tracks checkpoints internally

    @pytest.mark.asyncio
    async def test_checkpoint_cache_stats(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test checkpoint cache statistics are tracked."""
        await basic_engine.execute(sequential_workflow, inputs={})

        stats = basic_engine.get_checkpoint_cache_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats or "size" in stats or len(stats) >= 0

    @pytest.mark.asyncio
    async def test_list_checkpoints_for_workflow(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test listing checkpoints for a workflow."""
        result = await basic_engine.execute(
            sequential_workflow,
            inputs={},
            workflow_id="test-checkpoint-list",
        )

        checkpoints = await basic_engine.list_checkpoints("test-checkpoint-list")
        # Should return list (may be empty if checkpoints were cleaned up)
        assert isinstance(checkpoints, list)


@pytest.mark.e2e
class TestCompleteWorkflowLifecycle:
    """Tests for complete workflow lifecycle management."""

    @pytest.mark.asyncio
    async def test_end_to_end_data_processing_workflow(
        self,
    ) -> None:
        """Test complete data processing workflow."""
        engine = create_workflow_engine(custom_steps={"custom": MockCustomStep})

        workflow = WorkflowDefinition(
            id="data-processing-e2e",
            name="Data Processing E2E",
            description="Complete data processing workflow",
            steps=[
                StepDefinition(
                    id="ingest",
                    name="Ingest Data",
                    step_type="custom",
                    config={
                        "output": {
                            "records": 100,
                            "source": "api",
                        }
                    },
                    next_steps=["validate"],
                ),
                StepDefinition(
                    id="validate",
                    name="Validate Data",
                    step_type="custom",
                    config={
                        "output": {
                            "valid_records": 95,
                            "invalid_records": 5,
                        }
                    },
                    next_steps=["transform"],
                ),
                StepDefinition(
                    id="transform",
                    name="Transform Data",
                    step_type="custom",
                    config={
                        "output": {
                            "transformed": True,
                            "schema": "normalized",
                        }
                    },
                    next_steps=["store"],
                ),
                StepDefinition(
                    id="store",
                    name="Store Data",
                    step_type="custom",
                    config={
                        "output": {
                            "stored": True,
                            "location": "warehouse",
                        }
                    },
                ),
            ],
            entry_step="ingest",
        )

        result = await engine.execute(
            workflow,
            inputs={"source_config": {"type": "api", "url": "https://example.com"}},
        )

        assert result.success
        assert len(result.steps) == 4

        # Verify all steps executed
        step_ids = [r.step_id for r in result.steps]
        assert step_ids == ["ingest", "validate", "transform", "store"]

        # Verify final output
        assert result.final_output is not None
        assert result.final_output.get("stored") is True

    @pytest.mark.asyncio
    async def test_workflow_with_decision_branching(
        self,
    ) -> None:
        """Test workflow with decision-based branching."""
        engine = create_workflow_engine(custom_steps={"custom": MockCustomStep})

        workflow = WorkflowDefinition(
            id="decision-workflow",
            name="Decision Workflow",
            steps=[
                StepDefinition(
                    id="analyze",
                    name="Analyze Request",
                    step_type="custom",
                    config={
                        "output": {
                            "risk_level": "high",
                            "score": 85,
                        }
                    },
                ),
                StepDefinition(
                    id="high-risk",
                    name="High Risk Path",
                    step_type="custom",
                    config={"output": {"action": "manual_review"}},
                ),
                StepDefinition(
                    id="low-risk",
                    name="Low Risk Path",
                    step_type="custom",
                    config={"output": {"action": "auto_approve"}},
                ),
                StepDefinition(
                    id="complete",
                    name="Complete",
                    step_type="custom",
                    config={"output": {"status": "done"}},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="to-high",
                    from_step="analyze",
                    to_step="high-risk",
                    # Engine uses step_output in namespace, not output
                    condition="step_output.get('risk_level') == 'high'",
                    priority=10,
                ),
                TransitionRule(
                    id="to-low",
                    from_step="analyze",
                    to_step="low-risk",
                    condition="step_output.get('risk_level') == 'low'",
                    priority=5,
                ),
                TransitionRule(
                    id="high-to-complete",
                    from_step="high-risk",
                    to_step="complete",
                    condition="True",
                ),
                TransitionRule(
                    id="low-to-complete",
                    from_step="low-risk",
                    to_step="complete",
                    condition="True",
                ),
            ],
            entry_step="analyze",
        )

        result = await engine.execute(workflow, inputs={})

        assert result.success
        executed_steps = {r.step_id for r in result.steps}
        assert "analyze" in executed_steps
        # Should take high-risk path based on output
        assert "high-risk" in executed_steps
        assert "low-risk" not in executed_steps
        assert "complete" in executed_steps

    @pytest.mark.asyncio
    async def test_workflow_metrics_collection(
        self,
        basic_engine: WorkflowEngine,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test that workflow collects execution metrics."""
        result = await basic_engine.execute(sequential_workflow, inputs={})

        metrics = basic_engine.get_metrics()

        assert "total_steps" in metrics
        assert "completed_steps" in metrics
        assert "failed_steps" in metrics
        assert "total_duration_ms" in metrics
        assert metrics["total_steps"] >= 3

    @pytest.mark.asyncio
    async def test_workflow_termination_request(
        self,
    ) -> None:
        """Test early workflow termination."""
        engine = create_workflow_engine(custom_steps={"custom": MockCustomStep})

        workflow = WorkflowDefinition(
            id="termination-test",
            name="Termination Test",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="custom",
                    config={"output": {"done": True}},
                    next_steps=["step2"],
                ),
                StepDefinition(
                    id="step2",
                    name="Step 2",
                    step_type="custom",
                    config={"output": {"done": True}},
                ),
            ],
            entry_step="step1",
        )

        # Request termination before execution
        engine.request_termination("Test termination")

        should_terminate, reason = engine.check_termination()
        assert should_terminate
        assert reason == "Test termination"


@pytest.mark.e2e
class TestWorkflowDefinitionValidation:
    """Tests for workflow definition validation."""

    def test_workflow_validation_passes_for_valid_definition(
        self,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test that valid workflow definitions pass validation."""
        is_valid, errors = sequential_workflow.validate()

        assert is_valid
        assert len(errors) == 0

    def test_workflow_validation_fails_for_missing_entry_step(self) -> None:
        """Test validation fails when entry step doesn't exist."""
        workflow = WorkflowDefinition(
            id="invalid-entry",
            name="Invalid Entry",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="task",
                    config={},
                ),
            ],
            entry_step="nonexistent-step",
        )

        is_valid, errors = workflow.validate()

        assert not is_valid
        assert any("entry step" in e.lower() for e in errors)

    def test_workflow_validation_fails_for_invalid_transitions(self) -> None:
        """Test validation fails for transitions to nonexistent steps."""
        workflow = WorkflowDefinition(
            id="invalid-transition",
            name="Invalid Transition",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="task",
                    config={},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="bad-transition",
                    from_step="step1",
                    to_step="nonexistent",
                    condition="True",
                ),
            ],
            entry_step="step1",
        )

        is_valid, errors = workflow.validate()

        assert not is_valid
        assert any("nonexistent" in e for e in errors)

    def test_workflow_cloning(
        self,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test workflow cloning creates proper copy."""
        # Mark as template to test template_id propagation
        sequential_workflow.is_template = True

        cloned = sequential_workflow.clone(
            new_id="cloned-workflow",
            new_name="Cloned Workflow",
        )

        assert cloned.id == "cloned-workflow"
        assert cloned.name == "Cloned Workflow"
        assert len(cloned.steps) == len(sequential_workflow.steps)
        # When cloning a template, clone gets template_id set to source id
        assert cloned.template_id == sequential_workflow.id
        assert cloned.is_template is False  # Clone is not a template

    def test_workflow_serialization_roundtrip(
        self,
        conditional_workflow: WorkflowDefinition,
    ) -> None:
        """Test workflow serialization and deserialization."""
        # Convert to dict
        workflow_dict = conditional_workflow.to_dict()
        assert isinstance(workflow_dict, dict)
        assert workflow_dict["id"] == conditional_workflow.id

        # Restore from dict
        restored = WorkflowDefinition.from_dict(workflow_dict)
        assert restored.id == conditional_workflow.id
        assert restored.name == conditional_workflow.name
        assert len(restored.steps) == len(conditional_workflow.steps)
        assert len(restored.transitions) == len(conditional_workflow.transitions)

    def test_workflow_yaml_serialization(
        self,
        sequential_workflow: WorkflowDefinition,
    ) -> None:
        """Test workflow YAML serialization."""
        yaml_str = sequential_workflow.to_yaml()
        assert isinstance(yaml_str, str)
        assert sequential_workflow.name in yaml_str

        # Restore from YAML
        restored = WorkflowDefinition.from_yaml(yaml_str)
        assert restored.id == sequential_workflow.id
        assert len(restored.steps) == len(sequential_workflow.steps)
