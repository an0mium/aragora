"""Tests for Workflow Engine."""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict

from aragora.workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowConfig,
    WorkflowStep,
    WorkflowContext,
    WorkflowResult,
    StepDefinition,
    StepResult,
    StepStatus,
    TransitionRule,
    ExecutionPattern,
    WorkflowCheckpoint,
    BaseStep,
    AgentStep,
    ParallelStep,
    LoopStep,
)


class TestExecutionPattern:
    """Test ExecutionPattern enum."""

    def test_pattern_values(self):
        """Test all execution patterns are defined."""
        assert ExecutionPattern.SEQUENTIAL.value == "sequential"
        assert ExecutionPattern.PARALLEL.value == "parallel"
        assert ExecutionPattern.CONDITIONAL.value == "conditional"
        assert ExecutionPattern.LOOP.value == "loop"


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


class TestTransitionRule:
    """Test TransitionRule dataclass."""

    def test_simple_transition(self):
        """Test creating a simple transition."""
        rule = TransitionRule(
            id="tr_1",
            from_step="step_1",
            to_step="step_2",
            condition="True",  # Always true = unconditional
        )

        assert rule.id == "tr_1"
        assert rule.from_step == "step_1"
        assert rule.to_step == "step_2"

    def test_conditional_transition(self):
        """Test creating a conditional transition."""
        rule = TransitionRule(
            id="tr_cond",
            from_step="validation",
            to_step="success_handler",
            condition="${validation.result} == 'passed'",
        )

        assert rule.from_step == "validation"
        assert rule.to_step == "success_handler"
        assert rule.condition == "${validation.result} == 'passed'"


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

    def test_definition_with_metadata(self):
        """Test creating a definition with metadata."""
        definition = WorkflowDefinition(
            id="wf_meta",
            name="Workflow with Metadata",
            description="A workflow for testing",
            version="1.0.0",
            steps=[StepDefinition(id="s1", name="S1", step_type="agent")],
            metadata={"author": "test", "category": "testing"},
        )

        assert definition.description == "A workflow for testing"
        assert definition.version == "1.0.0"
        assert definition.metadata["author"] == "test"


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

    def test_set_step_output(self):
        """Test setting step outputs."""
        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        context.step_outputs["step_1"] = {"result": "success"}

        output = context.get_step_output("step_1")
        assert output["result"] == "success"

    def test_get_nonexistent_output(self):
        """Test getting output for non-existent step."""
        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        output = context.get_step_output("nonexistent")
        assert output is None


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
            created_at=datetime.now(),
        )

        assert checkpoint.id == "cp_1"
        assert checkpoint.workflow_id == "wf_1"
        assert checkpoint.definition_id == "def_1"
        assert checkpoint.current_step == "s3"
        assert len(checkpoint.completed_steps) == 2


class TestStepResult:
    """Test StepResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful step result."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.COMPLETED,
            output={"value": 42},
        )

        assert result.step_id == "s1"
        assert result.status == StepStatus.COMPLETED
        assert result.success is True
        assert result.output["value"] == 42

    def test_failed_result(self):
        """Test creating a failed step result."""
        result = StepResult(
            step_id="s1",
            step_name="Step 1",
            status=StepStatus.FAILED,
            error="Step execution failed",
        )

        assert result.status == StepStatus.FAILED
        assert result.success is False
        assert result.error == "Step execution failed"


class SimpleTestStep(BaseStep):
    """Simple step for testing."""

    def __init__(
        self, name: str = "test", config: Dict[str, Any] = None, output_value: Any = "done"
    ):
        super().__init__(name, config)
        self._output_value = output_value

    async def execute(self, context: WorkflowContext) -> Any:
        return {"result": self._output_value}


class FailingStep(BaseStep):
    """Step that always fails for testing."""

    def __init__(self, name: str = "failing", config: Dict[str, Any] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        raise RuntimeError("Step failed intentionally")


class TestWorkflowConfig:
    """Test WorkflowConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WorkflowConfig()

        assert config.total_timeout_seconds == 3600.0
        assert config.step_timeout_seconds == 120.0
        assert config.stop_on_failure is True
        assert config.enable_checkpointing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            total_timeout_seconds=600.0,
            step_timeout_seconds=60.0,
            stop_on_failure=False,
        )

        assert config.total_timeout_seconds == 600.0
        assert config.step_timeout_seconds == 60.0
        assert config.stop_on_failure is False


class TestWorkflowEngine:
    """Test WorkflowEngine class."""

    @pytest.fixture
    def engine(self):
        """Create workflow engine for testing."""
        config = WorkflowConfig()
        return WorkflowEngine(config)

    @pytest.fixture
    def simple_definition(self):
        """Create a simple workflow definition."""
        return WorkflowDefinition(
            id="wf_simple",
            name="Simple Workflow",
            steps=[
                StepDefinition(id="step_1", name="First Step", step_type="test"),
            ],
        )

    def test_register_step_type(self, engine):
        """Test registering a step type."""
        engine.register_step_type("test_step", SimpleTestStep)

        assert "test_step" in engine._step_types

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, engine):
        """Test executing a simple workflow."""
        # Register custom step type
        engine.register_step_type("test", SimpleTestStep)

        definition = WorkflowDefinition(
            id="wf_test",
            name="Test",
            steps=[
                StepDefinition(id="s1", name="Step 1", step_type="test"),
            ],
        )

        result = await engine.execute(definition, inputs={"test": "value"})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_sequential_steps(self, engine):
        """Test executing sequential steps."""
        outputs = []

        class RecordingStep(BaseStep):
            def __init__(self, name: str = "record", config: Dict[str, Any] = None, order: int = 1):
                super().__init__(name, config)
                self._order = order

            async def execute(self, context: WorkflowContext) -> Any:
                outputs.append(self._order)
                return {"order": self._order}

        # Register steps - engine will pass name/config, we add order via subclass
        class Record1Step(RecordingStep):
            def __init__(self, name: str = "record_1", config: Dict[str, Any] = None):
                super().__init__(name, config, order=1)

        class Record2Step(RecordingStep):
            def __init__(self, name: str = "record_2", config: Dict[str, Any] = None):
                super().__init__(name, config, order=2)

        class Record3Step(RecordingStep):
            def __init__(self, name: str = "record_3", config: Dict[str, Any] = None):
                super().__init__(name, config, order=3)

        engine.register_step_type("record_1", Record1Step)
        engine.register_step_type("record_2", Record2Step)
        engine.register_step_type("record_3", Record3Step)

        definition = WorkflowDefinition(
            id="wf_seq",
            name="Sequential",
            steps=[
                StepDefinition(id="s1", name="First", step_type="record_1", next_steps=["s2"]),
                StepDefinition(id="s2", name="Second", step_type="record_2", next_steps=["s3"]),
                StepDefinition(id="s3", name="Third", step_type="record_3"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True
        assert outputs == [1, 2, 3]  # Steps executed in order

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, engine):
        """Test workflow with failing step."""
        engine.register_step_type("failing", FailingStep)

        definition = WorkflowDefinition(
            id="wf_fail",
            name="Failing Workflow",
            steps=[
                StepDefinition(id="s1", name="Fail Step", step_type="failing"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is False
        # Error is on the step result, not on the workflow result
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[0].error is not None

    @pytest.mark.asyncio
    async def test_execute_with_inputs(self, engine):
        """Test workflow with inputs."""
        received_inputs = {}

        class InputStep(BaseStep):
            def __init__(self, name: str = "input_step", config: Dict[str, Any] = None):
                super().__init__(name, config)

            async def execute(self, context: WorkflowContext) -> Any:
                nonlocal received_inputs
                received_inputs = context.inputs.copy()
                return {"processed": context.inputs.get("query")}

        engine.register_step_type("input", InputStep)

        definition = WorkflowDefinition(
            id="wf_input",
            name="Input Workflow",
            steps=[
                StepDefinition(id="s1", name="Input Step", step_type="input"),
            ],
        )

        result = await engine.execute(
            definition,
            inputs={"query": "test query", "limit": 10},
        )

        assert result.success is True
        assert received_inputs["query"] == "test query"
        assert received_inputs["limit"] == 10

    @pytest.mark.asyncio
    async def test_step_output_propagation(self, engine):
        """Test that step outputs are available to subsequent steps."""

        class ProducerStep(BaseStep):
            def __init__(self, name: str = "producer", config: Dict[str, Any] = None):
                super().__init__(name, config)

            async def execute(self, context: WorkflowContext) -> Any:
                return {"value": 42}

        class ConsumerStep(BaseStep):
            def __init__(self, name: str = "consumer", config: Dict[str, Any] = None):
                super().__init__(name, config)

            async def execute(self, context: WorkflowContext) -> Any:
                prev_output = context.get_step_output("s1")
                return {"received": prev_output.get("value") if prev_output else None}

        engine.register_step_type("producer", ProducerStep)
        engine.register_step_type("consumer", ConsumerStep)

        definition = WorkflowDefinition(
            id="wf_propagate",
            name="Propagation",
            steps=[
                StepDefinition(id="s1", name="Producer", step_type="producer", next_steps=["s2"]),
                StepDefinition(id="s2", name="Consumer", step_type="consumer"),
            ],
        )

        result = await engine.execute(definition)

        assert result.success is True


class TestParallelStep:
    """Test ParallelStep class."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel step execution."""
        execution_times = []

        class TimedStep(BaseStep):
            def __init__(self, name: str, delay: float):
                super().__init__(name)
                self._delay = delay

            async def execute(self, context: WorkflowContext) -> Any:
                await asyncio.sleep(self._delay)
                execution_times.append(self.name)
                return {"name": self.name}

        parallel = ParallelStep(
            name="parallel_test",
            sub_steps=[
                TimedStep("a", 0.05),
                TimedStep("b", 0.05),
                TimedStep("c", 0.05),
            ],
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await parallel.execute(context)

        # All steps should have completed
        assert len(result) == 3
        assert len(execution_times) == 3


class TestLoopStep:
    """Test LoopStep class."""

    @pytest.mark.asyncio
    async def test_loop_execution(self):
        """Test loop step execution."""
        iteration_count = 0

        class CountingStep(BaseStep):
            def __init__(self):
                super().__init__("counter")

            async def execute(self, context: WorkflowContext) -> Any:
                nonlocal iteration_count
                iteration_count += 1
                context.set_state("iteration", iteration_count)
                return {"iteration": iteration_count}

        # LoopStep uses a string condition that is evaluated
        # condition = "True" means stop when True (exit condition)
        # We want to stop after 3 iterations, so use: iteration >= 3
        loop = LoopStep(
            name="loop_test",
            wrapped_step=CountingStep(),
            max_iterations=5,
            condition="state.get('iteration', 0) >= 3",  # Stop when iteration >= 3
        )

        context = WorkflowContext(workflow_id="wf_1", definition_id="def_1")
        result = await loop.execute(context)

        # Should have looped 3 times (stops when iteration >= 3)
        assert iteration_count == 3
        assert len(result) == 3  # Results from each iteration
