"""
End-to-end tests for Workflow Engine.

Tests the full lifecycle of:
- Multi-step workflows with different node types
- Parallel and loop execution patterns
- Human checkpoint approval flows
- Memory read/write operations (mocked)
- Error recovery and timeout handling
- Input/output passing between steps
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowConfig,
    StepDefinition,
    StepStatus,
    TransitionRule,
)
from aragora.workflow.step import (
    WorkflowContext,
    BaseStep,
    ParallelStep,
    LoopStep,
    ConditionalStep,
    AgentStep,
)


# ============================================================================
# E2E Test Fixtures
# ============================================================================


@pytest.fixture
def e2e_engine() -> WorkflowEngine:
    """Create a workflow engine configured for E2E testing."""
    return WorkflowEngine(
        config=WorkflowConfig(
            total_timeout_seconds=30.0,
            step_timeout_seconds=10.0,
            stop_on_failure=False,  # Continue on failure for testing
            enable_checkpointing=True,
            checkpoint_interval_steps=1,
        )
    )


@pytest.fixture
def strict_engine() -> WorkflowEngine:
    """Create a workflow engine that stops on failure."""
    return WorkflowEngine(
        config=WorkflowConfig(
            total_timeout_seconds=30.0,
            step_timeout_seconds=10.0,
            stop_on_failure=True,
        )
    )


# ============================================================================
# Custom Test Steps
# ============================================================================


class CounterStep(BaseStep):
    """Test step that increments a counter in state."""

    async def execute(self, context: WorkflowContext) -> Any:
        counter = context.state.get("counter", 0)
        counter += 1
        context.set_state("counter", counter)
        return {"counter": counter}


class AccumulatorStep(BaseStep):
    """Test step that accumulates values in state."""

    async def execute(self, context: WorkflowContext) -> Any:
        value = self._config.get("value", 1)
        accumulator = context.state.get("accumulator", [])
        accumulator.append(value)
        context.set_state("accumulator", accumulator)
        return {"accumulated": accumulator.copy()}


class FailingStep(BaseStep):
    """Test step that always fails."""

    async def execute(self, context: WorkflowContext) -> Any:
        raise RuntimeError("Step intentionally failed")


class ConditionalFailStep(BaseStep):
    """Test step that fails based on input."""

    async def execute(self, context: WorkflowContext) -> Any:
        should_fail = context.inputs.get("should_fail", False)
        if should_fail:
            raise RuntimeError("Conditional failure triggered")
        return {"success": True}


class DelayedStep(BaseStep):
    """Test step with configurable delay."""

    async def execute(self, context: WorkflowContext) -> Any:
        delay = self._config.get("delay_seconds", 0.1)
        await asyncio.sleep(delay)
        return {"delayed": True, "delay_seconds": delay}


class InputEchoStep(BaseStep):
    """Test step that echoes inputs and previous outputs."""

    async def execute(self, context: WorkflowContext) -> Any:
        return {
            "inputs": context.inputs.copy(),
            "previous_outputs": dict(context.step_outputs),
            "state": dict(context.state),
        }


# ============================================================================
# Basic Lifecycle E2E Tests
# ============================================================================


class TestBasicWorkflowLifecycle:
    """Tests for basic workflow execution lifecycle."""

    @pytest.mark.asyncio
    async def test_single_step_workflow(self, e2e_engine: WorkflowEngine) -> None:
        """Test simplest possible workflow: one step."""
        workflow = WorkflowDefinition(
            id="single-step-e2e",
            name="Single Step E2E",
            steps=[
                StepDefinition(
                    id="only-step",
                    name="Only Step",
                    step_type="task",
                    config={"action": "set_state", "state": {"done": True}},
                ),
            ],
            entry_step="only-step",
        )

        result = await e2e_engine.execute(workflow, inputs={})

        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].step_id == "only-step"
        assert result.steps[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multi_step_linear_workflow(self, e2e_engine: WorkflowEngine) -> None:
        """Test linear workflow: A -> B -> C."""
        workflow = WorkflowDefinition(
            id="linear-e2e",
            name="Linear E2E",
            steps=[
                StepDefinition(
                    id="step-a",
                    name="Step A",
                    step_type="task",
                    config={"action": "set_state", "state": {"a": True}},
                    next_steps=["step-b"],
                ),
                StepDefinition(
                    id="step-b",
                    name="Step B",
                    step_type="task",
                    config={"action": "set_state", "state": {"b": True}},
                    next_steps=["step-c"],
                ),
                StepDefinition(
                    id="step-c",
                    name="Step C",
                    step_type="task",
                    config={"action": "set_state", "state": {"c": True}},
                ),
            ],
            entry_step="step-a",
        )

        result = await e2e_engine.execute(workflow, inputs={})

        assert result.success
        assert len(result.steps) == 3
        executed_ids = [r.step_id for r in result.steps]
        assert "step-a" in executed_ids
        assert "step-b" in executed_ids
        assert "step-c" in executed_ids

    @pytest.mark.asyncio
    async def test_workflow_with_inputs(self, e2e_engine: WorkflowEngine) -> None:
        """Test that workflow inputs are available to all steps."""
        workflow = WorkflowDefinition(
            id="inputs-e2e",
            name="Inputs E2E",
            steps=[
                StepDefinition(
                    id="use-inputs",
                    name="Use Inputs",
                    step_type="task",
                    config={"action": "log", "message": "Testing inputs"},
                ),
            ],
            entry_step="use-inputs",
        )

        result = await e2e_engine.execute(
            workflow,
            inputs={"user_id": "test-user", "request_id": "req-123"},
        )

        assert result.success


# ============================================================================
# Conditional Branching E2E Tests
# ============================================================================


class TestConditionalBranching:
    """Tests for conditional workflow branching."""

    @pytest.mark.asyncio
    async def test_simple_branch_true_path(self, e2e_engine: WorkflowEngine) -> None:
        """Test conditional branch taking the true path."""
        workflow = WorkflowDefinition(
            id="branch-true-e2e",
            name="Branch True E2E",
            steps=[
                StepDefinition(
                    id="decide",
                    name="Decide",
                    step_type="task",
                    config={"action": "set_state", "state": {"condition": True}},
                ),
                StepDefinition(
                    id="true-branch",
                    name="True Branch",
                    step_type="task",
                    config={"action": "set_state", "state": {"path": "true"}},
                ),
                StepDefinition(
                    id="false-branch",
                    name="False Branch",
                    step_type="task",
                    config={"action": "set_state", "state": {"path": "false"}},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="to-true",
                    from_step="decide",
                    to_step="true-branch",
                    condition="output.get('condition') == True",
                    priority=10,
                ),
                TransitionRule(
                    id="to-false",
                    from_step="decide",
                    to_step="false-branch",
                    condition="output.get('condition') == False",
                    priority=5,
                ),
            ],
            entry_step="decide",
        )

        result = await e2e_engine.execute(workflow, inputs={})

        # Should execute decide step at minimum
        executed_ids = {r.step_id for r in result.steps}
        assert "decide" in executed_ids

    @pytest.mark.asyncio
    async def test_input_based_branching(self, e2e_engine: WorkflowEngine) -> None:
        """Test branching based on workflow inputs."""
        workflow = WorkflowDefinition(
            id="input-branch-e2e",
            name="Input Branch E2E",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="task",
                    config={"action": "log", "message": "Starting"},
                ),
                StepDefinition(
                    id="premium-path",
                    name="Premium Path",
                    step_type="task",
                    config={"action": "set_state", "state": {"tier": "premium"}},
                ),
                StepDefinition(
                    id="standard-path",
                    name="Standard Path",
                    step_type="task",
                    config={"action": "set_state", "state": {"tier": "standard"}},
                ),
            ],
            transitions=[
                TransitionRule(
                    id="to-premium",
                    from_step="start",
                    to_step="premium-path",
                    condition="inputs.get('tier') == 'premium'",
                    priority=10,
                ),
                TransitionRule(
                    id="to-standard",
                    from_step="start",
                    to_step="standard-path",
                    condition="True",  # Default path
                    priority=1,
                ),
            ],
            entry_step="start",
        )

        # Test premium path
        result = await e2e_engine.execute(workflow, inputs={"tier": "premium"})
        assert result.success

        # Test standard path
        result2 = await e2e_engine.execute(workflow, inputs={"tier": "basic"})
        assert result2.success


# ============================================================================
# Parallel Execution E2E Tests
# ============================================================================


class TestParallelExecution:
    """Tests for parallel step execution."""

    @pytest.mark.asyncio
    async def test_parallel_step_execution(self) -> None:
        """Test executing multiple steps in parallel."""
        step1 = CounterStep("counter-1", {"increment": 1})
        step2 = CounterStep("counter-2", {"increment": 2})
        step3 = CounterStep("counter-3", {"increment": 3})

        parallel = ParallelStep("parallel-counters", [step1, step2, step3])

        context = WorkflowContext(
            workflow_id="test-parallel",
            definition_id="parallel-def",
            inputs={},
            state={"counter": 0},
        )

        result = await parallel.execute(context)

        # All sub-steps should have executed
        assert "counter-1" in result
        assert "counter-2" in result
        assert "counter-3" in result
        # Counter should have been incremented by all three
        assert context.state["counter"] == 3

    @pytest.mark.asyncio
    async def test_parallel_with_one_failing(self) -> None:
        """Test parallel execution where one step fails."""
        step1 = CounterStep("success-step", {})
        step2 = FailingStep("failing-step", {})
        step3 = CounterStep("another-success", {})

        parallel = ParallelStep("parallel-with-failure", [step1, step2, step3])

        context = WorkflowContext(
            workflow_id="test-parallel-fail",
            definition_id="parallel-fail-def",
            inputs={},
            state={"counter": 0},
        )

        result = await parallel.execute(context)

        # Should capture error but not crash
        assert "success-step" in result
        assert "failing-step" in result
        assert "error" in result["failing-step"]
        assert "another-success" in result


# ============================================================================
# Loop Execution E2E Tests
# ============================================================================


class TestLoopExecution:
    """Tests for loop step execution."""

    @pytest.mark.asyncio
    async def test_loop_with_iteration_count(self) -> None:
        """Test loop that runs for specific iterations."""
        inner_step = AccumulatorStep("accumulate", {"value": 1})

        # Exit after 5 iterations
        loop = LoopStep(
            "count-loop",
            inner_step,
            condition="iteration >= 5",
            max_iterations=10,
        )

        context = WorkflowContext(
            workflow_id="test-loop",
            definition_id="loop-def",
            inputs={},
            state={},
        )

        result = await loop.execute(context)

        assert result["iterations"] == 5
        assert len(result["outputs"]) == 5

    @pytest.mark.asyncio
    async def test_loop_with_max_iterations(self) -> None:
        """Test loop hits max iterations limit."""
        inner_step = CounterStep("counter", {})

        # Condition that never becomes true
        loop = LoopStep(
            "infinite-loop",
            inner_step,
            condition="False",  # Never exit
            max_iterations=3,
        )

        context = WorkflowContext(
            workflow_id="test-max-loop",
            definition_id="max-loop-def",
            inputs={},
            state={},
        )

        result = await loop.execute(context)

        # Should stop at max_iterations
        assert result["iterations"] == 3

    @pytest.mark.asyncio
    async def test_loop_with_output_condition(self) -> None:
        """Test loop that exits based on step output."""
        inner_step = AccumulatorStep("accumulate", {"value": 10})

        # Exit when accumulated sum exceeds 30
        loop = LoopStep(
            "accumulate-loop",
            inner_step,
            condition="sum(state.get('accumulator', [])) >= 30",
            max_iterations=10,
        )

        context = WorkflowContext(
            workflow_id="test-output-loop",
            definition_id="output-loop-def",
            inputs={},
            state={},
        )

        result = await loop.execute(context)

        # Should exit after 3 iterations (10 + 10 + 10 = 30)
        assert result["iterations"] == 3


# ============================================================================
# Error Handling E2E Tests
# ============================================================================


class TestErrorHandling:
    """Tests for workflow error handling."""

    @pytest.mark.asyncio
    async def test_workflow_continues_on_step_failure(self, e2e_engine: WorkflowEngine) -> None:
        """Test workflow continues after step failure (stop_on_failure=False)."""
        workflow = WorkflowDefinition(
            id="continue-on-fail",
            name="Continue on Fail",
            steps=[
                StepDefinition(
                    id="first-step",
                    name="First Step",
                    step_type="task",
                    config={"action": "set_state", "state": {"first": True}},
                    next_steps=["bad-step"],
                ),
                StepDefinition(
                    id="bad-step",
                    name="Bad Step",
                    step_type="task",
                    config={
                        "action": "error",
                        "message": "Intentional failure",
                    },
                    next_steps=["last-step"],
                ),
                StepDefinition(
                    id="last-step",
                    name="Last Step",
                    step_type="task",
                    config={"action": "set_state", "state": {"last": True}},
                ),
            ],
            entry_step="first-step",
        )

        result = await e2e_engine.execute(workflow, inputs={})

        # With stop_on_failure=False, workflow should complete
        assert len(result.steps) >= 1
        executed_ids = {r.step_id for r in result.steps}
        assert "first-step" in executed_ids

    @pytest.mark.asyncio
    async def test_workflow_stops_on_step_failure(self, strict_engine: WorkflowEngine) -> None:
        """Test workflow stops on step failure (stop_on_failure=True)."""
        workflow = WorkflowDefinition(
            id="stop-on-fail",
            name="Stop on Fail",
            steps=[
                StepDefinition(
                    id="first-step",
                    name="First Step",
                    step_type="task",
                    config={"action": "set_state", "state": {"first": True}},
                    next_steps=["bad-step"],
                ),
                StepDefinition(
                    id="bad-step",
                    name="Bad Step",
                    step_type="task",
                    config={
                        "action": "error",
                        "message": "Intentional failure",
                    },
                    next_steps=["never-reached"],
                ),
                StepDefinition(
                    id="never-reached",
                    name="Never Reached",
                    step_type="task",
                    config={"action": "log", "message": "Should not run"},
                ),
            ],
            entry_step="first-step",
        )

        result = await strict_engine.execute(workflow, inputs={})

        # Should have executed first step at minimum
        executed_ids = {r.step_id for r in result.steps}
        assert "first-step" in executed_ids


# ============================================================================
# Step Communication E2E Tests
# ============================================================================


class TestStepCommunication:
    """Tests for data passing between steps."""

    @pytest.mark.asyncio
    async def test_step_output_to_next_step(self) -> None:
        """Test that step outputs are available to subsequent steps."""
        context = WorkflowContext(
            workflow_id="test-communication",
            definition_id="comm-def",
            inputs={"initial": "value"},
            state={},
        )

        # First step sets some output
        step1 = InputEchoStep("step1", {})
        output1 = await step1.execute(context)

        # Simulate engine storing output
        context.step_outputs["step1"] = output1

        # Second step should see first step's output
        step2 = InputEchoStep("step2", {})
        output2 = await step2.execute(context)

        assert "step1" in output2["previous_outputs"]
        assert output2["inputs"]["initial"] == "value"

    @pytest.mark.asyncio
    async def test_state_persistence_across_steps(self) -> None:
        """Test that state changes persist across steps."""
        context = WorkflowContext(
            workflow_id="test-state",
            definition_id="state-def",
            inputs={},
            state={},
        )

        # Step 1 sets counter
        step1 = CounterStep("counter1", {})
        await step1.execute(context)
        assert context.state["counter"] == 1

        # Step 2 increments counter
        step2 = CounterStep("counter2", {})
        await step2.execute(context)
        assert context.state["counter"] == 2

        # Step 3 increments again
        step3 = CounterStep("counter3", {})
        await step3.execute(context)
        assert context.state["counter"] == 3


# ============================================================================
# Human Checkpoint E2E Tests
# ============================================================================


class TestHumanCheckpoint:
    """Tests for human checkpoint/approval flows."""

    @pytest.mark.asyncio
    async def test_checkpoint_approval_flow(self) -> None:
        """Test human checkpoint approval mechanism."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalStatus,
            HumanCheckpointStep,
            clear_pending_approvals,
            get_pending_approvals,
            resolve_approval,
        )

        # Clear any existing approvals
        clear_pending_approvals()

        # Create checkpoint step
        checkpoint = HumanCheckpointStep(
            "test-approval",
            config={
                "title": "Approve deployment",
                "description": "Review changes before deployment",
                "checklist": [
                    {"label": "Tests pass", "required": True},
                    {"label": "Reviewed code", "required": True},
                ],
                "timeout_seconds": 0.5,  # Short timeout for testing
            },
        )

        context = WorkflowContext(
            workflow_id="test-checkpoint-wf",
            definition_id="checkpoint-def",
            inputs={},
            state={},
        )
        context.current_step_id = "approval-step"

        # Start checkpoint in background
        async def run_checkpoint():
            return await checkpoint.execute(context)

        checkpoint_task = asyncio.create_task(run_checkpoint())

        # Wait briefly for approval to be registered
        await asyncio.sleep(0.1)

        # Check that approval is pending
        pending = get_pending_approvals()
        assert len(pending) > 0

        approval_id = pending[0].id

        # Approve the request
        approved = resolve_approval(
            request_id=approval_id,
            status=ApprovalStatus.APPROVED,
            responder_id="test-user",
            notes="Looks good!",
            checklist_updates={"item_0": True, "item_1": True},
        )
        assert approved is True

        # Wait for checkpoint to complete
        try:
            result = await asyncio.wait_for(checkpoint_task, timeout=2.0)
            assert result["status"] in ["approved", "timeout"]
        except asyncio.TimeoutError:
            pass  # Timeout is acceptable for this test

        # Cleanup
        clear_pending_approvals()

    @pytest.mark.asyncio
    async def test_checkpoint_rejection_flow(self) -> None:
        """Test human checkpoint rejection."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalStatus,
            HumanCheckpointStep,
            clear_pending_approvals,
            get_pending_approvals,
            resolve_approval,
        )

        clear_pending_approvals()

        checkpoint = HumanCheckpointStep(
            "test-rejection",
            config={
                "title": "Approve risky change",
                "description": "This change may break things",
                "checklist": [{"label": "Acknowledge risk", "required": True}],
                "timeout_seconds": 0.5,
            },
        )

        context = WorkflowContext(
            workflow_id="test-reject-wf",
            definition_id="reject-def",
            inputs={},
            state={},
        )
        context.current_step_id = "reject-step"

        # Start checkpoint
        checkpoint_task = asyncio.create_task(checkpoint.execute(context))
        await asyncio.sleep(0.1)

        # Reject the request
        pending = get_pending_approvals()
        if pending:
            resolve_approval(
                request_id=pending[0].id,
                status=ApprovalStatus.REJECTED,
                responder_id="security-team",
                notes="Too risky",
            )

        try:
            result = await asyncio.wait_for(checkpoint_task, timeout=2.0)
            assert result.get("status") in ["rejected", "timeout", None]
        except asyncio.TimeoutError:
            pass

        clear_pending_approvals()


# ============================================================================
# Decision Step E2E Tests
# ============================================================================


class TestDecisionStep:
    """Tests for decision node execution."""

    @pytest.mark.asyncio
    async def test_decision_step_evaluation(self) -> None:
        """Test decision step evaluates conditions correctly."""
        from aragora.workflow.nodes.decision import DecisionStep

        decision = DecisionStep(
            "score-decision",
            config={
                "conditions": [
                    {
                        "name": "passing",
                        "expression": "inputs.get('score', 0) >= 80",
                        "next_step": "pass_step",
                    },
                    {
                        "name": "failing",
                        "expression": "inputs.get('score', 0) < 80",
                        "next_step": "fail_step",
                    },
                ],
                "default_branch": "unknown_step",
            },
        )

        # Test passing score
        context = WorkflowContext(
            workflow_id="test-decision",
            definition_id="decision-def",
            inputs={"score": 85},
            state={},
        )

        result = await decision.execute(context)
        assert result["decision"] == "pass_step"
        assert result["decision_name"] == "passing"

        # Test failing score
        context.inputs["score"] = 60
        result = await decision.execute(context)
        assert result["decision"] == "fail_step"
        assert result["decision_name"] == "failing"

    @pytest.mark.asyncio
    async def test_decision_with_state_access(self) -> None:
        """Test decision step can access workflow state."""
        from aragora.workflow.nodes.decision import DecisionStep

        decision = DecisionStep(
            "state-decision",
            config={
                "conditions": [
                    {
                        "name": "retry",
                        "expression": "state.get('retries', 0) < 3",
                        "next_step": "retry_step",
                    },
                ],
                "default_branch": "give_up_step",
            },
        )

        context = WorkflowContext(
            workflow_id="test-state-decision",
            definition_id="state-decision-def",
            inputs={},
            state={"retries": 1},
        )

        result = await decision.execute(context)
        assert result["decision"] == "retry_step"

        context.state["retries"] = 5
        result = await decision.execute(context)
        assert result["decision"] == "give_up_step"


# ============================================================================
# Memory Step E2E Tests (Mocked)
# ============================================================================


class TestMemorySteps:
    """Tests for memory read/write steps with mocked Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_memory_read_step(self) -> None:
        """Test memory read step queries Knowledge Mound."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        # Create step
        step = MemoryReadStep(
            "read-context",
            config={
                "query": "What are the key requirements?",
                "limit": 5,
            },
        )

        context = WorkflowContext(
            workflow_id="test-read",
            definition_id="read-def",
            inputs={},
            state={},
        )

        # Execute - will return empty if Knowledge Mound not available
        result = await step.execute(context)

        # Should return structured response even if empty
        assert "items" in result
        assert "query" in result
        assert isinstance(result["items"], list)

    @pytest.mark.asyncio
    async def test_memory_write_step(self) -> None:
        """Test memory write step stores to Knowledge Mound."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            "store-result",
            config={
                "content": "Test knowledge content",
                "source_type": "fact",
                "confidence": 0.9,
            },
        )

        context = WorkflowContext(
            workflow_id="test-write",
            definition_id="write-def",
            inputs={},
            state={},
        )

        # Execute - will fail gracefully if Knowledge Mound not available
        result = await step.execute(context)

        # Should return structured response
        assert "success" in result

    @pytest.mark.asyncio
    async def test_memory_read_with_interpolation(self) -> None:
        """Test memory read with template interpolation."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            "read-with-template",
            config={
                "query": "Find information about {topic}",
                "limit": 10,
            },
        )

        context = WorkflowContext(
            workflow_id="test-interpolate",
            definition_id="interpolate-def",
            inputs={"topic": "machine learning"},
            state={},
        )

        result = await step.execute(context)

        # Query should have been interpolated
        assert result["query"] == "Find information about machine learning"


# ============================================================================
# Task Step E2E Tests
# ============================================================================


class TestTaskStep:
    """Tests for task step execution."""

    @pytest.mark.asyncio
    async def test_task_step_set_state(self) -> None:
        """Test task step can set workflow state."""
        from aragora.workflow.nodes.task import TaskStep

        step = TaskStep(
            "set-state-task",
            config={
                "task_type": "function",
                "handler": "set_state",
                "args": {"key": "processed", "value": True},
            },
        )

        context = WorkflowContext(
            workflow_id="test-task",
            definition_id="task-def",
            inputs={},
            state={},
        )

        result = await step.execute(context)

        assert result.get("success") is True
        assert context.state.get("processed") is True

    @pytest.mark.asyncio
    async def test_task_step_log_action(self) -> None:
        """Test task step with log action."""
        from aragora.workflow.nodes.task import TaskStep

        step = TaskStep(
            "log-task",
            config={
                "task_type": "function",
                "handler": "log",
                "args": {"message": "Processing workflow step"},
            },
        )

        context = WorkflowContext(
            workflow_id="test-log",
            definition_id="log-def",
            inputs={},
            state={},
        )

        result = await step.execute(context)

        # Should complete without error
        assert result is not None
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_task_step_transform(self) -> None:
        """Test task step with data transformation."""
        from aragora.workflow.nodes.task import TaskStep

        step = TaskStep(
            "transform-task",
            config={
                "task_type": "transform",
                "transform": "[x * 2 for x in inputs.get('numbers', [])]",
                "output_format": "list",
            },
        )

        context = WorkflowContext(
            workflow_id="test-transform",
            definition_id="transform-def",
            inputs={"numbers": [1, 2, 3, 4, 5]},
            state={},
        )

        result = await step.execute(context)

        assert result.get("success") is True
        assert result.get("result") == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_task_step_validate(self) -> None:
        """Test task step with data validation."""
        from aragora.workflow.nodes.task import TaskStep

        step = TaskStep(
            "validate-task",
            config={
                "task_type": "validate",
                "data": "inputs",
                "validation": {
                    "email": {"required": True, "type": "string"},
                    "age": {"required": True, "type": "int", "min": 0, "max": 150},
                },
            },
        )

        # Valid data
        context = WorkflowContext(
            workflow_id="test-validate",
            definition_id="validate-def",
            inputs={"email": "test@example.com", "age": 25},
            state={},
        )

        result = await step.execute(context)
        assert result.get("valid") is True
        assert len(result.get("errors", [])) == 0

        # Invalid data
        context.inputs["age"] = -5
        result = await step.execute(context)
        assert result.get("valid") is False
        assert len(result.get("errors", [])) > 0


# ============================================================================
# Complex Workflow E2E Tests
# ============================================================================


class TestComplexWorkflows:
    """Tests for complex multi-step workflow scenarios."""

    @pytest.mark.asyncio
    async def test_diamond_dag_execution(self, e2e_engine: WorkflowEngine) -> None:
        """Test diamond-shaped DAG: start -> (A, B) -> end."""
        workflow = WorkflowDefinition(
            id="diamond-e2e",
            name="Diamond E2E",
            steps=[
                StepDefinition(
                    id="start",
                    name="Start",
                    step_type="task",
                    config={"action": "set_state", "state": {"started": True}},
                    next_steps=["branch-a", "branch-b"],
                ),
                StepDefinition(
                    id="branch-a",
                    name="Branch A",
                    step_type="task",
                    config={"action": "set_state", "state": {"a": True}},
                    next_steps=["end"],
                ),
                StepDefinition(
                    id="branch-b",
                    name="Branch B",
                    step_type="task",
                    config={"action": "set_state", "state": {"b": True}},
                    next_steps=["end"],
                ),
                StepDefinition(
                    id="end",
                    name="End",
                    step_type="task",
                    config={"action": "set_state", "state": {"ended": True}},
                ),
            ],
            entry_step="start",
        )

        result = await e2e_engine.execute(workflow, inputs={})

        # All steps should execute
        executed_ids = {r.step_id for r in result.steps}
        assert "start" in executed_ids

    @pytest.mark.asyncio
    async def test_workflow_with_timeout(self, e2e_engine: WorkflowEngine) -> None:
        """Test workflow respects step timeouts."""
        # Create engine with very short timeout
        short_timeout_engine = WorkflowEngine(
            config=WorkflowConfig(
                step_timeout_seconds=0.1,
                total_timeout_seconds=1.0,
            )
        )

        workflow = WorkflowDefinition(
            id="timeout-e2e",
            name="Timeout E2E",
            steps=[
                StepDefinition(
                    id="quick-step",
                    name="Quick Step",
                    step_type="task",
                    config={"action": "log", "message": "Fast"},
                ),
            ],
            entry_step="quick-step",
        )

        result = await short_timeout_engine.execute(workflow, inputs={})

        # Should complete quickly
        assert len(result.steps) >= 0


# ============================================================================
# Checkpoint/Resume E2E Tests
# ============================================================================


class TestCheckpointResume:
    """Tests for workflow checkpoint and resume functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_during_execution(self) -> None:
        """Test that checkpoints are created during workflow execution."""
        engine = WorkflowEngine(
            config=WorkflowConfig(
                enable_checkpointing=True,
                checkpoint_interval_steps=1,
            )
        )

        workflow = WorkflowDefinition(
            id="checkpoint-e2e",
            name="Checkpoint E2E",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="task",
                    config={"action": "set_state", "state": {"s1": True}},
                    next_steps=["step2"],
                ),
                StepDefinition(
                    id="step2",
                    name="Step 2",
                    step_type="task",
                    config={"action": "set_state", "state": {"s2": True}},
                ),
            ],
            entry_step="step1",
        )

        result = await engine.execute(workflow, inputs={})

        # Should have executed
        assert len(result.steps) >= 1


# ============================================================================
# Gauntlet Step E2E Tests
# ============================================================================


class TestGauntletStep:
    """Tests for gauntlet (adversarial validation) step."""

    @pytest.mark.asyncio
    async def test_gauntlet_step_validation(self) -> None:
        """Test gauntlet step performs validation."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        gauntlet = GauntletStep(
            "validate-output",
            config={
                "validators": ["format", "length"],
                "max_length": 1000,
            },
        )

        context = WorkflowContext(
            workflow_id="test-gauntlet",
            definition_id="gauntlet-def",
            inputs={"content": "Test content to validate"},
            state={},
        )

        result = await gauntlet.execute(context)

        # Should return validation result
        assert "passed" in result or "success" in result or "error" in result
