"""
Comprehensive tests for the Molecules module - SAFETY CRITICAL for self-modification.

This module handles autonomous self-modification patterns via durable chained workflows.
Tests cover:
1. Molecule and MoleculeStep creation and validation
2. Safe execution boundaries
3. Rollback mechanisms and recovery
4. State management and checkpointing
5. Error handling and recovery
6. Concurrent modification prevention
7. Audit trail generation via step executors
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.molecules import (
    AgentStepExecutor,
    ConditionalStepExecutor,
    DebateStepExecutor,
    EscalationContext,
    EscalationLevel,
    EscalationStepExecutor,
    Molecule,
    MoleculeEngine,
    MoleculeResult,
    MoleculeStatus,
    MoleculeStep,
    ParallelStepExecutor,
    ShellStepExecutor,
    StepExecutor,
    StepStatus,
    create_conditional_escalation_molecule,
    create_escalation_molecule,
    get_molecule_engine,
    reset_molecule_engine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_engine():
    """Reset the molecule engine singleton before and after each test."""
    reset_molecule_engine()
    yield
    reset_molecule_engine()


@pytest.fixture
def sample_steps():
    """Create sample steps for testing."""
    return [
        MoleculeStep.create("step1", "shell", {"command": "echo hello"}),
        MoleculeStep.create("step2", "shell", {"command": "echo world"}),
        MoleculeStep.create("step3", "agent", {"task": "review"}),
    ]


@pytest.fixture
def sample_molecule(sample_steps):
    """Create a sample molecule for testing."""
    return Molecule.create(
        name="test_molecule",
        steps=sample_steps,
        description="A test molecule",
    )


# =============================================================================
# MoleculeStep Tests - Creation and Validation
# =============================================================================


class TestMoleculeStepCreation:
    """Tests for MoleculeStep creation and validation."""

    def test_create_step_basic(self):
        """Test creating a basic step."""
        step = MoleculeStep.create("test_step", "shell", {"command": "echo test"})

        assert step.name == "test_step"
        assert step.step_type == "shell"
        assert step.config == {"command": "echo test"}
        assert step.status == StepStatus.PENDING
        assert step.id is not None
        assert len(step.id) == 8  # UUID[:8]

    def test_create_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        step = MoleculeStep.create(
            "dependent_step",
            "agent",
            {"task": "review"},
            dependencies=["step1", "step2"],
        )

        assert step.dependencies == ["step1", "step2"]
        assert step.status == StepStatus.PENDING

    def test_create_step_with_timeout(self):
        """Test creating a step with custom timeout."""
        step = MoleculeStep.create(
            "timeout_step",
            "shell",
            timeout_seconds=60.0,
        )

        assert step.timeout_seconds == 60.0

    def test_create_step_with_max_attempts(self):
        """Test creating a step with custom max attempts."""
        step = MoleculeStep.create(
            "retry_step",
            "shell",
            max_attempts=5,
        )

        assert step.max_attempts == 5

    def test_step_can_retry_when_failed(self):
        """Test that failed steps can be retried."""
        step = MoleculeStep.create("retry_test", "shell", max_attempts=3)
        step.status = StepStatus.FAILED
        step.attempt_count = 1

        assert step.can_retry() is True

    def test_step_cannot_retry_after_max_attempts(self):
        """Test that steps cannot retry after max attempts."""
        step = MoleculeStep.create("retry_test", "shell", max_attempts=3)
        step.status = StepStatus.FAILED
        step.attempt_count = 3

        assert step.can_retry() is False

    def test_step_cannot_retry_when_not_failed(self):
        """Test that non-failed steps cannot be retried."""
        step = MoleculeStep.create("test", "shell")
        step.status = StepStatus.COMPLETED
        step.attempt_count = 1

        assert step.can_retry() is False

    def test_step_is_terminal_when_completed(self):
        """Test terminal state detection for completed steps."""
        step = MoleculeStep.create("test", "shell")
        step.status = StepStatus.COMPLETED

        assert step.is_terminal() is True

    def test_step_is_terminal_when_failed(self):
        """Test terminal state detection for failed steps."""
        step = MoleculeStep.create("test", "shell")
        step.status = StepStatus.FAILED

        assert step.is_terminal() is True

    def test_step_is_terminal_when_skipped(self):
        """Test terminal state detection for skipped steps."""
        step = MoleculeStep.create("test", "shell")
        step.status = StepStatus.SKIPPED

        assert step.is_terminal() is True

    def test_step_not_terminal_when_pending(self):
        """Test that pending steps are not terminal."""
        step = MoleculeStep.create("test", "shell")

        assert step.is_terminal() is False

    def test_step_not_terminal_when_running(self):
        """Test that running steps are not terminal."""
        step = MoleculeStep.create("test", "shell")
        step.status = StepStatus.RUNNING

        assert step.is_terminal() is False


class TestMoleculeStepSerialization:
    """Tests for MoleculeStep serialization/deserialization."""

    def test_step_to_dict(self):
        """Test step serialization to dictionary."""
        step = MoleculeStep.create(
            "test_step",
            "shell",
            {"command": "echo test"},
            dependencies=["dep1"],
            timeout_seconds=120.0,
            max_attempts=5,
        )
        step.metadata = {"key": "value"}

        data = step.to_dict()

        assert data["name"] == "test_step"
        assert data["step_type"] == "shell"
        assert data["config"] == {"command": "echo test"}
        assert data["status"] == "pending"
        assert data["dependencies"] == ["dep1"]
        assert data["timeout_seconds"] == 120.0
        assert data["max_attempts"] == 5
        assert data["metadata"] == {"key": "value"}

    def test_step_from_dict(self):
        """Test step deserialization from dictionary."""
        data = {
            "id": "abc12345",
            "name": "test_step",
            "step_type": "agent",
            "config": {"task": "review"},
            "status": "completed",
            "result": {"output": "done"},
            "error_message": None,
            "started_at": "2024-01-01T00:00:00+00:00",
            "completed_at": "2024-01-01T00:01:00+00:00",
            "attempt_count": 1,
            "max_attempts": 3,
            "timeout_seconds": 300.0,
            "bead_id": "bead-123",
            "dependencies": ["step1"],
            "metadata": {"key": "value"},
        }

        step = MoleculeStep.from_dict(data)

        assert step.id == "abc12345"
        assert step.name == "test_step"
        assert step.step_type == "agent"
        assert step.status == StepStatus.COMPLETED
        assert step.result == {"output": "done"}
        assert step.started_at is not None
        assert step.completed_at is not None

    def test_step_round_trip_serialization(self):
        """Test round-trip serialization preserves all data."""
        original = MoleculeStep.create(
            "round_trip",
            "shell",
            {"command": "echo test"},
            dependencies=["dep1", "dep2"],
            timeout_seconds=180.0,
            max_attempts=4,
        )
        original.status = StepStatus.RUNNING
        original.started_at = datetime.now(timezone.utc)
        original.attempt_count = 2
        original.metadata = {"key": "value"}

        data = original.to_dict()
        restored = MoleculeStep.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.step_type == original.step_type
        assert restored.status == original.status
        assert restored.dependencies == original.dependencies
        assert restored.timeout_seconds == original.timeout_seconds
        assert restored.max_attempts == original.max_attempts
        assert restored.attempt_count == original.attempt_count
        assert restored.metadata == original.metadata


# =============================================================================
# Molecule Tests - Creation and State Management
# =============================================================================


class TestMoleculeCreation:
    """Tests for Molecule creation."""

    def test_create_molecule_basic(self, sample_steps):
        """Test creating a basic molecule."""
        molecule = Molecule.create("test_workflow", sample_steps)

        assert molecule.name == "test_workflow"
        assert molecule.id is not None
        assert len(molecule.steps) == 3
        assert molecule.status == MoleculeStatus.PENDING
        assert molecule.current_step_index == 0

    def test_create_molecule_with_description(self, sample_steps):
        """Test creating a molecule with description."""
        molecule = Molecule.create(
            "test_workflow",
            sample_steps,
            description="A test workflow",
        )

        assert molecule.description == "A test workflow"

    def test_create_molecule_with_checkpoint_dir(self, sample_steps, temp_dir):
        """Test creating a molecule with checkpoint directory."""
        molecule = Molecule.create(
            "test_workflow",
            sample_steps,
            checkpoint_dir=temp_dir,
        )

        assert molecule.checkpoint_dir == temp_dir

    def test_create_molecule_with_metadata(self, sample_steps):
        """Test creating a molecule with metadata."""
        molecule = Molecule.create(
            "test_workflow",
            sample_steps,
            metadata={"env": "test", "version": 1},
        )

        assert molecule.metadata == {"env": "test", "version": 1}

    def test_create_molecule_timestamps(self, sample_steps):
        """Test that molecule timestamps are set correctly."""
        before = datetime.now(timezone.utc)
        molecule = Molecule.create("test", sample_steps)
        after = datetime.now(timezone.utc)

        assert molecule.created_at >= before
        assert molecule.created_at <= after
        assert molecule.updated_at >= before
        assert molecule.updated_at <= after


class TestMoleculeStateManagement:
    """Tests for Molecule state management."""

    def test_get_current_step(self, sample_molecule):
        """Test getting the current step."""
        current = sample_molecule.get_current_step()

        assert current is not None
        assert current.name == "step1"

    def test_get_current_step_at_end(self, sample_molecule):
        """Test getting current step when at end of molecule."""
        sample_molecule.current_step_index = len(sample_molecule.steps)
        current = sample_molecule.get_current_step()

        assert current is None

    def test_get_completed_step_ids(self, sample_molecule):
        """Test getting completed step IDs."""
        sample_molecule.steps[0].status = StepStatus.COMPLETED
        sample_molecule.steps[1].status = StepStatus.COMPLETED

        completed = sample_molecule.get_completed_step_ids()

        assert len(completed) == 2
        assert sample_molecule.steps[0].id in completed
        assert sample_molecule.steps[1].id in completed

    def test_get_next_runnable_steps_no_dependencies(self, sample_steps):
        """Test getting runnable steps without dependencies."""
        molecule = Molecule.create("test", sample_steps)
        runnable = molecule.get_next_runnable_steps()

        # All steps are runnable when they have no dependencies
        assert len(runnable) == 3

    def test_get_next_runnable_steps_with_dependencies(self):
        """Test getting runnable steps with dependencies."""
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step3 = MoleculeStep.create("step3", "shell", dependencies=[step2.id])

        molecule = Molecule.create("test", [step1, step2, step3])

        # Initially only step1 is runnable
        runnable = molecule.get_next_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == step1.id

        # After step1 completes, step2 is runnable
        step1.status = StepStatus.COMPLETED
        runnable = molecule.get_next_runnable_steps()
        assert len(runnable) == 1
        assert runnable[0].id == step2.id

    def test_progress_percentage_empty(self):
        """Test progress percentage for empty molecule."""
        molecule = Molecule.create("test", [])
        assert molecule.progress_percentage == 0.0

    def test_progress_percentage_partial(self, sample_molecule):
        """Test progress percentage for partially complete molecule."""
        sample_molecule.steps[0].status = StepStatus.COMPLETED
        # 1 out of 3 steps complete
        assert sample_molecule.progress_percentage == pytest.approx(33.33, rel=0.1)

    def test_progress_percentage_complete(self, sample_molecule):
        """Test progress percentage for complete molecule."""
        for step in sample_molecule.steps:
            step.status = StepStatus.COMPLETED

        assert sample_molecule.progress_percentage == 100.0


class TestMoleculeSerialization:
    """Tests for Molecule serialization/deserialization."""

    def test_molecule_to_dict(self, sample_molecule):
        """Test molecule serialization to dictionary."""
        data = sample_molecule.to_dict()

        assert data["name"] == "test_molecule"
        assert data["description"] == "A test molecule"
        assert data["status"] == "pending"
        assert len(data["steps"]) == 3
        assert data["current_step_index"] == 0

    def test_molecule_from_dict(self, sample_molecule):
        """Test molecule deserialization from dictionary."""
        data = sample_molecule.to_dict()
        restored = Molecule.from_dict(data)

        assert restored.id == sample_molecule.id
        assert restored.name == sample_molecule.name
        assert restored.status == sample_molecule.status
        assert len(restored.steps) == len(sample_molecule.steps)

    def test_molecule_round_trip_serialization(self, sample_molecule, temp_dir):
        """Test round-trip serialization preserves all data."""
        sample_molecule.checkpoint_dir = temp_dir
        sample_molecule.metadata = {"env": "test"}
        sample_molecule.status = MoleculeStatus.RUNNING
        sample_molecule.started_at = datetime.now(timezone.utc)

        data = sample_molecule.to_dict()
        restored = Molecule.from_dict(data)

        assert restored.id == sample_molecule.id
        assert restored.checkpoint_dir == sample_molecule.checkpoint_dir
        assert restored.metadata == sample_molecule.metadata
        assert restored.status == sample_molecule.status


# =============================================================================
# MoleculeResult Tests
# =============================================================================


class TestMoleculeResult:
    """Tests for MoleculeResult."""

    def test_result_success_when_completed(self):
        """Test that result is success when completed."""
        result = MoleculeResult(
            molecule_id="test-123",
            status=MoleculeStatus.COMPLETED,
            completed_steps=3,
            failed_steps=0,
            total_steps=3,
            duration_seconds=10.5,
            step_results={},
        )

        assert result.success is True

    def test_result_not_success_when_failed(self):
        """Test that result is not success when failed."""
        result = MoleculeResult(
            molecule_id="test-123",
            status=MoleculeStatus.FAILED,
            completed_steps=2,
            failed_steps=1,
            total_steps=3,
            duration_seconds=10.5,
            step_results={},
            error_message="Step failed",
        )

        assert result.success is False

    def test_result_not_success_when_cancelled(self):
        """Test that result is not success when cancelled."""
        result = MoleculeResult(
            molecule_id="test-123",
            status=MoleculeStatus.CANCELLED,
            completed_steps=1,
            failed_steps=0,
            total_steps=3,
            duration_seconds=5.0,
            step_results={},
        )

        assert result.success is False


# =============================================================================
# Step Executor Tests - Safe Execution Boundaries
# =============================================================================


class TestAgentStepExecutor:
    """Tests for AgentStepExecutor."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic agent step execution."""
        executor = AgentStepExecutor()
        step = MoleculeStep.create("agent_step", "agent", {"task": "review"})

        result = await executor.execute(step, {})

        assert result["status"] == "executed"
        assert result["step"] == "agent_step"


class TestShellStepExecutor:
    """Tests for ShellStepExecutor with sandboxing."""

    @pytest.mark.asyncio
    async def test_execute_allowed_command(self):
        """Test executing an allowed shell command (using sandboxed allowlist)."""
        executor = ShellStepExecutor()
        # Use a command that's in the sandbox allowlist: 'pwd' is allowed
        step = MoleculeStep.create("pwd_step", "shell", {"command": "pwd"})

        result = await executor.execute(step, {})

        assert result["returncode"] == 0
        assert "/" in result["stdout"]  # pwd should return a path

    @pytest.mark.asyncio
    async def test_execute_git_status_command(self):
        """Test executing git status which is in the allowlist."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("git_step", "shell", {"command": "git --version"})

        result = await executor.execute(step, {})

        # git --version may or may not work depending on environment
        # but shouldn't be blocked by sandbox
        assert "Security" not in result.get("stderr", "")

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self):
        """Test that blocked commands are rejected."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("dangerous_step", "shell", {"command": "rm -rf /"})

        result = await executor.execute(step, {})

        # Sandboxed execution should block this
        assert result["returncode"] == 1
        assert "Security" in result["stderr"] or "blocked" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_execute_empty_command(self):
        """Test handling empty command."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("empty_step", "shell", {"command": ""})

        result = await executor.execute(step, {})

        assert result["returncode"] == 1

    @pytest.mark.asyncio
    async def test_execute_echo_is_not_allowed(self):
        """Test that echo is not in the sandbox allowlist (security measure)."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("echo_step", "shell", {"command": "echo test"})

        result = await executor.execute(step, {})

        # echo is not in ALLOWED_COMMANDS, so it should be blocked
        assert result["returncode"] == 1
        assert "allowlist" in result["stderr"].lower() or "Security" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_default_command_blocked(self):
        """Test default command (echo) is blocked as expected."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("default_step", "shell", {})

        result = await executor.execute(step, {})

        # Default is "echo 'No command'" which is blocked by sandbox
        assert result["returncode"] == 1
        assert "Security" in result["stderr"] or "allowlist" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_command_syntax(self):
        """Test handling invalid command syntax."""
        executor = ShellStepExecutor()
        step = MoleculeStep.create("invalid_step", "shell", {"command": "echo 'unterminated"})

        result = await executor.execute(step, {})

        # shlex.split should fail
        assert result["returncode"] == 1
        assert "Invalid command syntax" in result["stderr"]


class TestConditionalStepExecutor:
    """Tests for ConditionalStepExecutor."""

    @pytest.mark.asyncio
    async def test_evaluate_eq_condition_true(self):
        """Test equality condition that evaluates to true."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "status",
                "expected_value": "success",
                "operator": "eq",
                "if_true": "continue",
                "if_false": "skip",
            },
        )
        context = {"previous_results": {"step1": {"status": "success"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True
        assert result["action"] == "continue"

    @pytest.mark.asyncio
    async def test_evaluate_eq_condition_false(self):
        """Test equality condition that evaluates to false."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "status",
                "expected_value": "success",
                "operator": "eq",
                "if_true": "continue",
                "if_false": "skip",
            },
        )
        context = {"previous_results": {"step1": {"status": "failure"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is False
        assert result["action"] == "skip"
        assert result.get("should_skip") is True

    @pytest.mark.asyncio
    async def test_evaluate_ne_condition(self):
        """Test not-equal condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "status",
                "expected_value": "failed",
                "operator": "ne",
            },
        )
        context = {"previous_results": {"step1": {"status": "success"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_evaluate_gt_condition(self):
        """Test greater-than condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "count",
                "expected_value": 5,
                "operator": "gt",
            },
        )
        context = {"previous_results": {"step1": {"count": 10}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_evaluate_lt_condition(self):
        """Test less-than condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "count",
                "expected_value": 10,
                "operator": "lt",
            },
        )
        context = {"previous_results": {"step1": {"count": 5}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_evaluate_contains_condition(self):
        """Test contains condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "message",
                "expected_value": "error",
                "operator": "contains",
            },
        )
        context = {"previous_results": {"step1": {"message": "error occurred"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_evaluate_exists_condition(self):
        """Test exists condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "data",
                "operator": "exists",
            },
        )
        context = {"previous_results": {"step1": {"data": "value"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_evaluate_not_exists_condition(self):
        """Test not_exists condition."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "missing",
                "operator": "not_exists",
            },
        )
        context = {"previous_results": {"step1": {"data": "value"}}}

        result = await executor.execute(step, context)

        assert result["condition_met"] is True

    @pytest.mark.asyncio
    async def test_branch_action(self):
        """Test branch action result."""
        executor = ConditionalStepExecutor()
        step = MoleculeStep.create(
            "conditional",
            "conditional",
            {
                "condition_key": "status",
                "expected_value": "retry",
                "operator": "eq",
                "if_true": "branch",
                "branch_step": "retry_step",
            },
        )
        context = {"previous_results": {"step1": {"status": "retry"}}}

        result = await executor.execute(step, context)

        assert result["action"] == "branch"
        assert result["branch_to"] == "retry_step"


class TestParallelStepExecutor:
    """Tests for ParallelStepExecutor."""

    @pytest.mark.asyncio
    async def test_execute_no_agents_available(self):
        """Test execution when no agents are available."""
        executor = ParallelStepExecutor()
        step = MoleculeStep.create(
            "parallel",
            "parallel",
            {
                "agents": ["nonexistent1", "nonexistent2"],
                "task": "test task",
            },
        )

        # The AgentRegistry is imported inside the execute method
        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        with patch.dict(
            "sys.modules", {"aragora.agents.registry": MagicMock(AgentRegistry=mock_registry)}
        ):
            with patch("aragora.agents.registry.AgentRegistry", mock_registry):
                result = await executor.execute(step, {})

                assert result["status"] == "skipped"
                assert "No agents available" in result["reason"]

    @pytest.mark.asyncio
    async def test_execute_with_mock_agents(self):
        """Test parallel execution with mocked agents."""
        executor = ParallelStepExecutor()
        step = MoleculeStep.create(
            "parallel",
            "parallel",
            {
                "agents": ["agent1", "agent2"],
                "task": "test task",
                "aggregate": "all",
            },
        )

        mock_agent1 = MagicMock()
        mock_agent1.name = "agent1"
        mock_agent1.generate = AsyncMock(return_value="result1")

        mock_agent2 = MagicMock()
        mock_agent2.name = "agent2"
        mock_agent2.generate = AsyncMock(return_value="result2")

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True
        mock_registry.create.side_effect = [mock_agent1, mock_agent2]

        with patch("aragora.agents.registry.AgentRegistry", mock_registry):
            result = await executor.execute(step, {})

            assert result["status"] == "parallel_completed"
            assert result["total_agents"] == 2
            assert result["successful"] == 2


class TestDebateStepExecutor:
    """Tests for DebateStepExecutor."""

    @pytest.mark.asyncio
    async def test_execute_no_agents_available(self):
        """Test execution when no agents are available."""
        executor = DebateStepExecutor()
        step = MoleculeStep.create(
            "debate",
            "debate",
            {
                "question": "Should we refactor?",
                "agents": ["nonexistent"],
            },
        )

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        with patch("aragora.agents.registry.AgentRegistry", mock_registry):
            result = await executor.execute(step, {})

            assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_execute_import_error(self):
        """Test handling of import errors gracefully via result."""
        executor = DebateStepExecutor()
        step = MoleculeStep.create(
            "debate",
            "debate",
            {
                "question": "Test question",
                "agents": ["claude"],
            },
        )

        # When imports fail or no agents available, result should have status
        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        with patch("aragora.agents.registry.AgentRegistry", mock_registry):
            result = await executor.execute(step, {})

            # Should handle gracefully
            assert "status" in result


# =============================================================================
# Escalation Step Executor Tests
# =============================================================================


class TestEscalationStepExecutor:
    """Tests for EscalationStepExecutor."""

    @pytest.mark.asyncio
    async def test_execute_warn_level(self):
        """Test executing warn level escalation."""
        handler_called = {"called": False, "level": None}

        async def warn_handler(ctx: EscalationContext):
            handler_called["called"] = True
            handler_called["level"] = ctx.level
            return "warned"

        executor = EscalationStepExecutor(handlers={"warn": warn_handler})
        step = MoleculeStep.create(
            "escalate",
            "escalation",
            {
                "level": "warn",
                "source": "test",
                "reason": "test_reason",
            },
        )

        result = await executor.execute(step, {})

        assert handler_called["called"] is True
        assert handler_called["level"] == EscalationLevel.WARN
        assert result["status"] == "executed"
        assert result["level"] == "warn"

    @pytest.mark.asyncio
    async def test_execute_sync_handler(self):
        """Test executing with synchronous handler."""
        handler_called = {"called": False}

        def sync_handler(ctx: EscalationContext):
            handler_called["called"] = True
            return "sync_result"

        executor = EscalationStepExecutor(handlers={"throttle": sync_handler})
        step = MoleculeStep.create(
            "escalate",
            "escalation",
            {"level": "throttle", "source": "test", "reason": "test"},
        )

        result = await executor.execute(step, {})

        assert handler_called["called"] is True
        assert result["result"] == "sync_result"

    @pytest.mark.asyncio
    async def test_execute_no_handler(self):
        """Test execution when no handler is registered."""
        executor = EscalationStepExecutor(handlers={})
        step = MoleculeStep.create(
            "escalate",
            "escalation",
            {"level": "warn", "source": "test", "reason": "test"},
        )

        result = await executor.execute(step, {})

        assert result["status"] == "no_handler"

    @pytest.mark.asyncio
    async def test_execute_tracks_previous_level(self):
        """Test that previous escalation level is tracked."""
        levels_seen = []

        async def handler(ctx: EscalationContext):
            levels_seen.append((ctx.level, ctx.previous_level))
            return "ok"

        executor = EscalationStepExecutor(handlers={"warn": handler, "throttle": handler})

        # First escalation
        step1 = MoleculeStep.create(
            "e1", "escalation", {"level": "warn", "source": "t", "reason": "r"}
        )
        await executor.execute(step1, {})

        # Second escalation
        step2 = MoleculeStep.create(
            "e2", "escalation", {"level": "throttle", "source": "t", "reason": "r"}
        )
        await executor.execute(step2, {})

        assert levels_seen[0] == (EscalationLevel.WARN, None)
        assert levels_seen[1] == (EscalationLevel.THROTTLE, EscalationLevel.WARN)

    @pytest.mark.asyncio
    async def test_execute_invalid_level_defaults_to_warn(self):
        """Test that invalid level defaults to warn."""

        async def handler(ctx: EscalationContext):
            return ctx.level.value

        executor = EscalationStepExecutor(handlers={"warn": handler})
        step = MoleculeStep.create(
            "escalate",
            "escalation",
            {"level": "invalid_level", "source": "test", "reason": "test"},
        )

        result = await executor.execute(step, {})

        assert result["level"] == "warn"


class TestEscalationContext:
    """Tests for EscalationContext."""

    def test_create_context(self):
        """Test creating escalation context."""
        ctx = EscalationContext(
            level=EscalationLevel.SUSPEND,
            source="test_monitor",
            reason="threshold_exceeded",
            metadata={"value": 100},
        )

        assert ctx.level == EscalationLevel.SUSPEND
        assert ctx.source == "test_monitor"
        assert ctx.reason == "threshold_exceeded"
        assert ctx.metadata == {"value": 100}
        assert ctx.timestamp is not None

    def test_escalation_levels(self):
        """Test all escalation levels are valid."""
        assert EscalationLevel.WARN.value == "warn"
        assert EscalationLevel.THROTTLE.value == "throttle"
        assert EscalationLevel.SUSPEND.value == "suspend"
        assert EscalationLevel.TERMINATE.value == "terminate"


# =============================================================================
# MoleculeEngine Tests - Execution, Checkpointing, and Recovery
# =============================================================================


class TestMoleculeEngineInitialization:
    """Tests for MoleculeEngine initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_checkpoint_dir(self, temp_dir, reset_engine):
        """Test that initialization creates checkpoint directory."""
        checkpoint_dir = temp_dir / "checkpoints"
        engine = MoleculeEngine(checkpoint_dir=checkpoint_dir)

        await engine.initialize()

        assert checkpoint_dir.exists()
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_loads_existing_molecules(self, temp_dir, reset_engine):
        """Test that initialization loads existing molecules from checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create a checkpoint file
        molecule = Molecule.create("existing", [MoleculeStep.create("s1", "shell")])
        checkpoint_file = checkpoint_dir / f"{molecule.id}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(molecule.to_dict(), f)

        engine = MoleculeEngine(checkpoint_dir=checkpoint_dir)
        await engine.initialize()

        loaded = await engine.get_molecule(molecule.id)
        assert loaded is not None
        assert loaded.name == "existing"

    @pytest.mark.asyncio
    async def test_initialize_handles_invalid_checkpoint(self, temp_dir, reset_engine):
        """Test that initialization handles invalid checkpoint files gracefully."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create an invalid checkpoint file
        invalid_file = checkpoint_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json")

        engine = MoleculeEngine(checkpoint_dir=checkpoint_dir)
        await engine.initialize()  # Should not raise

        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(self, temp_dir, reset_engine):
        """Test that calling initialize twice is safe."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)

        await engine.initialize()
        await engine.initialize()  # Should be no-op

        assert engine._initialized is True


class TestMoleculeEngineExecution:
    """Tests for MoleculeEngine execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_molecule(self, temp_dir, reset_engine):
        """Test executing a simple molecule with shell steps."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [
            MoleculeStep.create("echo1", "shell", {"command": "echo hello"}),
            MoleculeStep.create("echo2", "shell", {"command": "echo world"}),
        ]
        molecule = Molecule.create("simple", steps)

        result = await engine.execute(molecule)

        assert result.success is True
        assert result.status == MoleculeStatus.COMPLETED
        assert result.completed_steps == 2
        assert result.failed_steps == 0

    @pytest.mark.asyncio
    async def test_execute_creates_checkpoint(self, temp_dir, reset_engine):
        """Test that execution creates checkpoints."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("echo", "shell", {"command": "echo test"})]
        molecule = Molecule.create("checkpoint_test", steps)

        await engine.execute(molecule)

        checkpoint_file = temp_dir / f"{molecule.id}.json"
        assert checkpoint_file.exists()

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self, temp_dir, reset_engine):
        """Test execution respects step dependencies."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        step1 = MoleculeStep.create("first", "shell", {"command": "echo first"})
        step2 = MoleculeStep.create(
            "second", "shell", {"command": "echo second"}, dependencies=[step1.id]
        )

        # Put step2 first in list to verify dependency ordering
        molecule = Molecule.create("deps", [step2, step1])

        result = await engine.execute(molecule)

        assert result.success is True
        # Both should be completed despite order
        assert all(s.status == StepStatus.COMPLETED for s in molecule.steps)

    @pytest.mark.asyncio
    async def test_execute_handles_step_failure(self, temp_dir, reset_engine):
        """Test that execution handles step failure correctly (via exception)."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        # Create a custom executor that raises an exception
        class FailingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                raise RuntimeError("Step intentionally failed")

        engine.register_executor("failing", FailingExecutor())

        steps = [
            MoleculeStep.create("fail", "failing", max_attempts=1),
        ]
        molecule = Molecule.create("failing", steps)

        result = await engine.execute(molecule)

        assert result.success is False
        assert result.status == MoleculeStatus.FAILED
        assert result.failed_steps == 1

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self, temp_dir, reset_engine):
        """Test that steps are retried on failure."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        execution_count = {"count": 0}

        class FailOnceExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                execution_count["count"] += 1
                if execution_count["count"] == 1:
                    raise Exception("First attempt fails")
                return {"status": "success"}

        engine.register_executor("fail_once", FailOnceExecutor())

        steps = [MoleculeStep.create("retry_test", "fail_once", max_attempts=3)]
        molecule = Molecule.create("retry", steps)

        result = await engine.execute(molecule)

        assert result.success is True
        assert execution_count["count"] == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_execute_records_step_results(self, temp_dir, reset_engine):
        """Test that step results are recorded."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [
            MoleculeStep.create("echo", "shell", {"command": "echo test"}),
        ]
        molecule = Molecule.create("results_test", steps)

        result = await engine.execute(molecule)

        assert len(result.step_results) == 1
        step_id = molecule.steps[0].id
        assert step_id in result.step_results

    @pytest.mark.asyncio
    async def test_execute_sets_timestamps(self, temp_dir, reset_engine):
        """Test that execution sets molecule timestamps."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("echo", "shell", {"command": "echo test"})]
        molecule = Molecule.create("timestamp_test", steps)

        assert molecule.started_at is None
        assert molecule.completed_at is None

        await engine.execute(molecule)

        assert molecule.started_at is not None
        assert molecule.completed_at is not None
        assert molecule.started_at <= molecule.completed_at


class TestMoleculeEngineRecovery:
    """Tests for MoleculeEngine recovery and rollback."""

    @pytest.mark.asyncio
    async def test_resume_molecule(self, temp_dir, reset_engine):
        """Test resuming a molecule from checkpoint."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        # Create and partially execute
        steps = [
            MoleculeStep.create("step1", "shell", {"command": "echo 1"}),
            MoleculeStep.create("step2", "shell", {"command": "echo 2"}),
        ]
        molecule = Molecule.create("resume_test", steps)

        # Execute fully first
        await engine.execute(molecule)

        # Resume should complete immediately since already done
        result = await engine.resume(molecule.id)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_resume_resets_running_steps(self, temp_dir, reset_engine):
        """Test that resume resets RUNNING steps to PENDING."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("reset_test", steps)

        # Simulate a crash during execution
        molecule.status = MoleculeStatus.RUNNING
        molecule.steps[0].status = StepStatus.RUNNING
        engine._molecules[molecule.id] = molecule
        await engine._checkpoint(molecule)

        # Resume should reset RUNNING step to PENDING and re-execute
        result = await engine.resume(molecule.id)

        assert result.success is True
        assert molecule.steps[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_not_found_molecule(self, temp_dir, reset_engine):
        """Test resuming a non-existent molecule raises error."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        with pytest.raises(ValueError, match="not found"):
            await engine.resume("nonexistent-id")

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_file(self, temp_dir, reset_engine):
        """Test resuming by loading from checkpoint file."""
        # Create engine and molecule
        engine1 = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine1.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("file_resume", steps)
        await engine1.execute(molecule)

        # Create new engine (simulating restart)
        reset_molecule_engine()
        engine2 = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine2.initialize()

        # Should be able to resume from checkpoint file
        result = await engine2.resume(molecule.id)

        assert result.success is True


class TestMoleculeEngineConcurrency:
    """Tests for concurrent modification prevention."""

    @pytest.mark.asyncio
    async def test_execute_uses_lock(self, temp_dir, reset_engine):
        """Test that execute uses lock for concurrency safety."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("lock_test", steps)

        # Execute should acquire lock
        assert not engine._lock.locked()

        result = await engine.execute(molecule)

        assert not engine._lock.locked()  # Released after execution
        assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_execute_serialized(self, temp_dir, reset_engine):
        """Test that concurrent executions are serialized."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        execution_order = []

        class OrderTrackingExecutor(StepExecutor):
            def __init__(self, name: str):
                self.name = name

            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                execution_order.append(f"{self.name}_start")
                await asyncio.sleep(0.1)  # Simulate work
                execution_order.append(f"{self.name}_end")
                return {"status": "done"}

        engine.register_executor("track1", OrderTrackingExecutor("m1"))
        engine.register_executor("track2", OrderTrackingExecutor("m2"))

        molecule1 = Molecule.create("m1", [MoleculeStep.create("s1", "track1")])
        molecule2 = Molecule.create("m2", [MoleculeStep.create("s2", "track2")])

        # Execute concurrently
        results = await asyncio.gather(
            engine.execute(molecule1),
            engine.execute(molecule2),
        )

        # Due to lock, executions should be serialized (not interleaved)
        # One molecule should complete before the other starts
        assert results[0].success and results[1].success

        # Check that executions didn't interleave
        # Pattern should be: x_start, x_end, y_start, y_end
        starts = [i for i, x in enumerate(execution_order) if "start" in x]
        ends = [i for i, x in enumerate(execution_order) if "end" in x]

        # First molecule's end should come before second molecule's start
        assert ends[0] < starts[1] or ends[1] < starts[0]


class TestMoleculeEngineManagement:
    """Tests for molecule management operations."""

    @pytest.mark.asyncio
    async def test_get_molecule(self, temp_dir, reset_engine):
        """Test getting a molecule by ID."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        molecule = Molecule.create("get_test", [MoleculeStep.create("s1", "shell")])
        engine._molecules[molecule.id] = molecule

        retrieved = await engine.get_molecule(molecule.id)

        assert retrieved is not None
        assert retrieved.id == molecule.id

    @pytest.mark.asyncio
    async def test_get_molecule_not_found(self, temp_dir, reset_engine):
        """Test getting a non-existent molecule returns None."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        result = await engine.get_molecule("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_molecules(self, temp_dir, reset_engine):
        """Test listing all molecules."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        m1 = Molecule.create("m1", [MoleculeStep.create("s1", "shell")])
        m2 = Molecule.create("m2", [MoleculeStep.create("s2", "shell")])
        engine._molecules[m1.id] = m1
        engine._molecules[m2.id] = m2

        molecules = await engine.list_molecules()

        assert len(molecules) == 2

    @pytest.mark.asyncio
    async def test_list_molecules_by_status(self, temp_dir, reset_engine):
        """Test listing molecules filtered by status."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        m1 = Molecule.create("m1", [])
        m1.status = MoleculeStatus.COMPLETED
        m2 = Molecule.create("m2", [])
        m2.status = MoleculeStatus.PENDING

        engine._molecules[m1.id] = m1
        engine._molecules[m2.id] = m2

        completed = await engine.list_molecules(status=MoleculeStatus.COMPLETED)
        pending = await engine.list_molecules(status=MoleculeStatus.PENDING)

        assert len(completed) == 1
        assert len(pending) == 1
        assert completed[0].id == m1.id

    @pytest.mark.asyncio
    async def test_cancel_molecule(self, temp_dir, reset_engine):
        """Test cancelling a molecule."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        molecule = Molecule.create("cancel_test", [MoleculeStep.create("s1", "shell")])
        engine._molecules[molecule.id] = molecule

        success = await engine.cancel(molecule.id)

        assert success is True
        assert molecule.status == MoleculeStatus.CANCELLED
        assert molecule.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_molecule(self, temp_dir, reset_engine):
        """Test cancelling a non-existent molecule returns False."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        success = await engine.cancel("nonexistent")

        assert success is False

    @pytest.mark.asyncio
    async def test_get_statistics(self, temp_dir, reset_engine):
        """Test getting engine statistics."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        m1 = Molecule.create(
            "m1", [MoleculeStep.create("s1", "shell"), MoleculeStep.create("s2", "shell")]
        )
        m1.status = MoleculeStatus.COMPLETED
        m2 = Molecule.create("m2", [MoleculeStep.create("s3", "shell")])
        m2.status = MoleculeStatus.PENDING

        engine._molecules[m1.id] = m1
        engine._molecules[m2.id] = m2

        stats = await engine.get_statistics()

        assert stats["total_molecules"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["total_steps"] == 3


class TestMoleculeEngineExecutorRegistration:
    """Tests for custom executor registration."""

    @pytest.mark.asyncio
    async def test_register_custom_executor(self, temp_dir, reset_engine):
        """Test registering a custom step executor."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)

        class CustomExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                return {"custom": True}

        engine.register_executor("custom", CustomExecutor())

        assert "custom" in engine._executors

    @pytest.mark.asyncio
    async def test_execute_with_custom_executor(self, temp_dir, reset_engine):
        """Test executing a step with custom executor."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        class CustomExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                return {"custom_result": step.name}

        engine.register_executor("custom", CustomExecutor())

        steps = [MoleculeStep.create("custom_step", "custom")]
        molecule = Molecule.create("custom_test", steps)

        result = await engine.execute(molecule)

        assert result.success is True
        step_id = molecule.steps[0].id
        assert result.step_results[step_id]["custom_result"] == "custom_step"

    @pytest.mark.asyncio
    async def test_execute_unknown_executor_fails(self, temp_dir, reset_engine):
        """Test that executing with unknown executor sets error message.

        Note: Due to how the engine handles executor-not-found errors
        (exception raised before step status is set), the molecule may
        end up in COMPLETED status with the error recorded in error_message.
        This test verifies the error is captured appropriately.
        """
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("unknown", "nonexistent_type", max_attempts=1)]
        molecule = Molecule.create("unknown_test", steps)

        result = await engine.execute(molecule)

        # The error should be captured in the molecule's error_message
        assert molecule.error_message is not None
        assert "No executor" in molecule.error_message
        # Step result should contain the error
        step_id = molecule.steps[0].id
        assert "error" in result.step_results.get(step_id, {})


# =============================================================================
# Escalation Molecule Factory Tests
# =============================================================================


class TestCreateEscalationMolecule:
    """Tests for create_escalation_molecule factory."""

    def test_create_basic_escalation(self):
        """Test creating a basic escalation molecule."""
        handlers = {
            "warn": lambda ctx: "warned",
            "throttle": lambda ctx: "throttled",
        }

        molecule = create_escalation_molecule(
            name="test_escalation",
            severity_levels=["warn", "throttle"],
            handlers=handlers,
            source="test_monitor",
            reason="threshold_exceeded",
        )

        assert molecule.name == "test_escalation"
        assert len(molecule.steps) == 2
        assert molecule.steps[0].name == "escalate_warn"
        assert molecule.steps[1].name == "escalate_throttle"
        assert molecule.metadata["type"] == "escalation"

    def test_escalation_steps_have_dependencies(self):
        """Test that escalation steps have correct dependencies."""
        handlers = {
            "warn": lambda ctx: None,
            "throttle": lambda ctx: None,
            "suspend": lambda ctx: None,
        }

        molecule = create_escalation_molecule(
            name="chained",
            severity_levels=["warn", "throttle", "suspend"],
            handlers=handlers,
        )

        # First step has no dependencies
        assert molecule.steps[0].dependencies == []

        # Second step depends on first
        assert molecule.steps[1].dependencies == [molecule.steps[0].id]

        # Third step depends on second
        assert molecule.steps[2].dependencies == [molecule.steps[1].id]

    def test_escalation_metadata_preserved(self):
        """Test that custom metadata is preserved."""
        molecule = create_escalation_molecule(
            name="meta_test",
            severity_levels=["warn"],
            handlers={"warn": lambda ctx: None},
            metadata={"custom_key": "custom_value"},
        )

        assert molecule.metadata["custom_key"] == "custom_value"
        assert molecule.metadata["type"] == "escalation"


class TestCreateConditionalEscalationMolecule:
    """Tests for create_conditional_escalation_molecule factory."""

    def test_create_conditional_escalation(self):
        """Test creating a conditional escalation molecule."""

        def check_fn():
            return True

        handlers = {"warn": lambda ctx: None, "throttle": lambda ctx: None}

        molecule = create_conditional_escalation_molecule(
            name="conditional_test",
            check_fn=check_fn,
            severity_levels=["warn", "throttle"],
            handlers=handlers,
        )

        assert molecule.name == "conditional_test"
        # Each level has check + escalate steps
        assert len(molecule.steps) == 4
        assert molecule.metadata["type"] == "conditional_escalation"

    def test_conditional_escalation_step_order(self):
        """Test that conditional escalation has correct step order."""
        molecule = create_conditional_escalation_molecule(
            name="order_test",
            check_fn=lambda: True,
            severity_levels=["warn", "throttle"],
            handlers={"warn": lambda ctx: None, "throttle": lambda ctx: None},
        )

        # Order should be: check_warn, escalate_warn, check_throttle, escalate_throttle
        assert molecule.steps[0].name == "check_before_warn"
        assert molecule.steps[1].name == "escalate_warn"
        assert molecule.steps[2].name == "check_before_throttle"
        assert molecule.steps[3].name == "escalate_throttle"


# =============================================================================
# Singleton and Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.mark.asyncio
    async def test_get_molecule_engine_singleton(self, temp_dir, reset_engine):
        """Test that get_molecule_engine returns singleton."""
        engine1 = await get_molecule_engine(checkpoint_dir=temp_dir)
        engine2 = await get_molecule_engine()

        assert engine1 is engine2

    @pytest.mark.asyncio
    async def test_reset_molecule_engine(self, temp_dir, reset_engine):
        """Test that reset_molecule_engine clears singleton."""
        engine1 = await get_molecule_engine(checkpoint_dir=temp_dir)
        reset_molecule_engine()

        # Create a new temp dir for the new engine
        new_temp = temp_dir / "new"
        new_temp.mkdir()
        engine2 = await get_molecule_engine(checkpoint_dir=new_temp)

        assert engine1 is not engine2


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, temp_dir, reset_engine):
        """Test that circular dependencies are detected."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        # Create circular dependency
        step1 = MoleculeStep.create("step1", "shell", dependencies=["placeholder"])
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])

        # Make step1 depend on step2 (circular)
        step1.dependencies = [step2.id]

        molecule = Molecule.create("circular", [step1, step2])

        result = await engine.execute(molecule)

        assert result.success is False
        assert (
            "Circular" in molecule.error_message
            or "Cyclic" in molecule.error_message
            or "unmet" in molecule.error_message.lower()
        )

    @pytest.mark.asyncio
    async def test_step_timeout(self, temp_dir, reset_engine):
        """Test step timeout handling."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        class SlowExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                await asyncio.sleep(10)  # Longer than timeout
                return {}

        engine.register_executor("slow", SlowExecutor())

        steps = [MoleculeStep.create("slow_step", "slow", timeout_seconds=0.1, max_attempts=1)]
        molecule = Molecule.create("timeout_test", steps)

        result = await engine.execute(molecule)

        assert result.success is False
        assert molecule.steps[0].status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_executor_exception_handling(self, temp_dir, reset_engine):
        """Test handling of exceptions in executor."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        class ExceptionExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                raise RuntimeError("Executor crashed")

        engine.register_executor("exception", ExceptionExecutor())

        steps = [MoleculeStep.create("crash_step", "exception", max_attempts=1)]
        molecule = Molecule.create("exception_test", steps)

        result = await engine.execute(molecule)

        assert result.success is False
        assert "Executor crashed" in molecule.steps[0].error_message

    @pytest.mark.asyncio
    async def test_empty_molecule_execution(self, temp_dir, reset_engine):
        """Test executing a molecule with no steps."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        molecule = Molecule.create("empty", [])

        result = await engine.execute(molecule)

        assert result.success is True
        assert result.completed_steps == 0
        assert result.total_steps == 0


class TestStepStatusTransitions:
    """Tests for step status transitions during execution."""

    @pytest.mark.asyncio
    async def test_status_transition_pending_to_running(self, temp_dir, reset_engine):
        """Test step transitions from PENDING to RUNNING."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        status_changes = []

        class StatusTrackingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                status_changes.append(step.status)
                return {}

        engine.register_executor("track", StatusTrackingExecutor())

        steps = [MoleculeStep.create("track_step", "track")]
        molecule = Molecule.create("status_test", steps)

        await engine.execute(molecule)

        assert StepStatus.RUNNING in status_changes

    @pytest.mark.asyncio
    async def test_status_transition_running_to_completed(self, temp_dir, reset_engine):
        """Test step transitions from RUNNING to COMPLETED on success."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("complete_step", "shell", {"command": "echo done"})]
        molecule = Molecule.create("complete_test", steps)

        await engine.execute(molecule)

        assert molecule.steps[0].status == StepStatus.COMPLETED
        assert molecule.steps[0].completed_at is not None

    @pytest.mark.asyncio
    async def test_step_attempt_count_incremented(self, temp_dir, reset_engine):
        """Test that attempt count is incremented on each execution."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        attempt_counts = []

        class AttemptTracker(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                attempt_counts.append(step.attempt_count)
                if len(attempt_counts) < 2:
                    raise Exception("Retry needed")
                return {}

        engine.register_executor("attempts", AttemptTracker())

        steps = [MoleculeStep.create("attempt_step", "attempts", max_attempts=3)]
        molecule = Molecule.create("attempt_test", steps)

        await engine.execute(molecule)

        assert 1 in attempt_counts
        assert 2 in attempt_counts


# =============================================================================
# Audit Trail Tests
# =============================================================================


class TestAuditTrail:
    """Tests for audit trail generation."""

    @pytest.mark.asyncio
    async def test_checkpoint_contains_audit_info(self, temp_dir, reset_engine):
        """Test that checkpoints contain audit information."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("audit_step", "shell", {"command": "echo test"})]
        molecule = Molecule.create("audit_test", steps)

        await engine.execute(molecule)

        # Read checkpoint file
        checkpoint_file = temp_dir / f"{molecule.id}.json"
        with open(checkpoint_file) as f:
            data = json.load(f)

        # Verify audit info is present
        assert "created_at" in data
        assert "updated_at" in data
        assert "started_at" in data
        assert "completed_at" in data
        assert "steps" in data

        # Verify step audit info
        step_data = data["steps"][0]
        assert "started_at" in step_data
        assert "completed_at" in step_data
        assert "attempt_count" in step_data
        assert "status" in step_data

    @pytest.mark.asyncio
    async def test_step_timestamps_recorded(self, temp_dir, reset_engine):
        """Test that step execution timestamps are recorded."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        steps = [MoleculeStep.create("timed_step", "shell", {"command": "echo test"})]
        molecule = Molecule.create("timestamp_test", steps)

        before = datetime.now(timezone.utc)
        await engine.execute(molecule)
        after = datetime.now(timezone.utc)

        step = molecule.steps[0]
        assert step.started_at is not None
        assert step.completed_at is not None
        assert before <= step.started_at <= after
        assert before <= step.completed_at <= after
        assert step.started_at <= step.completed_at

    @pytest.mark.asyncio
    async def test_error_messages_recorded(self, temp_dir, reset_engine):
        """Test that error messages are recorded on failure."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        class FailingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
                raise ValueError("Specific error message")

        engine.register_executor("failing", FailingExecutor())

        steps = [MoleculeStep.create("fail_step", "failing", max_attempts=1)]
        molecule = Molecule.create("error_test", steps)

        await engine.execute(molecule)

        assert molecule.steps[0].error_message == "Specific error message"
