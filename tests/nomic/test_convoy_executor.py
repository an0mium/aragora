"""Tests for ConvoyImplementExecutor including execute_plan bridge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.nomic.implement_executor import (
    BeadTaskResult,
    ConvoyImplementExecutor,
    ConvoyResult,
    ImplementationBead,
    _generate_bead_id,
)


@pytest.fixture
def executor(tmp_path):
    return ConvoyImplementExecutor(
        aragora_path=tmp_path,
        agents=["agent-a", "agent-b", "agent-c"],
        max_parallel=2,
    )


class TestBeadTaskResult:
    """Tests for BeadTaskResult dataclass."""

    def test_success_result(self):
        r = BeadTaskResult(success=True, task_id="t1", files_modified=["a.py"])
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = BeadTaskResult(success=False, error="boom", task_id="t2")
        assert r.success is False
        assert r.error == "boom"


class TestDecomposeDesign:
    """Tests for _decompose_design parsing."""

    def test_parses_file_prefixes(self, executor):
        design = (
            "File: aragora/foo.py\n"
            "  Add function bar()\n"
            "\n"
            "File: aragora/baz.py\n"
            "  Update class Qux\n"
        )
        beads = executor._decompose_design(design, "test improvement", set())
        assert len(beads) == 2
        assert beads[0].file_path == "aragora/foo.py"
        assert beads[1].file_path == "aragora/baz.py"

    def test_parses_create_prefix(self, executor):
        design = "Create: aragora/new_module.py\n  New feature code\n"
        beads = executor._decompose_design(design, "test", set())
        assert len(beads) == 1
        assert beads[0].change_type == "create"

    def test_parses_modify_prefix(self, executor):
        design = "Modify: aragora/existing.py\n  Update handler\n"
        beads = executor._decompose_design(design, "test", set())
        assert len(beads) == 1
        assert beads[0].change_type == "modify"

    def test_protected_files_skipped(self, executor):
        design = "File: CLAUDE.md\n  Modify something\n\nFile: aragora/foo.py\n  Add feature\n"
        beads = executor._decompose_design(design, "test", {"CLAUDE.md"})
        assert len(beads) == 1
        assert beads[0].file_path == "aragora/foo.py"

    def test_single_bead_fallback(self, executor):
        design = "Just do something with the codebase."
        beads = executor._decompose_design(design, "improve stuff", set())
        assert len(beads) == 1
        assert beads[0].file_path == "<unstructured>"


class TestAssignAgents:
    """Tests for round-robin agent assignment."""

    def test_round_robin_assignment(self, executor):
        beads = [
            ImplementationBead(
                bead_id=f"b{i}",
                title=f"bead {i}",
                description="",
                file_path=f"f{i}.py",
                change_type="modify",
            )
            for i in range(5)
        ]
        executor._assign_agents(beads)
        assert beads[0].assigned_agent == "agent-a"
        assert beads[1].assigned_agent == "agent-b"
        assert beads[2].assigned_agent == "agent-c"
        assert beads[3].assigned_agent == "agent-a"  # wraps around
        assert beads[4].assigned_agent == "agent-b"

    def test_reviewer_different_from_implementer(self, executor):
        beads = [
            ImplementationBead(
                bead_id="b0",
                title="test",
                description="",
                file_path="f.py",
                change_type="modify",
            ),
        ]
        executor._assign_agents(beads)
        assert beads[0].assigned_agent != beads[0].reviewer_agent

    def test_no_agents_no_assignment(self, tmp_path):
        executor = ConvoyImplementExecutor(
            aragora_path=tmp_path,
            agents=[],
            max_parallel=1,
        )
        beads = [
            ImplementationBead(
                bead_id="b0",
                title="test",
                description="",
                file_path="f.py",
                change_type="modify",
            ),
        ]
        executor._assign_agents(beads)
        assert beads[0].assigned_agent is None


class TestExecutePlanBridge:
    """Tests for the execute_plan() bridge method."""

    @pytest.mark.asyncio
    async def test_execute_plan_basic(self, executor):
        task = MagicMock()
        task.id = "task-1"
        task.description = "Add helper function"
        task.files = ["src/helpers.py"]
        task.dependencies = []

        results = await executor.execute_plan([task], set())
        assert len(results) == 1
        assert isinstance(results[0], BeadTaskResult)
        assert results[0].task_id == "task-1"
        # Without agent_factory, beads are marked "done" (external execution mode)
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_plan_filters_completed(self, executor):
        task1 = MagicMock()
        task1.id = "task-1"
        task1.description = "Already done"
        task1.files = ["a.py"]
        task1.dependencies = []

        task2 = MagicMock()
        task2.id = "task-2"
        task2.description = "New work"
        task2.files = ["b.py"]
        task2.dependencies = []

        results = await executor.execute_plan([task1, task2], {"task-1"})
        assert len(results) == 1
        assert results[0].task_id == "task-2"

    @pytest.mark.asyncio
    async def test_execute_plan_all_completed(self, executor):
        task = MagicMock()
        task.id = "task-1"
        task.description = "Done"
        task.files = ["a.py"]

        results = await executor.execute_plan([task], {"task-1"})
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_plan_calls_on_task_complete(self, executor):
        task = MagicMock()
        task.id = "task-1"
        task.description = "Work"
        task.files = ["a.py"]

        callback = MagicMock()
        results = await executor.execute_plan([task], set(), on_task_complete=callback)
        assert callback.call_count == 1
        call_args = callback.call_args
        assert call_args[0][0] == "task-1"
        assert isinstance(call_args[0][1], BeadTaskResult)

    @pytest.mark.asyncio
    async def test_execute_plan_with_agent_factory(self, tmp_path):
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="implemented code")

        executor = ConvoyImplementExecutor(
            aragora_path=tmp_path,
            agents=["agent-a"],
            agent_factory=lambda name: mock_agent,
            max_parallel=1,
            enable_cross_check=False,
        )

        task = MagicMock()
        task.id = "task-1"
        task.description = "Implement feature"
        task.files = ["src/feature.py"]

        results = await executor.execute_plan([task], set())
        assert len(results) == 1
        assert results[0].success is True
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_plan_with_cross_check(self, tmp_path):
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="APPROVE - looks good")

        executor = ConvoyImplementExecutor(
            aragora_path=tmp_path,
            agents=["agent-a", "agent-b"],
            agent_factory=lambda name: mock_agent,
            max_parallel=2,
            enable_cross_check=True,
        )

        task = MagicMock()
        task.id = "task-1"
        task.description = "Implement feature"
        task.files = ["src/feature.py"]

        results = await executor.execute_plan([task], set())
        assert len(results) == 1
        assert results[0].success is True
        # Agent called for implementation + cross-check review
        assert mock_agent.generate.call_count == 2


class TestExecute:
    """Tests for the main execute() method."""

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        result = await executor.execute(
            design="File: aragora/new.py\n  Add function foo()\n",
            improvement="Add foo feature",
        )
        assert isinstance(result, ConvoyResult)
        assert result.beads_completed >= 1

    @pytest.mark.asyncio
    async def test_execute_empty_design(self, executor):
        result = await executor.execute(design="", improvement="nothing")
        # Empty design still creates a single unstructured bead
        assert isinstance(result, ConvoyResult)


class TestGenerateBeadId:
    """Tests for bead ID generation."""

    def test_generates_unique_ids(self):
        id1 = _generate_bead_id("task 1")
        id2 = _generate_bead_id("task 2")
        assert id1.startswith("impl-")
        assert id2.startswith("impl-")
        assert len(id1) == 11  # "impl-" + 6 hex chars
