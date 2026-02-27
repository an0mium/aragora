"""Tests for InterrogationExecutor -- bridging Spec to HardenedOrchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.interrogation.executor import (
    InterrogationExecutor,
    ExecutionRequest,
    ExecutionResult,
)
from aragora.interrogation.crystallizer import Spec, Requirement, RequirementLevel


class TestInterrogationExecutor:
    @pytest.fixture
    def spec(self):
        return Spec(
            problem_statement="Fix performance",
            requirements=[
                Requirement(
                    description="Reduce latency",
                    level=RequirementLevel.MUST,
                    dimension="performance",
                ),
            ],
            success_criteria=["API response < 200ms"],
        )

    @pytest.mark.asyncio
    async def test_execute_creates_worktree_and_runs(self, spec):
        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_goal.return_value = MagicMock(
            success=True,
            completed_subtasks=1,
            failed_subtasks=0,
            total_subtasks=1,
            skipped_subtasks=0,
        )
        executor = InterrogationExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(ExecutionRequest(spec=spec, target="self"))
        assert isinstance(result, ExecutionResult)
        assert result.success
        mock_orchestrator.execute_goal.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_passes_goal_text(self, spec):
        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_goal.return_value = MagicMock(
            success=True,
            completed_subtasks=1,
            failed_subtasks=0,
            total_subtasks=1,
            skipped_subtasks=0,
        )
        executor = InterrogationExecutor(orchestrator=mock_orchestrator)
        await executor.execute(ExecutionRequest(spec=spec, target="self"))
        call_args = mock_orchestrator.execute_goal.call_args
        goal_arg = call_args.kwargs.get("goal", call_args.args[0] if call_args.args else "")
        assert "Reduce latency" in goal_arg

    @pytest.mark.asyncio
    async def test_execute_dry_run(self, spec):
        executor = InterrogationExecutor()
        result = await executor.execute(ExecutionRequest(spec=spec, target="self", dry_run=True))
        assert isinstance(result, ExecutionResult)
        assert result.dry_run
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_handles_orchestrator_failure(self, spec):
        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_goal.side_effect = RuntimeError("Orchestrator crashed")
        executor = InterrogationExecutor(orchestrator=mock_orchestrator)
        result = await executor.execute(ExecutionRequest(spec=spec, target="self"))
        assert not result.success
        assert result.error

    @pytest.mark.asyncio
    async def test_execution_result_includes_goal_text(self, spec):
        executor = InterrogationExecutor()
        result = await executor.execute(ExecutionRequest(spec=spec, target="self", dry_run=True))
        assert "Reduce latency" in result.goal_text
