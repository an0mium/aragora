"""Tests for fabric implementation integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.fabric import AgentFabric
from aragora.implement.fabric_integration import (
    FabricImplementationConfig,
    FabricImplementationRunner,
)
from aragora.implement.types import ImplementTask, TaskResult


@pytest.mark.asyncio
async def test_fabric_runner_executes_tasks(monkeypatch):
    tasks = [
        ImplementTask(
            id="task-1",
            description="Update README",
            files=["README.md"],
            complexity="simple",
        ),
        ImplementTask(
            id="task-2",
            description="Add helper",
            files=["aragora/helpers.py"],
            complexity="moderate",
        ),
    ]

    async def fake_execute(task, agent_handle, **_kwargs):  # noqa: ANN001
        return TaskResult(task_id=task.id, success=True)

    async with AgentFabric() as fabric:
        runner = FabricImplementationRunner(
            fabric,
            repo_path=Path.cwd(),
            implementation_profile=None,
        )
        monkeypatch.setattr(runner, "_execute_task", fake_execute)

        completed: list[str] = []

        def on_complete(task_id, result):  # noqa: ANN001
            completed.append(task_id)

        results = await runner.run_plan(
            tasks,
            config=FabricImplementationConfig(models=["claude"], min_agents=1),
            on_task_complete=on_complete,
        )

    assert len(results) == 2
    assert all(result.success for result in results)
    assert set(completed) == {"task-1", "task-2"}
