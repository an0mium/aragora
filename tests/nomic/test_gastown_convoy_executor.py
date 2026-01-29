import pytest

from aragora.implement.types import ImplementTask, TaskResult
from aragora.nomic.convoy_executor import GastownConvoyExecutor, ReviewResult


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.asyncio
async def test_gastown_convoy_executor_executes_tasks(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    implementers = [DummyAgent("claude-implementer"), DummyAgent("codex-implementer")]
    reviewers = [DummyAgent("reviewer-a"), DummyAgent("reviewer-b")]

    executor = GastownConvoyExecutor(
        repo_path=repo_path,
        implementers=implementers,
        reviewers=reviewers,
        bead_dir=tmp_path / "beads",
    )

    tasks = [
        ImplementTask(
            id="task-1",
            description="Add new helper",
            files=["aragora/helpers/new_helper.py"],
            complexity="simple",
        ),
        ImplementTask(
            id="task-2",
            description="Wire helper into module",
            files=["aragora/module.py"],
            complexity="simple",
            dependencies=["task-1"],
        ),
    ]

    async def fake_execute(task: ImplementTask, agent):
        return TaskResult(
            task_id=task.id,
            success=True,
            diff=f"diff for {task.id}",
            model_used=getattr(agent, "name", "mock"),
            duration_seconds=0.01,
        )

    async def fake_review(task: ImplementTask, agent, diff: str):
        return ReviewResult(approved=True, notes="ok")

    # Patch internal execution/review to avoid LLM calls
    executor._execute_task_with_agent = fake_execute  # type: ignore[assignment]
    executor._review_task = fake_review  # type: ignore[assignment]

    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert completed == {"task-1", "task-2"}
