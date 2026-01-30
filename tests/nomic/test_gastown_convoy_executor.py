import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement.types import ImplementTask, TaskResult
from aragora.nomic.convoy_executor import GastownConvoyExecutor, ReviewResult


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name


def _make_executor(tmp_path, implementers=None, reviewers=None, **kwargs):
    """Helper to create a GastownConvoyExecutor with sensible defaults."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir(exist_ok=True)
    if implementers is None:
        implementers = [DummyAgent("claude-impl"), DummyAgent("codex-impl")]
    if reviewers is None:
        reviewers = [DummyAgent("reviewer-a"), DummyAgent("reviewer-b")]
    return GastownConvoyExecutor(
        repo_path=repo_path,
        implementers=implementers,
        reviewers=reviewers,
        bead_dir=tmp_path / "beads",
        **kwargs,
    )


def _patch_executor(executor, execute_fn=None, review_fn=None, test_fn=None):
    """Patch internal methods to avoid real LLM / subprocess calls."""
    if execute_fn is None:

        async def execute_fn(task, agent):
            return TaskResult(
                task_id=task.id,
                success=True,
                diff=f"diff for {task.id}",
                model_used=getattr(agent, "name", "mock"),
                duration_seconds=0.01,
            )

    if review_fn is None:

        async def review_fn(task, agent, diff):
            return ReviewResult(approved=True, notes="ok")

    executor._execute_task_with_agent = execute_fn  # type: ignore[assignment]
    executor._review_task = review_fn  # type: ignore[assignment]
    if test_fn is not None:
        executor._run_task_tests = test_fn  # type: ignore[assignment]


def _simple_task(task_id, description="Do something", files=None, deps=None, complexity="simple"):
    return ImplementTask(
        id=task_id,
        description=description,
        files=files or [f"aragora/{task_id}.py"],
        complexity=complexity,
        dependencies=deps or [],
    )


# ===========================================================================
# Basic execution tests
# ===========================================================================


@pytest.mark.asyncio
async def test_execute_plan_completes_all_tasks(tmp_path):
    """All tasks should execute and complete successfully."""
    executor = _make_executor(tmp_path)
    _patch_executor(executor)

    tasks = [
        _simple_task("task-1", "Add helper"),
        _simple_task("task-2", "Wire helper", deps=["task-1"]),
    ]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert completed == {"task-1", "task-2"}


@pytest.mark.asyncio
async def test_execute_plan_single_task(tmp_path):
    """A plan with a single task should work correctly."""
    executor = _make_executor(tmp_path)
    _patch_executor(executor)

    tasks = [_simple_task("only-task")]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 1
    assert results[0].success
    assert results[0].task_id == "only-task"
    assert completed == {"only-task"}


@pytest.mark.asyncio
async def test_execute_plan_skips_already_completed(tmp_path):
    """Tasks already in the completed set should not be re-executed."""
    executor = _make_executor(tmp_path)
    _patch_executor(executor)

    tasks = [
        _simple_task("task-a"),
        _simple_task("task-b", deps=["task-a"]),
    ]
    completed = {"task-a"}
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 1
    assert results[0].task_id == "task-b"
    assert completed == {"task-a", "task-b"}


# ===========================================================================
# Failure & rollback tests
# ===========================================================================


@pytest.mark.asyncio
async def test_task_failure_marks_result_failed(tmp_path):
    """When execution returns failure, the result should reflect that."""
    executor = _make_executor(tmp_path)

    async def failing_execute(task, agent):
        return TaskResult(
            task_id=task.id,
            success=False,
            error="compile_error",
            model_used="mock",
            duration_seconds=0.01,
        )

    _patch_executor(executor, execute_fn=failing_execute)

    tasks = [_simple_task("fail-task")]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 1
    assert not results[0].success
    assert "fail-task" not in completed


@pytest.mark.asyncio
async def test_review_rejection_marks_failure(tmp_path):
    """When a reviewer blocks, the task should be marked failed."""
    executor = _make_executor(tmp_path)

    async def rejecting_review(task, agent, diff):
        return ReviewResult(approved=False, notes="unsafe pattern detected")

    _patch_executor(executor, review_fn=rejecting_review)

    tasks = [_simple_task("reviewed-task")]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 1
    assert not results[0].success
    assert "reviewed-task" not in completed


@pytest.mark.asyncio
async def test_stop_on_failure_halts_plan(tmp_path):
    """With stop_on_failure=True, the plan stops after the first failure."""
    executor = _make_executor(tmp_path)

    async def counting_execute(task, agent):
        return TaskResult(
            task_id=task.id,
            success=False,
            error="error",
            model_used="mock",
            duration_seconds=0.01,
        )

    _patch_executor(executor, execute_fn=counting_execute)

    tasks = [
        _simple_task("t1"),
        _simple_task("t2"),
        _simple_task("t3"),
    ]
    completed = set()
    results = await executor.execute_plan(tasks, completed, stop_on_failure=True)

    assert len(results) == 1
    assert not results[0].success


@pytest.mark.asyncio
async def test_unmet_dependencies_all_fail(tmp_path):
    """Tasks whose dependencies never complete should be marked failed."""
    executor = _make_executor(tmp_path)

    async def failing_execute(task, agent):
        return TaskResult(
            task_id=task.id,
            success=False,
            error="fail",
            model_used="mock",
            duration_seconds=0.01,
        )

    _patch_executor(executor, execute_fn=failing_execute)

    tasks = [
        _simple_task("parent"),
        _simple_task("child", deps=["parent"]),
    ]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 2
    child_result = [r for r in results if r.task_id == "child"][0]
    assert not child_result.success
    assert "dependencies_unmet" in (child_result.error or "")


# ===========================================================================
# No implementers
# ===========================================================================


@pytest.mark.asyncio
async def test_no_implementers_raises(tmp_path):
    """execute_plan should raise RuntimeError when no implementers are available."""
    executor = _make_executor(tmp_path, implementers=[])
    _patch_executor(executor)

    with pytest.raises(RuntimeError, match="No implementer agents"):
        await executor.execute_plan([_simple_task("x")], set())


# ===========================================================================
# Progress / callback tests
# ===========================================================================


@pytest.mark.asyncio
async def test_on_task_complete_callback(tmp_path):
    """The on_task_complete callback should be called for each successful task."""
    executor = _make_executor(tmp_path)
    _patch_executor(executor)

    completed_ids = []

    def on_complete(task_id, result):
        completed_ids.append(task_id)

    tasks = [_simple_task("a"), _simple_task("b")]
    await executor.execute_plan(tasks, set(), on_task_complete=on_complete)

    assert "a" in completed_ids
    assert "b" in completed_ids


@pytest.mark.asyncio
async def test_on_task_complete_not_called_on_failure(tmp_path):
    """The callback should NOT be called for failed tasks."""
    executor = _make_executor(tmp_path)

    async def fail_exec(task, agent):
        return TaskResult(
            task_id=task.id,
            success=False,
            error="err",
            model_used="m",
            duration_seconds=0.01,
        )

    _patch_executor(executor, execute_fn=fail_exec)

    completed_ids = []

    def on_complete(task_id, result):
        completed_ids.append(task_id)

    tasks = [_simple_task("fail")]
    await executor.execute_plan(tasks, set(), on_task_complete=on_complete)

    assert completed_ids == []


# ===========================================================================
# Agent selection
# ===========================================================================


def test_select_agent_for_assignment_known(tmp_path):
    """_select_agent_for_assignment should return the matching agent by name."""
    executor = _make_executor(tmp_path)
    agent = executor._select_agent_for_assignment("codex-impl")
    assert agent.name == "codex-impl"


def test_select_agent_for_assignment_unknown_falls_back(tmp_path):
    """Unknown agent_id should fall back to the first implementer."""
    executor = _make_executor(tmp_path)
    agent = executor._select_agent_for_assignment("nonexistent-agent")
    assert agent.name == executor.implementers[0].name


def test_select_agent_for_assignment_none_falls_back(tmp_path):
    """None agent_id should fall back to the first implementer."""
    executor = _make_executor(tmp_path)
    agent = executor._select_agent_for_assignment(None)
    assert agent.name == executor.implementers[0].name


# ===========================================================================
# Test file detection (_is_test_file)
# ===========================================================================


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/test_foo.py", True),
        ("tests/nomic/test_bar.py", True),
        ("aragora/tests/test_baz.py", True),
        ("src/test_utils.py", True),
        ("src/utils_test.py", True),
        ("aragora/module.py", False),
        ("aragora/test_helpers/support.py", False),
        ("scripts/run.py", False),
    ],
)
def test_is_test_file(path, expected):
    """_is_test_file should correctly identify test files."""
    assert GastownConvoyExecutor._is_test_file(path) is expected


# ===========================================================================
# Codex agent detection (_is_codex_agent)
# ===========================================================================


def test_is_codex_agent_by_name():
    """Agents with 'codex' in name should be detected."""
    agent = DummyAgent("codex-impl")
    assert GastownConvoyExecutor._is_codex_agent(agent) is True


def test_is_codex_agent_negative():
    """Agents without 'codex' in name/class should not be detected."""
    agent = DummyAgent("claude-impl")
    assert GastownConvoyExecutor._is_codex_agent(agent) is False


def test_is_codex_agent_wrapped():
    """Wrapped agents (AirlockProxy-style) should be detected via wrapped_agent."""
    inner = DummyAgent("codex-fallback")
    wrapper = MagicMock()
    wrapper.wrapped_agent = inner
    wrapper.__class__.__name__ = "AirlockProxy"
    assert GastownConvoyExecutor._is_codex_agent(wrapper) is True


# ===========================================================================
# Test running (_run_task_tests)
# ===========================================================================


@pytest.mark.asyncio
async def test_run_task_tests_disabled(tmp_path):
    """When tests are disabled, _run_task_tests returns True immediately."""
    executor = _make_executor(tmp_path, enable_tests=False)
    ok, notes = await executor._run_task_tests(_simple_task("t"))
    assert ok is True
    assert notes == "tests_disabled"


@pytest.mark.asyncio
async def test_run_task_tests_no_test_files(tmp_path):
    """When no test files and no test command, return no_tests_selected."""
    executor = _make_executor(tmp_path, enable_tests=True)
    task = _simple_task("t", files=["aragora/module.py"])
    ok, notes = await executor._run_task_tests(task)
    assert ok is True
    assert notes == "no_tests_selected"


# ===========================================================================
# _get_test_command
# ===========================================================================


def test_get_test_command_with_custom_command(tmp_path):
    """When test_command is set, it should be returned as-is."""
    executor = _make_executor(tmp_path, test_command="make test")
    cmd, use_shell = executor._get_test_command(_simple_task("t"))
    assert cmd == "make test"
    assert use_shell is True


def test_get_test_command_with_test_files(tmp_path):
    """When task has test files, pytest command should be built."""
    executor = _make_executor(tmp_path)
    task = _simple_task("t", files=["tests/test_foo.py", "aragora/core.py"])
    cmd, use_shell = executor._get_test_command(task)
    assert cmd == ["pytest", "tests/test_foo.py"]
    assert use_shell is False


def test_get_test_command_no_test_files(tmp_path):
    """When task has no test files, returns None."""
    executor = _make_executor(tmp_path)
    task = _simple_task("t", files=["aragora/core.py"])
    cmd, use_shell = executor._get_test_command(task)
    assert cmd is None
    assert use_shell is False


# ===========================================================================
# Initialization / configuration
# ===========================================================================


def test_init_filters_none_agents(tmp_path):
    """None agents in the implementers/reviewers list should be filtered out."""
    executor = _make_executor(
        tmp_path,
        implementers=[DummyAgent("a"), None, DummyAgent("b")],
        reviewers=[None, DummyAgent("r")],
    )
    assert len(executor.implementers) == 2
    assert len(executor.reviewers) == 1


def test_init_defaults_from_env(tmp_path, monkeypatch):
    """Config defaults should come from environment variables when not explicit."""
    monkeypatch.setenv("NOMIC_CONVOY_PARALLEL_TASKS", "1")
    monkeypatch.setenv("NOMIC_CONVOY_MAX_PARALLEL", "4")
    monkeypatch.setenv("NOMIC_CONVOY_TESTS", "1")
    monkeypatch.setenv("NOMIC_CONVOY_TEST_TIMEOUT", "300")

    executor = _make_executor(tmp_path)
    assert executor._allow_parallel is True
    assert executor._max_parallel == 4
    assert executor._enable_tests is True
    assert executor._test_timeout == 300


def test_init_explicit_overrides_env(tmp_path, monkeypatch):
    """Explicit constructor args should override env vars."""
    monkeypatch.setenv("NOMIC_CONVOY_PARALLEL_TASKS", "1")
    executor = _make_executor(tmp_path, allow_parallel=False, max_parallel=1)
    assert executor._allow_parallel is False
    assert executor._max_parallel == 1


# ===========================================================================
# Review logic
# ===========================================================================


@pytest.mark.asyncio
async def test_review_task_empty_diff_auto_approves(tmp_path):
    """Empty diff should auto-approve without consulting reviewers."""
    executor = _make_executor(tmp_path)
    result = await executor._review_task(_simple_task("t"), DummyAgent("impl"), "   ")
    assert result.approved is True
    assert result.notes == "no_diff"


# ===========================================================================
# Test failure during test run marks task as failed
# ===========================================================================


@pytest.mark.asyncio
async def test_task_fails_when_tests_fail(tmp_path):
    """If tests fail after successful execution + review, task should be marked failed."""
    executor = _make_executor(tmp_path)

    async def good_execute(task, agent):
        return TaskResult(
            task_id=task.id,
            success=True,
            diff="some diff",
            model_used="mock",
            duration_seconds=0.01,
        )

    async def good_review(task, agent, diff):
        return ReviewResult(approved=True, notes="ok")

    async def failing_tests(task):
        return False, "assertion error in test_foo"

    _patch_executor(executor, execute_fn=good_execute, review_fn=good_review, test_fn=failing_tests)

    tasks = [_simple_task("tested-task")]
    completed = set()
    results = await executor.execute_plan(tasks, completed)

    assert len(results) == 1
    assert not results[0].success
    assert "tests_failed" in (results[0].error or "")
    assert "tested-task" not in completed
