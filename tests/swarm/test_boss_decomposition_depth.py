from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.boss_loop import BossLoop, BossLoopConfig, GitHubIssue


def _fake_gh_subprocess(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Return a subprocess.run side_effect that emulates gh CLI success."""

    def side_effect(args, **_kwargs):
        if isinstance(args, list):
            if "issue" in args and "list" in args:
                return SimpleNamespace(returncode=returncode, stdout="[]", stderr=stderr)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return side_effect


@pytest.mark.parametrize(
    ("depth", "should_decompose"), [(0, True), (1, True), (2, True), (3, False)]
)
def test_auto_decompose_stuck_issue_caps_depth(depth: int, should_decompose: bool) -> None:
    issue = GitHubIssue(
        number=7,
        title=((" ".join("[from #1]" for _ in range(depth)) + " Parent issue").strip()),
        body="Decompose me",
        labels=["boss-ready"],
        url="https://github.com/synaptent/aragora/issues/7",
        state="OPEN",
        created_at="2026-04-05T00:00:00Z",
    )
    loop = BossLoop(BossLoopConfig(repo="synaptent/aragora", max_retries_per_issue=2))
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(
        should_decompose=True,
        subtasks=[
            SimpleNamespace(
                title="Child task",
                description=(
                    "A sufficiently long sub-task description to pass the "
                    "minimum length gate in the decomposer."
                ),
                file_scope=["aragora/swarm/boss_loop.py"],
                estimated_complexity="low",
            )
        ],
    )

    with (
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("subprocess.run", side_effect=_fake_gh_subprocess()),
        patch.object(BossLoop, "_label_boss_stuck") as label_stuck,
    ):
        loop._auto_decompose_stuck_issue(issue.number, [issue])

    if should_decompose:
        decomposer.analyze.assert_called_once()
    else:
        decomposer.analyze.assert_not_called()
    # `_label_boss_stuck` is always called once at function exit (or the
    # early-return at the depth cap). Asserting call-count guards against
    # regressions that spam the parent with repeated stuck-labels.
    label_stuck.assert_called_once()


def test_auto_decompose_skips_when_issue_already_boss_stuck() -> None:
    """Idempotency guard: if the issue is already boss-stuck, do nothing.

    Without this guard, the boss loop can re-enter `_auto_decompose_stuck_issue`
    on the same issue (via stale feed caches or concurrent loops) and post
    duplicate "exhausted N attempts..." comments. See issue #5894 for the
    field observation (4 such comments in 13 minutes).
    """
    issue = GitHubIssue(
        number=5894,
        title="[TW-02] Already-stuck issue",
        body="Body text",
        labels=["autonomous", "boss-stuck"],
        url="https://github.com/synaptent/aragora/issues/5894",
        state="OPEN",
        created_at="2026-04-16T03:22:33Z",
    )
    loop = BossLoop(BossLoopConfig(repo="synaptent/aragora"))
    decomposer = MagicMock()

    with (
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("subprocess.run") as mock_subprocess,
        patch.object(BossLoop, "_label_boss_stuck") as label_stuck,
    ):
        loop._auto_decompose_stuck_issue(issue.number, [issue])

    decomposer.analyze.assert_not_called()
    label_stuck.assert_not_called()
    assert mock_subprocess.call_count == 0, (
        "No gh CLI calls should fire when the issue is already boss-stuck; "
        f"saw {mock_subprocess.call_count} calls."
    )


def test_auto_decompose_does_not_label_children_boss_ready() -> None:
    """Canonical queue policy: decomposer children must not inherit `boss-ready`.

    Per docs/status/NEXT_STEPS_CANONICAL.md, only CS-01..03 work carries
    `boss-ready` until Foreman reliability is proven. Historically the
    decomposer auto-applied `boss-ready` to every child it spawned, which
    leaked the queue.
    """
    issue = GitHubIssue(
        number=100,
        title="Parent task",
        body="Decompose me",
        labels=["boss-ready", "autonomous"],
        url="https://github.com/synaptent/aragora/issues/100",
        state="OPEN",
        created_at="2026-04-10T00:00:00Z",
    )
    loop = BossLoop(BossLoopConfig(repo="synaptent/aragora", max_retries_per_issue=2))
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(
        should_decompose=True,
        subtasks=[
            SimpleNamespace(
                title="Child task",
                description=(
                    "A sufficiently long sub-task description to pass the "
                    "minimum length gate in the decomposer."
                ),
                file_scope=["aragora/swarm/boss_loop.py"],
                estimated_complexity="low",
            )
        ],
    )

    with (
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("subprocess.run", side_effect=_fake_gh_subprocess()) as mock_run,
        patch.object(BossLoop, "_label_boss_stuck"),
    ):
        loop._auto_decompose_stuck_issue(issue.number, [issue])

    create_calls = [
        call
        for call in mock_run.call_args_list
        if call.args
        and isinstance(call.args[0], list)
        and "issue" in call.args[0]
        and "create" in call.args[0]
    ]
    assert len(create_calls) == 1, (
        f"expected exactly one `gh issue create` call, got {len(create_calls)}"
    )
    cmd = create_calls[0].args[0]
    assert "boss-ready" not in cmd, (
        f"decomposer children must not be auto-labeled boss-ready: {cmd}"
    )
