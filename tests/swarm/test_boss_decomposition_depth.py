from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.boss_loop import BossLoop, BossLoopConfig, GitHubIssue


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
                title="Child", description="desc", file_scope=[], estimated_complexity="low"
            )
        ],
    )

    with (
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
        patch.object(BossLoop, "_label_boss_stuck") as label_stuck,
    ):
        loop._auto_decompose_stuck_issue(issue.number, [issue])

    if should_decompose:
        decomposer.analyze.assert_called_once()
        label_stuck.assert_not_called()
    else:
        decomposer.analyze.assert_not_called()
        label_stuck.assert_called_once()
