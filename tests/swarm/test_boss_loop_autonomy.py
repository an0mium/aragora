from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.swarm.boss_loop import (
    BossLoop,
    BossLoopConfig,
    GitHubIssue,
    GitHubIssueFeed,
    RunnerFreshnessResult,
)

UTC = timezone.utc


def _fresh_result() -> RunnerFreshnessResult:
    return RunnerFreshnessResult(
        fresh=True,
        runner_ids=["claude-runner-1"],
        checked_at=datetime.now(UTC).isoformat(),
        details={
            "routing": {
                "selected_runner_ids": ["claude-runner-1"],
                "selected_runners": [
                    {
                        "runner_id": "claude-runner-1",
                        "runner_type": "claude",
                        "cost_class": "subscription",
                    }
                ],
                "fallback_reason": None,
            }
        },
    )


def _issue() -> GitHubIssue:
    return GitHubIssue(
        number=101,
        title="Improve routed execution",
        body="## Acceptance Criteria\n- [ ] It works\n\n## Validation\npytest -q tests/swarm/",
        labels=["boss-ready"],
        url="https://github.com/synaptent/aragora/issues/101",
        state="open",
        created_at="2026-03-27T00:00:00Z",
    )


def test_issue_feed_supports_explicit_issue_numbers() -> None:
    responses = {
        101: {
            "number": 101,
            "title": "Issue 101",
            "body": "body",
            "labels": [{"name": "boss-ready"}],
            "url": "https://github.com/synaptent/aragora/issues/101",
            "state": "OPEN",
            "createdAt": "2026-03-27T00:00:00Z",
        },
        102: {
            "number": 102,
            "title": "Issue 102",
            "body": "body",
            "labels": [{"name": "boss-ready"}],
            "url": "https://github.com/synaptent/aragora/issues/102",
            "state": "OPEN",
            "createdAt": "2026-03-27T00:00:00Z",
        },
    }

    def _run(cmd, **_kwargs):
        number = int(cmd[3])
        return SimpleNamespace(returncode=0, stdout=json.dumps(responses[number]), stderr="")

    with patch("subprocess.run", side_effect=_run):
        issues = GitHubIssueFeed(
            repo="synaptent/aragora",
            label_filter="boss-ready",
            issue_numbers=[101, 102],
        ).fetch()

    assert [issue.number for issue in issues] == [101, 102]
    assert [issue.title for issue in issues] == ["Issue 101", "Issue 102"]


def test_full_auto_continues_when_needs_human_has_deliverable() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [_issue()]
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            auto_continue_on_needs_human=True,
            default_target_agent="claude",
            default_reviewer_agent="codex",
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: _fresh_result(),
    )

    with patch.object(
        loop,
        "_dispatch_issue",
        AsyncMock(
            return_value={
                "status": "needs_human",
                "deliverable": {"work_order_id": "wo-1"},
                "run": {
                    "work_orders": [
                        {
                            "work_order_id": "wo-1",
                            "target_agent": "claude",
                            "reviewer_agent": "codex",
                        }
                    ]
                },
                "receipt_metadata": {"requested_target_agent": "claude"},
            }
        ),
    ):
        status = asyncio.run(loop._run_iteration(1))

    assert status.worker_status == "completed"
    assert status.stop_reason is None
    assert "Auto-continuing" in status.next_actions[0]


def test_full_auto_skips_needs_human_without_deliverable() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [_issue()]
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            auto_continue_on_needs_human=True,
            default_target_agent="claude",
            default_reviewer_agent="codex",
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: _fresh_result(),
    )

    with patch.object(
        loop,
        "_dispatch_issue",
        AsyncMock(return_value={"status": "needs_human", "reasons": ["blocked"]}),
    ):
        status = asyncio.run(loop._run_iteration(1))

    assert status.worker_status == "needs_human"
    assert status.stop_reason is None
    assert "Skipping to next issue" in status.next_actions[0]


def test_run_iteration_passes_profile_pool_and_rotation_to_freshness_checker() -> None:
    feed = MagicMock()
    feed.fetch.return_value = []
    captured: dict[str, object] = {}

    def _freshness_checker(**kwargs):
        captured.update(kwargs)
        return _fresh_result()

    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            default_target_agent="claude",
            allowed_runner_profiles={"max-02", "max-03"},
            runner_rotation_interval_seconds=900.0,
        ),
        issue_feed=feed,
        freshness_checker=_freshness_checker,
    )

    asyncio.run(loop._run_iteration(1))

    assert captured["requested_runner_type"] == "claude"
    assert captured["allowed_profiles"] == {"max-02", "max-03"}
    assert captured["rotation_interval_seconds"] == 900.0
