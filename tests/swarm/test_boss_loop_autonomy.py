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


def _issue(number: int = 101, title: str = "Improve routed execution") -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=title,
        body="## Acceptance Criteria\n- [ ] It works\n\n## Validation\npytest -q tests/swarm/",
        labels=["boss-ready"],
        url=f"https://github.com/synaptent/aragora/issues/{number}",
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

    assert status.worker_status == "needs_human"
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


def test_auto_publish_promotes_branch_deliverable_to_pr_metadata() -> None:
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            auto_publish_deliverables=True,
            target_branch="release/2026.04",
        )
    )
    worker_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "receipt_id": "lane-101",
        "receipt_metadata": {"requested_target_agent": "claude"},
        "deliverable": {
            "type": "branch",
            "branch": "codex/issue-101",
            "commit_shas": ["abc123"],
        },
    }

    with (
        patch(
            "aragora.swarm.tranche_integrate.publish_lane_deliverable",
            return_value={
                "published": True,
                "branch": "codex/issue-101",
                "pr_url": "https://github.com/synaptent/aragora/pull/1919",
            },
        ) as publish_mock,
        patch("aragora.ralph.github_control.GitHubControl") as github_control_cls,
        patch("aragora.swarm.pr_registry.PullRequestRegistry"),
    ):
        github_control_cls.return_value.upsert_issue_comment.return_value = {
            "commented": True,
            "action": "created",
            "comment_url": "https://github.com/synaptent/aragora/issues/101#issuecomment-1",
        }
        result = loop._postprocess_issue_result(_issue(), worker_result)

    assert result["status"] == "completed"
    assert result["outcome"] == "pr_adopted"
    assert result["deliverable"]["type"] == "pr"
    assert result["deliverable"]["pr_url"] == "https://github.com/synaptent/aragora/pull/1919"
    assert result["pr_number"] == 1919
    assert result["publish_result"]["published"] is True
    assert result["receipt_metadata"]["publish_result"]["pr_url"] == (
        "https://github.com/synaptent/aragora/pull/1919"
    )
    assert result["issue_comment_result"]["action"] == "created"
    assert result["receipt_metadata"]["issue_comment_result"]["commented"] is True
    assert result["receipt_metadata"]["postprocess_promoted_from_status"] == "needs_human"
    assert result["receipt_metadata"]["postprocess_promoted_from_outcome"] == "blocked"
    assert publish_mock.call_args.kwargs["target_branch"] == "release/2026.04"
    assert publish_mock.call_args.args[0].branch == "codex/issue-101"
    github_control_cls.return_value.upsert_issue_comment.assert_called_once()


def test_auto_close_marks_already_done_issue_resolved() -> None:
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            auto_close_already_done_issues=True,
            repo="synaptent/aragora",
        )
    )
    worker_result = {
        "status": "needs_human",
        "run": {
            "work_orders": [
                {
                    "worker_outcome": "clean_exit_no_effect",
                    "stdout_tail": "Already implemented; nothing to commit.",
                    "verification_results": [{"passed": True}],
                    "tests_run": ["pytest -q tests/swarm/test_boss_loop_autonomy.py"],
                    "commit_shas": [],
                    "changed_paths": [],
                }
            ]
        },
    }

    with patch(
        "aragora.swarm.boss_loop.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
    ) as close_mock:
        result = loop._postprocess_issue_result(_issue(), worker_result)

    assert result["outcome"] == "issue_already_resolved"
    assert result["issue_resolution"]["action"] == "closed"
    assert result["issue_resolution"]["reason"] == "already_implemented"
    assert "Verification passed on 1 check" in result["issue_resolution"]["comment"]
    close_cmd = close_mock.call_args.args[0]
    assert close_cmd[:3] == ["gh", "issue", "close"]
    assert "--repo" in close_cmd
    assert "synaptent/aragora" in close_cmd


def test_run_iteration_treats_auto_closed_issue_as_completed() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [_issue()]
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: _fresh_result(),
    )
    loop._emit_lane_receipt = MagicMock(return_value="lane-receipt-1")
    loop._log_value_outcome = MagicMock()

    with patch.object(
        loop,
        "_dispatch_issue",
        AsyncMock(
            return_value={
                "status": "needs_human",
                "outcome": "issue_already_resolved",
                "issue_resolution": {
                    "action": "closed",
                    "reason": "already_implemented",
                },
            }
        ),
    ):
        status = asyncio.run(loop._run_iteration(1))

    assert status.worker_status == "completed"
    assert status.stop_reason is None
    assert "auto-closed" in status.next_actions[0]
    assert loop._completed_issues[0]["number"] == 101


def test_run_iteration_surfaces_published_pr_after_blocked_deliverable() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [_issue()]
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: _fresh_result(),
    )
    loop._emit_lane_receipt = MagicMock(return_value="lane-receipt-1")
    loop._log_value_outcome = MagicMock()

    with patch.object(
        loop,
        "_dispatch_issue",
        AsyncMock(
            return_value={
                "status": "completed",
                "outcome": "pr_adopted",
                "pr_url": "https://github.com/synaptent/aragora/pull/1919",
                "deliverable": {
                    "type": "pr",
                    "pr_url": "https://github.com/synaptent/aragora/pull/1919",
                },
                "publish_result": {
                    "published": True,
                    "action": "pr_created",
                    "pr_url": "https://github.com/synaptent/aragora/pull/1919",
                },
            }
        ),
    ):
        status = asyncio.run(loop._run_iteration(1))

    assert status.worker_status == "completed"
    assert status.stop_reason is None
    assert "published PR https://github.com/synaptent/aragora/pull/1919" in status.next_actions[0]


def test_run_iteration_passes_profile_pool_and_rotation_to_freshness_checker() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [_issue()]
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
            verified_runner_target=0,
            runner_probe_limit=3,
        ),
        issue_feed=feed,
        freshness_checker=_freshness_checker,
    )

    with patch.object(loop, "_dispatch_issue", AsyncMock(return_value={"status": "completed"})):
        asyncio.run(loop._run_iteration(1))

    assert captured["requested_runner_type"] == "claude"
    assert captured["allowed_profiles"] == {"max-02", "max-03"}
    assert captured["rotation_interval_seconds"] == 900.0
    assert captured["verified_runner_target"] == 0
    assert captured["runner_probe_limit"] == 3


def test_boss_loop_can_dispatch_multiple_issues_in_one_iteration() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [
        _issue(101, "First issue"),
        _issue(102, "Second issue"),
    ]
    freshness = RunnerFreshnessResult(
        fresh=True,
        runner_ids=["claude-runner-1", "claude-runner-2"],
        checked_at=datetime.now(UTC).isoformat(),
        details={
            "routing": {
                "selected_runner_ids": ["claude-runner-1", "claude-runner-2"],
                "selected_runners": [
                    {
                        "runner_id": "claude-runner-1",
                        "runner_type": "claude",
                        "cost_class": "subscription",
                        "available_capacity": 1,
                    },
                    {
                        "runner_id": "claude-runner-2",
                        "runner_type": "claude",
                        "cost_class": "subscription",
                        "available_capacity": 1,
                    },
                ],
                "fallback_reason": None,
            }
        },
    )
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            default_target_agent="claude",
            default_reviewer_agent="codex",
            max_parallel_dispatches=2,
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: freshness,
    )

    with patch.object(
        loop,
        "_dispatch_issue",
        AsyncMock(side_effect=[{"status": "completed"}, {"status": "completed"}]),
    ):
        result = asyncio.run(loop.run())

    assert len(result.issues_attempted) == 2
    assert len(result.issues_completed) == 2
    assert len(result.iteration_statuses) == 2
    assert sorted(status["selected_issue"]["number"] for status in result.iteration_statuses) == [
        101,
        102,
    ]


def test_boss_loop_refills_parallel_capacity_with_next_issue() -> None:
    feed = MagicMock()
    feed.fetch.return_value = [
        _issue(101, "First issue"),
        _issue(102, "Second issue"),
        _issue(103, "Third issue"),
    ]
    freshness = RunnerFreshnessResult(
        fresh=True,
        runner_ids=["claude-runner-1", "claude-runner-2"],
        checked_at=datetime.now(UTC).isoformat(),
        details={
            "routing": {
                "selected_runner_ids": ["claude-runner-1", "claude-runner-2"],
                "selected_runners": [
                    {
                        "runner_id": "claude-runner-1",
                        "runner_type": "claude",
                        "cost_class": "subscription",
                        "available_capacity": 1,
                    },
                    {
                        "runner_id": "claude-runner-2",
                        "runner_type": "claude",
                        "cost_class": "subscription",
                        "available_capacity": 1,
                    },
                ],
                "fallback_reason": None,
            }
        },
    )
    loop = BossLoop(
        config=BossLoopConfig(
            max_iterations=1,
            iteration_interval_seconds=0.0,
            default_target_agent="claude",
            default_reviewer_agent="codex",
            max_parallel_dispatches=2,
        ),
        issue_feed=feed,
        freshness_checker=lambda **_kwargs: freshness,
    )
    started_numbers: list[int] = []

    async def _dispatch(issue, _freshness):
        started_numbers.append(issue.number)
        if issue.number == 101:
            await asyncio.sleep(0.02)
        elif issue.number == 102:
            await asyncio.sleep(0.01)
        return {"status": "completed"}

    with patch.object(loop, "_dispatch_issue", AsyncMock(side_effect=_dispatch)):
        result = asyncio.run(loop.run())

    assert started_numbers[:2] == [101, 102]
    assert 103 in started_numbers
    assert len(result.issues_attempted) == 3
    assert len(result.issues_completed) == 3
