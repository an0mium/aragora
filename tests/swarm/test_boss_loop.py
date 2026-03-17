"""Tests for the long-running Boss loop MVP.

Covers:
- No fresh runner -> blocked
- No suitable issue -> blocked
- Bounded retry / iteration behavior
- Status payload shape
- Resumable / terminal stop-state reporting
- GitHub issue feed parsing
- Runner freshness checks
- CLI boss-loop action
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.boss_loop import (
    BossIterationStatus,
    BossLoop,
    BossLoopConfig,
    BossLoopResult,
    BossStopReason,
    GitHubIssue,
    GitHubIssueFeed,
    RunnerFreshnessResult,
    check_runner_freshness,
    dispatch_bounded_spec,
    select_eligible_issue,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_issue(
    number: int = 1,
    title: str = "Fix the dashboard",
    body: str = "The dashboard is slow, improve query performance in aragora/analytics/dashboard.py",
    labels: list[str] | None = None,
    state: str = "OPEN",
) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=title,
        body=body,
        labels=labels or [],
        url=f"https://github.com/synaptent/aragora/issues/{number}",
        state=state,
        created_at="2026-03-07T00:00:00Z",
    )


def _fresh_result(
    fresh: bool = True,
    runner_ids: list[str] | None = None,
    blocked_reason: str | None = None,
) -> RunnerFreshnessResult:
    return RunnerFreshnessResult(
        fresh=fresh,
        runner_ids=runner_ids or (["codex-runner-1"] if fresh else []),
        checked_at=datetime.now(UTC).isoformat(),
        blocked_reason=blocked_reason,
    )


def _boss_config(**overrides: Any) -> BossLoopConfig:
    defaults = {
        "max_iterations": 3,
        "iteration_interval_seconds": 0.0,
        "freshness_ttl_seconds": 3600.0,
        "max_consecutive_failures": 2,
        "max_retries_per_issue": 2,
    }
    defaults.update(overrides)
    return BossLoopConfig(**defaults)


# ---------------------------------------------------------------------------
# GitHubIssueFeed tests
# ---------------------------------------------------------------------------


class TestGitHubIssueFeed:
    def test_fetch_parses_gh_json_output(self, monkeypatch):
        gh_output = json.dumps(
            [
                {
                    "number": 42,
                    "title": "Improve error handling",
                    "body": "Add retry logic to API calls",
                    "labels": [{"name": "enhancement"}],
                    "url": "https://github.com/synaptent/aragora/issues/42",
                    "state": "OPEN",
                    "createdAt": "2026-03-07T10:00:00Z",
                }
            ]
        )

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = gh_output
            result.stderr = ""
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        feed = GitHubIssueFeed(repo="synaptent/aragora")
        issues = feed.fetch()

        assert len(issues) == 1
        assert issues[0].number == 42
        assert issues[0].title == "Improve error handling"
        assert issues[0].labels == ["enhancement"]
        assert issues[0].state == "OPEN"

    def test_fetch_returns_empty_on_gh_failure(self, monkeypatch):
        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "gh: not found"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        feed = GitHubIssueFeed()
        issues = feed.fetch()
        assert issues == []

    def test_fetch_returns_empty_on_invalid_json(self, monkeypatch):
        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "not json"
            result.stderr = ""
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        feed = GitHubIssueFeed()
        assert feed.fetch() == []

    def test_fetch_returns_empty_on_file_not_found(self, monkeypatch):
        import subprocess as sp

        def _run(cmd, **kwargs):
            raise FileNotFoundError("gh not found")

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        feed = GitHubIssueFeed()
        assert feed.fetch() == []

    def test_fetch_passes_label_filter_and_repo(self, monkeypatch):
        captured_cmds: list[list[str]] = []

        def _run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = "[]"
            result.stderr = ""
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        feed = GitHubIssueFeed(repo="org/repo", label_filter="boss-ready", limit=10)
        feed.fetch()

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "--repo" in cmd
        assert cmd[cmd.index("--repo") + 1] == "org/repo"
        assert "--label" in cmd
        assert cmd[cmd.index("--label") + 1] == "boss-ready"
        assert "--limit" in cmd
        assert cmd[cmd.index("--limit") + 1] == "10"


# ---------------------------------------------------------------------------
# Issue selection tests
# ---------------------------------------------------------------------------


class TestSelectEligibleIssue:
    def test_selects_first_open_issue(self):
        issues = [_make_issue(1, "First"), _make_issue(2, "Second")]
        selected = select_eligible_issue(issues)
        assert selected is not None
        assert selected.number == 1

    def test_skips_closed_issues(self):
        issues = [_make_issue(1, "Closed", state="CLOSED"), _make_issue(2, "Open")]
        selected = select_eligible_issue(issues)
        assert selected is not None
        assert selected.number == 2

    def test_skips_issues_with_skip_labels(self):
        issues = [
            _make_issue(1, "Dup", labels=["duplicate"]),
            _make_issue(2, "Valid"),
        ]
        selected = select_eligible_issue(issues, skip_labels={"duplicate"})
        assert selected is not None
        assert selected.number == 2

    def test_requires_labels_when_specified(self):
        issues = [
            _make_issue(1, "No label"),
            _make_issue(2, "Has label", labels=["boss-ready"]),
        ]
        selected = select_eligible_issue(issues, require_labels={"boss-ready"})
        assert selected is not None
        assert selected.number == 2

    def test_returns_none_when_no_eligible_issue(self):
        issues = [_make_issue(1, "Invalid", labels=["wontfix"])]
        selected = select_eligible_issue(issues, skip_labels={"wontfix"})
        assert selected is None

    def test_returns_none_on_empty_list(self):
        assert select_eligible_issue([]) is None

    def test_skips_issues_with_empty_title(self):
        issues = [_make_issue(1, ""), _make_issue(2, "Valid")]
        selected = select_eligible_issue(issues)
        assert selected is not None
        assert selected.number == 2


# ---------------------------------------------------------------------------
# Runner freshness tests
# ---------------------------------------------------------------------------


class TestRunnerFreshness:
    def test_fresh_runner_passes(self, tmp_path, monkeypatch):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "codex-runner-1",
                            "runner_type": "codex",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "chatgpt_login",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inspection = MagicMock()
        inspection.available = True
        inspection.auth_mode = "chatgpt_login"
        inspection.to_dict.return_value = {"available": True, "auth_mode": "chatgpt_login"}

        with patch("aragora.swarm.runner_registry.CodexRunnerInspector") as inspector_cls:
            inspector_cls.return_value.inspect.return_value = inspection
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
            )

        assert result.fresh is True
        assert "codex-runner-1" in result.runner_ids
        assert result.blocked_reason is None

    def test_missing_owner_context_blocks(self):
        result = check_runner_freshness(env={})
        assert result.fresh is False
        assert result.blocked_reason == "missing_owner_context"

    def test_no_eligible_runners_blocks(self, tmp_path):
        registry_path = tmp_path / "runners.json"
        registry_path.write_text(json.dumps({"registrations": []}), encoding="utf-8")

        result = check_runner_freshness(
            registry_path=str(registry_path),
            env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
        )
        assert result.fresh is False
        assert result.blocked_reason == "no_eligible_registered_runners"

    def test_stale_runner_blocks(self, tmp_path, monkeypatch):
        registry_path = tmp_path / "runners.json"
        # Heartbeat from long ago → routing layer rejects as stale
        old_time = "2020-01-01T00:00:00+00:00"
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "codex-runner-1",
                            "runner_type": "codex",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "chatgpt_login",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": old_time,
                            "heartbeat_at": old_time,
                            "freshness_status": "stale",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inspection = MagicMock()
        inspection.available = True
        inspection.auth_mode = "chatgpt_login"
        inspection.to_dict.return_value = {"available": True}

        with patch("aragora.swarm.runner_registry.CodexRunnerInspector") as inspector_cls:
            inspector_cls.return_value.inspect.return_value = inspection
            result = check_runner_freshness(
                freshness_ttl_seconds=60.0,  # 1 minute TTL
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
            )

        assert result.fresh is False
        # Routing layer catches staleness via heartbeat before TTL check runs
        assert result.blocked_reason == "no_fresh_registered_runners"

    def test_runner_not_responding_blocks(self, tmp_path, monkeypatch):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "codex-runner-1",
                            "runner_type": "codex",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "chatgpt_login",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        inspection = MagicMock()
        inspection.available = False
        inspection.to_dict.return_value = {"available": False}

        with patch("aragora.swarm.runner_registry.CodexRunnerInspector") as inspector_cls:
            inspector_cls.return_value.inspect.return_value = inspection
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
            )

        assert result.fresh is False
        assert result.blocked_reason == "runner_not_responding"


# ---------------------------------------------------------------------------
# BossLoop core tests
# ---------------------------------------------------------------------------


class TestBossLoop:
    def test_no_fresh_runner_stops_immediately(self):
        config = _boss_config()
        loop = BossLoop(
            config=config,
            freshness_checker=lambda **kw: _fresh_result(
                fresh=False, blocked_reason="no_eligible_registered_runners"
            ),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_FRESH_RUNNER.value
        assert result.iterations_completed == 1
        assert len(result.issues_attempted) == 0
        assert len(result.needs_human_reasons) > 0
        assert "No fresh runner" in result.needs_human_reasons[0]

    def test_no_suitable_issue_stops(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config()
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert result.iterations_completed == 1
        assert len(result.issues_attempted) == 0
        assert "No suitable open issue" in result.needs_human_reasons[0]

    def test_bounded_iteration_limit(self):
        """Loop respects max_iterations even when issues keep flowing."""
        feed = MagicMock(spec=GitHubIssueFeed)
        # Return a fresh issue each time
        call_count = 0

        def _fetch():
            nonlocal call_count
            call_count += 1
            return [_make_issue(call_count, f"Issue {call_count}")]

        feed.fetch.side_effect = _fetch

        config = _boss_config(max_iterations=3)

        async def _fake_dispatch(issue, freshness):
            return {"status": "completed"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _fake_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 3
        assert len(result.issues_completed) == 3

    def test_consecutive_failures_stops(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        call_count = 0

        def _fetch():
            nonlocal call_count
            call_count += 1
            return [_make_issue(call_count, f"Failing issue {call_count}")]

        feed.fetch.side_effect = _fetch

        config = _boss_config(max_consecutive_failures=2, max_iterations=10)

        async def _failing_dispatch(issue, freshness):
            return {"status": "failed", "error": "worker crashed"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _failing_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.CONSECUTIVE_FAILURES.value
        assert result.iterations_completed == 2
        assert len(result.issues_failed) == 2

    def test_needs_human_stops(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(1, "Needs human review")]

        config = _boss_config()

        async def _needs_human_dispatch(issue, freshness):
            return {
                "status": "needs_human",
                "reasons": ["Approval required for merge."],
            }

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _needs_human_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value
        assert "Approval required for merge." in result.needs_human_reasons

    def test_retry_skips_maxed_issues(self):
        """Issues that have been attempted max_retries_per_issue times are skipped."""
        feed = MagicMock(spec=GitHubIssueFeed)
        # Always return the same issue
        feed.fetch.return_value = [_make_issue(42, "Flaky issue")]

        config = _boss_config(
            max_iterations=5,
            max_retries_per_issue=2,
            max_consecutive_failures=10,
        )

        async def _failing_dispatch(issue, freshness):
            return {"status": "failed", "error": "flaky"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _failing_dispatch

        result = asyncio.run(loop.run())

        # After 2 attempts on issue #42, it gets maxed out and the 3rd iteration
        # finds no suitable issue
        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert result.iterations_completed == 3  # 2 failures + 1 no-issue

    def test_on_status_callback_called(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config(max_iterations=1)
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        statuses: list[Any] = []
        result = asyncio.run(loop.run(on_status=statuses.append))

        assert len(statuses) == 1
        assert isinstance(statuses[0], BossIterationStatus)
        assert statuses[0].iteration == 1

    def test_successful_completion_resets_consecutive_failures(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        call_count = 0

        def _fetch():
            nonlocal call_count
            call_count += 1
            return [_make_issue(call_count, f"Issue {call_count}")]

        feed.fetch.side_effect = _fetch

        config = _boss_config(max_iterations=3, max_consecutive_failures=2)

        dispatch_count = 0

        async def _alternating_dispatch(issue, freshness):
            nonlocal dispatch_count
            dispatch_count += 1
            if dispatch_count == 1:
                return {"status": "failed", "error": "transient"}
            return {"status": "completed"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _alternating_dispatch

        result = asyncio.run(loop.run())

        # Should complete all 3 iterations: fail, succeed, succeed
        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 3
        assert len(result.issues_completed) == 2
        assert len(result.issues_failed) == 1


# ---------------------------------------------------------------------------
# Status payload shape tests
# ---------------------------------------------------------------------------


class TestStatusPayloadShape:
    def test_iteration_status_has_required_fields(self):
        status = BossIterationStatus(
            iteration=1,
            run_id="boss-test-123",
            timestamp="2026-03-07T00:00:00+00:00",
            runner_freshness={"fresh": True, "runner_ids": ["r-1"]},
            selected_issue={"number": 42, "title": "Test"},
            worker_status="completed",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=["Proceeding to next issue."],
            elapsed_seconds=1.5,
        )
        payload = status.to_dict()

        assert "iteration" in payload
        assert "run_id" in payload
        assert "timestamp" in payload
        assert "runner_freshness" in payload
        assert "selected_issue" in payload
        assert "worker_status" in payload
        assert "stop_reason" in payload
        assert "needs_human_reasons" in payload
        assert "next_actions" in payload
        assert "elapsed_seconds" in payload

    def test_loop_result_has_required_fields(self):
        result = BossLoopResult(
            run_id="boss-test-456",
            iterations_completed=5,
            total_elapsed_seconds=150.0,
            stop_reason="max_iterations",
            issues_attempted=[{"number": 1}],
            issues_completed=[{"number": 1}],
            issues_failed=[],
            iteration_statuses=[],
            needs_human_reasons=[],
            next_actions=["Done."],
        )
        payload = result.to_dict()

        assert payload["mode"] == "boss-loop"
        assert "run_id" in payload
        assert "iterations_completed" in payload
        assert "total_elapsed_seconds" in payload
        assert "stop_reason" in payload
        assert "issues_attempted" in payload
        assert "issues_completed" in payload
        assert "issues_failed" in payload
        assert "iteration_statuses" in payload
        assert "needs_human_reasons" in payload
        assert "next_actions" in payload

    def test_loop_result_is_json_serializable(self):
        result = BossLoopResult(
            run_id="boss-test-789",
            iterations_completed=1,
            total_elapsed_seconds=10.0,
            stop_reason="no_suitable_issue",
            issues_attempted=[],
            issues_completed=[],
            issues_failed=[],
            iteration_statuses=[
                {
                    "iteration": 1,
                    "run_id": "boss-test-789",
                    "worker_status": "idle",
                }
            ],
            needs_human_reasons=["No suitable issue."],
            next_actions=["Create an issue."],
        )
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["mode"] == "boss-loop"
        assert parsed["run_id"] == "boss-test-789"

    def test_freshness_result_serializable(self):
        result = _fresh_result(fresh=True)
        payload = result.to_dict()
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        assert parsed["fresh"] is True

    def test_github_issue_serializable(self):
        issue = _make_issue(99, "Test issue", labels=["bug", "priority"])
        payload = issue.to_dict()
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        assert parsed["number"] == 99
        assert "bug" in parsed["labels"]


# ---------------------------------------------------------------------------
# Stop-state reporting tests
# ---------------------------------------------------------------------------


class TestStopStateReporting:
    def test_max_iterations_next_actions(self):
        loop = BossLoop(config=_boss_config())
        loop._stop_reason = BossStopReason.MAX_ITERATIONS.value
        loop._iteration_statuses = [MagicMock(needs_human_reasons=[])] * 3
        actions = loop._derive_next_actions()
        assert any("completed" in a.lower() for a in actions)

    def test_no_fresh_runner_next_actions(self):
        loop = BossLoop(config=_boss_config())
        loop._stop_reason = BossStopReason.NO_FRESH_RUNNER.value
        actions = loop._derive_next_actions()
        assert any("register" in a.lower() or "refresh" in a.lower() for a in actions)

    def test_no_suitable_issue_next_actions(self):
        loop = BossLoop(config=_boss_config())
        loop._stop_reason = BossStopReason.NO_SUITABLE_ISSUE.value
        actions = loop._derive_next_actions()
        assert any("issue" in a.lower() for a in actions)

    def test_consecutive_failures_next_actions(self):
        loop = BossLoop(config=_boss_config())
        loop._stop_reason = BossStopReason.CONSECUTIVE_FAILURES.value
        loop._consecutive_failures = 3
        actions = loop._derive_next_actions()
        assert any("failure" in a.lower() or "investigate" in a.lower() for a in actions)

    def test_needs_human_next_actions(self):
        loop = BossLoop(config=_boss_config())
        loop._stop_reason = BossStopReason.NEEDS_HUMAN.value
        actions = loop._derive_next_actions()
        assert any("human" in a.lower() or "review" in a.lower() for a in actions)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestBossLoopCLI:
    def _swarm_args(self, **overrides: Any) -> argparse.Namespace:
        defaults = {
            "swarm_action_or_goal": "boss-loop",
            "swarm_goal": None,
            "spec": None,
            "skip_interrogation": False,
            "dry_run": False,
            "budget_limit": 5.0,
            "require_approval": False,
            "save_spec": None,
            "from_obsidian": None,
            "obsidian_vault": None,
            "no_obsidian_receipts": False,
            "profile": "developer",
            "autonomy": "propose",
            "max_parallel": 20,
            "no_loop": False,
            "target_branch": "main",
            "concurrency_cap": 8,
            "managed_dir_pattern": ".worktrees/{agent}-auto",
            "json": False,
            "run_id": None,
            "status_limit": 20,
            "refresh_scaling": False,
            "no_dispatch": False,
            "watch": False,
            "interval_seconds": 0.0,
            "max_ticks": 2,
            "all_runs": False,
            "dispatch_only": False,
            "no_wait": False,
            "freshness_ttl": 3600.0,
            "boss_repo": None,
            "boss_label_filter": None,
            "max_consecutive_failures": 3,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_boss_loop_json_output_blocked_no_runner(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=True, max_ticks=1)

        with patch(
            "aragora.swarm.boss_loop.check_runner_freshness",
            return_value=_fresh_result(fresh=False, blocked_reason="missing_owner_context"),
        ):
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "boss-loop"
        assert parsed["stop_reason"] == "no_fresh_runner"
        assert parsed["iterations_completed"] == 1
        assert len(parsed["needs_human_reasons"]) > 0
        assert "run_id" in parsed

    def test_boss_loop_json_output_blocked_no_issue(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=True, max_ticks=1)

        with (
            patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                return_value=_fresh_result(fresh=True),
            ),
            patch.object(GitHubIssueFeed, "fetch", return_value=[]),
        ):
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "boss-loop"
        assert parsed["stop_reason"] == "no_suitable_issue"
        assert parsed["iterations_completed"] == 1

    def test_boss_loop_text_output(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=False, max_ticks=1)

        with patch(
            "aragora.swarm.boss_loop.check_runner_freshness",
            return_value=_fresh_result(
                fresh=False, blocked_reason="no_eligible_registered_runners"
            ),
        ):
            cmd_swarm(args)

        out = capsys.readouterr().out
        assert "Boss loop finished" in out
        assert "no_fresh_runner" in out
        assert "iterations=1" in out

    def test_boss_loop_parser_accepts_action(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "boss-loop",
                "--max-ticks",
                "10",
                "--freshness-ttl",
                "1800",
                "--boss-repo",
                "synaptent/aragora",
                "--boss-label-filter",
                "boss-ready",
                "--max-consecutive-failures",
                "5",
                "--json",
            ]
        )
        assert args.command == "swarm"
        assert args.swarm_action_or_goal == "boss-loop"
        assert args.max_ticks == 10
        assert args.freshness_ttl == 1800.0
        assert args.boss_repo == "synaptent/aragora"
        assert args.boss_label_filter == "boss-ready"
        assert args.max_consecutive_failures == 5
        assert args.json is True


@pytest.mark.asyncio
async def test_dispatch_bounded_spec_enables_force_collect_on_max_ticks() -> None:
    spec = MagicMock()
    spec.is_dispatch_bounded.return_value = True

    fake_run = MagicMock()
    fake_run.to_dict.return_value = {
        "status": "completed",
        "run_id": "run-123",
        "work_orders": [
            {
                "status": "completed",
                "branch": "codex/example",
                "commit_shas": ["abc123"],
            }
        ],
    }

    with patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls:
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)

        await dispatch_bounded_spec(spec, max_ticks=7)

    kwargs = mock_commander_cls.return_value.run_supervised_from_spec.await_args.kwargs
    assert kwargs["max_ticks"] == 7
    assert kwargs["force_collect_on_max_ticks"] is True


# ---------------------------------------------------------------------------
# Fixture-backed Boss-loop invocation test
# ---------------------------------------------------------------------------


class TestBossLoopFixtureInvocation:
    """Proves the Boss loop selects from a fixture issue feed, requires fresh
    runners, and emits the required JSON payload shape."""

    def test_fixture_boss_loop_selects_task_or_blocks(self):
        """Run a fixture-backed Boss loop proving task selection, runner
        requirement, and JSON payload with run_id, iteration, and next_actions."""

        fixture_issues = [
            _make_issue(
                100,
                "Add retry to aragora/resilience/retry.py",
                body="Add exponential backoff to the retry module",
                labels=["enhancement"],
            ),
            _make_issue(200, "Fix typo in docs", labels=["docs"]),
        ]

        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = fixture_issues

        config = _boss_config(max_iterations=1)

        # Case 1: Fresh runner available -- task is selected and dispatched
        async def _completing_dispatch(issue, freshness):
            return {"status": "completed"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _completing_dispatch

        result = asyncio.run(loop.run())
        payload = result.to_dict()

        assert payload["mode"] == "boss-loop"
        assert payload["run_id"].startswith("boss-")
        assert payload["iterations_completed"] == 1
        assert payload["stop_reason"] == "max_iterations"
        assert len(payload["issues_completed"]) == 1
        assert payload["issues_completed"][0]["number"] == 100
        assert payload["issues_completed"][0]["title"] == "Add retry to aragora/resilience/retry.py"
        assert isinstance(payload["next_actions"], list)
        assert len(payload["next_actions"]) > 0

        # Verify JSON serialization works
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        assert parsed["mode"] == "boss-loop"
        assert "run_id" in parsed

        # Case 2: No fresh runner -- blocks truthfully
        loop2 = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(
                fresh=False, blocked_reason="all_runners_stale"
            ),
        )

        result2 = asyncio.run(loop2.run())
        payload2 = result2.to_dict()

        assert payload2["stop_reason"] == "no_fresh_runner"
        assert len(payload2["issues_attempted"]) == 0
        assert len(payload2["needs_human_reasons"]) > 0
        assert "run_id" in payload2
        assert isinstance(payload2["next_actions"], list)
        assert len(payload2["next_actions"]) > 0
