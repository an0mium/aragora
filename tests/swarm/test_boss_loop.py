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
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.boss_loop import (
    _should_replace_with_focused_tests,
    BossIterationStatus,
    BossLoop,
    BossLoopConfig,
    BossLoopResult,
    BossStopReason,
    GitHubIssue,
    GitHubIssueFeed,
    RunnerFreshnessResult,
    check_runner_freshness,
    discover_focused_tests,
    dispatch_bounded_spec,
    extract_pre_dispatch_validation_commands,
    extract_issue_validation_contract,
    run_pre_dispatch_validation_commands,
    sanitize_issue_body_for_dispatch,
    select_eligible_issue,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_issue(
    number: int = 1,
    title: str = "Fix the dashboard",
    body: str = (
        "The dashboard is slow, improve query performance in aragora/analytics/dashboard.py\n\n"
        "Acceptance Criteria:\n"
        "- pytest -q tests/swarm/test_boss_loop.py\n"
        "- Dashboard query path remains bounded to aragora/analytics/dashboard.py\n"
    ),
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


class TestPreDispatchValidationCommands:
    def test_extract_pre_dispatch_validation_commands_filters_non_commands(self):
        body = """
Acceptance Criteria:
- Dashboard query path remains bounded to aragora/analytics/dashboard.py
- python -m pytest tests/swarm/test_boss_loop.py -q
- `aragora quickstart --topic test --rounds 1 --json | python3 -c "import json,sys; json.load(sys.stdin)"`
"""
        assert extract_pre_dispatch_validation_commands(body) == [
            "python -m pytest tests/swarm/test_boss_loop.py -q",
            'python3 -m aragora.cli.main quickstart --topic test --rounds 1 --json | python3 -c "import json,sys; json.load(sys.stdin)"',
        ]

    def test_extract_pre_dispatch_validation_commands_normalizes_inline_acceptance(self):
        body = """**Test:** `aragora quickstart --topic "test" --rounds 1 --json | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'receipt_id' in d"`

**Acceptance:** `pytest tests/cli/test_quickstart.py -x -q` passes."""

        assert extract_pre_dispatch_validation_commands(body) == [
            'python3 -m aragora.cli.main quickstart --topic "test" --rounds 1 --json | '
            "python3 -c \"import json,sys; d=json.load(sys.stdin); assert 'receipt_id' in d\"",
            "pytest tests/cli/test_quickstart.py -x -q",
        ]

    def test_run_pre_dispatch_validation_commands_stops_on_failure(self, monkeypatch):
        calls: list[str] = []

        def _run(cmd, **kwargs):
            calls.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "failed"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        result = run_pre_dispatch_validation_commands(
            ["python -m pytest tests/swarm/test_boss_loop.py -q"],
            cwd=Path.cwd(),
            timeout_seconds=15,
        )

        assert calls == [["/bin/bash", "-lc", "python -m pytest tests/swarm/test_boss_loop.py -q"]]
        assert result["satisfied"] is False
        assert result["results"][0]["status"] == "failed"

    def test_requires_labels_when_specified(self):
        issues = [
            _make_issue(1, "No label"),
            _make_issue(2, "Has label", labels=["boss-ready"]),
        ]
        selected = select_eligible_issue(issues, require_labels={"boss-ready"})
        assert selected is not None
        assert selected.number == 2

    def test_require_labels_all_match(self):
        """When multiple labels are required, ALL must be present on the issue."""
        issues = [
            _make_issue(1, "Only one label", labels=["P0"]),
            _make_issue(2, "Both labels", labels=["P0", "queue-eligible"]),
            _make_issue(3, "Extra labels too", labels=["P0", "queue-eligible", "enhancement"]),
        ]
        selected = select_eligible_issue(issues, require_labels={"P0", "queue-eligible"})
        assert selected is not None
        assert selected.number == 2

    def test_require_labels_rejects_partial_match(self):
        """An issue with only some of the required labels is rejected."""
        issues = [
            _make_issue(1, "Partial", labels=["P0"]),
        ]
        selected = select_eligible_issue(issues, require_labels={"P0", "queue-eligible"})
        assert selected is None

    def test_no_require_labels_means_no_filtering(self):
        """When require_labels is None, all issues are eligible (label-wise)."""
        issues = [
            _make_issue(1, "No labels"),
            _make_issue(2, "Has labels", labels=["P0"]),
        ]
        selected = select_eligible_issue(issues, require_labels=None)
        assert selected is not None
        assert selected.number == 1


class TestValidationContractExtraction:
    def test_extracts_bullets_from_acceptance_section(self):
        body = """
Summary text.

Acceptance Criteria:
- pytest -q tests/swarm/test_boss_loop.py
- No files outside aragora/swarm/ are changed
"""

        assert extract_issue_validation_contract(body) == [
            "pytest -q tests/swarm/test_boss_loop.py",
            "No files outside aragora/swarm/ are changed",
        ]

    def test_extracts_inline_validation_and_pytest_lines(self):
        body = """
Validation: verify JSON output includes stop_reason

python -m pytest tests/swarm/test_boss_loop.py -q
"""

        assert extract_issue_validation_contract(body) == [
            "verify JSON output includes stop_reason",
            "python -m pytest tests/swarm/test_boss_loop.py -q",
        ]

    def test_stops_at_scope_hints_and_restart_sections_without_colons(self):
        body = """
Summary text.

Acceptance Criteria
- pytest -q tests/swarm/test_boss_loop.py -k focused
- Keep the boss loop bounded

Scope hints
- aragora/swarm/boss_loop.py

Implementation Rules
- do not include these in validation
"""

        assert extract_issue_validation_contract(body) == [
            "pytest -q tests/swarm/test_boss_loop.py -k focused",
            "Keep the boss loop bounded",
        ]

    def test_extracts_bold_inline_test_and_acceptance_markers(self):
        body = """
Add `--json` flag to `aragora quickstart`.

**Test:** `aragora quickstart --topic 'test' --rounds 1 --json | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'receipt_id' in d"`

**Acceptance:** `pytest tests/cli/test_quickstart.py -x -q` passes.
"""

        assert extract_issue_validation_contract(body) == [
            "`aragora quickstart --topic 'test' --rounds 1 --json | python3 -c \"import json,sys; d=json.load(sys.stdin); assert 'receipt_id' in d\"`",
            "`pytest tests/cli/test_quickstart.py -x -q` passes.",
        ]

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


class TestFocusedVerificationReplacement:
    def test_keeps_explicit_test_file_commands(self):
        command = "python -m pytest tests/swarm/test_boss_loop.py tests/swarm/test_spec.py -q"
        assert _should_replace_with_focused_tests(command) is False

    def test_rewrites_directory_level_test_commands(self):
        command = "python -m pytest tests/swarm/ -q"
        assert _should_replace_with_focused_tests(command) is True


class TestDispatchIssueNormalization:
    def test_sanitize_issue_body_for_dispatch_keeps_context_only(self):
        body = """
Summary:
- Fix the boss loop prompt pollution.

Context:
Workers should only see bounded context.

Acceptance Criteria:
- pytest -q tests/swarm/test_boss_loop.py

Scope hints:
- aragora/swarm/boss_loop.py
"""

        sanitized = sanitize_issue_body_for_dispatch(body)

        assert "Fix the boss loop prompt pollution." in sanitized
        assert "Workers should only see bounded context." in sanitized
        assert "Acceptance Criteria" not in sanitized
        assert "Scope hints" not in sanitized


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

    def test_missing_owner_context_blocks(self, monkeypatch):
        monkeypatch.setattr("aragora.swarm.runner_registry.getpass.getuser", lambda: "")
        monkeypatch.setattr(
            "aragora.swarm.runner_registry.subprocess.run",
            lambda *args, **kwargs: type(
                "_Proc",
                (),
                {"returncode": 1, "stdout": "", "stderr": "fatal: not a git repository"},
            )(),
        )
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

    def test_runner_freshness_reinspects_selected_claude_profile(self, tmp_path):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "claude-runner-1",
                            "runner_type": "claude",
                            "profile": "max-01",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "subscription",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                            "probe_status": "passed",
                            "probe_checked_at": now,
                            "probe_ttl_seconds": 3600,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        class _Inspector:
            def inspect(self) -> MagicMock:
                inspection = MagicMock()
                inspection.available = True
                inspection.auth_mode = "subscription"
                inspection.runner_id = "claude-runner-1"
                inspection.to_dict.return_value = {
                    "runner_id": "claude-runner-1",
                    "profile": "max-01",
                    "available": True,
                    "auth_mode": "subscription",
                }
                return inspection

        with patch("aragora.swarm.runner_registry.make_runner_inspector") as inspector_factory:
            inspector_factory.return_value = _Inspector()
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="claude",
            )

        assert result.fresh is True
        inspector_factory.assert_called_with("claude", env=ANY, profile="max-01")

    def test_runner_freshness_uses_explicit_profile_pool_and_probe_policy(self, tmp_path):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "claude-runner-1",
                            "runner_type": "claude",
                            "profile": "max-01",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "subscription",
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

        class _Inspector:
            def inspect(self) -> MagicMock:
                inspection = MagicMock()
                inspection.available = True
                inspection.auth_mode = "subscription"
                inspection.runner_id = "claude-runner-1"
                inspection.to_dict.return_value = {
                    "runner_id": "claude-runner-1",
                    "profile": "max-01",
                    "available": True,
                    "auth_mode": "subscription",
                }
                return inspection

        with (
            patch(
                "aragora.swarm.runner_registry.refresh_discovered_runners",
                return_value=[],
            ) as refresh_mock,
            patch("aragora.swarm.runner_registry.make_runner_inspector", return_value=_Inspector()),
        ):
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="claude",
                allowed_profiles={"max-01"},
                verified_runner_target=0,
                runner_probe_limit=4,
            )

        assert result.fresh is True
        assert result.details["probe"]["verified_target"] == 0
        refresh_mock.assert_called_once()
        assert refresh_mock.call_args.kwargs["profiles"] == {"max-01"}

    def test_runner_freshness_probes_toward_verified_target(self, tmp_path, monkeypatch):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "claude-runner-1",
                            "runner_type": "claude",
                            "profile": "max-01",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "subscription",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                            "probe_status": "passed",
                            "probe_checked_at": now,
                            "probe_ttl_seconds": 3600,
                        },
                        {
                            "runner_id": "claude-runner-2",
                            "runner_type": "claude",
                            "profile": "max-02",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "subscription",
                            "owner_binding": {"user_id": "user-1", "workspace_id": "ws-1"},
                            "capabilities": {"max_parallel_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ARAGORA_BOSS_VERIFIED_RUNNER_TARGET", "2")
        monkeypatch.setenv("ARAGORA_BOSS_RUNNER_PROBE_LIMIT", "1")
        inspection = SimpleNamespace(runner_id="claude-runner-2", profile="max-02")
        probe = SimpleNamespace(
            status="passed",
            to_runner_fields=lambda: {
                "probe_status": "passed",
                "probe_checked_at": now,
                "probe_detail": "Live prompt probe succeeded.",
                "probe_latency_seconds": 1.0,
                "probe_ttl_seconds": 3600,
            },
            to_dict=lambda: {
                "runner_id": "claude-runner-2",
                "runner_type": "claude",
                "probe_status": "passed",
            },
        )

        class _Inspector:
            def __init__(self, runner_id: str) -> None:
                self._runner_id = runner_id

            def inspect(self) -> MagicMock:
                inspected = MagicMock()
                inspected.available = True
                inspected.auth_mode = "subscription"
                inspected.runner_id = self._runner_id
                inspected.to_dict.return_value = {
                    "runner_id": self._runner_id,
                    "available": True,
                    "auth_mode": "subscription",
                }
                return inspected

        def _make_inspector(runner_type: str, *, env=None, profile=None):
            runner_id = "claude-runner-1" if profile == "max-01" else "claude-runner-2"
            return _Inspector(runner_id)

        with (
            patch(
                "aragora.swarm.runner_registry.refresh_discovered_runners",
                return_value=[inspection],
            ),
            patch(
                "aragora.swarm.runner_registry.prioritized_probe_candidates",
                return_value=[inspection],
            ),
            patch("aragora.swarm.runner_registry.probe_runner_execution", return_value=probe),
            patch(
                "aragora.swarm.runner_registry.make_runner_inspector", side_effect=_make_inspector
            ),
        ):
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="claude",
            )

        assert result.fresh is True
        assert result.details["probe"]["auto_probe_triggered"] is True
        assert result.details["probe"]["passed"] == 1

    def test_runner_freshness_blocks_when_no_selected_runner_is_execution_verified(
        self, tmp_path, monkeypatch
    ):
        registry_path = tmp_path / "runners.json"
        now = datetime.now(UTC).isoformat()
        registry_path.write_text(
            json.dumps(
                {
                    "registrations": [
                        {
                            "runner_id": "claude-runner-1",
                            "runner_type": "claude",
                            "profile": "max-01",
                            "registered": True,
                            "availability": "available",
                            "available": True,
                            "auth_mode": "subscription",
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
        monkeypatch.setenv("ARAGORA_BOSS_VERIFIED_RUNNER_TARGET", "1")
        monkeypatch.setenv("ARAGORA_BOSS_RUNNER_PROBE_LIMIT", "1")
        inspection = SimpleNamespace(runner_id="claude-runner-1", profile="max-01")
        failed_probe = SimpleNamespace(
            status="failed",
            detail="Probe failed (exit 1): org access denied",
            to_runner_fields=lambda: {
                "probe_status": "failed",
                "probe_checked_at": now,
                "probe_detail": "Probe failed (exit 1): org access denied",
                "probe_latency_seconds": 1.0,
                "probe_ttl_seconds": 3600,
            },
            to_dict=lambda: {
                "runner_id": "claude-runner-1",
                "runner_type": "claude",
                "probe_status": "failed",
            },
        )

        class _Inspector:
            def inspect(self) -> MagicMock:
                inspected = MagicMock()
                inspected.available = True
                inspected.auth_mode = "subscription"
                inspected.runner_id = "claude-runner-1"
                inspected.to_dict.return_value = {
                    "runner_id": "claude-runner-1",
                    "available": True,
                    "auth_mode": "subscription",
                }
                return inspected

        with (
            patch(
                "aragora.swarm.runner_registry.refresh_discovered_runners",
                return_value=[inspection],
            ),
            patch(
                "aragora.swarm.runner_registry.prioritized_probe_candidates",
                return_value=[inspection],
            ),
            patch(
                "aragora.swarm.runner_registry.probe_runner_execution",
                return_value=failed_probe,
            ),
            patch(
                "aragora.swarm.runner_registry.make_runner_inspector",
                return_value=_Inspector(),
            ),
        ):
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="claude",
            )

        assert result.fresh is False
        assert result.blocked_reason == "no_execution_verified_runner"
        assert result.details["probe"]["failed"] == 1


# ---------------------------------------------------------------------------
# BossLoop core tests
# ---------------------------------------------------------------------------


class TestBossLoop:
    def test_no_fresh_runner_stops_immediately(self):
        config = _boss_config()
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(1, "Runner blocked issue")]
        loop = BossLoop(
            config=config,
            issue_feed=feed,
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

    def test_specific_issue_number_selects_target_issue(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [
            _make_issue(909, "Meta benchmark issue"),
            _make_issue(873, "Bounded execution issue"),
        ]

        config = _boss_config(max_iterations=1, issue_number=873)

        async def _complete(issue, freshness):
            assert issue.number == 873
            return {"status": "completed"}

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _complete

        result = asyncio.run(loop.run())

        assert result.iterations_completed == 1
        assert result.issues_attempted[0]["number"] == 873

    def test_specific_issue_number_missing_stops_truthfully(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(909, "Meta benchmark issue")]

        loop = BossLoop(
            config=_boss_config(max_iterations=1, issue_number=873),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert "Target issue #873" in result.needs_human_reasons[0]
        assert "Verify issue #873" in result.next_actions[0]
        assert "Remove --boss-issue-number" in result.next_actions[1]

    def test_specific_issue_number_closed_stops_with_closed_reason(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []
        feed._fetch_issue.return_value = _make_issue(873, "Closed target", state="CLOSED")

        loop = BossLoop(
            config=_boss_config(max_iterations=1, issue_number=873),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert "is closed and cannot be selected" in result.needs_human_reasons[0]
        assert "Reopen issue #873" in result.next_actions[0]
        assert "Remove --boss-issue-number" in result.next_actions[1]

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

    def test_retry_rotation_switches_target_agent_after_needs_human(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(42, "Retry with rotated agent")]

        freshness_requests: list[str | None] = []

        def _freshness_checker(**kwargs):
            freshness_requests.append(kwargs.get("requested_runner_type"))
            return RunnerFreshnessResult(
                fresh=True,
                runner_ids=["claude-runner-1", "codex-runner-1"],
                checked_at=datetime.now(UTC).isoformat(),
                details={
                    "routing": {
                        "selected_runners": [
                            {"runner_id": "claude-runner-1", "runner_type": "claude"},
                            {"runner_id": "codex-runner-1", "runner_type": "codex"},
                        ],
                        "selected_runner_ids": ["claude-runner-1", "codex-runner-1"],
                    }
                },
            )

        loop = BossLoop(
            config=_boss_config(
                max_iterations=2,
                max_retries_per_issue=3,
                auto_continue_on_needs_human=True,
                default_target_agent="claude",
                model_rotation=["claude", "codex"],
            ),
            issue_feed=feed,
            freshness_checker=_freshness_checker,
        )

        def _claim_runner(freshness, *, requested_target_agent=None):
            runner_type = requested_target_agent or "claude"
            return (
                {
                    "runner_id": f"{runner_type}-runner-1",
                    "runner_type": runner_type,
                },
                f"{runner_type}-runner-1",
            )

        loop._claim_runner_for_dispatch = _claim_runner
        loop._release_runner_claim = lambda runner_id: None

        dispatch_results = AsyncMock(
            side_effect=[
                {"status": "needs_human", "reasons": ["Approval required."]},
                {
                    "status": "completed",
                    "outcome": "deliverable_created",
                    "deliverable": {"type": "branch"},
                },
            ]
        )

        with patch("aragora.swarm.boss_loop.dispatch_bounded_spec", dispatch_results):
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert freshness_requests == ["claude", None]
        first_call = dispatch_results.await_args_list[0].kwargs
        second_call = dispatch_results.await_args_list[1].kwargs
        assert first_call["default_target_agent"] == "claude"
        assert first_call["selected_runner"]["runner_type"] == "claude"
        assert second_call["default_target_agent"] == "codex"
        assert second_call["selected_runner"]["runner_type"] == "codex"

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

    @pytest.mark.asyncio
    async def test_on_status_callback_emits_dispatching_before_final_status(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(1749, "Receipt markdown flag")]

        loop = BossLoop(
            config=_boss_config(max_iterations=1, default_target_agent="codex"),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

        statuses: list[BossIterationStatus] = []
        result = await loop.run(on_status=statuses.append)

        assert [status.worker_status for status in statuses] == ["dispatching", "completed"]
        assert statuses[0].selected_issue["number"] == 1749
        assert statuses[1].selected_issue["number"] == 1749
        assert statuses[0].next_actions == ["Dispatching issue #1749 with codex."]
        assert [status["worker_status"] for status in result.iteration_statuses] == ["completed"]

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

    def test_iteration_appends_jsonl_metrics(self, tmp_path: Path):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(42, "Emit boss metrics")]

        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                metrics_jsonl_path=str(tmp_path / "boss_metrics.jsonl"),
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        async def _completed_dispatch(issue, freshness):
            return {
                "status": "completed",
                "run": {
                    "work_orders": [
                        {
                            "changed_paths": [
                                "aragora/swarm/boss_loop.py",
                                "tests/swarm/test_boss_loop.py",
                            ],
                            "tests_run": ["python -m pytest tests/swarm/test_boss_loop.py -q"],
                            "verification_results": [
                                {
                                    "command": "python -m pytest tests/swarm/test_boss_loop.py -q",
                                    "passed": True,
                                }
                            ],
                        }
                    ]
                },
            }

        loop._dispatch_issue = _completed_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        assert payload["iteration"] == 1
        assert payload["issue_number"] == 42
        assert payload["worker_status"] == "completed"
        assert payload["files_changed"] == 2
        assert payload["tests_run"] == 1
        assert payload["tests_passed"] == 1
        assert payload["elapsed_seconds"] >= 0.0

    def test_missing_validation_contract_stops_with_needs_human(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [
            _make_issue(
                7,
                "Issue missing validation",
                body="Tighten the boss loop selection logic in aragora/swarm/boss_loop.py",
            )
        ]

        loop = BossLoop(
            config=_boss_config(max_iterations=1),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value
        assert "lacks an explicit validation contract" in result.needs_human_reasons[0]

    def test_bold_markdown_validation_contract_allows_dispatch(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [
            _make_issue(
                1639,
                "Add --json output flag to aragora quickstart CLI",
                body="""Add `--json` flag to `aragora quickstart` so the debate result is printed as structured JSON.

**Files:** `aragora/cli/commands/quickstart.py`, `aragora/cli/parser.py`

**Test:** `aragora quickstart --topic "test" --rounds 1 --json | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'receipt_id' in d"`

**Acceptance:** `pytest tests/cli/test_quickstart.py -x -q` passes.""",
            )
        ]

        loop = BossLoop(
            config=_boss_config(max_iterations=1),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        async def _completed_dispatch(issue, freshness):
            return {
                "status": "completed",
                "outcome": "deliverable_created",
                "deliverable": {"type": "branch"},
            }

        loop._dispatch_issue = _completed_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 1
        assert result.issues_completed[0]["number"] == 1639

    def test_no_dispatch_preview_stops_truthfully(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(8, "Preview only issue")]

        loop = BossLoop(
            config=_boss_config(max_iterations=1, dispatch_enabled=False),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value
        assert "No-dispatch preview only" in result.needs_human_reasons[0]
        assert "Rerun without --no-dispatch" in result.next_actions[1]
        assert result.iteration_statuses[0]["worker_outcome"] == "preview_only"

    def test_single_tick_live_dispatch_returns_after_launch(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(873, "Bounded live issue")]

        loop = BossLoop(
            config=_boss_config(max_iterations=1),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        async def _running_dispatch(issue, freshness):
            return {
                "status": "running",
                "outcome": "dispatched",
                "run_id": "run-873",
            }

        loop._dispatch_issue = _running_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 1
        assert len(result.issues_attempted) == 1
        assert len(result.issues_completed) == 0
        assert len(result.issues_failed) == 0
        assert "Supervisor run run-873 is active" in result.next_actions[0]
        assert result.iteration_statuses[0]["worker_status"] == "running"
        assert result.iteration_statuses[0]["worker_outcome"] == "dispatched"


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

    def test_no_suitable_issue_prefers_iteration_specific_next_actions(self):
        loop = BossLoop(config=_boss_config(issue_number=873))
        loop._stop_reason = BossStopReason.NO_SUITABLE_ISSUE.value
        loop._iteration_statuses = [
            BossIterationStatus(
                iteration=1,
                run_id="run-1",
                timestamp="2026-03-29T00:00:00+00:00",
                runner_freshness={},
                selected_issue=None,
                worker_status="idle",
                stop_reason=BossStopReason.NO_SUITABLE_ISSUE.value,
                needs_human_reasons=[
                    "Target issue #873 was not found in the issue feed or is not eligible under current filters/retry state."
                ],
                next_actions=[
                    "Verify issue #873 is still open, eligible, and has not exceeded retry limits.",
                    "Remove --boss-issue-number to return to feed-driven selection.",
                ],
            )
        ]

        actions = loop._derive_next_actions()

        assert actions == [
            "Verify issue #873 is still open, eligible, and has not exceeded retry limits.",
            "Remove --boss-issue-number to return to feed-driven selection.",
        ]

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
            "boss_issue_number": None,
            "max_consecutive_failures": 3,
            "labels": None,
            "allow_missing_validation_contract": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_boss_loop_json_output_blocked_no_runner(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=True, max_ticks=1)

        with (
            patch.object(
                GitHubIssueFeed,
                "fetch",
                return_value=[_make_issue(1, "Runner blocked issue")],
            ),
            patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                return_value=_fresh_result(fresh=False, blocked_reason="missing_owner_context"),
            ),
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

    def test_boss_loop_no_issue_skips_runner_freshness_check(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=True, max_ticks=1)

        with patch.object(GitHubIssueFeed, "fetch", return_value=[]):
            with patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                side_effect=AssertionError("freshness should not be checked for an empty queue"),
            ):
                cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["stop_reason"] == "no_suitable_issue"

    def test_boss_loop_text_output(self, capsys):
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(json=False, max_ticks=1)

        with (
            patch.object(
                GitHubIssueFeed,
                "fetch",
                return_value=[_make_issue(1, "Runner blocked issue")],
            ),
            patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                return_value=_fresh_result(
                    fresh=False, blocked_reason="no_eligible_registered_runners"
                ),
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
                "--boss-issue-number",
                "873",
                "--max-consecutive-failures",
                "5",
                "--allow-missing-validation-contract",
                "--json",
            ]
        )
        assert args.command == "swarm"
        assert args.swarm_action_or_goal == "boss-loop"
        assert args.max_ticks == 10
        assert args.freshness_ttl == 1800.0
        assert args.boss_repo == "synaptent/aragora"
        assert args.boss_label_filter == "boss-ready"
        assert args.boss_issue_number == 873
        assert args.max_consecutive_failures == 5
        assert args.allow_missing_validation_contract is True
        assert args.json is True

    def test_boss_loop_parser_accepts_label_repeatable(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "boss-loop",
                "--label",
                "P0",
                "--label",
                "queue-eligible",
                "--json",
            ]
        )
        assert args.command == "swarm"
        assert args.swarm_action_or_goal == "boss-loop"
        assert args.labels == ["P0", "queue-eligible"]
        assert args.json is True

    def test_boss_loop_parser_no_labels_defaults_to_none(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["swarm", "boss-loop"])
        assert args.labels is None

    def test_boss_loop_label_filter_wired_to_config(self):
        """--label args are wired to BossLoopConfig.require_labels as a set."""
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._swarm_args(
            json=True,
            max_ticks=1,
            labels=["P0", "queue-eligible"],
        )

        with patch(
            "aragora.swarm.boss_loop.check_runner_freshness",
            return_value=_fresh_result(fresh=False, blocked_reason="missing_owner_context"),
        ) as _:
            cmd_swarm(args)

        # The loop ran (and stopped because no fresh runner). We verify that
        # the config was constructed with the right require_labels by checking
        # that it didn't crash and produced valid output.  A more direct test
        # of the wiring follows below.

    def test_boss_loop_label_filter_wired_require_labels_all_match(self, capsys):
        """Issues must carry ALL --label values to be selected."""
        from aragora.cli.commands.swarm import cmd_swarm

        # Issue only has P0, not queue-eligible
        fixture_issues = [
            _make_issue(1, "Partial match", labels=["P0"]),
        ]
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = fixture_issues

        args = self._swarm_args(
            json=True,
            max_ticks=1,
            labels=["P0", "queue-eligible"],
        )

        with (
            patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                return_value=_fresh_result(fresh=True),
            ),
            patch(
                "aragora.swarm.boss_loop.GitHubIssueFeed",
                return_value=feed,
            ),
        ):
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["stop_reason"] == "no_suitable_issue"

    def test_boss_loop_label_filter_passes_with_all_labels(self, capsys):
        """Issues carrying ALL required labels pass the filter."""
        from aragora.cli.commands.swarm import cmd_swarm

        fixture_issues = [
            _make_issue(
                1,
                "Full match",
                labels=["P0", "queue-eligible", "enhancement"],
                body=(
                    "Fix the thing\n\n"
                    "Acceptance Criteria:\n"
                    "- pytest -q tests/swarm/test_boss_loop.py\n"
                ),
            ),
        ]
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = fixture_issues

        args = self._swarm_args(
            json=True,
            max_ticks=1,
            labels=["P0", "queue-eligible"],
            no_dispatch=True,
        )

        with (
            patch(
                "aragora.swarm.boss_loop.check_runner_freshness",
                return_value=_fresh_result(fresh=True),
            ),
            patch(
                "aragora.swarm.boss_loop.GitHubIssueFeed",
                return_value=feed,
            ),
        ):
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        # The issue was selected (didn't stop with no_suitable_issue)
        assert parsed["stop_reason"] != "no_suitable_issue"


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


@pytest.mark.asyncio
async def test_dispatch_bounded_spec_wait_false_returns_running_after_launch() -> None:
    spec = MagicMock()
    spec.is_dispatch_bounded.return_value = True

    active_run = MagicMock()
    active_run.to_dict.return_value = {
        "status": "active",
        "run_id": "run-873",
        "work_orders": [
            {
                "status": "dispatched",
                "branch": "",
                "commit_shas": [],
            }
        ],
    }

    with patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls:
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(
            return_value=active_run
        )

        result = await dispatch_bounded_spec(spec, wait_for_completion=False)

    kwargs = mock_commander_cls.return_value.run_supervised_from_spec.await_args.kwargs
    assert kwargs["wait"] is False
    assert result["status"] == "running"
    assert result["outcome"] == "dispatched"
    assert result["run_id"] == "run-873"


@pytest.mark.asyncio
async def test_dispatch_issue_refine_exports_worker_env_to_commander() -> None:
    issue = _make_issue(
        1641,
        "Wire prompt refiner env",
        body=(
            "Pass prompt-refiner file and test hints as worker env vars.\n\n"
            "Acceptance Criteria:\n"
            "- pytest -q tests/swarm/test_boss_loop.py -k refine\n"
        ),
    )
    loop = BossLoop(config=_boss_config(max_iterations=1))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    fake_run = MagicMock()
    fake_run.to_dict.return_value = {
        "status": "completed",
        "run_id": "run-1641",
        "work_orders": [
            {
                "status": "completed",
                "branch": "codex/refine-env",
                "commit_shas": ["abc123"],
            }
        ],
    }

    refinement = {
        "refined_prompt": "Refined goal",
        "files_to_change": ["aragora/swarm/boss_loop.py", "aragora/swarm/prompt_refiner.py"],
        "test_patterns": ["tests/swarm/test_boss_loop.py"],
        "constraints": [],
        "context_gathered": True,
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(return_value=refinement),
        ),
        patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
    ):
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    kwargs = mock_commander_cls.return_value.run_supervised_from_spec.await_args.kwargs
    assert result["status"] == "completed"
    assert kwargs["worker_env"] == {
        "ARAGORA_RELEVANT_FILES": os.pathsep.join(refinement["files_to_change"]),
        "ARAGORA_TEST_PATTERNS": os.pathsep.join(refinement["test_patterns"]),
    }


@pytest.mark.asyncio
async def test_dispatch_issue_builds_clean_spec_from_issue_body() -> None:
    issue = _make_issue(
        1733,
        "Tighten supervisor merge gate",
        body=(
            "Summary:\n"
            "- Use a clean worker goal instead of the whole issue blob.\n\n"
            "Context:\n"
            "Workers should keep only dispatch-relevant context.\n\n"
            "Acceptance Criteria:\n"
            "- pytest -q tests/swarm/test_boss_loop.py\n\n"
            "Scope hints:\n"
            "- aragora/swarm/supervisor.py\n"
        ),
    )
    loop = BossLoop(config=_boss_config(max_iterations=1))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    fake_run = MagicMock()
    fake_run.to_dict.return_value = {
        "status": "completed",
        "run_id": "run-1733",
        "work_orders": [
            {"status": "completed", "branch": "codex/merge-gate", "commit_shas": ["abc123"]}
        ],
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(
                return_value={
                    "refined_prompt": "",
                    "files_to_change": [],
                    "test_patterns": [],
                    "constraints": [],
                    "context_gathered": False,
                }
            ),
        ),
        patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
    ):
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    spec = mock_commander_cls.return_value.run_supervised_from_spec.await_args.args[0]
    assert "[Issue #1733] Tighten supervisor merge gate" in spec.raw_goal
    assert "Workers should keep only dispatch-relevant context." in spec.raw_goal
    assert "Scope hints" not in spec.raw_goal
    assert "aragora/swarm/supervisor.py" in spec.file_scope_hints
    assert spec.acceptance_criteria == ["pytest -q tests/swarm/test_boss_loop.py"]


@pytest.mark.asyncio
async def test_dispatch_issue_consumes_pending_handoff_prompt_and_target_agent() -> None:
    issue = _make_issue(1701, "Retry same issue with handoff")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="claude"))
    loop._pending_handoff_prompts[issue.number] = (
        "## Goal\nRetry with prior context\n\n## Context (from claude, round 1)\nKnown bad edge case.",
        "codex",
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": requested_target_agent or "codex",
    }

    fake_result = {
        "status": "completed",
        "run": {"work_orders": [{"status": "completed", "target_agent": "codex"}]},
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value=fake_result),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    dispatched_spec = dispatch_mock.await_args.args[0]
    dispatch_kwargs = dispatch_mock.await_args.kwargs
    assert result["status"] == "completed"
    assert "Retry with prior context" in dispatched_spec.raw_goal
    assert dispatch_kwargs["default_target_agent"] == "codex"
    assert issue.number not in loop._pending_handoff_prompts


@pytest.mark.asyncio
async def test_dispatch_issue_preserves_pending_handoff_on_pre_run_failure() -> None:
    issue = _make_issue(1703, "Retry handoff should survive dispatch crash")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="claude"))
    loop._pending_handoff_prompts[issue.number] = (
        "## Goal\nRetry with preserved context\n\n## Context (from claude, round 1)\nStill relevant.",
        "codex",
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": requested_target_agent or "codex",
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "failed", "outcome": "crash", "error": "boom"}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    dispatch_kwargs = dispatch_mock.await_args.kwargs
    assert result["status"] == "failed"
    assert dispatch_kwargs["default_target_agent"] == "codex"
    assert loop._pending_handoff_prompts[issue.number] == (
        "## Goal\nRetry with preserved context\n\n## Context (from claude, round 1)\nStill relevant.",
        "codex",
    )


@pytest.mark.asyncio
async def test_dispatch_issue_uses_configured_dispatch_max_ticks() -> None:
    issue = _make_issue(1706, "Use configured max ticks")
    loop = BossLoop(config=_boss_config(max_iterations=1, dispatch_max_ticks=777))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert dispatch_mock.await_args.kwargs["max_ticks"] == 777


@pytest.mark.asyncio
async def test_dispatch_issue_preserves_issue_header_with_refined_prompt() -> None:
    issue = _make_issue(
        1641,
        "Wire prompt refiner env",
        body=(
            "Pass prompt-refiner file and test hints as worker env vars.\n\n"
            "Acceptance Criteria:\n"
            "- pytest -q tests/swarm/test_boss_loop.py -k refine\n"
        ),
    )
    loop = BossLoop(config=_boss_config(max_iterations=1))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    fake_run = MagicMock()
    fake_run.to_dict.return_value = {
        "status": "completed",
        "run_id": "run-1641",
        "work_orders": [
            {"status": "completed", "branch": "codex/refine-env", "commit_shas": ["abc123"]}
        ],
    }

    refinement = {
        "refined_prompt": "Use the refined goal only.",
        "files_to_change": ["aragora/swarm/boss_loop.py"],
        "test_patterns": ["tests/swarm/test_boss_loop.py"],
        "constraints": [],
        "context_gathered": True,
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(return_value=refinement),
        ),
        patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
    ):
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)
        await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    spec = mock_commander_cls.return_value.run_supervised_from_spec.await_args.args[0]
    assert spec.raw_goal.startswith("[Issue #1641] Wire prompt refiner env")
    assert "Use the refined goal only." in spec.raw_goal


@pytest.mark.asyncio
async def test_ping_pong_retry_prioritizes_same_issue_and_uses_handoff_prompt() -> None:
    issue = _make_issue(1702, "Ping-pong retry")
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [issue]

    loop = BossLoop(
        config=_boss_config(
            max_iterations=2,
            default_target_agent="claude",
            enable_ping_pong_retry=True,
            model_rotation=["claude", "codex"],
        ),
        issue_feed=feed,
        freshness_checker=lambda **kw: _fresh_result(fresh=True),
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": f"{requested_target_agent or 'claude'}-runner-1",
        "runner_type": requested_target_agent or "claude",
    }

    calls: list[dict[str, Any]] = []

    async def _dispatch(spec, **kwargs):
        calls.append(
            {
                "goal": spec.raw_goal,
                "target_agent": kwargs.get("default_target_agent"),
            }
        )
        if len(calls) == 1:
            return {
                "status": "needs_human",
                "reasons": ["Need a second implementation pass"],
                "run": {
                    "work_orders": [
                        {
                            "stdout_tail": (
                                "Found the failure in the boss loop retry path. "
                                "The next agent needs the transcript to finish the fix cleanly."
                            ),
                            "changed_paths": ["aragora/swarm/boss_loop.py"],
                            "target_agent": "claude",
                        }
                    ]
                },
            }
        return {
            "status": "completed",
            "run": {"work_orders": [{"status": "completed", "target_agent": "codex"}]},
        }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec", new=AsyncMock(side_effect=_dispatch)
        ),
    ):
        result = await loop.run()

    assert [status["worker_status"] for status in result.iteration_statuses] == [
        "ping_pong_retry",
        "completed",
    ]
    assert len(calls) == 2
    assert calls[0]["target_agent"] == "claude"
    assert calls[1]["target_agent"] == "codex"
    assert "## Context (from claude, round 1)" in calls[1]["goal"]
    assert "Need a second implementation pass" in calls[1]["goal"]
    assert not loop._pending_handoff_prompts


@pytest.mark.asyncio
async def test_ping_pong_retry_survives_max_retry_cap() -> None:
    issue = _make_issue(1704, "Ping-pong under strict retry cap")
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [issue]

    loop = BossLoop(
        config=_boss_config(
            max_iterations=2,
            max_retries_per_issue=1,
            default_target_agent="claude",
            enable_ping_pong_retry=True,
            model_rotation=["claude", "codex"],
        ),
        issue_feed=feed,
        freshness_checker=lambda **kw: _fresh_result(fresh=True),
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": f"{requested_target_agent or 'claude'}-runner-1",
        "runner_type": requested_target_agent or "claude",
    }

    calls: list[dict[str, Any]] = []

    async def _dispatch(spec, **kwargs):
        calls.append(
            {
                "goal": spec.raw_goal,
                "target_agent": kwargs.get("default_target_agent"),
            }
        )
        if len(calls) == 1:
            return {
                "status": "needs_human",
                "reasons": ["Need one more pass"],
                "run": {
                    "work_orders": [
                        {
                            "stdout_tail": (
                                "First pass isolated the failing edge case. "
                                "A second agent should finish the handoff-based fix."
                            ),
                            "changed_paths": ["aragora/swarm/boss_loop.py"],
                            "target_agent": "claude",
                        }
                    ]
                },
            }
        return {
            "status": "completed",
            "run": {"work_orders": [{"status": "completed", "target_agent": "codex"}]},
        }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec", new=AsyncMock(side_effect=_dispatch)
        ),
    ):
        result = await loop.run()

    assert [status["worker_status"] for status in result.iteration_statuses] == [
        "ping_pong_retry",
        "completed",
    ]
    assert len(calls) == 2
    assert calls[0]["target_agent"] == "claude"
    assert calls[1]["target_agent"] == "codex"
    assert "## Context (from claude, round 1)" in calls[1]["goal"]
    assert not loop._pending_handoff_prompts


@pytest.mark.asyncio
async def test_dispatch_issue_keeps_explicit_validation_commands_when_focused_enabled() -> None:
    issue = _make_issue(
        1640,
        "Keep explicit validation",
        body=(
            "Do the narrow fix.\n\n"
            "Acceptance Criteria:\n"
            "- python -m pytest tests/swarm/test_boss_loop.py tests/swarm/test_spec.py -q\n"
        ),
    )
    loop = BossLoop(config=_boss_config(max_iterations=1))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    fake_run = MagicMock()
    fake_run.to_dict.return_value = {
        "status": "completed",
        "run_id": "run-1640",
        "work_orders": [
            {"status": "completed", "branch": "codex/keep-tests", "commit_shas": ["abc123"]}
        ],
    }

    with (
        patch(
            "aragora.swarm.boss_loop.discover_focused_tests",
            return_value=["tests/swarm/test_other.py"],
        ),
        patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
    ):
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)
        await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    spec = mock_commander_cls.return_value.run_supervised_from_spec.await_args.args[0]
    assert spec.acceptance_criteria == [
        "python -m pytest tests/swarm/test_boss_loop.py tests/swarm/test_spec.py -q"
    ]


@pytest.mark.asyncio
async def test_run_iteration_prioritizes_pending_handoff_issue() -> None:
    issue_a = _make_issue(1801, "Regular issue")
    issue_b = _make_issue(1802, "Pending handoff issue")
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [issue_a, issue_b]

    loop = BossLoop(
        config=_boss_config(max_iterations=1),
        issue_feed=feed,
        freshness_checker=lambda **kw: _fresh_result(fresh=True),
    )
    loop._pending_handoff_prompts[issue_b.number] = ("handoff prompt", "codex")

    seen: list[int] = []

    async def _dispatch_issue(issue, freshness):
        seen.append(issue.number)
        return {"status": "completed"}

    loop._dispatch_issue = _dispatch_issue

    status = await loop._run_iteration(1)

    assert seen == [issue_b.number]
    assert status.selected_issue["number"] == issue_b.number


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
        assert payload["issues_completed"][0]["number"] in {100, 200}
        assert payload["issues_completed"][0]["title"] in {
            "Add retry to aragora/resilience/retry.py",
            "Fix typo in docs",
        }
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


def test_boss_loop_batch_no_issue_skips_runner_freshness_check() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = []
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=2),
        issue_feed=feed,
        freshness_checker=lambda **kw: (_ for _ in ()).throw(
            AssertionError("freshness should not be checked for an empty queue")
        ),
    )

    result = asyncio.run(loop.run())

    assert result.stop_reason == "no_suitable_issue"
    assert result.iterations_completed == 1


# ---------------------------------------------------------------------------
# _classify_terminal_run_outcome regression tests
# ---------------------------------------------------------------------------


class TestClassifyTerminalRunOutcome:
    """Regression tests for deliverable extraction from blocked/reviewable runs."""

    def test_needs_human_with_deliverable_stays_needs_human(self):
        """A blocked/reviewable run may still expose a concrete deliverable.

        The deliverable should remain extractable, but the terminal outcome must
        stay truthful instead of being promoted to unconditional success.
        """
        from aragora.swarm.boss_loop import _classify_terminal_run_outcome

        run_dict = {
            "status": "needs_human",
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "codex/swarm-abc-subtask_1",
                    "commit_shas": ["abc123"],
                },
                {
                    "work_order_id": "wo-2",
                    "status": "needs_human",
                    "branch": "",
                    "commit_shas": [],
                },
            ],
        }
        assert _classify_terminal_run_outcome(run_dict) == "needs_human"

    def test_needs_human_without_deliverable_returns_needs_human(self):
        from aragora.swarm.boss_loop import _classify_terminal_run_outcome

        run_dict = {
            "status": "needs_human",
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "needs_human",
                    "branch": "",
                    "commit_shas": [],
                },
            ],
        }
        assert _classify_terminal_run_outcome(run_dict) == "needs_human"

    def test_dispatch_bounded_spec_includes_deliverable_for_needs_human(self):
        """dispatch_bounded_spec should include the deliverable dict even when
        the outcome would have been needs_human (now reclassified)."""
        from aragora.swarm.boss_loop import _extract_deliverable

        run_dict = {
            "status": "needs_human",
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "codex/swarm-abc-subtask_1",
                    "commit_shas": ["abc123"],
                },
            ],
        }
        deliverable = _extract_deliverable(run_dict)
        assert deliverable is not None
        assert deliverable["type"] == "branch"
        assert deliverable["branch"] == "codex/swarm-abc-subtask_1"
        assert deliverable["commit_shas"] == ["abc123"]


# ---------------------------------------------------------------------------
# Focused test discovery
# ---------------------------------------------------------------------------


class TestDiscoverFocusedTests:
    """Tests for discover_focused_tests — keyed with 'focused'."""

    def test_focused_maps_source_to_test(self, tmp_path, monkeypatch):
        """Source file under aragora/ maps to tests/ mirror."""
        # Create the test file so the existence check passes
        test_dir = tmp_path / "tests" / "swarm"
        test_dir.mkdir(parents=True)
        (test_dir / "test_boss_loop.py").write_text("# test")

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "aragora/swarm/boss_loop.py\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == ["tests/swarm/test_boss_loop.py"]

    def test_focused_includes_changed_test_files(self, tmp_path, monkeypatch):
        """Changed test files under tests/ are included directly."""
        test_dir = tmp_path / "tests" / "swarm"
        test_dir.mkdir(parents=True)
        (test_dir / "test_queue.py").write_text("# test")

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "tests/swarm/test_queue.py\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == ["tests/swarm/test_queue.py"]

    def test_focused_skips_nonexistent_test(self, tmp_path, monkeypatch):
        """When the mapped test file doesn't exist, it is omitted."""

        # Do NOT create the test file
        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "aragora/swarm/boss_loop.py\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_skips_non_python_files(self, tmp_path, monkeypatch):
        """Non-.py files (docs, configs) are ignored."""

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "docs/STATUS.md\nREADME.md\nsetup.cfg\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_deduplicates(self, tmp_path, monkeypatch):
        """Same test file mapped from two source files appears only once."""
        test_dir = tmp_path / "tests" / "swarm"
        test_dir.mkdir(parents=True)
        (test_dir / "test_boss_loop.py").write_text("# test")

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            # Two different source files that map to the same test
            result.stdout = "aragora/swarm/boss_loop.py\ntests/swarm/test_boss_loop.py\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == ["tests/swarm/test_boss_loop.py"]

    def test_focused_returns_empty_on_git_failure(self, tmp_path, monkeypatch):
        """Non-zero git exit code returns empty list gracefully."""

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 128
            result.stdout = ""
            result.stderr = "fatal: not a git repository"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_returns_empty_on_missing_git(self, tmp_path, monkeypatch):
        """FileNotFoundError from git binary returns empty list."""

        def _run(cmd, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_respects_base_ref(self, tmp_path, monkeypatch):
        """Custom base_ref is passed through to git diff."""
        captured: list[list[str]] = []

        def _run(cmd, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        discover_focused_tests(tmp_path, base_ref="origin/develop")
        assert any("origin/develop..HEAD" in c for c in captured[0])

    def test_focused_multiple_source_files(self, tmp_path, monkeypatch):
        """Multiple source files each map to their own test file."""
        (tmp_path / "tests" / "swarm").mkdir(parents=True)
        (tmp_path / "tests" / "cli").mkdir(parents=True)
        (tmp_path / "tests" / "swarm" / "test_boss_loop.py").write_text("# t")
        (tmp_path / "tests" / "cli" / "test_parser.py").write_text("# t")

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "aragora/swarm/boss_loop.py\naragora/cli/parser.py\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_loop.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert "tests/swarm/test_boss_loop.py" in paths
        assert "tests/cli/test_parser.py" in paths
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# Runner heartbeat refresh tests
# ---------------------------------------------------------------------------


class TestRunnerHeartbeatRefresh:
    """Verify that the boss loop refreshes runner heartbeats each iteration."""

    def test_heartbeats_refreshed_each_iteration(self, tmp_path):
        """_refresh_runner_heartbeats updates updated_at so runners stay fresh."""
        import json as _json

        registry_path = str(tmp_path / "runners.json")
        old_ts = "2025-01-01T00:00:00+00:00"
        _json.dump(
            {
                "registrations": [
                    {
                        "runner_id": "codex-runner-1",
                        "runner_type": "codex",
                        "availability": "ready",
                        "available": True,
                        "auth_mode": "api_key",
                        "registered": True,
                        "registered_at": old_ts,
                        "updated_at": old_ts,
                        "heartbeat_at": old_ts,
                        "owner_binding": {
                            "user_id": "u1",
                            "workspace_id": "w1",
                        },
                    }
                ]
            },
            open(registry_path, "w"),
        )

        config = _boss_config(registry_path=registry_path)
        loop = BossLoop(
            config=config,
            env={"ARAGORA_USER_ID": "u1", "ARAGORA_WORKSPACE_ID": "w1"},
        )

        loop._refresh_runner_heartbeats()

        with open(registry_path) as f:
            data = _json.load(f)

        reg = data["registrations"][0]
        assert reg["updated_at"] != old_ts, "updated_at should be refreshed"
        assert reg["heartbeat_at"] != old_ts, "heartbeat_at should be refreshed"

    def test_heartbeat_refresh_called_during_run(self, tmp_path):
        """The main run() loop calls _refresh_runner_heartbeats each iteration."""
        feed = MagicMock()
        feed.fetch.return_value = [_make_issue()]

        config = _boss_config(max_iterations=2)
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        refresh_calls: list[int] = []
        original_refresh = loop._refresh_runner_heartbeats

        def _tracking_refresh():
            refresh_calls.append(1)

        loop._refresh_runner_heartbeats = _tracking_refresh

        async def _complete_dispatch(issue, freshness):
            return {"status": "completed", "deliverable": "done"}

        loop._dispatch_issue = _complete_dispatch

        asyncio.run(loop.run())

        assert len(refresh_calls) == 2, (
            f"Expected 2 heartbeat refreshes (one per iteration), got {len(refresh_calls)}"
        )

    def test_heartbeat_refresh_skipped_without_owner_context(self, tmp_path):
        """Without ARAGORA_USER_ID, heartbeat refresh is a no-op (no crash)."""
        config = _boss_config(registry_path=str(tmp_path / "runners.json"))
        loop = BossLoop(config=config, env={})

        # Should not raise
        loop._refresh_runner_heartbeats()


# ---------------------------------------------------------------------------
# Backbone ledger wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_creates_backbone_entry():
    """Boss loop dispatch should attempt to create a backbone run entry."""
    issue = _make_issue(42, "Backbone wiring test")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    created_run_ids: list[str] = []
    updated_calls: list[dict] = []

    class MockRuntime:
        def create_run(self, ledger):
            created_run_ids.append(ledger.run_id)

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})

    fake_result = {"status": "completed", "run_id": "run-42", "receipt_id": "receipt-42"}

    with (
        patch(
            "aragora.pipeline.backbone_runtime.BackboneRuntime",
            MockRuntime,
        ),
        patch(
            "aragora.pipeline.backbone_contracts.RunLedger",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value=fake_result),
        ),
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    assert len(created_run_ids) == 1
    assert "boss-" in created_run_ids[0]
    assert "issue42" in created_run_ids[0]
    assert len(updated_calls) == 1
    assert updated_calls[0]["status"] == "completed"
    assert updated_calls[0]["execution_id"] == "run-42"
    assert updated_calls[0]["receipt_id"] == "receipt-42"


@pytest.mark.asyncio
async def test_dispatch_backbone_failure_does_not_block():
    """Backbone runtime errors must never prevent dispatch from proceeding."""
    issue = _make_issue(99, "Backbone failure resilience")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    class CrashingRuntime:
        def create_run(self, ledger):
            raise RuntimeError("backbone unavailable")

        def update_run(self, run_id, **kw):
            raise RuntimeError("backbone unavailable")

    fake_result = {"status": "completed", "run_id": "run-99"}

    with (
        patch(
            "aragora.pipeline.backbone_runtime.BackboneRuntime",
            CrashingRuntime,
        ),
        patch(
            "aragora.pipeline.backbone_contracts.RunLedger",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value=fake_result),
        ),
    ):
        # Should NOT raise despite backbone failures
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
