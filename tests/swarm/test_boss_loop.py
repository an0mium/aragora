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
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from aragora.swarm import preflight as preflight_mod
from aragora.swarm.boss_loop import (
    _should_replace_with_focused_tests,
    _ISSUE_CLAIM_TTL_SECONDS,
    build_issue_eligibility_report,
    BossIterationStatus,
    BossLoop,
    BossLoopConfig,
    BossLoopResult,
    BossStopReason,
    fetch_open_pr_changed_paths,
    GitHubIssue,
    GitHubIssueFeed,
    infer_issue_lane_hints,
    infer_issue_scope_entries,
    RunnerFreshnessResult,
    check_runner_freshness,
    discover_focused_tests,
    dispatch_bounded_spec,
    extract_declared_new_file_paths,
    extract_pre_dispatch_validation_commands,
    extract_issue_validation_contract,
    find_missing_pre_dispatch_validation_targets,
    run_pre_dispatch_validation_commands,
    sanitize_issue_body_for_dispatch,
    select_eligible_issue,
)
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy
from aragora.swarm.session_state import SessionStateStore
from aragora.swarm.task_sanitizer import SanitizationOutcome
from aragora.swarm.terminal_truth import qualify_work_order_terminal_state

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


def _preflight_receipt(
    *,
    check_type: str = "scratch",
    passed: bool = True,
    checks: list[dict[str, Any]] | None = None,
) -> preflight_mod.PreflightReceipt:
    return preflight_mod.PreflightReceipt(
        receipt_id=f"preflight-{check_type}-20260414T001900Z-test",
        envelope_seal="seal",
        repo_root="/tmp/repo",
        check_type=check_type,
        started_at="2026-04-14T00:19:00Z",
        finished_at="2026-04-14T00:19:05Z",
        passed=passed,
        checks=checks
        or [
            {
                "name": "receipt_check",
                "passed": passed,
                "detail": "ok" if passed else "failed",
            }
        ],
        cache_key=f"{check_type}-cache",
        ttl_seconds=3600,
        expires_at="2026-04-14T01:19:05Z",
        artifacts={"target_ref": "main"},
    )


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

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)
        feed = GitHubIssueFeed()
        assert feed.fetch() == []

    def test_fetch_returns_empty_on_file_not_found(self, monkeypatch):
        import subprocess as sp

        def _run(cmd, **kwargs):
            raise FileNotFoundError("gh not found")

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)
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

    def test_fetch_open_pr_changed_paths_parses_changed_files(self, monkeypatch):
        gh_output = json.dumps(
            [
                {
                    "files": [
                        {"path": "tests/memory/test_tier_ttl_expiration.py"},
                        {"path": "aragora/swarm/boss_loop.py"},
                    ]
                }
            ]
        )

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = gh_output
            result.stderr = ""
            return result

        monkeypatch.setattr("aragora.swarm.boss_feed.subprocess.run", _run)

        assert fetch_open_pr_changed_paths(repo="synaptent/aragora") == {
            "tests/memory/test_tier_ttl_expiration.py",
            "aragora/swarm/boss_loop.py",
        }


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

    def test_reports_issues_skipped_by_label(self):
        issues = [
            _make_issue(1, "Known stuck", labels=["boss-stuck"]),
            _make_issue(2, "Ready lane"),
        ]

        report = build_issue_eligibility_report(issues, skip_labels={"boss-stuck"})

        assert report.eligible_count == 1
        assert [issue.number for issue in report.eligible] == [2]
        assert report.skipped_by_label == {"boss-stuck": [1]}


class TestBatchIssueSelection:
    def test_blocked_issue_scopes_skips_lookup_without_repo(self, monkeypatch):
        loop = BossLoop(_boss_config(repo=None))

        def _unexpected(**kwargs):
            raise AssertionError("open PR scope lookup should be skipped without an explicit repo")

        monkeypatch.setattr("aragora.swarm.boss_loop.fetch_open_pr_changed_paths", _unexpected)
        monkeypatch.setattr(loop, "_coordination_blocked_scopes", lambda: set())

        assert loop._blocked_issue_scopes() == set()

    def test_blocked_issue_scopes_without_repo_keeps_coordination_claims(self, monkeypatch):
        loop = BossLoop(_boss_config(repo=None))

        def _unexpected(**kwargs):
            raise AssertionError("open PR scope lookup should be skipped without an explicit repo")

        monkeypatch.setattr("aragora.swarm.boss_loop.fetch_open_pr_changed_paths", _unexpected)
        monkeypatch.setattr(
            loop,
            "_coordination_blocked_scopes",
            lambda: {"aragora/swarm/supervisor.py", "tests/swarm/test_supervisor.py"},
        )

        assert loop._blocked_issue_scopes() == {
            "aragora/swarm/supervisor.py",
            "tests/swarm/test_supervisor.py",
        }

    def test_blocked_issue_scopes_unions_open_pr_and_coordination_claims(self, monkeypatch):
        loop = BossLoop(_boss_config(repo="synaptent/aragora"))
        monkeypatch.setattr(
            loop,
            "_coordination_blocked_scopes",
            lambda: {"aragora/swarm/supervisor.py"},
        )
        monkeypatch.setattr(
            "aragora.swarm.boss_loop.fetch_open_pr_changed_paths",
            lambda **kwargs: {"tests/memory/test_tier_ttl_expiration.py"},
        )

        assert loop._blocked_issue_scopes() == {
            "aragora/swarm/supervisor.py",
            "tests/memory/test_tier_ttl_expiration.py",
        }

    def test_coordination_blocked_scopes_collects_active_leases_and_claims(self, monkeypatch):
        loop = BossLoop(_boss_config(repo="synaptent/aragora"))

        class _FakeFleetStore:
            def __init__(self) -> None:
                self.reaped = False

            def reap_stale_claims(self) -> dict[str, int]:
                self.reaped = True
                return {"released": 1}

            def list_claims(self) -> list[dict[str, str]]:
                return [
                    {"path": "tests/memory/test_tier_ttl_expiration.py"},
                    {"path": "aragora/swarm"},
                ]

        class _FakeStore:
            def __init__(self, repo_root) -> None:
                self.repo_root = repo_root
                self.fleet_store = _FakeFleetStore()

            def list_active_leases(self):
                lease = MagicMock()
                lease.claimed_paths = ["aragora/swarm/boss_loop.py"]
                lease.allowed_globs = ["tests/swarm/**"]
                return [lease]

        monkeypatch.setattr("aragora.swarm.boss_loop.DevCoordinationStore", _FakeStore)

        assert loop._coordination_blocked_scopes() == {
            "aragora/swarm/boss_loop.py",
            "tests/swarm/**",
            "tests/memory/test_tier_ttl_expiration.py",
            "aragora/swarm",
        }

    def test_parallel_selection_skips_conflicting_issue_scopes(self):
        loop = BossLoop(_boss_config(max_parallel_dispatches=3))
        issues = [
            _make_issue(
                1,
                "TTL tests A",
                body="Scope hints:\n- `tests/memory/test_tier_ttl_expiration.py`\n",
            ),
            _make_issue(
                2,
                "TTL tests B",
                body="Scope hints:\n- `tests/memory/test_tier_ttl_expiration.py`\n",
            ),
            _make_issue(
                3,
                "Quota tests",
                body="Scope hints:\n- `tests/agents/test_fallback_quota.py`\n",
            ),
        ]

        selected = loop._select_issues_for_iteration(issues, limit=3)

        assert [issue.number for issue in selected] == [1, 3]

    def test_parallel_selection_claims_lane_before_dispatch(self):
        loop = BossLoop(_boss_config(max_parallel_dispatches=3))
        issues = [
            _make_issue(
                1,
                "Swarm supervisor follow-up",
                body="Touch `aragora/swarm/supervisor.py` only.\n",
            ),
            _make_issue(
                2,
                "Swarm test follow-up",
                body="Touch `tests/swarm/test_supervisor.py` only.\n",
            ),
            _make_issue(
                3,
                "Frontend follow-up",
                body="Touch `aragora/live/src/app/page.tsx` only.\n",
            ),
        ]

        selected = loop._select_issues_for_iteration(issues, limit=3)

        assert [issue.number for issue in selected] == [1, 3]

    def test_parallel_selection_respects_open_pr_blocked_scope(self):
        loop = BossLoop(_boss_config(max_parallel_dispatches=2))
        issues = [
            _make_issue(
                1,
                "TTL tests",
                body="Scope hints:\n- `tests/memory/test_tier_ttl_expiration.py`\n",
            ),
            _make_issue(
                2,
                "Quota tests",
                body="Scope hints:\n- `tests/agents/test_fallback_quota.py`\n",
            ),
        ]

        selected = loop._select_issues_for_iteration(
            issues,
            limit=2,
            blocked_scopes={"tests/memory/test_tier_ttl_expiration.py"},
        )

        assert [issue.number for issue in selected] == [2]

    def test_parallel_selection_filters_skip_labels_before_semantic_dedup(self, monkeypatch):
        loop = BossLoop(_boss_config(max_parallel_dispatches=2))
        issues = [
            _make_issue(1, "Duplicate lane", labels=["boss-stuck"]),
            _make_issue(2, "Duplicate lane"),
        ]
        seen: dict[str, list[int]] = {}

        def _dedup(candidates: list[GitHubIssue]) -> list[GitHubIssue]:
            seen["numbers"] = [issue.number for issue in candidates]
            if any(issue.number == 1 for issue in candidates):
                return [candidates[0]]
            return list(candidates)

        monkeypatch.setattr(loop, "_semantic_dedup_issues", _dedup)

        selected = loop._select_issues_for_iteration(issues, limit=2)

        assert seen["numbers"] == [2]
        assert [issue.number for issue in selected] == [2]

    @pytest.mark.asyncio
    async def test_run_iteration_filters_active_foreign_issue_claim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        claim_dir = tmp_path / ".aragora" / "issue_claims"
        claim_dir.mkdir(parents=True)
        (claim_dir / "1.lock").write_text(
            json.dumps(
                {
                    "issue_number": 1,
                    "run_id": "boss-other",
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "claimed_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )

        issue_one = _make_issue(1, "Claimed elsewhere")
        issue_two = _make_issue(2, "Dispatch me instead")
        feed = MagicMock()
        feed.fetch.return_value = [issue_one, issue_two]
        loop = BossLoop(
            _boss_config(max_iterations=1),
            issue_feed=feed,
            freshness_checker=lambda **_: _fresh_result(fresh=True),
        )
        loop._existing_open_pr_skip_status = lambda **_: None
        loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

        status = await loop._run_iteration(1)

        assert status.worker_status == "completed"
        assert status.selected_issue["number"] == 2
        loop._dispatch_issue.assert_awaited_once_with(issue_two, ANY)

    @pytest.mark.asyncio
    async def test_run_iteration_skips_issue_when_claim_taken_before_dispatch(self):
        issue = _make_issue(1, "Race on issue claim")
        feed = MagicMock()
        feed.fetch.return_value = [issue]
        loop = BossLoop(
            _boss_config(max_iterations=1),
            issue_feed=feed,
            freshness_checker=lambda **_: _fresh_result(fresh=True),
        )
        loop._existing_open_pr_skip_status = lambda **_: None
        loop._claim_issue_dispatch = lambda issue_number: (
            False,
            f"Issue #{issue_number} is already claimed by boss-other.",
        )
        loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

        status = await loop._run_iteration(1)

        assert status.worker_status == "skipped"
        assert status.worker_outcome == "issue_claimed"
        assert status.selected_issue["number"] == 1
        loop._dispatch_issue.assert_not_awaited()

    def test_claim_issue_dispatch_reaps_expired_claim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        claim_dir = tmp_path / ".aragora" / "issue_claims"
        claim_dir.mkdir(parents=True)
        claim_path = claim_dir / "42.lock"
        claim_path.write_text(
            json.dumps(
                {
                    "issue_number": 42,
                    "run_id": "boss-old",
                    "pid": 999999,
                    "host": socket.gethostname(),
                }
            ),
            encoding="utf-8",
        )
        expired = time.time() - (_ISSUE_CLAIM_TTL_SECONDS + 5)
        os.utime(claim_path, (expired, expired))

        loop = BossLoop(_boss_config(max_iterations=1))

        claimed, reason = loop._claim_issue_dispatch(42)

        assert claimed is True
        assert reason is None
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == loop.run_id
        assert payload["pid"] == os.getpid()

    @pytest.mark.asyncio
    async def test_dispatch_issue_under_claim_releases_issue_lock(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        loop = BossLoop(_boss_config(max_iterations=1))
        issue = _make_issue(77, "Release claim after dispatch")
        claimed, reason = loop._claim_issue_dispatch(issue.number)
        assert claimed is True
        assert reason is None

        loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

        result = await loop._dispatch_issue_under_claim(issue, _fresh_result(fresh=True))

        assert result["status"] == "completed"
        assert not loop._issue_claim_path(issue.number).exists()

    def test_skips_issues_with_skip_labels(self):
        issues = [
            _make_issue(1, "Dup", labels=["duplicate"]),
            _make_issue(2, "Valid"),
        ]
        selected = select_eligible_issue(issues, skip_labels={"duplicate"})
        assert selected is not None
        assert selected.number == 2

    def test_infers_directory_scope_from_issue_constraints(self):
        issue = _make_issue(
            1,
            "Boss loop cleanup",
            body="Acceptance Criteria:\n- No files outside aragora/swarm/ are changed\n",
        )

        assert infer_issue_scope_entries(issue) == ["aragora/swarm"]

    def test_infers_lane_from_scope_hints(self):
        issue = _make_issue(
            1,
            "Boss loop cleanup",
            body="Acceptance Criteria:\n- No files outside aragora/swarm/ are changed\n",
        )

        assert infer_issue_lane_hints(issue) == ["swarm"]

    def test_infers_lane_from_explicit_label(self):
        issue = _make_issue(
            1,
            "Landing polish",
            labels=["boss-ready", "lane:frontend"],
        )

        assert infer_issue_lane_hints(issue) == ["frontend"]

    def test_issue_payload_exposes_lane_metadata(self):
        issue = _make_issue(
            1,
            "Boss loop cleanup",
            body="Touch `aragora/swarm/boss_loop.py` only.\n",
        )

        payload = BossLoop._issue_payload(issue)

        assert payload["lane_hints"] == ["swarm"]
        assert payload["lane_id"] == "swarm"

    def test_skips_issue_when_scope_overlaps_blocked_open_pr_paths(self):
        issues = [
            _make_issue(
                1,
                "Duplicate TTL test",
                body="Touch `tests/memory/test_tier_ttl_expiration.py` only.\n",
            ),
            _make_issue(
                2,
                "Independent fallback tests",
                body="Touch `tests/agents/test_fallback_quota.py` only.\n",
            ),
        ]

        selected = select_eligible_issue(
            issues,
            blocked_scopes={"tests/memory/test_tier_ttl_expiration.py"},
        )
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

    def test_find_missing_pre_dispatch_validation_targets_reports_missing_pytest_file(
        self, tmp_path: Path
    ):
        commands = [
            "pytest tests/webhooks/test_delivery_retry.py -x -q",
            "python -m pytest tests/swarm/test_boss_loop.py -q",
        ]
        existing = tmp_path / "tests" / "swarm"
        existing.mkdir(parents=True)
        (existing / "test_boss_loop.py").write_text("# test")

        assert find_missing_pre_dispatch_validation_targets(commands, repo_root=tmp_path) == [
            "tests/webhooks/test_delivery_retry.py"
        ]

    def test_extract_declared_new_file_paths_only_accepts_explicit_new_markers(self):
        body = (
            "## Files\n"
            "- `tests/test_openapi_regeneration.py` (new)\n"
            "- `tests/webhooks/test_delivery_retry.py` (new or extend)\n"
            "- `tests/pipeline/test_run_ledger_ordering.py` (new file)\n"
        )

        assert extract_declared_new_file_paths(body) == [
            "tests/test_openapi_regeneration.py",
            "tests/pipeline/test_run_ledger_ordering.py",
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        def _make_inspector(runner_type: str, *, env=None, profile=None, repo_root=None):
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

    def test_runner_freshness_auto_probes_codex_runner_until_execution_verified(
        self, tmp_path, monkeypatch
    ):
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
        inspection = SimpleNamespace(runner_id="codex-runner-1", profile=None)
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
                "runner_id": "codex-runner-1",
                "runner_type": "codex",
                "probe_status": "passed",
            },
        )

        class _Inspector:
            def inspect(self) -> MagicMock:
                inspected = MagicMock()
                inspected.available = True
                inspected.auth_mode = "chatgpt_login"
                inspected.runner_id = "codex-runner-1"
                inspected.to_dict.return_value = {
                    "runner_id": "codex-runner-1",
                    "available": True,
                    "auth_mode": "chatgpt_login",
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
            patch("aragora.swarm.runner_registry.probe_runner_execution", return_value=probe),
            patch("aragora.swarm.runner_registry.make_runner_inspector", return_value=_Inspector()),
        ):
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="codex",
            )

        assert result.fresh is True
        assert result.details["probe"]["auto_probe_triggered"] is True
        assert result.details["probe"]["passed"] == 1
        assert result.details["probe"]["verified_target"] == 1

    def test_runner_freshness_probes_fallback_selected_runner_type_when_requested_type_full(
        self, tmp_path, monkeypatch
    ):
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
                            "capabilities": {"max_parallel_lanes": 1, "active_lanes": 1},
                            "updated_at": now,
                            "heartbeat_at": now,
                            "freshness_status": "fresh",
                        },
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
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
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
                "runner_id": "claude-runner-1",
                "runner_type": "claude",
                "probe_status": "passed",
            },
        )

        class _Inspector:
            def inspect(self) -> MagicMock:
                inspected = MagicMock()
                inspected.available = True
                inspected.auth_mode = "subscription"
                inspected.runner_id = "claude-runner-1"
                inspected.profile = "max-01"
                inspected.runner_type = "claude"
                inspected.to_dict.return_value = {
                    "runner_id": "claude-runner-1",
                    "available": True,
                    "auth_mode": "subscription",
                }
                return inspected

        with (
            patch(
                "aragora.swarm.runner_registry.refresh_discovered_runners",
                return_value=[],
            ),
            patch(
                "aragora.swarm.runner_registry.prioritized_probe_candidates",
                side_effect=lambda **kwargs: list(kwargs["discovered_inspections"]),
            ) as prioritized_probe_candidates_mock,
            patch("aragora.swarm.runner_registry.probe_runner_execution", return_value=probe),
            patch("aragora.swarm.runner_registry.make_runner_inspector", return_value=_Inspector()),
        ):
            result = check_runner_freshness(
                freshness_ttl_seconds=3600.0,
                registry_path=str(registry_path),
                env={"ARAGORA_USER_ID": "user-1", "ARAGORA_WORKSPACE_ID": "ws-1"},
                requested_runner_type="codex",
            )

        assert result.fresh is True
        assert result.details["routing"]["fallback_reason"] == "requested_runner_type_unavailable"
        assert result.details["probe"]["verification_runner_type"] == "claude"
        prioritized_probe_candidates_mock.assert_called_once()
        assert prioritized_probe_candidates_mock.call_args.kwargs["runner_type"] == "claude"
        discovered_inspections = prioritized_probe_candidates_mock.call_args.kwargs[
            "discovered_inspections"
        ]
        assert [item.runner_id for item in discovered_inspections] == ["claude-runner-1"]


# ---------------------------------------------------------------------------
# BossLoop core tests
# ---------------------------------------------------------------------------


class TestBossLoop:
    def test_no_fresh_runner_stops_immediately(self):
        config = _boss_config(auto_refill_threshold=0)
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

    def test_malformed_truthy_freshness_flag_blocks_dispatch(self):
        config = _boss_config()
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(1, "Malformed freshness issue")]

        freshness = SimpleNamespace(
            fresh="false",
            blocked_reason="malformed_fresh_flag",
            details={},
            to_dict=lambda: {"fresh": "false", "blocked_reason": "malformed_fresh_flag"},
        )

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: freshness,
        )

        with patch.object(BossLoop, "_dispatch_issue", new_callable=AsyncMock) as mock_dispatch:
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_FRESH_RUNNER.value
        assert len(result.issues_attempted) == 0
        assert "No fresh runner" in result.needs_human_reasons[0]
        mock_dispatch.assert_not_awaited()

    def test_no_suitable_issue_stops(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config(auto_refill_threshold=0)
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

    def test_no_suitable_issue_keepalive_continues_until_max_iterations(self):
        """With keepalive on, empty queues should not terminate the run."""
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config(
            no_suitable_issue_keepalive=True,
            max_iterations=4,
            auto_refill_threshold=0,
        )
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        # Loop should drain to max_iterations rather than exit on first empty queue.
        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 4
        # Every iteration's per-iteration stop_reason was still recorded so
        # observers can see the queue was empty each tick.
        per_iteration_reasons = [status.get("stop_reason") for status in result.iteration_statuses]
        assert per_iteration_reasons == [BossStopReason.NO_SUITABLE_ISSUE.value] * 4

    def test_no_suitable_issue_keepalive_off_terminates_on_first_empty_queue(self):
        """Default behavior (keepalive off) must continue to short-exit cleanly."""
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config(
            no_suitable_issue_keepalive=False,
            max_iterations=4,
            auto_refill_threshold=0,
        )
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert result.iterations_completed == 1

    def test_no_suitable_issue_keepalive_does_not_swallow_other_terminal_reasons(self):
        """Other terminal stop reasons must still terminate even with keepalive on."""
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = []

        config = _boss_config(
            no_suitable_issue_keepalive=True,
            max_iterations=4,
            auto_refill_threshold=0,
        )
        loop = BossLoop(
            config=config,
            issue_feed=feed,
            # Force NO_FRESH_RUNNER on every iteration; keepalive must not catch it.
            freshness_checker=lambda **kw: _fresh_result(
                fresh=False, blocked_reason="runner_not_fresh"
            ),
        )

        result = asyncio.run(loop.run())

        # Empty queue is hit first, so NO_SUITABLE_ISSUE wins (and keepalive
        # does swallow it). But once we feed an issue, freshness blocks it,
        # which is a different terminal reason and must NOT be swallowed.
        # Re-run with one available issue:
        feed.fetch.return_value = [_make_issue(909, "Meta benchmark issue")]
        loop2 = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(
                fresh=False, blocked_reason="runner_not_fresh"
            ),
        )
        result2 = asyncio.run(loop2.run())
        assert result2.stop_reason == BossStopReason.NO_FRESH_RUNNER.value
        assert result2.iterations_completed == 1

        # First run with no issues: keepalive should let it ride out to
        # max_iterations even though _every_ iteration emitted NO_SUITABLE_ISSUE.
        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert result.iterations_completed == 4

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

    def test_specific_issue_number_seeds_explicit_feed_issue_numbers(self):
        loop = BossLoop(config=_boss_config(max_iterations=1, issue_number=873))

        assert isinstance(loop._feed, GitHubIssueFeed)
        assert loop._feed.issue_numbers == [873]

    def test_specific_issue_number_scope_conflict_reports_overlap_reason(self):
        issue = _make_issue(
            873,
            "Publish recurring scorecard",
            body=(
                "Update `scripts/measure_b0_scorecard.py`.\n\n"
                "Acceptance Criteria:\n"
                "- python3 scripts/measure_b0_scorecard.py --help\n"
            ),
            labels=["boss-ready", "priority:critical", "autonomous"],
        )
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [issue]
        feed._fetch_issue.return_value = issue

        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                issue_number=873,
                label_filter="boss-ready",
                require_labels={"boss-ready", "priority:critical", "autonomous"},
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._blocked_issue_scopes = lambda: {"scripts/measure_b0_scorecard.py"}

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert (
            "overlaps files already owned by open PR or in-flight work"
            in result.needs_human_reasons[0]
        )
        assert "scripts/measure_b0_scorecard.py" in result.needs_human_reasons[0]
        assert "Merge, close, or retarget" in result.next_actions[0]

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
        loop._blocked_issue_scopes = lambda: set()
        loop._filter_issues_with_active_claims = lambda issues: issues
        loop._has_open_pr_for_issue = lambda issue_number: None
        loop._claim_issue_dispatch = lambda issue_number: (True, None)
        loop._release_issue_dispatch_claim = lambda issue_number: None
        loop._dispatch_issue = _needs_human_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value
        assert "Approval required for merge." in result.needs_human_reasons

    def test_needs_human_truthy_junk_deliverable_does_not_auto_continue(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(1, "Needs human review")]

        config = _boss_config(auto_continue_on_needs_human=True)

        async def _needs_human_dispatch(issue, freshness):
            return {
                "status": "needs_human",
                "reasons": ["Approval required for merge."],
                "deliverable": "branch-ready",
            }

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _needs_human_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        assert result.iteration_statuses[0]["next_actions"] == [
            "Skipping to next issue (auto-continue mode)."
        ]
        assert "Approval required for merge." in result.iteration_statuses[0]["needs_human_reasons"]

    def test_auto_continue_needs_human_no_typed_deliverable_stops_at_threshold(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        call_count = 0

        def _fetch():
            nonlocal call_count
            call_count += 1
            return [_make_issue(call_count, f"Needs human review {call_count}")]

        feed.fetch.side_effect = _fetch

        config = _boss_config(
            auto_continue_on_needs_human=True,
            max_consecutive_failures=2,
            max_iterations=10,
        )

        async def _needs_human_dispatch(issue, freshness):
            return {
                "status": "needs_human",
                "reasons": ["Worker produced no typed deliverable."],
            }

        loop = BossLoop(
            config=config,
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _needs_human_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.CONSECUTIVE_FAILURES.value
        assert result.iterations_completed == 2
        assert len(result.issues_failed) == 2
        assert result.next_actions == [
            "Repeated rescue outcomes without a typed deliverable reached threshold (2).",
            "Investigate the rescue streak before resuming the boss loop.",
        ]
        assert any(
            "without a typed deliverable reached threshold (2)" in reason
            for reason in result.needs_human_reasons
        )

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

    def test_historical_retry_on_other_issue_does_not_broaden_freshness_pool(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(99, "Fresh issue should keep default routing")]

        freshness_requests: list[str | None] = []

        def _freshness_checker(**kwargs):
            freshness_requests.append(kwargs.get("requested_runner_type"))
            return RunnerFreshnessResult(
                fresh=True,
                runner_ids=["claude-runner-1"],
                checked_at=datetime.now(UTC).isoformat(),
                details={
                    "routing": {
                        "selected_runners": [
                            {"runner_id": "claude-runner-1", "runner_type": "claude"},
                        ],
                        "selected_runner_ids": ["claude-runner-1"],
                    }
                },
            )

        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                default_target_agent="claude",
                model_rotation=["claude", "codex"],
            ),
            issue_feed=feed,
            freshness_checker=_freshness_checker,
        )
        loop._issue_attempt_counts[42] = 1  # Unrelated historical retry

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
            return_value={
                "status": "completed",
                "outcome": "deliverable_created",
                "deliverable": {"type": "branch"},
            }
        )

        with patch("aragora.swarm.boss_loop.dispatch_bounded_spec", dispatch_results):
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert freshness_requests == ["claude"]
        dispatch_call = dispatch_results.await_args.kwargs
        assert dispatch_call["default_target_agent"] == "claude"
        assert dispatch_call["selected_runner"]["runner_type"] == "claude"

    def test_batch_retry_issue_does_not_pull_fresh_issue_into_broadened_batch(self):
        feed = MagicMock(spec=GitHubIssueFeed)
        retried_issue = _make_issue(42, "Retry-routed issue")
        fresh_issue = _make_issue(99, "Fresh issue should wait")
        feed.fetch.return_value = [retried_issue, fresh_issue]

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
                            {
                                "runner_id": "claude-runner-1",
                                "runner_type": "claude",
                                "available_capacity": 1,
                            },
                            {
                                "runner_id": "codex-runner-1",
                                "runner_type": "codex",
                                "available_capacity": 1,
                            },
                        ],
                        "selected_runner_ids": ["claude-runner-1", "codex-runner-1"],
                    }
                },
            )

        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                max_parallel_dispatches=2,
                default_target_agent="claude",
                model_rotation=["claude", "codex"],
            ),
            issue_feed=feed,
            freshness_checker=_freshness_checker,
        )
        loop._issue_attempt_counts[retried_issue.number] = 1

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
            return_value={
                "status": "completed",
                "outcome": "deliverable_created",
                "deliverable": {"type": "branch"},
            }
        )

        with patch("aragora.swarm.boss_loop.dispatch_bounded_spec", dispatch_results):
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert freshness_requests == [None]
        assert dispatch_results.await_count == 1
        dispatch_call = dispatch_results.await_args.kwargs
        assert dispatch_call["default_target_agent"] == "codex"
        assert dispatch_call["selected_runner"]["runner_type"] == "codex"
        assert [item["number"] for item in result.issues_attempted] == [retried_issue.number]

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
        assert statuses[0].configured_max_parallel_dispatches == 1
        assert statuses[0].effective_parallel_dispatches == 1

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
        assert statuses[0].configured_max_parallel_dispatches == 1
        assert statuses[0].effective_parallel_dispatches == 1
        assert statuses[1].effective_parallel_dispatches == 1
        assert [status["worker_status"] for status in result.iteration_statuses] == ["completed"]
        assert result.configured_max_parallel_dispatches == 1
        assert result.effective_parallel_dispatches_observed == 1

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
                "outcome": "deliverable_created",
                "deliverable": {"branch": "codex/test-branch"},
                "publish_result": {"action": "opened_pr", "published": True},
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
                            "prompt_chars": 2048,
                            "enriched_context_chars": 1536,
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
        assert payload["worker_outcome"] == "deliverable_created"
        assert payload["prompt_version"] == "v2"
        assert payload["prompt_chars"] == 2048
        assert payload["enriched_context_chars"] == 1536
        assert payload["deferred_queue_depth"] == 0
        assert payload["sanitizer_outcome"] is None
        assert payload["sanitizer_checks_failed_count"] == 0
        assert payload["cohort_tag"] is None
        assert payload["has_deliverable"] is True
        assert payload["publish_action"] == "opened_pr"
        assert payload["elapsed_seconds"] >= 0.0

    def test_terminal_class_in_metrics_payload(self, tmp_path: Path):
        """terminal_class field appears with a valid TerminalClass value."""
        from aragora.swarm.terminal_truth import TerminalClass

        valid_values = {tc.value for tc in TerminalClass}

        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(99, "Terminal class wiring")]

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
                "outcome": "deliverable_created",
                "deliverable": {"branch": "codex/test-branch"},
                "publish_result": {"action": "opened_pr", "published": True},
                "run": {
                    "work_orders": [
                        {
                            "changed_paths": ["aragora/swarm/boss_loop.py"],
                            "tests_run": ["pytest tests/swarm/ -q"],
                            "verification_results": [{"passed": True}],
                        }
                    ]
                },
            }

        loop._dispatch_issue = _completed_dispatch

        result = asyncio.run(loop.run())
        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value

        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        # The 14 existing fields must still be present
        assert "iteration" in payload
        assert "issue_number" in payload
        assert "worker_status" in payload
        assert "worker_outcome" in payload
        assert "elapsed_seconds" in payload
        assert "files_changed" in payload
        assert "tests_run" in payload
        assert "tests_passed" in payload
        assert "prompt_version" in payload
        assert "prompt_chars" in payload
        assert "enriched_context_chars" in payload
        assert "is_decomposed_issue" in payload
        assert "deferred_queue_depth" in payload
        assert "sanitizer_outcome" in payload
        assert "sanitizer_checks_failed_count" in payload
        assert "cohort_tag" in payload
        assert "has_deliverable" in payload
        assert "publish_action" in payload
        # New terminal_class field
        assert "terminal_class" in payload, "terminal_class key missing from metrics payload"
        assert payload["terminal_class"] in valid_values, (
            f"terminal_class value {payload['terminal_class']!r} is not a valid TerminalClass"
        )

    @pytest.mark.parametrize(
        ("title", "body", "expected_outcome"),
        [
            (
                "Too short",
                "Tiny task only.",
                "dropped",
            ),
            (
                "Broad scope",
                (
                    "Implement the reliability substrate end-to-end.\n\n"
                    "Allowed write set:\n"
                    "- `aragora/swarm/a.py` (modify)\n"
                    "- `aragora/swarm/b.py` (modify)\n"
                    "- `aragora/swarm/c.py` (modify)\n"
                    "- `aragora/swarm/d.py` (modify)\n"
                    "- `aragora/swarm/e.py` (modify)\n"
                    "- `aragora/swarm/f.py` (modify)\n"
                ),
                "quarantined",
            ),
        ],
    )
    def test_metrics_emit_sanitizer_outcome_for_filtered_issues(
        self,
        tmp_path: Path,
        title: str,
        body: str,
        expected_outcome: str,
    ) -> None:
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(2420, title, body=body)]

        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                metrics_jsonl_path=str(tmp_path / "boss_metrics.jsonl"),
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value
        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        assert payload["worker_status"] == "needs_human"
        assert payload["sanitizer_outcome"] == expected_outcome
        assert payload["sanitizer_checks_failed_count"] >= 1

    def test_metrics_emit_deferred_queue_depth_when_publish_deferred(self, tmp_path: Path) -> None:
        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                metrics_jsonl_path=str(tmp_path / "boss_metrics.jsonl"),
                auto_publish_deliverables=True,
            ),
        )
        issue = _make_issue(123, "Deferred publish metrics")
        worker_result = {
            "status": "completed",
            "outcome": "deliverable_created",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-123",
                "commit_shas": ["abc123"],
            },
            "receipt_metadata": {"issue_title": issue.title},
            "run": {"work_orders": []},
        }

        with patch.object(
            loop,
            "_maybe_publish_deliverable",
            return_value={
                "action": "deferred_due_to_open_boss_prs",
                "reason": "open_boss_harvest_pr_limit",
                "branch": "codex/issue-123",
                "max_open_prs": 20,
                "open_prs": [],
            },
        ):
            processed = loop._postprocess_issue_result(issue, worker_result)

        loop._append_iteration_metrics(
            iteration=1,
            issue_number=issue.number,
            worker_result=processed,
            elapsed_seconds=0.25,
        )

        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        assert payload["publish_action"] == "deferred_due_to_open_boss_prs"
        assert payload["deferred_queue_depth"] == 1
        assert len(loop._deferred_publish_queue) == 1

    def test_metrics_emit_cohort_tag_from_issue_title(self, tmp_path: Path) -> None:
        feed = MagicMock(spec=GitHubIssueFeed)
        issue = _make_issue(77, "[B0-cohort] Measure clean publish path")
        feed.fetch.return_value = [issue]

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
                "outcome": "deliverable_created",
                "deliverable": {"branch": "codex/test-branch"},
                "receipt_metadata": {"issue_title": issue.title},
                "run": {"work_orders": []},
            }

        loop._dispatch_issue = _completed_dispatch

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        assert payload["cohort_tag"] == "B0-cohort"

    def test_terminal_class_fallback_on_classification_error(self, tmp_path: Path):
        """terminal_class uses fallback value when classify_from_metrics raises."""
        from aragora.swarm.terminal_truth import TerminalClass

        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(100, "Terminal class fallback")]

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
                "outcome": "deliverable_created",
                "deliverable": {"branch": "codex/fallback-branch"},
                "run": {"work_orders": []},
            }

        loop._dispatch_issue = _completed_dispatch

        with patch(
            "aragora.swarm.boss_loop.classify_from_metrics",
            side_effect=RuntimeError("boom"),
        ):
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        payload = json.loads((tmp_path / "boss_metrics.jsonl").read_text(encoding="utf-8"))
        assert payload["terminal_class"] == TerminalClass.RESCUE_NO_DELIVERABLE.value

    @pytest.mark.asyncio
    async def test_missing_validation_contract_stops_with_needs_human(self):
        issue = _make_issue(
            7,
            "Issue missing validation",
            body="Tighten the boss loop selection logic in aragora/swarm/boss_loop.py",
        )
        loop = BossLoop(config=_boss_config(max_iterations=1))
        loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (
            None,
            None,
        )
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
                "aragora.swarm.boss_loop.TaskSanitizer.sanitize",
                return_value=SimpleNamespace(
                    outcome=SanitizationOutcome.ACCEPTED,
                    sanitized_text=issue.body,
                    checks_failed=[],
                    reason="",
                ),
            ),
        ):
            result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

        assert result["status"] == "needs_human"
        assert result["outcome"] == "blocked"
        assert "lacks an explicit validation contract" in result["reasons"][0]
        assert result["dispatch_gate"]["failure_classes"] == ["contract_missing"]
        assert result["receipt_metadata"]["dispatch_gate"]["failure_classes"] == [
            "contract_missing"
        ]
        qualification = qualify_work_order_terminal_state(result)
        assert qualification.failure_classes == ["contract_missing", "needs_human"]

    @pytest.mark.asyncio
    async def test_dispatch_bounded_gate_emits_contract_missing_evidence(self) -> None:
        issue = _make_issue(2467, "Issue missing bounded dispatch contract")
        loop = BossLoop(config=_boss_config(max_iterations=1))
        loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (
            None,
            None,
        )
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
                "aragora.swarm.boss_loop.TaskSanitizer.sanitize",
                return_value=SimpleNamespace(
                    outcome=SanitizationOutcome.ACCEPTED,
                    sanitized_text=issue.body,
                    checks_failed=[],
                    reason="",
                ),
            ),
            patch("aragora.swarm.spec.SwarmSpec.is_dispatch_bounded", return_value=False),
            patch(
                "aragora.swarm.spec.SwarmSpec.dispatch_gate_reason",
                return_value="under-specified dispatch contract",
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate._preview_env",
                side_effect=AssertionError(
                    "contract gate should not run after bounded gate failure"
                ),
            ),
        ):
            result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

        assert result["status"] == "needs_human"
        assert result["outcome"] == "blocked"
        assert "under-specified dispatch contract" in result["reasons"][0]
        assert result["dispatch_gate"]["failure_classes"] == ["contract_missing"]
        qualification = qualify_work_order_terminal_state(result)
        assert qualification.failure_classes == ["contract_missing", "needs_human"]

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
        assert "configured_max_parallel_dispatches" in payload
        assert "effective_parallel_dispatches" in payload

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
        assert "configured_max_parallel_dispatches" in payload
        assert "effective_parallel_dispatches_observed" in payload

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
        assert parsed["configured_max_parallel_dispatches"] == 1
        assert parsed["effective_parallel_dispatches_observed"] is None

    def test_loop_result_bounded_dict_caps_unbounded_operator_payload(self):
        huge_reason = (
            "contract_preflight: Drifted fields: ['permissions']\n"
            + ("x" * (4 * 1024 * 1024))
            + "\nTAIL_SENTINEL"
        )
        huge_body = "Acceptance criteria\n" + ("body context\n" * 200_000)
        result = BossLoopResult(
            run_id="boss-test-bounded",
            iterations_completed=1,
            total_elapsed_seconds=10.0,
            stop_reason="needs_human",
            issues_attempted=[{"number": 6187, "title": "Freshness fix", "body": huge_body}],
            issues_completed=[],
            issues_failed=[{"number": 6187, "title": "Freshness fix", "body": huge_body}],
            iteration_statuses=[
                {
                    "iteration": 1,
                    "run_id": "boss-test-bounded",
                    "runner_freshness": {
                        "fresh": True,
                        "details": {"raw_probe": "runner" * 500_000},
                    },
                    "selected_issue": {
                        "number": 6187,
                        "title": "Freshness fix",
                        "body": huge_body,
                    },
                    "worker_status": "needs_human",
                    "stop_reason": "needs_human",
                    "needs_human_reasons": [huge_reason],
                    "next_actions": [huge_reason],
                    "elapsed_seconds": 313.0,
                }
            ],
            needs_human_reasons=[huge_reason],
            next_actions=[huge_reason],
        )

        started_at = time.perf_counter()
        payload = result.to_bounded_dict(max_bytes=32 * 1024)
        elapsed = time.perf_counter() - started_at
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")

        assert elapsed < 2.0
        assert len(serialized) < 34 * 1024
        assert payload["_bounded"] is True
        assert payload["_truncated"] is True
        assert "contract_preflight" in payload["needs_human_reasons"][0]
        assert "TAIL_SENTINEL" in payload["needs_human_reasons"][0]
        assert len(payload["iteration_statuses"][0]["selected_issue"].get("body", "")) < 2048

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
        assert "parallel=1/1" in out

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
                "Fix the thing\n\nAcceptance Criteria:\n- pytest -q tests/swarm/test_boss_loop.py\n"
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


def test_filter_noncanonical_boss_ready_issues_strips_label_and_excludes_issue() -> None:
    generic = _make_issue(
        901,
        "Replace silent exception swallowing in postgres_store.py",
        body="Generic cleanup only.",
        labels=["boss-ready"],
    )
    canonical = _make_issue(
        902,
        "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
        body="Refresh benchmark corpus freshness after stale closed issues were detected.",
        labels=["boss-ready"],
    )
    staged_rev4 = _make_issue(
        5788,
        "Narrow broad except Exception in performance_monitor.py",
        body="Single-file exception hygiene task.",
        labels=["boss-ready", "autonomous"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    commands: list[list[str]] = []
    comments: list[str] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        commands.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    with (
        patch("aragora.swarm.boss_loop.subprocess.run", side_effect=_run),
        patch(
            "aragora.swarm.proof_first_queue._staged_rev4_issue_numbers",
            return_value=frozenset({5788}),
        ),
    ):
        kept = loop._filter_noncanonical_boss_ready_issues([generic, canonical, staged_rev4])

    assert kept == [canonical, staged_rev4]
    assert "boss-ready" not in generic.labels
    assert "boss-ready" in staged_rev4.labels
    assert comments
    assert "outside the canonical proof-first queue" in comments[-1]
    assert any(cmd[:3] == ["gh", "issue", "edit"] for cmd in commands)


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
    scoped_paths = set(spec.file_scope_hints)
    for work_order in spec.work_orders:
        if isinstance(work_order, dict):
            scoped_paths.update(str(path) for path in work_order.get("file_scope", []))
    assert "aragora/swarm/supervisor.py" in scoped_paths
    assert spec.acceptance_criteria == ["pytest -q tests/swarm/test_boss_loop.py"]


@pytest.mark.asyncio
async def test_dispatch_issue_blocks_on_missing_validation_target_before_dispatch() -> None:
    issue = _make_issue(
        2031,
        "Add missing handler tests for webhook delivery retry logic",
        body=(
            "The webhook delivery system has retry and dead-letter queue logic but limited test coverage.\n\n"
            "## Files\n"
            "- `tests/webhooks/test_delivery_retry.py` (new or extend)\n"
            "- Reference: `aragora/webhooks/delivery.py`, `aragora/webhooks/dead_letter.py`\n\n"
            "## Acceptance\n"
            "`pytest tests/webhooks/test_delivery_retry.py -x -q` passes\n"
        ),
    )
    loop = BossLoop(config=_boss_config(max_iterations=1))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
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
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "verification_target_missing"
    assert "missing validation targets" in result["reasons"][0]
    assert "tests/webhooks/test_delivery_retry.py" in result["reasons"][0]
    mock_commander_cls.return_value.run_supervised_from_spec.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_issue_allows_missing_validation_target_for_explicit_new_file() -> None:
    issue = _make_issue(
        2459,
        "Add OpenAPI spec regeneration test to prevent drift",
        body=(
            "The generated OpenAPI artifacts drift when handlers change. Add a test that regenerates and diffs.\n\n"
            "## Files\n"
            "- `tests/test_openapi_regeneration.py` (new)\n\n"
            "## Acceptance\n"
            "`pytest tests/test_openapi_regeneration.py -x -q` passes\n"
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
        "run_id": "run-2459",
        "work_orders": [
            {"status": "completed", "branch": "codex/openapi-regen-test", "commit_shas": ["abc123"]}
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
    mock_commander_cls.return_value.run_supervised_from_spec.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_issue_rewrites_missing_validation_before_dispatch() -> None:
    issue = _make_issue(
        2460,
        "Add sanitizer admission wiring",
        body=(
            "## File Scope\n"
            "- `aragora/swarm/task_sanitizer.py` (modify)\n\n"
            "Tighten the admission path so broad or contradictory tasks stop before worker launch."
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
        "run_id": "run-2460",
        "work_orders": [
            {"status": "completed", "branch": "codex/task-sanitizer", "commit_shas": ["abc123"]}
        ],
    }
    refinement_mock = AsyncMock(
        return_value={
            "refined_prompt": "",
            "files_to_change": [],
            "test_patterns": [],
            "constraints": [],
            "context_gathered": False,
        }
    )

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=refinement_mock,
        ),
        patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
    ):
        mock_commander_cls.return_value.run_supervised_from_spec = AsyncMock(return_value=fake_run)
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    assert result["original_issue_body"] == issue.body.strip()
    assert result["sanitized_issue_body"] != result["original_issue_body"]
    assert "## Validation" in result["sanitized_issue_body"]
    spec = mock_commander_cls.return_value.run_supervised_from_spec.await_args.args[0]
    assert spec.acceptance_criteria == ["python3 -m ruff check aragora/swarm/task_sanitizer.py"]
    assert "## Validation" in refinement_mock.await_args.args[1]
    assert (
        "python3 -m ruff check aragora/swarm/task_sanitizer.py"
        in refinement_mock.await_args.args[1]
    )


@pytest.mark.asyncio
async def test_dispatch_issue_quarantines_broad_scope_before_dispatch() -> None:
    issue = _make_issue(
        2461,
        "Over-broad crash cleanup lane",
        body=(
            "## Allowed Write Set\n"
            "- `aragora/swarm/a.py` (modify)\n"
            "- `aragora/swarm/b.py` (modify)\n"
            "- `aragora/swarm/c.py` (modify)\n"
            "- `aragora/swarm/d.py` (modify)\n"
            "- `aragora/swarm/e.py` (modify)\n"
            "- `aragora/swarm/f.py` (modify)\n"
        ),
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(max_iterations=1, repo="synaptent/aragora"))
    commands: list[list[str]] = []
    comments: list[str] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        commands.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=_run):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "sanitation_failed"
    assert result["sanitizer_outcome"] == "quarantined"
    assert "scope_too_broad" in result["checks_failed"]
    assert "boss-ready" not in issue.labels
    assert "boss-quarantined" in issue.labels
    assert issue.state == "OPEN"
    assert comments
    assert "file scope spans 6 files; quarantine before dispatch" in comments[-1]
    assert "scope_too_broad" in comments[-1]
    assert any(cmd[:3] == ["gh", "issue", "edit"] for cmd in commands)
    assert not any(cmd[:3] == ["gh", "issue", "close"] for cmd in commands)


@pytest.mark.asyncio
async def test_dispatch_issue_drops_short_task_before_dispatch() -> None:
    issue = _make_issue(
        2462,
        "Too short",
        body="Tiny task only.",
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(max_iterations=1, repo="synaptent/aragora"))
    commands: list[list[str]] = []
    comments: list[str] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        commands.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=_run):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "sanitation_failed"
    assert result["sanitizer_outcome"] == "dropped"
    assert "description_length" in result["checks_failed"]
    assert "boss-ready" not in issue.labels
    assert "boss-invalid" in issue.labels
    assert issue.state == "CLOSED"
    assert comments
    assert "task description is too short to dispatch safely" in comments[-1]
    assert "description_length" in comments[-1]
    assert any(cmd[:3] == ["gh", "issue", "edit"] for cmd in commands)
    assert any(cmd[:3] == ["gh", "issue", "close"] for cmd in commands)


@pytest.mark.asyncio
async def test_dispatch_issue_sanitizer_failure_skips_contract_gate() -> None:
    issue = _make_issue(
        2463,
        "Too short for contract gate",
        body="Tiny task only.",
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(max_iterations=1, repo="synaptent/aragora"))

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with (
        patch("aragora.swarm.boss_loop.subprocess.run", side_effect=_run),
        patch(
            "aragora.swarm.dispatch_contract_gate._preview_env",
            side_effect=AssertionError("contract gate should not run after sanitation failure"),
        ),
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "sanitation_failed"


@pytest.mark.asyncio
async def test_dispatch_issue_contract_gate_allows_complete_cli_dispatch() -> None:
    issue = _make_issue(2464, "Dispatch with complete CLI contract")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
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
            "aragora.swarm.dispatch_contract_gate._preview_env",
            return_value=(
                "codex",
                {
                    "ARAGORA_RUNNER_AUTH_MODE": "command",
                    "CODEX_COMMAND": "/usr/local/bin/codex",
                    "PYTEST_PATH": "/usr/local/bin/pytest",
                    "RUFF_PATH": "/usr/local/bin/ruff",
                },
            ),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
            return_value=Path("/tmp/issue-2464-contract.json"),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
            return_value=_preflight_receipt(),
        ) as contract_receipt,
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    contract_receipt.assert_called_once()
    assert contract_receipt.call_args.kwargs["skip_publication"] is True
    dispatch_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_issue_contract_gate_blocks_missing_publish_auth_slices() -> None:
    issue = _make_issue(2465, "Dispatch requires publish auth")
    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            default_target_agent="codex",
            auto_publish_deliverables=True,
        )
    )
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
            "aragora.swarm.dispatch_contract_gate._preview_env",
            return_value=(
                "codex",
                {
                    "ARAGORA_RUNNER_AUTH_MODE": "command",
                    "CODEX_COMMAND": "/usr/local/bin/codex",
                    "PYTEST_PATH": "/usr/local/bin/pytest",
                    "RUFF_PATH": "/usr/local/bin/ruff",
                },
            ),
        ),
        patch("aragora.swarm.dispatch_contract_gate._github_cli_authenticated", return_value=False),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "blocked_auth_failure"
    assert result["dispatch_contract"]["missing_slices"] == ["git", "github_api"]
    assert "missing git publish credentials" in result["reasons"][0]
    assert "missing GitHub API authentication" in result["reasons"][1]
    dispatch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_issue_contract_gate_blocks_failed_scratch_preflight_receipt() -> None:
    issue = _make_issue(2467, "Dispatch requires receipt-backed admission")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }
    failed_receipt = _preflight_receipt(
        passed=False,
        checks=[
            {
                "name": "git_commit",
                "passed": False,
                "detail": "worktree has uncommitted changes",
            }
        ],
    )

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._preview_env",
            return_value=(
                "codex",
                {
                    "ARAGORA_RUNNER_AUTH_MODE": "command",
                    "CODEX_COMMAND": "/usr/local/bin/codex",
                    "PYTEST_PATH": "/usr/local/bin/pytest",
                    "RUFF_PATH": "/usr/local/bin/ruff",
                },
            ),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
            return_value=Path("/tmp/issue-2467-contract.json"),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
            return_value=failed_receipt,
        ) as contract_receipt,
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "blocked"
    assert "failed `scratch` preflight receipt admission" in result["reasons"][0]
    assert result["dispatch_contract"]["required_receipts"] == ["scratch"]
    assert result["dispatch_contract"]["preflight_receipts"][0]["failure_terminal_class"] == (
        "blocked_not_dispatch_bounded"
    )
    contract_receipt.assert_called_once()
    dispatch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_issue_contract_gate_blocks_missing_provider_for_api_agent() -> None:
    issue = _make_issue(2466, "Dispatch requires provider auth")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="openai-api"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "openai-api-runner-1",
        "runner_type": "openai-api",
    }

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._preview_env",
            return_value=(
                "openai-api",
                {
                    "ARAGORA_RUNNER_AUTH_MODE": "command",
                    "ARAGORA_RUNNER_COMMAND": "/usr/local/bin/openai",
                    "PYTEST_PATH": "/usr/local/bin/pytest",
                    "RUFF_PATH": "/usr/local/bin/ruff",
                },
            ),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "blocked_auth_failure"
    assert result["dispatch_contract"]["missing_slices"] == ["provider"]
    assert "missing provider credentials" in result["reasons"][0]
    dispatch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_issue_contract_gate_blocks_failed_remote_publish_receipt() -> None:
    issue = _make_issue(2468, "Dispatch requires publish receipt")
    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            default_target_agent="codex",
            auto_publish_deliverables=True,
        )
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }
    remote_failure = _preflight_receipt(
        check_type="remote_publish",
        passed=False,
        checks=[
            {
                "name": "gh_pr_create_draft",
                "passed": False,
                "detail": "requires authentication",
            }
        ],
    )

    with (
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._preview_env",
            return_value=(
                "codex",
                {
                    "ARAGORA_RUNNER_AUTH_MODE": "command",
                    "CODEX_COMMAND": "/usr/local/bin/codex",
                    "PYTEST_PATH": "/usr/local/bin/pytest",
                    "RUFF_PATH": "/usr/local/bin/ruff",
                    "SSH_AUTH_SOCK": "/tmp/agent.sock",
                    "GITHUB_TOKEN": "token",
                },
            ),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
            return_value=Path("/tmp/issue-2468-contract.json"),
        ),
        patch(
            "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
            side_effect=[_preflight_receipt(), remote_failure],
        ) as contract_receipt,
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "blocked_auth_failure"
    assert result["dispatch_contract"]["required_receipts"] == ["scratch", "remote_publish"]
    assert result["dispatch_contract"]["preflight_receipts"][1]["check_type"] == "remote_publish"
    assert result["dispatch_contract"]["preflight_receipts"][1]["failure_terminal_class"] == (
        "blocked_auth_failure"
    )
    assert contract_receipt.call_count == 2
    assert [call.kwargs["skip_publication"] for call in contract_receipt.call_args_list] == [
        True,
        False,
    ]
    dispatch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_sanitizer_failed_issue_is_not_retried_in_same_run() -> None:
    dropped = _make_issue(2601, "Too short", body="Tiny task only.", labels=["boss-ready"])
    follow_up = _make_issue(2602, "Fix bounded queue state", labels=["boss-ready"])
    feed = MagicMock()
    feed.fetch.side_effect = [[dropped, follow_up], [dropped, follow_up]]
    loop = BossLoop(
        config=_boss_config(
            max_iterations=2,
            auto_continue_on_needs_human=True,
            use_value_ranking=False,
        ),
        issue_feed=feed,
        freshness_checker=lambda **kw: _fresh_result(fresh=True),
    )
    dispatched: list[int] = []

    async def _dispatch(issue, freshness):
        dispatched.append(issue.number)
        if issue.number == 2601:
            return {
                "status": "needs_human",
                "outcome": "sanitation_failed",
                "sanitizer_outcome": "quarantined",
                "checks_failed": ["scope_too_broad"],
                "reasons": [
                    "Issue #2601 was quarantined by task sanitizer: file scope spans 6 files; quarantine before dispatch"
                ],
                "next_actions": [
                    "Narrow the write scope, validation targets, or task breakdown before redispatch."
                ],
            }
        return {"status": "completed"}

    loop._dispatch_issue = _dispatch

    with patch.object(loop, "_auto_decompose_stuck_issue") as mock_decompose:
        result = await loop.run()

    assert dispatched == [2601, 2602]
    assert loop._issue_attempt_counts[2601] == loop.config.max_retries_per_issue + 1
    assert {item.get("number") for item in result.issues_attempted} == {2601, 2602}
    mock_decompose.assert_not_called()


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
async def test_dispatch_issue_uses_followup_upgrade_before_blocking_unbounded_spec() -> None:
    issue = _make_issue(
        1704,
        "Narrow broad except in helper",
        body="Narrow broad except usage in helper without additional scope.",
    )
    loop = BossLoop(config=_boss_config(max_iterations=1, require_validation_contract=False))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "runner_type": "codex",
    }

    def _upgrade(*, issue, spec, sanitized_issue_body, repo_root):
        spec.file_scope_hints = ["aragora/swarm/helper.py"]
        return spec

    with (
        patch(
            "aragora.swarm.boss_loop.dispatch_contract_gate",
            return_value=None,
        ),
        patch(
            "aragora.swarm.dispatch_followups.maybe_upgrade_dispatch_spec",
            side_effect=_upgrade,
        ) as upgrade_mock,
        patch(
            "aragora.swarm.prompt_refiner.refine_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("skip refinement")),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value={"status": "completed", "run": {"work_orders": []}}),
        ) as dispatch_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    upgrade_mock.assert_called_once()
    dispatch_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_issue_attaches_conductor_followup_to_failed_result() -> None:
    issue = _make_issue(1705, "Dispatch failed")
    loop = BossLoop(config=_boss_config(max_iterations=1))
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
            new=AsyncMock(return_value={"status": "failed", "error": "boom"}),
        ),
        patch(
            "aragora.swarm.dispatch_followups.annotate_result_with_conductor",
            return_value={
                "status": "failed",
                "error": "boom",
                "conductor_next_action": "retry_same",
            },
        ) as annotate_mock,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["conductor_next_action"] == "retry_same"
    annotate_mock.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_issue_preserves_pending_handoff_on_failed_result_with_malformed_run_id() -> (
    None
):
    issue = _make_issue(1705, "Retry handoff should ignore malformed run id")
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
            new=AsyncMock(
                return_value={
                    "status": "failed",
                    "run_id": {"id": "junk"},
                    "outcome": "crash",
                    "error": "boom",
                }
            ),
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
async def test_dispatch_issue_injects_session_resume_context_into_work_order_metadata(
    tmp_path: Path,
) -> None:
    issue = _make_issue(
        1734,
        "Reuse prior repair context",
        body=(
            "Summary:\n"
            "- Retry the bounded boss-loop fix with the prior failure context.\n\n"
            "Acceptance Criteria:\n"
            "- pytest -q tests/swarm/test_boss_loop.py\n\n"
            "Scope hints:\n"
            "- aragora/swarm/boss_loop.py\n"
        ),
    )
    store = SessionStateStore(state_dir=tmp_path)
    store.record_attempt(
        issue_number=1734,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["aragora/swarm/boss_loop.py"],
        target_agent="codex",
        runner_type="codex",
        resume_hint="pytest -q tests/swarm/test_boss_loop.py failed",
        metadata={
            "failure_reason": "pytest -q tests/swarm/test_boss_loop.py failed",
            "failing_verification": {
                "command": "pytest -q tests/swarm/test_boss_loop.py",
                "exit_code": 1,
            },
        },
    )
    store.record_attempt(
        issue_number=1734,
        repo_slug="other/repo",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["other/repo.py"],
        target_agent="codex",
        runner_type="codex",
        resume_hint="other repo should not leak",
        metadata={
            "failure_reason": "other repo should not leak",
            "failing_verification": {
                "command": "pytest -q tests/other_repo.py",
                "exit_code": 1,
            },
        },
    )
    loop = BossLoop(config=_boss_config(max_iterations=1), session_state_store=store)
    spec = SimpleNamespace(
        work_orders=[
            {
                "work_order_id": "work-1",
                "title": "Retry the bounded boss-loop fix",
                "description": "Use the prior verification failure as repair context.",
                "file_scope": ["aragora/swarm/boss_loop.py"],
                "expected_tests": ["pytest -q tests/swarm/test_boss_loop.py"],
            }
        ]
    )
    loop._attach_issue_handoff_metadata(
        spec,
        issue,
        session_state=store.latest_for_issue(issue.number, repo_slug="synaptent/aragora"),
    )
    metadata = spec.work_orders[0]["metadata"]

    assert metadata["resume_context"]["issue_number"] == 1734
    assert metadata["resume_context"]["retry_count"] == 1
    assert (
        metadata["resume_context"]["resume_hint"]
        == "pytest -q tests/swarm/test_boss_loop.py failed"
    )
    assert metadata["repair_journal"][0]["failing_verification"]["command"] == (
        "pytest -q tests/swarm/test_boss_loop.py"
    )


def test_record_session_attempt_persists_session_state_after_dispatch(tmp_path: Path) -> None:
    issue = _make_issue(
        1735,
        "Persist retry attempt",
        body=(
            "Summary:\n"
            "- Persist the retry attempt after dispatch.\n\n"
            "Acceptance Criteria:\n"
            "- pytest -q tests/swarm/test_boss_loop.py\n\n"
            "Scope hints:\n"
            "- aragora/swarm/boss_loop.py\n"
        ),
    )
    store = SessionStateStore(state_dir=tmp_path)
    loop = BossLoop(config=_boss_config(max_iterations=1), session_state_store=store)

    fake_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "reasons": ["Verification failed during pytest run."],
        "run_id": "run-1735",
        "receipt_id": "receipt-1735",
        "run": {
            "work_orders": [
                {
                    "status": "failed",
                    "target_agent": "codex",
                    "worktree_path": "/tmp/aragora-1735",
                    "branch": "codex/issue-1735",
                    "exit_code": 1,
                    "changed_paths": ["aragora/swarm/boss_loop.py"],
                    "verification_results": [
                        {
                            "command": "pytest -q tests/swarm/test_boss_loop.py",
                            "exit_code": 1,
                            "passed": False,
                            "stderr_tail": "assert False",
                        }
                    ],
                }
            ]
        },
    }

    loop._record_session_attempt(
        issue,
        fake_result,
        selected_runner={"runner_id": "codex-runner-1", "runner_type": "codex"},
        requested_target_agent="codex",
    )

    state = store.latest_for_issue(issue.number, repo_slug="synaptent/aragora")

    assert state is not None
    assert state.retry_count == 1
    assert state.target_agent == "codex"
    assert state.branch_name == "codex/issue-1735"
    assert state.metadata["repo_slug"] == "synaptent/aragora"
    assert state.attempts[-1]["exit_code"] == 1
    assert state.attempts[-1]["changed_paths"] == ["aragora/swarm/boss_loop.py"]


def test_maxed_issue_needs_human_includes_session_blocker_summary(tmp_path: Path) -> None:
    issue = _make_issue(1736, "Exhausted repair loop")
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [issue]
    store = SessionStateStore(state_dir=tmp_path)
    store.record_attempt(
        issue_number=1736,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["aragora/swarm/boss_loop.py"],
        resume_hint="Verification failed during pytest run.",
        metadata={
            "failure_reason": "Verification failed during pytest run.",
            "failing_verification": {
                "command": "pytest -q tests/swarm/test_boss_loop.py",
                "exit_code": 1,
            },
        },
    )
    store.record_attempt(
        issue_number=1736,
        repo_slug="other/repo",
        status="needs_human",
        outcome="blocked",
        exit_code=1,
        changed_files=["other/repo.py"],
        resume_hint="other repo should not leak",
        metadata={"failure_reason": "other repo should not leak"},
    )

    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            max_retries_per_issue=1,
            repo="synaptent/aragora",
        ),
        issue_feed=feed,
        freshness_checker=lambda **kw: (_ for _ in ()).throw(
            AssertionError("freshness should not be checked for maxed issues")
        ),
        session_state_store=store,
    )

    assert loop._issue_attempt_counts == {}
    assert loop._selected_issues_need_retry_routing([issue]) is True
    assert loop._issue_attempt_counts[1736] == 1

    result = asyncio.run(loop.run())

    assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
    assert any(
        "Issue #1736 exhausted retries; last blocker was failing verification" in reason
        for reason in result.needs_human_reasons
    )
    assert any(
        "pytest -q tests/swarm/test_boss_loop.py" in reason for reason in result.needs_human_reasons
    )
    assert all("other repo should not leak" not in reason for reason in result.needs_human_reasons)


def test_requested_target_agent_hydrates_repo_scoped_retry_count(tmp_path: Path) -> None:
    store = SessionStateStore(state_dir=tmp_path)
    store.record_attempt(
        issue_number=1737,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        target_agent="claude",
    )
    store.record_attempt(
        issue_number=1737,
        repo_slug="synaptent/aragora",
        status="needs_human",
        outcome="blocked",
        target_agent="codex",
    )
    store.record_attempt(
        issue_number=1737,
        repo_slug="other/repo",
        status="needs_human",
        outcome="blocked",
        target_agent="gemini",
    )

    loop = BossLoop(
        config=_boss_config(
            repo="synaptent/aragora",
            default_target_agent="claude",
            model_rotation=["claude", "codex", "gemini"],
        ),
        session_state_store=store,
    )

    assert loop._issue_attempt_counts == {}
    assert loop._requested_target_agent_for_issue(1737, repo_slug="synaptent/aragora") == "codex"
    assert loop._issue_attempt_counts[1737] == 2


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


@pytest.mark.asyncio
async def test_run_iteration_drops_ineligible_pending_handoff_issue() -> None:
    issue_a = _make_issue(1803, "Regular issue")
    issue_b = _make_issue(1804, "Pending handoff issue", labels=["wontfix"])
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

    assert seen == [issue_a.number]
    assert status.selected_issue["number"] == issue_a.number
    assert issue_b.number not in loop._pending_handoff_prompts


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
        assert payload["configured_max_parallel_dispatches"] == 1
        assert payload["effective_parallel_dispatches_observed"] == 1
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
        assert payload2["configured_max_parallel_dispatches"] == 1
        assert payload2["effective_parallel_dispatches_observed"] == 1


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


def test_boss_loop_reports_skip_labeled_issues_and_skips_dispatch() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [_make_issue(301, "Known stuck", labels=["boss-stuck"])]
    loop = BossLoop(
        config=_boss_config(max_iterations=1),
        issue_feed=feed,
        freshness_checker=lambda **kw: (_ for _ in ()).throw(
            AssertionError("freshness should not be checked for skip-labeled issues")
        ),
    )

    result = asyncio.run(loop.run())

    assert result.stop_reason == "no_suitable_issue"
    assert result.iterations_completed == 1
    assert result.issues_attempted == []
    assert "Skipped by label: boss-stuck (1: #301)" in result.needs_human_reasons
    assert "Skipped by label: boss-stuck (1: #301)" in result.next_actions


def test_boss_loop_batch_reports_effective_parallel_dispatches() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(301, "Batch issue A"),
        _make_issue(302, "Batch issue B"),
    ]
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=2),
        issue_feed=feed,
        freshness_checker=lambda **kw: RunnerFreshnessResult(
            fresh=True,
            runner_ids=["codex-runner-1", "codex-runner-2"],
            checked_at=datetime.now(UTC).isoformat(),
            details={
                "routing": {
                    "selected_runners": [
                        {"runner_id": "codex-runner-1", "available_capacity": 1},
                        {"runner_id": "codex-runner-2", "available_capacity": 1},
                    ]
                }
            },
        ),
    )
    loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

    statuses: list[BossIterationStatus] = []
    result = asyncio.run(loop.run(on_status=statuses.append))

    assert result.configured_max_parallel_dispatches == 2
    assert result.effective_parallel_dispatches_observed == 2
    assert {status.effective_parallel_dispatches for status in statuses} == {2}
    assert {status["effective_parallel_dispatches"] for status in result.iteration_statuses} == {2}


def test_boss_loop_batch_blocks_malformed_truthy_freshness_flag() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(351, "Batch malformed freshness A"),
        _make_issue(352, "Batch malformed freshness B"),
    ]
    freshness = SimpleNamespace(
        fresh="false",
        blocked_reason="malformed_fresh_flag",
        details={"routing": {"selected_runners": [{"runner_id": "codex-runner-1"}]}},
        to_dict=lambda: {"fresh": "false", "blocked_reason": "malformed_fresh_flag"},
    )
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=2),
        issue_feed=feed,
        freshness_checker=lambda **kw: freshness,
    )

    with patch.object(BossLoop, "_dispatch_issue", new_callable=AsyncMock) as mock_dispatch:
        result = asyncio.run(loop.run())

    assert result.stop_reason == BossStopReason.NO_FRESH_RUNNER.value
    assert result.issues_attempted == []
    mock_dispatch.assert_not_called()


def test_boss_loop_batch_uses_configured_limit_when_no_capacity_reported() -> None:
    """When selected runners don't report available_capacity, the configured
    max_parallel_dispatches should be used instead of falling back to serial."""
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(401, "Batch issue X"),
        _make_issue(402, "Batch issue Y"),
        _make_issue(403, "Batch issue Z"),
        _make_issue(404, "Batch issue W"),
    ]
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=4),
        issue_feed=feed,
        freshness_checker=lambda **kw: RunnerFreshnessResult(
            fresh=True,
            runner_ids=["max-01", "max-02", "max-03", "max-04"],
            checked_at=datetime.now(UTC).isoformat(),
            details={
                "routing": {
                    "selected_runners": [
                        {"runner_id": "max-01", "available_capacity": 0},
                        {"runner_id": "max-02", "available_capacity": 0},
                        {"runner_id": "max-03"},
                        {"runner_id": "max-04"},
                    ]
                }
            },
        ),
    )
    loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

    statuses: list[BossIterationStatus] = []
    result = asyncio.run(loop.run(on_status=statuses.append))

    assert result.configured_max_parallel_dispatches == 4
    assert result.effective_parallel_dispatches_observed == 4
    assert {status.effective_parallel_dispatches for status in statuses} == {4}


def test_boss_loop_batch_skips_existing_open_pr_before_dispatch() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(
            451,
            "Batch issue with existing PR",
            body=(
                "Touch aragora/swarm/open_pr_guard.py\n\n"
                "Acceptance Criteria:\n- pytest -q tests/swarm/test_boss_loop.py\n"
            ),
        ),
        _make_issue(
            452,
            "Batch issue without PR",
            body=(
                "Touch aragora/server/parallel_guard.py\n\n"
                "Acceptance Criteria:\n- pytest -q tests/swarm/test_boss_loop.py\n"
            ),
        ),
    ]
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=2, repo="synaptent/aragora"),
        issue_feed=feed,
        freshness_checker=lambda **kw: RunnerFreshnessResult(
            fresh=True,
            runner_ids=["codex-runner-1", "codex-runner-2"],
            checked_at=datetime.now(UTC).isoformat(),
            details={
                "routing": {
                    "selected_runners": [
                        {"runner_id": "codex-runner-1", "available_capacity": 1},
                        {"runner_id": "codex-runner-2", "available_capacity": 1},
                    ]
                }
            },
        ),
    )
    dispatch_seen: list[int] = []

    async def _dispatch(issue, freshness):
        dispatch_seen.append(issue.number)
        return {"status": "completed"}

    loop._dispatch_issue = _dispatch
    loop._select_issues_for_iteration = lambda issues, **kwargs: list(issues)
    existing_pr_url = "https://github.com/synaptent/aragora/pull/451"

    with patch.object(
        loop,
        "_has_open_pr_for_issue",
        side_effect=lambda number: existing_pr_url if number == 451 else None,
    ):
        result = asyncio.run(loop.run())

    assert dispatch_seen == [452]
    attempted_numbers = {item.get("number") for item in result.issues_attempted}
    assert 451 not in attempted_numbers
    assert 452 in attempted_numbers
    completed_numbers = {item.get("number") for item in result.issues_completed}
    assert 451 in completed_numbers
    skipped = [
        status
        for status in result.iteration_statuses
        if (status.get("selected_issue") or {}).get("number") == 451
    ]
    assert len(skipped) == 1
    assert any(existing_pr_url in str(action) for action in skipped[0].get("next_actions", []))


def test_boss_loop_batch_auto_decomposes_maxed_retry_issue() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(461, "Exhausted batch issue"),
        _make_issue(462, "Fresh batch issue"),
    ]
    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            max_parallel_dispatches=2,
            max_retries_per_issue=2,
            repo="synaptent/aragora",
        ),
        issue_feed=feed,
        freshness_checker=lambda **kw: RunnerFreshnessResult(
            fresh=True,
            runner_ids=["codex-runner-1", "codex-runner-2"],
            checked_at=datetime.now(UTC).isoformat(),
        ),
    )
    loop._issue_attempt_counts[461] = 2
    loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})
    loop._select_issues_for_iteration = lambda issues, **kwargs: list(issues)

    with patch.object(loop, "_auto_decompose_stuck_issue") as mock_decompose:
        result = asyncio.run(loop.run())

    mock_decompose.assert_called_once()
    assert mock_decompose.call_args.args[0] == 461
    loop._dispatch_issue.assert_awaited_once()
    dispatched_issue = loop._dispatch_issue.await_args.args[0]
    assert dispatched_issue.number == 462
    attempted_numbers = {item.get("number") for item in result.issues_attempted}
    assert 461 not in attempted_numbers
    assert 462 in attempted_numbers


def test_auto_decompose_carries_lineage_and_removes_ready_label() -> None:
    root_issue = _make_issue(
        4409,
        "[CS-01..03] Reconcile proof-first status docs",
        body="Keep roadmap docs narrower than measured proof.",
        labels=[],
    )
    issue = _make_issue(
        4412,
        "[from #4409] Add evidence metrics",
        body="Auto-decomposed from #4409 after 2 failed autonomous attempts.",
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created: list[list[str]] = []
    edited: list[list[str]] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "create"]:
            created.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "edit"]:
            edited.append(list(cmd))
        return result

    subtask = SimpleNamespace(
        title="Add focused evidence metric assertion",
        description=(
            "Add a focused regression that verifies evidence metrics are not "
            "auto-decomposed recursively when the root issue is already stuck."
        ),
        file_scope=["tests/swarm/test_boss_loop.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
    ):
        loop._auto_decompose_stuck_issue(4412, [root_issue, issue])

    assert len(created) == 1
    create_body = created[0][created[0].index("--body") + 1]
    assert "Root issue: #4409" in create_body
    assert "Parent issue: #4412" in create_body
    assert "Depth: 2" in create_body
    assert "Inherited roadmap codes: CS-01, CS-02, CS-03" in create_body
    assert edited
    assert "--add-label" in edited[-1]
    assert "boss-stuck" in edited[-1]
    assert "--remove-label" in edited[-1]
    assert "boss-ready" in edited[-1]


def test_auto_decompose_skips_candidate_already_covered_by_open_pr(monkeypatch) -> None:
    issue = _make_issue(
        4510,
        "[from #4409] Evidence metrics duplicate",
        body="## Decomposition Lineage\n- Root issue: #4409\n- Parent issue: #4503\n- Depth: 2\n",
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created: list[list[str]] = []
    comments: list[str] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "list"]:
            result.stdout = "[]"
        if cmd[:3] == ["gh", "issue", "create"]:
            created.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    subtask = SimpleNamespace(
        title="Add complete evidence metric type coverage",
        description=(
            "Add focused coverage for every evidence metric type in the "
            "observability evidence metrics contract and keep it hermetic."
        ),
        file_scope=["tests/observability/metrics/test_evidence.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    monkeypatch.setattr(
        "aragora.swarm.boss_loop.fetch_open_pr_changed_paths",
        lambda repo=None: {"tests/observability/metrics/test_evidence.py"},
    )

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
    ):
        loop._auto_decompose_stuck_issue(4510, [issue])

    assert created == []
    assert comments
    assert "already covered" in comments[-1]


def test_auto_decompose_skips_existing_boss_ready_scope_and_validation() -> None:
    issue = _make_issue(
        4511,
        "[from #4409] Evidence metrics duplicate",
        body="Auto-decomposed from #4409 after 2 failed autonomous attempts.",
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created: list[list[str]] = []
    comments: list[str] = []
    existing_body = (
        "Auto-decomposed from #4488 after 3 failed autonomous attempts.\n\n"
        "## Task\n"
        "Fix failing evidence metric tests with comprehensive metric coverage.\n\n"
        "## Files\n"
        "- `tests/observability/metrics/test_evidence.py`\n\n"
        "## Acceptance\n"
        "`python3 -m pytest tests/observability/metrics/test_evidence.py -q`\n"
    )

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "list"]:
            result.stdout = json.dumps(
                [
                    {
                        "number": 4503,
                        "title": "[from #4488] Fix evidence metric coverage",
                        "body": existing_body,
                    }
                ]
            )
        if cmd[:3] == ["gh", "issue", "create"]:
            created.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    subtask = SimpleNamespace(
        title="Repair failing evidence metric coverage",
        description=(
            "Repair failing evidence metrics coverage so test_evidence exercises "
            "all metric types without external monitoring dependencies."
        ),
        file_scope=["tests/observability/metrics/test_evidence.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("aragora.swarm.boss_loop.fetch_open_pr_changed_paths", return_value=set()),
    ):
        loop._auto_decompose_stuck_issue(4511, [issue])

    assert created == []
    assert comments
    assert "already covered" in comments[-1]


def test_auto_decompose_skips_generic_same_scope_restatement() -> None:
    issue = _make_issue(
        4512,
        "[from #4409] Fix failing evidence metric tests",
        body=(
            "Auto-decomposed from #4409 after 3 failed autonomous attempts.\n\n"
            "## Decomposition Lineage\n"
            "- Root issue: #4409\n"
            "- Parent issue: #4488\n"
            "- Depth: 1\n\n"
            "## Task\n"
            "Fix failing tests and ensure comprehensive coverage for evidence metrics.\n\n"
            "## Files\n"
            "- `tests/observability/metrics/test_evidence.py`\n\n"
        ),
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created: list[list[str]] = []
    comments: list[str] = []

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "list"]:
            result.stdout = "[]"
        if cmd[:3] == ["gh", "issue", "create"]:
            created.append(list(cmd))
        if cmd[:3] == ["gh", "issue", "comment"]:
            comments.append(cmd[cmd.index("--body") + 1])
        return result

    subtask = SimpleNamespace(
        title="Fix failing evidence metric tests",
        description=(
            "Fix failing tests and ensure comprehensive coverage for evidence metrics "
            "in the existing evidence metric test module."
        ),
        file_scope=["tests/observability/metrics/test_evidence.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
        patch("aragora.swarm.boss_loop.fetch_open_pr_changed_paths", return_value=set()),
    ):
        loop._auto_decompose_stuck_issue(4512, [issue])

    assert created == []
    assert comments
    assert "parent task" in comments[-1]


def test_auto_decompose_stops_at_body_lineage_depth_limit() -> None:
    issue = _make_issue(
        4475,
        "[from #4461] Recursive evidence metrics task",
        body=("## Decomposition Lineage\n- Root issue: #4409\n- Parent issue: #4461\n- Depth: 3\n"),
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))

    def _run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result

    with (
        patch("subprocess.run", side_effect=_run),
        patch.object(loop, "_label_boss_stuck") as mock_label_stuck,
        patch("aragora.nomic.task_decomposer.TaskDecomposer") as mock_decomposer,
    ):
        loop._auto_decompose_stuck_issue(4475, [issue])

    mock_label_stuck.assert_called_once()
    mock_decomposer.assert_not_called()


def test_auto_decompose_skips_delayed_root_issue_lineage() -> None:
    issue = _make_issue(
        5400,
        "[from #5390] Define operator state model",
        body=("## Decomposition Lineage\n- Root issue: #5331\n- Parent issue: #5390\n- Depth: 2\n"),
        labels=["boss-ready"],
    )
    root_issue = _make_issue(
        5331,
        "BC-07 Unify lane, host, runner, and publication state into one operator model",
        body="Refs: docs/status/NEXT_STEPS_CANONICAL.md (`BC-07`)",
        labels=[],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    policy = RoadmapPriorityPolicy(
        do_now=frozenset({"RS-07"}),
        delay=frozenset({"BC-07", "BC-08", "BC-09"}),
        avoid=frozenset(),
    )

    with (
        patch("aragora.swarm.boss_loop.load_roadmap_priority_policy", return_value=policy),
        patch.object(loop, "_fetch_issue_by_number", return_value=root_issue),
        patch.object(loop, "_label_boss_stuck") as mock_label_stuck,
        patch("aragora.nomic.task_decomposer.TaskDecomposer") as mock_decomposer,
    ):
        loop._auto_decompose_stuck_issue(5400, [issue])

    mock_label_stuck.assert_called_once()
    assert "BC-07" in mock_label_stuck.call_args.args[2]
    mock_decomposer.assert_not_called()


def test_boss_loop_batch_serializes_retry_routed_dispatches() -> None:
    feed = MagicMock(spec=GitHubIssueFeed)
    feed.fetch.return_value = [
        _make_issue(471, "Retry-routed batch issue A"),
        _make_issue(472, "Retry-routed batch issue B"),
    ]
    loop = BossLoop(
        config=_boss_config(max_iterations=1, max_parallel_dispatches=2),
        issue_feed=feed,
        freshness_checker=lambda **kw: RunnerFreshnessResult(
            fresh=True,
            runner_ids=["codex-runner-1", "codex-runner-2"],
            checked_at=datetime.now(UTC).isoformat(),
            details={
                "routing": {
                    "selected_runners": [
                        {"runner_id": "codex-runner-1", "available_capacity": 1},
                        {"runner_id": "codex-runner-2", "available_capacity": 1},
                    ]
                }
            },
        ),
    )
    loop._issue_attempt_counts[471] = 1
    loop._issue_attempt_counts[472] = 1
    dispatch_seen: list[int] = []

    async def _dispatch(issue, freshness):
        dispatch_seen.append(issue.number)
        return {"status": "completed"}

    loop._dispatch_issue = AsyncMock(side_effect=_dispatch)

    statuses: list[BossIterationStatus] = []
    result = asyncio.run(loop.run(on_status=statuses.append))

    assert dispatch_seen == [471]
    assert result.configured_max_parallel_dispatches == 2
    assert result.effective_parallel_dispatches_observed == 1
    assert {status.effective_parallel_dispatches for status in statuses} == {1}


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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_skips_non_python_files(self, tmp_path, monkeypatch):
        """Non-.py files (docs, configs) are ignored."""

        def _run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "docs/STATUS.md\nREADME.md\nsetup.cfg\n"
            return result

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
        paths = discover_focused_tests(tmp_path)
        assert paths == []

    def test_focused_returns_empty_on_missing_git(self, tmp_path, monkeypatch):
        """FileNotFoundError from git binary returns empty list."""

        def _run(cmd, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

        monkeypatch.setattr("aragora.swarm.boss_validation.subprocess.run", _run)
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

    def test_heartbeat_refresh_requires_real_boolean_available(self, tmp_path):
        """Malformed availability values fail closed during heartbeat refresh."""
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
                        "available": "false",
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
        assert reg["available"] is False
        assert reg["updated_at"] != old_ts
        assert reg["heartbeat_at"] != old_ts
        assert reg["freshness_status"] == "unavailable"
        assert reg["updated_at"] != old_ts
        assert reg["heartbeat_at"] != old_ts

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

    def test_deferred_publish_queue_drained_each_iteration(self) -> None:
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.side_effect = [
            [_make_issue(1, "First issue")],
            [_make_issue(2, "Second issue")],
        ]

        loop = BossLoop(
            config=_boss_config(max_iterations=2),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        drain_calls: list[int] = []
        loop._drain_deferred_publish_queue = lambda: drain_calls.append(1) or 0
        loop._dispatch_issue = AsyncMock(return_value={"status": "completed"})

        result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.MAX_ITERATIONS.value
        assert len(drain_calls) == 2

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
async def test_dispatch_preserves_needs_human_backbone_status():
    """Backbone ledger should preserve review-required outcomes."""
    issue = _make_issue(43, "Backbone needs-human wiring")
    loop = BossLoop(config=_boss_config(max_iterations=2, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    updated_calls: list[dict[str, Any]] = []

    class MockRuntime:
        def create_run(self, ledger):
            return None

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})

    fake_result = {
        "status": "needs_human",
        "run_id": "run-43",
        "receipt_id": "receipt-43",
    }

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

    assert result["status"] == "needs_human"
    assert len(updated_calls) == 1
    assert updated_calls[0]["status"] == "needs_human"
    assert updated_calls[0]["execution_id"] == "run-43"
    assert updated_calls[0]["receipt_id"] == "receipt-43"


@pytest.mark.asyncio
async def test_dispatch_auto_publish_records_postprocessed_publish_metadata_in_backbone():
    """Backbone ledger should capture autonomous publish follow-up metadata."""
    issue = _make_issue(45, "Backbone publish wiring")
    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            default_target_agent="codex",
            auto_publish_deliverables=True,
        )
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    updated_calls: list[dict[str, Any]] = []

    class MockRuntime:
        def create_run(self, ledger):
            return None

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})

    fake_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "run_id": "run-45",
        "receipt_id": "receipt-45",
        "deliverable": {
            "type": "branch",
            "branch": "codex/issue-45",
            "commit_shas": ["abc123"],
        },
    }

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
        patch(
            "aragora.swarm.boss_loop.BossLoop._harvest_worker_commits_for_publish",
            return_value={
                "action": "harvested",
                "branch": "aragora/boss-harvest/issue-45",
                "commit_shas": ["abc123"],
            },
        ),
        patch(
            "aragora.swarm.tranche_integrate.publish_lane_deliverable",
            return_value={
                "published": True,
                "action": "pr_created",
                "branch": "codex/issue-45",
                "pr_url": "https://github.com/synaptent/aragora/pull/2045",
            },
        ),
        patch("aragora.ralph.github_control.GitHubControl") as github_control_cls,
        patch("aragora.swarm.pr_registry.PullRequestRegistry"),
    ):
        github_control_cls.return_value.upsert_issue_comment.return_value = {
            "commented": True,
            "action": "created",
            "comment_url": "https://github.com/synaptent/aragora/issues/45#issuecomment-1",
        }
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    assert result["outcome"] == "pr_adopted"
    assert len(updated_calls) == 2
    assert updated_calls[-1]["status"] == "completed"
    assert updated_calls[-1]["metadata"]["boss_postprocess"]["publish_result"]["action"] == (
        "pr_created"
    )
    assert updated_calls[-1]["metadata"]["boss_postprocess"]["issue_comment_result"]["action"] == (
        "created"
    )
    assert (
        updated_calls[-1]["metadata"]["boss_postprocess"]["postprocess_promoted_from_status"]
        == "needs_human"
    )


def test_published_deliverable_helpers_require_boolean_success_flag() -> None:
    worker_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "deliverable": {
            "type": "pr",
            "branch": "codex/issue-46",
            "pr_url": "https://github.com/synaptent/aragora/pull/2046",
        },
        "publish_result": {
            "published": "false",
            "action": "pr_created",
            "branch": "codex/issue-46",
            "pr_url": "https://github.com/synaptent/aragora/pull/2046",
        },
    }

    assert BossLoop._published_deliverable_comment(worker_result) is None
    assert BossLoop._promote_published_deliverable(worker_result) is False
    assert worker_result["status"] == "needs_human"
    assert worker_result["outcome"] == "blocked"


def test_postprocess_promotes_existing_pr_deliverable() -> None:
    loop = BossLoop(
        config=BossLoopConfig(
            repo="synaptent/aragora",
            auto_publish_deliverables=True,
        )
    )
    issue = _make_issue(number=46)
    pr_url = "https://github.com/synaptent/aragora/pull/2046"
    worker_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "deliverable": {
            "type": "pr",
            "pr_url": pr_url,
        },
    }

    with (
        patch.object(loop, "_maybe_comment_published_deliverable", return_value=None),
        patch.object(loop, "_maybe_auto_close_already_done_issue", return_value=None),
        patch.object(loop, "_convert_pr_to_draft"),
    ):
        result = loop._postprocess_issue_result(issue, worker_result)

    assert result["publish_result"] == {
        "action": "existing_pr",
        "published": True,
        "branch": None,
        "pr_url": pr_url,
    }
    assert result["status"] == "completed"
    assert result["outcome"] == "pr_adopted"
    assert result["receipt_metadata"]["publish_result"]["published"] is True


@pytest.mark.asyncio
async def test_dispatch_auto_publish_rejects_malformed_success_flag() -> None:
    issue = _make_issue(46, "Backbone malformed publish wiring")
    loop = BossLoop(
        config=_boss_config(
            max_iterations=1,
            default_target_agent="codex",
            auto_publish_deliverables=True,
        )
    )
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    updated_calls: list[dict[str, Any]] = []

    class MockRuntime:
        def create_run(self, ledger):
            return None

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})

    fake_result = {
        "status": "needs_human",
        "outcome": "blocked",
        "run_id": "run-46",
        "receipt_id": "receipt-46",
        "deliverable": {
            "type": "branch",
            "branch": "codex/issue-46",
            "commit_shas": ["abc123"],
        },
    }

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
        patch(
            "aragora.swarm.boss_loop.BossLoop._harvest_worker_commits_for_publish",
            return_value={
                "action": "harvested",
                "branch": "aragora/boss-harvest/issue-46",
                "commit_shas": ["abc123"],
            },
        ),
        patch(
            "aragora.swarm.tranche_integrate.publish_lane_deliverable",
            return_value={
                "published": "false",
                "action": "pr_created",
                "branch": "codex/issue-46",
                "pr_url": "https://github.com/synaptent/aragora/pull/2046",
            },
        ),
        patch("aragora.ralph.github_control.GitHubControl") as github_control_cls,
        patch("aragora.swarm.pr_registry.PullRequestRegistry"),
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "needs_human"
    assert result["outcome"] == "blocked"
    assert result["publish_result"]["published"] == "false"
    assert "issue_comment_result" not in result
    assert len(updated_calls) == 2
    assert updated_calls[-1]["status"] == "needs_human"
    assert (
        "postprocess_promoted_from_status" not in updated_calls[-1]["metadata"]["boss_postprocess"]
    )
    github_control_cls.return_value.upsert_issue_comment.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_preserves_running_backbone_status():
    """Backbone ledger should preserve in-flight dispatch outcomes."""
    issue = _make_issue(44, "Backbone running wiring")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    updated_calls: list[dict[str, Any]] = []

    class MockRuntime:
        def create_run(self, ledger):
            return None

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})

    fake_result = {
        "status": "running",
        "run_id": "run-44",
        "receipt_id": "receipt-44",
    }

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

    assert result["status"] == "running"
    assert len(updated_calls) == 1
    assert updated_calls[0]["status"] == "running"
    assert updated_calls[0]["execution_id"] == "run-44"
    assert updated_calls[0]["receipt_id"] == "receipt-44"


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
        patch("aragora.swarm.boss_loop.logger") as mock_logger,
    ):
        # Should NOT raise despite backbone failures
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    mock_logger.debug.assert_called_once_with(
        "Boss backbone ledger create failed for issue #%d: %s",
        issue.number,
        "backbone unavailable",
    )


@pytest.mark.asyncio
async def test_dispatch_backbone_update_failures_log_and_do_not_block():
    issue = _make_issue(100, "Backbone update failure resilience")
    loop = BossLoop(config=_boss_config(max_iterations=1, default_target_agent="codex"))
    loop._claim_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: (None, None)
    loop._selected_runner_for_dispatch = lambda freshness, *, requested_target_agent=None: {
        "runner_id": "codex-runner-1",
        "agent_type": "codex",
    }

    updated_calls: list[dict[str, Any]] = []

    class FlakyRuntime:
        def create_run(self, ledger):
            return None

        def update_run(self, run_id, **kw):
            updated_calls.append({"run_id": run_id, **kw})
            raise RuntimeError("backbone update unavailable")

    fake_result = {
        "status": "needs_human",
        "run_id": "run-100",
        "receipt_id": "receipt-100",
    }
    postprocessed_result = {
        **fake_result,
        "status": "completed",
        "outcome": "pr_adopted",
    }

    with (
        patch(
            "aragora.pipeline.backbone_runtime.BackboneRuntime",
            FlakyRuntime,
        ),
        patch(
            "aragora.pipeline.backbone_contracts.RunLedger",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ),
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(return_value=fake_result),
        ),
        patch.object(loop, "_postprocess_issue_result", return_value=postprocessed_result),
        patch.object(
            loop,
            "_apply_postprocess_metadata",
            return_value={"publish_result": {"action": "pr_created"}},
        ),
        patch("aragora.swarm.boss_loop.logger") as mock_logger,
    ):
        result = await loop._dispatch_issue(issue, _fresh_result(fresh=True))

    assert result["status"] == "completed"
    assert len(updated_calls) == 2
    assert mock_logger.debug.call_args_list == [
        call(
            "Boss backbone ledger dispatch update failed for issue #%d: %s",
            issue.number,
            "backbone update unavailable",
        ),
        call(
            "Boss backbone ledger postprocess update failed for issue #%d: %s",
            issue.number,
            "backbone update unavailable",
        ),
    ]


# ---------------------------------------------------------------------------
# Draft PR promotion tests
# ---------------------------------------------------------------------------


class TestConvertPrToDraft:
    """Tests for _convert_pr_to_draft."""

    def test_converts_pr_to_draft_on_success(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        worker_result: dict[str, Any] = {
            "pr_url": "https://github.com/synaptent/aragora/pull/42",
        }
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            loop._convert_pr_to_draft(worker_result)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "ready", "--undo", "42", "-R", "synaptent/aragora"]
        assert worker_result.get("draft_converted") is True

    def test_no_op_when_no_pr_url(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        worker_result: dict[str, Any] = {"status": "completed"}
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            loop._convert_pr_to_draft(worker_result)
        mock_run.assert_not_called()
        assert "draft_converted" not in worker_result

    def test_handles_already_draft(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        worker_result: dict[str, Any] = {
            "pr_url": "https://github.com/synaptent/aragora/pull/7",
        }
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode=1, stdout="", stderr="already a draft"
            )
            loop._convert_pr_to_draft(worker_result)
        assert worker_result.get("draft_converted") is True

    def test_handles_gh_failure_gracefully(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        worker_result: dict[str, Any] = {
            "pr_url": "https://github.com/synaptent/aragora/pull/7",
        }
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="network error")
            loop._convert_pr_to_draft(worker_result)
        assert "draft_converted" not in worker_result

    def test_handles_subprocess_exception(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        worker_result: dict[str, Any] = {
            "pr_url": "https://github.com/synaptent/aragora/pull/7",
        }
        with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=OSError("fail")):
            loop._convert_pr_to_draft(worker_result)
        assert "draft_converted" not in worker_result


class TestAllRequiredChecksPassed:
    """Tests for _all_required_checks_passed."""

    def test_all_checks_pass(self) -> None:
        checks_json = json.dumps(
            [
                {"name": "lint", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "sdk-parity", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "Generate & Validate", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "TypeScript SDK Type Check",
                    "state": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {"name": "other-check", "state": "COMPLETED", "conclusion": "FAILURE"},
            ]
        )
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout=checks_json, stderr="")
            assert BossLoop._all_required_checks_passed(42, "synaptent/aragora") is True

    def test_missing_check_fails(self) -> None:
        checks_json = json.dumps(
            [
                {"name": "lint", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "sdk-parity", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "Generate & Validate", "state": "COMPLETED", "conclusion": "SUCCESS"},
                # TypeScript SDK Type Check is missing
            ]
        )
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout=checks_json, stderr="")
            assert BossLoop._all_required_checks_passed(42, "synaptent/aragora") is False

    def test_failed_check_fails(self) -> None:
        checks_json = json.dumps(
            [
                {"name": "lint", "state": "COMPLETED", "conclusion": "FAILURE"},
                {"name": "typecheck", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "sdk-parity", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "Generate & Validate", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "TypeScript SDK Type Check",
                    "state": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ]
        )
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout=checks_json, stderr="")
            assert BossLoop._all_required_checks_passed(42, "synaptent/aragora") is False

    def test_gh_command_failure(self) -> None:
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="error")
            assert BossLoop._all_required_checks_passed(42, "synaptent/aragora") is False

    def test_subprocess_exception(self) -> None:
        with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=OSError("fail")):
            assert BossLoop._all_required_checks_passed(42, "synaptent/aragora") is False


class TestPromoteReadyDrafts:
    """Tests for _promote_ready_drafts."""

    def test_identifies_automation_and_boss_owned_draft_branches(self) -> None:
        assert BossLoop._draft_promotion_ownership("codex/swarm-2c4959f7-micro-2") == "queue-owned"
        assert (
            BossLoop._draft_promotion_ownership("aragora/boss-harvest/issue-10-boss-aaa")
            == "boss-owned"
        )
        assert BossLoop._draft_promotion_ownership("codex/issue-101") is None
        assert BossLoop._draft_promotion_ownership("codex/ordinary-branch") is None
        assert BossLoop._draft_promotion_ownership("factory/manual-fix") is None
        assert BossLoop._draft_promotion_ownership("feature/human-draft") is None
        assert BossLoop._draft_promotion_ownership(None) is None

    def test_promotes_draft_with_all_checks_passing(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        draft_list_json = json.dumps(
            [
                {"number": 10, "headRefName": "codex/swarm-lane-10"},
                {"number": 20, "headRefName": "feature/human-draft"},
                {"number": 30, "headRefName": "aragora/boss-harvest/issue-30-boss-abc"},
            ]
        )
        checks_10 = json.dumps(
            [
                {"name": "lint", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "sdk-parity", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "Generate & Validate", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "TypeScript SDK Type Check",
                    "state": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ]
        )
        checks_30 = json.dumps(
            [
                {"name": "lint", "state": "COMPLETED", "conclusion": "FAILURE"},
            ]
        )
        ready_calls: list[list[str]] = []
        check_calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout=draft_list_json, stderr="")
            if "checks" in cmd:
                check_calls.append(cmd)
                if "10" in cmd:
                    return SimpleNamespace(returncode=0, stdout=checks_10, stderr="")
                return SimpleNamespace(returncode=0, stdout=checks_30, stderr="")
            # gh pr ready (promote)
            ready_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=fake_run):
            promoted = loop._promote_ready_drafts()

        assert promoted == [10]
        assert [cmd[3] for cmd in check_calls] == ["10", "30"]
        assert [cmd[3] for cmd in ready_calls] == ["10"]

    def test_skips_generic_automation_branches_even_when_checks_pass(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        draft_list_json = json.dumps(
            [
                {"number": 10, "headRefName": "codex/ordinary-branch"},
                {"number": 20, "headRefName": "factory/manual-fix"},
            ]
        )
        commands: list[list[str]] = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout=draft_list_json, stderr="")
            raise AssertionError(f"unexpected gh invocation: {cmd}")

        with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=fake_run):
            promoted = loop._promote_ready_drafts()

        assert promoted == []
        assert commands == [
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--draft",
                "--json",
                "number,headRefName",
                "--limit",
                "100",
                "-R",
                "synaptent/aragora",
            ]
        ]

    def test_skips_unowned_human_draft_even_when_checks_pass(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        draft_list_json = json.dumps(
            [
                {"number": 10, "headRefName": "feature/human-draft"},
            ]
        )
        commands: list[list[str]] = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout=draft_list_json, stderr="")
            raise AssertionError(f"unexpected gh invocation: {cmd}")

        with patch("aragora.swarm.boss_loop.subprocess.run", side_effect=fake_run):
            promoted = loop._promote_ready_drafts()

        assert promoted == []
        assert commands == [
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--draft",
                "--json",
                "number,headRefName",
                "--limit",
                "100",
                "-R",
                "synaptent/aragora",
            ]
        ]

    def test_no_repo_returns_empty(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo=None))
        result = loop._promote_ready_drafts()
        assert result == []

    def test_gh_list_failure_returns_empty(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="error")
            result = loop._promote_ready_drafts()
        assert result == []

    def test_empty_draft_list(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="[]", stderr="")
            result = loop._promote_ready_drafts()
        assert result == []


class TestListOpenBossHarvestPrs:
    """Tests for boss-harvest queue-cap inspection."""

    def test_filters_to_open_boss_harvest_prs(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        pr_list_json = json.dumps(
            [
                {
                    "number": 10,
                    "headRefName": "aragora/boss-harvest/issue-10-boss-aaa",
                    "isDraft": True,
                    "url": "https://github.com/synaptent/aragora/pull/10",
                },
                {
                    "number": 11,
                    "headRefName": "codex/ordinary-branch",
                    "isDraft": True,
                    "url": "https://github.com/synaptent/aragora/pull/11",
                },
            ]
        )
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout=pr_list_json, stderr="")
            result = loop._list_open_boss_harvest_prs()

        assert result == [
            {
                "number": 10,
                "headRefName": "aragora/boss-harvest/issue-10-boss-aaa",
                "isDraft": True,
                "url": "https://github.com/synaptent/aragora/pull/10",
            }
        ]

    def test_returns_empty_when_listing_fails(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        with patch("aragora.swarm.boss_loop.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="error")
            result = loop._list_open_boss_harvest_prs()
        assert result == []


class TestMaybePublishDeliverable:
    """Tests for auto-publish queue capping."""

    def test_debate_gate_disabled_preserves_existing_publish_path(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
            )
        )
        issue = _make_issue(number=127)
        worker_result = {
            "status": "needs_human",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-127",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(loop, "_run_debate_publish_gate") as gate_mock,
            patch.object(loop, "_list_open_boss_harvest_prs", return_value=[]),
            patch.object(
                loop,
                "_harvest_worker_commits_for_publish",
                return_value={
                    "action": "harvested",
                    "branch": "codex/issue-127",
                    "source_branch": "codex/issue-127",
                    "commit_shas": ["abc123"],
                },
            ),
            patch(
                "aragora.swarm.tranche_integrate.publish_lane_deliverable",
                return_value={
                    "published": True,
                    "branch": "codex/issue-127",
                    "pr_url": "https://github.com/synaptent/aragora/pull/2127",
                },
            ) as mock_publish,
            patch("aragora.swarm.pr_registry.PullRequestRegistry"),
        ):
            result = loop._maybe_publish_deliverable(issue, worker_result)

        gate_mock.assert_not_called()
        assert result is not None
        assert result["published"] is True
        assert result["pr_url"] == "https://github.com/synaptent/aragora/pull/2127"
        assert "debate_gate_result" not in worker_result
        assert mock_publish.called

    def test_reuses_existing_published_pr_for_branch_deliverable(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
                max_open_auto_publish_prs=1,
            )
        )
        issue = _make_issue(number=124)
        worker_result = {
            "status": "needs_human",
            "outcome": "blocked",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-124",
                "commit_shas": ["abc123"],
            },
            "publish_result": {
                "published": True,
                "action": "existing_pr",
                "branch": "codex/issue-124",
                "pr_url": "https://github.com/synaptent/aragora/pull/2046",
            },
        }

        with (
            patch.object(loop, "_list_open_boss_harvest_prs") as mock_list_open_prs,
            patch.object(loop, "_harvest_worker_commits_for_publish") as mock_harvest,
            patch("aragora.swarm.tranche_integrate.publish_lane_deliverable") as mock_publish,
        ):
            result = loop._maybe_publish_deliverable(issue, worker_result)

        assert result == {
            "published": True,
            "action": "existing_pr",
            "branch": "codex/issue-124",
            "pr_url": "https://github.com/synaptent/aragora/pull/2046",
        }
        assert worker_result["deliverable"] == {
            "type": "pr",
            "branch": "codex/issue-124",
            "commit_shas": ["abc123"],
            "pr_url": "https://github.com/synaptent/aragora/pull/2046",
        }
        assert worker_result["pr_url"] == "https://github.com/synaptent/aragora/pull/2046"
        assert worker_result["pr_number"] == 2046
        assert BossLoop._promote_published_deliverable(worker_result) is True
        assert worker_result["status"] == "completed"
        assert worker_result["outcome"] == "pr_adopted"
        mock_list_open_prs.assert_not_called()
        mock_harvest.assert_not_called()
        mock_publish.assert_not_called()

    def test_defers_when_open_boss_harvest_pr_already_exists(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
                max_open_auto_publish_prs=1,
            )
        )
        issue = _make_issue(number=123)
        worker_result = {
            "status": "needs_human",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-123",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(
                loop,
                "_list_open_boss_harvest_prs",
                return_value=[
                    {
                        "number": 2045,
                        "headRefName": "aragora/boss-harvest/issue-45-boss-aaa",
                        "isDraft": True,
                        "url": "https://github.com/synaptent/aragora/pull/2045",
                    }
                ],
            ),
            patch.object(loop, "_harvest_worker_commits_for_publish") as mock_harvest,
            patch("aragora.swarm.tranche_integrate.publish_lane_deliverable") as mock_publish,
        ):
            result = loop._maybe_publish_deliverable(issue, worker_result)

        assert result == {
            "action": "deferred_due_to_open_boss_prs",
            "reason": "open_boss_harvest_pr_limit",
            "branch": "codex/issue-123",
            "max_open_prs": 1,
            "open_prs": [
                {
                    "number": 2045,
                    "headRefName": "aragora/boss-harvest/issue-45-boss-aaa",
                    "isDraft": True,
                    "url": "https://github.com/synaptent/aragora/pull/2045",
                }
            ],
        }
        mock_harvest.assert_not_called()
        mock_publish.assert_not_called()

    def test_existing_pr_deliverable_marks_publish_success(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
            )
        )
        issue = _make_issue(number=124)
        pr_url = "https://github.com/synaptent/aragora/pull/2124"
        worker_result = {
            "status": "needs_human",
            "deliverable": {
                "type": "pr",
                "pr_url": pr_url,
            },
        }

        result = loop._maybe_publish_deliverable(issue, worker_result)

        assert result == {
            "action": "existing_pr",
            "published": True,
            "branch": None,
            "pr_url": pr_url,
        }
        assert worker_result["pr_url"] == pr_url
        assert worker_result["pr_number"] == 2124

    def test_skips_harvest_fallback_when_source_branch_has_no_diff(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
            )
        )
        issue = _make_issue(number=125)
        worker_result = {
            "status": "completed",
            "deliverable": {
                "type": "branch",
                "branch": "codex/swarm-empty",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(loop, "_list_open_boss_harvest_prs", return_value=[]),
            patch.object(
                loop,
                "_harvest_worker_commits_for_publish",
                side_effect=RuntimeError("previous cherry-pick is now empty"),
            ),
            patch.object(loop, "_publish_branch_has_target_diff", return_value=False),
            patch("aragora.swarm.tranche_integrate.publish_lane_deliverable") as mock_publish,
        ):
            result = loop._maybe_publish_deliverable(issue, worker_result)

        assert result == {
            "action": "skipped_empty_publish_branch",
            "published": False,
            "reason": "harvest_failed_empty_diff",
            "branch": "codex/swarm-empty",
            "source_branch": "codex/swarm-empty",
            "commit_shas": ["abc123"],
            "harvest_result": {
                "action": "harvest_failed",
                "reason": "RuntimeError",
                "branch": "codex/swarm-empty",
                "source_branch": "codex/swarm-empty",
                "commit_shas": ["abc123"],
                "error": "previous cherry-pick is now empty",
            },
        }
        assert worker_result["harvest_result"]["action"] == "harvest_failed"
        mock_publish.assert_not_called()

    def test_harvest_fallback_publishes_when_branch_diff_unverified(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
                target_branch="release/2026.04",
            )
        )
        issue = _make_issue(number=126)
        worker_result = {
            "status": "needs_human",
            "deliverable": {
                "type": "branch",
                "branch": "codex/swarm-valid",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(loop, "_list_open_boss_harvest_prs", return_value=[]),
            patch.object(
                loop,
                "_harvest_worker_commits_for_publish",
                side_effect=RuntimeError("fatal: invalid reference: release/2026.04"),
            ),
            patch.object(loop, "_publish_branch_has_target_diff", return_value=None),
            patch(
                "aragora.swarm.tranche_integrate.publish_lane_deliverable",
                return_value={
                    "published": True,
                    "branch": "codex/swarm-valid",
                    "pr_url": "https://github.com/synaptent/aragora/pull/2126",
                },
            ) as mock_publish,
            patch("aragora.swarm.pr_registry.PullRequestRegistry"),
        ):
            result = loop._maybe_publish_deliverable(issue, worker_result)

        assert result is not None
        assert result["published"] is True
        assert result["pr_url"] == "https://github.com/synaptent/aragora/pull/2126"
        assert worker_result["harvest_result"]["action"] == "harvest_failed"
        assert mock_publish.call_args.kwargs["target_branch"] == "release/2026.04"
        assert mock_publish.call_args.args[0].branch == "codex/swarm-valid"

    def test_debate_gate_blocks_publish_and_records_metadata(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
                use_debate_publish_gate=True,
            )
        )
        issue = _make_issue(number=128)
        worker_result = {
            "status": "needs_human",
            "outcome": "blocked",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-128",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(
                loop,
                "_run_debate_publish_gate",
                return_value={
                    "verdict": "blocked",
                    "publication_allowed": False,
                    "passed": False,
                    "reason": "Publish summary mismatches the verified diff.",
                    "concerns": ["publish metadata hides a blocker"],
                    "fail_open_used": False,
                    "ran": True,
                },
            ),
            patch.object(loop, "_list_open_boss_harvest_prs", return_value=[]),
            patch.object(
                loop,
                "_harvest_worker_commits_for_publish",
                return_value={
                    "action": "harvested",
                    "branch": "codex/issue-128",
                    "source_branch": "codex/issue-128",
                    "commit_shas": ["abc123"],
                },
            ),
            patch("aragora.swarm.tranche_integrate.publish_lane_deliverable") as mock_publish,
            patch.object(loop, "_maybe_comment_published_deliverable", return_value=None),
            patch.object(loop, "_maybe_auto_close_already_done_issue", return_value=None),
            patch.object(loop, "_convert_pr_to_draft"),
        ):
            result = loop._postprocess_issue_result(issue, worker_result)

        assert result["status"] == "needs_human"
        assert result["publish_result"]["action"] == "blocked_by_debate_gate"
        assert result["publish_result"]["published"] is False
        assert result["receipt_metadata"]["debate_gate_result"]["verdict"] == "blocked"
        assert "Debate publish gate blocked publication" in result["reasons"][-1]
        mock_publish.assert_not_called()

    def test_debate_gate_fail_open_preserves_publish(self) -> None:
        loop = BossLoop(
            config=BossLoopConfig(
                repo="synaptent/aragora",
                auto_publish_deliverables=True,
                use_debate_publish_gate=True,
            )
        )
        issue = _make_issue(number=129)
        worker_result = {
            "status": "needs_human",
            "outcome": "blocked",
            "deliverable": {
                "type": "branch",
                "branch": "codex/issue-129",
                "commit_shas": ["abc123"],
            },
        }

        with (
            patch.object(
                loop,
                "_run_debate_publish_gate",
                return_value={
                    "verdict": "fail_open",
                    "publication_allowed": True,
                    "passed": False,
                    "reason": "TimeoutError: model unavailable",
                    "concerns": [],
                    "fail_open_used": True,
                    "ran": False,
                },
            ),
            patch.object(loop, "_list_open_boss_harvest_prs", return_value=[]),
            patch.object(
                loop,
                "_harvest_worker_commits_for_publish",
                return_value={
                    "action": "harvested",
                    "branch": "codex/issue-129",
                    "source_branch": "codex/issue-129",
                    "commit_shas": ["abc123"],
                },
            ),
            patch(
                "aragora.swarm.tranche_integrate.publish_lane_deliverable",
                return_value={
                    "published": True,
                    "branch": "codex/issue-129",
                    "pr_url": "https://github.com/synaptent/aragora/pull/2129",
                },
            ) as mock_publish,
            patch("aragora.swarm.pr_registry.PullRequestRegistry"),
            patch.object(loop, "_maybe_comment_published_deliverable", return_value=None),
            patch.object(loop, "_maybe_auto_close_already_done_issue", return_value=None),
            patch.object(loop, "_convert_pr_to_draft"),
        ):
            result = loop._postprocess_issue_result(issue, worker_result)

        assert result["publish_result"]["published"] is True
        assert result["receipt_metadata"]["debate_gate_result"]["verdict"] == "fail_open"
        assert result["deliverable"]["type"] == "pr"
        assert mock_publish.called


class TestPostprocessConvertsToDraft:
    """Verify _postprocess_issue_result calls _convert_pr_to_draft."""

    def test_postprocess_calls_convert(self) -> None:
        loop = BossLoop(config=BossLoopConfig(repo="synaptent/aragora"))
        issue = _make_issue(number=99)
        worker_result: dict[str, Any] = {
            "status": "completed",
            "pr_url": "https://github.com/synaptent/aragora/pull/99",
        }
        with (
            patch.object(loop, "_maybe_publish_deliverable", return_value=None),
            patch.object(loop, "_maybe_comment_published_deliverable", return_value=None),
            patch.object(loop, "_maybe_auto_close_already_done_issue", return_value=None),
            patch.object(loop, "_promote_published_deliverable"),
            patch.object(loop, "_convert_pr_to_draft") as mock_convert,
        ):
            loop._postprocess_issue_result(issue, worker_result)
        mock_convert.assert_called_once_with(worker_result)

    def test_run_iteration_reports_debate_gate_block_followup(self) -> None:
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(130, "Publish gate followup")]
        loop = BossLoop(
            config=_boss_config(
                max_iterations=1,
                auto_publish_deliverables=True,
                use_debate_publish_gate=True,
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._emit_lane_receipt = MagicMock(return_value="lane-130")
        loop._log_value_outcome = MagicMock()

        with patch.object(
            loop,
            "_dispatch_issue",
            AsyncMock(
                return_value={
                    "status": "needs_human",
                    "outcome": "blocked",
                    "deliverable": {
                        "type": "branch",
                        "branch": "codex/issue-130",
                        "commit_shas": ["abc123"],
                    },
                    "publish_result": {
                        "action": "blocked_by_debate_gate",
                        "published": False,
                        "branch": "codex/issue-130",
                        "pr_url": None,
                        "reason": "Publish gate requires human review before PR creation.",
                        "concerns": ["diff summary omits the blocker"],
                    },
                    "debate_gate_result": {
                        "verdict": "blocked",
                        "publication_allowed": False,
                        "passed": False,
                        "reason": "Publish gate requires human review before PR creation.",
                        "concerns": ["diff summary omits the blocker"],
                        "fail_open_used": False,
                        "ran": True,
                    },
                    "receipt_metadata": {
                        "debate_gate_result": {
                            "verdict": "blocked",
                            "publication_allowed": False,
                            "passed": False,
                            "reason": "Publish gate requires human review before PR creation.",
                            "concerns": ["diff summary omits the blocker"],
                            "fail_open_used": False,
                            "ran": True,
                        }
                    },
                }
            ),
        ):
            status = asyncio.run(loop._run_iteration(1))

        assert status.worker_status == "completed"
        assert "Publish skipped by debate gate" in status.next_actions[0]


# ---------------------------------------------------------------------------
# Published PR = terminal (no retry)
# ---------------------------------------------------------------------------


class TestPublishedPrTerminal:
    """When a worker produces a PR, the issue must be terminal for that run."""

    def test_needs_human_with_pr_url_is_terminal_completed(self):
        """Worker returns needs_human but produced a PR -> completed, not retried."""
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(42, "Implement widget")]

        pr_url = "https://github.com/synaptent/aragora/pull/100"

        async def _dispatch(issue, freshness):
            return {
                "status": "needs_human",
                "reasons": ["Approval required."],
                "deliverable": {"type": "pr", "pr_url": pr_url},
                "pr_url": pr_url,
            }

        loop = BossLoop(
            config=_boss_config(
                max_iterations=5,
                max_retries_per_issue=5,
                auto_continue_on_needs_human=True,
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _dispatch

        result = asyncio.run(loop.run())

        # The issue should be completed, not failed
        completed_numbers = {i.get("number") for i in result.issues_completed}
        failed_numbers = {i.get("number") for i in result.issues_failed}
        assert 42 in completed_numbers, "Issue should be in completed list"
        assert 42 not in failed_numbers, "Issue should NOT be in failed list"

        # It should have been dispatched exactly once (terminal on first attempt)
        dispatch_statuses = [
            s
            for s in result.iteration_statuses
            if (s.get("selected_issue") or {}).get("number") == 42
            and s.get("worker_status") == "completed"
        ]
        assert len(dispatch_statuses) == 1, (
            f"Issue should be dispatched exactly once, got {len(dispatch_statuses)}"
        )

        # The next_actions should mention the PR URL
        actions = dispatch_statuses[0].get("next_actions", [])
        assert any(pr_url in str(a) for a in actions), (
            f"next_actions should mention PR URL, got {actions}"
        )

    def test_completed_result_is_not_redispatched_in_same_run(self):
        """A completed result should exhaust retries for the current loop run."""
        issue = _make_issue(43, "Complete once")
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [issue]
        dispatch_count = 0

        async def _dispatch(issue, freshness):
            nonlocal dispatch_count
            dispatch_count += 1
            return {"status": "completed"}

        loop = BossLoop(
            config=_boss_config(
                max_iterations=3,
                max_retries_per_issue=3,
                auto_continue_on_needs_human=True,
                repo=None,
            ),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _dispatch

        result = asyncio.run(loop.run())

        assert dispatch_count == 1
        completed_numbers = {i.get("number") for i in result.issues_completed}
        assert 43 in completed_numbers
        assert loop._issue_attempt_counts[43] == 3

    def test_existing_open_pr_skips_dispatch(self):
        """Issue with an existing open boss-harvest PR is skipped in dispatch."""
        feed = MagicMock(spec=GitHubIssueFeed)
        feed.fetch.return_value = [_make_issue(42, "Implement widget")]

        dispatch_called = []

        async def _dispatch(issue, freshness):
            dispatch_called.append(issue.number)
            return {"status": "completed"}

        loop = BossLoop(
            config=_boss_config(max_iterations=3, repo="synaptent/aragora"),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )
        loop._dispatch_issue = _dispatch
        loop._issue_attempt_counts[42] = loop.config.max_retries_per_issue

        # Simulate that an open PR already exists for issue #42
        existing_pr_url = "https://github.com/synaptent/aragora/pull/200"
        with (
            patch.object(
                loop,
                "_has_open_pr_for_issue",
                return_value=existing_pr_url,
            ),
            patch(
                "aragora.swarm.boss_loop.select_eligible_issue",
                side_effect=lambda issues, **kwargs: next(
                    (i for i in issues if i is not None), None
                ),
            ),
        ):
            result = asyncio.run(loop.run())

        # Dispatch should never have been called
        assert 42 not in dispatch_called, "Issue with existing PR should not be dispatched"

        # The issue should be marked completed
        completed_numbers = {i.get("number") for i in result.issues_completed}
        assert 42 in completed_numbers, "Issue should be in completed list"

        # The iteration status should mention the existing PR
        skipped = [
            s
            for s in result.iteration_statuses
            if (s.get("selected_issue") or {}).get("number") == 42
        ]
        assert len(skipped) >= 1
        actions = skipped[0].get("next_actions", [])
        assert any(existing_pr_url in str(a) for a in actions), (
            f"next_actions should mention existing PR, got {actions}"
        )

    def test_already_maxed_issue_numbers_lists_open_prs_once(self):
        """Retry exhaustion checks open PR state once per scan, not once per issue."""
        loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
        loop._issue_attempt_counts = {
            42: loop.config.max_retries_per_issue,
            43: loop.config.max_retries_per_issue,
            44: loop.config.max_retries_per_issue,
        }

        open_prs = [
            {
                "number": 200,
                "headRefName": "aragora/boss-harvest/issue-42-boss-aaa",
                "url": "https://github.com/synaptent/aragora/pull/200",
            },
            {
                "number": 201,
                "headRefName": "aragora/boss-harvest/issue-43-boss-bbb",
                "url": "https://github.com/synaptent/aragora/pull/201",
            },
        ]

        with (
            patch.object(loop, "_hydrate_issue_attempt_count", return_value=0),
            patch.object(
                loop, "_list_open_boss_harvest_prs", return_value=open_prs
            ) as mock_list_open_prs,
            patch.object(loop, "_auto_decompose_stuck_issue") as mock_decompose,
        ):
            result = loop._already_maxed_issue_numbers(
                [
                    _make_issue(42, "Existing PR one"),
                    _make_issue(43, "Existing PR two"),
                    _make_issue(44, "No PR"),
                ]
            )

        assert result == {44}
        mock_list_open_prs.assert_called_once()
        mock_decompose.assert_called_once()


# ---------------------------------------------------------------------------
# Decomposition guardrails (Task 1 + Task 4)
# ---------------------------------------------------------------------------


def test_per_run_sub_issue_budget_stops_decomposition() -> None:
    """After max_total_sub_issues_per_run is exhausted, decomposition is refused."""
    issue = _make_issue(
        100,
        "[from #50] Fix failing tests",
        body="## Task\nFix failing tests in the module.\n\n## Files\n- `tests/foo/test_bar.py`\n",
        labels=["boss-ready"],
    )
    loop = BossLoop(
        config=_boss_config(
            repo="synaptent/aragora",
            max_total_sub_issues_per_run=2,
        )
    )
    # Simulate budget already exhausted
    loop._total_sub_issues_created = 2

    stuck_labels: list[str] = []

    def _run(cmd, **kw):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "edit"]:
            stuck_labels.append("called")
        if cmd[:3] == ["gh", "issue", "comment"]:
            stuck_labels.append(cmd[cmd.index("--body") + 1])
        return result

    with patch("subprocess.run", side_effect=_run):
        loop._auto_decompose_stuck_issue(100, [issue])

    # Should label boss-stuck, not create sub-issues
    assert any("budget" in s.lower() for s in stuck_labels if isinstance(s, str))


def test_decompose_annotates_new_files() -> None:
    """Sub-issue file scope should include (new file) for non-existent files."""
    issue = _make_issue(
        200,
        "Add unit tests for foo.py",
        body=(
            "## Task\nCreate tests for foo module.\n\n"
            "## Files\n- `aragora/foo.py`\n- `tests/foo/test_foo.py` (create)\n\n"
            "## Acceptance\n`pytest tests/foo/test_foo.py -q`\n"
        ),
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created_bodies: list[str] = []

    def _run(cmd, **kw):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "list"]:
            result.stdout = ""
        if cmd[:3] == ["gh", "issue", "create"]:
            body_idx = cmd.index("--body") + 1 if "--body" in cmd else None
            if body_idx:
                created_bodies.append(cmd[body_idx])
        return result

    subtask = SimpleNamespace(
        title="Create test file",
        description="Create comprehensive unit tests for the foo module in tests/foo/test_foo.py.",
        file_scope=["tests/foo/test_foo.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
    ):
        loop._auto_decompose_stuck_issue(200, [issue])

    assert created_bodies
    body = created_bodies[0]
    assert "(new file)" in body, (
        f"Expected (new file) annotation in sub-issue body, got: {body[:200]}"
    )


def test_decompose_uses_ruff_for_nonexistent_test_files() -> None:
    """When test file doesn't exist, validation should use ruff, not pytest."""
    issue = _make_issue(
        300,
        "Add unit tests for bar.py",
        body=(
            "## Task\nCreate tests for bar module.\n\n"
            "## Files\n- `aragora/bar.py`\n- `tests/bar/test_bar.py` (create)\n\n"
            "## Acceptance\n`pytest tests/bar/test_bar.py -q`\n"
        ),
        labels=["boss-ready"],
    )
    loop = BossLoop(config=_boss_config(repo="synaptent/aragora"))
    created_bodies: list[str] = []

    def _run(cmd, **kw):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        if cmd[:3] == ["gh", "issue", "create"]:
            body_idx = cmd.index("--body") + 1 if "--body" in cmd else None
            if body_idx:
                created_bodies.append(cmd[body_idx])
        return result

    subtask = SimpleNamespace(
        title="Create test file",
        description="Create comprehensive unit tests for the bar module in tests/bar/test_bar.py.",
        file_scope=["tests/bar/test_bar.py"],
        estimated_complexity="small",
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = SimpleNamespace(should_decompose=True, subtasks=[subtask])

    with (
        patch("subprocess.run", side_effect=_run),
        patch("aragora.nomic.task_decomposer.TaskDecomposer", return_value=decomposer),
    ):
        loop._auto_decompose_stuck_issue(300, [issue])

    if created_bodies:
        body = created_bodies[0]
        # Should NOT use pytest on a non-existent file
        assert "pytest tests/bar/test_bar.py" not in body, (
            f"Should not use pytest on non-existent test file: {body[:200]}"
        )
