"""Tests for the admin merge arbiter."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.merge_arbiter import (
    REQUIRED_CHECKS,
    ArbiterSummary,
    MergeArbiter,
    MergeArbiterConfig,
    MergeResult,
    _evaluate_pr,
    _get_check_status,
    _list_candidate_prs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gh_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _all_passing_checks() -> list[dict]:
    return [{"name": name, "state": "SUCCESS"} for name in REQUIRED_CHECKS]


def _pr(number: int = 1, branch: str = "boss-harvest/fix-1", draft: bool = False) -> dict:
    return {"number": number, "headRefName": branch, "isDraft": draft}


# ---------------------------------------------------------------------------
# _list_candidate_prs
# ---------------------------------------------------------------------------


class TestListCandidatePrs:
    def test_filters_by_prefix(self):
        prs = [
            _pr(1, "boss-harvest/fix-1"),
            _pr(2, "codex/task-2"),
            _pr(3, "dependabot/npm"),
            _pr(4, "feat/manual-feature"),
        ]
        config = MergeArbiterConfig(branch_prefixes=["boss-harvest", "codex/"])
        with patch("aragora.swarm.merge_arbiter._run_gh") as mock_gh:
            mock_gh.return_value = _make_gh_result(stdout=json.dumps(prs))
            result = _list_candidate_prs(config)
        assert len(result) == 2
        assert result[0]["number"] == 1
        assert result[1]["number"] == 2

    def test_returns_empty_on_gh_failure(self):
        config = MergeArbiterConfig()
        with patch("aragora.swarm.merge_arbiter._run_gh") as mock_gh:
            mock_gh.return_value = _make_gh_result(returncode=1, stderr="error")
            assert _list_candidate_prs(config) == []

    def test_returns_empty_on_bad_json(self):
        config = MergeArbiterConfig()
        with patch("aragora.swarm.merge_arbiter._run_gh") as mock_gh:
            mock_gh.return_value = _make_gh_result(stdout="not json")
            assert _list_candidate_prs(config) == []


# ---------------------------------------------------------------------------
# _get_check_status
# ---------------------------------------------------------------------------


class TestGetCheckStatus:
    def test_parses_check_output(self):
        checks = _all_passing_checks()
        with patch("aragora.swarm.merge_arbiter._run_gh") as mock_gh:
            mock_gh.return_value = _make_gh_result(stdout=json.dumps(checks))
            result = _get_check_status(1, "owner/repo")
        assert len(result) == 5
        for name in REQUIRED_CHECKS:
            assert result[name] == "SUCCESS"

    def test_returns_empty_on_failure(self):
        with patch("aragora.swarm.merge_arbiter._run_gh") as mock_gh:
            mock_gh.return_value = _make_gh_result(returncode=1)
            assert _get_check_status(1, "owner/repo") == {}


# ---------------------------------------------------------------------------
# _evaluate_pr — all checks passing → merge
# ---------------------------------------------------------------------------


class TestEvaluatePrAllPassing:
    def test_merges_when_all_checks_pass(self):
        config = MergeArbiterConfig()
        checks = _all_passing_checks()
        with (
            patch("aragora.swarm.merge_arbiter._get_check_status") as mock_checks,
            patch("aragora.swarm.merge_arbiter._merge_pr") as mock_merge,
        ):
            mock_checks.return_value = {c["name"]: "SUCCESS" for c in checks}
            mock_merge.return_value = (True, "merged")
            result = _evaluate_pr(_pr(42, "boss-harvest/ok"), config)
        assert result.success is True
        assert result.pr_number == 42
        mock_merge.assert_called_once_with(42, config.repo)


# ---------------------------------------------------------------------------
# _evaluate_pr — failing check → skipped
# ---------------------------------------------------------------------------


class TestEvaluatePrFailingCheck:
    def test_skips_when_check_fails(self):
        config = MergeArbiterConfig()
        status = dict.fromkeys(REQUIRED_CHECKS, "SUCCESS")
        status["lint"] = "FAILURE"
        with patch("aragora.swarm.merge_arbiter._get_check_status") as mock_checks:
            mock_checks.return_value = status
            result = _evaluate_pr(_pr(10, "boss-harvest/bad"), config)
        assert result.success is False
        assert "failing" in result.reason
        assert "lint" in result.reason

    def test_skips_when_check_missing(self):
        config = MergeArbiterConfig()
        # Only return 4 of 5 checks
        status = dict.fromkeys(REQUIRED_CHECKS[:4], "SUCCESS")
        with patch("aragora.swarm.merge_arbiter._get_check_status") as mock_checks:
            mock_checks.return_value = status
            result = _evaluate_pr(_pr(11, "boss-harvest/partial"), config)
        assert result.success is False
        assert "missing" in result.reason


# ---------------------------------------------------------------------------
# _evaluate_pr — dry-run mode
# ---------------------------------------------------------------------------


class TestEvaluatePrDryRun:
    def test_dry_run_does_not_merge(self):
        config = MergeArbiterConfig(dry_run=True)
        checks = dict.fromkeys(REQUIRED_CHECKS, "SUCCESS")
        with (
            patch("aragora.swarm.merge_arbiter._get_check_status") as mock_checks,
            patch("aragora.swarm.merge_arbiter._merge_pr") as mock_merge,
        ):
            mock_checks.return_value = checks
            result = _evaluate_pr(_pr(99, "boss-harvest/dry"), config)
        assert result.success is True
        assert "dry-run" in result.reason
        mock_merge.assert_not_called()


# ---------------------------------------------------------------------------
# _evaluate_pr — draft promotion
# ---------------------------------------------------------------------------


class TestEvaluatePrDraft:
    def test_promotes_draft_pr(self):
        config = MergeArbiterConfig()
        with patch("aragora.swarm.merge_arbiter._promote_draft") as mock_promote:
            mock_promote.return_value = True
            result = _evaluate_pr(_pr(5, "boss-harvest/draft", draft=True), config)
        assert result.success is False
        assert "promoted" in result.reason
        mock_promote.assert_called_once_with(5, config.repo)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_stops_after_consecutive_failures(self):
        config = MergeArbiterConfig(
            max_consecutive_failures=3,
            poll_interval_seconds=0.01,
            max_runtime_hours=0.01,
        )
        arbiter = MergeArbiter(config=config)

        failing_pr = _pr(1, "boss-harvest/fail")
        failing_checks = dict.fromkeys(REQUIRED_CHECKS, "SUCCESS")
        failing_checks["lint"] = "FAILURE"

        call_count = 0

        def fake_list(_cfg):
            nonlocal call_count
            call_count += 1
            return [failing_pr]

        with (
            patch("aragora.swarm.merge_arbiter._list_candidate_prs", side_effect=fake_list),
            patch("aragora.swarm.merge_arbiter._get_check_status", return_value=failing_checks),
        ):
            summary = await arbiter.run()

        assert "circuit breaker" in summary.stop_reason
        assert call_count == 3


# ---------------------------------------------------------------------------
# ArbiterSummary
# ---------------------------------------------------------------------------


class TestArbiterSummary:
    def test_to_dict_roundtrip(self):
        summary = ArbiterSummary(
            merged=[1, 2],
            skipped=[3],
            failed=[4],
            polls=5,
            stop_reason="done",
            elapsed_seconds=123.456,
        )
        d = summary.to_dict()
        assert d["merged"] == [1, 2]
        assert d["elapsed_seconds"] == 123.5
        assert d["stop_reason"] == "done"


# ---------------------------------------------------------------------------
# Full run with merges
# ---------------------------------------------------------------------------


class TestFullRun:
    @pytest.mark.asyncio
    async def test_merges_eligible_pr(self):
        config = MergeArbiterConfig(
            poll_interval_seconds=0.01,
            max_runtime_hours=0.0001,  # Very short — exits after 1 poll
        )
        arbiter = MergeArbiter(config=config)

        pr = _pr(7, "boss-harvest/good")
        checks = dict.fromkeys(REQUIRED_CHECKS, "SUCCESS")

        with (
            patch("aragora.swarm.merge_arbiter._list_candidate_prs", return_value=[pr]),
            patch("aragora.swarm.merge_arbiter._get_check_status", return_value=checks),
            patch("aragora.swarm.merge_arbiter._merge_pr", return_value=(True, "merged")),
        ):
            summary = await arbiter.run()

        assert 7 in summary.merged
        assert summary.polls >= 1
