"""Tests for reviewer diff-fidelity: _fetch_diff_content and _build_prompt with diff."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.errors import AgentConnectionError, CLISubprocessError
from aragora.swarm.campaign import (
    CampaignProject,
    CampaignReviewGate,
    CampaignReviewStatus,
    CampaignReviewer,
    _DIFF_MAX_CHARS,
    _fetch_diff_content,
)
from aragora.swarm.review_routing import ReviewRoutingError


def _make_run_dict(*, branch: str = "codex/worker-branch", commit: str = "abc123") -> dict:
    return {
        "work_orders": [
            {
                "work_order_id": "wo-1",
                "status": "completed",
                "branch": branch,
                "commit_shas": [commit],
            }
        ],
    }


def _make_project(**overrides) -> CampaignProject:
    defaults = {
        "project_id": "phase0a-007",
        "title": "Test project",
        "status": "active",
        "acceptance_criteria": ["File exists"],
        "file_scope_hints": ["docs/test.md"],
    }
    defaults.update(overrides)
    return CampaignProject(**defaults)


class TestFetchDiffContent:
    def test_returns_none_when_no_repo_root(self) -> None:
        assert _fetch_diff_content(_make_run_dict(), repo_root=None) is None

    def test_returns_none_when_no_deliverable(self) -> None:
        run_dict: dict = {"work_orders": []}
        assert _fetch_diff_content(run_dict, repo_root=Path("/tmp")) is None

    def test_returns_none_when_deliverable_has_no_branch(self) -> None:
        run_dict: dict = {
            "work_orders": [{"status": "completed", "pr_url": "https://github.com/org/repo/pull/1"}]
        }
        # PR type without branch — should return None
        assert _fetch_diff_content(run_dict, repo_root=Path("/tmp")) is None

    def test_returns_diff_on_success(self, tmp_path: Path) -> None:
        diff_text = "diff --git a/docs/test.md b/docs/test.md\n+hello world\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = diff_text

        with patch("aragora.swarm.campaign.subprocess.run", return_value=mock_result) as mock_run:
            result = _fetch_diff_content(_make_run_dict(), repo_root=tmp_path, target_branch="main")

        assert result == diff_text
        # Two calls: fetch + diff
        assert mock_run.call_count == 2
        diff_call = mock_run.call_args_list[1]
        assert diff_call[0][0] == ["git", "diff", "origin/main...codex/worker-branch"]
        assert diff_call[1]["cwd"] == str(tmp_path)

    def test_truncates_large_diffs(self, tmp_path: Path) -> None:
        large_diff = "x" * (_DIFF_MAX_CHARS + 1000)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = large_diff

        with patch("aragora.swarm.campaign.subprocess.run", return_value=mock_result):
            result = _fetch_diff_content(_make_run_dict(), repo_root=tmp_path, max_chars=100)

        assert result is not None
        assert len(result) < 200  # truncated + message
        assert "truncated" in result

    def test_returns_none_on_git_error(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: bad revision"

        with patch("aragora.swarm.campaign.subprocess.run", return_value=mock_result):
            result = _fetch_diff_content(_make_run_dict(), repo_root=tmp_path)

        assert result is None

    def test_returns_none_on_timeout(self, tmp_path: Path) -> None:
        with patch(
            "aragora.swarm.campaign.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30),
        ):
            result = _fetch_diff_content(_make_run_dict(), repo_root=tmp_path)

        assert result is None

    def test_returns_none_on_empty_diff(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "   \n  "

        with patch("aragora.swarm.campaign.subprocess.run", return_value=mock_result):
            result = _fetch_diff_content(_make_run_dict(), repo_root=tmp_path)

        assert result is None

    def test_extracts_branch_from_work_order_for_pr_deliverable(self, tmp_path: Path) -> None:
        run_dict: dict = {
            "work_orders": [
                {
                    "status": "completed",
                    "pr_url": "https://github.com/org/repo/pull/1",
                    "branch": "codex/pr-branch",
                    "commit_shas": ["abc"],
                }
            ]
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "+new content\n"

        with patch("aragora.swarm.campaign.subprocess.run", return_value=mock_result) as mock_run:
            result = _fetch_diff_content(run_dict, repo_root=tmp_path)

        assert result is not None
        args = mock_run.call_args
        assert "codex/pr-branch" in args[0][0][2]


class TestBuildPromptWithDiff:
    def test_prompt_includes_diff_when_provided(self) -> None:
        project = _make_project()
        run_dict = _make_run_dict()
        diff = "diff --git a/docs/test.md\n+hello world\n"

        prompt = CampaignReviewer._build_prompt(project, run_dict, "claude", diff_content=diff)

        assert "ACTUAL DIFF" in prompt
        assert "+hello world" in prompt
        assert "END DIFF" in prompt
        parsed_json = json.loads(prompt.split("\n")[-1])
        assert parsed_json["project_id"] == "phase0a-007"

    def test_prompt_excludes_diff_when_none(self) -> None:
        project = _make_project()
        run_dict = _make_run_dict()

        prompt = CampaignReviewer._build_prompt(project, run_dict, "claude", diff_content=None)

        assert "ACTUAL DIFF" not in prompt
        assert "END DIFF" not in prompt

    def test_prompt_excludes_diff_when_empty(self) -> None:
        project = _make_project()
        run_dict = _make_run_dict()

        prompt = CampaignReviewer._build_prompt(project, run_dict, "claude", diff_content="")

        assert "ACTUAL DIFF" not in prompt

    def test_prompt_backward_compatible_without_diff(self) -> None:
        """Calling without diff_content produces same format as before."""
        project = _make_project()
        run_dict = _make_run_dict()

        prompt = CampaignReviewer._build_prompt(project, run_dict, "claude")

        assert "Review this completed implementation" in prompt
        assert '"status":"passed|changes_requested|blocked_nonreviewable"' in prompt
        parsed_json = json.loads(prompt.split("\n")[-1])
        assert parsed_json["project_id"] == "phase0a-007"

    def test_prompt_includes_budget_context(self) -> None:
        project = _make_project()
        run_dict = _make_run_dict()

        prompt = CampaignReviewer._build_prompt(
            project,
            run_dict,
            "claude",
            budget_context={
                "budget_limit_usd": 5.0,
                "spent_cost_usd": 1.5,
                "reserved_cost_usd": 0.5,
                "available_budget_usd": 3.0,
                "project_estimated_cost_usd": 1.0,
            },
        )

        parsed_json = json.loads(prompt.split("\n")[-1])
        assert parsed_json["budget"]["budget_limit_usd"] == 5.0
        assert parsed_json["budget"]["available_budget_usd"] == 3.0
        assert parsed_json["budget"]["project_estimated_cost_usd"] == 1.0


class TestReviewerBillingErrorDetection:
    """Tests that billing/credit errors from the Claude CLI are surfaced clearly."""

    @pytest.mark.asyncio
    async def test_credit_balance_error_surfaces_billing_message(self) -> None:
        """CLISubprocessError with 'credit balance' triggers actionable finding."""
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()

        billing_error = CLISubprocessError(
            message="CLI command failed with return code 1 (stdout): Credit balance is too low",
            agent_name="campaign-review",
            returncode=1,
            stderr="Credit balance is too low",
        )

        with patch(
            "aragora.swarm.campaign.generate_review_response",
            new=AsyncMock(side_effect=billing_error),
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
            )

        assert gate.status == CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
        assert any("billing" in f.lower() for f in gate.findings)
        assert any("claude auth" in f.lower() for f in gate.findings)

    @pytest.mark.asyncio
    async def test_review_prompt_includes_actual_git_diff_content(self, tmp_path: Path) -> None:
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()
        diff_text = "diff --git a/docs/test.md b/docs/test.md\n+hello from git diff\n"

        fetch_result = MagicMock()
        fetch_result.returncode = 0
        fetch_result.stdout = ""

        diff_result = MagicMock()
        diff_result.returncode = 0
        diff_result.stdout = diff_text

        with (
            patch(
                "aragora.swarm.campaign.subprocess.run",
                side_effect=[fetch_result, diff_result],
            ) as mock_run,
            patch(
                "aragora.swarm.campaign.generate_review_response",
                new=AsyncMock(
                    return_value={
                        "candidate": {"provider": "claude", "label": "claude:max-12"},
                        "response": '{"status":"passed","findings":[]}',
                        "attempts": [
                            {"candidate": "claude:max-12", "stage": "generate", "detail": "ok"}
                        ],
                    }
                ),
            ) as mock_route,
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
                repo_root=tmp_path,
                target_branch="main",
            )

        assert gate.status == CampaignReviewStatus.PASSED.value
        prompt = mock_route.await_args.args[0]
        assert "ACTUAL DIFF" in prompt
        assert diff_text in prompt
        assert "hello from git diff" in prompt
        assert mock_run.call_args_list[1][0][0] == [
            "git",
            "diff",
            "origin/main...codex/worker-branch",
        ]

    @pytest.mark.asyncio
    async def test_non_billing_cli_error_does_not_trigger_billing_message(self) -> None:
        """Regular CLISubprocessError does not produce billing guidance."""
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()

        cli_error = CLISubprocessError(
            message="CLI command failed with return code 1: connection refused",
            agent_name="campaign-review",
            returncode=1,
            stderr="connection refused",
        )

        with patch(
            "aragora.swarm.campaign.generate_review_response",
            new=AsyncMock(side_effect=cli_error),
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
            )

        assert gate.status == CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
        assert not any("billing" in f.lower() for f in gate.findings)
        assert any("CLISubprocessError" in f for f in gate.findings)

    @pytest.mark.asyncio
    async def test_generic_exception_does_not_leak_details(self) -> None:
        """Non-CLI exceptions report type name only, not raw str(exc)."""
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()

        with patch(
            "aragora.swarm.campaign.generate_review_response",
            new=AsyncMock(side_effect=ValueError("secret internal detail")),
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
            )

        assert gate.status == CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
        assert any("ValueError" in f for f in gate.findings)
        # Raw exception detail should not be in findings
        assert not any("secret internal detail" in f for f in gate.findings)
        assert gate.raw_review["error"] == "ValueError"
        assert gate.raw_review["detail"] == "secret internal detail"

    @pytest.mark.asyncio
    async def test_agent_connection_error_keeps_detail_in_raw_review(self) -> None:
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()

        review_error = AgentConnectionError(
            "Connection failed",
            agent_name="campaign-review",
            cause=RuntimeError("certificate verify failed"),
        )

        with patch(
            "aragora.swarm.campaign.generate_review_response",
            new=AsyncMock(side_effect=review_error),
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
            )

        assert gate.status == CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
        assert gate.findings == ["Review failed: AgentConnectionError"]
        assert gate.raw_review["error"] == "AgentConnectionError"
        assert "certificate verify failed" in gate.raw_review["detail"]

    @pytest.mark.asyncio
    async def test_review_routing_error_surfaces_attempt_chain(self) -> None:
        reviewer = CampaignReviewer()
        project = _make_project()
        run_dict = _make_run_dict()
        routing_error = ReviewRoutingError(
            [
                {"candidate": "codex", "detail": "codex CLI not found", "kind": "preflight"},
                {
                    "candidate": "claude:max-01",
                    "detail": "Reviewer credits are exhausted.",
                    "kind": "billing_exhausted",
                },
                {
                    "candidate": "openrouter",
                    "detail": "OpenRouter TLS check failed",
                    "kind": "preflight",
                },
            ],
            category="billing_exhausted",
        )

        with patch(
            "aragora.swarm.campaign.generate_review_response",
            new=AsyncMock(side_effect=routing_error),
        ):
            gate = await reviewer.review(
                project=project,
                worker_model="codex",
                review_model="claude",
                run_dict=run_dict,
            )

        assert gate.status == CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value
        assert any("billing" in item.lower() for item in gate.findings)
        assert any("claude auth status" in item.lower() for item in gate.findings)
        assert gate.raw_review["error"] == "ReviewRoutingError"
        assert len(gate.raw_review["attempts"]) == 3
