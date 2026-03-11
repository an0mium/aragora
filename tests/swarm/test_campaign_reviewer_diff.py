"""Tests for reviewer diff-fidelity: _fetch_diff_content and _build_prompt with diff."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.campaign import (
    CampaignProject,
    CampaignReviewGate,
    CampaignReviewStatus,
    CampaignReviewer,
    _DIFF_MAX_CHARS,
    _fetch_diff_content,
)


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
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ["git", "diff", "main...codex/worker-branch"]
        assert args[1]["cwd"] == str(tmp_path)

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
