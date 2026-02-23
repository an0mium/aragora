"""Tests for review CLI Knowledge Mound persistence and --post-comment flag.

Tests that:
1. Review findings are persisted to KM after extraction
2. Graceful fallback when KM dependencies are unavailable
3. Graceful fallback when KM persistence fails at runtime
4. _persist_review_to_km helper passes correct args to DecisionReceipt
5. --post-comment flag formats and posts findings via `gh pr comment`
6. --post-comment without PR URL returns error
7. --post-comment handles gh CLI failures gracefully
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.cli.review import (
    _persist_review_to_km,
    cmd_review,
    format_github_comment,
)


@dataclass
class MockCritique:
    agent: str = "anthropic-api"
    target_agent: str = "openai-api"
    issues: list = field(default_factory=lambda: ["SQL injection risk"])
    suggestions: list = field(default_factory=lambda: ["Use parameterized queries"])
    severity: float = 0.8


@dataclass
class MockMessage:
    agent: str = "anthropic-api"
    content: str = "Found a security issue"
    role: str = "reviewer"


@dataclass
class MockVote:
    voter: str = "anthropic-api"
    choice: str = "reject"
    confidence: float = 0.9


@dataclass
class MockDebateResult:
    final_answer: str = "Review complete: 2 issues found"
    confidence: float = 0.85
    consensus_reached: bool = True
    winner: str = "anthropic-api"
    rounds_used: int = 2
    duration_seconds: float = 10.0
    messages: list = field(default_factory=lambda: [MockMessage()])
    votes: list = field(default_factory=lambda: [MockVote()])
    critiques: list = field(
        default_factory=lambda: [
            MockCritique(severity=0.95),
            MockCritique(agent="openai-api", severity=0.5),
        ]
    )
    participants: list = field(default_factory=lambda: ["anthropic-api", "openai-api"])


def _mock_findings():
    return {
        "unanimous_critiques": ["SQL injection vulnerability"],
        "split_opinions": [],
        "risk_areas": [],
        "agreement_score": 0.8,
        "agent_alignment": {},
        "critical_issues": [],
        "high_issues": [],
        "medium_issues": [],
        "low_issues": [],
        "all_critiques": [],
        "final_summary": "Review complete.",
        "agents_used": ["anthropic-api", "openai-api"],
    }


def _make_args(**overrides):
    defaults = {
        "pr_url": None,
        "diff_file": None,
        "agents": "anthropic-api,openai-api",
        "rounds": 2,
        "focus": "security,performance,quality",
        "output_format": "github",
        "output_dir": None,
        "demo": False,
        "share": False,
        "sarif": None,
        "gauntlet": False,
        "ci": False,
        "post_comment": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestPersistReviewToKM:
    """Unit tests for the _persist_review_to_km helper function."""

    def test_persist_passes_findings_dict_to_from_review_result(self):
        """from_review_result receives the findings dict, not the DebateResult."""
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {"receipt_id": "test-123"}
        mock_adapter = MagicMock()

        findings = _mock_findings()
        result = MockDebateResult()

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ) as mock_get_adapter:
            mock_cls.from_review_result.return_value = mock_receipt
            mock_get_adapter.return_value = mock_adapter

            success = _persist_review_to_km(result, findings)

        assert success is True
        # Verify findings dict is passed as the first positional arg
        mock_cls.from_review_result.assert_called_once_with(
            findings,
            pr_url=None,
            reviewer_agents=["anthropic-api", "openai-api"],
        )

    def test_persist_passes_pr_url(self):
        """pr_url is forwarded to from_review_result."""
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {"receipt_id": "test-456"}
        mock_adapter = MagicMock()

        findings = _mock_findings()
        result = MockDebateResult()

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ) as mock_get_adapter:
            mock_cls.from_review_result.return_value = mock_receipt
            mock_get_adapter.return_value = mock_adapter

            pr = "https://github.com/owner/repo/pull/42"
            success = _persist_review_to_km(result, findings, pr_url=pr)

        assert success is True
        mock_cls.from_review_result.assert_called_once_with(
            findings,
            pr_url=pr,
            reviewer_agents=["anthropic-api", "openai-api"],
        )

    def test_persist_calls_ingest_with_dict(self):
        """adapter.ingest() receives receipt.to_dict(), not the receipt object."""
        receipt_dict = {"receipt_id": "r-789", "verdict": "PASS"}
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = receipt_dict
        mock_adapter = MagicMock()

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ) as mock_get_adapter:
            mock_cls.from_review_result.return_value = mock_receipt
            mock_get_adapter.return_value = mock_adapter

            _persist_review_to_km(MockDebateResult(), _mock_findings())

        mock_adapter.ingest.assert_called_once_with(receipt_dict)

    def test_persist_import_error_returns_false(self):
        """ImportError from missing KM deps returns False without raising."""
        original_import = __import__

        def block_km(name, *args, **kwargs):
            if "receipt_adapter" in name or "receipt_models" in name:
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_km):
            result = _persist_review_to_km(MockDebateResult(), _mock_findings())

        assert result is False

    def test_persist_runtime_error_returns_false(self):
        """Runtime error during persistence returns False without raising."""
        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ):
            mock_cls.from_review_result.side_effect = RuntimeError("KM down")
            result = _persist_review_to_km(MockDebateResult(), _mock_findings())

        assert result is False

    def test_persist_empty_agents_passes_none(self):
        """When agents_used is empty, reviewer_agents is passed as None."""
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {}

        findings = _mock_findings()
        findings["agents_used"] = []

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ) as mock_get_adapter:
            mock_cls.from_review_result.return_value = mock_receipt
            mock_get_adapter.return_value = MagicMock()

            _persist_review_to_km(MockDebateResult(), findings)

        _, kwargs = mock_cls.from_review_result.call_args
        assert kwargs["reviewer_agents"] is None


class TestReviewKMPersistence:
    """Integration tests for KM persistence in cmd_review flow."""

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    def test_km_persistence_attempted(
        self, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """KM persistence is attempted after findings extraction."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {"receipt_id": "test"}
        mock_adapter = MagicMock()

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_receipt_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ) as mock_get_adapter:
            mock_receipt_cls.from_review_result.return_value = mock_receipt
            mock_get_adapter.return_value = mock_adapter

            args = _make_args(diff_file=str(diff_file))
            result = cmd_review(args)

        assert result == 0
        mock_receipt_cls.from_review_result.assert_called_once()
        mock_adapter.ingest.assert_called_once_with(mock_receipt.to_dict())

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    def test_km_import_error_graceful(
        self, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """ImportError in KM persistence does not fail the review."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        original_import = __import__

        def block_km_import(name, *args, **kwargs):
            if "receipt_adapter" in name or "aragora.gauntlet.receipt_models" in name:
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_km_import):
            args = _make_args(diff_file=str(diff_file))
            result = cmd_review(args)

        assert result == 0

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    def test_km_runtime_error_graceful(
        self, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """Runtime error in KM persistence does not fail the review."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        with patch(
            "aragora.gauntlet.receipt_models.DecisionReceipt"
        ) as mock_receipt_cls, patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter"
        ):
            mock_receipt_cls.from_review_result.side_effect = RuntimeError("KM down")

            args = _make_args(diff_file=str(diff_file))
            result = cmd_review(args)

        assert result == 0


class TestPostComment:
    """Tests for --post-comment flag."""

    @patch("aragora.cli.review.subprocess.run")
    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review._persist_review_to_km")
    def test_post_comment_calls_gh(
        self, mock_persist, mock_agents, mock_extract, mock_debate, mock_run, tmp_path
    ):
        """--post-comment posts formatted comment via gh pr comment."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        findings = _mock_findings()
        mock_extract.return_value = findings
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = _make_args(
            diff_file=str(diff_file),
            pr_url="https://github.com/owner/repo/pull/99",
            post_comment=True,
        )
        result = cmd_review(args)

        assert result == 0
        # Verify gh pr comment was called
        gh_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "gh"]
        assert len(gh_calls) >= 1
        gh_cmd = gh_calls[-1][0][0]
        assert gh_cmd[:3] == ["gh", "pr", "comment"]
        assert "99" in gh_cmd
        assert "--body" in gh_cmd

    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review._persist_review_to_km")
    def test_post_comment_without_pr_url_errors(
        self, mock_persist, mock_agents, mock_extract, mock_debate, tmp_path
    ):
        """--post-comment without pr_url returns error code 1."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = _make_args(
            diff_file=str(diff_file),
            pr_url=None,
            post_comment=True,
        )
        result = cmd_review(args)

        assert result == 1

    @patch("aragora.cli.review.subprocess.run")
    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review._persist_review_to_km")
    def test_post_comment_gh_failure_graceful(
        self, mock_persist, mock_agents, mock_extract, mock_debate, mock_run, tmp_path
    ):
        """gh pr comment failure does not crash the review (returns 0)."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth error")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = _make_args(
            diff_file=str(diff_file),
            pr_url="https://github.com/owner/repo/pull/55",
            post_comment=True,
        )
        result = cmd_review(args)

        # gh failure is a warning, not a fatal error
        assert result == 0

    @patch("aragora.cli.review.subprocess.run")
    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review._persist_review_to_km")
    def test_post_comment_gh_not_found_graceful(
        self, mock_persist, mock_agents, mock_extract, mock_debate, mock_run, tmp_path
    ):
        """Missing gh CLI does not crash the review."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()
        mock_run.side_effect = FileNotFoundError("gh not found")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = _make_args(
            diff_file=str(diff_file),
            pr_url="https://github.com/owner/repo/pull/10",
            post_comment=True,
        )
        result = cmd_review(args)

        assert result == 0

    @patch("aragora.cli.review.subprocess.run")
    @patch("aragora.cli.review.run_review_debate")
    @patch("aragora.cli.review.extract_review_findings")
    @patch("aragora.cli.review.get_available_agents")
    @patch("aragora.cli.review._persist_review_to_km")
    def test_post_comment_includes_repo_arg(
        self, mock_persist, mock_agents, mock_extract, mock_debate, mock_run, tmp_path
    ):
        """gh pr comment includes --repo for cross-repo support."""
        mock_agents.return_value = "anthropic-api,openai-api"
        mock_debate.return_value = MockDebateResult()
        mock_extract.return_value = _mock_findings()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n")

        args = _make_args(
            diff_file=str(diff_file),
            pr_url="https://github.com/myorg/myrepo/pull/77",
            post_comment=True,
        )
        cmd_review(args)

        gh_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "gh"]
        gh_cmd = gh_calls[-1][0][0]
        assert "--repo" in gh_cmd
        repo_idx = gh_cmd.index("--repo")
        assert gh_cmd[repo_idx + 1] == "myorg/myrepo"
