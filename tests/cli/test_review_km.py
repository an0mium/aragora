"""Tests for review CLI Knowledge Mound persistence.

Tests that:
1. Review findings are persisted to KM after extraction
2. Graceful fallback when KM dependencies are unavailable
3. Graceful fallback when KM persistence fails at runtime
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.review import cmd_review


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


class TestReviewKMPersistence:
    """Tests for KM persistence in review flow."""

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
        mock_adapter = MagicMock()

        with patch(
            "aragora.cli.review.DecisionReceipt", create=True
        ) as mock_receipt_cls, patch(
            "aragora.cli.review.get_receipt_adapter", create=True
        ) as mock_get_adapter:
            # These patches won't intercept the lazy imports inside the try block.
            # Instead, patch the actual import targets.
            pass

        # Use the real import paths inside the try block
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
        mock_adapter.ingest.assert_called_once_with(mock_receipt)

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
        ) as mock_get_adapter:
            mock_receipt_cls.from_review_result.side_effect = RuntimeError("KM down")

            args = _make_args(diff_file=str(diff_file))
            result = cmd_review(args)

        assert result == 0
