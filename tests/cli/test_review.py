"""
Tests for aragora.cli.review module.

Tests multi-agent code review CLI commands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.review import (
    DEFAULT_REVIEW_AGENTS,
    DEFAULT_ROUNDS,
    MAX_DIFF_SIZE,
    build_review_prompt,
    cmd_review,
    create_review_parser,
    extract_review_findings,
    format_github_comment,
    generate_review_id,
    get_available_agents,
    get_demo_findings,
    get_shareable_url,
    save_review_for_sharing,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockCritique:
    """Mock critique."""

    agent: str = "anthropic-api"
    target_agent: str = "openai-api"
    issues: list = field(default_factory=lambda: ["Issue 1", "Issue 2"])
    suggestions: list = field(default_factory=lambda: ["Fix this"])
    severity: float = 0.7


@dataclass
class MockMessage:
    """Mock message."""

    agent: str = "anthropic-api"


@dataclass
class MockDebateResult:
    """Mock debate result."""

    final_answer: str = "Review complete"
    votes: dict = field(default_factory=dict)
    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)


# ===========================================================================
# Tests: generate_review_id
# ===========================================================================


class TestGenerateReviewId:
    """Tests for generate_review_id function."""

    def test_generates_unique_id(self):
        """Test that unique IDs are generated."""
        findings = {"critical_issues": []}
        diff_hash = "abc123"

        id1 = generate_review_id(findings, diff_hash)
        id2 = generate_review_id(findings, diff_hash)

        assert id1 != id2  # UUIDs should be different
        assert len(id1) == 8
        assert len(id2) == 8

    def test_returns_8_char_string(self):
        """Test that ID is exactly 8 characters."""
        findings = {}
        diff_hash = "test_hash"

        review_id = generate_review_id(findings, diff_hash)

        assert len(review_id) == 8
        assert review_id.isalnum()


# ===========================================================================
# Tests: save_review_for_sharing
# ===========================================================================


class TestSaveReviewForSharing:
    """Tests for save_review_for_sharing function."""

    def test_saves_review_to_file(self, tmp_path, monkeypatch):
        """Test that review is saved to file."""
        # Mock the REVIEWS_DIR to use temp directory
        reviews_dir = tmp_path / "reviews"
        monkeypatch.setattr("aragora.cli.review.REVIEWS_DIR", reviews_dir)

        findings = {
            "unanimous_critiques": ["Issue 1"],
            "split_opinions": [],
            "risk_areas": [],
            "agreement_score": 0.9,
            "critical_issues": [],
            "high_issues": [],
            "medium_issues": [],
            "low_issues": [],
            "final_summary": "Test summary",
        }

        result = save_review_for_sharing(
            review_id="test123",
            findings=findings,
            diff="diff content",
            agents="anthropic-api,openai-api",
            pr_url="https://github.com/owner/repo/pull/1",
        )

        assert result.exists()
        content = json.loads(result.read_text())
        assert content["id"] == "test123"
        assert content["pr_url"] == "https://github.com/owner/repo/pull/1"
        assert len(content["agents"]) == 2

    def test_truncates_long_diff(self, tmp_path, monkeypatch):
        """Test that long diffs are truncated in preview."""
        reviews_dir = tmp_path / "reviews"
        monkeypatch.setattr("aragora.cli.review.REVIEWS_DIR", reviews_dir)

        long_diff = "x" * 1000
        findings = {"unanimous_critiques": [], "final_summary": ""}

        result = save_review_for_sharing(
            review_id="test456",
            findings=findings,
            diff=long_diff,
            agents="anthropic-api",
        )

        content = json.loads(result.read_text())
        assert len(content["diff_preview"]) < len(long_diff)
        assert content["diff_preview"].endswith("...")


# ===========================================================================
# Tests: get_shareable_url
# ===========================================================================


class TestGetShareableUrl:
    """Tests for get_shareable_url function."""

    def test_returns_correct_url(self):
        """Test that URL is correctly formatted."""
        url = get_shareable_url("abc123")
        assert url == "https://aragora.ai/reviews/abc123"


# ===========================================================================
# Tests: get_available_agents
# ===========================================================================


class TestGetAvailableAgents:
    """Tests for get_available_agents function."""

    def test_no_keys_returns_empty(self, monkeypatch):
        """Test returns empty when no API keys."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = get_available_agents()
        assert result == ""

    def test_anthropic_only(self, monkeypatch):
        """Test with only Anthropic key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = get_available_agents()
        assert "anthropic-api" in result

    def test_openai_only(self, monkeypatch):
        """Test with only OpenAI key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = get_available_agents()
        assert "openai-api" in result

    def test_both_default_agents(self, monkeypatch):
        """Test with both default agent keys."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = get_available_agents()
        assert "anthropic-api" in result
        assert "openai-api" in result

    def test_openrouter_fallback(self, monkeypatch):
        """Test OpenRouter is used as fallback."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = get_available_agents()
        assert "anthropic-api" in result
        assert "openrouter-api" in result


# ===========================================================================
# Tests: get_demo_findings
# ===========================================================================


class TestGetDemoFindings:
    """Tests for get_demo_findings function."""

    def test_returns_demo_data(self):
        """Test that demo findings are returned."""
        findings = get_demo_findings()

        assert "unanimous_critiques" in findings
        assert len(findings["unanimous_critiques"]) > 0
        assert "SQL injection" in findings["unanimous_critiques"][0]
        assert "critical_issues" in findings
        assert "final_summary" in findings
        assert findings["agreement_score"] > 0


# ===========================================================================
# Tests: build_review_prompt
# ===========================================================================


class TestBuildReviewPrompt:
    """Tests for build_review_prompt function."""

    def test_builds_prompt_with_diff(self):
        """Test prompt includes diff."""
        diff = "diff --git a/test.py"
        prompt = build_review_prompt(diff)

        assert diff in prompt
        assert "Security" in prompt
        assert "Performance" in prompt
        assert "Code Quality" in prompt

    def test_focus_areas_filter(self):
        """Test that focus areas filter content."""
        diff = "test diff"

        # Only security
        prompt = build_review_prompt(diff, focus_areas=["security"])
        assert "Security" in prompt
        # Performance section should not be present
        assert "N+1 query" not in prompt

        # Only performance
        prompt = build_review_prompt(diff, focus_areas=["performance"])
        assert "N+1 query" in prompt
        # Security-specific terms not present
        assert "SQL/NoSQL injection" not in prompt

    def test_truncates_large_diff(self):
        """Test that large diffs are truncated."""
        large_diff = "x" * (MAX_DIFF_SIZE + 1000)
        prompt = build_review_prompt(large_diff)

        assert "[... diff truncated ...]" in prompt
        assert len(prompt) < len(large_diff) + 1000


# ===========================================================================
# Tests: extract_review_findings
# ===========================================================================


class TestExtractReviewFindings:
    """Tests for extract_review_findings function."""

    def test_extracts_from_debate_result(self):
        """Test extraction from debate result."""
        result = MockDebateResult(
            critiques=[MockCritique(severity=0.95), MockCritique(severity=0.5)],
            messages=[MockMessage(agent="anthropic-api")],
        )

        # Mock the DisagreementReporter
        mock_report = MagicMock()
        mock_report.unanimous_critiques = ["Issue agreed by all"]
        mock_report.split_opinions = []
        mock_report.risk_areas = []
        mock_report.agreement_score = 0.8
        mock_report.agent_alignment = {}

        with patch("aragora.cli.review.DisagreementReporter") as mock_reporter_class:
            mock_reporter_class.return_value.generate_report.return_value = mock_report
            findings = extract_review_findings(result)

        assert "unanimous_critiques" in findings
        assert "critical_issues" in findings
        assert "agents_used" in findings

    def test_categorizes_by_severity(self):
        """Test issues are categorized by severity."""
        result = MockDebateResult(
            critiques=[
                MockCritique(severity=0.95),  # Critical
                MockCritique(severity=0.75),  # High
                MockCritique(severity=0.5),  # Medium
                MockCritique(severity=0.2),  # Low
            ],
            messages=[],
        )

        mock_report = MagicMock()
        mock_report.unanimous_critiques = []
        mock_report.split_opinions = []
        mock_report.risk_areas = []
        mock_report.agreement_score = 0.5
        mock_report.agent_alignment = {}

        with patch("aragora.cli.review.DisagreementReporter") as mock_reporter_class:
            mock_reporter_class.return_value.generate_report.return_value = mock_report
            findings = extract_review_findings(result)

        assert len(findings["critical_issues"]) >= 1
        assert len(findings["high_issues"]) >= 1


# ===========================================================================
# Tests: format_github_comment
# ===========================================================================


class TestFormatGithubComment:
    """Tests for format_github_comment function."""

    def test_formats_basic_comment(self):
        """Test basic comment formatting."""
        findings = {
            "unanimous_critiques": [],
            "split_opinions": [],
            "risk_areas": [],
            "agreement_score": 0.8,
            "critical_issues": [],
            "high_issues": [],
            "agents_used": ["anthropic-api", "openai-api"],
            "final_summary": "",
        }

        comment = format_github_comment(None, findings)

        assert "## Multi Agent Code Review" in comment
        assert "2 agents reviewed" in comment
        assert "Agreement score: 80%" in comment

    def test_includes_unanimous_issues(self):
        """Test unanimous issues are included."""
        findings = {
            "unanimous_critiques": ["SQL injection found", "Missing validation"],
            "split_opinions": [],
            "risk_areas": [],
            "agreement_score": 0.9,
            "critical_issues": [],
            "high_issues": [],
            "agents_used": ["anthropic-api"],
            "final_summary": "",
        }

        comment = format_github_comment(None, findings)

        assert "### Unanimous Issues" in comment
        assert "SQL injection found" in comment

    def test_includes_critical_issues(self):
        """Test critical issues are included."""
        findings = {
            "unanimous_critiques": [],
            "split_opinions": [],
            "risk_areas": [],
            "agreement_score": 0.7,
            "critical_issues": [{"issue": "Critical security flaw"}],
            "high_issues": [{"issue": "High severity bug"}],
            "agents_used": [],
            "final_summary": "",
        }

        comment = format_github_comment(None, findings)

        assert "### Critical & High Severity Issues" in comment
        assert "CRITICAL" in comment

    def test_includes_split_opinions_table(self):
        """Test split opinions table is formatted."""
        findings = {
            "unanimous_critiques": [],
            "split_opinions": [
                ("Add rate limiting", ["anthropic-api"], ["openai-api"]),
            ],
            "risk_areas": [],
            "agreement_score": 0.6,
            "critical_issues": [],
            "high_issues": [],
            "agents_used": [],
            "final_summary": "",
        }

        comment = format_github_comment(None, findings)

        assert "### Split Opinions" in comment
        assert "Add rate limiting" in comment
        assert "| Topic |" in comment  # Table header


# ===========================================================================
# Tests: create_review_parser
# ===========================================================================


class TestCreateReviewParser:
    """Tests for create_review_parser function."""

    def test_creates_parser_with_defaults(self):
        """Test parser creation with defaults."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review"])
        assert args.agents == DEFAULT_REVIEW_AGENTS
        assert args.rounds == DEFAULT_ROUNDS
        assert args.output_format == "github"

    def test_parser_with_pr_url(self):
        """Test parser with PR URL argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(["review", "https://github.com/owner/repo/pull/123"])
        assert args.pr_url == "https://github.com/owner/repo/pull/123"

    def test_parser_with_all_options(self):
        """Test parser with all options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_review_parser(subparsers)

        args = parser.parse_args(
            [
                "review",
                "--diff-file",
                "test.diff",
                "--agents",
                "anthropic-api,gemini-api",
                "--rounds",
                "3",
                "--focus",
                "security",
                "--output-format",
                "json",
                "--output-dir",
                "./output",
                "--demo",
                "--share",
            ]
        )

        assert args.diff_file == "test.diff"
        assert args.agents == "anthropic-api,gemini-api"
        assert args.rounds == 3
        assert args.focus == "security"
        assert args.output_format == "json"
        assert args.output_dir == "./output"
        assert args.demo is True
        assert args.share is True


# ===========================================================================
# Tests: cmd_review
# ===========================================================================


class TestCmdReview:
    """Tests for cmd_review function."""

    @pytest.fixture
    def review_args(self):
        """Create base review args."""
        args = argparse.Namespace()
        args.pr_url = None
        args.diff_file = None
        args.agents = DEFAULT_REVIEW_AGENTS
        args.rounds = DEFAULT_ROUNDS
        args.focus = "security,performance,quality"
        args.output_format = "github"
        args.output_dir = None
        args.demo = False
        args.share = False
        return args

    def test_demo_mode_github_output(self, review_args, capsys, monkeypatch):
        """Test demo mode with github output."""
        review_args.demo = True
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "[DEMO MODE]" in captured.out
        assert "## Multi Agent Code Review" in captured.out

    def test_demo_mode_json_output(self, review_args, capsys, monkeypatch):
        """Test demo mode with JSON output."""
        review_args.demo = True
        review_args.output_format = "json"
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["demo_mode"] is True
        assert "unanimous_critiques" in output

    def test_demo_mode_saves_to_output_dir(self, review_args, tmp_path, monkeypatch):
        """Test demo mode saves to output directory."""
        review_args.demo = True
        review_args.output_format = "github"
        review_args.output_dir = str(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 0
        assert (tmp_path / "comment.md").exists()

    def test_error_no_diff_provided(self, review_args, capsys, monkeypatch):
        """Test error when no diff provided."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No diff provided" in captured.err

    def test_diff_file_not_found(self, review_args, capsys, monkeypatch):
        """Test error when diff file not found."""
        review_args.diff_file = "/nonexistent/diff.patch"
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Diff file not found" in captured.err

    def test_reads_diff_from_file(self, review_args, tmp_path, capsys, monkeypatch):
        """Test reading diff from file."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("diff --git a/test.py\n+new line")
        review_args.diff_file = str(diff_file)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        # Mock the available agents check
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock the async debate
        mock_result = MockDebateResult(
            critiques=[],
            messages=[MockMessage(agent="anthropic-api")],
            final_answer="Review complete",
        )

        mock_report = MagicMock()
        mock_report.unanimous_critiques = []
        mock_report.split_opinions = []
        mock_report.risk_areas = []
        mock_report.agreement_score = 0.9
        mock_report.agent_alignment = {}

        with patch("aragora.cli.review.asyncio.run", return_value=mock_result):
            with patch("aragora.cli.review.DisagreementReporter") as mock_reporter:
                mock_reporter.return_value.generate_report.return_value = mock_report
                result = cmd_review(review_args)

        assert result == 0

    def test_empty_diff_error(self, review_args, tmp_path, capsys, monkeypatch):
        """Test error on empty diff."""
        diff_file = tmp_path / "empty.diff"
        diff_file.write_text("")
        review_args.diff_file = str(diff_file)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Empty diff" in captured.err

    def test_no_api_keys_error(self, review_args, tmp_path, capsys, monkeypatch):
        """Test error when no API keys configured."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("diff content")
        review_args.diff_file = str(diff_file)

        # Remove all API keys
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No API keys configured" in captured.err

    def test_json_output_format(self, review_args, tmp_path, capsys, monkeypatch):
        """Test JSON output format."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("diff content")
        review_args.diff_file = str(diff_file)
        review_args.output_format = "json"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        mock_result = MockDebateResult(
            critiques=[],
            messages=[],
            final_answer="Test summary",
        )

        mock_report = MagicMock()
        mock_report.unanimous_critiques = ["Issue 1"]
        mock_report.split_opinions = []
        mock_report.risk_areas = []
        mock_report.agreement_score = 0.8
        mock_report.agent_alignment = {}

        with patch("aragora.cli.review.asyncio.run", return_value=mock_result):
            with patch("aragora.cli.review.DisagreementReporter") as mock_reporter:
                mock_reporter.return_value.generate_report.return_value = mock_report
                result = cmd_review(review_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "unanimous_critiques" in output

    def test_review_error_handling(self, review_args, tmp_path, capsys, monkeypatch):
        """Test error handling during review."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("diff content")
        review_args.diff_file = str(diff_file)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("aragora.cli.review.asyncio.run", side_effect=Exception("API error")):
            result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error running review" in captured.err

    def test_pr_url_invalid_format(self, review_args, capsys, monkeypatch):
        """Test error on invalid PR URL format."""
        review_args.pr_url = "https://invalid-url.com/not-a-pr"
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        # Mock subprocess to fail on invalid URL
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Invalid PR")
            result = cmd_review(review_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid PR URL format" in captured.err

    def test_share_option_generates_link(self, review_args, tmp_path, capsys, monkeypatch):
        """Test that share option generates shareable link."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("diff content")
        review_args.diff_file = str(diff_file)
        review_args.share = True

        reviews_dir = tmp_path / "reviews"
        monkeypatch.setattr("aragora.cli.review.REVIEWS_DIR", reviews_dir)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        mock_result = MockDebateResult(
            critiques=[],
            messages=[],
            final_answer="Test summary",
        )

        mock_report = MagicMock()
        mock_report.unanimous_critiques = []
        mock_report.split_opinions = []
        mock_report.risk_areas = []
        mock_report.agreement_score = 0.9
        mock_report.agent_alignment = {}

        with patch("aragora.cli.review.asyncio.run", return_value=mock_result):
            with patch("aragora.cli.review.DisagreementReporter") as mock_reporter:
                mock_reporter.return_value.generate_report.return_value = mock_report
                result = cmd_review(review_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Shareable link:" in captured.err
        assert "aragora.ai/reviews" in captured.err
