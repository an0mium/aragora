"""
Tests for CLI review module.

Tests PR code review functionality.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.review import (
    DEFAULT_REVIEW_AGENTS,
    DEFAULT_ROUNDS,
    MAX_DIFF_SIZE,
    REVIEWS_DIR,
    SHARE_BASE_URL,
    build_review_prompt,
    cmd_review,
    format_github_comment,
    generate_review_id,
    get_available_agents,
    get_demo_findings,
    get_shareable_url,
    save_review_for_sharing,
)


class TestConstants:
    """Tests for module constants."""

    def test_default_agents_defined(self):
        """Default review agents are defined."""
        assert DEFAULT_REVIEW_AGENTS
        assert "," in DEFAULT_REVIEW_AGENTS  # At least 2 agents

    def test_default_rounds_positive(self):
        """Default rounds is positive."""
        assert DEFAULT_ROUNDS > 0

    def test_max_diff_size_reasonable(self):
        """Max diff size is reasonable (10KB - 100KB)."""
        assert 10000 <= MAX_DIFF_SIZE <= 100000

    def test_share_base_url_is_https(self):
        """Share base URL is HTTPS."""
        assert SHARE_BASE_URL.startswith("https://")


class TestGenerateReviewId:
    """Tests for generate_review_id function."""

    def test_returns_string(self):
        """generate_review_id returns a string."""
        findings = {"test": "data"}
        diff_hash = "abc123"
        review_id = generate_review_id(findings, diff_hash)
        assert isinstance(review_id, str)

    def test_returns_unique_ids(self):
        """generate_review_id returns unique IDs."""
        findings = {"test": "data"}
        ids = {generate_review_id(findings, "hash") for _ in range(10)}
        assert len(ids) == 10  # All unique

    def test_returns_reasonable_length(self):
        """generate_review_id returns reasonable length ID."""
        review_id = generate_review_id({}, "hash")
        assert 5 <= len(review_id) <= 20


class TestGetShareableUrl:
    """Tests for get_shareable_url function."""

    def test_returns_full_url(self):
        """get_shareable_url returns full URL with ID."""
        url = get_shareable_url("abc123")
        assert SHARE_BASE_URL in url
        assert "abc123" in url

    def test_url_format(self):
        """get_shareable_url has correct format."""
        url = get_shareable_url("test-id")
        assert url == f"{SHARE_BASE_URL}/test-id"


class TestGetAvailableAgents:
    """Tests for get_available_agents function."""

    @patch.dict("os.environ", {}, clear=True)
    def test_no_keys_returns_empty(self):
        """Returns empty string when no API keys."""
        agents = get_available_agents()
        assert agents == ""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
    def test_anthropic_only(self):
        """Returns anthropic-api when only Anthropic key set."""
        agents = get_available_agents()
        assert "anthropic-api" in agents

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_openai_only(self):
        """Returns openai-api when only OpenAI key set."""
        agents = get_available_agents()
        assert "openai-api" in agents

    @patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "key1", "OPENAI_API_KEY": "key2"},
        clear=True,
    )
    def test_both_primary_providers(self):
        """Returns both when Anthropic and OpenAI keys set."""
        agents = get_available_agents()
        assert "anthropic-api" in agents
        assert "openai-api" in agents


class TestGetDemoFindings:
    """Tests for get_demo_findings function."""

    def test_returns_dict(self):
        """get_demo_findings returns a dictionary."""
        findings = get_demo_findings()
        assert isinstance(findings, dict)

    def test_has_required_keys(self):
        """get_demo_findings has all required keys."""
        findings = get_demo_findings()
        required_keys = [
            "unanimous_critiques",
            "split_opinions",
            "risk_areas",
            "agreement_score",
            "critical_issues",
            "high_issues",
            "final_summary",
        ]
        for key in required_keys:
            assert key in findings, f"Missing key: {key}"

    def test_unanimous_critiques_not_empty(self):
        """Demo findings have unanimous critiques."""
        findings = get_demo_findings()
        assert len(findings["unanimous_critiques"]) > 0

    def test_agreement_score_valid(self):
        """Agreement score is between 0 and 1."""
        findings = get_demo_findings()
        assert 0 <= findings["agreement_score"] <= 1


class TestBuildReviewPrompt:
    """Tests for build_review_prompt function."""

    def test_includes_diff(self):
        """build_review_prompt includes the diff."""
        diff = "--- a/file.py\n+++ b/file.py\n+new line"
        prompt = build_review_prompt(diff)
        assert diff in prompt or "new line" in prompt

    def test_includes_security_focus(self):
        """build_review_prompt includes security focus by default."""
        prompt = build_review_prompt("test diff")
        assert "security" in prompt.lower()

    def test_includes_performance_focus(self):
        """build_review_prompt includes performance focus by default."""
        prompt = build_review_prompt("test diff")
        assert "performance" in prompt.lower()

    def test_includes_quality_focus(self):
        """build_review_prompt includes quality focus by default."""
        prompt = build_review_prompt("test diff")
        assert "quality" in prompt.lower()

    def test_truncates_large_diffs(self):
        """build_review_prompt truncates very large diffs."""
        large_diff = "A" * (MAX_DIFF_SIZE * 2)
        prompt = build_review_prompt(large_diff)
        assert len(prompt) < len(large_diff)
        assert "truncated" in prompt.lower()

    def test_custom_focus_areas(self):
        """build_review_prompt respects custom focus areas."""
        prompt = build_review_prompt("diff", focus_areas=["security"])
        assert "security" in prompt.lower()
        # Performance should not be prominently featured when not requested
        # (though it may still appear in default template text)


class TestFormatGithubComment:
    """Tests for format_github_comment function."""

    def test_includes_header(self):
        """format_github_comment includes AI Red Team header."""
        findings = get_demo_findings()
        comment = format_github_comment(None, findings)
        assert "AI Red Team" in comment

    def test_includes_unanimous_issues(self):
        """format_github_comment includes unanimous issues."""
        findings = get_demo_findings()
        comment = format_github_comment(None, findings)
        assert "Unanimous" in comment

    def test_includes_agreement_score(self):
        """format_github_comment includes agreement score."""
        findings = get_demo_findings()
        comment = format_github_comment(None, findings)
        # Agreement score should be formatted as percentage
        assert "%" in comment

    def test_includes_aragora_link(self):
        """format_github_comment includes Aragora link."""
        findings = get_demo_findings()
        comment = format_github_comment(None, findings)
        assert "aragora" in comment.lower()

    def test_returns_string(self):
        """format_github_comment returns a string."""
        findings = get_demo_findings()
        comment = format_github_comment(None, findings)
        assert isinstance(comment, str)


class TestSaveReviewForSharing:
    """Tests for save_review_for_sharing function."""

    def test_creates_file(self, tmp_path):
        """save_review_for_sharing creates a JSON file."""
        with patch("aragora.cli.review.REVIEWS_DIR", tmp_path):
            review_id = "test-123"
            path = save_review_for_sharing(
                review_id=review_id,
                findings={"test": "data"},
                diff="test diff",
                agents="agent1,agent2",
            )

            assert path.exists()
            assert path.suffix == ".json"

    def test_file_contains_review_data(self, tmp_path):
        """Saved file contains review data."""
        import json

        with patch("aragora.cli.review.REVIEWS_DIR", tmp_path):
            review_id = "test-456"
            save_review_for_sharing(
                review_id=review_id,
                findings={"agreement_score": 0.8},
                diff="test diff content",
                agents="agent1,agent2",
                pr_url="https://github.com/test/repo/pull/1",
            )

            file_path = tmp_path / f"{review_id}.json"
            with open(file_path) as f:
                data = json.load(f)

            assert data["id"] == review_id
            assert "test diff" in data["diff_preview"]
            assert data["pr_url"] == "https://github.com/test/repo/pull/1"


class TestCmdReview:
    """Tests for cmd_review CLI function."""

    def test_demo_mode_works(self, capsys):
        """cmd_review demo mode works without API keys."""
        args = argparse.Namespace(
            demo=True,
            output_format="github",
            output_dir=None,
            diff_file=None,
            pr_url=None,
            agents=DEFAULT_REVIEW_AGENTS,
            rounds=DEFAULT_ROUNDS,
            focus="security,performance,quality",
            share=False,
        )

        result = cmd_review(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DEMO MODE" in captured.out or "demo" in captured.err.lower()

    def test_demo_mode_json_output(self, capsys):
        """cmd_review demo mode can output JSON."""
        args = argparse.Namespace(
            demo=True,
            output_format="json",
            output_dir=None,
            diff_file=None,
            pr_url=None,
            agents=DEFAULT_REVIEW_AGENTS,
            rounds=DEFAULT_ROUNDS,
            focus="security",
            share=False,
        )

        result = cmd_review(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should be valid JSON
        import json

        json.loads(captured.out)

    def test_missing_diff_returns_error(self, capsys, monkeypatch):
        """cmd_review returns error when no diff provided."""
        # Simulate TTY (no stdin piping)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        args = argparse.Namespace(
            demo=False,
            output_format="github",
            output_dir=None,
            diff_file=None,
            pr_url=None,
            agents=DEFAULT_REVIEW_AGENTS,
            rounds=DEFAULT_ROUNDS,
            focus="security",
            share=False,
        )

        result = cmd_review(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_diff_file_not_found_returns_error(self, capsys):
        """cmd_review returns error for non-existent diff file."""
        args = argparse.Namespace(
            demo=False,
            output_format="github",
            output_dir=None,
            diff_file="/nonexistent/file.diff",
            pr_url=None,
            agents=DEFAULT_REVIEW_AGENTS,
            rounds=DEFAULT_ROUNDS,
            focus="security",
            share=False,
        )

        result = cmd_review(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_reads_diff_from_file(self, tmp_path, capsys):
        """cmd_review reads diff from file."""
        diff_file = tmp_path / "test.diff"
        diff_file.write_text("--- a/file.py\n+++ b/file.py\n+new line")

        # Patch to avoid actual API calls
        with patch.dict("os.environ", {}, clear=True):
            args = argparse.Namespace(
                demo=False,
                output_format="github",
                output_dir=None,
                diff_file=str(diff_file),
                pr_url=None,
                agents=DEFAULT_REVIEW_AGENTS,
                rounds=DEFAULT_ROUNDS,
                focus="security",
                share=False,
            )

            result = cmd_review(args)

            # Should fail due to no API keys, but will have attempted to read file
            captured = capsys.readouterr()
            assert result == 1  # No API keys
            assert "API" in captured.err or "key" in captured.err.lower()
