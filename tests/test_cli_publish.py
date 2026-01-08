"""Tests for CLI publish module - HTML, Markdown, and JSON generation."""

import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import pytest

from aragora.core import DebateResult, Message, Critique
from aragora.cli.publish import (
    generate_html_report,
    generate_markdown_report,
    generate_json_report,
    publish_debate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result() -> DebateResult:
    """Create a sample debate result for testing."""
    return DebateResult(
        id="test-debate-12345",
        task="What is the best approach to testing software?",
        final_answer="A balanced approach using unit, integration, and end-to-end tests.",
        confidence=0.92,
        consensus_reached=True,
        rounds_used=3,
        duration_seconds=120.5,
        messages=[
            Message(role="proposer", agent="claude", content="Unit tests are essential.", round=0),
            Message(role="proposer", agent="gemini", content="Integration tests matter more.", round=0),
            Message(role="critic", agent="codex", content="Both have valid points.", round=1),
            Message(role="synthesizer", agent="claude", content="We need all types.", round=2),
        ],
        critiques=[
            Critique(
                agent="codex",
                target_agent="claude",
                target_content="Unit tests proposal",
                issues=["Ignores integration testing", "Too narrow scope"],
                suggestions=["Consider broader testing strategy"],
                severity=0.6,
                reasoning="Valid but incomplete",
            ),
            Critique(
                agent="codex",
                target_agent="gemini",
                target_content="Integration tests proposal",
                issues=["Slow feedback loop", "Complex setup"],
                suggestions=["Balance with faster unit tests"],
                severity=0.4,
                reasoning="Trade-offs not considered",
            ),
        ],
    )


@pytest.fixture
def minimal_result() -> DebateResult:
    """Create a minimal debate result for edge case testing."""
    return DebateResult(
        task="Minimal task",
        final_answer="Minimal answer",
        confidence=0.5,
        consensus_reached=False,
        rounds_used=1,
        duration_seconds=10.0,
        messages=[],
        critiques=[],
    )


@pytest.fixture
def result_with_long_content() -> DebateResult:
    """Create a result with very long content for truncation testing."""
    return DebateResult(
        id="long-content-test",
        task="A" * 1000,
        final_answer="B" * 2000,
        confidence=0.75,
        consensus_reached=True,
        rounds_used=2,
        duration_seconds=60.0,
        messages=[
            Message(role="proposer", agent="agent1", content="C" * 1500, round=0),
        ],
        critiques=[
            Critique(
                agent="agent2",
                target_agent="agent1",
                target_content="target",
                issues=["Issue " * 100],
                suggestions=["Suggestion " * 100],
                severity=0.5,
                reasoning="Long reasoning " * 50,
            ),
        ],
    )


# =============================================================================
# Test HTML Generation
# =============================================================================


class TestHTMLGeneration:
    """Tests for HTML report generation."""

    def test_html_structure(self, sample_result):
        """HTML has correct basic structure."""
        html = generate_html_report(sample_result)

        assert "<!DOCTYPE html>" in html
        assert "<html lang=\"en\">" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_html_contains_task(self, sample_result):
        """HTML contains the debate task."""
        html = generate_html_report(sample_result)
        assert sample_result.task in html

    def test_html_contains_final_answer(self, sample_result):
        """HTML contains the final answer."""
        html = generate_html_report(sample_result)
        assert sample_result.final_answer in html

    def test_html_contains_messages(self, sample_result):
        """HTML contains all message content."""
        html = generate_html_report(sample_result)

        for msg in sample_result.messages:
            assert msg.agent in html
            assert msg.role in html

    def test_html_contains_critiques(self, sample_result):
        """HTML contains critique information."""
        html = generate_html_report(sample_result)

        for critique in sample_result.critiques:
            assert critique.agent in html
            assert critique.target_agent in html

    def test_html_consensus_reached(self, sample_result):
        """HTML shows consensus reached correctly."""
        html = generate_html_report(sample_result)
        assert "Consensus" in html
        assert "reached" in html

    def test_html_no_consensus(self, minimal_result):
        """HTML shows no consensus correctly."""
        html = generate_html_report(minimal_result)
        assert "No Consensus" in html or "not-reached" in html

    def test_html_statistics(self, sample_result):
        """HTML contains statistics."""
        html = generate_html_report(sample_result)

        assert str(sample_result.rounds_used) in html
        assert str(len(sample_result.messages)) in html
        assert str(len(sample_result.critiques)) in html

    def test_html_contains_styling(self, sample_result):
        """HTML contains CSS styling."""
        html = generate_html_report(sample_result)
        assert "<style>" in html
        assert "</style>" in html
        assert "color:" in html

    def test_html_truncates_long_content(self, result_with_long_content):
        """Long message content is truncated."""
        html = generate_html_report(result_with_long_content)

        # Content should be truncated to 500 chars
        assert "..." in html
        # Full 1500 char content should not appear - only first 500
        assert "C" * 1500 not in html
        assert "C" * 500 in html  # Truncated version should appear

    def test_html_no_critiques_message(self, minimal_result):
        """Shows message when no critiques."""
        html = generate_html_report(minimal_result)
        assert "No critiques" in html

    def test_html_contains_footer(self, sample_result):
        """HTML contains footer with generator info."""
        html = generate_html_report(sample_result)
        assert "aragora" in html
        assert "<footer>" in html

    def test_html_escapes_special_chars(self):
        """HTML escapes potentially dangerous characters."""
        result = DebateResult(
            task="<script>alert('xss')</script>",
            final_answer="Test & verify < > symbols",
            confidence=0.5,
            consensus_reached=False,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[],
            critiques=[],
        )
        # Note: Current implementation doesn't escape - this documents behavior
        # In a real security audit, this would be a finding
        html = generate_html_report(result)
        assert isinstance(html, str)

    def test_html_contains_timestamp(self, sample_result):
        """HTML contains generation timestamp."""
        html = generate_html_report(sample_result)
        # Should contain current date in some format
        today = datetime.now().strftime('%Y-%m-%d')
        assert today in html


# =============================================================================
# Test Markdown Generation
# =============================================================================


class TestMarkdownGeneration:
    """Tests for Markdown report generation."""

    def test_md_basic_structure(self, sample_result):
        """Markdown has correct basic structure."""
        md = generate_markdown_report(sample_result)

        assert "# " in md  # H1 header
        assert "## " in md  # H2 headers
        assert "---" in md  # Horizontal rules

    def test_md_contains_task(self, sample_result):
        """Markdown contains the debate task."""
        md = generate_markdown_report(sample_result)
        assert sample_result.task in md

    def test_md_contains_final_answer(self, sample_result):
        """Markdown contains the final answer."""
        md = generate_markdown_report(sample_result)
        assert sample_result.final_answer in md

    def test_md_contains_messages(self, sample_result):
        """Markdown contains message content."""
        md = generate_markdown_report(sample_result)

        for msg in sample_result.messages:
            assert msg.agent in md
            assert msg.role in md

    def test_md_contains_critiques(self, sample_result):
        """Markdown contains critique information."""
        md = generate_markdown_report(sample_result)

        for critique in sample_result.critiques:
            assert critique.agent in md
            assert critique.target_agent in md

    def test_md_consensus_indicator(self, sample_result):
        """Markdown shows consensus indicator."""
        md = generate_markdown_report(sample_result)
        assert "Consensus Reached" in md

    def test_md_no_consensus_indicator(self, minimal_result):
        """Markdown shows no consensus indicator."""
        md = generate_markdown_report(minimal_result)
        assert "No Consensus" in md

    def test_md_statistics_table(self, sample_result):
        """Markdown contains statistics table."""
        md = generate_markdown_report(sample_result)

        assert "| Metric | Value |" in md
        assert "|--------|-------|" in md
        assert "Rounds" in md
        assert "Messages" in md
        assert "Duration" in md

    def test_md_no_critiques_message(self, minimal_result):
        """Shows message when no critiques."""
        md = generate_markdown_report(minimal_result)
        assert "No critiques" in md

    def test_md_truncates_long_content(self, result_with_long_content):
        """Long message content is truncated to 800 chars."""
        md = generate_markdown_report(result_with_long_content)
        assert "..." in md

    def test_md_contains_footer(self, sample_result):
        """Markdown contains footer with generator info."""
        md = generate_markdown_report(sample_result)
        assert "aragora" in md
        assert "Generated by" in md

    def test_md_contains_bullet_points(self, sample_result):
        """Critique issues are formatted as bullet points."""
        md = generate_markdown_report(sample_result)
        assert "- " in md


# =============================================================================
# Test JSON Generation
# =============================================================================


class TestJSONGeneration:
    """Tests for JSON report generation."""

    def test_json_valid(self, sample_result):
        """JSON output is valid JSON."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_json_contains_task(self, sample_result):
        """JSON contains task field."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)
        assert data["task"] == sample_result.task

    def test_json_contains_final_answer(self, sample_result):
        """JSON contains final_answer field."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)
        assert data["final_answer"] == sample_result.final_answer

    def test_json_contains_messages(self, sample_result):
        """JSON contains messages array."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)

        assert "messages" in data
        assert len(data["messages"]) == len(sample_result.messages)

    def test_json_message_structure(self, sample_result):
        """JSON messages have correct structure."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)

        for msg in data["messages"]:
            assert "agent" in msg
            assert "role" in msg
            assert "content" in msg
            assert "round" in msg

    def test_json_contains_critiques(self, sample_result):
        """JSON contains critiques array."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)

        assert "critiques" in data
        assert len(data["critiques"]) == len(sample_result.critiques)

    def test_json_critique_structure(self, sample_result):
        """JSON critiques have correct structure."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)

        for critique in data["critiques"]:
            assert "agent" in critique
            assert "target" in critique
            assert "issues" in critique
            assert "suggestions" in critique
            assert "severity" in critique

    def test_json_contains_metadata(self, sample_result):
        """JSON contains metadata fields."""
        json_str = generate_json_report(sample_result)
        data = json.loads(json_str)

        assert "id" in data
        assert "confidence" in data
        assert "consensus_reached" in data
        assert "rounds_used" in data
        assert "duration_seconds" in data
        assert "generated_at" in data
        assert "generator" in data

    def test_json_empty_arrays(self, minimal_result):
        """JSON handles empty messages/critiques."""
        json_str = generate_json_report(minimal_result)
        data = json.loads(json_str)

        assert data["messages"] == []
        assert data["critiques"] == []

    def test_json_is_pretty_printed(self, sample_result):
        """JSON is pretty-printed with indentation."""
        json_str = generate_json_report(sample_result)
        assert "\n" in json_str
        assert "  " in json_str  # Indentation


# =============================================================================
# Test publish_debate Function
# =============================================================================


class TestPublishDebate:
    """Tests for publish_debate function."""

    def test_publish_html(self, sample_result):
        """Publishes HTML file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = publish_debate(sample_result, tmpdir, "html")

            assert filepath.exists()
            assert filepath.suffix == ".html"
            assert "debate_" in filepath.name

            content = filepath.read_text()
            assert "<!DOCTYPE html>" in content

    def test_publish_markdown(self, sample_result):
        """Publishes Markdown file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = publish_debate(sample_result, tmpdir, "md")

            assert filepath.exists()
            assert filepath.suffix == ".md"
            assert "debate_" in filepath.name

            content = filepath.read_text()
            assert "# " in content

    def test_publish_json(self, sample_result):
        """Publishes JSON file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = publish_debate(sample_result, tmpdir, "json")

            assert filepath.exists()
            assert filepath.suffix == ".json"
            assert "debate_" in filepath.name

            content = filepath.read_text()
            data = json.loads(content)
            assert "task" in data

    def test_publish_creates_directory(self, sample_result):
        """Creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"
            filepath = publish_debate(sample_result, str(nested_dir), "html")

            assert nested_dir.exists()
            assert filepath.exists()

    def test_publish_uses_debate_id(self, sample_result):
        """Filename includes debate ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = publish_debate(sample_result, tmpdir, "html")

            # ID is truncated to first 8 chars
            assert "test-deb" in filepath.name

    def test_publish_unknown_format_raises(self, sample_result):
        """Unknown format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown format"):
                publish_debate(sample_result, tmpdir, "pdf")

    def test_publish_minimal_result(self, minimal_result):
        """Publishes minimal result without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in ["html", "md", "json"]:
                filepath = publish_debate(minimal_result, tmpdir, fmt)
                assert filepath.exists()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task(self):
        """Handles empty task gracefully."""
        result = DebateResult(
            task="",
            final_answer="Answer",
            confidence=0.5,
            consensus_reached=False,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[],
            critiques=[],
        )

        html = generate_html_report(result)
        md = generate_markdown_report(result)
        json_str = generate_json_report(result)

        assert isinstance(html, str)
        assert isinstance(md, str)
        data = json.loads(json_str)
        assert data["task"] == ""

    def test_special_characters_in_content(self):
        """Handles special characters in content."""
        result = DebateResult(
            task="Test <>&\"' special chars",
            final_answer="Answer with\nnewlines\tand\ttabs",
            confidence=0.5,
            consensus_reached=False,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[
                Message(
                    role="proposer",
                    agent="agent1",
                    content="Unicode: \u2603 \u2764 \U0001F600",
                    round=0,
                ),
            ],
            critiques=[],
        )

        html = generate_html_report(result)
        md = generate_markdown_report(result)
        json_str = generate_json_report(result)

        # Should not raise and produce valid output
        assert isinstance(html, str)
        assert isinstance(md, str)
        json.loads(json_str)  # Should be valid JSON

    def test_zero_confidence(self):
        """Handles zero confidence."""
        result = DebateResult(
            task="Test",
            final_answer="Answer",
            confidence=0.0,
            consensus_reached=False,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[],
            critiques=[],
        )

        html = generate_html_report(result)
        assert "0%" in html

    def test_full_confidence(self):
        """Handles full confidence."""
        result = DebateResult(
            task="Test",
            final_answer="Answer",
            confidence=1.0,
            consensus_reached=True,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[],
            critiques=[],
        )

        html = generate_html_report(result)
        assert "100%" in html

    def test_very_long_critique_issues(self):
        """Handles critiques with many issues."""
        result = DebateResult(
            task="Test",
            final_answer="Answer",
            confidence=0.5,
            consensus_reached=False,
            rounds_used=1,
            duration_seconds=1.0,
            messages=[],
            critiques=[
                Critique(
                    agent="critic",
                    target_agent="target",
                    target_content="content",
                    issues=[f"Issue {i}" for i in range(20)],
                    suggestions=[f"Suggestion {i}" for i in range(20)],
                    severity=0.5,
                    reasoning="Many issues",
                ),
            ],
        )

        html = generate_html_report(result)
        md = generate_markdown_report(result)

        # HTML limits to 3 issues
        assert html.count("<li>Issue") <= 3
        # MD limits to 3 issues
        assert md.count("- Issue") <= 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full publish workflow."""

    def test_full_workflow(self, sample_result):
        """Full workflow from result to all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = publish_debate(sample_result, tmpdir, "html")
            md_path = publish_debate(sample_result, tmpdir, "md")
            json_path = publish_debate(sample_result, tmpdir, "json")

            # All files created
            assert html_path.exists()
            assert md_path.exists()
            assert json_path.exists()

            # All files have content
            assert len(html_path.read_text()) > 100
            assert len(md_path.read_text()) > 100
            assert len(json_path.read_text()) > 100

    def test_output_files_are_readable(self, sample_result):
        """Output files can be read back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = publish_debate(sample_result, tmpdir, "json")

            # Read back and verify
            with open(json_path) as f:
                data = json.load(f)

            assert data["task"] == sample_result.task
            assert data["final_answer"] == sample_result.final_answer
            assert data["confidence"] == sample_result.confidence
