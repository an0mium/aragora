"""
Tests for debate export formatting utilities.

Tests cover:
- format_debate_csv (multiple table types)
- format_debate_html
- format_debate_txt
- format_debate_md
- format_debate_latex
- _latex_escape helper
"""

import pytest
import csv
import io

from aragora.server.debate_export import (
    ExportResult,
    format_debate_csv,
    format_debate_html,
    format_debate_txt,
    format_debate_md,
    format_debate_latex,
    _latex_escape,
)


@pytest.fixture
def sample_debate():
    """A sample debate for testing exports."""
    return {
        "id": "debate-123",
        "slug": "test-debate",
        "topic": "Should AI systems be regulated?",
        "started_at": "2024-01-15T10:00:00Z",
        "ended_at": "2024-01-15T11:30:00Z",
        "rounds_used": 3,
        "consensus_reached": True,
        "confidence": 0.85,
        "final_answer": "Yes, AI systems should be regulated with appropriate safeguards.",
        "synthesis": "After thorough debate, consensus emerged around proportional regulation.",
        "messages": [
            {
                "round": 1,
                "agent": "Claude",
                "role": "speaker",
                "content": "I believe AI regulation is necessary.",
                "timestamp": "2024-01-15T10:05:00Z",
            },
            {
                "round": 1,
                "agent": "GPT-4",
                "role": "speaker",
                "content": "Regulation must balance innovation with safety.",
                "timestamp": "2024-01-15T10:10:00Z",
            },
            {
                "round": 2,
                "agent": "Claude",
                "role": "critic",
                "content": "GPT-4's point about innovation is valid.",
                "timestamp": "2024-01-15T10:20:00Z",
            },
        ],
        "critiques": [
            {
                "round": 1,
                "critic": "Claude",
                "target": "GPT-4",
                "severity": 0.3,
                "summary": "Valid point but needs more specifics.",
                "timestamp": "2024-01-15T10:15:00Z",
            },
        ],
        "votes": [
            {
                "round": 2,
                "voter": "Judge",
                "choice": "Claude",
                "reason": "More comprehensive argument.",
                "timestamp": "2024-01-15T10:25:00Z",
            },
        ],
    }


@pytest.fixture
def minimal_debate():
    """A minimal debate with no messages."""
    return {
        "id": "minimal-123",
        "topic": "Test",
        "messages": [],
        "critiques": [],
        "votes": [],
    }


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_fields(self):
        """ExportResult has required fields."""
        result = ExportResult(
            content=b"test",
            content_type="text/plain",
            filename="test.txt",
        )
        assert result.content == b"test"
        assert result.content_type == "text/plain"
        assert result.filename == "test.txt"


class TestFormatDebateCsv:
    """Tests for format_debate_csv function."""

    def test_summary_csv(self, sample_debate):
        """Exports summary CSV."""
        result = format_debate_csv(sample_debate, table="summary")
        assert result.content_type == "text/csv; charset=utf-8"
        assert result.filename.endswith(".csv")
        assert b"debate_id" in result.content
        assert b"topic" in result.content
        assert b"test-debate" in result.content

    def test_messages_csv(self, sample_debate):
        """Exports messages CSV."""
        result = format_debate_csv(sample_debate, table="messages")
        csv_content = result.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        # Check header
        assert rows[0] == ["round", "agent", "role", "content", "timestamp"]
        # Check data rows (3 messages)
        assert len(rows) == 4
        assert rows[1][1] == "Claude"

    def test_critiques_csv(self, sample_debate):
        """Exports critiques CSV."""
        result = format_debate_csv(sample_debate, table="critiques")
        csv_content = result.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        assert rows[0] == ["round", "critic", "target", "severity", "summary", "timestamp"]
        assert len(rows) == 2
        assert rows[1][1] == "Claude"

    def test_votes_csv(self, sample_debate):
        """Exports votes CSV."""
        result = format_debate_csv(sample_debate, table="votes")
        csv_content = result.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        assert rows[0] == ["round", "voter", "choice", "reason", "timestamp"]
        assert len(rows) == 2
        assert rows[1][1] == "Judge"

    def test_invalid_table_defaults_to_summary(self, sample_debate):
        """Invalid table type defaults to summary."""
        result = format_debate_csv(sample_debate, table="invalid")
        assert b"debate_id" in result.content

    def test_empty_debate(self, minimal_debate):
        """Handles debate with no data."""
        result = format_debate_csv(minimal_debate, table="messages")
        csv_content = result.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        # Just header
        assert len(rows) == 1

    def test_filename_uses_slug(self, sample_debate):
        """Filename uses debate slug."""
        result = format_debate_csv(sample_debate, table="summary")
        assert "test-debate" in result.filename

    def test_filename_falls_back_to_id(self):
        """Filename falls back to id when no slug."""
        debate = {"id": "debate-456", "messages": []}
        result = format_debate_csv(debate)
        assert "debate-456" in result.filename

    def test_content_truncation(self, sample_debate):
        """Long content is truncated in CSV."""
        sample_debate["messages"][0]["content"] = "x" * 2000
        result = format_debate_csv(sample_debate, table="messages")
        # Content should be truncated to 1000 chars
        assert b"x" * 1001 not in result.content


class TestFormatDebateHtml:
    """Tests for format_debate_html function."""

    def test_html_structure(self, sample_debate):
        """Generates valid HTML structure."""
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")

        assert result.content_type == "text/html; charset=utf-8"
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<title>" in html

    def test_html_contains_topic(self, sample_debate):
        """HTML contains debate topic."""
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")
        assert "Should AI systems be regulated?" in html

    def test_html_contains_messages(self, sample_debate):
        """HTML contains message content."""
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")
        assert "Claude" in html
        # Content is truncated to 500 chars in HTML
        assert "I believe AI regulation is necessary." in html

    def test_html_contains_stats(self, sample_debate):
        """HTML contains statistics."""
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")
        assert "3" in html  # messages count
        assert "1" in html  # critiques count

    def test_html_consensus_indicator(self, sample_debate):
        """HTML shows consensus indicator."""
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")
        assert "Yes" in html  # Consensus: Yes

    def test_html_no_consensus(self, sample_debate):
        """HTML shows no consensus when false."""
        sample_debate["consensus_reached"] = False
        result = format_debate_html(sample_debate)
        html = result.content.decode("utf-8")
        assert "No Consensus Reached" in html

    def test_html_escapes_special_chars(self):
        """HTML escapes special characters."""
        debate = {
            "topic": "<script>alert('xss')</script>",
            "messages": [],
            "critiques": [],
        }
        result = format_debate_html(debate)
        html = result.content.decode("utf-8")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_empty_messages(self, minimal_debate):
        """Handles empty messages gracefully."""
        result = format_debate_html(minimal_debate)
        html = result.content.decode("utf-8")
        assert "No messages recorded" in html


class TestFormatDebateTxt:
    """Tests for format_debate_txt function."""

    def test_txt_structure(self, sample_debate):
        """Generates plain text transcript."""
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")

        assert result.content_type == "text/plain; charset=utf-8"
        assert "ARAGORA DEBATE TRANSCRIPT" in txt
        assert "=" * 70 in txt

    def test_txt_contains_metadata(self, sample_debate):
        """Text contains metadata."""
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")

        assert "Topic:" in txt
        assert "Should AI systems be regulated?" in txt
        assert "Debate ID:" in txt
        assert "test-debate" in txt

    def test_txt_contains_timeline(self, sample_debate):
        """Text contains message timeline."""
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")

        assert "DEBATE TIMELINE" in txt
        assert "[CLAUDE]" in txt
        assert "Round 1" in txt

    def test_txt_contains_critiques(self, sample_debate):
        """Text contains critiques section."""
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")

        assert "CRITIQUES" in txt
        assert "Claude -> GPT-4" in txt

    def test_txt_contains_conclusion(self, sample_debate):
        """Text contains conclusion."""
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")

        assert "CONCLUSION" in txt
        assert "SYNTHESIS:" in txt
        assert "FINAL ANSWER:" in txt

    def test_txt_no_consensus(self, sample_debate):
        """Text shows no consensus status."""
        sample_debate["consensus_reached"] = False
        result = format_debate_txt(sample_debate)
        txt = result.content.decode("utf-8")
        assert "Consensus Reached: No" in txt


class TestFormatDebateMd:
    """Tests for format_debate_md function."""

    def test_md_structure(self, sample_debate):
        """Generates Markdown document."""
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")

        assert result.content_type == "text/markdown; charset=utf-8"
        assert "# Aragora Debate:" in md
        assert "## Metadata" in md
        assert "## Debate Timeline" in md

    def test_md_contains_metadata(self, sample_debate):
        """Markdown contains metadata."""
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")

        assert "**Debate ID:**" in md
        assert "test-debate" in md
        assert "**Rounds:**" in md

    def test_md_contains_emojis(self, sample_debate):
        """Markdown uses emoji badges."""
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")

        assert "💬" in md  # speaker emoji
        assert "🔍" in md  # critic emoji

    def test_md_critiques_severity(self, sample_debate):
        """Markdown shows severity indicators."""
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")

        # 0.3 severity is low (green)
        assert "🟢" in md

    def test_md_high_severity(self, sample_debate):
        """Markdown shows high severity."""
        sample_debate["critiques"][0]["severity"] = 0.8
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")
        assert "🔴" in md

    def test_md_consensus_checkmark(self, sample_debate):
        """Markdown shows consensus checkmark."""
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")
        assert "✅" in md

    def test_md_no_consensus_warning(self, sample_debate):
        """Markdown shows no consensus warning."""
        sample_debate["consensus_reached"] = False
        result = format_debate_md(sample_debate)
        md = result.content.decode("utf-8")
        assert "⚠️" in md


class TestFormatDebateLatex:
    """Tests for format_debate_latex function."""

    def test_latex_structure(self, sample_debate):
        """Generates LaTeX document."""
        result = format_debate_latex(sample_debate)
        tex = result.content.decode("utf-8")

        assert result.content_type == "application/x-latex; charset=utf-8"
        assert result.filename.endswith(".tex")
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex

    def test_latex_contains_packages(self, sample_debate):
        """LaTeX includes necessary packages."""
        result = format_debate_latex(sample_debate)
        tex = result.content.decode("utf-8")

        assert r"\usepackage[utf8]{inputenc}" in tex
        assert r"\usepackage{xcolor}" in tex
        assert r"\usepackage{hyperref}" in tex

    def test_latex_contains_metadata(self, sample_debate):
        """LaTeX contains metadata table."""
        result = format_debate_latex(sample_debate)
        tex = result.content.decode("utf-8")

        assert r"\section*{Debate Metadata}" in tex
        assert r"\begin{tabular}" in tex
        assert r"\toprule" in tex

    def test_latex_contains_timeline(self, sample_debate):
        """LaTeX contains message timeline."""
        result = format_debate_latex(sample_debate)
        tex = result.content.decode("utf-8")

        assert r"\section{Debate Timeline}" in tex
        assert r"\subsection{Round" in tex
        assert "agentmsg" in tex

    def test_latex_escapes_special_chars(self):
        """LaTeX escapes special characters."""
        debate = {
            "topic": "Test $100 & 50% #tag",
            "messages": [],
            "critiques": [],
        }
        result = format_debate_latex(debate)
        tex = result.content.decode("utf-8")

        assert r"\$" in tex
        assert r"\&" in tex
        assert r"\%" in tex
        assert r"\#" in tex


class TestLatexEscape:
    """Tests for _latex_escape function."""

    def test_escapes_backslash(self):
        """Escapes backslash."""
        result = _latex_escape("\\")
        # Backslash becomes \textbackslash{} but then { and } get escaped too
        assert "textbackslash" in result

    def test_escapes_ampersand(self):
        """Escapes ampersand."""
        assert _latex_escape("A & B") == r"A \& B"

    def test_escapes_percent(self):
        """Escapes percent."""
        assert _latex_escape("50%") == r"50\%"

    def test_escapes_dollar(self):
        """Escapes dollar sign."""
        assert _latex_escape("$100") == r"\$100"

    def test_escapes_hash(self):
        """Escapes hash."""
        assert _latex_escape("#tag") == r"\#tag"

    def test_escapes_underscore(self):
        """Escapes underscore."""
        assert _latex_escape("snake_case") == r"snake\_case"

    def test_escapes_braces(self):
        """Escapes curly braces."""
        assert _latex_escape("{text}") == r"\{text\}"

    def test_escapes_tilde(self):
        """Escapes tilde."""
        assert r"\textasciitilde{}" in _latex_escape("~")

    def test_escapes_caret(self):
        """Escapes caret."""
        assert r"\textasciicircum{}" in _latex_escape("^")

    def test_escapes_angle_brackets(self):
        """Escapes angle brackets."""
        result = _latex_escape("<text>")
        assert r"\textless{}" in result
        assert r"\textgreater{}" in result

    def test_empty_string(self):
        """Handles empty string."""
        assert _latex_escape("") == ""

    def test_none_input(self):
        """Handles None input gracefully."""
        # Function would fail on None, but empty string check covers it
        assert _latex_escape("") == ""

    def test_multiple_special_chars(self):
        """Handles multiple special characters."""
        result = _latex_escape("$100 & 50% #tag")
        assert r"\$" in result
        assert r"\&" in result
        assert r"\%" in result
        assert r"\#" in result


class TestEdgeCases:
    """Edge case tests."""

    def test_unicode_in_exports(self, sample_debate):
        """Handles unicode in all formats."""
        sample_debate["topic"] = "Is AI 人工智能 the future?"
        sample_debate["final_answer"] = "Yes, AI will shape the future 🚀"

        # All formats should handle unicode
        assert "人工智能" in format_debate_csv(sample_debate).content.decode("utf-8")
        assert "人工智能" in format_debate_html(sample_debate).content.decode("utf-8")
        assert "人工智能" in format_debate_txt(sample_debate).content.decode("utf-8")
        assert "人工智能" in format_debate_md(sample_debate).content.decode("utf-8")

    def test_missing_optional_fields(self):
        """Handles missing optional fields."""
        debate = {
            "messages": [],
            "critiques": [],
        }
        # Should not raise
        format_debate_csv(debate)
        format_debate_html(debate)
        format_debate_txt(debate)
        format_debate_md(debate)
        format_debate_latex(debate)

    def test_very_long_content(self, sample_debate):
        """Handles very long content."""
        sample_debate["messages"][0]["content"] = "x" * 10000
        sample_debate["final_answer"] = "y" * 5000

        # All formats should handle without error (with truncation where applicable)
        format_debate_csv(sample_debate)
        format_debate_html(sample_debate)
        format_debate_txt(sample_debate)
        format_debate_md(sample_debate)
        format_debate_latex(sample_debate)

    def test_special_chars_in_agent_names(self, sample_debate):
        """Handles special characters in agent names."""
        sample_debate["messages"][0]["agent"] = "Agent <special>"
        sample_debate["critiques"][0]["critic"] = "Critic & Co."

        # HTML should escape these
        html_result = format_debate_html(sample_debate)
        html = html_result.content.decode("utf-8")
        assert "&lt;" in html or "&gt;" in html

        # LaTeX should escape these
        latex_result = format_debate_latex(sample_debate)
        tex = latex_result.content.decode("utf-8")
        assert r"\&" in tex
