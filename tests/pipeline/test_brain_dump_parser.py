"""Tests for the BrainDumpParser — natural language to ideas extraction."""

from __future__ import annotations

import pytest

from aragora.pipeline.brain_dump_parser import BrainDumpParser


@pytest.fixture
def parser() -> BrainDumpParser:
    return BrainDumpParser()


# ──────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────


class TestDetectFormat:
    def test_detects_bullets_dash(self, parser: BrainDumpParser) -> None:
        text = "- Build a dashboard\n- Fix error handling\n- Improve API speed"
        assert parser._detect_format(text) == "bullets"

    def test_detects_bullets_asterisk(self, parser: BrainDumpParser) -> None:
        text = "* Build a dashboard\n* Fix error handling"
        assert parser._detect_format(text) == "bullets"

    def test_detects_bullets_angle_bracket(self, parser: BrainDumpParser) -> None:
        text = "> Build a dashboard\n> Fix error handling"
        assert parser._detect_format(text) == "bullets"

    def test_detects_numbered_list(self, parser: BrainDumpParser) -> None:
        text = "1. Build a dashboard\n2. Fix error handling\n3. Improve speed"
        assert parser._detect_format(text) == "numbered"

    def test_detects_numbered_with_paren(self, parser: BrainDumpParser) -> None:
        text = "1) Build a dashboard\n2) Fix error handling"
        assert parser._detect_format(text) == "numbered"

    def test_detects_paragraphs(self, parser: BrainDumpParser) -> None:
        text = "First thought here.\n\nSecond thought here."
        assert parser._detect_format(text) == "paragraphs"

    def test_detects_prose(self, parser: BrainDumpParser) -> None:
        text = "I think we need to build a dashboard and fix error handling."
        assert parser._detect_format(text) == "prose"


# ──────────────────────────────────────────────────────────────────────
# Bullet list parsing
# ──────────────────────────────────────────────────────────────────────


class TestBulletParsing:
    def test_dash_bullets(self, parser: BrainDumpParser) -> None:
        text = "- Build a dashboard\n- Fix error handling\n- Improve API speed"
        ideas = parser.parse(text)
        assert len(ideas) == 3
        assert ideas[0] == "Build a dashboard"
        assert ideas[1] == "Fix error handling"

    def test_asterisk_bullets(self, parser: BrainDumpParser) -> None:
        text = "* Build a dashboard\n* Fix error handling"
        ideas = parser.parse(text)
        assert len(ideas) == 2

    def test_mixed_bullets_with_empty_lines(self, parser: BrainDumpParser) -> None:
        text = "- Build a dashboard\n\n- Fix error handling\n- Improve API speed"
        ideas = parser.parse(text)
        assert len(ideas) == 3


# ──────────────────────────────────────────────────────────────────────
# Numbered list parsing
# ──────────────────────────────────────────────────────────────────────


class TestNumberedParsing:
    def test_dot_numbered(self, parser: BrainDumpParser) -> None:
        text = "1. Build a dashboard\n2. Fix error handling\n3. Improve speed"
        ideas = parser.parse(text)
        assert len(ideas) == 3
        assert ideas[0] == "Build a dashboard"

    def test_paren_numbered(self, parser: BrainDumpParser) -> None:
        text = "1) Build a dashboard\n2) Fix error handling"
        ideas = parser.parse(text)
        assert len(ideas) == 2
        assert ideas[0] == "Build a dashboard"


# ──────────────────────────────────────────────────────────────────────
# Paragraph splitting
# ──────────────────────────────────────────────────────────────────────


class TestParagraphSplitting:
    def test_double_newline_paragraphs(self, parser: BrainDumpParser) -> None:
        text = (
            "We should build a dashboard for real-time analytics.\n\n"
            "Error handling needs improvement across the codebase.\n\n"
            "The API latency is too high for production use."
        )
        ideas = parser.parse(text)
        assert len(ideas) == 3

    def test_multiple_blank_lines(self, parser: BrainDumpParser) -> None:
        text = "First idea.\n\n\nSecond idea."
        ideas = parser.parse(text)
        assert len(ideas) == 2


# ──────────────────────────────────────────────────────────────────────
# Prose sentence splitting
# ──────────────────────────────────────────────────────────────────────


class TestProseSplitting:
    def test_period_separated(self, parser: BrainDumpParser) -> None:
        text = (
            "I think we should build a dashboard. "
            "Also need better error handling. "
            "The API is too slow."
        )
        ideas = parser.parse(text)
        assert len(ideas) == 3

    def test_exclamation_and_question(self, parser: BrainDumpParser) -> None:
        text = "Why is the API so slow? We need to fix this! Add caching to the endpoints."
        ideas = parser.parse(text)
        assert len(ideas) == 3

    def test_single_sentence(self, parser: BrainDumpParser) -> None:
        text = "Build a dashboard for the analytics team"
        ideas = parser.parse(text)
        assert len(ideas) == 1
        assert ideas[0] == "Build a dashboard for the analytics team"


# ──────────────────────────────────────────────────────────────────────
# Fragment merging
# ──────────────────────────────────────────────────────────────────────


class TestFragmentMerging:
    def test_short_fragment_merged_with_previous(self, parser: BrainDumpParser) -> None:
        ideas = parser._merge_fragments(
            ["Build a real-time dashboard", "yes", "Fix error handling across codebase"]
        )
        assert len(ideas) == 2
        assert "yes" in ideas[0]

    def test_short_first_fragment_merged_with_next(self, parser: BrainDumpParser) -> None:
        ideas = parser._merge_fragments(["ok", "Build a real-time dashboard for analytics"])
        assert len(ideas) == 1
        assert "ok" in ideas[0]

    def test_no_merge_needed(self, parser: BrainDumpParser) -> None:
        ideas = parser._merge_fragments(
            ["Build a real-time dashboard", "Fix error handling across codebase"]
        )
        assert len(ideas) == 2

    def test_empty_list(self, parser: BrainDumpParser) -> None:
        assert parser._merge_fragments([]) == []


# ──────────────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_exact_duplicate_removed(self, parser: BrainDumpParser) -> None:
        ideas = parser._deduplicate(["Build a dashboard", "Build a dashboard"])
        assert len(ideas) == 1

    def test_near_duplicate_removed(self, parser: BrainDumpParser) -> None:
        ideas = parser._deduplicate(
            ["Build a real-time dashboard", "Build a real-time dashboard for analytics"],
            threshold=0.6,
        )
        assert len(ideas) == 1

    def test_different_ideas_kept(self, parser: BrainDumpParser) -> None:
        ideas = parser._deduplicate(
            ["Build a dashboard", "Fix error handling", "Improve API speed"]
        )
        assert len(ideas) == 3

    def test_empty_list(self, parser: BrainDumpParser) -> None:
        assert parser._deduplicate([]) == []


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self, parser: BrainDumpParser) -> None:
        assert parser.parse("") == []

    def test_whitespace_only(self, parser: BrainDumpParser) -> None:
        assert parser.parse("   \n\n  ") == []

    def test_single_word(self, parser: BrainDumpParser) -> None:
        result = parser.parse("dashboard")
        assert len(result) == 1
        assert result[0] == "dashboard"

    def test_preserves_content(self, parser: BrainDumpParser) -> None:
        text = "- Implement Redis caching to reduce API response latency by 40%"
        ideas = parser.parse(text)
        assert len(ideas) == 1
        assert "Redis caching" in ideas[0]
        assert "40%" in ideas[0]


# ──────────────────────────────────────────────────────────────────────
# Integration: brain dump text → pipeline-compatible idea count
# ──────────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_prose_to_ideas_count(self, parser: BrainDumpParser) -> None:
        text = (
            "I think we should build a dashboard. "
            "Also need better error handling. "
            "The API is too slow."
        )
        ideas = parser.parse(text)
        assert len(ideas) == 3

    def test_bullet_to_ideas_count(self, parser: BrainDumpParser) -> None:
        text = (
            "- Build a real-time analytics dashboard\n"
            "- Fix error handling across the codebase\n"
            "- Reduce API response latency below 100ms\n"
            "- Add rate limiting to all public endpoints"
        )
        ideas = parser.parse(text)
        assert len(ideas) == 4

    def test_mixed_content_produces_ideas(self, parser: BrainDumpParser) -> None:
        text = (
            "We need to improve our infrastructure. "
            "The database is slow. "
            "Users are complaining about timeouts."
        )
        ideas = parser.parse(text)
        assert len(ideas) >= 2
        assert all(isinstance(i, str) for i in ideas)
        assert all(len(i) > 0 for i in ideas)
