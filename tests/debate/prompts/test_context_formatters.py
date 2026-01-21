"""
Tests for the context formatters module.

Tests the pure formatting functions for debate prompts.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from aragora.debate.prompts.context_formatters import (
    format_patterns_for_prompt,
    format_successful_patterns,
    format_evidence_for_prompt,
    format_trending_for_prompt,
    format_elo_ranking_context,
    format_calibration_context,
    format_belief_context,
)


# ============================================================================
# Mock Data Classes
# ============================================================================


@dataclass
class MockEvidenceSnippet:
    """Mock evidence snippet for testing."""

    title: str = "Test Article"
    source: str = "Test Source"
    snippet: str = "This is test content"
    url: Optional[str] = "https://example.com"
    reliability_score: float = 0.8


@dataclass
class MockTrendingTopic:
    """Mock trending topic for testing."""

    topic: str = "AI Safety"
    platform: str = "hackernews"
    volume: int = 1500
    category: str = "technology"


# ============================================================================
# Pattern Formatting Tests
# ============================================================================


class TestFormatPatternsForPrompt:
    """Tests for format_patterns_for_prompt function."""

    def test_empty_patterns(self):
        """Test empty patterns returns empty string."""
        assert format_patterns_for_prompt([]) == ""

    def test_single_pattern(self):
        """Test formatting a single pattern."""
        patterns = [
            {
                "category": "reasoning",
                "pattern": "Circular arguments detected",
                "occurrences": 5,
                "avg_severity": 0.6,
            }
        ]
        result = format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "REASONING" in result
        assert "Circular arguments detected" in result
        assert "5 times" in result
        assert "[MEDIUM]" in result

    def test_high_severity_label(self):
        """Test high severity gets proper label."""
        patterns = [
            {
                "category": "logic",
                "pattern": "Major flaw",
                "occurrences": 10,
                "avg_severity": 0.8,
            }
        ]
        result = format_patterns_for_prompt(patterns)
        assert "[HIGH SEVERITY]" in result

    def test_limits_to_five_patterns(self):
        """Test that output is limited to 5 patterns."""
        patterns = [
            {"category": f"category{i}", "pattern": f"Pattern {i}", "occurrences": 1}
            for i in range(10)
        ]
        result = format_patterns_for_prompt(patterns)

        # Should have exactly 5 patterns (categories are uppercased)
        assert result.count("CATEGORY") == 5
        assert "Pattern 0" in result
        assert "Pattern 4" in result
        assert "Pattern 5" not in result


class TestFormatSuccessfulPatterns:
    """Tests for format_successful_patterns function."""

    def test_empty_patterns(self):
        """Test empty patterns returns empty string."""
        assert format_successful_patterns([]) == ""

    def test_single_pattern(self):
        """Test formatting a single successful pattern."""
        patterns = [
            {
                "strategy": "Evidence-based arguments",
                "success_rate": 0.85,
                "context": "Technical debates",
            }
        ]
        result = format_successful_patterns(patterns)

        assert "SUCCESSFUL STRATEGIES" in result
        assert "Evidence-based arguments" in result
        assert "85%" in result
        assert "Technical debates" in result

    def test_respects_limit(self):
        """Test that limit parameter is respected."""
        patterns = [{"strategy": f"Strategy {i}", "success_rate": 0.5} for i in range(10)]
        result = format_successful_patterns(patterns, limit=2)

        assert result.count("Strategy") == 2


# ============================================================================
# Evidence Formatting Tests
# ============================================================================


class TestFormatEvidenceForPrompt:
    """Tests for format_evidence_for_prompt function."""

    def test_empty_snippets(self):
        """Test empty snippets returns empty string."""
        assert format_evidence_for_prompt([]) == ""

    def test_single_snippet(self):
        """Test formatting a single evidence snippet."""
        snippets = [MockEvidenceSnippet()]
        result = format_evidence_for_prompt(snippets)

        assert "AVAILABLE EVIDENCE" in result
        assert "[EVID-1]" in result
        assert "Test Article" in result
        assert "Test Source" in result
        assert "80%" in result  # reliability
        assert "example.com" in result

    def test_multiple_snippets(self):
        """Test formatting multiple snippets."""
        snippets = [MockEvidenceSnippet(title=f"Article {i}") for i in range(3)]
        result = format_evidence_for_prompt(snippets)

        assert "[EVID-1]" in result
        assert "[EVID-2]" in result
        assert "[EVID-3]" in result

    def test_max_snippets_limit(self):
        """Test max_snippets parameter."""
        snippets = [MockEvidenceSnippet(title=f"Article {i}") for i in range(10)]
        result = format_evidence_for_prompt(snippets, max_snippets=3)

        assert "[EVID-1]" in result
        assert "[EVID-3]" in result
        assert "[EVID-4]" not in result

    def test_truncates_long_snippet(self):
        """Test that long snippets are truncated."""
        snippets = [MockEvidenceSnippet(snippet="x" * 500)]
        result = format_evidence_for_prompt(snippets)

        assert "..." in result

    def test_handles_missing_reliability(self):
        """Test handling of missing reliability score."""
        snippet = MockEvidenceSnippet()
        snippet.reliability_score = None  # type: ignore
        result = format_evidence_for_prompt([snippet])

        assert "50%" in result  # default fallback


# ============================================================================
# Trending Topics Formatting Tests
# ============================================================================


class TestFormatTrendingForPrompt:
    """Tests for format_trending_for_prompt function."""

    def test_empty_topics(self):
        """Test empty topics returns empty string."""
        assert format_trending_for_prompt([]) == ""

    def test_single_topic(self):
        """Test formatting a single topic."""
        topics = [MockTrendingTopic()]
        result = format_trending_for_prompt(topics)

        assert "TRENDING CONTEXT" in result
        assert "AI Safety" in result
        assert "hackernews" in result
        assert "1,500" in result

    def test_relevance_filtering(self):
        """Test relevance filtering with task."""
        topics = [
            MockTrendingTopic(topic="AI Safety"),
            MockTrendingTopic(topic="Weather Patterns"),
            MockTrendingTopic(topic="Machine Learning"),
        ]
        result = format_trending_for_prompt(
            topics,
            task="Discussion about AI and machine learning",
            max_topics=2,
            use_relevance_filter=True,
        )

        # AI-related topics should be prioritized
        assert "AI Safety" in result or "Machine Learning" in result

    def test_no_filtering(self):
        """Test without relevance filtering."""
        topics = [MockTrendingTopic(topic=f"Topic {i}") for i in range(5)]
        result = format_trending_for_prompt(
            topics,
            max_topics=3,
            use_relevance_filter=False,
        )

        assert "Topic 0" in result
        assert "Topic 1" in result
        assert "Topic 2" in result
        assert "Topic 3" not in result


# ============================================================================
# ELO Ranking Tests
# ============================================================================


class TestFormatEloRankingContext:
    """Tests for format_elo_ranking_context function."""

    def test_empty_ratings(self):
        """Test empty ratings returns empty string."""
        assert format_elo_ranking_context("agent1", [], {}) == ""

    def test_single_agent(self):
        """Test formatting for single agent."""
        result = format_elo_ranking_context(
            "claude",
            ["claude"],
            {"claude": 1500},
        )

        assert "Agent Rankings" in result
        assert "claude (you)" in result
        assert "1500" in result

    def test_multiple_agents_sorted(self):
        """Test agents are sorted by rating."""
        result = format_elo_ranking_context(
            "claude",
            ["claude", "gpt-4", "gemini"],
            {"claude": 1500, "gpt-4": 1600, "gemini": 1400},
        )

        # Should be in order: gpt-4, claude, gemini
        gpt4_pos = result.find("gpt-4")
        claude_pos = result.find("claude")
        gemini_pos = result.find("gemini")

        assert gpt4_pos < claude_pos < gemini_pos

    def test_domain_suffix(self):
        """Test domain suffix in header."""
        result = format_elo_ranking_context(
            "claude",
            ["claude"],
            {"claude": 1500},
            domain="technical",
        )

        assert "(technical)" in result

    def test_general_domain_no_suffix(self):
        """Test no suffix for general domain."""
        result = format_elo_ranking_context(
            "claude",
            ["claude"],
            {"claude": 1500},
            domain="general",
        )

        assert "(general)" not in result


# ============================================================================
# Calibration Context Tests
# ============================================================================


class TestFormatCalibrationContext:
    """Tests for format_calibration_context function."""

    def test_insufficient_predictions(self):
        """Test that insufficient predictions returns empty."""
        result = format_calibration_context(
            brier_score=0.3,
            is_overconfident=True,
            is_underconfident=False,
            total_predictions=3,  # Less than 5
        )
        assert result == ""

    def test_good_calibration_no_feedback(self):
        """Test that well-calibrated agents get no feedback."""
        result = format_calibration_context(
            brier_score=0.2,  # Below 0.25 threshold
            is_overconfident=False,
            is_underconfident=False,
            total_predictions=10,
        )
        assert result == ""

    def test_overconfident_feedback(self):
        """Test overconfident agent gets appropriate feedback."""
        result = format_calibration_context(
            brier_score=0.35,
            is_overconfident=True,
            is_underconfident=False,
            total_predictions=10,
        )

        assert "Calibration Feedback" in result
        assert "OVERCONFIDENT" in result
        assert "0.35" in result

    def test_underconfident_feedback(self):
        """Test underconfident agent gets appropriate feedback."""
        result = format_calibration_context(
            brier_score=0.30,
            is_overconfident=False,
            is_underconfident=True,
            total_predictions=10,
        )

        assert "UNDERCONFIDENT" in result
        assert "more confidence" in result


# ============================================================================
# Belief Context Tests
# ============================================================================


class TestFormatBeliefContext:
    """Tests for format_belief_context function."""

    def test_empty_beliefs(self):
        """Test empty beliefs returns empty string."""
        assert format_belief_context([]) == ""

    def test_single_belief(self):
        """Test formatting a single belief."""
        beliefs = [
            {
                "statement": "AI safety is important",
                "confidence": 0.85,
                "support_count": 4,
            }
        ]
        result = format_belief_context(beliefs)

        assert "ESTABLISHED BELIEFS" in result
        assert "AI safety is important" in result
        assert "85%" in result
        assert "4" in result

    def test_respects_limit(self):
        """Test that limit parameter is respected."""
        beliefs = [
            {"statement": f"Belief {i}", "confidence": 0.7, "support_count": 2} for i in range(10)
        ]
        result = format_belief_context(beliefs, limit=2)

        assert result.count("Belief") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
