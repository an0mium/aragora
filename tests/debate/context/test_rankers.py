"""Tests for aragora.debate.context.rankers — Ranking and scoring utilities."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any

from aragora.debate.context.rankers import (
    CONFIDENCE_MAP,
    ContentRanker,
    KnowledgeItemCategorizer,
    TopicRelevanceDetector,
    confidence_label,
    confidence_to_float,
    pattern_confidence_label,
)


# ---------------------------------------------------------------------------
# confidence_to_float
# ---------------------------------------------------------------------------


class TestConfidenceToFloat:
    def test_int(self):
        assert confidence_to_float(1) == 1.0

    def test_float(self):
        assert confidence_to_float(0.75) == pytest.approx(0.75)

    def test_string_verified(self):
        assert confidence_to_float("verified") == pytest.approx(0.95)

    def test_string_high(self):
        assert confidence_to_float("high") == pytest.approx(0.8)

    def test_string_medium(self):
        assert confidence_to_float("medium") == pytest.approx(0.6)

    def test_string_low(self):
        assert confidence_to_float("low") == pytest.approx(0.3)

    def test_string_unverified(self):
        assert confidence_to_float("unverified") == pytest.approx(0.2)

    def test_string_unknown(self):
        assert confidence_to_float("unknown_level") == pytest.approx(0.5)

    def test_string_case_insensitive(self):
        assert confidence_to_float("HIGH") == pytest.approx(0.8)
        assert confidence_to_float("Verified") == pytest.approx(0.95)

    def test_enum_like_with_value(self):
        class FakeEnum:
            value = "high"

        assert confidence_to_float(FakeEnum()) == pytest.approx(0.8)

    def test_enum_like_numeric_value_falls_through(self):
        # After extracting .value=0.9, the function only checks isinstance(str)
        # so numeric enum values fall through to default 0.5
        class FakeEnum:
            value = 0.9

        assert confidence_to_float(FakeEnum()) == pytest.approx(0.5)

    def test_none_fallback(self):
        assert confidence_to_float(None) == pytest.approx(0.5)

    def test_list_fallback(self):
        assert confidence_to_float([1, 2]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# confidence_label / pattern_confidence_label
# ---------------------------------------------------------------------------


class TestConfidenceLabel:
    def test_high(self):
        assert confidence_label(0.8) == "HIGH"

    def test_medium(self):
        assert confidence_label(0.5) == "MEDIUM"

    def test_low(self):
        assert confidence_label(0.2) == "LOW"

    def test_boundary_high(self):
        assert confidence_label(0.71) == "HIGH"
        assert confidence_label(0.7) == "MEDIUM"

    def test_boundary_medium(self):
        assert confidence_label(0.41) == "MEDIUM"
        assert confidence_label(0.4) == "LOW"


class TestPatternConfidenceLabel:
    def test_strong(self):
        assert pattern_confidence_label(0.8) == "Strong"

    def test_moderate(self):
        assert pattern_confidence_label(0.5) == "Moderate"

    def test_emerging(self):
        assert pattern_confidence_label(0.2) == "Emerging"


# ---------------------------------------------------------------------------
# TopicRelevanceDetector
# ---------------------------------------------------------------------------


class TestTopicRelevanceDetector:
    def test_is_aragora_topic_true(self):
        assert TopicRelevanceDetector.is_aragora_topic("How should Aragora handle rate limiting?")

    def test_is_aragora_topic_keyword_multi_agent(self):
        assert TopicRelevanceDetector.is_aragora_topic("Design a multi-agent debate system")

    def test_is_aragora_topic_false(self):
        assert not TopicRelevanceDetector.is_aragora_topic("What is the weather today?")

    def test_is_aragora_topic_case_insensitive(self):
        assert TopicRelevanceDetector.is_aragora_topic("NOMIC LOOP improvements needed")

    def test_is_security_topic_true(self):
        assert TopicRelevanceDetector.is_security_topic("How to prevent SQL injection attacks?")

    def test_is_security_topic_cve(self):
        assert TopicRelevanceDetector.is_security_topic("Analyze CVE-2024-1234 impact")

    def test_is_security_topic_false(self):
        assert not TopicRelevanceDetector.is_security_topic("Design a REST API for todo lists")

    def test_get_topic_categories_both(self):
        cats = TopicRelevanceDetector.get_topic_categories(
            "Aragora security vulnerability in authentication"
        )
        assert "aragora" in cats
        assert "security" in cats

    def test_get_topic_categories_none(self):
        cats = TopicRelevanceDetector.get_topic_categories("Discuss best pizza toppings")
        assert cats == []

    def test_get_topic_categories_security_only(self):
        cats = TopicRelevanceDetector.get_topic_categories("XSS attack prevention strategies")
        assert "security" in cats
        assert "aragora" not in cats


# ---------------------------------------------------------------------------
# ContentRanker
# ---------------------------------------------------------------------------


class TestContentRankerRankByConfidence:
    def test_ranks_descending(self):
        items = [("a", 0.3), ("b", 0.9), ("c", 0.6)]
        result = ContentRanker.rank_by_confidence(items)
        assert result[0] == ("b", 0.9)
        assert result[1] == ("c", 0.6)
        assert result[2] == ("a", 0.3)

    def test_limit(self):
        items = [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4)]
        result = ContentRanker.rank_by_confidence(items, limit=2)
        assert len(result) == 2

    def test_min_confidence(self):
        items = [("a", 0.1), ("b", 0.5), ("c", 0.9)]
        result = ContentRanker.rank_by_confidence(items, min_confidence=0.3)
        assert len(result) == 2
        assert all(conf >= 0.3 for _, conf in result)

    def test_empty(self):
        assert ContentRanker.rank_by_confidence([]) == []


class TestContentRankerFilterByLength:
    def test_within_budget(self):
        items = ["short", "also short"]
        result = ContentRanker.filter_by_length(items, max_total_chars=100)
        assert len(result) == 2

    def test_exceeds_budget(self):
        items = ["a" * 50, "b" * 50, "c" * 50]
        result = ContentRanker.filter_by_length(items, max_total_chars=100)
        assert len(result) == 2

    def test_truncates_long_items(self):
        items = ["a" * 3000]
        result = ContentRanker.filter_by_length(items, max_item_chars=100)
        assert len(result[0]) <= 100
        assert "truncated" in result[0]

    def test_empty(self):
        assert ContentRanker.filter_by_length([]) == []


class TestContentRankerDeduplicateBySimilarity:
    def test_removes_duplicates(self):
        items = [
            "the quick brown fox jumps",
            "the quick brown fox jumps",  # Exact duplicate
            "completely different text here",
        ]
        result = ContentRanker.deduplicate_by_similarity(items)
        assert len(result) == 2

    def test_keeps_different_items(self):
        items = ["hello world", "goodbye moon", "foo bar baz"]
        result = ContentRanker.deduplicate_by_similarity(items)
        assert len(result) == 3

    def test_empty(self):
        assert ContentRanker.deduplicate_by_similarity([]) == []

    def test_single_item(self):
        result = ContentRanker.deduplicate_by_similarity(["only one"])
        assert result == ["only one"]

    def test_threshold_control(self):
        # Low threshold means more items considered duplicates
        items = ["hello world test", "hello world check", "completely different"]
        strict = ContentRanker.deduplicate_by_similarity(items, similarity_threshold=0.3)
        lenient = ContentRanker.deduplicate_by_similarity(items, similarity_threshold=0.9)
        assert len(strict) <= len(lenient)


# ---------------------------------------------------------------------------
# KnowledgeItemCategorizer
# ---------------------------------------------------------------------------


@dataclass
class FakeKnowledgeItem:
    content: str
    source: Any = None
    confidence: float = 0.5


class FakeSource:
    def __init__(self, value: str):
        self.value = value


class TestKnowledgeItemCategorizer:
    def test_categorize_facts(self):
        items = [FakeKnowledgeItem(content="a fact", source=FakeSource("fact"))]
        facts, evidence, insights = KnowledgeItemCategorizer.categorize_items(items)
        assert len(facts) == 1
        assert facts[0][0] == "a fact"

    def test_categorize_evidence(self):
        items = [FakeKnowledgeItem(content="evidence text", source=FakeSource("evidence_store"))]
        facts, evidence, insights = KnowledgeItemCategorizer.categorize_items(items)
        assert len(evidence) == 1

    def test_categorize_insights(self):
        items = [FakeKnowledgeItem(content="insight text", source=FakeSource("pattern"))]
        facts, evidence, insights = KnowledgeItemCategorizer.categorize_items(items)
        assert len(insights) == 1
        assert insights[0][2] == "pattern"

    def test_categorize_mixed(self):
        items = [
            FakeKnowledgeItem(content="fact", source=FakeSource("fact")),
            FakeKnowledgeItem(content="evidence", source=FakeSource("evidence")),
            FakeKnowledgeItem(content="insight", source=FakeSource("analysis")),
        ]
        facts, evidence, insights = KnowledgeItemCategorizer.categorize_items(items)
        assert len(facts) == 1
        assert len(evidence) == 1
        assert len(insights) == 1

    def test_categorize_no_source(self):
        items = [FakeKnowledgeItem(content="orphan", source=None)]
        facts, evidence, insights = KnowledgeItemCategorizer.categorize_items(items)
        assert len(insights) == 1  # Defaults to insights

    def test_truncates_long_content(self):
        items = [FakeKnowledgeItem(content="x" * 1000, source=FakeSource("fact"))]
        facts, _, _ = KnowledgeItemCategorizer.categorize_items(items)
        assert len(facts[0][0]) == 500

    def test_format_categorized_context_facts(self):
        facts = [("AES-256 is secure", 0.9)]
        parts = KnowledgeItemCategorizer.format_categorized_context(facts, [], [])
        assert any("Verified Facts" in p for p in parts)
        assert any("[HIGH]" in p for p in parts)

    def test_format_categorized_context_evidence(self):
        evidence = [("Study shows encryption works", 0.7)]
        parts = KnowledgeItemCategorizer.format_categorized_context([], evidence, [])
        assert any("Supporting Evidence" in p for p in parts)

    def test_format_categorized_context_insights(self):
        insights = [("Pattern observed", 0.5, "analysis")]
        parts = KnowledgeItemCategorizer.format_categorized_context([], [], insights)
        assert any("Related Insights" in p for p in parts)
        assert any("(analysis)" in p for p in parts)

    def test_format_respects_limits(self):
        facts = [("f1", 0.9), ("f2", 0.8), ("f3", 0.7), ("f4", 0.6)]
        parts = KnowledgeItemCategorizer.format_categorized_context(facts, [], [], max_facts=2)
        fact_lines = [p for p in parts if p.startswith("- [")]
        assert len(fact_lines) == 2

    def test_format_empty(self):
        parts = KnowledgeItemCategorizer.format_categorized_context([], [], [])
        assert parts == []
