"""
Relevance ranking and scoring utilities for context gathering.

Provides utilities for:
- Confidence score normalization
- Topic relevance detection
- Content ranking and selection
- Quality filtering
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Confidence mapping for various confidence representations
CONFIDENCE_MAP = {
    "verified": 0.95,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.3,
    "unverified": 0.2,
}


def confidence_to_float(value: Any) -> float:
    """Convert various confidence representations to float.

    Handles:
    - Numeric values (int, float)
    - Enum values with .value attribute
    - String labels (verified, high, medium, low, unverified)

    Args:
        value: The confidence value to convert.

    Returns:
        Float confidence score between 0 and 1.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        return CONFIDENCE_MAP.get(value.lower(), 0.5)
    return 0.5


def confidence_label(confidence: float) -> str:
    """Get a human-readable confidence label.

    Args:
        confidence: Float confidence score between 0 and 1.

    Returns:
        Label string: HIGH, MEDIUM, or LOW.
    """
    if confidence > 0.7:
        return "HIGH"
    elif confidence > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def pattern_confidence_label(confidence: float) -> str:
    """Get a human-readable pattern confidence label.

    Args:
        confidence: Float confidence score between 0 and 1.

    Returns:
        Label string: Strong, Moderate, or Emerging.
    """
    if confidence > 0.7:
        return "Strong"
    elif confidence > 0.4:
        return "Moderate"
    else:
        return "Emerging"


class TopicRelevanceDetector:
    """
    Detects topic relevance for specialized context gathering.

    Determines whether a debate task is related to specific domains
    to enable targeted context enrichment.
    """

    # Aragora-related keywords
    ARAGORA_KEYWORDS = frozenset(
        [
            "aragora",
            "multi-agent debate",
            "decision stress-test",
            "ai red team",
            "adversarial validation",
            "gauntlet",
            "nomic loop",
            "debate framework",
        ]
    )

    # Security-related keywords (for threat intel enrichment)
    SECURITY_KEYWORDS = frozenset(
        [
            "security",
            "vulnerability",
            "cve",
            "exploit",
            "malware",
            "attack",
            "threat",
            "breach",
            "ransomware",
            "phishing",
            "injection",
            "xss",
            "csrf",
            "authentication",
            "authorization",
            "encryption",
            "cybersecurity",
            "penetration test",
            "zero day",
            "backdoor",
        ]
    )

    @classmethod
    def is_aragora_topic(cls, task: str) -> bool:
        """Check if task is related to Aragora.

        Args:
            task: The debate task description.

        Returns:
            True if task mentions Aragora-related concepts.
        """
        task_lower = task.lower()
        return any(kw in task_lower for kw in cls.ARAGORA_KEYWORDS)

    @classmethod
    def is_security_topic(cls, task: str) -> bool:
        """Check if task is related to security.

        Args:
            task: The debate task description.

        Returns:
            True if task mentions security-related concepts.
        """
        task_lower = task.lower()
        return any(kw in task_lower for kw in cls.SECURITY_KEYWORDS)

    @classmethod
    def get_topic_categories(cls, task: str) -> list[str]:
        """Get all applicable topic categories for a task.

        Args:
            task: The debate task description.

        Returns:
            List of category names (e.g., ["aragora", "security"]).
        """
        categories = []
        if cls.is_aragora_topic(task):
            categories.append("aragora")
        if cls.is_security_topic(task):
            categories.append("security")
        return categories


class ContentRanker:
    """
    Ranks and filters content based on relevance and quality.

    Provides utilities for selecting the most relevant content
    from multiple sources within token/character budgets.
    """

    @staticmethod
    def rank_by_confidence(
        items: list[tuple[str, float]],
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Rank items by confidence score.

        Args:
            items: List of (content, confidence) tuples.
            limit: Maximum number of items to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            Sorted list of items with highest confidence first.
        """
        filtered = [(c, conf) for c, conf in items if conf >= min_confidence]
        sorted_items = sorted(filtered, key=lambda x: x[1], reverse=True)
        return sorted_items[:limit]

    @staticmethod
    def filter_by_length(
        items: list[str],
        max_total_chars: int = 10000,
        max_item_chars: int = 2000,
    ) -> list[str]:
        """Filter items to fit within character budget.

        Args:
            items: List of content strings.
            max_total_chars: Maximum total characters to return.
            max_item_chars: Maximum characters per item.

        Returns:
            Filtered list that fits within the budget.
        """
        result = []
        total_chars = 0

        for item in items:
            # Truncate individual item if too long
            if len(item) > max_item_chars:
                item = item[: max_item_chars - 30] + "... [truncated]"

            # Check if we have room
            if total_chars + len(item) > max_total_chars:
                break

            result.append(item)
            total_chars += len(item)

        return result

    @staticmethod
    def deduplicate_by_similarity(
        items: list[str],
        similarity_threshold: float = 0.8,
    ) -> list[str]:
        """Remove near-duplicate items based on simple overlap.

        Uses a simple word overlap metric for fast deduplication.
        For more sophisticated semantic deduplication, use embeddings.

        Args:
            items: List of content strings.
            similarity_threshold: Threshold for considering items duplicates.

        Returns:
            Deduplicated list of items.
        """
        if not items:
            return []

        def word_overlap(a: str, b: str) -> float:
            """Calculate word overlap ratio."""
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            intersection = words_a & words_b
            union = words_a | words_b
            return len(intersection) / len(union) if union else 0.0

        result = [items[0]]
        for item in items[1:]:
            # Check if similar to any existing item
            is_duplicate = any(
                word_overlap(item, existing) > similarity_threshold for existing in result
            )
            if not is_duplicate:
                result.append(item)

        return result


class KnowledgeItemCategorizer:
    """
    Categorizes knowledge items by source type.

    Groups items from Knowledge Mound queries into logical categories
    for structured presentation in debate context.
    """

    FACT_SOURCES = frozenset(["fact", "fact_store"])
    EVIDENCE_SOURCES = frozenset(["evidence", "evidence_store"])

    @classmethod
    def categorize_items(
        cls,
        items: list[Any],
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float, str]]]:
        """Categorize knowledge items by source type.

        Args:
            items: List of knowledge items with source, content, confidence attributes.

        Returns:
            Tuple of:
            - facts: List of (content, confidence) tuples
            - evidence: List of (content, confidence) tuples
            - insights: List of (content, confidence, source) tuples
        """
        facts = []
        evidence = []
        insights = []

        for item in items:
            source = getattr(item, "source", None)
            source_name = (
                source.value if hasattr(source, "value") else str(source) if source else "unknown"
            )
            content = item.content[:500] if item.content else ""
            confidence = confidence_to_float(getattr(item, "confidence", 0.5))

            if source_name in cls.FACT_SOURCES:
                facts.append((content, confidence))
            elif source_name in cls.EVIDENCE_SOURCES:
                evidence.append((content, confidence))
            else:
                insights.append((content, confidence, source_name))

        return facts, evidence, insights

    @classmethod
    def format_categorized_context(
        cls,
        facts: list[tuple[str, float]],
        evidence: list[tuple[str, float]],
        insights: list[tuple[str, float, str]],
        max_facts: int = 3,
        max_evidence: int = 3,
        max_insights: int = 4,
    ) -> list[str]:
        """Format categorized items into context parts.

        Args:
            facts: List of (content, confidence) tuples.
            evidence: List of (content, confidence) tuples.
            insights: List of (content, confidence, source) tuples.
            max_facts: Maximum facts to include.
            max_evidence: Maximum evidence items to include.
            max_insights: Maximum insights to include.

        Returns:
            List of formatted context strings.
        """
        context_parts = []

        # Format facts
        if facts:
            context_parts.append("### Verified Facts")
            for content, conf in facts[:max_facts]:
                conf_label = confidence_label(conf)
                context_parts.append(f"- [{conf_label}] {content}")
            context_parts.append("")

        # Format evidence
        if evidence:
            context_parts.append("### Supporting Evidence")
            for content, conf in evidence[:max_evidence]:
                context_parts.append(f"- {content}")
            context_parts.append("")

        # Format insights
        if insights:
            context_parts.append("### Related Insights")
            for content, conf, source in insights[:max_insights]:
                context_parts.append(f"- ({source}) {content}")
            context_parts.append("")

        return context_parts


# Re-export all public symbols
__all__ = [
    "CONFIDENCE_MAP",
    "confidence_to_float",
    "confidence_label",
    "pattern_confidence_label",
    "TopicRelevanceDetector",
    "ContentRanker",
    "KnowledgeItemCategorizer",
]
