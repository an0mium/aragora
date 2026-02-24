"""
Pulse Debate Enrichment -- closes the trending-topics-to-debate loop.

Provides PulseDebateEnricher, a standalone bridge that queries PulseStore
(or PulseManager) for trending topics relevant to a debate task, scores
them by quality and freshness, and returns ranked context snippets ready
for injection into debate prompts.

Usage:
    enricher = PulseDebateEnricher(pulse_store=store)
    snippets = enricher.enrich("Design a rate limiter", max_topics=5)
    for s in snippets:
        print(f"{s.title} ({s.source}, Q:{s.quality_score:.1f})")

Wire into PromptBuilder via ``_inject_pulse_enrichment``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "they",
        "them",
        "their",
        "our",
        "your",
        "my",
        "his",
        "her",
    }
)


@dataclass
class EnrichedSnippet:
    """A single trending-topic context snippet ready for prompt injection."""

    title: str
    source: str  # platform name
    quality_score: float  # 0.0 - 1.0
    freshness_score: float  # 0.0 - 1.0
    relevance_rationale: str  # human-readable explanation of relevance
    volume: int = 0
    category: str = ""
    hours_ago: float = 0.0

    @property
    def combined_score(self) -> float:
        """Weighted composite of quality, freshness, and implied relevance."""
        # Relevance is already filtered; rank by quality * freshness
        return self.quality_score * 0.5 + self.freshness_score * 0.5


@dataclass
class EnrichmentResult:
    """Container for the enrichment output."""

    snippets: list[EnrichedSnippet] = field(default_factory=list)
    task: str = ""
    elapsed_ms: float = 0.0

    @property
    def has_context(self) -> bool:
        return len(self.snippets) > 0


# ---------------------------------------------------------------------------
# Core enricher
# ---------------------------------------------------------------------------


class PulseDebateEnricher:
    """Query PulseStore for topics relevant to a debate and rank them.

    Parameters
    ----------
    pulse_store:
        A ``ScheduledDebateStore`` or any object exposing
        ``get_recent_topics(hours=N) -> list[record]`` where each record has
        ``topic_text``, ``platform``, ``volume``, ``category``, ``hours_ago``.
    pulse_manager:
        Alternative: a ``PulseManager`` instance whose
        ``get_trending_topics`` async method can be used.
    quality_threshold:
        Minimum quality score (0-1) for a topic to be included.
    freshness_max_hours:
        Maximum age in hours; topics older than this are excluded.
    min_keyword_overlap:
        Minimum number of overlapping keywords between the task and a topic
        for the topic to be considered relevant.  Set to 0 to skip filtering.
    """

    def __init__(
        self,
        pulse_store: Any | None = None,
        pulse_manager: Any | None = None,
        quality_threshold: float = 0.3,
        freshness_max_hours: float = 48.0,
        min_keyword_overlap: int = 1,
    ) -> None:
        self.pulse_store = pulse_store
        self.pulse_manager = pulse_manager
        self.quality_threshold = quality_threshold
        self.freshness_max_hours = freshness_max_hours
        self.min_keyword_overlap = min_keyword_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        task: str,
        max_topics: int = 5,
    ) -> EnrichmentResult:
        """Synchronously enrich a debate task with relevant pulse context.

        Args:
            task: The debate task/question text.
            max_topics: Maximum number of enriched snippets to return (3-5).

        Returns:
            EnrichmentResult with ranked snippets.
        """
        start = time.monotonic()
        max_topics = max(1, min(max_topics, 10))  # clamp

        if not task:
            return EnrichmentResult(task=task)

        # 1. Fetch raw records from the store
        records = self._fetch_records()
        if not records:
            elapsed = (time.monotonic() - start) * 1000
            return EnrichmentResult(task=task, elapsed_ms=elapsed)

        # 2. Extract task keywords for relevance matching
        task_keywords = self._extract_keywords(task)

        # 3. Score each record
        scored: list[tuple[Any, EnrichedSnippet]] = []
        for record in records:
            snippet = self._score_record(record, task_keywords)
            if snippet is not None:
                scored.append((snippet.combined_score, snippet))

        # 4. Sort by combined score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        snippets = [s for _, s in scored[:max_topics]]

        elapsed = (time.monotonic() - start) * 1000
        return EnrichmentResult(snippets=snippets, task=task, elapsed_ms=elapsed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_records(self) -> list[Any]:
        """Fetch recent topic records from the pulse store."""
        if not self.pulse_store:
            return []
        try:
            records = self.pulse_store.get_recent_topics(hours=24)
            return records or []
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.debug("Pulse store fetch failed: %s", e)
            return []

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from text (simple TF approach)."""
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {w for w in words if w not in _STOP_WORDS}

    def _compute_quality(self, record: Any) -> float:
        """Compute a quality score for a record.

        Uses the TopicQualityFilter when available, otherwise falls back to
        a lightweight heuristic based on text length and volume.
        """
        topic_text = getattr(record, "topic_text", "") or ""

        try:
            from aragora.pulse.quality import TopicQualityFilter
            from aragora.pulse.ingestor import TrendingTopic

            topic_obj = TrendingTopic(
                platform=getattr(record, "platform", "unknown"),
                topic=topic_text,
                volume=getattr(record, "volume", 0),
                category=getattr(record, "category", ""),
            )
            qf = TopicQualityFilter()
            qs = qf.score_topic(topic_obj)
            return qs.overall_score
        except (ImportError, TypeError, ValueError, AttributeError):
            # Lightweight fallback
            length = len(topic_text)
            if length < 10:
                return 0.1
            if length < 30:
                return 0.4
            return 0.7

    def _compute_freshness(self, record: Any) -> float:
        """Compute freshness score (1.0 = brand new, 0.0 = stale)."""
        hours_ago = getattr(record, "hours_ago", 0.0)
        if not isinstance(hours_ago, (int, float)):
            hours_ago = 0.0

        if hours_ago <= 0:
            return 1.0
        if hours_ago >= self.freshness_max_hours:
            return 0.0

        # Exponential decay with half-life of 6 hours
        import math

        half_life = 6.0
        return math.exp(-0.693 * hours_ago / half_life)

    def _compute_relevance(
        self,
        record: Any,
        task_keywords: set[str],
    ) -> tuple[int, str]:
        """Compute relevance as keyword overlap count and rationale.

        Returns:
            (overlap_count, human-readable rationale)
        """
        topic_text = getattr(record, "topic_text", "") or ""
        topic_keywords = self._extract_keywords(topic_text)
        overlap = task_keywords & topic_keywords

        if not overlap:
            return 0, ""

        overlap_list = sorted(overlap)[:5]
        rationale = f"Shared terms: {', '.join(overlap_list)}"
        return len(overlap), rationale

    def _score_record(
        self,
        record: Any,
        task_keywords: set[str],
    ) -> EnrichedSnippet | None:
        """Score a single record and return an EnrichedSnippet, or None if filtered."""
        # Relevance gate
        overlap_count, rationale = self._compute_relevance(record, task_keywords)
        if self.min_keyword_overlap > 0 and overlap_count < self.min_keyword_overlap:
            return None

        # Quality gate
        quality = self._compute_quality(record)
        if quality < self.quality_threshold:
            return None

        # Freshness gate
        freshness = self._compute_freshness(record)
        hours_ago = getattr(record, "hours_ago", 0.0)
        if not isinstance(hours_ago, (int, float)):
            hours_ago = 0.0
        if hours_ago > self.freshness_max_hours:
            return None

        # Build snippet
        topic_text = getattr(record, "topic_text", "") or ""
        platform = getattr(record, "platform", "unknown") or "unknown"
        volume = getattr(record, "volume", 0) or 0
        category = getattr(record, "category", "") or ""

        if not rationale:
            rationale = "General trending context"

        return EnrichedSnippet(
            title=topic_text,
            source=platform,
            quality_score=round(quality, 2),
            freshness_score=round(freshness, 2),
            relevance_rationale=rationale,
            volume=volume,
            category=category,
            hours_ago=hours_ago,
        )


# ---------------------------------------------------------------------------
# Prompt formatting helper
# ---------------------------------------------------------------------------

MAX_PULSE_ENRICHMENT_WORDS = 200


def format_enrichment_for_prompt(
    result: EnrichmentResult,
    max_words: int = MAX_PULSE_ENRICHMENT_WORDS,
) -> str:
    """Format an EnrichmentResult as a concise prompt section.

    Returns empty string if there are no snippets.

    The output follows the format:
        ## Current Context
        Recent relevant developments:
        - [Topic] (Source, Quality: X/10)
          Relevance: <rationale>
        ...
    """
    if not result.has_context:
        return ""

    lines: list[str] = [
        "## Current Context",
        "Recent relevant developments:\n",
    ]

    for snippet in result.snippets:
        quality_10 = round(snippet.quality_score * 10, 1)
        freshness_label = _freshness_label(snippet.hours_ago)
        line = f"- {snippet.title} ({snippet.source}, Quality: {quality_10}/10, {freshness_label})"
        lines.append(line)
        if snippet.relevance_rationale:
            lines.append(f"  Relevance: {snippet.relevance_rationale}")

    lines.append("")
    lines.append("Consider these developments when formulating your response.")

    text = "\n".join(lines)

    # Enforce word cap
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "..."

    return text


def _freshness_label(hours_ago: float) -> str:
    """Human-readable freshness label."""
    if hours_ago <= 0:
        return "just now"
    if hours_ago < 1:
        minutes = int(hours_ago * 60)
        return f"{minutes}m ago"
    if hours_ago < 24:
        return f"{hours_ago:.1f}h ago"
    days = hours_ago / 24
    return f"{days:.1f}d ago"
