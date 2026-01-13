"""
Evidence Quality Scoring.

Provides comprehensive quality scoring for evidence snippets including:
- Relevance scoring (how relevant to the query/topic)
- Freshness scoring (temporal relevance)
- Authority scoring (source trustworthiness)
- Overall quality score (weighted combination)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from aragora.evidence.metadata import EnrichedMetadata, SourceType

logger = logging.getLogger(__name__)


class QualityTier(str, Enum):
    """Quality tier classification for evidence."""

    EXCELLENT = "excellent"  # >= 0.85
    GOOD = "good"            # >= 0.70
    FAIR = "fair"            # >= 0.50
    POOR = "poor"            # >= 0.30
    UNRELIABLE = "unreliable"  # < 0.30

    @classmethod
    def from_score(cls, score: float) -> "QualityTier":
        """Classify a score into a quality tier."""
        if score >= 0.85:
            return cls.EXCELLENT
        elif score >= 0.70:
            return cls.GOOD
        elif score >= 0.50:
            return cls.FAIR
        elif score >= 0.30:
            return cls.POOR
        else:
            return cls.UNRELIABLE


@dataclass
class QualityScores:
    """Quality scores for an evidence snippet."""

    relevance_score: float = 0.5   # 0-1, how relevant to the query
    freshness_score: float = 0.5   # 0-1, temporal freshness
    authority_score: float = 0.5   # 0-1, source authority/trustworthiness
    completeness_score: float = 0.5  # 0-1, how complete the evidence is
    consistency_score: float = 0.5   # 0-1, internal consistency

    # Weights for overall score calculation
    relevance_weight: float = 0.35
    freshness_weight: float = 0.15
    authority_weight: float = 0.30
    completeness_weight: float = 0.10
    consistency_weight: float = 0.10

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        return (
            self.relevance_score * self.relevance_weight +
            self.freshness_score * self.freshness_weight +
            self.authority_score * self.authority_weight +
            self.completeness_score * self.completeness_weight +
            self.consistency_score * self.consistency_weight
        )

    @property
    def quality_tier(self) -> QualityTier:
        """Get the quality tier based on overall score."""
        return QualityTier.from_score(self.overall_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "relevance_score": self.relevance_score,
            "freshness_score": self.freshness_score,
            "authority_score": self.authority_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "overall_score": self.overall_score,
            "quality_tier": self.quality_tier.value,
            "weights": {
                "relevance": self.relevance_weight,
                "freshness": self.freshness_weight,
                "authority": self.authority_weight,
                "completeness": self.completeness_weight,
                "consistency": self.consistency_weight,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityScores":
        """Create from dictionary."""
        weights = data.get("weights", {})
        return cls(
            relevance_score=data.get("relevance_score", 0.5),
            freshness_score=data.get("freshness_score", 0.5),
            authority_score=data.get("authority_score", 0.5),
            completeness_score=data.get("completeness_score", 0.5),
            consistency_score=data.get("consistency_score", 0.5),
            relevance_weight=weights.get("relevance", 0.35),
            freshness_weight=weights.get("freshness", 0.15),
            authority_weight=weights.get("authority", 0.30),
            completeness_weight=weights.get("completeness", 0.10),
            consistency_weight=weights.get("consistency", 0.10),
        )


@dataclass
class QualityContext:
    """Context for quality scoring."""

    query: str = ""                      # The original query/topic
    keywords: List[str] = field(default_factory=list)  # Keywords to match
    required_topics: Set[str] = field(default_factory=set)  # Must-have topics
    preferred_sources: Set[str] = field(default_factory=set)  # Preferred sources
    blocked_sources: Set[str] = field(default_factory=set)  # Sources to penalize
    max_age_days: int = 365              # Maximum age for freshness
    min_word_count: int = 50             # Minimum content length
    require_citations: bool = False      # Require citations for high score


class QualityScorer:
    """Scores evidence quality based on multiple dimensions."""

    # Authority scores by source type
    SOURCE_TYPE_AUTHORITY = {
        SourceType.ACADEMIC: 0.9,
        SourceType.DOCUMENTATION: 0.85,
        SourceType.CODE: 0.75,
        SourceType.NEWS: 0.65,
        SourceType.LOCAL: 0.70,
        SourceType.API: 0.70,
        SourceType.DATABASE: 0.75,
        SourceType.WEB: 0.50,
        SourceType.SOCIAL: 0.40,
        SourceType.UNKNOWN: 0.30,
    }

    # Domain authority scores (well-known authoritative sources)
    DOMAIN_AUTHORITY = {
        # Academic
        "arxiv.org": 0.95,
        "nature.com": 0.95,
        "science.org": 0.95,
        "ieee.org": 0.90,
        "acm.org": 0.90,
        # Documentation
        "docs.python.org": 0.90,
        "developer.mozilla.org": 0.90,
        "docs.microsoft.com": 0.85,
        "docs.aws.amazon.com": 0.85,
        # Code
        "github.com": 0.80,
        "gitlab.com": 0.75,
        "stackoverflow.com": 0.70,
        # News
        "reuters.com": 0.80,
        "bbc.com": 0.75,
        "nytimes.com": 0.75,
    }

    def __init__(
        self,
        default_context: Optional[QualityContext] = None,
    ):
        """Initialize the quality scorer.

        Args:
            default_context: Default context for scoring
        """
        self.default_context = default_context or QualityContext()

    def score(
        self,
        content: str,
        metadata: Optional[EnrichedMetadata] = None,
        context: Optional[QualityContext] = None,
        url: Optional[str] = None,
        source: Optional[str] = None,
        fetched_at: Optional[datetime] = None,
    ) -> QualityScores:
        """Score evidence quality.

        Args:
            content: The evidence content
            metadata: Optional enriched metadata
            context: Optional scoring context
            url: Optional source URL
            source: Optional source name
            fetched_at: Optional fetch timestamp

        Returns:
            QualityScores with all dimension scores
        """
        ctx = context or self.default_context

        scores = QualityScores()

        # Calculate individual scores
        scores.relevance_score = self._score_relevance(content, ctx)
        scores.freshness_score = self._score_freshness(
            metadata.timestamp if metadata else fetched_at,
            metadata.provenance.publication_date if metadata and metadata.provenance else None,
            ctx.max_age_days,
        )
        scores.authority_score = self._score_authority(
            metadata.source_type if metadata else None,
            url,
            source,
            metadata.provenance if metadata else None,
            ctx,
        )
        scores.completeness_score = self._score_completeness(
            content,
            metadata,
            ctx,
        )
        scores.consistency_score = self._score_consistency(content)

        return scores

    def _score_relevance(self, content: str, context: QualityContext) -> float:
        """Score relevance to the query/topic."""
        if not context.query and not context.keywords:
            return 0.5  # Neutral if no context

        score = 0.0
        content_lower = content.lower()

        # Query matching
        if context.query:
            query_words = context.query.lower().split()
            matched = sum(1 for w in query_words if w in content_lower)
            if query_words:
                score += 0.3 * (matched / len(query_words))

        # Keyword matching
        if context.keywords:
            matched = sum(1 for k in context.keywords if k.lower() in content_lower)
            score += 0.4 * (matched / len(context.keywords))

        # Required topics
        if context.required_topics:
            matched = sum(1 for t in context.required_topics if t.lower() in content_lower)
            topic_score = matched / len(context.required_topics)
            score += 0.3 * topic_score
        else:
            score += 0.3 * 0.5  # Neutral if no required topics

        return min(1.0, score)

    def _score_freshness(
        self,
        fetched_at: Optional[datetime],
        published_at: Optional[datetime],
        max_age_days: int,
    ) -> float:
        """Score temporal freshness."""
        # Use publication date if available, otherwise fetch date
        reference_date = published_at or fetched_at

        if not reference_date:
            return 0.5  # Neutral if no date

        age = datetime.now() - reference_date
        age_days = age.total_seconds() / 86400

        if age_days < 1:
            return 1.0  # Very fresh
        elif age_days < 7:
            return 0.95
        elif age_days < 30:
            return 0.85
        elif age_days < 90:
            return 0.70
        elif age_days < 365:
            return 0.55
        elif age_days < max_age_days:
            # Linear decay from 0.55 to 0.30
            progress = (age_days - 365) / (max_age_days - 365)
            return 0.55 - (0.25 * progress)
        else:
            return 0.30  # Stale

    def _score_authority(
        self,
        source_type: Optional[SourceType],
        url: Optional[str],
        source: Optional[str],
        provenance: Optional[Any],  # Provenance
        context: QualityContext,
    ) -> float:
        """Score source authority and trustworthiness."""
        score = 0.5  # Base score

        # Source type authority
        if source_type:
            score = self.SOURCE_TYPE_AUTHORITY.get(source_type, 0.5)

        # Domain-specific authority
        if url:
            from urllib.parse import urlparse
            try:
                domain = urlparse(url).netloc.lower().replace("www.", "")
                if domain in self.DOMAIN_AUTHORITY:
                    # Blend with source type score
                    score = (score + self.DOMAIN_AUTHORITY[domain]) / 2
            except Exception as e:
                logger.debug(f"Could not parse URL for authority scoring: {e}")

        # Provenance adjustments
        if provenance:
            if provenance.peer_reviewed:
                score = min(1.0, score + 0.15)
            if provenance.doi:
                score = min(1.0, score + 0.1)
            if provenance.citation_count and provenance.citation_count > 10:
                score = min(1.0, score + 0.05)
            if provenance.author:
                score = min(1.0, score + 0.03)

        # Context adjustments
        if context.preferred_sources and source:
            if source.lower() in {s.lower() for s in context.preferred_sources}:
                score = min(1.0, score + 0.1)

        if context.blocked_sources and source:
            if source.lower() in {s.lower() for s in context.blocked_sources}:
                score = max(0.0, score - 0.3)

        return score

    def _score_completeness(
        self,
        content: str,
        metadata: Optional[EnrichedMetadata],
        context: QualityContext,
    ) -> float:
        """Score content completeness."""
        score = 0.5

        word_count = len(content.split())

        # Length scoring
        if word_count < context.min_word_count:
            score = 0.2 + (0.3 * word_count / context.min_word_count)
        elif word_count < 100:
            score = 0.5
        elif word_count < 300:
            score = 0.65
        elif word_count < 500:
            score = 0.75
        elif word_count < 1000:
            score = 0.85
        else:
            score = 0.90

        # Metadata completeness
        if metadata:
            if metadata.has_citations:
                score = min(1.0, score + 0.05)
            if metadata.has_data:
                score = min(1.0, score + 0.05)
            if metadata.topics:
                score = min(1.0, score + 0.03)

        # Citation requirement
        if context.require_citations and metadata and not metadata.has_citations:
            score = max(0.0, score - 0.2)

        return score

    def _score_consistency(self, content: str) -> float:
        """Score internal consistency of content."""
        # Simple heuristics for consistency
        score = 0.7  # Base assumption of reasonable consistency

        # Check for contradictory language patterns
        contradiction_patterns = [
            r'\bhowever\b.*\bbut\b',
            r'\balthough\b.*\bnevertheless\b',
            r'\bon one hand\b.*\bon the other hand\b',
        ]

        # Multiple contradictions may indicate nuanced but potentially confusing content
        contradictions = sum(
            1 for pattern in contradiction_patterns
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        )

        if contradictions > 2:
            score -= 0.1

        # Check for uncertain language
        uncertain_patterns = [
            r'\bmaybe\b',
            r'\bperhaps\b',
            r'\bpossibly\b',
            r'\bmight be\b',
            r'\bcould be\b',
            r'\bunclear\b',
            r'\bunknown\b',
        ]

        uncertainties = sum(
            1 for pattern in uncertain_patterns
            if re.search(pattern, content, re.IGNORECASE)
        )

        if uncertainties > 3:
            score -= 0.1

        # Presence of definitive statements is positive
        definitive_patterns = [
            r'\bdefinitely\b',
            r'\bcertainly\b',
            r'\bproven\b',
            r'\bdemonstrated\b',
            r'\bestablished\b',
        ]

        definitives = sum(
            1 for pattern in definitive_patterns
            if re.search(pattern, content, re.IGNORECASE)
        )

        if definitives > 0:
            score = min(1.0, score + 0.1)

        return max(0.3, min(1.0, score))


class QualityFilter:
    """Filter evidence based on quality thresholds."""

    def __init__(
        self,
        scorer: Optional[QualityScorer] = None,
        min_overall_score: float = 0.5,
        min_relevance_score: float = 0.3,
        min_authority_score: float = 0.3,
    ):
        """Initialize the quality filter.

        Args:
            scorer: QualityScorer instance
            min_overall_score: Minimum overall score to pass
            min_relevance_score: Minimum relevance score
            min_authority_score: Minimum authority score
        """
        self.scorer = scorer or QualityScorer()
        self.min_overall_score = min_overall_score
        self.min_relevance_score = min_relevance_score
        self.min_authority_score = min_authority_score

    def filter(
        self,
        evidence_list: List[Any],  # List of EvidenceSnippet
        context: Optional[QualityContext] = None,
    ) -> List[Any]:
        """Filter evidence by quality.

        Args:
            evidence_list: List of evidence snippets
            context: Optional quality context

        Returns:
            Filtered list of evidence that passes quality thresholds
        """
        passed = []

        for evidence in evidence_list:
            scores = self.scorer.score(
                content=evidence.snippet,
                url=evidence.url,
                source=evidence.source,
                fetched_at=evidence.fetched_at,
                context=context,
            )

            if (
                scores.overall_score >= self.min_overall_score
                and scores.relevance_score >= self.min_relevance_score
                and scores.authority_score >= self.min_authority_score
            ):
                passed.append(evidence)

        return passed

    def rank(
        self,
        evidence_list: List[Any],
        context: Optional[QualityContext] = None,
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """Rank evidence by quality score.

        Args:
            evidence_list: List of evidence snippets
            context: Optional quality context
            top_k: Optional limit on results

        Returns:
            List of (evidence, scores) tuples sorted by overall score
        """
        scored = []

        for evidence in evidence_list:
            scores = self.scorer.score(
                content=evidence.snippet,
                url=evidence.url,
                source=evidence.source,
                fetched_at=evidence.fetched_at,
                context=context,
            )
            scored.append((evidence, scores))

        # Sort by overall score descending
        scored.sort(key=lambda x: x[1].overall_score, reverse=True)

        if top_k:
            scored = scored[:top_k]

        return scored


def score_evidence_snippet(
    snippet: Any,  # EvidenceSnippet
    query: str = "",
    keywords: Optional[List[str]] = None,
    scorer: Optional[QualityScorer] = None,
) -> QualityScores:
    """Convenience function to score an evidence snippet.

    Args:
        snippet: An EvidenceSnippet object
        query: Optional query for relevance scoring
        keywords: Optional keywords for relevance scoring
        scorer: Optional QualityScorer instance

    Returns:
        QualityScores for the snippet
    """
    if scorer is None:
        scorer = QualityScorer()

    context = QualityContext(
        query=query,
        keywords=keywords or [],
    )

    return scorer.score(
        content=snippet.snippet,
        url=snippet.url,
        source=snippet.source,
        fetched_at=snippet.fetched_at,
        context=context,
    )
