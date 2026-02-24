"""
Knowledge Gap Detection and Recommendations.

Identifies gaps in the organization's knowledge base and recommends
areas to strengthen. Works with the Knowledge Mound to analyze:
- Coverage gaps: domains/topics with sparse or missing knowledge
- Staleness: outdated entries that need refresh
- Contradictions: conflicting knowledge entries
- Debate receipt analysis: low-confidence decisions and question topics
- Frequently asked topics: query patterns with sparse knowledge coverage
- Coverage map: domain-by-domain coverage overview
- Recommendations: prioritized actions to improve knowledge quality

Usage:
    from aragora.knowledge.gap_detector import KnowledgeGapDetector

    detector = KnowledgeGapDetector(mound)
    gaps = await detector.detect_coverage_gaps("legal")
    stale = await detector.detect_staleness(max_age_days=90)
    contradictions = await detector.detect_contradictions()
    recommendations = await detector.get_recommendations()
    score = await detector.get_coverage_score("technical")
    coverage_map = await detector.get_coverage_map()

    # Track debate receipts and queries for gap signal enrichment
    await detector.analyze_debate_receipt(receipt)
    detector.record_query("contract termination clauses")
    frequent_gaps = detector.get_frequently_asked_gaps(min_queries=3)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


class Priority(str, Enum):
    """Priority level for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendedAction(str, Enum):
    """Action type for recommendations."""

    CREATE = "create"
    UPDATE = "update"
    REVIEW = "review"
    ARCHIVE = "archive"
    ACQUIRE = "acquire"


@dataclass
class KnowledgeGap:
    """A detected gap in knowledge coverage."""

    domain: str
    topic: str
    description: str
    severity: float  # 0-1, how critical the gap is
    expected_items: int
    actual_items: int
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "topic": self.topic,
            "description": self.description,
            "severity": round(self.severity, 3),
            "expected_items": self.expected_items,
            "actual_items": self.actual_items,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class StaleKnowledge:
    """A stale knowledge entry that needs updating."""

    item_id: str
    content_preview: str
    domain: str
    age_days: float
    confidence: float
    last_updated: datetime | None
    staleness_score: float  # 0-1, how stale the entry is

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "content_preview": self.content_preview[:200],
            "domain": self.domain,
            "age_days": round(self.age_days, 1),
            "confidence": round(self.confidence, 3),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "staleness_score": round(self.staleness_score, 3),
        }


@dataclass
class Contradiction:
    """A contradiction between knowledge entries."""

    item_a_id: str
    item_b_id: str
    item_a_preview: str
    item_b_preview: str
    domain: str
    conflict_score: float  # 0-1, how conflicting the entries are
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_a_id": self.item_a_id,
            "item_b_id": self.item_b_id,
            "item_a_preview": self.item_a_preview[:200],
            "item_b_preview": self.item_b_preview[:200],
            "domain": self.domain,
            "conflict_score": round(self.conflict_score, 3),
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class Recommendation:
    """A prioritized recommendation for improving knowledge quality."""

    priority: Priority
    action: RecommendedAction
    description: str
    domain: str
    impact_score: float  # 0-1, estimated impact of the action
    related_gap_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "priority": self.priority.value,
            "action": self.action.value,
            "description": self.description,
            "domain": self.domain,
            "impact_score": round(self.impact_score, 3),
            "related_gap_id": self.related_gap_id,
            "metadata": self.metadata,
        }


@dataclass
class DebateInsight:
    """Insight extracted from a debate receipt indicating a knowledge gap.

    When a debate finishes with low confidence or high disagreement, that
    signals a domain where the Knowledge Mound may be lacking.
    """

    debate_id: str
    topic: str
    domain: str
    confidence: float  # Final debate confidence (lower = bigger gap signal)
    disagreement_score: float  # 0-1, how much agents disagreed
    question: str  # The question or task that was debated
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "disagreement_score": round(self.disagreement_score, 3),
            "question": self.question[:300],
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class FrequentlyAskedGap:
    """A topic that is frequently queried but has sparse knowledge coverage.

    Tracks the intersection of high demand (many queries) and low supply
    (few or low-quality knowledge items).
    """

    topic: str
    query_count: int
    coverage_score: float  # 0-1, how well covered this topic is
    gap_severity: float  # 0-1, higher = more urgent
    sample_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "query_count": self.query_count,
            "coverage_score": round(self.coverage_score, 3),
            "gap_severity": round(self.gap_severity, 3),
            "sample_queries": self.sample_queries[:5],
        }


@dataclass
class DomainCoverageEntry:
    """Coverage information for a single domain in the coverage map."""

    domain: str
    coverage_score: float
    total_items: int
    expected_items: int
    average_confidence: float
    gap_count: int
    stale_count: int
    contradiction_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "coverage_score": round(self.coverage_score, 3),
            "total_items": self.total_items,
            "expected_items": self.expected_items,
            "average_confidence": round(self.average_confidence, 3),
            "gap_count": self.gap_count,
            "stale_count": self.stale_count,
            "contradiction_count": self.contradiction_count,
        }


# =============================================================================
# Gap Detector
# =============================================================================


# Minimum expected items per domain for coverage analysis
_DOMAIN_EXPECTATIONS: dict[str, int] = {
    "legal": 20,
    "financial": 20,
    "technical": 15,
    "healthcare": 15,
    "operational": 10,
}

# Subdomain minimum expectations
_SUBDOMAIN_EXPECTATIONS: dict[str, int] = {
    "legal/contracts": 10,
    "legal/compliance": 8,
    "financial/accounting": 8,
    "financial/audit": 6,
    "technical/architecture": 8,
    "technical/security": 8,
    "healthcare/clinical": 8,
    "healthcare/compliance": 6,
    "operational/hr": 5,
    "operational/strategy": 5,
}


class KnowledgeGapDetector:
    """Detects gaps in the knowledge base and generates improvement recommendations.

    Analyzes the Knowledge Mound to find:
    - Topics with sparse coverage
    - Outdated entries needing refresh
    - Conflicting knowledge entries
    - Debate receipts with low confidence (knowledge gap signals)
    - Frequently asked topics with sparse coverage
    - Prioritized improvement recommendations

    The detector also tracks queries and debate outcomes over time to identify
    patterns in what knowledge is missing. Call ``record_query()`` and
    ``analyze_debate_receipt()`` to feed these signals into the gap analysis.

    Args:
        mound: KnowledgeMound instance to analyze
        workspace_id: Workspace scope for analysis
    """

    def __init__(
        self,
        mound: KnowledgeMound,
        workspace_id: str = "default",
    ) -> None:
        self._mound = mound
        self._workspace_id = workspace_id
        # Track debate receipt insights (low-confidence decisions)
        self._debate_insights: list[DebateInsight] = []
        # Track query topics for frequently-asked-gap detection
        # Maps normalized topic -> list of raw queries
        self._query_topics: dict[str, list[str]] = defaultdict(list)

    async def detect_coverage_gaps(
        self,
        domain: str,
        min_expected: int | None = None,
    ) -> list[KnowledgeGap]:
        """Find topics within a domain that have sparse or missing coverage.

        Queries the Knowledge Mound for items in the specified domain and
        compares against expected coverage thresholds.

        Args:
            domain: The domain to analyze (e.g., "legal", "technical")
            min_expected: Override for minimum expected items (default: use built-in thresholds)

        Returns:
            List of KnowledgeGap entries sorted by severity (highest first)
        """
        gaps: list[KnowledgeGap] = []

        # Determine expected item count for this domain
        expected = min_expected or _DOMAIN_EXPECTATIONS.get(domain, 10)

        try:
            result = await self._mound.query(
                query=domain,
                workspace_id=self._workspace_id,
                limit=200,
            )
            items = result.items if hasattr(result, "items") else []
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to query domain '%s': %s", domain, e)
            items = []

        actual_count = len(items)

        # Check overall domain coverage
        if actual_count < expected:
            severity = 1.0 - (actual_count / max(expected, 1))
            gaps.append(
                KnowledgeGap(
                    domain=domain,
                    topic=domain,
                    description=f"Domain '{domain}' has {actual_count} items, expected at least {expected}",
                    severity=min(severity, 1.0),
                    expected_items=expected,
                    actual_items=actual_count,
                )
            )

        # Check subdomain coverage
        for subdomain, sub_expected in _SUBDOMAIN_EXPECTATIONS.items():
            if not subdomain.startswith(f"{domain}/"):
                continue

            sub_topic = subdomain.split("/", 1)[1] if "/" in subdomain else subdomain

            try:
                sub_result = await self._mound.query(
                    query=sub_topic,
                    workspace_id=self._workspace_id,
                    limit=100,
                )
                sub_items = sub_result.items if hasattr(sub_result, "items") else []
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to query subdomain '%s': %s", subdomain, e)
                sub_items = []

            sub_count = len(sub_items)
            if sub_count < sub_expected:
                severity = 1.0 - (sub_count / max(sub_expected, 1))
                gaps.append(
                    KnowledgeGap(
                        domain=subdomain,
                        topic=sub_topic,
                        description=(
                            f"Subdomain '{subdomain}' has {sub_count} items, "
                            f"expected at least {sub_expected}"
                        ),
                        severity=min(severity, 1.0),
                        expected_items=sub_expected,
                        actual_items=sub_count,
                    )
                )

        # Sort by severity descending
        gaps.sort(key=lambda g: g.severity, reverse=True)
        return gaps

    async def detect_staleness(
        self,
        max_age_days: int = 90,
    ) -> list[StaleKnowledge]:
        """Find knowledge entries that are outdated based on age.

        Args:
            max_age_days: Maximum age in days before an entry is considered stale

        Returns:
            List of StaleKnowledge entries sorted by staleness score (highest first)
        """
        stale_entries: list[StaleKnowledge] = []
        cutoff = datetime.now() - timedelta(days=max_age_days)

        try:
            # Use the mound's staleness detection if available
            if hasattr(self._mound, "get_stale_knowledge"):
                stale_items = await self._mound.get_stale_knowledge(
                    threshold=0.5,
                    workspace_id=self._workspace_id,
                )
                items = stale_items if isinstance(stale_items, list) else []
            else:
                # Fallback: query all and filter by age
                result = await self._mound.query(
                    query="",
                    workspace_id=self._workspace_id,
                    limit=500,
                )
                items = result.items if hasattr(result, "items") else []
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to detect staleness: %s", e)
            items = []

        now = datetime.now()
        for item in items:
            updated_at = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
            if updated_at is None:
                continue

            # Parse datetime if string
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at)
                except (ValueError, TypeError):
                    continue

            if updated_at < cutoff:
                age_days = (now - updated_at).total_seconds() / 86400
                staleness_score = min(1.0, age_days / (max_age_days * 2))
                confidence = getattr(item, "confidence", 0.5)

                # Lower confidence items that are also stale are more concerning
                staleness_score = min(1.0, staleness_score * (2.0 - confidence))

                content = getattr(item, "content", "") or ""
                domain = getattr(item, "domain", "") or getattr(item, "source_type", "unknown")
                if hasattr(domain, "value"):
                    domain = domain.value

                stale_entries.append(
                    StaleKnowledge(
                        item_id=getattr(item, "id", "") or "",
                        content_preview=content[:200],
                        domain=str(domain),
                        age_days=age_days,
                        confidence=confidence,
                        last_updated=updated_at,
                        staleness_score=staleness_score,
                    )
                )

        stale_entries.sort(key=lambda s: s.staleness_score, reverse=True)
        return stale_entries

    async def detect_contradictions(self) -> list[Contradiction]:
        """Find conflicting knowledge entries in the mound.

        Uses the KnowledgeMound's built-in contradiction detection when available,
        otherwise returns an empty list.

        Returns:
            List of Contradiction entries sorted by conflict score (highest first)
        """
        contradictions: list[Contradiction] = []

        try:
            if hasattr(self._mound, "detect_contradictions"):
                raw = await self._mound.detect_contradictions(
                    workspace_id=self._workspace_id,
                )
                raw_list = raw if isinstance(raw, list) else []
            elif hasattr(self._mound, "get_contradictions"):
                raw_list = await self._mound.get_contradictions(  # type: ignore[call-arg]
                    workspace_id=self._workspace_id,
                )
            else:
                logger.debug("Mound does not support contradiction detection")
                return []
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to detect contradictions: %s", e)
            return []

        for c in raw_list:
            item_a_id = getattr(c, "item_a_id", "") or ""
            item_b_id = getattr(c, "item_b_id", "") or ""
            conflict_score = getattr(c, "conflict_score", 0.5)

            # Try to get content previews
            item_a_preview = ""
            item_b_preview = ""
            try:
                if hasattr(self._mound, "get"):
                    item_a = await self._mound.get(item_a_id)
                    item_b = await self._mound.get(item_b_id)
                    if item_a:
                        item_a_preview = getattr(item_a, "content", "")[:200]
                    if item_b:
                        item_b_preview = getattr(item_b, "content", "")[:200]
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

            domain = getattr(c, "domain", "") or "unknown"
            detected_at = getattr(c, "detected_at", None) or datetime.now()

            contradictions.append(
                Contradiction(
                    item_a_id=item_a_id,
                    item_b_id=item_b_id,
                    item_a_preview=item_a_preview,
                    item_b_preview=item_b_preview,
                    domain=str(domain),
                    conflict_score=conflict_score,
                    detected_at=detected_at,
                )
            )

        contradictions.sort(key=lambda c: c.conflict_score, reverse=True)
        return contradictions

    async def analyze_debate_receipt(
        self,
        receipt: dict[str, Any] | Any,
    ) -> DebateInsight | None:
        """Analyze a debate receipt to identify knowledge gaps.

        Examines the outcome of a debate to determine if low confidence
        or high disagreement indicates a domain where knowledge is lacking.
        Records the insight for future recommendation generation.

        Args:
            receipt: A debate receipt dict or object with fields like
                debate_id, topic/task, confidence, consensus_score, domain.

        Returns:
            DebateInsight if a gap signal was detected, None otherwise.
        """
        # Extract fields from dict or object
        if isinstance(receipt, dict):
            debate_id = receipt.get("debate_id", receipt.get("id", ""))
            topic = receipt.get("topic", receipt.get("task", ""))
            confidence = float(receipt.get("confidence", receipt.get("consensus_confidence", 1.0)))
            consensus_score = float(receipt.get("consensus_score", receipt.get("agreement", 1.0)))
            domain = receipt.get("domain", "general")
            question = receipt.get("question", receipt.get("task", topic))
        else:
            debate_id = getattr(receipt, "debate_id", getattr(receipt, "id", ""))
            topic = getattr(receipt, "topic", getattr(receipt, "task", ""))
            confidence = float(
                getattr(receipt, "confidence", getattr(receipt, "consensus_confidence", 1.0))
            )
            consensus_score = float(
                getattr(receipt, "consensus_score", getattr(receipt, "agreement", 1.0))
            )
            domain = getattr(receipt, "domain", "general")
            question = getattr(receipt, "question", getattr(receipt, "task", topic))

        # Only flag as a gap signal if confidence is below threshold
        # or disagreement is high
        disagreement_score = max(0.0, 1.0 - consensus_score)
        if confidence >= 0.7 and disagreement_score < 0.3:
            return None

        # Classify the domain from the topic text if not provided
        if not domain or domain == "general":
            domain = self._classify_domain(str(topic) + " " + str(question))

        insight = DebateInsight(
            debate_id=str(debate_id),
            topic=str(topic),
            domain=str(domain),
            confidence=confidence,
            disagreement_score=disagreement_score,
            question=str(question),
        )

        self._debate_insights.append(insight)
        logger.info(
            "Debate gap signal: debate=%s domain=%s confidence=%.2f disagreement=%.2f",
            debate_id,
            domain,
            confidence,
            disagreement_score,
        )
        return insight

    def _classify_domain(self, text: str) -> str:
        """Classify text into a domain using keyword matching.

        Uses the DOMAIN_KEYWORDS from taxonomy if available, otherwise
        uses the built-in subdomain expectations keys.

        Args:
            text: Text to classify

        Returns:
            Domain string (e.g., "legal", "technical/security")
        """
        text_lower = text.lower()

        try:
            from aragora.knowledge.mound.taxonomy import DOMAIN_KEYWORDS

            best_domain = "general"
            best_score = 0
            for domain_path, keywords in DOMAIN_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > best_score:
                    best_score = score
                    best_domain = domain_path
            if best_score > 0:
                return best_domain
        except ImportError:
            pass

        # Fallback: check top-level domain names
        for domain in _DOMAIN_EXPECTATIONS:
            if domain in text_lower:
                return domain

        return "general"

    def record_query(self, query: str) -> None:
        """Record a query for frequently-asked-gap tracking.

        Call this each time a user or agent queries the Knowledge Mound.
        The detector accumulates queries to identify topics that are
        frequently requested but poorly covered.

        Args:
            query: The query string submitted to the Knowledge Mound.
        """
        if not query or not query.strip():
            return

        # Normalize: lowercase, strip whitespace
        normalized = query.strip().lower()

        # Extract a topic key (first 3 significant words)
        words = [w for w in normalized.split() if len(w) > 2]
        topic_key = " ".join(words[:3]) if words else normalized

        self._query_topics[topic_key].append(query)

    def get_frequently_asked_gaps(
        self,
        min_queries: int = 3,
        max_coverage: float = 0.5,
    ) -> list[FrequentlyAskedGap]:
        """Identify topics that are frequently queried but have sparse coverage.

        Cross-references recorded query patterns against coverage scores
        to find high-demand, low-supply knowledge areas.

        This is a synchronous method that returns cached data. For
        coverage scores that require async mound queries, call
        ``get_frequently_asked_gaps_async()`` instead.

        Args:
            min_queries: Minimum number of queries for a topic to be considered
            max_coverage: Maximum coverage score to be considered a gap

        Returns:
            List of FrequentlyAskedGap entries sorted by gap severity
        """
        gaps: list[FrequentlyAskedGap] = []

        for topic_key, queries in self._query_topics.items():
            if len(queries) < min_queries:
                continue

            # Estimate coverage from query count vs domain expectations
            # Higher query count with no resolution = bigger gap
            query_count = len(queries)
            # Placeholder coverage score (will be 0 since we have no async here)
            coverage_score = 0.0
            gap_severity = min(1.0, query_count / 20.0) * (1.0 - coverage_score)

            if coverage_score <= max_coverage:
                gaps.append(
                    FrequentlyAskedGap(
                        topic=topic_key,
                        query_count=query_count,
                        coverage_score=coverage_score,
                        gap_severity=gap_severity,
                        sample_queries=list(dict.fromkeys(queries))[:5],
                    )
                )

        gaps.sort(key=lambda g: g.gap_severity, reverse=True)
        return gaps

    async def get_frequently_asked_gaps_async(
        self,
        min_queries: int = 3,
        max_coverage: float = 0.5,
    ) -> list[FrequentlyAskedGap]:
        """Identify frequently queried topics with sparse coverage (async version).

        Like ``get_frequently_asked_gaps()`` but queries the Knowledge Mound
        for actual coverage scores.

        Args:
            min_queries: Minimum number of queries for a topic to be considered
            max_coverage: Maximum coverage score to be considered a gap

        Returns:
            List of FrequentlyAskedGap entries sorted by gap severity
        """
        gaps: list[FrequentlyAskedGap] = []

        for topic_key, queries in self._query_topics.items():
            if len(queries) < min_queries:
                continue

            query_count = len(queries)

            # Query the mound for this topic to estimate coverage
            try:
                result = await self._mound.query(
                    query=topic_key,
                    workspace_id=self._workspace_id,
                    limit=50,
                )
                items = result.items if hasattr(result, "items") else []
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to query topic '%s' for gap analysis: %s", topic_key, e)
                items = []

            # Coverage score based on item count and confidence
            if items:
                item_count = len(items)
                confidences = [
                    float(getattr(item, "confidence", 0.5))
                    for item in items
                    if getattr(item, "confidence", None) is not None
                ]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
                # Coverage combines quantity and quality
                coverage_score = min(1.0, (item_count / 10.0) * avg_conf)
            else:
                coverage_score = 0.0

            if coverage_score <= max_coverage:
                # Gap severity combines demand (query frequency) with supply shortage
                demand_factor = min(1.0, query_count / 20.0)
                gap_severity = demand_factor * (1.0 - coverage_score)

                gaps.append(
                    FrequentlyAskedGap(
                        topic=topic_key,
                        query_count=query_count,
                        coverage_score=round(coverage_score, 3),
                        gap_severity=round(gap_severity, 3),
                        sample_queries=list(dict.fromkeys(queries))[:5],
                    )
                )

        gaps.sort(key=lambda g: g.gap_severity, reverse=True)
        return gaps

    def get_debate_insights(
        self,
        domain: str | None = None,
        min_disagreement: float = 0.0,
    ) -> list[DebateInsight]:
        """Get debate receipt insights indicating knowledge gaps.

        Args:
            domain: Optional domain filter
            min_disagreement: Minimum disagreement score to include

        Returns:
            List of DebateInsight entries sorted by confidence (lowest first)
        """
        insights = self._debate_insights
        if domain:
            insights = [
                i for i in insights if i.domain == domain or i.domain.startswith(f"{domain}/")
            ]
        if min_disagreement > 0:
            insights = [i for i in insights if i.disagreement_score >= min_disagreement]

        return sorted(insights, key=lambda i: i.confidence)

    async def get_coverage_map(self) -> list[DomainCoverageEntry]:
        """Generate a coverage map across all known domains.

        Queries each domain in the taxonomy for item counts, confidence,
        gaps, staleness, and contradictions to build a comprehensive
        overview of knowledge health.

        Returns:
            List of DomainCoverageEntry sorted by coverage score (lowest first,
            so the weakest domains appear first).
        """
        entries: list[DomainCoverageEntry] = []

        for domain, expected in _DOMAIN_EXPECTATIONS.items():
            # Get coverage score
            coverage_score = await self.get_coverage_score(domain)

            # Query items for statistics
            try:
                result = await self._mound.query(
                    query=domain,
                    workspace_id=self._workspace_id,
                    limit=200,
                )
                items = result.items if hasattr(result, "items") else []
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to query domain '%s' for coverage map: %s", domain, e)
                items = []

            total_items = len(items)

            # Average confidence
            confidences = [
                float(getattr(item, "confidence", 0.5))
                for item in items
                if getattr(item, "confidence", None) is not None
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Count gaps
            try:
                gaps = await self.detect_coverage_gaps(domain)
                gap_count = len(gaps)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                gap_count = 0

            # Estimate stale and contradiction counts from cached data
            # (avoid expensive re-queries by using domain-filtered insights)
            stale_count = 0
            try:
                now = datetime.now()
                for item in items:
                    updated_at = getattr(item, "updated_at", None) or getattr(
                        item, "created_at", None
                    )
                    if updated_at is None:
                        continue
                    if isinstance(updated_at, str):
                        try:
                            updated_at = datetime.fromisoformat(updated_at)
                        except (ValueError, TypeError):
                            continue
                    if (now - updated_at).days > 90:
                        stale_count += 1
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

            contradiction_count = 0  # Would require domain-specific contradiction query

            entries.append(
                DomainCoverageEntry(
                    domain=domain,
                    coverage_score=coverage_score,
                    total_items=total_items,
                    expected_items=expected,
                    average_confidence=avg_confidence,
                    gap_count=gap_count,
                    stale_count=stale_count,
                    contradiction_count=contradiction_count,
                )
            )

        # Sort by coverage score ascending (weakest first)
        entries.sort(key=lambda e: e.coverage_score)
        return entries

    async def get_recommendations(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[Recommendation]:
        """Generate prioritized recommendations for improving knowledge quality.

        Combines coverage gaps, staleness, and contradictions into actionable
        recommendations sorted by priority and impact.

        Args:
            domain: Optional domain filter; if None, analyzes all known domains
            limit: Maximum number of recommendations to return

        Returns:
            List of Recommendation entries sorted by priority then impact
        """
        recommendations: list[Recommendation] = []

        # Determine domains to analyze
        domains_to_check = [domain] if domain else list(_DOMAIN_EXPECTATIONS.keys())

        # Analyze coverage gaps
        for d in domains_to_check:
            try:
                gaps = await self.detect_coverage_gaps(d)
                for gap in gaps:
                    priority = (
                        Priority.HIGH
                        if gap.severity > 0.7
                        else (Priority.MEDIUM if gap.severity > 0.4 else Priority.LOW)
                    )
                    recommendations.append(
                        Recommendation(
                            priority=priority,
                            action=RecommendedAction.CREATE,
                            description=f"Add knowledge for {gap.topic}: {gap.description}",
                            domain=gap.domain,
                            impact_score=gap.severity,
                            metadata={
                                "gap_type": "coverage",
                                "actual": gap.actual_items,
                                "expected": gap.expected_items,
                            },
                        )
                    )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to get coverage gaps for '%s': %s", d, e)

        # Analyze staleness
        try:
            stale = await self.detect_staleness()
            for entry in stale[:limit]:
                priority = (
                    Priority.HIGH
                    if entry.staleness_score > 0.8
                    else (Priority.MEDIUM if entry.staleness_score > 0.5 else Priority.LOW)
                )
                action = (
                    RecommendedAction.ARCHIVE
                    if entry.staleness_score > 0.9 and entry.confidence < 0.3
                    else RecommendedAction.UPDATE
                )
                recommendations.append(
                    Recommendation(
                        priority=priority,
                        action=action,
                        description=(
                            f"{'Archive' if action == RecommendedAction.ARCHIVE else 'Update'} "
                            f"stale entry in {entry.domain} "
                            f"(age: {entry.age_days:.0f} days, confidence: {entry.confidence:.2f})"
                        ),
                        domain=entry.domain,
                        impact_score=entry.staleness_score,
                        metadata={"item_id": entry.item_id, "age_days": entry.age_days},
                    )
                )
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get staleness recommendations: %s", e)

        # Analyze contradictions
        try:
            contras = await self.detect_contradictions()
            for c in contras[:limit]:
                priority = (
                    Priority.HIGH
                    if c.conflict_score > 0.7
                    else (Priority.MEDIUM if c.conflict_score > 0.4 else Priority.LOW)
                )
                recommendations.append(
                    Recommendation(
                        priority=priority,
                        action=RecommendedAction.REVIEW,
                        description=(
                            f"Resolve contradiction in {c.domain} between "
                            f"items {c.item_a_id[:8]}... and {c.item_b_id[:8]}..."
                        ),
                        domain=c.domain,
                        impact_score=c.conflict_score,
                        metadata={"item_a_id": c.item_a_id, "item_b_id": c.item_b_id},
                    )
                )
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get contradiction recommendations: %s", e)

        # Analyze debate receipt insights (low-confidence decisions)
        insights = self.get_debate_insights(domain=domain)
        for insight in insights[:limit]:
            # Low confidence in a debate means the domain needs more knowledge
            impact = max(0.0, 1.0 - insight.confidence) * 0.5 + insight.disagreement_score * 0.5
            priority = (
                Priority.HIGH
                if impact > 0.7
                else (Priority.MEDIUM if impact > 0.4 else Priority.LOW)
            )
            recommendations.append(
                Recommendation(
                    priority=priority,
                    action=RecommendedAction.ACQUIRE,
                    description=(
                        f"Acquire knowledge for '{insight.topic}' in {insight.domain} "
                        f"(debate confidence: {insight.confidence:.2f}, "
                        f"disagreement: {insight.disagreement_score:.2f})"
                    ),
                    domain=insight.domain,
                    impact_score=round(impact, 3),
                    metadata={
                        "gap_type": "debate_signal",
                        "debate_id": insight.debate_id,
                        "confidence": insight.confidence,
                        "disagreement": insight.disagreement_score,
                    },
                )
            )

        # Analyze frequently asked topics with sparse coverage
        try:
            faq_gaps = self.get_frequently_asked_gaps()
            for faq in faq_gaps[:limit]:
                priority = (
                    Priority.HIGH
                    if faq.gap_severity > 0.7
                    else (Priority.MEDIUM if faq.gap_severity > 0.4 else Priority.LOW)
                )
                recommendations.append(
                    Recommendation(
                        priority=priority,
                        action=RecommendedAction.ACQUIRE,
                        description=(
                            f"Frequently queried topic '{faq.topic}' has low coverage "
                            f"(queried {faq.query_count} times, coverage: {faq.coverage_score:.2f})"
                        ),
                        domain="general",
                        impact_score=faq.gap_severity,
                        metadata={
                            "gap_type": "frequently_asked",
                            "query_count": faq.query_count,
                            "coverage_score": faq.coverage_score,
                        },
                    )
                )
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get frequently-asked recommendations: %s", e)

        # Sort: HIGH first, then by impact_score descending
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 2), -r.impact_score),
        )

        return recommendations[:limit]

    async def get_coverage_score(self, domain: str) -> float:
        """Calculate a 0-1 coverage score for the given domain.

        Combines item count, average confidence, and freshness into
        a single score. Higher is better.

        Args:
            domain: The domain to score (e.g., "legal", "technical")

        Returns:
            Float between 0.0 and 1.0 representing coverage quality
        """
        expected = _DOMAIN_EXPECTATIONS.get(domain, 10)

        try:
            result = await self._mound.query(
                query=domain,
                workspace_id=self._workspace_id,
                limit=200,
            )
            items = result.items if hasattr(result, "items") else []
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get coverage score for '%s': %s", domain, e)
            return 0.0

        if not items:
            return 0.0

        # Depth factor: how many items vs expected
        depth_factor = min(1.0, len(items) / max(expected, 1))

        # Confidence factor: average confidence across items
        confidences = []
        for item in items:
            conf = getattr(item, "confidence", None)
            if conf is not None:
                confidences.append(float(conf))
        confidence_factor = sum(confidences) / len(confidences) if confidences else 0.5

        # Freshness factor: based on average age
        now = datetime.now()
        ages: list[float] = []
        for item in items:
            updated_at = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
            if updated_at is None:
                continue
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at)
                except (ValueError, TypeError):
                    continue
            age_days = (now - updated_at).total_seconds() / 86400
            ages.append(age_days)

        if ages:
            avg_age = sum(ages) / len(ages)
            freshness_factor = max(0.0, 1.0 - (avg_age / 365.0))
        else:
            freshness_factor = 0.5

        # Weighted combination
        score = depth_factor * 0.4 + confidence_factor * 0.35 + freshness_factor * 0.25
        return round(min(1.0, max(0.0, score)), 3)


__all__ = [
    "KnowledgeGapDetector",
    "KnowledgeGap",
    "StaleKnowledge",
    "Contradiction",
    "Recommendation",
    "Priority",
    "RecommendedAction",
    "DebateInsight",
    "FrequentlyAskedGap",
    "DomainCoverageEntry",
]
