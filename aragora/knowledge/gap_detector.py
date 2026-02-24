"""
Knowledge Gap Detection and Recommendations.

Identifies gaps in the organization's knowledge base and recommends
areas to strengthen. Works with the Knowledge Mound to analyze:
- Coverage gaps: domains/topics with sparse or missing knowledge
- Staleness: outdated entries that need refresh
- Contradictions: conflicting knowledge entries
- Recommendations: prioritized actions to improve knowledge quality

Usage:
    from aragora.knowledge.gap_detector import KnowledgeGapDetector

    detector = KnowledgeGapDetector(mound)
    gaps = await detector.detect_coverage_gaps("legal")
    stale = await detector.detect_staleness(max_age_days=90)
    contradictions = await detector.detect_contradictions()
    recommendations = await detector.get_recommendations()
    score = await detector.get_coverage_score("technical")
"""

from __future__ import annotations

import logging
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
    - Prioritized improvement recommendations

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
                raw_list = await self._mound.get_contradictions(
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
                    priority = Priority.HIGH if gap.severity > 0.7 else (
                        Priority.MEDIUM if gap.severity > 0.4 else Priority.LOW
                    )
                    recommendations.append(
                        Recommendation(
                            priority=priority,
                            action=RecommendedAction.CREATE,
                            description=f"Add knowledge for {gap.topic}: {gap.description}",
                            domain=gap.domain,
                            impact_score=gap.severity,
                            metadata={"gap_type": "coverage", "actual": gap.actual_items,
                                      "expected": gap.expected_items},
                        )
                    )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to get coverage gaps for '%s': %s", d, e)

        # Analyze staleness
        try:
            stale = await self.detect_staleness()
            for entry in stale[:limit]:
                priority = Priority.HIGH if entry.staleness_score > 0.8 else (
                    Priority.MEDIUM if entry.staleness_score > 0.5 else Priority.LOW
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
                priority = Priority.HIGH if c.conflict_score > 0.7 else (
                    Priority.MEDIUM if c.conflict_score > 0.4 else Priority.LOW
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
        score = (
            depth_factor * 0.4
            + confidence_factor * 0.35
            + freshness_factor * 0.25
        )
        return round(min(1.0, max(0.0, score)), 3)


__all__ = [
    "KnowledgeGapDetector",
    "KnowledgeGap",
    "StaleKnowledge",
    "Contradiction",
    "Recommendation",
    "Priority",
    "RecommendedAction",
]
