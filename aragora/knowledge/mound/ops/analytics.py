"""
Analytics Module for Knowledge Mound.

Provides comprehensive analytics capabilities:
- Coverage metrics by domain and topic
- Usage patterns and trends
- Quality heatmaps over time
- Growth and churn analysis

Phase A2 - Knowledge Analytics Dashboard
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


# =============================================================================
# Coverage Analytics
# =============================================================================


@dataclass
class DomainCoverage:
    """Coverage metrics for a domain."""

    domain: str
    total_items: int
    high_confidence_items: int  # confidence > 0.7
    medium_confidence_items: int  # 0.4 < confidence <= 0.7
    low_confidence_items: int  # confidence <= 0.4
    average_confidence: float
    average_age_days: float
    stale_items: int  # Items older than threshold
    topics: List[str]  # Main topics in this domain

    @property
    def coverage_score(self) -> float:
        """Calculate overall coverage score (0-1)."""
        if self.total_items == 0:
            return 0.0

        # Weight factors
        confidence_factor = self.average_confidence
        freshness_factor = max(0, 1 - (self.average_age_days / 365))  # Decay over year
        depth_factor = min(1.0, self.total_items / 100)  # Cap at 100 items

        return confidence_factor * 0.4 + freshness_factor * 0.3 + depth_factor * 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "total_items": self.total_items,
            "high_confidence_items": self.high_confidence_items,
            "medium_confidence_items": self.medium_confidence_items,
            "low_confidence_items": self.low_confidence_items,
            "average_confidence": round(self.average_confidence, 3),
            "average_age_days": round(self.average_age_days, 1),
            "stale_items": self.stale_items,
            "topics": self.topics[:10],  # Limit to top 10
            "coverage_score": round(self.coverage_score, 3),
        }


@dataclass
class CoverageReport:
    """Full coverage analysis report."""

    workspace_id: str
    total_items: int
    domains: List[DomainCoverage]
    well_covered_domains: List[str]  # score > 0.7
    sparse_domains: List[str]  # score < 0.3
    overall_coverage_score: float
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "total_items": self.total_items,
            "domains": [d.to_dict() for d in self.domains],
            "well_covered_domains": self.well_covered_domains,
            "sparse_domains": self.sparse_domains,
            "overall_coverage_score": round(self.overall_coverage_score, 3),
            "analyzed_at": self.analyzed_at.isoformat(),
        }


# =============================================================================
# Usage Analytics
# =============================================================================


class UsageEventType(str, Enum):
    """Types of usage events tracked."""

    QUERY = "query"
    VIEW = "view"
    CITE = "cite"
    SHARE = "share"
    EXPORT = "export"


@dataclass
class UsageEvent:
    """A usage event record."""

    id: str
    event_type: UsageEventType
    item_id: Optional[str] = None
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    query: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemUsageStats:
    """Usage statistics for a single item."""

    item_id: str
    view_count: int = 0
    query_hits: int = 0
    citation_count: int = 0
    share_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score."""
        return (
            self.view_count * 1.0
            + self.query_hits * 2.0
            + self.citation_count * 5.0
            + self.share_count * 3.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "view_count": self.view_count,
            "query_hits": self.query_hits,
            "citation_count": self.citation_count,
            "share_count": self.share_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "engagement_score": round(self.engagement_score, 2),
        }


@dataclass
class UsageReport:
    """Usage analytics report."""

    workspace_id: str
    period_days: int
    total_queries: int
    total_views: int
    unique_users: int
    most_accessed_items: List[ItemUsageStats]
    least_accessed_items: List[ItemUsageStats]
    query_patterns: Dict[str, int]  # Common query terms
    daily_activity: Dict[str, int]  # date -> count
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "period_days": self.period_days,
            "total_queries": self.total_queries,
            "total_views": self.total_views,
            "unique_users": self.unique_users,
            "most_accessed_items": [i.to_dict() for i in self.most_accessed_items],
            "least_accessed_items": [i.to_dict() for i in self.least_accessed_items],
            "query_patterns": dict(sorted(self.query_patterns.items(), key=lambda x: -x[1])[:20]),
            "daily_activity": self.daily_activity,
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# Quality Analytics
# =============================================================================


@dataclass
class QualitySnapshot:
    """Quality metrics at a point in time."""

    timestamp: datetime
    total_items: int
    average_confidence: float
    stale_percentage: float
    contradiction_count: int
    high_quality_count: int  # confidence > 0.8, recently updated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_items": self.total_items,
            "average_confidence": round(self.average_confidence, 3),
            "stale_percentage": round(self.stale_percentage, 3),
            "contradiction_count": self.contradiction_count,
            "high_quality_count": self.high_quality_count,
        }


@dataclass
class QualityTrend:
    """Quality trend over time."""

    workspace_id: str
    snapshots: List[QualitySnapshot]
    trend_direction: str  # "improving", "stable", "declining"
    period_days: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "trend_direction": self.trend_direction,
            "period_days": self.period_days,
        }


# =============================================================================
# Growth Analytics
# =============================================================================


@dataclass
class GrowthMetrics:
    """Growth and churn metrics."""

    workspace_id: str
    period_days: int
    items_added: int
    items_deleted: int
    items_updated: int
    net_growth: int
    growth_rate: float  # percentage
    churn_rate: float  # deleted / (start_count + added)
    velocity: float  # items per day

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "period_days": self.period_days,
            "items_added": self.items_added,
            "items_deleted": self.items_deleted,
            "items_updated": self.items_updated,
            "net_growth": self.net_growth,
            "growth_rate": round(self.growth_rate, 3),
            "churn_rate": round(self.churn_rate, 3),
            "velocity": round(self.velocity, 2),
        }


# =============================================================================
# Analytics Manager
# =============================================================================


class KnowledgeAnalytics:
    """Manages analytics for Knowledge Mound."""

    def __init__(self):
        """Initialize analytics manager."""
        self._usage_events: List[UsageEvent] = []
        self._item_usage: Dict[str, ItemUsageStats] = defaultdict(
            lambda: ItemUsageStats(item_id="")
        )
        self._quality_snapshots: Dict[str, List[QualitySnapshot]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_usage(
        self,
        event_type: UsageEventType,
        item_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageEvent:
        """Record a usage event.

        Args:
            event_type: Type of event
            item_id: Item involved
            user_id: User who triggered
            workspace_id: Workspace context
            query: Query string if applicable
            metadata: Additional metadata

        Returns:
            Created UsageEvent
        """
        import uuid

        event = UsageEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            item_id=item_id,
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
            metadata=metadata or {},
        )

        async with self._lock:
            self._usage_events.append(event)

            # Update item stats
            if item_id:
                stats = self._item_usage[item_id]
                stats.item_id = item_id
                stats.last_accessed = datetime.now()

                if event_type == UsageEventType.VIEW:
                    stats.view_count += 1
                elif event_type == UsageEventType.QUERY:
                    stats.query_hits += 1
                elif event_type == UsageEventType.CITE:
                    stats.citation_count += 1
                elif event_type == UsageEventType.SHARE:
                    stats.share_count += 1

            # Trim old events
            if len(self._usage_events) > 100000:
                self._usage_events = self._usage_events[-100000:]

        return event

    async def analyze_coverage(
        self,
        mound: "KnowledgeMound",
        workspace_id: str,
        stale_threshold_days: int = 90,
    ) -> CoverageReport:
        """Analyze domain coverage.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to analyze
            stale_threshold_days: Days before item is considered stale

        Returns:
            CoverageReport
        """
        # Get all items
        result = await mound.query(
            workspace_id=workspace_id,
            query="",
            limit=10000,
        )
        items = result.items if hasattr(result, "items") else []

        # Group by domain (first topic)
        domain_items: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            topics = getattr(item, "topics", []) or []
            domain = topics[0] if topics else "uncategorized"
            domain_items[domain].append(item)

        now = datetime.now()
        stale_threshold = now - timedelta(days=stale_threshold_days)

        domain_coverages = []
        for domain, items_in_domain in domain_items.items():
            high_conf = 0
            medium_conf = 0
            low_conf = 0
            total_conf = 0.0
            total_age = 0.0
            stale = 0
            topic_counts: Dict[str, int] = defaultdict(int)

            for item in items_in_domain:
                conf = getattr(item, "confidence", 0.5) or 0.5

                if conf > 0.7:
                    high_conf += 1
                elif conf > 0.4:
                    medium_conf += 1
                else:
                    low_conf += 1

                total_conf += conf

                # Age
                created_at = getattr(item, "created_at", None)
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except ValueError:
                        created_at = now
                elif created_at is None:
                    created_at = now

                age_days = (now - created_at).total_seconds() / 86400
                total_age += age_days

                if created_at < stale_threshold:
                    stale += 1

                # Topics
                for topic in getattr(item, "topics", []) or []:
                    topic_counts[topic] += 1

            count = len(items_in_domain)
            top_topics = sorted(topic_counts.keys(), key=lambda t: -topic_counts[t])

            domain_coverages.append(
                DomainCoverage(
                    domain=domain,
                    total_items=count,
                    high_confidence_items=high_conf,
                    medium_confidence_items=medium_conf,
                    low_confidence_items=low_conf,
                    average_confidence=total_conf / count if count else 0.0,
                    average_age_days=total_age / count if count else 0.0,
                    stale_items=stale,
                    topics=top_topics,
                )
            )

        # Sort by coverage score
        domain_coverages.sort(key=lambda d: -d.coverage_score)

        well_covered = [d.domain for d in domain_coverages if d.coverage_score > 0.7]
        sparse = [d.domain for d in domain_coverages if d.coverage_score < 0.3]

        overall_score = (
            sum(d.coverage_score for d in domain_coverages) / len(domain_coverages)
            if domain_coverages
            else 0.0
        )

        return CoverageReport(
            workspace_id=workspace_id,
            total_items=len(items),
            domains=domain_coverages,
            well_covered_domains=well_covered,
            sparse_domains=sparse,
            overall_coverage_score=overall_score,
        )

    async def analyze_usage(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> UsageReport:
        """Analyze usage patterns.

        Args:
            workspace_id: Workspace to analyze
            days: Number of days to look back

        Returns:
            UsageReport
        """
        start_time = datetime.now() - timedelta(days=days)

        async with self._lock:
            # Filter events
            events = [
                e
                for e in self._usage_events
                if e.timestamp >= start_time
                and (e.workspace_id == workspace_id or workspace_id is None)
            ]

        total_queries = sum(1 for e in events if e.event_type == UsageEventType.QUERY)
        total_views = sum(1 for e in events if e.event_type == UsageEventType.VIEW)
        unique_users = len({e.user_id for e in events if e.user_id})

        # Query patterns
        query_terms: Dict[str, int] = defaultdict(int)
        for event in events:
            if event.query:
                for term in event.query.lower().split():
                    if len(term) > 2:
                        query_terms[term] += 1

        # Daily activity
        daily: Dict[str, int] = defaultdict(int)
        for event in events:
            day = event.timestamp.strftime("%Y-%m-%d")
            daily[day] += 1

        # Item stats
        item_stats = list(self._item_usage.values())
        item_stats.sort(key=lambda s: -s.engagement_score)

        most_accessed = item_stats[:10]
        least_accessed = [s for s in item_stats if s.engagement_score > 0][-10:]

        return UsageReport(
            workspace_id=workspace_id,
            period_days=days,
            total_queries=total_queries,
            total_views=total_views,
            unique_users=unique_users,
            most_accessed_items=most_accessed,
            least_accessed_items=least_accessed,
            query_patterns=dict(query_terms),
            daily_activity=dict(daily),
        )

    async def capture_quality_snapshot(
        self,
        mound: "KnowledgeMound",
        workspace_id: str,
    ) -> QualitySnapshot:
        """Capture current quality metrics.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to snapshot

        Returns:
            QualitySnapshot
        """
        result = await mound.query(
            workspace_id=workspace_id,
            query="",
            limit=10000,
        )
        items = result.items if hasattr(result, "items") else []

        now = datetime.now()
        stale_threshold = now - timedelta(days=90)

        total_conf = 0.0
        stale_count = 0
        high_quality = 0

        for item in items:
            conf = getattr(item, "confidence", 0.5) or 0.5
            total_conf += conf

            created_at = getattr(item, "created_at", None)
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = now
            elif created_at is None:
                created_at = now

            if created_at < stale_threshold:
                stale_count += 1
            elif conf > 0.8:
                high_quality += 1

        count = len(items)
        snapshot = QualitySnapshot(
            timestamp=now,
            total_items=count,
            average_confidence=total_conf / count if count else 0.0,
            stale_percentage=stale_count / count if count else 0.0,
            contradiction_count=0,  # Would need contradiction detector
            high_quality_count=high_quality,
        )

        async with self._lock:
            self._quality_snapshots[workspace_id].append(snapshot)
            # Keep last 365 snapshots per workspace
            if len(self._quality_snapshots[workspace_id]) > 365:
                self._quality_snapshots[workspace_id] = self._quality_snapshots[workspace_id][-365:]

        return snapshot

    async def get_quality_trend(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> QualityTrend:
        """Get quality trend over time.

        Args:
            workspace_id: Workspace to analyze
            days: Number of days to look back

        Returns:
            QualityTrend
        """
        start_time = datetime.now() - timedelta(days=days)

        async with self._lock:
            snapshots = [
                s
                for s in self._quality_snapshots.get(workspace_id, [])
                if s.timestamp >= start_time
            ]

        # Determine trend
        if len(snapshots) < 2:
            trend = "stable"
        else:
            first_avg = sum(s.average_confidence for s in snapshots[: len(snapshots) // 2]) / (
                len(snapshots) // 2
            )
            second_avg = sum(s.average_confidence for s in snapshots[len(snapshots) // 2 :]) / (
                len(snapshots) - len(snapshots) // 2
            )

            if second_avg > first_avg + 0.05:
                trend = "improving"
            elif second_avg < first_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"

        return QualityTrend(
            workspace_id=workspace_id,
            snapshots=snapshots,
            trend_direction=trend,
            period_days=days,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return {
            "total_usage_events": len(self._usage_events),
            "tracked_items": len(self._item_usage),
            "workspaces_with_snapshots": len(self._quality_snapshots),
        }


# =============================================================================
# Analytics Mixin
# =============================================================================


class AnalyticsMixin:
    """Mixin for analytics operations on KnowledgeMound."""

    _analytics: Optional[KnowledgeAnalytics] = None

    def _get_analytics(self) -> KnowledgeAnalytics:
        """Get or create analytics manager."""
        if self._analytics is None:
            self._analytics = KnowledgeAnalytics()
        return self._analytics

    async def record_usage_event(
        self,
        event_type: UsageEventType,
        item_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> UsageEvent:
        """Record a usage event."""
        return await self._get_analytics().record_usage(
            event_type, item_id, user_id, workspace_id, query
        )

    async def analyze_coverage(
        self,
        workspace_id: str,
        stale_threshold_days: int = 90,
    ) -> CoverageReport:
        """Analyze domain coverage."""
        return await self._get_analytics().analyze_coverage(
            self, workspace_id, stale_threshold_days
        )

    async def analyze_usage(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> UsageReport:
        """Analyze usage patterns."""
        return await self._get_analytics().analyze_usage(workspace_id, days)

    async def capture_quality_snapshot(
        self,
        workspace_id: str,
    ) -> QualitySnapshot:
        """Capture current quality metrics."""
        return await self._get_analytics().capture_quality_snapshot(self, workspace_id)

    async def get_quality_trend(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> QualityTrend:
        """Get quality trend over time."""
        return await self._get_analytics().get_quality_trend(workspace_id, days)

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return self._get_analytics().get_stats()


# Singleton instance
_analytics: Optional[KnowledgeAnalytics] = None


def get_knowledge_analytics() -> KnowledgeAnalytics:
    """Get the global knowledge analytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = KnowledgeAnalytics()
    return _analytics
