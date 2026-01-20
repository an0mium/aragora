"""
PulseAdapter - Bridges the Pulse system to the Knowledge Mound.

This adapter enables bidirectional integration between the Pulse
(trending topics and scheduled debates) system and the Knowledge Mound:

- Data flow IN: Trending topics, scheduled debates, and outcomes stored in KM
- Data flow OUT: Past debates on topic retrieved for deduplication
- Reverse flow: KM patterns inform trend detection thresholds

The adapter provides:
- Trending topic storage with quality filtering
- Scheduled debate record persistence
- Debate outcome tracking for analytics
- Topic deduplication via historical lookup

ID Prefix: pl_
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.pulse.ingestor import TrendingTopic, TrendingTopicOutcome
    from aragora.pulse.store import ScheduledDebateRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Reverse Flow Dataclasses (KM → Pulse)
# =============================================================================


@dataclass
class KMQualityThresholdUpdate:
    """Result of updating quality thresholds from KM patterns."""

    old_min_quality: float
    new_min_quality: float
    old_category_bonuses: Dict[str, float] = field(default_factory=dict)
    new_category_bonuses: Dict[str, float] = field(default_factory=dict)
    patterns_analyzed: int = 0
    adjustments_made: int = 0
    confidence: float = 0.7
    recommendation: str = "keep"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMTopicCoverage:
    """Analysis of KM coverage for a potential debate topic."""

    topic_text: str
    coverage_score: float = 0.0
    related_debates_count: int = 0
    avg_outcome_confidence: float = 0.0
    consensus_rate: float = 0.0
    km_items_found: int = 0
    recommendation: str = "proceed"  # proceed, skip, defer
    priority_adjustment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMSchedulingRecommendation:
    """Recommendation for scheduling adjustments from KM."""

    topic_id: str
    original_priority: float = 0.5
    adjusted_priority: float = 0.5
    reason: str = "no_change"
    km_confidence: float = 0.7
    coverage: Optional[KMTopicCoverage] = None
    was_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMTopicValidation:
    """Validation result for a topic from KM patterns."""

    topic_id: str
    km_confidence: float = 0.7
    outcome_success_rate: float = 0.0
    similar_debates_count: int = 0
    avg_rounds_needed: float = 0.0
    recommendation: str = "keep"  # keep, boost, demote, skip
    priority_adjustment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PulseKMSyncResult:
    """Result of batch sync from KM validations."""

    topics_analyzed: int = 0
    topics_adjusted: int = 0
    threshold_updates: int = 0
    scheduling_changes: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicSearchResult:
    """Wrapper for topic search results with adapter metadata."""

    topic: Dict[str, Any]
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        pass


class PulseAdapter:
    """
    Adapter that bridges Pulse system to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_trending_topic: Store quality trending topics
    - store_scheduled_debate: Store scheduled debate records
    - store_outcome: Store debate outcome for analytics
    - search_past_debates: Find historical debates on topic
    - get_trending_patterns: Find recurring trending patterns

    Usage:
        from aragora.pulse.store import ScheduledDebateStore
        from aragora.knowledge.mound.adapters import PulseAdapter

        store = ScheduledDebateStore()
        adapter = PulseAdapter(store)

        # Store quality trending topic
        adapter.store_trending_topic(topic)

        # Check for duplicates before scheduling
        past = adapter.search_past_debates(topic.topic, hours=48)
    """

    ID_PREFIX = "pl_"

    # Thresholds from plan
    MIN_TOPIC_QUALITY = 0.6  # Only store quality topics (volume/relevance weighted)

    def __init__(
        self,
        debate_store: Optional[Any] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            debate_store: Optional ScheduledDebateStore instance
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._debate_store = debate_store
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._topics: Dict[str, Dict[str, Any]] = {}
        self._debates: Dict[str, Dict[str, Any]] = {}
        self._outcomes: Dict[str, Dict[str, Any]] = {}

        # Indices for fast lookup
        self._platform_topics: Dict[str, List[str]] = {}  # platform -> [topic_ids]
        self._category_topics: Dict[str, List[str]] = {}  # category -> [topic_ids]
        self._topic_hash_map: Dict[str, str] = {}  # topic_hash -> topic_id

    @property
    def debate_store(self) -> Optional[Any]:
        """Access the underlying ScheduledDebateStore."""
        return self._debate_store

    def _calculate_quality_score(
        self,
        volume: int,
        category: str,
    ) -> float:
        """
        Calculate quality score for a trending topic.

        Higher volume and certain categories (tech, science) get higher scores.

        Args:
            volume: Engagement volume
            category: Topic category

        Returns:
            Quality score 0-1
        """
        # Volume scoring: log scale, max 1000000
        import math
        volume_score = min(1.0, math.log10(max(1, volume)) / 6)

        # Category bonus
        category_bonuses = {
            "tech": 0.2,
            "science": 0.2,
            "business": 0.1,
            "politics": 0.0,  # No bonus, but not penalized
            "entertainment": -0.1,
        }
        cat_bonus = category_bonuses.get(category.lower(), 0.0)

        return min(1.0, max(0.0, volume_score + cat_bonus))

    def store_trending_topic(
        self,
        topic: "TrendingTopic",
        min_quality: float = None,
    ) -> Optional[str]:
        """
        Store a trending topic in the Knowledge Mound.

        Args:
            topic: The TrendingTopic to store
            min_quality: Minimum quality threshold

        Returns:
            The topic ID if stored, None if below threshold
        """
        min_q = min_quality or self.MIN_TOPIC_QUALITY
        quality = self._calculate_quality_score(topic.volume, topic.category)

        if quality < min_q:
            logger.debug(f"Topic '{topic.topic[:50]}' below quality threshold: {quality:.2f}")
            return None

        import hashlib
        topic_hash = hashlib.sha256(topic.topic.lower().encode()).hexdigest()[:16]
        topic_id = f"{self.ID_PREFIX}topic_{topic_hash}"

        topic_data = {
            "id": topic_id,
            "topic_hash": topic_hash,
            "topic": topic.topic,
            "platform": topic.platform,
            "volume": topic.volume,
            "category": topic.category,
            "quality_score": quality,
            "raw_data": topic.raw_data,
            "created_at": datetime.utcnow().isoformat(),
        }

        self._topics[topic_id] = topic_data
        self._topic_hash_map[topic_hash] = topic_id

        # Update indices
        if topic.platform not in self._platform_topics:
            self._platform_topics[topic.platform] = []
        self._platform_topics[topic.platform].append(topic_id)

        if topic.category:
            if topic.category not in self._category_topics:
                self._category_topics[topic.category] = []
            self._category_topics[topic.category].append(topic_id)

        logger.info(f"Stored trending topic: {topic_id} (quality={quality:.2f})")
        return topic_id

    def store_scheduled_debate(
        self,
        record: "ScheduledDebateRecord",
    ) -> str:
        """
        Store a scheduled debate record in the Knowledge Mound.

        Args:
            record: The ScheduledDebateRecord to store

        Returns:
            The debate record ID
        """
        debate_id = f"{self.ID_PREFIX}debate_{record.id}"

        debate_data = {
            "id": debate_id,
            "original_id": record.id,
            "topic_hash": record.topic_hash,
            "topic_text": record.topic_text,
            "platform": record.platform,
            "category": record.category,
            "volume": record.volume,
            "debate_id": record.debate_id,
            "created_at": record.created_at,
            "consensus_reached": record.consensus_reached,
            "confidence": record.confidence,
            "rounds_used": record.rounds_used,
            "scheduler_run_id": record.scheduler_run_id,
            "stored_at": datetime.utcnow().isoformat(),
        }

        self._debates[debate_id] = debate_data

        logger.info(f"Stored scheduled debate: {debate_id}")
        return debate_id

    def store_outcome(
        self,
        outcome: "TrendingTopicOutcome",
    ) -> str:
        """
        Store a debate outcome for a trending topic.

        This is always stored (for analytics) regardless of consensus.

        Args:
            outcome: The TrendingTopicOutcome to store

        Returns:
            The outcome ID
        """
        import hashlib
        topic_hash = hashlib.sha256(outcome.topic.lower().encode()).hexdigest()[:16]
        outcome_id = f"{self.ID_PREFIX}outcome_{outcome.debate_id}"

        outcome_data = {
            "id": outcome_id,
            "topic": outcome.topic,
            "topic_hash": topic_hash,
            "platform": outcome.platform,
            "debate_id": outcome.debate_id,
            "consensus_reached": outcome.consensus_reached,
            "confidence": outcome.confidence,
            "rounds_used": outcome.rounds_used,
            "timestamp": outcome.timestamp,
            "category": outcome.category,
            "volume": outcome.volume,
            "stored_at": datetime.utcnow().isoformat(),
        }

        self._outcomes[outcome_id] = outcome_data

        logger.info(f"Stored outcome: {outcome_id} (consensus={outcome.consensus_reached})")
        return outcome_id

    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific topic by ID.

        Args:
            topic_id: The topic ID (may be prefixed with "pl_topic_")

        Returns:
            Topic dict or None
        """
        if not topic_id.startswith(self.ID_PREFIX):
            topic_id = f"{self.ID_PREFIX}topic_{topic_id}"
        return self._topics.get(topic_id)

    def get_debate(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific debate record by ID.

        Args:
            debate_id: The debate ID

        Returns:
            Debate dict or None
        """
        if not debate_id.startswith(self.ID_PREFIX):
            debate_id = f"{self.ID_PREFIX}debate_{debate_id}"
        return self._debates.get(debate_id)

    def search_past_debates(
        self,
        topic_text: str,
        hours: int = 48,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find historical debates on a similar topic.

        This is the key deduplication query for the scheduler.

        Args:
            topic_text: Topic to search for
            hours: Look back this many hours
            limit: Maximum results

        Returns:
            List of debate dicts
        """
        import time
        import hashlib

        topic_hash = hashlib.sha256(topic_text.lower().encode()).hexdigest()[:16]
        cutoff = time.time() - (hours * 3600)

        results = []
        query_words = set(topic_text.lower().split())

        for debate in self._debates.values():
            # Check time window
            created_at = debate.get("created_at", 0)
            if isinstance(created_at, str):
                continue  # Skip if not unix timestamp
            if created_at < cutoff:
                continue

            # Check exact hash match
            if debate.get("topic_hash") == topic_hash:
                results.append({**debate, "match_type": "exact"})
                continue

            # Check keyword overlap
            debate_words = set(debate.get("topic_text", "").lower().split())
            overlap = len(query_words & debate_words)
            if overlap >= 2:  # At least 2 words in common
                results.append({
                    **debate,
                    "match_type": "similar",
                    "overlap_count": overlap,
                })

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", 0), reverse=True)

        return results[:limit]

    def get_platform_topics(
        self,
        platform: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics from a specific platform.

        Args:
            platform: Platform name (twitter, reddit, etc.)
            limit: Maximum results

        Returns:
            List of topic dicts
        """
        topic_ids = self._platform_topics.get(platform, [])
        results = []

        for topic_id in topic_ids[:limit]:
            topic = self._topics.get(topic_id)
            if topic:
                results.append(topic)

        return results

    def get_category_topics(
        self,
        category: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics from a specific category.

        Args:
            category: Category name
            limit: Maximum results

        Returns:
            List of topic dicts
        """
        topic_ids = self._category_topics.get(category, [])
        results = []

        for topic_id in topic_ids[:limit]:
            topic = self._topics.get(topic_id)
            if topic:
                results.append(topic)

        return results

    def get_trending_patterns(
        self,
        min_occurrences: int = 3,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find recurring trending patterns across topics.

        Args:
            min_occurrences: Minimum times a pattern must appear
            limit: Maximum patterns to return

        Returns:
            List of pattern dicts
        """
        # Extract keywords from all topics
        keyword_counts: Dict[str, int] = {}
        keyword_topics: Dict[str, List[str]] = {}

        for topic in self._topics.values():
            words = topic.get("topic", "").lower().split()
            for word in words:
                if len(word) < 3:  # Skip short words
                    continue
                keyword_counts[word] = keyword_counts.get(word, 0) + 1
                if word not in keyword_topics:
                    keyword_topics[word] = []
                keyword_topics[word].append(topic["id"])

        # Filter by min occurrences
        patterns = [
            {
                "keyword": kw,
                "occurrence_count": count,
                "topic_ids": keyword_topics[kw][:10],  # Sample topics
            }
            for kw, count in keyword_counts.items()
            if count >= min_occurrences
        ]

        patterns.sort(key=lambda x: x["occurrence_count"], reverse=True)

        return patterns[:limit]

    def get_outcome_analytics(
        self,
        platform: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get analytics on debate outcomes.

        Args:
            platform: Optional platform filter
            category: Optional category filter

        Returns:
            Analytics dict with success rates, etc.
        """
        outcomes = list(self._outcomes.values())

        if platform:
            outcomes = [o for o in outcomes if o.get("platform") == platform]
        if category:
            outcomes = [o for o in outcomes if o.get("category") == category]

        if not outcomes:
            return {
                "total_debates": 0,
                "consensus_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_rounds": 0.0,
            }

        consensus_count = sum(1 for o in outcomes if o.get("consensus_reached"))
        total_confidence = sum(o.get("confidence", 0) or 0 for o in outcomes)
        total_rounds = sum(o.get("rounds_used", 0) or 0 for o in outcomes)

        return {
            "total_debates": len(outcomes),
            "consensus_rate": consensus_count / len(outcomes),
            "avg_confidence": total_confidence / len(outcomes),
            "avg_rounds": total_rounds / len(outcomes),
            "platform": platform,
            "category": category,
        }

    def to_knowledge_item(self, topic: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a topic dict to a KnowledgeItem.

        Args:
            topic: The topic dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        quality = topic.get("quality_score", 0.5)
        if quality >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif quality >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif quality >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = topic.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        return KnowledgeItem(
            id=topic["id"],
            content=topic.get("topic", ""),
            source=KnowledgeSource.PULSE,
            source_id=topic.get("topic_hash", topic["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "platform": topic.get("platform", ""),
                "category": topic.get("category", ""),
                "volume": topic.get("volume", 0),
                "quality_score": quality,
            },
            importance=quality,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored Pulse data."""
        return {
            "total_topics": len(self._topics),
            "total_debates": len(self._debates),
            "total_outcomes": len(self._outcomes),
            "platforms_tracked": len(self._platform_topics),
            "categories_tracked": len(self._category_topics),
        }


__all__ = [
    "PulseAdapter",
    "TopicSearchResult",
]
