"""
InsightsAdapter - Bridges Insights/Trickster to the Knowledge Mound.

This adapter enables bidirectional integration between the Insights system
(including Trickster flip detection) and the Knowledge Mound:

- Data flow IN: Insights, FlipEvents, and PatternClusters stored in KM
- Data flow OUT: Similar insights and flip history retrieved for context
- Reverse flow: KM patterns inform flip detection thresholds

The adapter provides:
- Insight storage after debate extraction
- FlipEvent persistence for Trickster tracking
- Pattern cluster aggregation
- Historical flip retrieval for consistency prediction

ID Prefixes:
- in_: Insights
- fl_: Flip events
- pt_: Pattern clusters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.insights.extractor import DebateInsights, Insight, InsightType
    from aragora.insights.flip_detector import FlipEvent, AgentConsistencyScore

logger = logging.getLogger(__name__)


@dataclass
class InsightSearchResult:
    """Wrapper for insight search results with adapter metadata."""

    insight: Dict[str, Any]
    relevance_score: float = 0.0
    matched_topics: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_topics is None:
            self.matched_topics = []


@dataclass
class FlipSearchResult:
    """Wrapper for flip event search results."""

    flip: Dict[str, Any]
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        pass


class InsightsAdapter:
    """
    Adapter that bridges InsightStore and FlipDetector to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_insight: Store individual insights with confidence filtering
    - store_debate_insights: Store all insights from a debate
    - store_flip: Store flip events for Trickster tracking
    - store_pattern: Store recurring pattern clusters
    - search_similar_insights: Find insights on similar topics
    - get_agent_flip_history: Retrieve flip history for an agent

    Usage:
        from aragora.insights.store import InsightStore
        from aragora.knowledge.mound.adapters import InsightsAdapter

        store = InsightStore()
        adapter = InsightsAdapter(store)

        # Store insights from debate
        adapter.store_debate_insights(debate_insights)

        # Store flip events
        for flip in detected_flips:
            adapter.store_flip(flip)
    """

    INSIGHT_PREFIX = "in_"
    FLIP_PREFIX = "fl_"
    PATTERN_PREFIX = "pt_"

    # Thresholds from plan
    MIN_INSIGHT_CONFIDENCE = 0.7  # Only store high-confidence insights
    MIN_PATTERN_OCCURRENCES = 3  # Store patterns with 3+ occurrences

    def __init__(
        self,
        insight_store: Optional[Any] = None,
        flip_detector: Optional[Any] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            insight_store: Optional InsightStore instance
            flip_detector: Optional FlipDetector instance
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._insight_store = insight_store
        self._flip_detector = flip_detector
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._insights: Dict[str, Dict[str, Any]] = {}
        self._flips: Dict[str, Dict[str, Any]] = {}
        self._patterns: Dict[str, Dict[str, Any]] = {}

        # Indices for fast lookup
        self._debate_insights: Dict[str, List[str]] = {}  # debate_id -> [insight_ids]
        self._type_insights: Dict[str, List[str]] = {}  # type -> [insight_ids]
        self._agent_flips: Dict[str, List[str]] = {}  # agent_name -> [flip_ids]
        self._domain_flips: Dict[str, List[str]] = {}  # domain -> [flip_ids]

    @property
    def insight_store(self) -> Optional[Any]:
        """Access the underlying InsightStore."""
        return self._insight_store

    @property
    def flip_detector(self) -> Optional[Any]:
        """Access the underlying FlipDetector."""
        return self._flip_detector

    def store_insight(
        self,
        insight: "Insight",
        min_confidence: float = None,
    ) -> Optional[str]:
        """
        Store an insight in the Knowledge Mound.

        Args:
            insight: The Insight to store
            min_confidence: Minimum confidence threshold (default: MIN_INSIGHT_CONFIDENCE)

        Returns:
            The insight ID if stored, None if below threshold
        """
        min_conf = min_confidence or self.MIN_INSIGHT_CONFIDENCE

        if insight.confidence < min_conf:
            logger.debug(f"Insight {insight.id} below confidence threshold: {insight.confidence:.2f}")
            return None

        insight_id = f"{self.INSIGHT_PREFIX}{insight.id}"

        insight_data = {
            "id": insight_id,
            "original_id": insight.id,
            "type": insight.type.value if hasattr(insight.type, 'value') else str(insight.type),
            "title": insight.title,
            "description": insight.description,
            "confidence": insight.confidence,
            "debate_id": insight.debate_id,
            "agents_involved": insight.agents_involved,
            "evidence": insight.evidence,
            "created_at": insight.created_at,
            "metadata": insight.metadata,
        }

        self._insights[insight_id] = insight_data

        # Update indices
        if insight.debate_id:
            if insight.debate_id not in self._debate_insights:
                self._debate_insights[insight.debate_id] = []
            self._debate_insights[insight.debate_id].append(insight_id)

        insight_type = insight_data["type"]
        if insight_type not in self._type_insights:
            self._type_insights[insight_type] = []
        self._type_insights[insight_type].append(insight_id)

        logger.info(f"Stored insight: {insight_id} (confidence={insight.confidence:.2f})")
        return insight_id

    def store_debate_insights(
        self,
        debate_insights: "DebateInsights",
        min_confidence: float = None,
    ) -> List[str]:
        """
        Store all insights from a debate above the threshold.

        Args:
            debate_insights: The DebateInsights collection
            min_confidence: Minimum confidence threshold

        Returns:
            List of stored insight IDs
        """
        stored_ids = []

        for insight in debate_insights.all_insights():
            insight_id = self.store_insight(insight, min_confidence)
            if insight_id:
                stored_ids.append(insight_id)

        logger.info(f"Stored {len(stored_ids)} insights from debate {debate_insights.debate_id}")
        return stored_ids

    def store_flip(
        self,
        flip: "FlipEvent",
    ) -> str:
        """
        Store a flip event in the Knowledge Mound.

        Flip events are always stored (meta-learning) regardless of threshold.

        Args:
            flip: The FlipEvent to store

        Returns:
            The flip ID
        """
        flip_id = f"{self.FLIP_PREFIX}{flip.id}"

        flip_data = {
            "id": flip_id,
            "original_id": flip.id,
            "agent_name": flip.agent_name,
            "original_claim": flip.original_claim,
            "new_claim": flip.new_claim,
            "original_confidence": flip.original_confidence,
            "new_confidence": flip.new_confidence,
            "original_debate_id": flip.original_debate_id,
            "new_debate_id": flip.new_debate_id,
            "original_position_id": flip.original_position_id,
            "new_position_id": flip.new_position_id,
            "similarity_score": flip.similarity_score,
            "flip_type": flip.flip_type,
            "domain": flip.domain,
            "detected_at": flip.detected_at,
        }

        self._flips[flip_id] = flip_data

        # Update indices
        if flip.agent_name not in self._agent_flips:
            self._agent_flips[flip.agent_name] = []
        self._agent_flips[flip.agent_name].append(flip_id)

        if flip.domain:
            if flip.domain not in self._domain_flips:
                self._domain_flips[flip.domain] = []
            self._domain_flips[flip.domain].append(flip_id)

        logger.info(f"Stored flip event: {flip_id} (type={flip.flip_type})")
        return flip_id

    def store_flips_batch(
        self,
        flips: List["FlipEvent"],
    ) -> List[str]:
        """
        Store multiple flip events.

        Args:
            flips: List of FlipEvents to store

        Returns:
            List of stored flip IDs
        """
        return [self.store_flip(flip) for flip in flips]

    def store_pattern(
        self,
        category: str,
        pattern_text: str,
        occurrence_count: int,
        avg_severity: float = 0.5,
        debate_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Store a pattern cluster in the Knowledge Mound.

        Args:
            category: Pattern category
            pattern_text: The pattern description
            occurrence_count: Number of occurrences
            avg_severity: Average severity (0-1)
            debate_ids: List of debate IDs where pattern appeared

        Returns:
            The pattern ID if stored, None if below threshold
        """
        if occurrence_count < self.MIN_PATTERN_OCCURRENCES:
            logger.debug(f"Pattern '{pattern_text[:50]}' below occurrence threshold: {occurrence_count}")
            return None

        import hashlib
        pattern_hash = hashlib.sha256(f"{category}:{pattern_text}".encode()).hexdigest()[:12]
        pattern_id = f"{self.PATTERN_PREFIX}{pattern_hash}"

        pattern_data = {
            "id": pattern_id,
            "category": category,
            "pattern_text": pattern_text,
            "occurrence_count": occurrence_count,
            "avg_severity": avg_severity,
            "debate_ids": debate_ids or [],
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
        }

        # Update if exists, otherwise create
        if pattern_id in self._patterns:
            existing = self._patterns[pattern_id]
            pattern_data["first_seen"] = existing["first_seen"]
            pattern_data["occurrence_count"] = existing["occurrence_count"] + occurrence_count
            pattern_data["debate_ids"] = list(set(existing["debate_ids"] + (debate_ids or [])))

        self._patterns[pattern_id] = pattern_data

        logger.info(f"Stored pattern: {pattern_id} (occurrences={pattern_data['occurrence_count']})")
        return pattern_id

    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific insight by ID.

        Args:
            insight_id: The insight ID (may be prefixed with "in_")

        Returns:
            Insight dict or None
        """
        if not insight_id.startswith(self.INSIGHT_PREFIX):
            insight_id = f"{self.INSIGHT_PREFIX}{insight_id}"
        return self._insights.get(insight_id)

    def get_flip(self, flip_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific flip event by ID.

        Args:
            flip_id: The flip ID (may be prefixed with "fl_")

        Returns:
            Flip dict or None
        """
        if not flip_id.startswith(self.FLIP_PREFIX):
            flip_id = f"{self.FLIP_PREFIX}{flip_id}"
        return self._flips.get(flip_id)

    def search_similar_insights(
        self,
        query: str,
        limit: int = 10,
        insight_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find insights similar to the query.

        Args:
            query: Search query (keywords from title/description)
            limit: Maximum results
            insight_type: Optional filter by insight type
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching insight dicts
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for insight in self._insights.values():
            if insight["confidence"] < min_confidence:
                continue

            if insight_type and insight["type"] != insight_type:
                continue

            # Combine title and description for matching
            text = f"{insight['title']} {insight['description']}".lower()
            text_words = set(text.split())

            overlap = len(query_words & text_words)
            if overlap > 0:
                relevance = overlap / max(len(query_words), 1)
                results.append({
                    **insight,
                    "relevance_score": relevance,
                })

        results.sort(
            key=lambda x: x["relevance_score"] * x["confidence"],
            reverse=True,
        )

        return results[:limit]

    def get_insights_by_type(
        self,
        insight_type: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get insights of a specific type.

        Args:
            insight_type: The insight type to filter by
            limit: Maximum results

        Returns:
            List of insight dicts
        """
        insight_ids = self._type_insights.get(insight_type, [])
        results = []

        for insight_id in insight_ids[:limit]:
            insight = self._insights.get(insight_id)
            if insight:
                results.append(insight)

        return results

    def get_debate_insights(
        self,
        debate_id: str,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all insights from a specific debate.

        Args:
            debate_id: The debate ID
            min_confidence: Minimum confidence filter

        Returns:
            List of insight dicts
        """
        insight_ids = self._debate_insights.get(debate_id, [])
        results = []

        for insight_id in insight_ids:
            insight = self._insights.get(insight_id)
            if insight and insight.get("confidence", 0) >= min_confidence:
                results.append(insight)

        return results

    def get_agent_flip_history(
        self,
        agent_name: str,
        limit: int = 50,
        flip_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get flip history for a specific agent.

        This is the key query for Trickster consistency tracking.

        Args:
            agent_name: The agent name
            limit: Maximum results
            flip_type: Optional filter by flip type

        Returns:
            List of flip dicts ordered by detected_at (newest first)
        """
        flip_ids = self._agent_flips.get(agent_name, [])
        results = []

        for flip_id in flip_ids:
            flip = self._flips.get(flip_id)
            if flip:
                if flip_type and flip["flip_type"] != flip_type:
                    continue
                results.append(flip)

        # Sort by detected_at descending
        results.sort(key=lambda x: x.get("detected_at", ""), reverse=True)

        return results[:limit]

    def get_domain_flips(
        self,
        domain: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get all flips in a specific domain.

        Args:
            domain: The domain to filter by
            limit: Maximum results

        Returns:
            List of flip dicts
        """
        flip_ids = self._domain_flips.get(domain, [])
        results = []

        for flip_id in flip_ids[:limit]:
            flip = self._flips.get(flip_id)
            if flip:
                results.append(flip)

        return results

    def get_common_patterns(
        self,
        min_occurrences: int = 3,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recurring patterns above occurrence threshold.

        Args:
            min_occurrences: Minimum occurrence count
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of pattern dicts sorted by occurrence count
        """
        results = []

        for pattern in self._patterns.values():
            if pattern["occurrence_count"] < min_occurrences:
                continue
            if category and pattern["category"] != category:
                continue
            results.append(pattern)

        results.sort(key=lambda x: x["occurrence_count"], reverse=True)

        return results[:limit]

    def to_knowledge_item(self, insight: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert an insight dict to a KnowledgeItem.

        Args:
            insight: The insight dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        confidence_val = insight.get("confidence", 0.5)
        if confidence_val >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif confidence_val >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif confidence_val >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_val >= 0.3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = insight.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        return KnowledgeItem(
            id=insight["id"],
            content=f"{insight.get('title', '')}: {insight.get('description', '')}",
            source=KnowledgeSource.INSIGHT,
            source_id=insight.get("original_id", insight["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "type": insight.get("type", ""),
                "debate_id": insight.get("debate_id", ""),
                "agents_involved": insight.get("agents_involved", []),
                "evidence": insight.get("evidence", []),
            },
            importance=confidence_val,
        )

    def flip_to_knowledge_item(self, flip: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a flip dict to a KnowledgeItem.

        Args:
            flip: The flip dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Flip confidence based on similarity score (higher = more confident detection)
        similarity = flip.get("similarity_score", 0.5)
        if similarity >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif similarity >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif similarity >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        detected_at = flip.get("detected_at")
        if isinstance(detected_at, str):
            try:
                detected_at = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
            except ValueError:
                detected_at = datetime.utcnow()
        elif detected_at is None:
            detected_at = datetime.utcnow()

        content = (
            f"Agent {flip.get('agent_name', 'unknown')} flipped from "
            f"'{flip.get('original_claim', '')[:100]}' to "
            f"'{flip.get('new_claim', '')[:100]}'"
        )

        return KnowledgeItem(
            id=flip["id"],
            content=content,
            source=KnowledgeSource.FLIP,
            source_id=flip.get("original_id", flip["id"]),
            confidence=confidence,
            created_at=detected_at,
            updated_at=detected_at,
            metadata={
                "agent_name": flip.get("agent_name", ""),
                "flip_type": flip.get("flip_type", ""),
                "domain": flip.get("domain", ""),
                "original_debate_id": flip.get("original_debate_id", ""),
                "new_debate_id": flip.get("new_debate_id", ""),
                "similarity_score": similarity,
            },
            importance=similarity,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored insights and flips."""
        flip_types = {}
        for flip in self._flips.values():
            ft = flip.get("flip_type", "unknown")
            flip_types[ft] = flip_types.get(ft, 0) + 1

        return {
            "total_insights": len(self._insights),
            "total_flips": len(self._flips),
            "total_patterns": len(self._patterns),
            "debates_with_insights": len(self._debate_insights),
            "agents_with_flips": len(self._agent_flips),
            "domains_with_flips": len(self._domain_flips),
            "insight_types": dict(
                (t, len(ids)) for t, ids in self._type_insights.items()
            ),
            "flip_types": flip_types,
        }


__all__ = [
    "InsightsAdapter",
    "InsightSearchResult",
    "FlipSearchResult",
]
