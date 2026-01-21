"""
RlmAdapter - Bridges RLM (Recursive Language Models) to the Knowledge Mound.

This adapter enables bidirectional integration between the RLM compression
system and the Knowledge Mound:

- Data flow IN: Compression patterns stored in KM for optimization
- Data flow OUT: Priority content retrieved for compression decisions
- Reverse flow: KM access patterns inform compression strategies

The adapter provides:
- Compression pattern storage for learning
- Content priority hints from access patterns
- Value score tracking for content
- Compression ratio optimization

ID Prefixes:
- cp_: Compression patterns
- pr_: Priority records
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CompressionPattern:
    """Represents a successful compression pattern."""

    id: str
    compression_ratio: float
    value_score: float
    content_markers: List[str]
    content_type: str = "general"
    usage_count: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentPriority:
    """Represents content priority for compression decisions."""

    content_id: str
    access_count: int
    last_accessed: str
    priority_score: float
    content_type: str = "general"


class RlmAdapter:
    """
    Adapter that bridges RLM Compressor to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_compression_pattern: Store successful compression patterns
    - get_priority_content: Retrieve high-priority content IDs
    - update_access_patterns: Record content access for priority calculation
    - get_compression_hints: Get compression strategy hints

    Usage:
        from aragora.rlm.compressor import RLMCompressor
        from aragora.knowledge.mound.adapters import RlmAdapter

        compressor = RLMCompressor()
        adapter = RlmAdapter(compressor)

        # After successful compression, store pattern
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api", "authentication"],
        )

        # For compression decisions, query priorities
        priorities = adapter.get_priority_content(limit=20)
    """

    PATTERN_PREFIX = "cp_"
    PRIORITY_PREFIX = "pr_"

    # Thresholds
    MIN_VALUE_SCORE = 0.7  # Minimum value score to store pattern
    MIN_ACCESS_COUNT = 3  # Minimum accesses before priority calculation

    def __init__(
        self,
        compressor: Optional[Any] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            compressor: Optional RLM Compressor instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._compressor = compressor
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._priorities: Dict[str, Dict[str, Any]] = {}  # {content_id: priority_data}

        # Statistics
        self._total_compressions = 0
        self._successful_compressions = 0

    @property
    def compressor(self) -> Optional[Any]:
        """Access the underlying RLM Compressor."""
        return self._compressor

    def set_compressor(self, compressor: Any) -> None:
        """Set the RLM compressor to use."""
        self._compressor = compressor

    def store_compression_pattern(
        self,
        compression_ratio: float,
        value_score: float,
        content_markers: List[str],
        content_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a compression pattern in the Knowledge Mound.

        Args:
            compression_ratio: The compression ratio achieved (0-1, lower is better)
            value_score: Value score of retained content (0-1, higher is better)
            content_markers: List of content markers/tags
            content_type: Type of content (e.g., "code", "debate", "analysis")
            metadata: Optional additional metadata

        Returns:
            The pattern ID if stored, None if below threshold
        """
        self._total_compressions += 1

        # Only store high-value patterns
        if value_score < self.MIN_VALUE_SCORE:
            logger.debug(f"Pattern value score too low: {value_score:.2f}")
            return None

        self._successful_compressions += 1

        # Generate pattern ID from markers
        import hashlib

        marker_str = ":".join(sorted(content_markers))
        pattern_hash = hashlib.sha256(marker_str.encode()).hexdigest()[:12]
        pattern_id = f"{self.PATTERN_PREFIX}{pattern_hash}"

        # Check if pattern exists and update usage count
        existing = self._patterns.get(pattern_id)
        usage_count = 1

        if existing:
            usage_count = existing.get("usage_count", 0) + 1
            # Update with weighted average of compression ratios
            compression_ratio = (
                existing.get("compression_ratio", compression_ratio) * 0.7 + compression_ratio * 0.3
            )
            value_score = existing.get("value_score", value_score) * 0.7 + value_score * 0.3

        pattern_data = {
            "id": pattern_id,
            "compression_ratio": compression_ratio,
            "value_score": value_score,
            "content_markers": content_markers,
            "content_type": content_type,
            "usage_count": usage_count,
            "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        self._patterns[pattern_id] = pattern_data

        logger.info(
            f"Stored compression pattern: {pattern_id} "
            f"(ratio={compression_ratio:.2f}, value={value_score:.2f}, usage={usage_count})"
        )
        return pattern_id

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: The pattern ID

        Returns:
            Pattern dict or None
        """
        if not pattern_id.startswith(self.PATTERN_PREFIX):
            pattern_id = f"{self.PATTERN_PREFIX}{pattern_id}"
        return self._patterns.get(pattern_id)

    def get_patterns_for_content(
        self,
        content_markers: List[str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find patterns that match content markers.

        Args:
            content_markers: List of content markers to match
            limit: Maximum patterns to return

        Returns:
            List of matching patterns sorted by value score
        """
        marker_set = set(m.lower() for m in content_markers)
        results = []

        for pattern in self._patterns.values():
            pattern_markers = set(m.lower() for m in pattern.get("content_markers", []))
            overlap = len(marker_set & pattern_markers)

            if overlap > 0:
                relevance = overlap / max(len(marker_set), 1)
                results.append(
                    {
                        **pattern,
                        "relevance": relevance,
                    }
                )

        # Sort by relevance * value_score
        results.sort(
            key=lambda x: x.get("relevance", 0) * x.get("value_score", 0),
            reverse=True,
        )

        return results[:limit]

    def update_access_pattern(
        self,
        content_id: str,
        content_type: str = "general",
    ) -> None:
        """
        Record content access for priority calculation.

        Args:
            content_id: ID of the accessed content
            content_type: Type of content
        """
        if content_id not in self._priorities:
            self._priorities[content_id] = {
                "content_id": content_id,
                "access_count": 0,
                "last_accessed": "",
                "content_type": content_type,
            }

        self._priorities[content_id]["access_count"] += 1
        self._priorities[content_id]["last_accessed"] = datetime.now(timezone.utc).isoformat()

        # Recalculate priority score
        access_count = self._priorities[content_id]["access_count"]
        # Priority decays with time but increases with access count
        self._priorities[content_id]["priority_score"] = min(1.0, access_count / 10)

    def get_priority_content(
        self,
        limit: int = 20,
        content_type: Optional[str] = None,
        min_access_count: int = None,
    ) -> List[ContentPriority]:
        """
        Get high-priority content IDs for compression decisions.

        Args:
            limit: Maximum content items to return
            content_type: Optional content type filter
            min_access_count: Minimum access count threshold

        Returns:
            List of ContentPriority sorted by priority score descending
        """
        min_count = min_access_count or self.MIN_ACCESS_COUNT
        results = []

        for priority_data in self._priorities.values():
            if priority_data.get("access_count", 0) < min_count:
                continue

            if content_type and priority_data.get("content_type") != content_type:
                continue

            results.append(
                ContentPriority(
                    content_id=priority_data["content_id"],
                    access_count=priority_data.get("access_count", 0),
                    last_accessed=priority_data.get("last_accessed", ""),
                    priority_score=priority_data.get("priority_score", 0.0),
                    content_type=priority_data.get("content_type", "general"),
                )
            )

        # Sort by priority score descending
        results.sort(key=lambda x: x.priority_score, reverse=True)

        return results[:limit]

    def get_compression_hints(
        self,
        content_markers: List[str],
    ) -> Dict[str, Any]:
        """
        Get compression strategy hints based on stored patterns.

        Args:
            content_markers: Content markers for the content to compress

        Returns:
            Dict with compression hints including recommended ratio and strategy
        """
        patterns = self.get_patterns_for_content(content_markers, limit=3)

        if not patterns:
            return {
                "recommended_ratio": 0.5,
                "strategy": "default",
                "confidence": 0.0,
                "based_on_patterns": 0,
            }

        # Calculate weighted average of compression ratios
        total_weight = 0
        weighted_ratio = 0

        for pattern in patterns:
            weight = pattern.get("relevance", 0) * pattern.get("usage_count", 1)
            weighted_ratio += pattern.get("compression_ratio", 0.5) * weight
            total_weight += weight

        avg_ratio = weighted_ratio / total_weight if total_weight > 0 else 0.5
        confidence = min(1.0, total_weight / 10)

        # Determine strategy based on patterns
        strategy = (
            "aggressive" if avg_ratio < 0.3 else "balanced" if avg_ratio < 0.6 else "conservative"
        )

        return {
            "recommended_ratio": avg_ratio,
            "strategy": strategy,
            "confidence": confidence,
            "based_on_patterns": len(patterns),
            "top_pattern_id": patterns[0]["id"] if patterns else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns and priorities."""
        pattern_types: Dict[str, int] = {}
        for pattern in self._patterns.values():
            ct = pattern.get("content_type", "general")
            pattern_types[ct] = pattern_types.get(ct, 0) + 1

        return {
            "total_patterns": len(self._patterns),
            "total_priorities": len(self._priorities),
            "total_compressions": self._total_compressions,
            "successful_compressions": self._successful_compressions,
            "success_rate": (
                self._successful_compressions / self._total_compressions
                if self._total_compressions > 0
                else 0.0
            ),
            "pattern_types": pattern_types,
            "avg_pattern_usage": (
                sum(p.get("usage_count", 0) for p in self._patterns.values()) / len(self._patterns)
                if self._patterns
                else 0.0
            ),
        }

    # =========================================================================
    # Knowledge Mound Persistence Methods
    # =========================================================================

    async def sync_to_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Persist compression patterns to the Knowledge Mound.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID for storage

        Returns:
            Dict with sync statistics
        """
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        result: Dict[str, Any] = {
            "patterns_synced": 0,
            "errors": [],
        }

        for pattern_id, pattern_data in self._patterns.items():
            try:
                # Only sync patterns with significant usage
                if pattern_data.get("usage_count", 0) < 2:
                    continue

                markers_str = ", ".join(pattern_data.get("content_markers", []))
                content = (
                    f"Compression Pattern: {pattern_id}\n"
                    f"Content Type: {pattern_data.get('content_type', 'general')}\n"
                    f"Compression Ratio: {pattern_data.get('compression_ratio', 0):.2f}\n"
                    f"Value Score: {pattern_data.get('value_score', 0):.2f}\n"
                    f"Usage Count: {pattern_data.get('usage_count', 0)}\n"
                    f"Content Markers: {markers_str}"
                )

                request = IngestionRequest(
                    content=content,
                    source_type=SourceType.RLM,
                    workspace_id=workspace_id,
                    confidence=pattern_data.get("value_score", 0.5),
                    tier="slow",  # Slow tier for pattern data
                    metadata={
                        "type": "compression_pattern",
                        "pattern_id": pattern_id,
                        "compression_ratio": pattern_data.get("compression_ratio"),
                        "value_score": pattern_data.get("value_score"),
                        "content_type": pattern_data.get("content_type"),
                        "content_markers": pattern_data.get("content_markers"),
                        "usage_count": pattern_data.get("usage_count"),
                    },
                )

                await mound.ingest(request)
                result["patterns_synced"] += 1

            except Exception as e:
                result["errors"].append(f"Pattern {pattern_id}: {e}")

        logger.info(
            f"RLM sync to KM: patterns={result['patterns_synced']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    async def load_from_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Load compression patterns from the Knowledge Mound.

        This restores adapter state from KM persistence.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID to load from

        Returns:
            Dict with load statistics
        """
        result: Dict[str, Any] = {
            "patterns_loaded": 0,
            "errors": [],
        }

        try:
            # Query KM for compression patterns
            nodes = await mound.query_nodes(
                workspace_id=workspace_id,
                source_type="rlm",
                limit=500,
            )

            for node in nodes:
                metadata = node.metadata or {}
                if metadata.get("type") != "compression_pattern":
                    continue

                pattern_id = metadata.get("pattern_id")
                if not pattern_id:
                    continue

                self._patterns[pattern_id] = {
                    "id": pattern_id,
                    "compression_ratio": metadata.get("compression_ratio", 0.5),
                    "value_score": metadata.get("value_score", 0.7),
                    "content_type": metadata.get("content_type", "general"),
                    "content_markers": metadata.get("content_markers", []),
                    "usage_count": metadata.get("usage_count", 1),
                    "created_at": (
                        node.created_at.isoformat()
                        if node.created_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "updated_at": (
                        node.updated_at.isoformat()
                        if node.updated_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "metadata": {},
                }

                result["patterns_loaded"] += 1

        except Exception as e:
            result["errors"].append(f"Load failed: {e}")
            logger.error(f"Failed to load patterns from KM: {e}")

        logger.info(
            f"RLM load from KM: loaded={result['patterns_loaded']}, "
            f"errors={len(result['errors'])}"
        )
        return result


__all__ = [
    "RlmAdapter",
    "CompressionPattern",
    "ContentPriority",
]
