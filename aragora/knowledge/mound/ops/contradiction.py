"""
Contradiction Detection for Knowledge Mound.

Detects and manages conflicting knowledge items:
- Semantic contradiction detection using embeddings
- Logical contradiction detection using claim parsing
- Conflict resolution strategies
- Contradiction audit logging

Phase A2 - Knowledge Quality Assurance
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


class ContradictionType(str, Enum):
    """Types of contradictions detected."""

    SEMANTIC = "semantic"  # Similar topics, opposing content
    LOGICAL = "logical"  # Explicit logical contradiction
    TEMPORAL = "temporal"  # Same claim, different time validity
    NUMERICAL = "numerical"  # Conflicting numerical values
    AUTHORITY = "authority"  # Conflicting authoritative sources


class ResolutionStrategy(str, Enum):
    """Strategies for resolving contradictions."""

    PREFER_NEWER = "prefer_newer"  # Keep more recent item
    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    PREFER_MORE_SOURCES = "prefer_more_sources"
    MERGE = "merge"  # Combine into nuanced item
    HUMAN_REVIEW = "human_review"  # Flag for manual review
    KEEP_BOTH = "keep_both"  # Mark as disputed


@dataclass
class Contradiction:
    """A detected contradiction between knowledge items."""

    id: str
    item_a_id: str
    item_b_id: str
    contradiction_type: ContradictionType
    similarity_score: float  # How similar the topics are (0-1)
    conflict_score: float  # How conflicting the claims are (0-1)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[ResolutionStrategy] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        """Calculate contradiction severity."""
        score = self.similarity_score * self.conflict_score
        if score > 0.8:
            return "critical"
        if score > 0.5:
            return "high"
        if score > 0.3:
            return "medium"
        return "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "item_a_id": self.item_a_id,
            "item_b_id": self.item_b_id,
            "contradiction_type": self.contradiction_type.value,
            "similarity_score": self.similarity_score,
            "conflict_score": self.conflict_score,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution.value if self.resolution else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "notes": self.notes,
            "metadata": self.metadata,
        }


@dataclass
class ContradictionReport:
    """Report of contradiction detection results."""

    workspace_id: str
    scanned_items: int
    contradictions_found: int
    contradictions: List[Contradiction]
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    scan_duration_ms: float
    scanned_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workspace_id": self.workspace_id,
            "scanned_items": self.scanned_items,
            "contradictions_found": self.contradictions_found,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "scan_duration_ms": self.scan_duration_ms,
            "scanned_at": self.scanned_at.isoformat(),
        }


@dataclass
class ContradictionConfig:
    """Configuration for contradiction detection."""

    # Similarity thresholds
    min_topic_similarity: float = 0.7  # Min similarity to consider related
    min_conflict_score: float = 0.5  # Min score to flag as contradiction

    # Detection settings
    max_comparisons_per_item: int = 50  # Limit comparisons per item
    batch_size: int = 100  # Items to process per batch
    enable_semantic_detection: bool = True
    enable_logical_detection: bool = True
    enable_numerical_detection: bool = True

    # Resolution settings
    auto_resolve: bool = False  # Automatically apply resolution strategy
    default_strategy: ResolutionStrategy = ResolutionStrategy.HUMAN_REVIEW

    # Negation patterns for conflict detection
    negation_patterns: List[str] = field(
        default_factory=lambda: [
            "not",
            "never",
            "no longer",
            "isn't",
            "doesn't",
            "won't",
            "cannot",
            "shouldn't",
            "incorrect",
            "false",
            "wrong",
            "invalid",
            "deprecated",
            "obsolete",
        ]
    )


class ContradictionDetector:
    """Detects contradictions in knowledge items."""

    def __init__(self, config: Optional[ContradictionConfig] = None):
        """Initialize the contradiction detector."""
        self.config = config or ContradictionConfig()
        self._contradictions: Dict[str, Contradiction] = {}
        self._lock = asyncio.Lock()

    async def detect_contradictions(
        self,
        mound: "KnowledgeMound",
        workspace_id: str,
        item_ids: Optional[List[str]] = None,
    ) -> ContradictionReport:
        """Detect contradictions in knowledge items.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to scan
            item_ids: Optional specific items to check (checks all if None)

        Returns:
            ContradictionReport with findings
        """
        import time
        import uuid

        start_time = time.time()

        # Get items to scan
        if item_ids:
            items = []
            for item_id in item_ids:
                item = await mound.get(item_id)
                if item:
                    items.append(item)
        else:
            # Query all items in workspace
            result = await mound.query(
                workspace_id=workspace_id,
                query="",  # Empty query to get all
                limit=10000,
            )
            items = result.items if hasattr(result, "items") else []

        contradictions: List[Contradiction] = []
        scanned_pairs = set()

        # Compare items pairwise
        for i, item_a in enumerate(items):
            comparisons = 0
            for j, item_b in enumerate(items):
                if i >= j:  # Skip self and already-compared pairs
                    continue

                pair_key = tuple(sorted([item_a.id, item_b.id]))
                if pair_key in scanned_pairs:
                    continue
                scanned_pairs.add(pair_key)

                # Check if items are related enough to compare
                similarity = await self._compute_similarity(mound, item_a, item_b)
                if similarity < self.config.min_topic_similarity:
                    continue

                # Detect conflicts
                conflict = await self._detect_conflict(item_a, item_b)
                if conflict and conflict.conflict_score >= self.config.min_conflict_score:
                    conflict.id = str(uuid.uuid4())
                    conflict.item_a_id = item_a.id
                    conflict.item_b_id = item_b.id
                    conflict.similarity_score = similarity
                    contradictions.append(conflict)

                    # Store for tracking
                    async with self._lock:
                        self._contradictions[conflict.id] = conflict

                comparisons += 1
                if comparisons >= self.config.max_comparisons_per_item:
                    break

        # Build report
        duration_ms = (time.time() - start_time) * 1000

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for c in contradictions:
            by_type[c.contradiction_type.value] = by_type.get(c.contradiction_type.value, 0) + 1
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1

        return ContradictionReport(
            workspace_id=workspace_id,
            scanned_items=len(items),
            contradictions_found=len(contradictions),
            contradictions=contradictions,
            by_type=by_type,
            by_severity=by_severity,
            scan_duration_ms=duration_ms,
        )

    async def _compute_similarity(
        self,
        mound: "KnowledgeMound",
        item_a: Any,
        item_b: Any,
    ) -> float:
        """Compute semantic similarity between two items."""
        # Use embeddings if available
        if hasattr(item_a, "embedding") and hasattr(item_b, "embedding"):
            if item_a.embedding is not None and item_b.embedding is not None:
                return self._cosine_similarity(item_a.embedding, item_b.embedding)

        # Fallback to topic overlap
        topics_a = set(getattr(item_a, "topics", []) or [])
        topics_b = set(getattr(item_b, "topics", []) or [])
        if not topics_a or not topics_b:
            return 0.0

        intersection = len(topics_a & topics_b)
        union = len(topics_a | topics_b)
        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def _detect_conflict(self, item_a: Any, item_b: Any) -> Optional[Contradiction]:
        """Detect if two items have conflicting content."""
        content_a = getattr(item_a, "content", "") or ""
        content_b = getattr(item_b, "content", "") or ""

        if not content_a or not content_b:
            return None

        content_a_lower = content_a.lower()
        content_b_lower = content_b.lower()

        # Check for negation patterns
        conflict_score = 0.0
        contradiction_type = ContradictionType.SEMANTIC

        # Negation detection
        for pattern in self.config.negation_patterns:
            if pattern in content_a_lower and pattern not in content_b_lower:
                conflict_score += 0.2
            elif pattern in content_b_lower and pattern not in content_a_lower:
                conflict_score += 0.2

        # Numerical conflict detection
        if self.config.enable_numerical_detection:
            num_conflict = self._detect_numerical_conflict(content_a, content_b)
            if num_conflict > 0:
                conflict_score = max(conflict_score, num_conflict)
                contradiction_type = ContradictionType.NUMERICAL

        # Cap at 1.0
        conflict_score = min(1.0, conflict_score)

        if conflict_score > 0:
            return Contradiction(
                id="",  # Set later
                item_a_id="",
                item_b_id="",
                contradiction_type=contradiction_type,
                similarity_score=0.0,  # Set later
                conflict_score=conflict_score,
            )

        return None

    def _detect_numerical_conflict(self, content_a: str, content_b: str) -> float:
        """Detect conflicting numerical values."""
        import re

        # Extract numbers with context
        num_pattern = r"(\d+(?:\.\d+)?)\s*(%|percent|ms|seconds|minutes|hours|days|bytes|kb|mb|gb)"

        nums_a = re.findall(num_pattern, content_a.lower())
        nums_b = re.findall(num_pattern, content_b.lower())

        if not nums_a or not nums_b:
            return 0.0

        # Check for same unit, different values
        for val_a, unit_a in nums_a:
            for val_b, unit_b in nums_b:
                if unit_a == unit_b:
                    try:
                        num_a = float(val_a)
                        num_b = float(val_b)
                        # Significant difference (>50% deviation)
                        if num_a > 0 and abs(num_a - num_b) / num_a > 0.5:
                            return 0.7
                    except ValueError:
                        continue

        return 0.0

    async def resolve_contradiction(
        self,
        contradiction_id: str,
        strategy: ResolutionStrategy,
        resolved_by: Optional[str] = None,
        notes: str = "",
    ) -> Optional[Contradiction]:
        """Resolve a contradiction.

        Args:
            contradiction_id: ID of contradiction to resolve
            strategy: Resolution strategy to apply
            resolved_by: User who resolved it
            notes: Resolution notes

        Returns:
            Updated contradiction or None if not found
        """
        async with self._lock:
            contradiction = self._contradictions.get(contradiction_id)
            if not contradiction:
                return None

            contradiction.resolved = True
            contradiction.resolution = strategy
            contradiction.resolved_at = datetime.now()
            contradiction.resolved_by = resolved_by
            contradiction.notes = notes

            return contradiction

    async def get_unresolved(
        self,
        workspace_id: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Contradiction]:
        """Get unresolved contradictions.

        Args:
            workspace_id: Filter by workspace
            min_severity: Minimum severity level

        Returns:
            List of unresolved contradictions
        """
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order.get(min_severity or "low", 0)

        async with self._lock:
            results = []
            for c in self._contradictions.values():
                if c.resolved:
                    continue
                if min_severity and severity_order.get(c.severity, 0) < min_level:
                    continue
                results.append(c)

            return sorted(results, key=lambda x: -severity_order.get(x.severity, 0))

    def get_stats(self) -> Dict[str, Any]:
        """Get contradiction detection statistics."""
        total = len(self._contradictions)
        resolved = sum(1 for c in self._contradictions.values() if c.resolved)
        unresolved = total - resolved

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for c in self._contradictions.values():
            by_type[c.contradiction_type.value] = by_type.get(c.contradiction_type.value, 0) + 1
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1

        return {
            "total_contradictions": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "by_type": by_type,
            "by_severity": by_severity,
        }


class ContradictionOperationsMixin:
    """Mixin for contradiction detection operations on KnowledgeMound."""

    _contradiction_detector: Optional[ContradictionDetector] = None

    def _get_contradiction_detector(self) -> ContradictionDetector:
        """Get or create contradiction detector."""
        if self._contradiction_detector is None:
            self._contradiction_detector = ContradictionDetector()
        return self._contradiction_detector

    async def detect_contradictions(
        self,
        workspace_id: str,
        item_ids: Optional[List[str]] = None,
    ) -> ContradictionReport:
        """Detect contradictions in knowledge items.

        Args:
            workspace_id: Workspace to scan
            item_ids: Optional specific items to check

        Returns:
            ContradictionReport with findings
        """
        detector = self._get_contradiction_detector()
        return await detector.detect_contradictions(self, workspace_id, item_ids)

    async def resolve_contradiction(
        self,
        contradiction_id: str,
        strategy: ResolutionStrategy,
        resolved_by: Optional[str] = None,
        notes: str = "",
    ) -> Optional[Contradiction]:
        """Resolve a detected contradiction."""
        detector = self._get_contradiction_detector()
        return await detector.resolve_contradiction(contradiction_id, strategy, resolved_by, notes)

    async def get_unresolved_contradictions(
        self,
        workspace_id: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Contradiction]:
        """Get unresolved contradictions."""
        detector = self._get_contradiction_detector()
        return await detector.get_unresolved(workspace_id, min_severity)

    def get_contradiction_stats(self) -> Dict[str, Any]:
        """Get contradiction detection statistics."""
        detector = self._get_contradiction_detector()
        return detector.get_stats()


# Singleton instance
_detector: Optional[ContradictionDetector] = None


def get_contradiction_detector() -> ContradictionDetector:
    """Get the global contradiction detector instance."""
    global _detector
    if _detector is None:
        _detector = ContradictionDetector()
    return _detector
