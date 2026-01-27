"""Shared type definitions for Knowledge Mound adapters.

Provides standardized result types used across all adapters to ensure
consistent interfaces and reduce duplication.

Usage:
    from aragora.knowledge.mound.adapters._types import (
        SyncResult,
        ValidationSyncResult,
        SearchResult,
        ValidationResult,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Generic type for adapter-specific items
T = TypeVar("T")


class SyncResult:
    """Result type for forward sync operations (Source -> Knowledge Mound).

    Attributes:
        records_synced: Number of records successfully synced.
        records_skipped: Number of records skipped (already synced or filtered).
        records_failed: Number of records that failed to sync.
        errors: List of error messages for debugging.
        duration_ms: Total operation duration in milliseconds.
    """

    __slots__ = ("records_synced", "records_skipped", "records_failed", "errors", "duration_ms")

    def __init__(
        self,
        records_synced: int = 0,
        records_skipped: int = 0,
        records_failed: int = 0,
        errors: Optional[List[str]] = None,
        duration_ms: float = 0.0,
    ):
        self.records_synced = records_synced
        self.records_skipped = records_skipped
        self.records_failed = records_failed
        self.errors = errors if errors is not None else []
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "records_synced": self.records_synced,
            "records_skipped": self.records_skipped,
            "records_failed": self.records_failed,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }

    @property
    def total_processed(self) -> int:
        """Total number of records processed."""
        return self.records_synced + self.records_skipped + self.records_failed

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        total = self.total_processed
        if total == 0:
            return 100.0
        return (self.records_synced / total) * 100


class ValidationSyncResult:
    """Result type for reverse validation sync operations (KM -> Source).

    Attributes:
        records_analyzed: Number of records analyzed from KM.
        records_updated: Number of records updated in source system.
        records_skipped: Number of records skipped (low confidence or no change).
        errors: List of error messages for debugging.
        duration_ms: Total operation duration in milliseconds.
    """

    __slots__ = (
        "records_analyzed",
        "records_updated",
        "records_skipped",
        "errors",
        "duration_ms",
    )

    def __init__(
        self,
        records_analyzed: int = 0,
        records_updated: int = 0,
        records_skipped: int = 0,
        errors: Optional[List[str]] = None,
        duration_ms: float = 0.0,
    ):
        self.records_analyzed = records_analyzed
        self.records_updated = records_updated
        self.records_skipped = records_skipped
        self.errors = errors if errors is not None else []
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "records_analyzed": self.records_analyzed,
            "records_updated": self.records_updated,
            "records_skipped": self.records_skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SearchResult(Generic[T]):
    """Generic search result wrapper with relevance metadata.

    Type Parameters:
        T: The type of the underlying item (e.g., ConsensusRecord, MemoryEntry).

    Attributes:
        item: The matched item.
        relevance_score: Relevance score (0.0 to 1.0).
        similarity: Semantic similarity score if applicable.
        matched_fields: List of fields that matched the query.
    """

    item: T
    relevance_score: float = 0.0
    similarity: float = 0.0
    matched_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        item_dict = self.item.to_dict() if hasattr(self.item, "to_dict") else self.item
        return {
            "item": item_dict,
            "relevance_score": self.relevance_score,
            "similarity": self.similarity,
            "matched_fields": self.matched_fields,
        }


@dataclass
class ValidationResult:
    """Result of validating a record against Knowledge Mound patterns.

    Used by reverse flow operations to determine if and how a source
    record should be updated based on KM validation.

    Attributes:
        source_id: ID of the source record being validated.
        confidence: Confidence score of the validation (0.0 to 1.0).
        recommendation: Recommended action (keep, boost, penalize, archive, etc.).
        adjustment: Numeric adjustment value if applicable (e.g., ELO delta).
        reason: Human-readable explanation of the validation result.
        metadata: Additional adapter-specific metadata.
    """

    source_id: str
    confidence: float = 0.0
    recommendation: str = "keep"  # keep, boost, penalize, archive, flag
    adjustment: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_apply(self, min_confidence: float = 0.7) -> bool:
        """Check if this validation should be applied based on confidence threshold.

        Args:
            min_confidence: Minimum confidence to apply (default 0.7).

        Returns:
            True if confidence meets threshold and recommendation is actionable.
        """
        if self.confidence < min_confidence:
            return False
        return self.recommendation != "keep" or self.adjustment != 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "adjustment": self.adjustment,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class BatchSyncResult:
    """Result for batch sync operations across multiple adapters.

    Aggregates results from multiple adapters during a coordinated sync.

    Attributes:
        adapter_results: Mapping of adapter name to its SyncResult.
        total_synced: Total records synced across all adapters.
        total_failed: Total records failed across all adapters.
        duration_ms: Total operation duration.
    """

    adapter_results: Dict[str, SyncResult] = field(default_factory=dict)
    total_synced: int = 0
    total_failed: int = 0
    duration_ms: float = 0.0

    def add_result(self, adapter_name: str, result: SyncResult) -> None:
        """Add an adapter's sync result to the batch.

        Args:
            adapter_name: Name of the adapter.
            result: The adapter's sync result.
        """
        self.adapter_results[adapter_name] = result
        self.total_synced += result.records_synced
        self.total_failed += result.records_failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_results": {
                name: result.to_dict() for name, result in self.adapter_results.items()
            },
            "total_synced": self.total_synced,
            "total_failed": self.total_failed,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "SyncResult",
    "ValidationSyncResult",
    "SearchResult",
    "ValidationResult",
    "BatchSyncResult",
]
