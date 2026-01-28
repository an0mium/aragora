"""Fusion mixin for Knowledge Mound adapters.

Provides unified multi-adapter fusion functionality that can be mixed into
any adapter. This mixin enables adapters to participate in cross-adapter
fusion operations orchestrated by the FusionCoordinator.

Follows the established mixin pattern from:
- ReverseFlowMixin (_reverse_flow_base.py) - Template method for KM → Source sync
- SemanticSearchMixin (_semantic_mixin.py) - Template method for vector search

Usage:
    from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin

    class MyAdapter(FusionMixin, KnowledgeMoundAdapter):
        adapter_name = "my_adapter"

        def _get_fusion_sources(self) -> List[str]:
            return ["consensus", "elo", "belief"]

        def _extract_fusible_data(self, km_item: Dict) -> Optional[Dict]:
            return {
                "confidence": km_item.get("confidence"),
                "source_id": km_item.get("id"),
            }

        def _apply_fusion_result(
            self, record, fusion_result, metadata
        ) -> bool:
            record.fused_confidence = fusion_result.fused_confidence
            return True
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from aragora.knowledge.mound.ops.fusion import (
    FusionStrategy,
    ConflictResolution,
    AdapterValidation,
    FusedValidation,
    get_fusion_coordinator,
)

logger = logging.getLogger(__name__)


class FusionSyncResult(TypedDict):
    """Result type for fusion sync operations."""

    items_analyzed: int
    items_fused: int
    items_skipped: int
    conflicts_detected: int
    conflicts_resolved: int
    errors: List[str]
    duration_ms: float


@dataclass
class FusionState:
    """State tracking for fusion operations."""

    fusions_performed: int = 0
    """Total number of fusion operations performed."""

    conflicts_detected: int = 0
    """Total conflicts detected across all fusions."""

    conflicts_resolved: int = 0
    """Total conflicts successfully resolved."""

    last_fusion_at: Optional[datetime] = None
    """Timestamp of the last fusion operation."""

    source_participation: Dict[str, int] = field(default_factory=dict)
    """Count of participation by source adapter."""

    avg_fusion_confidence: float = 0.0
    """Running average of fused confidence values."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fusions_performed": self.fusions_performed,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "last_fusion_at": (self.last_fusion_at.isoformat() if self.last_fusion_at else None),
            "source_participation": self.source_participation,
            "avg_fusion_confidence": round(self.avg_fusion_confidence, 4),
        }


class FusionMixin:
    """Mixin providing multi-adapter fusion capabilities for adapters.

    Provides:
    - fuse_validations_from_km(): Template method for fusion operations
    - _partition_by_source(): Organize items by source adapter
    - _compute_adapter_weight(): Calculate adapter reliability weights
    - Automatic metrics recording and event emission

    Required from inheriting class:
    - adapter_name: str identifying the adapter for metrics
    - _get_fusion_sources(): List of adapters to fuse data from
    - _extract_fusible_data(): Extract fusible data from KM items
    - _apply_fusion_result(): Apply fusion result to a record
    - _emit_event(): Event emission (from KnowledgeMoundAdapter)
    - _record_metric(): Metrics recording (from KnowledgeMoundAdapter)
    """

    # Expected from KnowledgeMoundAdapter or subclass
    adapter_name: str

    # Fusion-specific state
    _fusion_state: FusionState

    def _init_fusion_state(self) -> None:
        """Initialize fusion state tracking."""
        self._fusion_state = FusionState()

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    def _record_metric(
        self,
        operation: str,
        success: bool,
        latency: float,
        extra_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    @abstractmethod
    def _get_fusion_sources(self) -> List[str]:
        """Return list of source adapter names this adapter can fuse data from.

        Override in subclass to specify which adapters to accept data from.

        Returns:
            List of adapter names (e.g., ["consensus", "elo", "belief"]).
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_fusible_data(
        self,
        km_item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract fusible data from a KM item.

        Override in subclass to extract relevant fields for fusion.

        Args:
            km_item: A Knowledge Mound item with validation data.

        Returns:
            Dictionary of extracted data suitable for fusion,
            or None if item cannot be fused.
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: FusedValidation,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply a fusion result to a source record.

        Override in subclass to apply fused data to your records.

        Args:
            record: The target record to update.
            fusion_result: The fused validation result.
            metadata: Optional additional metadata.

        Returns:
            True if the record was updated, False otherwise.
        """
        raise NotImplementedError

    def _get_record_for_fusion(self, source_id: str) -> Optional[Any]:
        """Get a record by its source ID for fusion.

        Override if your adapter has a custom record lookup.
        Default looks for _get_record_for_validation from ReverseFlowMixin.

        Args:
            source_id: The source system record identifier.

        Returns:
            The record, or None if not found.
        """
        # Try to use ReverseFlowMixin method if available
        if hasattr(self, "_get_record_for_validation"):
            return self._get_record_for_validation(source_id)  # type: ignore
        return None

    def fuse_validations_from_km(
        self,
        km_items: List[Dict[str, Any]],
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        conflict_resolution: ConflictResolution = ConflictResolution.PREFER_HIGHER_CONFIDENCE,
        min_sources: int = 2,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> FusionSyncResult:
        """Main fusion template method - orchestrates multi-source fusion.

        Partitions KM items by source, extracts fusible data, coordinates
        with FusionCoordinator, and applies results to source records.

        Args:
            km_items: List of KM items to process for fusion.
            strategy: Fusion strategy to use (default: WEIGHTED_AVERAGE).
            conflict_resolution: How to resolve conflicts (default: PREFER_HIGHER_CONFIDENCE).
            min_sources: Minimum number of sources required for fusion (default: 2).
            min_confidence: Minimum confidence threshold for items to include.
            batch_size: Number of items to process per batch.

        Returns:
            FusionSyncResult with operation statistics.
        """
        start_time = time.time()
        result: FusionSyncResult = {
            "items_analyzed": 0,
            "items_fused": 0,
            "items_skipped": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "errors": [],
            "duration_ms": 0.0,
        }

        if not km_items:
            return result

        # Initialize fusion state if needed
        if not hasattr(self, "_fusion_state"):
            self._init_fusion_state()

        # Get fusion coordinator
        coordinator = get_fusion_coordinator()

        # Emit start event
        self._emit_event(
            "km_adapter_fusion_start",
            {
                "adapter": self.adapter_name,
                "item_count": len(km_items),
                "strategy": strategy.value,
            },
        )

        # Partition items by source
        source_items = self._partition_by_source(km_items)

        # Filter to allowed sources
        allowed_sources = set(self._get_fusion_sources())
        filtered_sources = {
            src: items for src, items in source_items.items() if src in allowed_sources
        }

        if len(filtered_sources) < min_sources:
            logger.info(
                f"[{self.adapter_name}] Insufficient sources for fusion: "
                f"{len(filtered_sources)} < {min_sources}"
            )
            result["items_skipped"] = len(km_items)
            result["duration_ms"] = (time.time() - start_time) * 1000
            return result

        # Group items by item_id for multi-source fusion
        item_groups = self._group_items_by_id(filtered_sources)

        # Process in batches
        for item_id, source_data in item_groups.items():
            result["items_analyzed"] += 1

            try:
                # Skip if insufficient sources for this item
                if len(source_data) < min_sources:
                    result["items_skipped"] += 1
                    continue

                # Build adapter validations
                validations = self._build_adapter_validations(source_data, min_confidence)

                if len(validations) < min_sources:
                    result["items_skipped"] += 1
                    continue

                # Fuse validations
                fusion_result = coordinator.fuse_validations(
                    validations,
                    strategy=strategy,
                    conflict_resolution=conflict_resolution,
                )

                # Track conflicts
                if fusion_result.conflict_detected:
                    result["conflicts_detected"] += 1
                    self._fusion_state.conflicts_detected += 1

                    if fusion_result.conflict_resolved:
                        result["conflicts_resolved"] += 1
                        self._fusion_state.conflicts_resolved += 1

                    # Emit conflict event
                    self._emit_event(
                        "km_adapter_fusion_conflict",
                        {
                            "adapter": self.adapter_name,
                            "item_id": item_id,
                            "sources": fusion_result.participating_adapters,
                            "resolved": fusion_result.conflict_resolved,
                        },
                    )

                # Apply fusion result
                record = self._get_record_for_fusion(item_id)
                if record is not None:
                    applied = self._apply_fusion_result(
                        record,
                        fusion_result,
                        metadata={
                            "fused_at": datetime.now(timezone.utc).isoformat(),
                            "strategy": strategy.value,
                            "sources": fusion_result.participating_adapters,
                        },
                    )

                    if applied:
                        result["items_fused"] += 1
                        self._update_fusion_stats(fusion_result)
                    else:
                        result["items_skipped"] += 1
                else:
                    result["items_skipped"] += 1

            except Exception as e:
                error_msg = f"Error fusing item {item_id}: {str(e)}"
                logger.warning(f"[{self.adapter_name}] {error_msg}")
                result["errors"].append(error_msg)

        # Calculate duration
        result["duration_ms"] = (time.time() - start_time) * 1000

        # Update state
        self._fusion_state.fusions_performed += result["items_fused"]
        self._fusion_state.last_fusion_at = datetime.now(timezone.utc)

        # Record metrics
        self._record_metric(
            "adapter_fusion",
            success=len(result["errors"]) == 0,
            latency=result["duration_ms"],
            extra_labels={
                "adapter": self.adapter_name,
                "strategy": strategy.value,
            },
        )

        # Emit completion event
        self._emit_event(
            "km_adapter_fusion_complete",
            {
                "adapter": self.adapter_name,
                "items_fused": result["items_fused"],
                "items_skipped": result["items_skipped"],
                "conflicts_detected": result["conflicts_detected"],
                "conflicts_resolved": result["conflicts_resolved"],
                "duration_ms": result["duration_ms"],
            },
        )

        logger.info(
            f"[{self.adapter_name}] Fusion complete: "
            f"{result['items_fused']} fused, {result['items_skipped']} skipped, "
            f"{result['conflicts_detected']} conflicts in {result['duration_ms']:.1f}ms"
        )

        return result

    def _partition_by_source(
        self,
        km_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Partition KM items by their source adapter.

        Args:
            km_items: List of KM items to partition.

        Returns:
            Dict mapping source adapter name to list of items.
        """
        partitions: Dict[str, List[Dict[str, Any]]] = {}

        for item in km_items:
            # Try to extract source from metadata
            source = self._extract_source_adapter(item)
            if source:
                if source not in partitions:
                    partitions[source] = []
                partitions[source].append(item)

        return partitions

    def _extract_source_adapter(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract the source adapter name from a KM item.

        Override to customize source extraction for your adapter.

        Args:
            item: The KM item.

        Returns:
            Source adapter name, or None if not found.
        """
        metadata = item.get("metadata", {})

        # Try various fields that might contain source info
        return (
            metadata.get("source_adapter")
            or metadata.get("adapter")
            or metadata.get("source_type")
            or item.get("source_type")
        )

    def _group_items_by_id(
        self,
        source_items: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group items from multiple sources by their item ID.

        Args:
            source_items: Dict of source -> items.

        Returns:
            Dict mapping item_id -> {source -> item_data}.
        """
        groups: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for source, items in source_items.items():
            for item in items:
                item_id = self._extract_item_id(item)
                if item_id:
                    if item_id not in groups:
                        groups[item_id] = {}

                    fusible = self._extract_fusible_data(item)
                    if fusible:
                        groups[item_id][source] = {
                            "item": item,
                            "fusible": fusible,
                        }

                    # Track source participation
                    if not hasattr(self, "_fusion_state"):
                        self._init_fusion_state()
                    self._fusion_state.source_participation[source] = (
                        self._fusion_state.source_participation.get(source, 0) + 1
                    )

        return groups

    def _extract_item_id(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract item ID for grouping.

        Args:
            item: The KM item.

        Returns:
            Item ID for grouping, or None if not found.
        """
        metadata = item.get("metadata", {})
        return metadata.get("source_id") or metadata.get("record_id") or item.get("id")

    def _build_adapter_validations(
        self,
        source_data: Dict[str, Dict[str, Any]],
        min_confidence: float,
    ) -> List[AdapterValidation]:
        """Build AdapterValidation objects from source data.

        Args:
            source_data: Dict of source -> {item, fusible}.
            min_confidence: Minimum confidence to include.

        Returns:
            List of AdapterValidation objects.
        """
        validations = []

        for source, data in source_data.items():
            fusible = data.get("fusible", {})
            item = data.get("item", {})

            confidence = fusible.get("confidence", 0.0)
            if confidence < min_confidence:
                continue

            validation = AdapterValidation(
                adapter_name=source,
                item_id=self._extract_item_id(item) or "unknown",
                confidence=confidence,
                is_valid=fusible.get("is_valid", confidence >= 0.5),
                sources=fusible.get("sources", []),
                reasoning=fusible.get("reasoning"),
                priority=self._get_source_priority(source),
                reliability=self._compute_adapter_weight(source),
            )
            validations.append(validation)

        return validations

    def _get_source_priority(self, source: str) -> int:
        """Get priority for a source adapter.

        Higher priority adapters are preferred in conflict resolution.
        Override to customize priorities for your use case.

        Args:
            source: Source adapter name.

        Returns:
            Priority value (higher = more priority).
        """
        # Default priorities based on typical adapter trustworthiness
        priorities = {
            "elo": 5,
            "consensus": 4,
            "belief": 3,
            "evidence": 3,
            "continuum": 2,
            "insights": 2,
            "pulse": 1,
        }
        return priorities.get(source, 1)

    def _compute_adapter_weight(
        self,
        adapter_name: str,
        reliability: Optional[float] = None,
    ) -> float:
        """Compute weight for an adapter based on reliability.

        Override to customize weighting for your use case.

        Args:
            adapter_name: Name of the adapter.
            reliability: Optional explicit reliability score (0-1).

        Returns:
            Weight value (0-1).
        """
        if reliability is not None:
            return reliability

        # Default weights based on typical adapter characteristics
        default_weights = {
            "elo": 0.9,
            "consensus": 0.85,
            "belief": 0.8,
            "evidence": 0.8,
            "continuum": 0.75,
            "insights": 0.7,
            "pulse": 0.6,
            "cost": 0.5,
        }
        return default_weights.get(adapter_name, 0.5)

    def _update_fusion_stats(self, fusion_result: FusedValidation) -> None:
        """Update running fusion statistics.

        Args:
            fusion_result: The fusion result to track.
        """
        if not hasattr(self, "_fusion_state"):
            self._init_fusion_state()

        # Update running average
        n = self._fusion_state.fusions_performed
        old_avg = self._fusion_state.avg_fusion_confidence

        if n == 0:
            self._fusion_state.avg_fusion_confidence = fusion_result.fused_confidence
        else:
            # Incremental mean update
            self._fusion_state.avg_fusion_confidence = old_avg + (
                fusion_result.fused_confidence - old_avg
            ) / (n + 1)

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion statistics for this adapter.

        Returns:
            Dict with fusion statistics.
        """
        if not hasattr(self, "_fusion_state"):
            self._init_fusion_state()

        return self._fusion_state.to_dict()

    @property
    def supports_fusion(self) -> bool:
        """Indicate that this adapter supports fusion operations."""
        return True


__all__ = [
    "FusionMixin",
    "FusionSyncResult",
    "FusionState",
]
