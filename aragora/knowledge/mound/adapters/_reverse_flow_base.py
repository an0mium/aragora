"""Base class for reverse flow (KM -> Source) sync operations.

Provides the template method pattern for sync_validations_from_km() that is
duplicated across 7 adapters with 80-90% identical code.

This consolidates ~800-1000 lines of duplicated reverse flow logic across:
- consensus_adapter.py
- continuum_adapter.py
- elo_adapter.py
- belief_adapter.py
- critique_adapter.py
- insights_adapter.py
- pulse_adapter.py

Usage:
    from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin

    class MyAdapter(ReverseFlowMixin, KnowledgeMoundAdapter):
        adapter_name = "my_adapter"

        def _get_record_for_validation(self, source_id: str) -> Optional[Any]:
            return self._source.get(source_id)

        def _apply_km_validation(
            self, record: Any, km_confidence: float, cross_refs: List[str]
        ) -> bool:
            record.km_validated = True
            return True

        def _extract_source_id(self, item: Dict) -> Optional[str]:
            meta = item.get("metadata", {})
            return meta.get("source_id")
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class ValidationSyncResult(TypedDict):
    """Result type for reverse validation sync operations."""

    records_analyzed: int
    records_updated: int
    records_skipped: int
    errors: List[str]
    duration_ms: float


class ReverseFlowMixin:
    """Mixin providing reverse flow (KM -> Source) sync operations.

    Provides:
    - sync_validations_from_km(): Template method for reverse sync
    - _parse_km_confidence(): Parse confidence from various formats
    - Common logging and event emission

    Required from inheriting class:
    - adapter_name: str identifying the adapter
    - _get_record_for_validation(): Get a record by source ID
    - _apply_km_validation(): Apply validation to a record
    - _extract_source_id(): Extract source ID from KM item
    - _emit_event(): Event emission (from KnowledgeMoundAdapter)
    """

    # Expected from KnowledgeMoundAdapter or subclass
    adapter_name: str

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    def _init_reverse_flow_state(self) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    @abstractmethod
    def _get_record_for_validation(self, source_id: str) -> Optional[Any]:
        """Get a record by its source ID for validation.

        Args:
            source_id: The source system record identifier.

        Returns:
            The record, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply KM validation to a source record.

        Args:
            record: The record to update.
            km_confidence: Confidence score from KM validation.
            cross_refs: Optional cross-references from KM.
            metadata: Optional additional metadata from KM.

        Returns:
            True if the record was updated, False otherwise.
        """
        raise NotImplementedError

    def _extract_source_id(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract the source ID from a KM item.

        Override to customize ID extraction for your adapter.
        Default looks in metadata.source_id, metadata.record_id, or item.id.

        Args:
            item: The KM item with validation data.

        Returns:
            The source ID, or None if not found.
        """
        meta = item.get("metadata", {})
        return meta.get("source_id") or meta.get("record_id") or item.get("id")

    def _parse_km_confidence(self, confidence_value: Any) -> float:
        """Parse confidence value from various formats.

        Handles:
        - Float values (0.0-1.0)
        - Integer percentages (0-100)
        - String levels ("low", "medium", "high")

        Args:
            confidence_value: The confidence value to parse.

        Returns:
            Confidence as a float (0.0-1.0).
        """
        if confidence_value is None:
            return 0.0

        if isinstance(confidence_value, (int, float)):
            value = float(confidence_value)
            # Convert percentage to decimal if needed
            if value > 1.0:
                return min(value / 100.0, 1.0)
            return max(0.0, min(value, 1.0))

        if isinstance(confidence_value, str):
            level_map = {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.9,
                "very_high": 0.95,
                "certain": 1.0,
            }
            return level_map.get(confidence_value.lower(), 0.5)

        return 0.0

    def _get_validation_timestamp(self) -> str:
        """Get the current timestamp for validation records.

        Returns:
            ISO8601 formatted timestamp in UTC.
        """
        return datetime.now(timezone.utc).isoformat()

    async def sync_validations_from_km(
        self,
        km_items: List[Dict[str, Any]],
        min_confidence: float = 0.7,
        batch_size: int = 100,
    ) -> ValidationSyncResult:
        """Sync KM validations back to the source system (reverse flow).

        This is the main template method for reverse flow operations.
        When KM validates or cross-references source records, this method
        updates the source records with validation metadata.

        Args:
            km_items: KM items with validation data.
            min_confidence: Minimum confidence for applying changes.
            batch_size: Maximum items to process per batch.

        Returns:
            ValidationSyncResult with sync statistics.
        """
        start_time = time.time()

        result: ValidationSyncResult = {
            "records_analyzed": 0,
            "records_updated": 0,
            "records_skipped": 0,
            "errors": [],
            "duration_ms": 0.0,
        }

        # Limit batch size for safety
        items_to_process = km_items[:batch_size]

        for item in items_to_process:
            source_id = self._extract_source_id(item)
            if not source_id:
                continue

            result["records_analyzed"] += 1

            try:
                # Get the source record
                record = self._get_record_for_validation(source_id)
                if record is None:
                    result["records_skipped"] += 1
                    continue

                # Parse and validate confidence
                km_confidence = self._parse_km_confidence(item.get("confidence", 0.0))

                if km_confidence < min_confidence:
                    result["records_skipped"] += 1
                    continue

                # Extract cross-references and metadata
                meta = item.get("metadata", {})
                cross_refs = meta.get("cross_references", [])
                validation_meta = {
                    "km_validated": True,
                    "km_validation_confidence": km_confidence,
                    "km_validation_timestamp": self._get_validation_timestamp(),
                }
                if extra_meta := meta.get("validation_data"):
                    validation_meta.update(extra_meta)

                # Apply the validation
                if self._apply_km_validation(record, km_confidence, cross_refs, validation_meta):
                    result["records_updated"] += 1

                    # Emit event for reverse sync
                    self._emit_event(
                        "km_adapter_reverse_sync",
                        {
                            "source": self.adapter_name,
                            "source_id": source_id,
                            "km_confidence": km_confidence,
                            "action": "validated",
                        },
                    )
                else:
                    result["records_skipped"] += 1

            except Exception as e:
                error_msg = f"Failed to update {source_id}: {str(e)}"
                result["errors"].append(error_msg)
                logger.warning(f"[{self.adapter_name}] Reverse sync failed for {source_id}: {e}")

        result["duration_ms"] = (time.time() - start_time) * 1000

        logger.info(
            f"[{self.adapter_name}] Reverse sync complete: "
            f"analyzed={result['records_analyzed']}, "
            f"updated={result['records_updated']}, "
            f"skipped={result['records_skipped']}"
        )

        return result


__all__ = ["ReverseFlowMixin", "ValidationSyncResult"]
