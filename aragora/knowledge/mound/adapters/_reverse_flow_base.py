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

        def _get_record_for_validation(self, source_id: str) -> Any | None:
            return self._source.get(source_id)

        def _apply_km_validation(
            self, record: Any, km_confidence: float, cross_refs: list[str]
        ) -> bool:
            record.km_validated = True
            return True

        def _extract_source_id(self, item: Dict) -> str | None:
            meta = item.get("metadata", {})
            return meta.get("source_id")
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from aragora.knowledge.mound.adapters._types import ValidationSyncResult

logger = logging.getLogger(__name__)


@dataclass
class BatchTimeoutConfig:
    """Configuration for per-item and total batch timeout enforcement.

    Attributes:
        per_item_timeout_seconds: Maximum time to process a single item.
        total_batch_timeout_seconds: Maximum time for the entire batch.
        fail_fast_on_timeout: If True, stop the batch on first timeout.
            If False, skip the timed-out item and continue.
    """

    per_item_timeout_seconds: float = 2.0
    total_batch_timeout_seconds: float = 60.0
    fail_fast_on_timeout: bool = False


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
    # Attribute declaration for mypy - _emit_event provided by KnowledgeMoundAdapter
    _emit_event: Any

    @abstractmethod
    def _get_record_for_validation(self, source_id: str) -> Any | None:
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
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
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

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
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
        km_items: list[dict[str, Any]],
        min_confidence: float = 0.7,
        batch_size: int = 100,
        timeout_config: BatchTimeoutConfig | None = None,
    ) -> dict[str, Any]:
        """Sync KM validations back to the source system (reverse flow).

        This is the main template method for reverse flow operations.
        When KM validates or cross-references source records, this method
        updates the source records with validation metadata.

        Args:
            km_items: KM items with validation data.
            min_confidence: Minimum confidence for applying changes.
            batch_size: Maximum items to process per batch.
            timeout_config: Optional per-item and total batch timeout config.

        Returns:
            Dict with sync statistics (records_analyzed, records_updated, etc.).
        """
        start_time = time.time()
        tc = timeout_config or BatchTimeoutConfig()

        result: dict[str, Any] = {
            "records_analyzed": 0,
            "records_updated": 0,
            "records_skipped": 0,
            "errors": [],
            "duration_ms": 0.0,
        }

        # Limit batch size for safety
        items_to_process = km_items[:batch_size]

        for item in items_to_process:
            # Check total batch timeout
            elapsed = time.time() - start_time
            if elapsed >= tc.total_batch_timeout_seconds:
                result["errors"].append(
                    f"Total batch timeout ({tc.total_batch_timeout_seconds}s) exceeded "
                    f"after {result['records_analyzed']} items"
                )
                break

            source_id = self._extract_source_id(item)
            if not source_id:
                continue

            result["records_analyzed"] += 1

            try:
                # Wrap per-item processing with timeout
                await asyncio.wait_for(
                    self._process_single_item(item, source_id, min_confidence, result),
                    timeout=tc.per_item_timeout_seconds,
                )

            except asyncio.TimeoutError:
                error_msg = f"Per-item timeout ({tc.per_item_timeout_seconds}s) for {source_id}"
                result["errors"].append(error_msg)
                logger.warning(f"[{self.adapter_name}] {error_msg}")

                if tc.fail_fast_on_timeout:
                    result["errors"].append("Batch stopped: fail_fast_on_timeout=True")
                    break

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

    async def _process_single_item(
        self,
        item: dict[str, Any],
        source_id: str,
        min_confidence: float,
        result: dict[str, Any],
    ) -> None:
        """Process a single KM item for reverse validation.

        Extracted to enable per-item timeout enforcement.

        Args:
            item: The KM item with validation data.
            source_id: The source record identifier.
            min_confidence: Minimum confidence threshold.
            result: Mutable result dict to update in place.
        """
        # Get the source record
        record = self._get_record_for_validation(source_id)
        if record is None:
            result["records_skipped"] += 1
            return

        # Parse and validate confidence
        km_confidence = self._parse_km_confidence(item.get("confidence", 0.0))

        if km_confidence < min_confidence:
            result["records_skipped"] += 1
            return

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


__all__ = ["BatchTimeoutConfig", "ReverseFlowMixin", "ValidationSyncResult"]
