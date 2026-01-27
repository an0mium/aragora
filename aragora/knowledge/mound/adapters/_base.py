"""Base class for Knowledge Mound adapters.

Provides shared utilities for event emission, metrics recording, SLO monitoring,
and reverse flow state management that are common across all adapters.

This consolidates ~200 lines of duplicated code from 10+ adapters.

Usage:
    from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter

    class MyAdapter(KnowledgeMoundAdapter):
        adapter_name = "my_adapter"

        def __init__(self, source_system, **kwargs):
            super().__init__(**kwargs)
            self._source = source_system
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]


class KnowledgeMoundAdapter:
    """Base class for all Knowledge Mound adapters.

    Provides:
    - Event emission with callback management
    - Prometheus metrics recording
    - SLO monitoring and alerting
    - Reverse flow state tracking
    - Common utility methods

    Subclasses should override:
    - adapter_name: Unique identifier for metrics/logging
    """

    adapter_name: str = "base"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
    ):
        """Initialize the adapter with common configuration.

        Args:
            enable_dual_write: If True, writes go to both systems during migration.
            event_callback: Optional callback for emitting events (event_type, data).
        """
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback
        self._init_reverse_flow_state()

    def _init_reverse_flow_state(self) -> None:
        """Initialize tracking state for reverse flow operations.

        Called automatically in __init__ and can be called to reset state.
        """
        if not hasattr(self, "_reverse_flow_state"):
            self._reverse_flow_state: Dict[str, Any] = {}

        self._reverse_flow_state.update(
            {
                "validations_applied": 0,
                "adjustments_made": 0,
                "validations_stored": [],
                "outcome_history": {},
                "pending_validations": [],
            }
        )

    def clear_reverse_flow_state(self) -> None:
        """Clear reverse flow state for testing or reset."""
        self._init_reverse_flow_state()

    def get_reverse_flow_stats(self) -> Dict[str, Any]:
        """Get statistics about reverse flow operations.

        Returns:
            Dict with validation counts, adjustment counts, and history.
        """
        self._init_reverse_flow_state()
        return {
            "validations_applied": self._reverse_flow_state.get("validations_applied", 0),
            "adjustments_made": self._reverse_flow_state.get("adjustments_made", 0),
            "pending_count": len(self._reverse_flow_state.get("pending_validations", [])),
            "history_size": len(self._reverse_flow_state.get("outcome_history", {})),
        }

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications.

        Args:
            callback: Function that receives (event_type, data) tuples.
        """
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured.

        Events are used for real-time WebSocket notifications.

        Args:
            event_type: Type of event (e.g., "km_sync", "validation_applied").
            data: Event payload.
        """
        if not self._event_callback:
            return

        try:
            self._event_callback(event_type, data)
        except Exception as e:
            logger.warning(f"[{self.adapter_name}] Failed to emit event {event_type}: {e}")

    def _record_metric(
        self,
        operation: str,
        success: bool,
        latency: float,
        extra_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record Prometheus metric for adapter operation and check SLOs.

        Args:
            operation: Operation name (search, store, sync, semantic_search).
            success: Whether operation succeeded.
            latency: Operation latency in seconds.
            extra_labels: Additional labels for the metric.
        """
        latency_ms = latency * 1000  # Convert to milliseconds

        try:
            from aragora.observability.metrics.km import (
                record_km_operation,
                record_km_adapter_sync,
            )

            record_km_operation(operation, success, latency)
            if operation in ("store", "sync", "forward_sync"):
                record_km_adapter_sync(self.adapter_name, "forward", success)
            elif operation in ("reverse_sync", "validate"):
                record_km_adapter_sync(self.adapter_name, "reverse", success)
        except ImportError:
            pass  # Metrics not available
        except Exception as e:
            logger.debug(f"[{self.adapter_name}] Failed to record metric: {e}")

        # Check SLOs and alert on violations
        self._check_slo(operation, latency_ms)

    def _check_slo(self, operation: str, latency_ms: float) -> None:
        """Check SLO thresholds and record violations.

        Args:
            operation: Operation name.
            latency_ms: Operation latency in milliseconds.
        """
        try:
            from aragora.observability.metrics.slo import check_and_record_slo_with_recovery

            # Map operation to SLO name
            slo_mapping = {
                "search": "adapter_reverse",
                "store": "adapter_forward_sync",
                "sync": "adapter_forward_sync",
                "semantic_search": "adapter_search",
                "reverse_sync": "adapter_reverse",
                "validate": "adapter_reverse",
            }

            slo_name = slo_mapping.get(operation)
            if slo_name:
                passed, message = check_and_record_slo_with_recovery(
                    operation=slo_name,
                    latency_ms=latency_ms,
                    context={"adapter": self.adapter_name, "operation": operation},
                )
                if not passed:
                    logger.warning(f"[{self.adapter_name}] SLO violation: {message}")
        except ImportError:
            pass  # SLO monitoring not available
        except Exception as e:
            logger.debug(f"[{self.adapter_name}] Failed to check SLO: {e}")

    def _record_validation_outcome(
        self,
        record_id: str,
        outcome: str,
        confidence: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record outcome of a validation for tracking.

        Args:
            record_id: ID of the validated record.
            outcome: Validation outcome (applied, skipped, failed).
            confidence: Confidence score of the validation.
            details: Additional details about the validation.
        """
        self._init_reverse_flow_state()

        self._reverse_flow_state["outcome_history"][record_id] = {
            "outcome": outcome,
            "confidence": confidence,
            "timestamp": time.time(),
            "details": details or {},
        }

        if outcome == "applied":
            self._reverse_flow_state["validations_applied"] += 1
        elif outcome == "adjusted":
            self._reverse_flow_state["adjustments_made"] += 1

    def _timed_operation(self, operation_name: str):
        """Context manager for timing and recording operations.

        Usage:
            with self._timed_operation("search") as timer:
                results = self._do_search()
            # Metrics automatically recorded

        Args:
            operation_name: Name of the operation for metrics.

        Returns:
            Context manager that records metrics on exit.
        """
        return _TimedOperation(self, operation_name)


class _TimedOperation:
    """Context manager for timing adapter operations."""

    def __init__(self, adapter: KnowledgeMoundAdapter, operation: str):
        self.adapter = adapter
        self.operation = operation
        self.start_time = 0.0
        self.success = True
        self.error: Optional[Exception] = None

    def __enter__(self) -> "_TimedOperation":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        latency = time.time() - self.start_time
        self.success = exc_type is None
        if not self.success:
            self.error = exc_val

        self.adapter._record_metric(self.operation, self.success, latency)
        return False  # Don't suppress exceptions


__all__ = [
    "KnowledgeMoundAdapter",
    "EventCallback",
]
