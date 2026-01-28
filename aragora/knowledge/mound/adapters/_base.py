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

from aragora.observability.tracing import get_tracer

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
        enable_tracing: bool = True,
    ):
        """Initialize the adapter with common configuration.

        Args:
            enable_dual_write: If True, writes go to both systems during migration.
            event_callback: Optional callback for emitting events (event_type, data).
            enable_tracing: If True, OpenTelemetry tracing is enabled for operations.
        """
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback
        self._enable_tracing = enable_tracing
        self._tracer = get_tracer() if enable_tracing else None
        self._last_operation_time: float = 0.0
        self._error_count: int = 0
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

    def _timed_operation(self, operation_name: str, **span_attributes: Any):
        """Context manager for timing, recording, and tracing operations.

        Usage:
            with self._timed_operation("search", query="test") as timer:
                results = self._do_search()
            # Metrics and traces automatically recorded

        Args:
            operation_name: Name of the operation for metrics/tracing.
            **span_attributes: Additional attributes to add to the trace span.

        Returns:
            Context manager that records metrics and traces on exit.
        """
        return _TimedOperation(self, operation_name, span_attributes)

    def health_check(self) -> Dict[str, Any]:
        """Return adapter health status for monitoring.

        Returns:
            Dict containing health status, last operation time, and error counts.
        """
        return {
            "adapter": self.adapter_name,
            "healthy": self._error_count < 5,  # Unhealthy if 5+ consecutive errors
            "last_operation_time": self._last_operation_time,
            "error_count": self._error_count,
            "reverse_flow_stats": self.get_reverse_flow_stats(),
        }

    def reset_health_counters(self) -> None:
        """Reset health counters (e.g., after recovering from errors)."""
        self._error_count = 0


class _TimedOperation:
    """Context manager for timing and tracing adapter operations."""

    def __init__(
        self,
        adapter: KnowledgeMoundAdapter,
        operation: str,
        span_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.adapter = adapter
        self.operation = operation
        self.span_attributes = span_attributes or {}
        self.start_time = 0.0
        self.success = True
        self.error: Optional[Exception] = None
        self._span = None

    def __enter__(self) -> "_TimedOperation":
        self.start_time = time.time()

        # Start trace span if tracing enabled
        if self.adapter._tracer is not None:
            span_name = f"{self.adapter.adapter_name}.{self.operation}"
            self._span = self.adapter._tracer.start_span(span_name)
            self._span.set_attribute("adapter.name", self.adapter.adapter_name)
            self._span.set_attribute("adapter.operation", self.operation)
            for key, value in self.span_attributes.items():
                if value is not None:
                    self._span.set_attribute(f"adapter.{key}", str(value))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency = time.time() - self.start_time
        self.success = exc_type is None
        if not self.success:
            self.error = exc_val

        # Update adapter state
        self.adapter._last_operation_time = time.time()
        if self.success:
            self.adapter._error_count = 0  # Reset on success
        else:
            self.adapter._error_count += 1

        # End trace span
        if self._span is not None:
            self._span.set_attribute("adapter.success", self.success)
            self._span.set_attribute("adapter.latency_ms", latency * 1000)
            if not self.success and exc_val is not None:
                self._span.record_exception(exc_val)
            self._span.end()

        self.adapter._record_metric(self.operation, self.success, latency)
        # Don't suppress exceptions (returning None is equivalent to False)


__all__ = [
    "KnowledgeMoundAdapter",
    "EventCallback",
]
