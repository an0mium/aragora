"""
Backward-compatibility convenience wrappers for metrics.

Extracted from metrics/__init__.py for maintainability.
Provides thin wrapper functions that compose lower-level metric calls.
"""

from __future__ import annotations

from aragora.observability.metrics.km import (
    record_km_event_emitted,
    record_km_operation,
)
from aragora.observability.metrics.webhook import record_webhook_delivery


def record_km_inbound_event(event_type: str, source: str, success: bool = True) -> None:
    """Record an inbound event to Knowledge Mound."""
    record_km_operation("ingest", success, 0.0)
    record_km_event_emitted(f"inbound_{event_type}")


def record_webhook_failure(endpoint: str, error_type: str) -> None:
    """Record a webhook delivery failure."""
    record_webhook_delivery(
        event_type=endpoint,
        success=False,
        duration_seconds=0.0,
        status_code=500 if error_type == "connection_error" else 400,
    )


def set_webhook_circuit_breaker_state(endpoint: str, state: str) -> None:
    """Set the circuit breaker state for a webhook endpoint."""
    # No-op for now - webhook module doesn't have this metric
    pass
