"""SLO monitoring for Knowledge Mound adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterSLOConfig:
    """SLO configuration for adapter operations.

    Defines latency thresholds for monitoring adapter performance.
    """

    # Forward sync (source -> KM) latencies in milliseconds
    forward_sync_p50_ms: float = 100.0
    forward_sync_p90_ms: float = 300.0
    forward_sync_p99_ms: float = 800.0

    # Reverse query (KM -> consumer) latencies in milliseconds
    reverse_query_p50_ms: float = 50.0
    reverse_query_p90_ms: float = 150.0
    reverse_query_p99_ms: float = 500.0

    # Semantic search latencies in milliseconds
    semantic_search_p50_ms: float = 100.0
    semantic_search_p90_ms: float = 300.0
    semantic_search_p99_ms: float = 1000.0

    # Operation timeouts in seconds
    forward_sync_timeout_s: float = 5.0
    reverse_query_timeout_s: float = 3.0
    semantic_search_timeout_s: float = 5.0


# Default SLO config
_adapter_slo_config: AdapterSLOConfig | None = None


def get_adapter_slo_config() -> AdapterSLOConfig:
    """Get the adapter SLO configuration."""
    global _adapter_slo_config
    if _adapter_slo_config is None:
        _adapter_slo_config = AdapterSLOConfig()
    return _adapter_slo_config


def set_adapter_slo_config(config: AdapterSLOConfig) -> None:
    """Set a custom adapter SLO configuration."""
    global _adapter_slo_config
    _adapter_slo_config = config


def check_adapter_slo(
    operation: str,
    latency_ms: float,
    adapter_name: str,
    percentile: str = "p99",
) -> tuple[bool, str]:
    """Check if adapter operation meets SLO.

    Args:
        operation: Operation type (forward_sync, reverse_query, semantic_search)
        latency_ms: Measured latency in milliseconds
        adapter_name: Name of the adapter
        percentile: SLO percentile to check (p50, p90, p99)

    Returns:
        Tuple of (is_within_slo, message)
    """
    config = get_adapter_slo_config()

    # Get threshold for operation and percentile
    attr_name = f"{operation}_{percentile}_ms"
    threshold = getattr(config, attr_name, None)

    if threshold is None:
        return True, f"No SLO defined for {operation}.{percentile}"

    is_within = latency_ms <= threshold

    if is_within:
        return (
            True,
            f"{adapter_name}.{operation} latency {latency_ms:.1f}ms "
            f"within {percentile} SLO ({threshold}ms)",
        )
    else:
        return (
            False,
            f"{adapter_name}.{operation} latency {latency_ms:.1f}ms "
            f"EXCEEDS {percentile} SLO ({threshold}ms)",
        )


def record_adapter_slo_check(
    adapter_name: str,
    operation: str,
    latency_ms: float,
    success: bool,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Record an adapter operation and check SLO compliance.

    Combines metric recording with SLO checking for convenience.

    Args:
        adapter_name: Name of the adapter
        operation: Operation type (forward_sync, reverse_query, semantic_search)
        latency_ms: Measured latency in milliseconds
        success: Whether the operation succeeded
        context: Optional context for SLO violation reporting

    Returns:
        Tuple of (is_within_slo, message)
    """
    # Record the operation latency
    try:
        from aragora.observability.metrics.km import (
            record_forward_sync_latency,
            record_reverse_query_latency,
            record_km_operation,
        )

        latency_seconds = latency_ms / 1000.0

        if operation == "forward_sync":
            record_forward_sync_latency(adapter_name, latency_seconds)
        elif operation == "reverse_query":
            record_reverse_query_latency(adapter_name, latency_seconds)

        record_km_operation(f"{adapter_name}_{operation}", success, latency_seconds)

    except ImportError:
        pass
    except (RuntimeError, ValueError, OSError, AttributeError) as e:
        logger.debug("Failed to record adapter metrics: %s", e)

    # Check SLO compliance
    passed, message = check_adapter_slo(operation, latency_ms, adapter_name)

    # Record SLO check if metrics available
    if not passed:
        try:
            from aragora.observability.metrics.slo import (
                check_and_record_slo_with_recovery,
            )

            # Map to standard SLO operation name
            slo_operation = f"adapter_{operation}"
            check_and_record_slo_with_recovery(
                operation=slo_operation,
                latency_ms=latency_ms,
                context={
                    "adapter": adapter_name,
                    "operation": operation,
                    "success": success,
                    **(context or {}),
                },
            )
        except ImportError:
            pass
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to record SLO check: %s", e)

        logger.warning(message)

    return passed, message
