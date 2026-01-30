"""Resilient adapter mixin and combined status."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from aragora.knowledge.mound.resilience._common import asyncio_timeout
from aragora.knowledge.mound.resilience.bulkhead import (
    AdapterBulkhead,
    BulkheadConfig,
    BulkheadFullError,
    get_adapter_bulkhead,
    get_all_adapter_bulkhead_stats,
)
from aragora.knowledge.mound.resilience.circuit_breaker import (
    AdapterCircuitBreaker,
    AdapterCircuitBreakerConfig,
    AdapterUnavailableError,
    _adapter_circuits,
    get_adapter_circuit_breaker,
    get_all_adapter_circuit_stats,
)
from aragora.knowledge.mound.resilience.retry import RetryConfig
from aragora.knowledge.mound.resilience.slo import (
    get_adapter_slo_config,
    record_adapter_slo_check,
)

logger = logging.getLogger(__name__)


class ResilientAdapterMixin:
    """
    Mixin providing resilience patterns for Knowledge Mound adapters.

    Combines circuit breaker, bulkhead, retry, and SLO monitoring
    into a consistent interface for adapter implementations.

    Usage:
        class MyAdapter(ResilientAdapterMixin):
            def __init__(self):
                self._init_resilience("my_adapter")

            async def my_operation(self):
                async with self._resilient_call("forward_sync"):
                    return await self._do_operation()
    """

    _adapter_name: str
    _circuit_breaker: AdapterCircuitBreaker | None = None
    _bulkhead: AdapterBulkhead | None = None
    _retry_config: RetryConfig | None = None
    _timeout_seconds: float = 5.0

    def _init_resilience(
        self,
        adapter_name: str,
        circuit_config: AdapterCircuitBreakerConfig | None = None,
        bulkhead_config: BulkheadConfig | None = None,
        retry_config: RetryConfig | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize resilience components.

        Args:
            adapter_name: Name of the adapter
            circuit_config: Circuit breaker configuration
            bulkhead_config: Bulkhead configuration
            retry_config: Retry configuration
            timeout_seconds: Default operation timeout
        """
        self._adapter_name = adapter_name
        self._circuit_breaker = get_adapter_circuit_breaker(adapter_name, circuit_config)
        self._bulkhead = get_adapter_bulkhead(adapter_name, bulkhead_config)
        self._retry_config = retry_config or RetryConfig()
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _resilient_call(
        self,
        operation: str,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute an operation with full resilience protection.

        Applies circuit breaker, bulkhead, timeout, and SLO monitoring.

        Args:
            operation: Operation name (forward_sync, reverse_query, etc.)
            timeout: Optional timeout override

        Yields:
            Context dict for storing operation metadata

        Raises:
            AdapterUnavailableError: If circuit is open
            BulkheadFullError: If bulkhead is full
            asyncio.TimeoutError: If operation times out
        """
        if not hasattr(self, "_adapter_name"):
            raise RuntimeError("Call _init_resilience() before using _resilient_call()")

        context: dict[str, Any] = {
            "adapter": self._adapter_name,
            "operation": operation,
        }

        start_time = time.time()
        success = False
        timeout_s = timeout or self._timeout_seconds

        try:
            # Check circuit breaker
            if self._circuit_breaker and not self._circuit_breaker.can_proceed():
                remaining = self._circuit_breaker.cooldown_remaining()
                raise AdapterUnavailableError(self._adapter_name, remaining)

            # Acquire bulkhead permit with timeout
            if self._bulkhead:
                async with self._bulkhead.acquire():
                    # Execute with timeout
                    async with asyncio_timeout(timeout_s):
                        yield context
                        success = True
            else:
                async with asyncio_timeout(timeout_s):
                    yield context
                    success = True

            # Record success in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

        except asyncio.TimeoutError:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(f"Timeout after {timeout_s}s")
            raise
        except (AdapterUnavailableError, BulkheadFullError):
            raise
        except Exception as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(str(e))
            raise
        finally:
            # Record SLO metrics
            latency_ms = (time.time() - start_time) * 1000
            record_adapter_slo_check(
                self._adapter_name,
                operation,
                latency_ms,
                success,
                context,
            )

    def get_resilience_stats(self) -> dict[str, Any]:
        """Get combined resilience statistics."""
        stats: dict[str, Any] = {
            "adapter_name": self._adapter_name,
            "timeout_seconds": self._timeout_seconds,
        }

        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats().to_dict()

        if self._bulkhead:
            stats["bulkhead"] = self._bulkhead.get_stats()

        return stats


def get_km_resilience_status() -> dict[str, Any]:
    """Get comprehensive resilience status for all KM components.

    Returns:
        Dict with circuit breaker, bulkhead, and SLO status
    """
    return {
        "circuit_breakers": get_all_adapter_circuit_stats(),
        "bulkheads": get_all_adapter_bulkhead_stats(),
        "slo_config": {
            "forward_sync_p99_ms": get_adapter_slo_config().forward_sync_p99_ms,
            "reverse_query_p99_ms": get_adapter_slo_config().reverse_query_p99_ms,
            "semantic_search_p99_ms": get_adapter_slo_config().semantic_search_p99_ms,
        },
        "adapters_with_open_circuits": [
            name for name, cb in _adapter_circuits.items() if cb.is_open
        ],
        "total_adapters_tracked": len(_adapter_circuits),
    }
