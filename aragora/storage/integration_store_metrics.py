"""
Integration Store Observability.

Provides metrics and monitoring for IntegrationStore operations.
Tracks latency, errors, encryption overhead, and health status.

Usage:
    from aragora.storage.integration_store_metrics import (
        InstrumentedIntegrationStore,
        get_integration_metrics,
    )

    # Wrap an existing store
    store = InstrumentedIntegrationStore(base_store)

    # Get metrics
    metrics = await get_integration_metrics()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_seconds: float = 0.0
    min_latency_seconds: Optional[float] = None
    max_latency_seconds: Optional[float] = None
    last_call_at: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    @property
    def avg_latency_seconds(self) -> float:
        """Average latency in seconds."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_seconds / self.successful_calls

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    def record_success(self, latency_seconds: float) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_latency_seconds += latency_seconds
        self.last_call_at = datetime.now(timezone.utc)

        if self.min_latency_seconds is None or latency_seconds < self.min_latency_seconds:
            self.min_latency_seconds = latency_seconds
        if self.max_latency_seconds is None or latency_seconds > self.max_latency_seconds:
            self.max_latency_seconds = latency_seconds

    def record_failure(self, error: str) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_error = error
        self.last_error_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "avg_latency_seconds": round(self.avg_latency_seconds, 4),
            "min_latency_seconds": (
                round(self.min_latency_seconds, 4) if self.min_latency_seconds else None
            ),
            "max_latency_seconds": (
                round(self.max_latency_seconds, 4) if self.max_latency_seconds else None
            ),
            "success_rate": round(self.success_rate, 2),
            "last_call_at": self.last_call_at.isoformat() if self.last_call_at else None,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
        }


@dataclass
class IntegrationStoreMetrics:
    """Comprehensive metrics for IntegrationStore."""

    # Operation metrics
    get_operations: OperationMetrics = field(default_factory=OperationMetrics)
    save_operations: OperationMetrics = field(default_factory=OperationMetrics)
    delete_operations: OperationMetrics = field(default_factory=OperationMetrics)
    list_operations: OperationMetrics = field(default_factory=OperationMetrics)
    refresh_token_operations: OperationMetrics = field(default_factory=OperationMetrics)

    # Encryption metrics
    encryption_operations: OperationMetrics = field(default_factory=OperationMetrics)
    decryption_operations: OperationMetrics = field(default_factory=OperationMetrics)

    # User mapping metrics
    user_mapping_operations: OperationMetrics = field(default_factory=OperationMetrics)

    # Health tracking
    backend_type: str = "unknown"
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    active_integrations: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_type": self.backend_type,
            "is_healthy": self.is_healthy,
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "active_integrations": self.active_integrations,
            "cache_hit_rate": round(self.cache_hit_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "operations": {
                "get": self.get_operations.to_dict(),
                "save": self.save_operations.to_dict(),
                "delete": self.delete_operations.to_dict(),
                "list": self.list_operations.to_dict(),
                "refresh_token": self.refresh_token_operations.to_dict(),
                "encryption": self.encryption_operations.to_dict(),
                "decryption": self.decryption_operations.to_dict(),
                "user_mapping": self.user_mapping_operations.to_dict(),
            },
        }


# Global metrics instance
_metrics: Optional[IntegrationStoreMetrics] = None
_metrics_lock = asyncio.Lock()


def get_metrics() -> IntegrationStoreMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = IntegrationStoreMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset global metrics (for testing)."""
    global _metrics
    _metrics = None


def track_operation(operation_type: str):
    """
    Decorator to track operation metrics.

    Args:
        operation_type: One of "get", "save", "delete", "list", "refresh_token",
                       "encryption", "decryption", "user_mapping"
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics()
            operation_metrics = getattr(metrics, f"{operation_type}_operations")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                operation_metrics.record_success(latency)

                # Reset failure tracking on success
                metrics.consecutive_failures = 0
                metrics.is_healthy = True

                return result
            except Exception as e:
                operation_metrics.record_failure(str(e))
                metrics.consecutive_failures += 1

                # Mark unhealthy after 3 consecutive failures
                if metrics.consecutive_failures >= 3:
                    metrics.is_healthy = False

                raise

        return wrapper

    return decorator


class InstrumentedIntegrationStore:
    """
    Wrapper that adds metrics to any IntegrationStore implementation.

    Tracks:
    - Operation latency
    - Success/failure rates
    - Encryption overhead
    - Cache hit rates
    - Health status
    """

    def __init__(self, base_store: Any, backend_type: str = "unknown"):
        """
        Initialize with a base store.

        Args:
            base_store: The underlying IntegrationStore implementation
            backend_type: Type of backend ("memory", "sqlite", "postgresql", "redis")
        """
        self._store = base_store
        self._metrics = get_metrics()
        self._metrics.backend_type = backend_type

    @property
    def metrics(self) -> IntegrationStoreMetrics:
        """Get current metrics."""
        return self._metrics

    async def get(self, integration_type: str, user_id: str = "default") -> Optional[Any]:
        """Get integration with metrics tracking."""
        start_time = time.time()
        try:
            result = await self._store.get_async(integration_type, user_id)
            latency = time.time() - start_time
            self._metrics.get_operations.record_success(latency)
            self._metrics.consecutive_failures = 0
            self._metrics.is_healthy = True

            if result:
                self._metrics.record_cache_miss()  # Actually fetched
            return result
        except Exception as e:
            self._metrics.get_operations.record_failure(str(e))
            self._metrics.consecutive_failures += 1
            if self._metrics.consecutive_failures >= 3:
                self._metrics.is_healthy = False
            raise

    async def save(self, config: Any) -> None:
        """Save integration with metrics tracking."""
        start_time = time.time()
        try:
            await self._store.save_async(config)
            latency = time.time() - start_time
            self._metrics.save_operations.record_success(latency)
            self._metrics.consecutive_failures = 0
            self._metrics.is_healthy = True
        except Exception as e:
            self._metrics.save_operations.record_failure(str(e))
            self._metrics.consecutive_failures += 1
            if self._metrics.consecutive_failures >= 3:
                self._metrics.is_healthy = False
            raise

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration with metrics tracking."""
        start_time = time.time()
        try:
            result = await self._store.delete_async(integration_type, user_id)
            latency = time.time() - start_time
            self._metrics.delete_operations.record_success(latency)
            self._metrics.consecutive_failures = 0
            self._metrics.is_healthy = True
            return result
        except Exception as e:
            self._metrics.delete_operations.record_failure(str(e))
            self._metrics.consecutive_failures += 1
            if self._metrics.consecutive_failures >= 3:
                self._metrics.is_healthy = False
            raise

    async def list_for_user(self, user_id: str = "default") -> List[Any]:
        """List integrations with metrics tracking."""
        start_time = time.time()
        try:
            result = await self._store.list_for_user(user_id)
            latency = time.time() - start_time
            self._metrics.list_operations.record_success(latency)
            self._metrics.active_integrations = len(result)
            self._metrics.consecutive_failures = 0
            self._metrics.is_healthy = True
            return result
        except Exception as e:
            self._metrics.list_operations.record_failure(str(e))
            self._metrics.consecutive_failures += 1
            if self._metrics.consecutive_failures >= 3:
                self._metrics.is_healthy = False
            raise

    async def list_all(self) -> List[Any]:
        """List all integrations with metrics tracking."""
        start_time = time.time()
        try:
            result = await self._store.list_all()
            latency = time.time() - start_time
            self._metrics.list_operations.record_success(latency)
            self._metrics.active_integrations = len(result)
            self._metrics.consecutive_failures = 0
            self._metrics.is_healthy = True
            return result
        except Exception as e:
            self._metrics.list_operations.record_failure(str(e))
            self._metrics.consecutive_failures += 1
            if self._metrics.consecutive_failures >= 3:
                self._metrics.is_healthy = False
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the store.

        Returns:
            Dict with health status and metrics summary
        """
        self._metrics.last_health_check = datetime.now(timezone.utc)

        try:
            # Try a lightweight operation
            await self._store.list_all()
            self._metrics.is_healthy = True
            self._metrics.consecutive_failures = 0
        except Exception as e:
            self._metrics.is_healthy = False
            logger.warning(f"Integration store health check failed: {e}")

        return {
            "healthy": self._metrics.is_healthy,
            "backend_type": self._metrics.backend_type,
            "last_check": self._metrics.last_health_check.isoformat(),
            "consecutive_failures": self._metrics.consecutive_failures,
            "active_integrations": self._metrics.active_integrations,
        }

    # Passthrough for other methods
    def __getattr__(self, name: str) -> Any:
        """Delegate to base store for untracked methods."""
        return getattr(self._store, name)


async def get_integration_metrics() -> Dict[str, Any]:
    """
    Get current integration store metrics.

    Returns:
        Dict with all metrics
    """
    return get_metrics().to_dict()


async def get_integration_health() -> Dict[str, Any]:
    """
    Get integration store health status.

    Returns:
        Dict with health info
    """
    metrics = get_metrics()
    return {
        "healthy": metrics.is_healthy,
        "backend_type": metrics.backend_type,
        "consecutive_failures": metrics.consecutive_failures,
        "last_health_check": (
            metrics.last_health_check.isoformat() if metrics.last_health_check else None
        ),
        "operations_summary": {
            "get_success_rate": metrics.get_operations.success_rate,
            "save_success_rate": metrics.save_operations.success_rate,
            "cache_hit_rate": metrics.cache_hit_rate,
        },
    }


__all__ = [
    "OperationMetrics",
    "IntegrationStoreMetrics",
    "InstrumentedIntegrationStore",
    "get_metrics",
    "reset_metrics",
    "track_operation",
    "get_integration_metrics",
    "get_integration_health",
]
