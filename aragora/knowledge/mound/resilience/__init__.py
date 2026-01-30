"""
Knowledge Mound Persistence Resilience.

Provides production-hardening capabilities:
- Retry logic with exponential backoff for transient failures
- Explicit transaction boundaries for multi-table operations
- Connection health monitoring
- Circuit breaker integration for adapters
- Cache invalidation events
- SLO monitoring with Prometheus metrics
- Bulkhead isolation for adapter operations
- Timeout configuration for all operations

"Reliability is a feature."
"""

# Retry
from aragora.knowledge.mound.resilience.retry import (
    RetryConfig,
    RetryStrategy,
    with_retry,
    with_timeout,
)

# Transaction
from aragora.knowledge.mound.resilience.transaction import (
    DeadlockError,
    TransactionConfig,
    TransactionIsolation,
    TransactionManager,
)

# Health
from aragora.knowledge.mound.resilience.health import (
    ConnectionHealthMonitor,
    HealthStatus,
)

# Cache Invalidation
from aragora.knowledge.mound.resilience.cache_invalidation import (
    CacheInvalidationBus,
    CacheInvalidationEvent,
    get_invalidation_bus,
)

# Integrity
from aragora.knowledge.mound.resilience.integrity import (
    IntegrityCheckResult,
    IntegrityVerifier,
)

# Circuit Breaker
from aragora.knowledge.mound.resilience.circuit_breaker import (
    AdapterCircuitBreaker,
    AdapterCircuitBreakerConfig,
    AdapterCircuitState,
    AdapterCircuitStats,
    AdapterUnavailableError,
    HealthAwareCircuitBreaker,
    LatencyTracker,
    get_adapter_circuit_breaker,
    get_all_adapter_circuit_stats,
    reset_adapter_circuit_breaker,
    reset_all_adapter_circuit_breakers,
)

# SLO
from aragora.knowledge.mound.resilience.slo import (
    AdapterSLOConfig,
    check_adapter_slo,
    get_adapter_slo_config,
    record_adapter_slo_check,
    set_adapter_slo_config,
)

# Bulkhead
from aragora.knowledge.mound.resilience.bulkhead import (
    AdapterBulkhead,
    BulkheadConfig,
    BulkheadFullError,
    get_adapter_bulkhead,
    get_all_adapter_bulkhead_stats,
)

# Postgres Store
from aragora.knowledge.mound.resilience.postgres_store import (
    ResilientPostgresStore,
)

# Adapter Mixin + Combined Status
from aragora.knowledge.mound.resilience.adapter_mixin import (
    ResilientAdapterMixin,
    get_km_resilience_status,
)

__all__ = [
    # Retry
    "RetryStrategy",
    "RetryConfig",
    "with_retry",
    "with_timeout",
    # Transaction
    "TransactionIsolation",
    "TransactionConfig",
    "DeadlockError",
    "TransactionManager",
    # Health
    "HealthStatus",
    "ConnectionHealthMonitor",
    # Cache Invalidation
    "CacheInvalidationEvent",
    "CacheInvalidationBus",
    "get_invalidation_bus",
    # Integrity
    "IntegrityCheckResult",
    "IntegrityVerifier",
    # Circuit Breaker
    "AdapterCircuitState",
    "AdapterCircuitBreakerConfig",
    "AdapterCircuitStats",
    "AdapterCircuitBreaker",
    "HealthAwareCircuitBreaker",
    "LatencyTracker",
    "AdapterUnavailableError",
    "get_adapter_circuit_breaker",
    "get_all_adapter_circuit_stats",
    "reset_adapter_circuit_breaker",
    "reset_all_adapter_circuit_breakers",
    # SLO
    "AdapterSLOConfig",
    "get_adapter_slo_config",
    "set_adapter_slo_config",
    "check_adapter_slo",
    "record_adapter_slo_check",
    # Bulkhead
    "BulkheadConfig",
    "AdapterBulkhead",
    "BulkheadFullError",
    "get_adapter_bulkhead",
    "get_all_adapter_bulkhead_stats",
    # Postgres Store
    "ResilientPostgresStore",
    # Adapter Mixin
    "ResilientAdapterMixin",
    "get_km_resilience_status",
]
