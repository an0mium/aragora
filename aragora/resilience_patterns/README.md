# Resilience Patterns Module

Unified resilience patterns for the Aragora codebase. This module provides production-hardened patterns for building fault-tolerant systems.

## Architecture Overview

```
aragora/resilience_patterns/
├── __init__.py          # Public API exports
├── circuit_breaker.py   # Circuit breaker pattern with global registry
├── retry.py             # Retry logic with configurable backoff
├── timeout.py           # Timeout management for async/sync operations
├── health.py            # Health checking and reporting
└── metrics.py           # Prometheus metrics callbacks
```

## Quick Start

```python
from aragora.resilience_patterns import (
    # Circuit Breaker
    get_circuit_breaker,
    with_circuit_breaker,
    CircuitBreakerConfig,

    # Retry
    with_retry,
    RetryConfig,

    # Timeout
    with_timeout,

    # Health
    HealthChecker,
)

# Using circuit breaker decorator
@with_circuit_breaker("external_api")
async def call_external_api():
    return await http_client.get("/api/data")

# Using retry decorator
@with_retry(RetryConfig(max_retries=3, strategy="exponential"))
async def flaky_operation():
    return await unreliable_service.call()

# Using timeout decorator
@with_timeout(5.0)
async def bounded_operation():
    return await slow_service.fetch()

# Manual circuit breaker usage
cb = get_circuit_breaker("my_service", failure_threshold=5)
if cb.can_execute():
    try:
        result = await my_service.call()
        cb.record_success()
    except Exception as e:
        cb.record_failure(e)
        raise
```

## Circuit Breaker

The circuit breaker pattern prevents cascading failures by failing fast when a service is unhealthy.

### States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failing fast, requests are rejected immediately
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Configuration

```python
from aragora.resilience_patterns import CircuitBreakerConfig, BaseCircuitBreaker

config = CircuitBreakerConfig(
    failure_threshold=5,        # Failures before opening
    success_threshold=3,        # Successes to close from half-open
    cooldown_seconds=60.0,      # Time in open state before half-open
    half_open_max_requests=3,   # Max concurrent requests in half-open
)

cb = BaseCircuitBreaker("my_service", config)
```

### Global Registry

Circuit breakers are registered globally for consistent state across components:

```python
from aragora.resilience_patterns import (
    get_circuit_breaker,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
)

# Get or create (returns existing if already created)
cb = get_circuit_breaker("agent_claude", failure_threshold=8)

# List all circuit breakers
all_cbs = get_all_circuit_breakers()

# Reset all for testing
reset_all_circuit_breakers()
```

## Retry Pattern

Configurable retry logic with multiple backoff strategies.

### Strategies

- **exponential**: 2^n * base_delay (default)
- **linear**: n * base_delay
- **constant**: base_delay always

### Configuration

```python
from aragora.resilience_patterns import RetryConfig, with_retry

config = RetryConfig(
    max_retries=3,
    base_delay=0.1,
    max_delay=10.0,
    strategy="exponential",
    jitter=True,  # Prevent thundering herd
    retryable_exceptions=(ConnectionError, TimeoutError),
)

@with_retry(config)
async def my_operation():
    ...
```

## Timeout Pattern

Enforce time bounds on operations.

```python
from aragora.resilience_patterns import with_timeout, TimeoutConfig

# Simple decorator
@with_timeout(5.0)
async def bounded_operation():
    ...

# With fallback
@with_timeout(5.0, fallback=lambda: [])
async def get_items_with_fallback():
    ...
```

## Health Checking

Monitor service health with configurable checks.

```python
from aragora.resilience_patterns import HealthChecker, HealthStatus

checker = HealthChecker(
    name="database",
    check_interval=30.0,
    failure_threshold=3,
)

# Register health check
async def check_db():
    await db.execute("SELECT 1")

checker.register_check("postgres", check_db)

# Get health report
report = await checker.check_all()
print(report.status)  # HealthStatus.HEALTHY
```

## Metrics Integration

Prometheus metrics for observability.

```python
from aragora.resilience_patterns import create_metrics_callbacks

# Register callbacks for metrics
callbacks = create_metrics_callbacks(namespace="aragora")

# Callbacks include:
# - circuit_breaker_state_changed(name, old_state, new_state)
# - retry_attempt(operation, attempt, delay)
# - retry_exhausted(operation, attempts)
# - timeout_occurred(operation, duration)
# - health_status_changed(name, old_status, new_status)
```

## Architecture Decision: Two Resilience Systems

The codebase has two resilience implementations:

### 1. General Resilience (`aragora/resilience_patterns/`)
- **Used by**: API agents, CLI agents, general services
- **Features**: Circuit breaker, retry, timeout, health monitoring
- **Pattern**: Decorators and context managers

### 2. KM Specialized Resilience (`aragora/knowledge/mound/resilience.py`)
- **Used by**: Knowledge Mound adapters only
- **Features**: Circuit breaker + Bulkhead + SLO monitoring
- **Pattern**: `ResilientAdapterMixin` with `_resilient_call()` context manager

**Rationale**: KM adapters require specialized features (bulkhead isolation, adapter-specific SLO monitoring) that the general module doesn't provide. Both systems are production-hardened and should be used in their respective domains.

## Migration Guide

### From `aragora.resilience` (legacy)

```python
# Before (legacy)
from aragora.resilience import CircuitBreaker, get_circuit_breaker

# After (new)
from aragora.resilience_patterns import BaseCircuitBreaker, get_circuit_breaker

# Method compatibility
cb.can_proceed()  # Works (alias for can_execute())
cb.can_execute()  # Preferred
```

### Type Changes

| Legacy | New |
|--------|-----|
| `CircuitBreaker` | `BaseCircuitBreaker` |
| `failures` property | `failures` property (same) |
| `get_status()` | `get_status()` (same) |

## Testing

```bash
# Run resilience pattern tests
pytest tests/resilience_patterns/ -v

# Expected: 126+ passed
```

## See Also

- `aragora/knowledge/mound/resilience.py` - KM-specific resilience patterns
- `docs/STATUS.md` - Feature status documentation
- `tests/resilience_patterns/` - Test suite
