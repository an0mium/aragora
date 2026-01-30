# Resilience Patterns

Fault-tolerant patterns for graceful failure handling in API calls, agent interactions, and distributed systems.

## Overview

This package provides production-ready resilience patterns:

| Pattern | Module | Purpose |
|---------|--------|---------|
| Circuit Breaker | `circuit_breaker.py` | Prevent cascading failures |
| Circuit Breaker v2 | `circuit_breaker_v2.py` | Async-first with advanced config |
| Retry | `retry.py` | Exponential backoff with jitter |
| Timeout | `timeout.py` | Bound execution time |
| Health | `health.py` | Liveness and readiness probes |
| Metrics | `metrics.py` | Prometheus-compatible monitoring |
| Persistence | `persistence.py` | SQLite state persistence |

## Quick Start

```python
from aragora.resilience import (
    CircuitBreaker,
    get_circuit_breaker,
    with_retry,
    with_timeout,
    HealthChecker,
)

# Circuit breaker via registry
cb = get_circuit_breaker("anthropic-api", provider="anthropic")

if cb.can_proceed():
    try:
        result = await call_api()
        cb.record_success()
    except Exception:
        cb.record_failure()

# Decorator-based retry
@with_retry(max_attempts=3, base_delay=1.0)
async def unreliable_call():
    return await external_api()

# Timeout wrapper
@with_timeout(seconds=30.0)
async def bounded_operation():
    return await slow_operation()
```

## Circuit Breaker

Three-state pattern: CLOSED → OPEN → HALF-OPEN → CLOSED

```python
from aragora.resilience import CircuitBreaker

# Single-entity tracking
breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)

# Multi-entity tracking (e.g., multiple agents)
if breaker.is_available("agent-claude"):
    breaker.record_success("agent-claude")
else:
    breaker.record_failure("agent-claude")

# Context manager usage
async with breaker.protect("my-service"):
    await risky_operation()
```

### Circuit Breaker v2 (Async-First)

```python
from aragora.resilience import (
    CircuitBreakerConfig,
    with_circuit_breaker,
    get_v2_circuit_breaker,
)

# Configure per service
config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    cooldown_seconds=60,
    half_open_max_calls=3,
)

cb = get_v2_circuit_breaker("my-service", config)

# Or use decorator
@with_circuit_breaker("external-api")
async def call_external():
    return await api.request()
```

## Retry Patterns

Exponential backoff with configurable jitter:

```python
from aragora.resilience import (
    RetryConfig,
    RetryStrategy,
    JitterMode,
    with_retry,
)

# Full configuration
config = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=JitterMode.FULL,
    retryable_exceptions=(ConnectionError, TimeoutError),
)

@with_retry(config=config)
async def fetch_data():
    return await api.get("/data")

# Simple usage
@with_retry(max_attempts=3, base_delay=0.5)
async def quick_retry():
    return await api.ping()
```

## Timeout Patterns

```python
from aragora.resilience import with_timeout, timeout_context

# Decorator
@with_timeout(seconds=10.0)
async def time_bounded():
    return await slow_operation()

# Context manager
async with timeout_context(seconds=5.0):
    await operation_that_might_hang()
```

## Health Monitoring

```python
from aragora.resilience import (
    HealthChecker,
    HealthStatus,
    get_global_health_registry,
)

# Register health checks
registry = get_global_health_registry()

async def check_database():
    await db.ping()
    return HealthStatus.HEALTHY

registry.register("database", check_database)
registry.register("cache", check_redis)

# Get health report
report = await registry.check_all()
print(f"Overall: {report.status}")
for name, status in report.checks.items():
    print(f"  {name}: {status}")
```

## Persistence

Circuit breaker state survives restarts:

```python
from aragora.resilience import (
    init_circuit_breaker_persistence,
    persist_all_circuit_breakers,
    load_circuit_breakers,
)

# Initialize at startup
init_circuit_breaker_persistence("/path/to/state.db")
load_circuit_breakers()

# Persist periodically or on shutdown
persist_all_circuit_breakers()
```

## Metrics Integration

```python
from aragora.resilience import (
    set_metrics_callback,
    get_circuit_breaker_metrics,
    get_all_circuit_breakers_status,
)

# Custom metrics callback
def my_metrics_handler(circuit_name: str, state: int):
    prometheus_gauge.labels(circuit=circuit_name).set(state)

set_metrics_callback(my_metrics_handler)

# Query status
status = get_all_circuit_breakers_status()
for name, info in status.items():
    print(f"{name}: {info['state']} (failures={info['failure_count']})")
```

## Integration with Aragora

Used throughout the codebase for:

- **Agent calls**: Prevent hammering failing LLM providers
- **Database connections**: Graceful handling of pool exhaustion
- **External APIs**: Rate limit compliance and backoff
- **Knowledge Mound**: Resilient storage operations

See `aragora.resilience.py` (34KB) for the main integration point.
