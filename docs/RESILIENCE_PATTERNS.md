# Resilience Patterns Module

Unified resilience patterns for fault-tolerant systems in Aragora.

## Overview

The `aragora.resilience_patterns` module provides consolidated resilience patterns used across the codebase:

- **Circuit Breakers** - Failure isolation for external services
- **Retry Logic** - Configurable backoff strategies for transient failures
- **Timeout Management** - Bounded operations with async support
- **Health Monitoring** - Component health tracking

## Location

`aragora/resilience_patterns/`

## Installation

The module is part of the core Aragora package. No additional dependencies required.

## Components

### Circuit Breaker

Prevents cascading failures by opening the circuit after repeated failures.

```python
from aragora.resilience_patterns import (
    get_circuit_breaker,
    with_circuit_breaker,
    CircuitBreakerConfig,
    CircuitState,
)

# Get or create a named circuit breaker
cb = get_circuit_breaker("stripe_api", failure_threshold=5, cooldown_seconds=60)

# Check if circuit allows execution
if cb.can_execute():
    try:
        result = await call_stripe()
        cb.record_success()
    except Exception as e:
        cb.record_failure(e)
        raise

# Decorator-based usage
@with_circuit_breaker("my_service")
async def call_service():
    return await http_client.get("/api/data")
```

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Failures before opening circuit |
| `cooldown_seconds` | 30 | Time before attempting recovery |
| `success_threshold` | 2 | Successes needed to close circuit |

**Circuit States:**
- `CLOSED` - Normal operation, requests pass through
- `OPEN` - Circuit tripped, requests fail immediately
- `HALF_OPEN` - Testing recovery, limited requests allowed

### Retry Logic

Configurable retry with multiple backoff strategies.

```python
from aragora.resilience_patterns import (
    with_retry,
    RetryConfig,
    RetryStrategy,
    JitterMode,
)

# Configure retry behavior
config = RetryConfig(
    max_retries=3,
    base_delay=0.5,
    max_delay=30.0,
    strategy="exponential",  # or "linear", "constant", "fibonacci"
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError),
)

# Decorator usage
@with_retry(config)
async def flaky_api_call():
    return await external_service.fetch()

# Or with simple defaults
@with_retry(RetryConfig(max_retries=3))
async def simple_retry():
    ...
```

**Strategies:**
- `exponential` - Delays grow exponentially (recommended)
- `linear` - Delays grow linearly
- `constant` - Fixed delay between retries
- `fibonacci` - Fibonacci sequence delays

**Jitter Modes:**
- `additive` - Add random jitter
- `multiplicative` - Multiply by random factor
- `full` - Full randomization

### Timeout Management

Bounded operations with configurable timeouts.

```python
from aragora.resilience_patterns import (
    with_timeout,
    TimeoutConfig,
    asyncio_timeout,
)

# Decorator usage
@with_timeout(5.0)  # 5 second timeout
async def bounded_operation():
    return await slow_service.call()

# Context manager
async with asyncio_timeout(10.0):
    result = await long_running_task()

# With configuration
config = TimeoutConfig(
    timeout_seconds=30.0,
    on_timeout=lambda: logger.warning("Operation timed out"),
)

@with_timeout(config)
async def configured_timeout():
    ...
```

### Health Monitoring

Track component health status.

```python
from aragora.resilience_patterns import (
    HealthChecker,
    HealthStatus,
    HealthReport,
)

# Create health checker
checker = HealthChecker(name="database")

# Record health status
checker.record_success()
checker.record_failure("Connection refused")

# Get current status
status: HealthStatus = checker.get_status()
print(f"Healthy: {status.is_healthy}")
print(f"Last error: {status.last_error}")

# Generate health report
report: HealthReport = checker.get_report()
```

## Prometheus Metrics

The module exports Prometheus metrics automatically:

| Metric | Type | Description |
|--------|------|-------------|
| `aragora_circuit_breaker_state` | Gauge | Current circuit state (0=closed, 1=open, 2=half-open) |
| `aragora_retry_attempts_total` | Counter | Total retry attempts |
| `aragora_retry_exhausted_total` | Counter | Retries that exhausted all attempts |
| `aragora_timeout_total` | Counter | Operations that timed out |
| `aragora_health_status` | Gauge | Component health (0=unhealthy, 1=healthy) |

## Migration from Legacy

### From `aragora.resilience`

The old `aragora/resilience.py` module is still available but new code should use `aragora.resilience_patterns`:

```python
# OLD (deprecated)
from aragora.resilience import CircuitBreaker
cb = CircuitBreaker("service", max_failures=5)

# NEW (recommended)
from aragora.resilience_patterns import get_circuit_breaker
cb = get_circuit_breaker("service", failure_threshold=5)
```

### From `aragora.knowledge.mound.resilience`

Knowledge Mound-specific resilience patterns have been consolidated:

```python
# OLD
from aragora.knowledge.mound.resilience import with_retry

# NEW
from aragora.resilience_patterns import with_retry, RetryConfig
```

## Best Practices

1. **Name your circuit breakers** - Use descriptive names like `"stripe_payments"` or `"github_api"`

2. **Configure for your SLAs** - Set timeouts based on expected response times

3. **Use exponential backoff** - Prevents thundering herd on service recovery

4. **Add jitter** - Spreads out retry attempts across clients

5. **Log failures** - Circuit breaker state changes are important operational events

6. **Monitor metrics** - Use the Prometheus metrics to set up alerts

## Examples

### Payment Handler with Full Resilience

```python
from aragora.resilience_patterns import (
    get_circuit_breaker,
    with_retry,
    RetryConfig,
)

# Circuit breaker for payment provider
_stripe_cb = get_circuit_breaker("stripe_payments", failure_threshold=5, cooldown_seconds=60)

# Retry config for transient failures
_retry_config = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    strategy="exponential",
    retryable_exceptions=(ConnectionError, TimeoutError),
)

async def charge_card(amount: int, token: str) -> dict:
    if not _stripe_cb.can_execute():
        raise ServiceUnavailableError("Payment service temporarily unavailable")

    @with_retry(_retry_config)
    async def _do_charge():
        return await stripe.charges.create(amount=amount, source=token)

    try:
        result = await _do_charge()
        _stripe_cb.record_success()
        return result
    except Exception as e:
        _stripe_cb.record_failure(e)
        raise
```

### Database with Health Monitoring

```python
from aragora.resilience_patterns import HealthChecker, with_timeout

db_health = HealthChecker("postgres")

@with_timeout(5.0)
async def query_database(sql: str) -> list:
    try:
        result = await db.execute(sql)
        db_health.record_success()
        return result
    except Exception as e:
        db_health.record_failure(str(e))
        raise
```

## See Also

- [RESILIENCE.md](./RESILIENCE.md) - Legacy resilience module documentation
- [ENTERPRISE_FEATURES.md](./ENTERPRISE_FEATURES.md) - Enterprise resilience features
- [OBSERVABILITY.md](./OBSERVABILITY.md) - Metrics and monitoring
