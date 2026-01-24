# Connector Error Handling Standards

This guide defines the standard error handling patterns for all Aragora connectors. Following these patterns ensures consistent resilience, observability, and user experience across all integrations.

## Core Principles

1. **Fail Gracefully** - Never crash; return structured error responses
2. **Retry Intelligently** - Use exponential backoff for transient failures
3. **Circuit Break** - Protect downstream systems from cascading failures
4. **Log Consistently** - Structured logging with correlation IDs
5. **Metric Everything** - Expose failure rates, latencies, circuit breaker states

## Standard Patterns

### Chat Connectors (ChatPlatformConnector)

Chat connectors should use the base class's `_http_request` helper for all HTTP operations:

```python
# GOOD: Uses standardized HTTP helper with retry, timeout, circuit breaker
async def send_message(self, channel_id: str, text: str, **kwargs) -> SendMessageResponse:
    success, data, error = await self._http_request(
        method="POST",
        url=f"{API_BASE}/channels/{channel_id}/messages",
        headers=self._get_headers(),
        json={"content": text},
        operation="send_message",
    )
    if success and data:
        return SendMessageResponse(success=True, message_id=data.get("id"))
    return SendMessageResponse(success=False, error=error)

# BAD: Direct HTTP calls without resilience
async def send_message(self, channel_id: str, text: str, **kwargs) -> SendMessageResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)  # No retry, no circuit breaker
        return SendMessageResponse(success=True, message_id=response.json()["id"])
```

#### Base Class Features

The `ChatPlatformConnector` base class provides:

| Method | Purpose |
|--------|---------|
| `_http_request()` | HTTP with retry, timeout, circuit breaker |
| `_with_retry()` | Generic async retry wrapper |
| `_check_circuit_breaker()` | Check if requests allowed |
| `_record_success()` | Record success for circuit breaker |
| `_record_failure()` | Record failure for circuit breaker |
| `_is_retryable_status_code()` | Identifies 429, 500-504 as retryable |

#### Configuration

```python
connector = MyConnector(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,     # Failures before opening
    circuit_breaker_cooldown=60.0,   # Seconds before retry
    request_timeout=30.0,            # HTTP timeout
)
```

### Enterprise Connectors (EnterpriseConnector)

Enterprise connectors should follow similar patterns with the `_make_request` helper:

```python
async def _make_request(
    self,
    method: str,
    path: str,
    data: Optional[dict] = None,
    **kwargs
) -> ConnectorResult:
    """Make authenticated request with retry and circuit breaker."""
    # 1. Check circuit breaker
    if not self._circuit_breaker.can_proceed():
        return ConnectorResult.failure("Service temporarily unavailable")

    # 2. Ensure token freshness
    await self._ensure_token_valid()

    # 3. Execute with retry
    for attempt in range(self.max_retries):
        try:
            response = await self._client.request(method, path, json=data)

            if self._is_retryable(response.status_code):
                await self._backoff(attempt)
                continue

            self._circuit_breaker.record_success()
            return ConnectorResult.success(response.json())

        except Exception as e:
            self._circuit_breaker.record_failure()
            if attempt == self.max_retries - 1:
                return ConnectorResult.failure(str(e))
            await self._backoff(attempt)
```

## Retryable Status Codes

These status codes should trigger automatic retry with backoff:

| Code | Meaning | Action |
|------|---------|--------|
| 429 | Too Many Requests | Retry with `Retry-After` header or backoff |
| 500 | Internal Server Error | Retry with backoff |
| 502 | Bad Gateway | Retry with backoff |
| 503 | Service Unavailable | Retry with backoff |
| 504 | Gateway Timeout | Retry with backoff |

Non-retryable errors (return immediately):

| Code | Meaning | Action |
|------|---------|--------|
| 400 | Bad Request | Return error (client issue) |
| 401 | Unauthorized | Refresh token or return error |
| 403 | Forbidden | Return error (permission issue) |
| 404 | Not Found | Return error (resource missing) |
| 422 | Unprocessable Entity | Return error (validation failed) |

## Exponential Backoff

Use this formula for retry delays:

```python
delay = min(base_delay * (2 ** attempt), max_delay)
jitter = random.uniform(0, delay * 0.1)
total_delay = delay + jitter
```

Default parameters:
- `base_delay`: 1.0 seconds
- `max_delay`: 30.0 seconds
- `max_retries`: 3

## Circuit Breaker States

```
CLOSED (healthy)
    ↓ failure_threshold failures
OPEN (blocking requests)
    ↓ cooldown_seconds elapsed
HALF-OPEN (testing recovery)
    ↓ success → CLOSED
    ↓ failure → OPEN
```

Configuration:
- `failure_threshold`: 5 (failures before opening)
- `cooldown_seconds`: 60.0 (time before recovery attempt)

## Error Response Format

All connectors should return structured error information:

```python
@dataclass
class ConnectorResult:
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    retryable: bool = False

    @classmethod
    def failure(cls, error: str, code: str = None, retryable: bool = False):
        return cls(success=False, error=error, error_code=code, retryable=retryable)

    @classmethod
    def success(cls, data: dict):
        return cls(success=True, data=data)
```

## Logging Standards

Use structured logging with consistent fields:

```python
logger.warning(
    f"{self.platform_name} {operation} failed "
    f"(attempt {attempt}/{max_retries}): {error}",
    extra={
        "connector": self.platform_name,
        "operation": operation,
        "attempt": attempt,
        "max_retries": max_retries,
        "status_code": response.status_code,
        "correlation_id": get_correlation_id(),
    }
)
```

## Metrics to Emit

Each connector should emit these Prometheus metrics:

```python
# Request metrics
platform_requests_total{platform, operation, status}
platform_request_latency_seconds{platform, operation}

# Circuit breaker metrics
platform_circuit_breaker_state{platform}  # 0=closed, 1=open, 2=half-open

# Retry metrics
platform_retry_total{platform, operation}
platform_retry_success_total{platform, operation}

# Rate limiting
platform_rate_limit_total{platform, result}  # allowed/blocked
```

## Platform-Specific Considerations

### Discord

- Interaction callbacks have 3-second window - use fewer retries
- Rate limits return `Retry-After` header - honor it
- Webhooks require Ed25519 signature verification

### Slack

- Rate limits are per-method - track separately
- Socket mode has different resilience needs than REST
- Workspace-level tokens require different scopes

### Microsoft Teams

- Uses OAuth 2.0 with refresh tokens
- Activity feed has strict rate limits
- Proactive messaging requires service URL storage

### Telegram

- Bot API has global rate limits
- Long polling needs different timeout handling
- File downloads have size limits

## Migration Guide

To migrate an existing connector to the standard pattern:

1. **Identify HTTP calls** - Find all direct `httpx` or `aiohttp` usage
2. **Replace with helper** - Use `_http_request()` or `_make_request()`
3. **Add circuit breaker** - Initialize in `__init__` if not inherited
4. **Update return types** - Use `ConnectorResult` or platform-specific response
5. **Add metrics** - Instrument success/failure/latency
6. **Update tests** - Mock `request()` instead of specific HTTP methods

### Example Migration

Before:
```python
async def get_user(self, user_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API}/users/{user_id}")
        response.raise_for_status()
        return response.json()
```

After:
```python
async def get_user(self, user_id: str) -> Optional[dict]:
    success, data, error = await self._http_request(
        method="GET",
        url=f"{API}/users/{user_id}",
        headers=self._get_headers(),
        operation="get_user",
    )
    if success:
        return data
    logger.warning(f"Failed to get user {user_id}: {error}")
    return None
```

## Testing Error Handling

Test files should cover:

```python
class TestConnectorResilience:
    async def test_retry_on_transient_error(self):
        """Verify retry on 503 with eventual success."""

    async def test_circuit_breaker_opens(self):
        """Verify circuit opens after threshold failures."""

    async def test_circuit_breaker_recovery(self):
        """Verify circuit recovers after cooldown."""

    async def test_timeout_handling(self):
        """Verify graceful timeout handling."""

    async def test_rate_limit_backoff(self):
        """Verify 429 triggers appropriate backoff."""
```

## Related Documentation

- [OBSERVABILITY.md](./OBSERVABILITY.md) - Metrics and monitoring
- [ENTERPRISE_FEATURES.md](./ENTERPRISE_FEATURES.md) - Enterprise connector capabilities
- Base classes:
  - `aragora/connectors/chat/base.py` - Chat connector base
  - `aragora/connectors/enterprise/base.py` - Enterprise connector base
  - `aragora/resilience.py` - Circuit breaker implementation
