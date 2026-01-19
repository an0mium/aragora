# Rate Limiting (Developer Guide)

> **Audience:** This is a **developer/maintainer guide** for understanding and extending the rate limiting implementation.
> For API consumer documentation (handling rate limits, best practices), see [API_RATE_LIMITS.md](./API_RATE_LIMITS.md).

Aragora implements comprehensive rate limiting to protect against abuse and ensure fair resource allocation. This document covers the rate limiting architecture, configuration, and usage.

## Table of Contents

- [Overview](#overview)
- [Token Bucket Algorithm](#token-bucket-algorithm)
- [Rate Limiter Types](#rate-limiter-types)
  - [IP-Based Rate Limiter](#ip-based-rate-limiter)
  - [User Rate Limiter](#user-rate-limiter)
  - [Tier Rate Limiter](#tier-rate-limiter)
- [Default Endpoint Limits](#default-endpoint-limits)
- [Usage](#usage)
  - [Decorator](#decorator)
  - [Manual Check](#manual-check)
- [HTTP Headers](#http-headers)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Reference](#api-reference)

---

## Overview

Rate limiting in Aragora uses a **token bucket** algorithm that provides:

- **Burst capacity**: Handle traffic spikes gracefully
- **Smooth rate limiting**: Tokens refill continuously
- **Fair allocation**: LRU eviction prevents memory bloat
- **Multiple key types**: IP, user, endpoint, or tier-based limits

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RateLimiterRegistry                       │
├─────────────────────────────────────────────────────────────┤
│  Default Limiter (preconfigured endpoints)                   │
│  Named Limiters (custom configurations)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  RateLimiter  │ │ UserRateLimiter│ │TierRateLimiter│
│  (IP-based)   │ │ (User-based)   │ │ (Tier-based)  │
└───────────────┘ └───────────────┘ └───────────────┘
        │               │               │
        ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    TokenBucket (per key)                     │
│  - Thread-safe token consumption                             │
│  - Automatic refill based on elapsed time                    │
│  - LRU eviction when max entries reached                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Token Bucket Algorithm

Each client/user gets a "bucket" of tokens that refills over time.

### How It Works

1. **Initial state**: Bucket starts full (burst capacity)
2. **Request arrives**: Consume 1 token from bucket
3. **Token available**: Request allowed, token consumed
4. **No tokens**: Request denied with 429 status
5. **Refill**: Tokens continuously refill at configured rate

### Example

```
Configuration: 60 req/min, burst 120

Time 0:00 - Bucket has 120 tokens
  → 100 requests arrive, bucket now has 20 tokens

Time 0:30 - 30 tokens refilled (30 seconds × 1 token/sec)
  → Bucket now has 50 tokens

Time 1:00 - 30 more tokens refilled
  → Bucket now has 80 tokens (capped at 120)
```

### Implementation

```python
class TokenBucket:
    def __init__(self, rate_per_minute: float, burst_size: int | None = None):
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size or int(rate_per_minute * 2)  # 2x burst
        self.tokens = float(self.burst_size)  # Start full
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        # Refill based on elapsed time
        now = time.monotonic()
        elapsed_minutes = (now - self.last_refill) / 60.0
        refill_amount = elapsed_minutes * self.rate_per_minute
        self.tokens = min(self.burst_size, self.tokens + refill_amount)
        self.last_refill = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

---

## Rate Limiter Types

### IP-Based Rate Limiter

Default rate limiter keyed by client IP address.

```python
from aragora.server.middleware.rate_limit import RateLimiter, get_rate_limiter

# Get default limiter
limiter = get_rate_limiter()

# Check if request allowed
result = limiter.allow(
    client_ip="192.168.1.1",
    endpoint="/api/debates",
)

if not result.allowed:
    print(f"Rate limited! Retry after {result.retry_after}s")
```

#### Features

- Automatic X-Forwarded-For handling (for proxies)
- Per-endpoint configuration
- Wildcard endpoint matching (`/api/debates/*`)
- LRU eviction to prevent memory bloat

### User Rate Limiter

Rate limiting by authenticated user ID.

```python
from aragora.server.middleware.rate_limit import (
    UserRateLimiter,
    get_user_rate_limiter,
    check_user_rate_limit,
)

limiter = get_user_rate_limiter()

# Check rate limit for specific action
result = limiter.allow(
    user_id="user-123",
    action="debate_create",  # 10 req/min
)

# Or use helper (extracts user from request)
result = check_user_rate_limit(handler, user_store, action="vote")
```

#### Action Limits

| Action | Limit (req/min) |
|--------|-----------------|
| `default` | 60 |
| `debate_create` | 10 |
| `vote` | 30 |
| `agent_call` | 120 |
| `export` | 5 |
| `admin` | 300 |

### Tier Rate Limiter

Rate limiting based on subscription tier.

```python
from aragora.server.middleware.rate_limit import (
    TierRateLimiter,
    get_tier_rate_limiter,
    check_tier_rate_limit,
)

limiter = get_tier_rate_limiter()

# Direct check
result = limiter.allow(
    client_key="user-123",
    tier="professional",  # 200 req/min
)

# Or use helper (extracts tier from user's org)
result = check_tier_rate_limit(handler, user_store)
```

#### Tier Limits

| Tier | Rate (req/min) | Burst |
|------|----------------|-------|
| `free` | 10 | 60 |
| `starter` | 50 | 100 |
| `professional` | 200 | 400 |
| `enterprise` | 1000 | 2000 |

---

## Default Endpoint Limits

The default rate limiter comes preconfigured with endpoint-specific limits:

| Endpoint | Limit (req/min) | Key Type |
|----------|-----------------|----------|
| `/api/debates` | 30 | IP |
| `/api/debates/*` | 60 | IP |
| `/api/debates/*/fork` | 5 | IP |
| `/api/agent/*` | 120 | IP |
| `/api/leaderboard*` | 60 | IP |
| `/api/pulse/*` | 30 | IP |
| `/api/memory/continuum/cleanup` | 2 | IP |
| `/api/memory/*` | 60 | IP |
| `/api/debates/*/broadcast` | 3 | IP |
| `/api/probes/*` | 10 | IP |
| `/api/verification/*` | 10 | IP |
| `/api/video/*` | 2 | IP |

---

## Usage

### Decorator

Apply rate limiting to endpoint handlers:

```python
from aragora.server.middleware.rate_limit import rate_limit, user_rate_limit

# IP-based limiting
@rate_limit(requests_per_minute=30)
def _create_debate(self, handler):
    ...

# With custom burst
@rate_limit(requests_per_minute=10, burst=20)
def _expensive_operation(self, handler):
    ...

# Share limiter across handlers
@rate_limit(requests_per_minute=5, limiter_name="expensive_ops")
def _operation_a(self, handler):
    ...

@rate_limit(requests_per_minute=5, limiter_name="expensive_ops")
def _operation_b(self, handler):
    ...

# User-based limiting
@user_rate_limit(action="debate_create")
def _create_debate(self, handler):
    ...
```

### Manual Check

For more control:

```python
from aragora.server.middleware.rate_limit import (
    get_rate_limiter,
    rate_limit_headers,
)
from aragora.server.handlers.base import error_response

def handle_request(self, handler):
    limiter = get_rate_limiter()
    client_ip = limiter.get_client_key(handler)

    result = limiter.allow(client_ip, endpoint="/api/debates")

    if not result.allowed:
        return error_response(
            "Rate limit exceeded",
            429,
            headers=rate_limit_headers(result),
        )

    # Process request...
    response = self._do_work()

    # Add headers to successful response
    if hasattr(response, "headers"):
        response.headers.update(rate_limit_headers(result))

    return response
```

---

## HTTP Headers

Rate limit information is included in response headers:

### Request Headers

```http
X-RateLimit-Limit: 30        # Max requests per minute
X-RateLimit-Remaining: 25    # Remaining requests in window
```

### Rate Limited Response (429)

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 5               # Seconds until next request allowed
X-RateLimit-Reset: 1704672045 # Unix timestamp when limit resets
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 0
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_RATE_LIMIT` | Default requests per minute | 60 |
| `ARAGORA_IP_RATE_LIMIT` | Per-IP requests per minute | 120 |
| `ARAGORA_BURST_MULTIPLIER` | Burst = rate × multiplier | 2.0 |

### Custom Configuration

```python
from aragora.server.middleware.rate_limit import RateLimiter

# Create custom limiter
limiter = RateLimiter(
    default_limit=60,      # Default req/min
    ip_limit=120,          # IP-specific limit
    cleanup_interval=300,  # Stats logging interval
    max_entries=10000,     # Max buckets before LRU eviction
)

# Configure specific endpoints
limiter.configure_endpoint(
    endpoint="/api/expensive",
    requests_per_minute=5,
    burst_size=10,
    key_type="combined",  # IP + endpoint
)
```

### Key Types

| Type | Description |
|------|-------------|
| `ip` | Keyed by client IP address (default) |
| `token` | Keyed by auth token |
| `endpoint` | Global per-endpoint limit |
| `combined` | IP + endpoint combination |

---

## Testing

### Reset Rate Limiters

```python
from aragora.server.middleware.rate_limit import reset_rate_limiters

def setup_test():
    reset_rate_limiters()  # Clear all state
```

### Cleanup Stale Entries

```python
from aragora.server.middleware.rate_limit import cleanup_rate_limiters

# Remove entries older than 5 minutes
removed = cleanup_rate_limiters(max_age_seconds=300)
print(f"Removed {removed} stale entries")
```

### Get Statistics

```python
limiter = get_rate_limiter()
stats = limiter.get_stats()
print(stats)
# {
#   "ip_buckets": 42,
#   "token_buckets": 10,
#   "endpoint_buckets": {"api/debates": 15},
#   "configured_endpoints": ["/api/debates", ...],
#   "default_limit": 60,
#   "ip_limit": 120,
# }
```

---

## API Reference

### RateLimitResult

```python
@dataclass
class RateLimitResult:
    allowed: bool          # Whether request is allowed
    remaining: int = 0     # Remaining requests
    limit: int = 0         # Limit for this key
    retry_after: float = 0 # Seconds until next request allowed
    key: str = ""          # The rate limit key used
```

### RateLimitConfig

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst_size: int | None = None
    key_type: str = "ip"    # "ip", "token", "endpoint", "combined"
    enabled: bool = True
```

### Functions

```python
# Get/create rate limiters
get_rate_limiter(name: str = "_default") -> RateLimiter
get_user_rate_limiter() -> UserRateLimiter
get_tier_rate_limiter() -> TierRateLimiter

# Check rate limits
check_user_rate_limit(handler, user_store, action) -> RateLimitResult
check_tier_rate_limit(handler, user_store) -> RateLimitResult

# Utilities
rate_limit_headers(result: RateLimitResult) -> Dict[str, str]
cleanup_rate_limiters(max_age_seconds: int = 300) -> int
reset_rate_limiters() -> None
```

### Decorators

```python
@rate_limit(
    requests_per_minute: int = 30,
    burst: int | None = None,
    limiter_name: str | None = None,
    key_type: str = "ip",
)

@user_rate_limit(
    action: str = "default",
    user_store_factory: Callable | None = None,
)
```

---

## See Also

- [Security](./SECURITY.md) - Overall security documentation
- [API Reference](./API_REFERENCE.md) - API endpoint documentation
- [Environment](./ENVIRONMENT.md) - Environment variables
