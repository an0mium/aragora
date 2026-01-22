---
title: API Rate Limiting
description: API Rate Limiting
---

# API Rate Limiting

This document describes the rate limiting policies for the Aragora API.

## Overview

Rate limiting protects the API from abuse and ensures fair usage across all clients. Aragora implements tiered rate limits based on authentication level and endpoint type.

## Rate Limit Tiers

### Anonymous (Unauthenticated)

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Read endpoints | 60 requests | 1 minute |
| Write endpoints | 10 requests | 1 minute |
| WebSocket connections | 2 | concurrent |

### Authenticated (API Key)

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Read endpoints | 1000 requests | 1 minute |
| Write endpoints | 100 requests | 1 minute |
| Debate creation | 20 debates | 1 hour |
| WebSocket connections | 10 | concurrent |

### Premium/Enterprise

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Read endpoints | 10000 requests | 1 minute |
| Write endpoints | 1000 requests | 1 minute |
| Debate creation | Unlimited | - |
| WebSocket connections | 100 | concurrent |

## Endpoint-Specific Limits

### High-Cost Endpoints

These endpoints have stricter limits due to computational cost:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `POST /api/debates` | 20/hour | per API key |
| `POST /api/debates/\{id\}/analyze` | 10/hour | per API key |
| `POST /api/agents/train` | 5/day | per API key |
| `POST /api/knowledge/ingest` | 100/hour | per API key |

### Bulk Operations

| Endpoint | Limit | Window |
|----------|-------|--------|
| `POST /api/batch/*` | 10/hour | per API key |
| `GET /api/export/*` | 5/hour | per API key |

## Response Headers

All API responses include rate limit headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640000000
X-RateLimit-Policy: authenticated
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |
| `X-RateLimit-Policy` | Active rate limit policy |

## Rate Limit Exceeded Response

When rate limit is exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 30

{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Please retry after 30 seconds.",
  "retry_after": 30,
  "limit": 1000,
  "reset_at": "2026-01-20T12:00:00Z"
}
```

## Implementation Details

### Token Bucket Algorithm

Aragora uses a token bucket algorithm with Redis for distributed rate limiting:

```python
# Rate limiter configuration
RATE_LIMIT_CONFIG = {
    "anonymous": {
        "read": {"tokens": 60, "interval": 60},
        "write": {"tokens": 10, "interval": 60},
    },
    "authenticated": {
        "read": {"tokens": 1000, "interval": 60},
        "write": {"tokens": 100, "interval": 60},
    },
    "premium": {
        "read": {"tokens": 10000, "interval": 60},
        "write": {"tokens": 1000, "interval": 60},
    },
}
```

### Redis Key Structure

```
ratelimit:\{api_key\}:\{endpoint_type\}:\{window\}
```

Example:
```
ratelimit:sk_abc123:read:1640000000
```

## Best Practices

### Client Implementation

1. **Respect Retry-After**: Always wait the specified time before retrying
2. **Implement exponential backoff**: For repeated 429s, increase wait time
3. **Cache responses**: Reduce API calls by caching read responses
4. **Batch requests**: Use batch endpoints when available

### Example: Python Client

```python
import time
import requests
from tenacity import retry, wait_exponential, retry_if_result

def is_rate_limited(response):
    return response.status_code == 429

@retry(
    retry=retry_if_result(is_rate_limited),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def api_request(url, **kwargs):
    response = requests.get(url, **kwargs)
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 30))
        time.sleep(retry_after)
    return response
```

### Example: JavaScript Client

```javascript
async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);

  if (response.status === 429) {
    const retryAfter = parseInt(response.headers.get('Retry-After') || '30');
    await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
    return apiRequest(url, options);  // Retry
  }

  return response;
}
```

## Monitoring

### Prometheus Metrics

```
# Rate limit hits
aragora_ratelimit_hits_total{policy="authenticated", endpoint_type="read"}

# Rate limit rejections
aragora_ratelimit_rejected_total{policy="authenticated", endpoint_type="read"}

# Current token count
aragora_ratelimit_tokens_remaining{api_key_hash="xxx"}
```

### Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Rejection Rate | >10% requests rejected | Warning |
| Single Client Abuse | >50% of capacity | Warning |
| Distributed Attack | Many clients at limit | Critical |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiting |
| `RATE_LIMIT_REDIS_URL` | - | Redis URL for distributed limiting |
| `RATE_LIMIT_DEFAULT_TIER` | `anonymous` | Default tier for unauth requests |

### API Key Tier Override

Admins can set custom limits per API key:

```python
# Admin API
POST /api/admin/rate-limits
{
  "api_key": "sk_xxx",
  "tier": "premium",
  "custom_limits": {
    "debates_per_hour": 50
  }
}
```

## Exemptions

The following are exempt from rate limiting:

1. Health check endpoint (`/api/health`)
2. Metrics endpoint (`/metrics`)
3. Internal service-to-service calls (via service mesh)
4. Whitelisted IP addresses (configurable)

## Troubleshooting

### Sudden Rate Limit Issues

1. Check for leaked API keys
2. Review client implementation for request loops
3. Check for WebSocket reconnection storms
4. Verify clock synchronization

### Capacity Planning

| Tier | Users | Expected RPS | Redis Memory |
|------|-------|--------------|--------------|
| 100 anonymous | - | 100 | 10 MB |
| 1000 authenticated | - | 1000 | 100 MB |
| 100 premium | - | 10000 | 100 MB |
