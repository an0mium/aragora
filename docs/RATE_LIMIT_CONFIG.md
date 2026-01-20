# API Rate Limit Configuration

Reference document for API rate limiting across subscription tiers.

## Default Tier Limits

| Tier | Requests/Min | Burst | Monthly Equivalent | Typical Use Case |
|------|--------------|-------|-------------------|------------------|
| **Free** | 10 | 60 | ~432K | Personal/evaluation |
| **Starter** | 50 | 100 | ~2.16M | Small teams |
| **Professional** | 200 | 400 | ~8.64M | Growing companies |
| **Enterprise** | 1000 | 2000 | ~43.2M | Large deployments |

## Rate Limit Behavior

### Token Bucket Algorithm
The rate limiter uses a token bucket algorithm with configurable:
- **Rate**: Tokens added per minute (sustained throughput)
- **Burst**: Maximum bucket capacity (handles traffic spikes)

### Response Headers
When rate limited, responses include:
```
X-RateLimit-Limit: <requests per minute>
X-RateLimit-Remaining: <remaining requests>
X-RateLimit-Reset: <Unix timestamp when bucket refills>
Retry-After: <seconds to wait>
```

### HTTP Status
- `429 Too Many Requests` when rate limited
- Response body includes `retry_after` seconds

## Configuration

### Environment Variables

```bash
# Override default tier limits (JSON format)
ARAGORA_RATE_LIMIT_TIERS='{"free": [10, 60], "starter": [50, 100], "professional": [200, 400], "enterprise": [1000, 2000]}'

# IP-based rate limit (for unauthenticated requests)
ARAGORA_RATE_LIMIT_IP_DEFAULT=60

# Maximum bucket entries before LRU eviction
ARAGORA_RATE_LIMIT_MAX_ENTRIES=10000
```

### Code Configuration

```python
from aragora.server.middleware.rate_limit import TierRateLimiter

# Custom tier limits
custom_limits = {
    "free": (5, 30),        # 5 req/min, 30 burst (more restrictive)
    "starter": (100, 200),  # 100 req/min, 200 burst (more generous)
    "professional": (500, 1000),
    "enterprise": (2000, 5000),
}

limiter = TierRateLimiter(tier_limits=custom_limits)
```

## Tuning Guidelines

### When to Increase Limits
- High-traffic legitimate use cases
- API integrations requiring higher throughput
- Batch processing workloads
- Real-time applications with frequent polling

### When to Decrease Limits
- Abuse patterns detected
- Resource constraints (CPU, memory, API costs)
- Provider rate limits being hit
- Quality-of-service concerns for other users

### Burst Size Recommendations
| Scenario | Burst Multiplier |
|----------|------------------|
| Steady traffic | 2-3x rate |
| Bursty webhooks | 5-10x rate |
| Batch jobs | 10-20x rate |
| Real-time apps | 2x rate (with low latency) |

## Endpoint-Specific Limits

Some endpoints have additional per-endpoint rate limits:

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `POST /api/debates` | 10/min | High compute cost |
| `POST /api/auth/login` | 5/min | Security (brute force) |
| `GET /api/health` | 100/min | Monitoring allows frequent checks |
| WebSocket connect | 1/sec | Connection overhead |

## Monitoring

### Prometheus Metrics
```promql
# Rate limit hit rate
rate(aragora_rate_limit_hits_total[5m])

# Rate limit by tier
aragora_rate_limit_hits_total{tier="free"}

# Percentage of requests rate limited
rate(aragora_http_requests_total{status="429"}[5m])
  / rate(aragora_http_requests_total[5m]) * 100
```

### Alerts
```yaml
- alert: HighRateLimitHits
  expr: rate(aragora_rate_limit_hits_total[5m]) > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: High rate limit hit rate
```

## Troubleshooting

### User Hitting Rate Limits

1. Check current limits for their tier:
```bash
curl -s http://localhost:8080/api/rate-limits/status \
  -H "Authorization: Bearer $TOKEN" | jq .
```

2. Check if they need a tier upgrade

3. Check for abuse patterns:
```bash
# Top rate-limited IPs
curl -s 'localhost:9090/api/v1/query?query=topk(10,aragora_rate_limit_hits_total)'
```

### Legitimate High-Volume Use

For legitimate high-volume integrations:
1. Consider Enterprise tier
2. Contact sales for custom limits
3. Implement client-side backoff
4. Use bulk/batch endpoints where available

## Security Considerations

- Rate limits help prevent:
  - API abuse and scraping
  - Brute force attacks
  - Denial of service
  - Excessive cost from AI providers

- IP extraction uses trusted proxy headers only when configured
- Authenticated users tracked by user_id (more stable than IP)

## Related Documentation

- [RUNBOOK.md](RUNBOOK.md) - Operational procedures
- [tenancy/quotas.py](../aragora/tenancy/quotas.py) - Quota management
- [rate_limit/tier_limiter.py](../aragora/server/middleware/rate_limit/tier_limiter.py) - Implementation
