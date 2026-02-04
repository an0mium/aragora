# Service Degradation Runbook

This runbook covers graceful degradation strategies and circuit breaker management for Aragora.

## Degradation Levels

| Level | Description | Actions |
|-------|-------------|---------|
| Green | Normal operation | All features available |
| Yellow | Elevated load | Rate limiting active, non-critical features throttled |
| Orange | Partial outage | Optional features disabled, core functionality prioritized |
| Red | Critical | Emergency mode, essential services only |

## Circuit Breaker States

Aragora uses circuit breakers to prevent cascade failures. States:
- **Closed**: Normal operation, requests flow through
- **Open**: Failures exceeded threshold, requests fail fast
- **Half-Open**: Testing if service recovered

### Monitoring Circuit Breakers

```bash
# Check circuit breaker states via API
curl https://api.aragora.ai/api/internal/circuit-breakers | jq .

# Example response:
# {
#   "openai_api": {"state": "closed", "failures": 2, "last_failure": null},
#   "anthropic_api": {"state": "half_open", "failures": 5, "last_failure": "2026-02-03T10:00:00Z"},
#   "database": {"state": "closed", "failures": 0, "last_failure": null}
# }

# Prometheus query for circuit breaker trips
curl -s 'https://prometheus.aragora.ai/api/v1/query?query=circuit_breaker_state{state="open"}'
```

### Manual Circuit Breaker Control

```python
# Python script to manually control circuit breakers
from aragora.resilience import CircuitBreakerRegistry

# Force open a circuit breaker (emergency)
CircuitBreakerRegistry.get("openai_api").force_open()

# Force close (after manual verification)
CircuitBreakerRegistry.get("openai_api").force_close()

# Reset to automatic mode
CircuitBreakerRegistry.get("openai_api").reset()
```

## Graceful Degradation Strategies

### 1. LLM Provider Fallback

When primary LLM provider fails:

```python
# Priority order configured in aragora/agents/fallback.py
FALLBACK_ORDER = [
    "anthropic",      # Primary
    "openai",         # First fallback
    "openrouter",     # Second fallback (DeepSeek, Llama)
]

# Manual override via environment
export ARAGORA_LLM_PROVIDER=openrouter
kubectl rollout restart deployment/aragora-api
```

### 2. Feature Flags for Degradation

```bash
# Disable non-critical features
curl -X POST https://api.aragora.ai/api/internal/feature-flags \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "cross_debate_memory": false,
    "rhetorical_observer": false,
    "belief_network": false,
    "knowledge_mound_sync": false
  }'

# Re-enable after recovery
curl -X POST https://api.aragora.ai/api/internal/feature-flags \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "cross_debate_memory": true,
    "rhetorical_observer": true,
    "belief_network": true,
    "knowledge_mound_sync": true
  }'
```

### 3. Rate Limiting Tiers

| Tier | Requests/min | Debates/hour | When to activate |
|------|--------------|--------------|------------------|
| Normal | 1000 | 100 | Green |
| Elevated | 500 | 50 | Yellow |
| Reduced | 100 | 10 | Orange |
| Emergency | 20 | 2 | Red |

```bash
# Switch to elevated rate limiting
kubectl set env deployment/aragora-api RATE_LIMIT_TIER=elevated

# Emergency rate limiting
kubectl set env deployment/aragora-api RATE_LIMIT_TIER=emergency
```

### 4. Cache Strategies

```bash
# Extend cache TTLs during high load
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET maxmemory 2gb

# Enable aggressive caching
kubectl set env deployment/aragora-api CACHE_STRATEGY=aggressive

# Warm critical caches
python scripts/warm_cache.py --critical-only
```

## Queue Management

### Debate Queue Backpressure

```bash
# Check queue depth
curl https://api.aragora.ai/api/internal/queues | jq .

# If queue is backing up:
# 1. Scale workers
kubectl scale deployment/aragora-worker --replicas=10

# 2. Enable priority processing
kubectl set env deployment/aragora-worker PRIORITY_ONLY=true

# 3. Pause non-critical queues
curl -X POST https://api.aragora.ai/api/internal/queues/pause \
  -d '{"queues": ["analytics", "notifications", "sync"]}'
```

### Dead Letter Queue Processing

```bash
# Check DLQ size
aws sqs get-queue-attributes \
  --queue-url $DLQ_URL \
  --attribute-names ApproximateNumberOfMessages

# Reprocess DLQ messages (after fixing underlying issue)
python scripts/reprocess_dlq.py --queue debate-dlq --limit 100

# Purge DLQ (if messages are unrecoverable)
aws sqs purge-queue --queue-url $DLQ_URL
```

## Health Check Configuration

```yaml
# Kubernetes health check with degradation awareness
livenessProbe:
  httpGet:
    path: /api/health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /api/health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 2

# During degradation, health endpoint returns degraded status
# but stays "ready" if core functions work
```

## Recovery Checklist

### Yellow to Green
- [ ] Load returned to normal
- [ ] All circuit breakers closed
- [ ] Rate limiting back to normal tier
- [ ] Queue depths normal
- [ ] Error rates < 0.1%

### Orange to Yellow
- [ ] Core services stable
- [ ] Primary providers responsive
- [ ] Re-enable non-critical features one by one
- [ ] Monitor for 15 minutes between each

### Red to Orange
- [ ] Emergency stabilized
- [ ] Identify root cause
- [ ] Scale back to normal capacity gradually
- [ ] Re-enable queues

## Monitoring Dashboards

| Dashboard | Purpose | URL |
|-----------|---------|-----|
| Service Health | Overall system health | /grafana/d/service-health |
| Circuit Breakers | Breaker states and trips | /grafana/d/circuit-breakers |
| LLM Providers | Provider latency/errors | /grafana/d/llm-providers |
| Queue Metrics | Queue depths and processing | /grafana/d/queues |

---
*Last updated: 2026-02-03*
