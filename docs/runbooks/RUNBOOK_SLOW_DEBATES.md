# Slow Debates Runbook

Procedures for investigating and resolving slow debate performance.

## Alert: SlowDebateDetected

**Condition:** p95 debate duration exceeds 60 seconds
**Severity:** Warning

## Quick Diagnosis

```bash
# Check current slow debates
curl -s http://localhost:8080/api/health/slow-debates | jq .

# Check active debate count
curl -s http://localhost:8080/api/health/detailed | jq '.debates'

# Check agent response times
curl -s http://localhost:8080/metrics | grep aragora_agent_response
```

## Common Causes

### 1. Agent Provider Latency

**Symptoms:** Single agent taking >10s per response
**Check:**
```bash
# Check circuit breaker states
curl -s http://localhost:8080/api/health/detailed | jq '.circuit_breakers'

# Check provider-specific metrics
curl -s http://localhost:8080/metrics | grep 'aragora_agent.*latency'
```

**Resolution:**
- Check provider status pages (OpenAI, Anthropic, etc.)
- Enable fallback to OpenRouter: `ARAGORA_OPENROUTER_FALLBACK_ENABLED=true`
- Reduce concurrent requests per provider

### 2. Convergence Backend Initialization

**Symptoms:** First debate of session takes 8-10s extra
**Cause:** SentenceTransformer model loading on first use

**Resolution:**
- This is expected behavior on cold start
- Models are cached after first load
- Consider warming up the service after deployment:
  ```bash
  curl -X POST http://localhost:8080/api/warmup
  ```

### 3. Database Contention

**Symptoms:** Multiple slow debates simultaneously
**Check:**
```bash
# Check SQLite lock contention
ls -la .nomic/*.db-wal

# Check debate store metrics
curl -s http://localhost:8080/metrics | grep 'aragora_db'
```

**Resolution:**
- Scale horizontally (requires PostgreSQL migration)
- Reduce concurrent debate limit
- Check disk I/O

### 4. Memory Pressure

**Symptoms:** All debates slowing down over time
**Check:**
```bash
# Check memory usage
curl -s http://localhost:8080/metrics | grep process_resident_memory

# Check RLM cache size
curl -s http://localhost:8080/metrics | grep aragora_rlm_memory
```

**Resolution:**
- Restart service to clear caches
- Increase memory limits in deployment
- Reduce RLM cache size: `ARAGORA_RLM_MAX_CACHE_ENTRIES=1000`

## Escalation

If issue persists after following above steps:

1. Collect diagnostics:
   ```bash
   curl -s http://localhost:8080/api/health/detailed > health.json
   curl -s http://localhost:8080/metrics > metrics.txt
   ```

2. Check logs for errors:
   ```bash
   kubectl logs -l app=aragora --tail=500 | grep -i error
   ```

3. Create incident following [RUNBOOK_INCIDENT.md](./RUNBOOK_INCIDENT.md)

## Prevention

- Monitor p95 debate latency dashboard
- Set up alerts at 30s threshold (warning) and 60s (critical)
- Run load tests before major releases
- Warm up service after deployments

## Related Alerts

- `DebateRoundLatencyHigh` - Individual round taking too long
- `NoActiveDebates` - No debates for extended period (upstream issue)
- `AgentFailureRateHigh` - Agent failures may cause retries and slowdowns
