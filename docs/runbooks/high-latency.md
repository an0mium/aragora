# Runbook: High Latency

**Alert:** `HighAPILatency`, `HighAgentLatency`
**Severity:** Warning
**Threshold:** p99 > 2s (API), p95 > 30s (Agent)

## Symptoms

- Slow API responses
- Request timeouts
- Users reporting "loading" states
- p99/p95 latency metrics elevated

## Diagnosis

### 1. Identify slow endpoints

```bash
# Check endpoint latencies
curl -s localhost:8080/api/health/detailed | jq '.endpoints | to_entries | sort_by(.value.p99_latency) | reverse | .[:10]'

# Check Prometheus metrics
curl -s localhost:8080/metrics | grep aragora_api_latency
```

### 2. Check AI provider latency

```bash
# Agent latency by provider
curl -s localhost:8080/metrics | grep aragora_agent_latency

# Test provider directly
time curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
```

### 3. Check database latency

```bash
# Query timing
psql $DATABASE_URL -c "
SELECT query, calls, total_time/calls as avg_ms
FROM pg_stat_statements
ORDER BY total_time/calls DESC
LIMIT 10;"
```

### 4. Check system resources

```bash
# CPU load
uptime

# Memory
free -h

# I/O wait
iostat -x 1 5
```

## Common Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| AI provider slow | Agent latency high | Wait or switch provider |
| Database slow | Query times high | Add indexes, optimize |
| CPU saturation | Load > cores | Scale up/out |
| Memory pressure | Swap usage | Add memory |
| Network issues | High RTT | Check network path |

## Resolution Steps

### AI provider latency

If AI provider is slow:

1. Check provider status page
2. Consider enabling fallback:
   ```bash
   export OPENROUTER_API_KEY="your-key"
   # OpenRouter provides automatic fallback
   ```
3. Reduce token limits temporarily:
   ```bash
   export ARAGORA_MAX_TOKENS=2000
   ```

### Database latency

1. **Identify slow queries:**
   ```bash
   psql $DATABASE_URL -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5"
   ```

2. **Add missing indexes:**
   ```bash
   # Check for sequential scans
   psql $DATABASE_URL -c "
   SELECT relname, seq_scan, seq_tup_read
   FROM pg_stat_user_tables
   WHERE seq_scan > 100
   ORDER BY seq_tup_read DESC;"
   ```

3. **Vacuum if needed:**
   ```bash
   psql $DATABASE_URL -c "VACUUM ANALYZE"
   ```

### Application-level fixes

1. **Enable caching (if not already):**
   ```bash
   export ARAGORA_CACHE_ENABLED=true
   export ARAGORA_CACHE_TTL=300
   ```

2. **Reduce concurrent requests:**
   ```bash
   export ARAGORA_MAX_CONCURRENT_DEBATES=5
   ```

3. **Enable request coalescing:**
   ```bash
   export ARAGORA_COALESCE_SIMILAR_REQUESTS=true
   ```

## Monitoring

After applying fixes:

```bash
# Watch latency metrics
watch -n 5 'curl -s localhost:8080/metrics | grep "aragora_api_latency_seconds_bucket.*le=\"2\"" | tail -5'

# Check p99 trend
curl -s localhost:8080/api/health/detailed | jq '.latency.p99'
```

## Escalation

If latency persists >30 minutes:
1. Consider scaling horizontally
2. Enable request queueing/throttling
3. Communicate expected degradation to users

## Prevention

- Set up latency budgets per endpoint
- Regular query performance reviews
- Load testing before major releases
- Auto-scaling based on latency metrics
