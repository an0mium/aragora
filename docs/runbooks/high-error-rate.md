# Runbook: High Error Rate

**Alert:** `HighErrorRate`
**Severity:** Critical
**Threshold:** >5% of requests failing over 5 minutes

## Symptoms

- Error rate above 5% for sustained period
- Users reporting failed requests
- Error logs increasing

## Diagnosis

### 1. Identify error patterns

```bash
# Check error distribution by endpoint
curl -s localhost:8080/api/health/detailed | jq '.error_counts'

# Check recent error logs
grep -i "error\|exception" /var/log/aragora/app.log | tail -50

# Check by status code (if using structured logs)
jq 'select(.status >= 400)' /var/log/aragora/app.log | head -20
```

### 2. Check upstream dependencies

```bash
# Test AI provider connectivity
curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":1,"messages":[{"role":"user","content":"test"}]}'

# Test Redis
redis-cli ping

# Test PostgreSQL
psql $DATABASE_URL -c "SELECT 1"
```

### 3. Check rate limits

```bash
# Check if hitting API rate limits
grep -i "rate.limit\|429\|too.many.requests" /var/log/aragora/app.log | tail -20

# Check current request rate
curl -s localhost:8080/metrics | grep aragora_api_requests_total
```

## Common Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| AI provider outage | 5xx from provider, timeouts | Wait or switch provider |
| Rate limiting | 429 responses | Implement backoff, check quotas |
| Database issues | Connection errors | Check connection pool |
| Memory pressure | OOM errors | Increase memory, reduce load |
| Bad deployment | Errors started at deploy time | Roll back |

## Resolution Steps

### 1. Identify the failing endpoint

```bash
# Get error breakdown
curl -s localhost:8080/api/health/detailed | jq '.endpoints | to_entries | map(select(.value.error_rate > 0.01))'
```

### 2. Check circuit breakers

```bash
curl -s localhost:8080/api/health/circuits | jq '.circuits | map(select(.state != "closed"))'
```

### 3. If AI provider issue

- Check provider status page
- Enable fallback provider:
  ```bash
  # Temporarily enable OpenRouter fallback
  export OPENROUTER_API_KEY="your-key"
  systemctl restart aragora
  ```

### 4. If database issue

```bash
# Check connection pool
curl -s localhost:8080/api/health/deep | jq '.database'

# Restart if pool exhausted (temporary fix)
systemctl restart aragora
```

## Mitigation

If errors persist:

1. **Enable maintenance mode** (if available):
   ```bash
   curl -X POST localhost:8080/api/admin/maintenance -d '{"enabled": true}'
   ```

2. **Scale down non-critical features**:
   ```bash
   # Disable background jobs
   export ARAGORA_DISABLE_BACKGROUND_JOBS=true
   ```

3. **Increase timeouts** (if timeout-related):
   ```bash
   export ARAGORA_AGENT_TIMEOUT=60
   ```

## Escalation

If error rate exceeds 20% or persists >30 minutes:
1. Page secondary on-call
2. Consider enabling maintenance mode
3. Prepare customer communication

## Post-Incident

- [ ] Create incident ticket
- [ ] Analyze error logs for root cause
- [ ] Add alerting for specific error type if new
- [ ] Update circuit breaker thresholds if needed
