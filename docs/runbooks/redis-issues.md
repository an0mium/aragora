# Runbook: Redis Issues

**Alert:** `RedisUnavailable`, `RedisHighMemory`, `RedisHighLatency`
**Severity:** Critical
**Impact:** Session store, caching, distributed state, RBAC cache

## Symptoms

- "Redis connection failed" in logs
- Session/auth failures
- Increased latency (cache misses)
- `DistributedStateError` exceptions

## Diagnosis

### 1. Check connectivity

```bash
# Basic ping
redis-cli -u $REDIS_URL ping

# If auth required
redis-cli -u $REDIS_URL --no-auth-warning ping

# Check info
redis-cli -u $REDIS_URL info server
```

### 2. Check memory

```bash
# Memory usage
redis-cli -u $REDIS_URL info memory | grep used_memory_human

# Memory breakdown by key pattern
redis-cli -u $REDIS_URL --bigkeys
```

### 3. Check connections

```bash
# Connected clients
redis-cli -u $REDIS_URL info clients

# Client list
redis-cli -u $REDIS_URL client list | head -20
```

## Common Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| Redis process down | Connection refused | Restart Redis |
| Memory limit reached | OOM errors | Increase memory or evict |
| Network issue | Timeouts | Check network/firewall |
| Slow commands | High latency | Check slowlog |
| Too many connections | Connection errors | Tune connection pool |

## Resolution Steps

### Redis process down

```bash
# Check status
systemctl status redis
# or
docker ps | grep redis

# Restart
systemctl restart redis
# or
docker restart redis
```

### Memory issues

```bash
# Check current memory
redis-cli -u $REDIS_URL info memory

# Set memory limit (if not set)
redis-cli -u $REDIS_URL config set maxmemory 2gb
redis-cli -u $REDIS_URL config set maxmemory-policy allkeys-lru

# Manual flush (CAUTION: clears all data)
# redis-cli -u $REDIS_URL flushall
```

### Slow commands

```bash
# Check slow log
redis-cli -u $REDIS_URL slowlog get 10

# Common offenders: KEYS * (use SCAN instead)
# If KEYS command found, identify and fix in code
```

### Application fallback

If Redis is down and cannot be immediately restored:

```bash
# Enable single-instance mode (disables distributed state requirement)
export ARAGORA_SINGLE_INSTANCE=true
systemctl restart aragora
```

**Warning:** This allows in-memory fallback which:
- Loses data on restart
- Doesn't work with multiple instances
- Should only be temporary

## Verification

After fix, verify Aragora can connect:

```bash
# Run environment validation
python -m aragora.cli validate-env

# Check health endpoint
curl -s localhost:8080/readyz | jq '.checks.redis'
```

## Escalation

If Redis cannot be restored within 15 minutes:
1. Enable single-instance mode as temporary workaround
2. Notify users of potential session issues
3. Contact infrastructure/platform team

## Prevention

- Set up Redis Sentinel or Cluster for HA
- Configure maxmemory with appropriate eviction policy
- Monitor memory usage and connection count
- Avoid KEYS commands in production code
