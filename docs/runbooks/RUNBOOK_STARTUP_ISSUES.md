# Runbook: Server Startup Issues

## Overview

This runbook covers troubleshooting server startup failures, SLO violations, and initialization issues.

## Alerts

| Alert | Severity | Description |
|-------|----------|-------------|
| `StartupSLOExceeded` | warning | Startup took longer than 30 seconds |
| `StartupFailed` | critical | Server failed to start |
| `ComponentInitFailed` | warning | A non-critical component failed to initialize |

## Quick Diagnostics

### Check Startup Status

```bash
# Check startup health endpoint
curl http://localhost:8080/api/v1/health/startup | jq

# Expected healthy response:
{
  "status": "healthy",
  "startup": {
    "success": true,
    "duration_seconds": 12.45,
    "slo_seconds": 30.0,
    "slo_met": true
  },
  "components": {
    "initialized": 25,
    "failed": []
  }
}
```

### Check Server Logs

```bash
# View startup sequence logs
grep -E "\[startup\]|\[STARTUP\]" /var/log/aragora/server.log | tail -50

# Check for failed component initialization
grep -i "failed to initialize" /var/log/aragora/server.log
```

## Common Issues

### 1. Startup Timeout (SLO Exceeded)

**Symptoms:**
- `/api/v1/health/startup` shows `slo_met: false`
- Startup takes > 30 seconds

**Causes:**
- Slow database connections
- Network latency to Redis
- Large number of stale jobs to recover
- Slow initialization of Knowledge Mound adapters

**Resolution:**
1. Check database connectivity:
   ```bash
   curl http://localhost:8080/api/v1/health/database | jq
   ```
2. Check Redis connectivity:
   ```bash
   curl http://localhost:8080/api/v1/health/stores | jq '.redis'
   ```
3. Consider parallelizing initialization by setting:
   ```bash
   export ARAGORA_PARALLEL_INIT=true
   ```

### 2. Component Initialization Failures

**Symptoms:**
- `components.failed` is non-empty in startup health
- Server runs in degraded mode

**Causes:**
- Missing environment variables
- Service dependencies unavailable
- Permission issues

**Resolution:**
1. Identify failed components from startup health
2. Check component-specific health endpoints:
   - `/api/v1/health/knowledge-mound` - Knowledge Mound adapters
   - `/api/v1/health/decay` - Confidence decay scheduler
   - `/api/v1/health/circuits` - Circuit breakers
3. Check required environment variables for the component
4. Restart with verbose logging:
   ```bash
   ARAGORA_LOG_LEVEL=DEBUG python -m aragora.server.unified_server
   ```

### 3. Degraded Mode

**Symptoms:**
- Server returns 503 for most endpoints
- Liveness probe passes but readiness fails

**Causes:**
- Critical initialization failure
- Missing required configuration
- Backend connectivity issues

**Resolution:**
1. Check degraded mode status:
   ```bash
   curl http://localhost:8080/readyz
   ```
2. Review degraded mode reason in logs:
   ```bash
   grep "set_degraded" /var/log/aragora/server.log
   ```
3. Fix the underlying issue and restart

## Rollback Procedures

If startup fails after an upgrade:

1. **Quick rollback:**
   ```bash
   # Stop the new version
   systemctl stop aragora

   # Switch to previous version
   ln -sf /opt/aragora/releases/previous /opt/aragora/current

   # Start previous version
   systemctl start aragora
   ```

2. **Check previous version starts successfully:**
   ```bash
   curl http://localhost:8080/healthz
   curl http://localhost:8080/readyz
   ```

## Prometheus Metrics

Monitor these startup-related metrics:

```promql
# Startup duration histogram
histogram_quantile(0.99, aragora_server_startup_duration_seconds_bucket)

# Components initialized
aragora_server_startup_components_initialized

# Startup SLO compliance
aragora_startup_slo_compliant
```

## Escalation

| Condition | Action |
|-----------|--------|
| Startup fails repeatedly | Escalate to on-call engineer |
| SLO exceeded > 2x threshold | Review initialization sequence |
| Multiple components failing | Review infrastructure health |

## Related Runbooks

- [RUNBOOK_DATABASE_ISSUES.md](./RUNBOOK_DATABASE_ISSUES.md)
- [redis-issues.md](./redis-issues.md)
- [service-down.md](./service-down.md)
