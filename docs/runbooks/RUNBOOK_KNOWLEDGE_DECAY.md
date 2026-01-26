# Runbook: Knowledge Confidence Decay Monitoring

## Overview

The confidence decay scheduler automatically reduces confidence scores for aging knowledge items in the Knowledge Mound. This prevents stale knowledge from inappropriately influencing debate outcomes.

## Alerts

| Alert | Severity | Description |
|-------|----------|-------------|
| `DecaySchedulerStopped` | warning | Decay scheduler is not running |
| `StaleWorkspace` | warning | Workspace has not had decay in >48 hours |
| `HighDecayErrorRate` | critical | Decay operations failing frequently |

## Quick Diagnostics

### Check Decay Health

```bash
# Check decay scheduler status
curl http://localhost:8080/api/v1/health/decay | jq

# Expected healthy response:
{
  "status": "healthy",
  "scheduler": {
    "running": true,
    "decay_interval_hours": 24,
    "min_confidence_threshold": 0.1,
    "decay_rate": 0.95
  },
  "statistics": {
    "total_cycles": 45,
    "total_items_processed": 12500,
    "total_items_expired": 230,
    "errors": 0
  },
  "workspaces": {
    "total": 5,
    "stale_count": 0,
    "stale_threshold_hours": 48
  }
}
```

### Check Knowledge Mound Health

```bash
curl http://localhost:8080/api/v1/health/knowledge-mound | jq '.components.confidence_decay'
```

## Common Issues

### 1. Scheduler Not Running

**Symptoms:**
- `scheduler.running: false` in decay health
- Confidence scores not decreasing over time

**Causes:**
- Server started in degraded mode
- Scheduler initialization failed
- Resource constraints

**Resolution:**
1. Check server startup status:
   ```bash
   curl http://localhost:8080/api/v1/health/startup | jq
   ```
2. Restart the decay scheduler manually:
   ```python
   from aragora.knowledge.mound.confidence_decay_scheduler import get_decay_scheduler
   scheduler = get_decay_scheduler()
   await scheduler.start()
   ```
3. Check logs for initialization errors:
   ```bash
   grep "confidence_decay" /var/log/aragora/server.log
   ```

### 2. Stale Workspaces

**Symptoms:**
- `workspaces.stale_count > 0` in decay health
- Warning messages about workspaces not having decay

**Causes:**
- Scheduler running but workspace-specific issues
- Knowledge Mound adapter connectivity issues
- Database lock contention

**Resolution:**
1. Identify stale workspaces:
   ```bash
   curl http://localhost:8080/api/v1/health/decay | jq '.workspaces.details'
   ```
2. Manually trigger decay for specific workspace:
   ```python
   from aragora.knowledge.mound.confidence_decay_scheduler import get_decay_scheduler
   scheduler = get_decay_scheduler()
   await scheduler.apply_decay_to_workspace("workspace_id")
   ```
3. Check Knowledge Mound adapter health:
   ```bash
   curl http://localhost:8080/api/v1/health/knowledge-mound | jq '.adapters'
   ```

### 3. High Error Rate

**Symptoms:**
- `statistics.errors > 0` increasing
- Prometheus metric `aragora_knowledge_decay_errors_total` rising

**Causes:**
- Database connectivity issues
- Invalid data in knowledge items
- Resource exhaustion

**Resolution:**
1. Check database health:
   ```bash
   curl http://localhost:8080/api/v1/health/database | jq
   ```
2. Review error logs:
   ```bash
   grep -E "decay.*error|Error.*decay" /var/log/aragora/server.log | tail -20
   ```
3. Check for corrupted knowledge items:
   ```sql
   SELECT id, confidence, workspace_id
   FROM knowledge_items
   WHERE confidence < 0 OR confidence > 1;
   ```

## Prometheus Metrics

Monitor these decay-related metrics:

```promql
# Total decay cycles
rate(aragora_knowledge_decay_cycles_total[1h])

# Items processed per cycle
rate(aragora_knowledge_decay_items_processed_total[1h])

# Decay operation duration
histogram_quantile(0.99, aragora_knowledge_decay_duration_seconds_bucket)

# Average confidence change
aragora_knowledge_decay_avg_change
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_DECAY_INTERVAL_HOURS` | `24` | Hours between decay cycles |
| `ARAGORA_DECAY_RATE` | `0.95` | Multiplier per cycle (0.95 = 5% reduction) |
| `ARAGORA_MIN_CONFIDENCE` | `0.1` | Threshold below which items are expired |

### Adjusting Decay Parameters

```python
# Access scheduler and update configuration
from aragora.knowledge.mound.confidence_decay_scheduler import get_decay_scheduler

scheduler = get_decay_scheduler()
scheduler.decay_rate = 0.90  # More aggressive decay
scheduler.decay_interval_hours = 12  # More frequent cycles
```

## Manual Operations

### Force Immediate Decay Cycle

```bash
# Via API (if endpoint available)
curl -X POST http://localhost:8080/api/v1/admin/knowledge/decay/trigger

# Via Python
python -c "
import asyncio
from aragora.knowledge.mound.confidence_decay_scheduler import get_decay_scheduler
scheduler = get_decay_scheduler()
asyncio.run(scheduler.apply_decay_to_workspaces())
"
```

### Reset Confidence for Specific Items

```sql
-- Reset confidence to 1.0 for recently verified items
UPDATE knowledge_items
SET confidence = 1.0, updated_at = CURRENT_TIMESTAMP
WHERE id IN (SELECT item_id FROM recent_verifications);
```

## Escalation

| Condition | Action |
|-----------|--------|
| All workspaces stale | Investigate scheduler immediately |
| Error rate > 10% | Check infrastructure health |
| Items not expiring | Review min_confidence threshold |

## Related Runbooks

- [RUNBOOK_DATABASE_ISSUES.md](./RUNBOOK_DATABASE_ISSUES.md)
- [service-down.md](./service-down.md)
