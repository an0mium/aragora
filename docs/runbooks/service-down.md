# Runbook: Service Down

**Alert:** `ServiceDown`
**Severity:** Critical
**SLO Impact:** Yes - affects availability target

## Symptoms

- Health endpoint `/healthz` returns non-200
- Prometheus `up{job="aragora"} == 0`
- Users report connection refused or timeouts

## Diagnosis

### 1. Check process status

```bash
# Check if process is running
ps aux | grep aragora

# Check systemd status (if using systemd)
systemctl status aragora

# Check container status (if using Docker/K8s)
docker ps | grep aragora
kubectl get pods -l app=aragora
```

### 2. Check logs

```bash
# Recent logs
journalctl -u aragora -n 100 --no-pager

# Docker logs
docker logs aragora-server --tail 100

# Kubernetes logs
kubectl logs -l app=aragora --tail=100
```

### 3. Check resources

```bash
# Memory/CPU
free -h
top -bn1 | head -20

# Disk space
df -h

# Open file descriptors
lsof -p $(pgrep -f aragora) | wc -l
```

## Common Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| OOM killed | `dmesg | grep -i kill` | Increase memory limits |
| Disk full | `df -h` shows 100% | Clear logs, expand disk |
| Port conflict | `netstat -tlnp | grep 8080` | Kill conflicting process |
| Config error | Startup logs show error | Fix config, restart |
| Dependency down | Can't connect to Redis/Postgres | Fix dependency first |

## Resolution Steps

### Quick Recovery

```bash
# Restart the service
systemctl restart aragora

# Or with Docker
docker restart aragora-server

# Or with Kubernetes
kubectl rollout restart deployment/aragora
```

### If restart doesn't help

1. **Check dependencies first:**
   ```bash
   # Test Redis
   redis-cli ping

   # Test PostgreSQL
   psql $DATABASE_URL -c "SELECT 1"
   ```

2. **Check configuration:**
   ```bash
   python -m aragora.cli validate-env
   ```

3. **Roll back if recent deployment:**
   ```bash
   kubectl rollout undo deployment/aragora
   ```

## Escalation

If not resolved within 15 minutes:
1. Page secondary on-call
2. Notify #incidents Slack channel
3. Consider failover to secondary region (if available)

## Post-Incident

- [ ] Create incident ticket
- [ ] Document root cause
- [ ] Update this runbook if needed
- [ ] Schedule post-mortem if SEV1/SEV2
