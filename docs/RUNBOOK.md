# Aragora Operational Runbook

This runbook provides procedures for common operational issues, debugging, and recovery.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Common Issues](#common-issues)
- [Debugging Procedures](#debugging-procedures)
- [Admin Console & Developer Portal](#admin-console--developer-portal)
- [Recovery Procedures](#recovery-procedures)
- [Health Checks](#health-checks)
- [Alerts and Responses](#alerts-and-responses)

---

## Quick Reference

### Service Ports
| Service | Port | Protocol |
|---------|------|----------|
| API Server | 8080 | HTTP |
| WebSocket Server | 8766 | WS |
| Prometheus | 9090 | HTTP |
| Grafana | 3000 | HTTP |
| Redis | 6379 | TCP |

### Key Commands
```bash
# Check service health
curl http://localhost:8080/api/health

# View recent logs
journalctl -u aragora -n 100 --no-pager

# Check WebSocket connections
curl http://localhost:8080/api/ws/stats

# Check rate limit status
curl http://localhost:8080/api/rate-limits/status

# Restart service
systemctl restart aragora
```

### Environment Variables
```bash
# Core
ARAGORA_API_TOKEN          # API authentication
ARAGORA_JWT_SECRET         # JWT signing key (required in production)

# Database
SUPABASE_URL               # Supabase URL
SUPABASE_KEY               # Supabase API key

# Redis (optional but recommended)
ARAGORA_REDIS_URL          # redis://host:port/db
ARAGORA_ENABLE_REDIS_RATE_LIMIT=1

# Observability
SENTRY_DSN                 # Sentry error tracking
METRICS_ENABLED=true       # Enable Prometheus metrics
```

---

## Common Issues

### 1. WebSocket Disconnections

**Symptoms:**
- Clients receiving `WebSocket closed unexpectedly`
- Missing real-time debate updates
- `ws_connection_errors` metric increasing

**Causes:**
1. Server restart
2. Network timeout
3. Rate limiting
4. Memory pressure

**Resolution:**

```bash
# Check WebSocket server status
curl http://localhost:8080/api/health | jq '.checks.websocket'

# Check connection count
curl http://localhost:8080/api/ws/stats

# If connections are stuck, restart WebSocket handler
# (This preserves API functionality)
kill -USR1 $(pgrep -f aragora)  # Graceful reload if supported
```

**Prevention:**
- Implement client-side reconnection with exponential backoff
- Monitor `aragora_websocket_connections` metric
- Set up alerts when connections exceed 80% of max

---

### 2. Rate Limit False Positives

**Symptoms:**
- Legitimate users receiving 429 responses
- `rate_limit_hits` metric spiking
- Complaints about API access

**Causes:**
1. Shared IP (corporate NAT, VPN)
2. Rate limit configuration too strict
3. Redis connectivity issues

**Resolution:**

```bash
# Check current rate limit config
curl http://localhost:8080/api/rate-limits/config

# Check if Redis is connected
redis-cli ping

# View blocked IPs (if logged)
grep "rate_limit" /var/log/aragora/api.log | tail -20

# Temporarily increase limits (requires restart)
export ARAGORA_RATE_LIMIT_DEBATE_RPM=20
systemctl restart aragora
```

**Whitelist Specific IPs:**
```python
# In config or environment
ARAGORA_RATE_LIMIT_WHITELIST=10.0.0.0/8,192.168.0.0/16
```

---

### 3. Agent Timeout Handling

**Symptoms:**
- Debates not completing
- `aragora_debates_total{status="timeout"}` increasing
- Individual agents timing out

**Causes:**
1. LLM provider latency spike
2. Complex debate topics
3. Network issues to AI providers

**Resolution:**

```bash
# Check circuit breaker status
curl http://localhost:8080/api/agents/circuit-breakers

# Check agent response times
curl http://localhost:8080/api/metrics | grep agent_response

# Reset a stuck circuit breaker
curl -X POST http://localhost:8080/api/agents/claude/reset-circuit

# Increase timeouts temporarily
export ARAGORA_AGENT_TIMEOUT_SECONDS=120
systemctl restart aragora
```

**Agent-Specific Checks:**
```bash
# Test individual agent
curl -X POST http://localhost:8080/api/debug/agent-test \
  -H "Content-Type: application/json" \
  -d '{"agent": "claude", "prompt": "Say hello"}'
```

---

### 4. Database Connection Exhaustion

**Symptoms:**
- `OperationalError: too many connections`
- Slow API responses
- Health check database failures

**Causes:**
1. Connection pool exhaustion
2. Long-running transactions
3. Connection leaks
4. Sudden traffic spike

**Resolution:**

```bash
# Check connection count (PostgreSQL)
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='aragora';"

# Kill idle connections older than 10 minutes
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
         WHERE datname='aragora' AND state='idle'
         AND state_change < now() - interval '10 minutes';"

# Increase pool size (if resources allow)
export ARAGORA_DB_POOL_SIZE=20
export ARAGORA_DB_MAX_OVERFLOW=10
systemctl restart aragora
```

---

### 5. Memory Pressure

**Symptoms:**
- OOM killer terminating process
- Slow response times
- `process_resident_memory_bytes` climbing

**Causes:**
1. Memory leak
2. Large debate results in memory
3. Cache not evicting
4. Too many concurrent debates

**Resolution:**

```bash
# Check memory usage
ps aux | grep aragora
cat /proc/$(pgrep -f aragora)/status | grep -E "VmRSS|VmSize"

# Trigger cache cleanup
curl -X POST http://localhost:8080/api/admin/cleanup-caches

# Reduce concurrent debates
export ARAGORA_MAX_CONCURRENT_DEBATES=5
systemctl restart aragora

# If OOM is imminent, graceful restart
systemctl restart aragora
```

---

### 6. Consensus Failures

**Symptoms:**
- Low `aragora_consensus_reached` rate
- Debates completing without resolution
- Users reporting inconclusive results

**Causes:**
1. Topic too contentious
2. Agent disagreement
3. Voting threshold too high
4. Insufficient debate rounds

**Resolution:**

```bash
# Check consensus rates
curl http://localhost:8080/api/metrics | grep consensus

# Review recent failed debates
curl http://localhost:8080/api/debates?status=no_consensus&limit=10

# Adjust consensus threshold (requires restart)
export ARAGORA_CONSENSUS_THRESHOLD=0.6
systemctl restart aragora
```

---

## Debugging Procedures

### Log Analysis

**Log Locations:**
```
/var/log/aragora/api.log         # API server logs
/var/log/aragora/websocket.log   # WebSocket logs
/var/log/aragora/debate.log      # Debate execution logs
/var/log/aragora/agent.log       # Agent communication logs
```

**Structured Log Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "aragora.debate.orchestrator",
  "message": "debate_start",
  "debate_id": "abc123",
  "correlation_id": "corr-xyz789",
  "agents": ["claude", "gpt-4"],
  "domain": "security"
}
```

**Common Log Queries:**
```bash
# Find all errors for a specific debate
grep "debate_id=abc123" /var/log/aragora/*.log | grep -i error

# Find rate limit events
grep "rate_limit" /var/log/aragora/api.log

# Find circuit breaker events
grep "circuit_breaker" /var/log/aragora/agent.log

# Find slow requests (>5s)
grep -E "duration_ms=[5-9][0-9]{3}|duration_ms=[0-9]{5}" /var/log/aragora/api.log
```

### Correlation ID Tracing

Every request generates a correlation ID for distributed tracing:

```bash
# Find all logs for a correlation ID
grep "corr-xyz789" /var/log/aragora/*.log

# In Sentry, search by correlation_id tag
# In Grafana, use: {correlation_id="corr-xyz789"}
```

### Memory Profiling

```bash
# Enable memory profiling (development only)
export ARAGORA_MEMORY_PROFILING=1
systemctl restart aragora

# Trigger memory dump
curl -X POST http://localhost:8080/api/debug/memory-dump

# View memory dump
cat /tmp/aragora_memory_dump.json | jq '.top_allocations[:10]'

# Check for leaks over time
watch -n 30 'curl -s http://localhost:8080/api/metrics | grep process_resident'
```

### Network Debugging

```bash
# Check API connectivity
curl -v http://localhost:8080/api/health

# Check WebSocket connectivity
websocat ws://localhost:8766 -v

# Check Redis connectivity
redis-cli -h localhost ping

# Check DNS resolution for AI providers
host api.anthropic.com
host api.openai.com

# Test agent connectivity
curl -X POST http://localhost:8080/api/debug/connectivity-check
```

---

## Admin Console & Developer Portal

### Admin Console Access Issues

**Symptoms:**
- `/admin` page loads but panels show “Unauthorized” or empty data
- 401/403 responses from `/api/system/*` endpoints

**Checks:**
```bash
# Validate API auth and role
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/auth/me

# Health endpoint should always respond
curl http://localhost:8080/api/health

# Admin endpoints (require admin/owner role)
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/system/circuit-breakers
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/system/errors?limit=5
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/system/rate-limits
```

**Common Fixes:**
- Ensure `ARAGORA_JWT_SECRET` is set in production.
- Verify the user role is `admin` or `owner`.
- Confirm CORS allowlist includes the admin host.

### Developer Portal Issues

**Symptoms:**
- API key missing or cannot be generated
- Usage data shows zeros or errors

**Checks:**
```bash
# Check authenticated user
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/auth/me

# Generate/revoke API key
curl -X POST -H "Authorization: Bearer <access_token>" http://localhost:8080/api/auth/api-key
curl -X DELETE -H "Authorization: Bearer <access_token>" http://localhost:8080/api/auth/api-key

# Usage stats
curl -H "Authorization: Bearer <access_token>" http://localhost:8080/api/billing/usage
```

**Common Fixes:**
- Check billing configuration if usage metrics are empty.
- Ensure API key storage backend is available (database health).

---

## Recovery Procedures

### 1. Service Restart

**Graceful Restart (preferred):**
```bash
# Send SIGTERM for graceful shutdown
systemctl restart aragora

# Or manually:
kill -TERM $(pgrep -f "aragora serve")
sleep 5
aragora serve --api-port 8080 --ws-port 8766 &
```

**Force Restart (if hung):**
```bash
# Force kill and restart
systemctl stop aragora
sleep 2
pkill -9 -f aragora
systemctl start aragora
```

### 2. Rollback Deployment

```bash
# List available versions
ls -la /opt/aragora/releases/

# Rollback to previous version
cd /opt/aragora
rm current
ln -s releases/v0.7.0 current
systemctl restart aragora

# Verify rollback
curl http://localhost:8080/api/health | jq '.version'
```

### 3. Database Recovery

**Point-in-Time Recovery (if using WAL):**
```bash
# Stop service
systemctl stop aragora

# Restore from backup
pg_restore -d aragora /backups/aragora_daily_20240115.dump

# Apply WAL to specific time
recovery_target_time = '2024-01-15 10:30:00'

# Start service
systemctl start aragora
```

**Clear Corrupted Cache:**
```bash
# Clear Redis cache
redis-cli FLUSHDB

# Clear SQLite caches
rm /var/lib/aragora/*.db
systemctl restart aragora
```

### 4. Emergency Procedures

**Complete Service Outage:**
```bash
#!/bin/bash
# emergency_recovery.sh

# 1. Stop everything
systemctl stop aragora
pkill -9 -f aragora

# 2. Clear potentially corrupted state
redis-cli FLUSHALL
rm /tmp/aragora_*.lock

# 3. Verify dependencies
redis-cli ping || systemctl restart redis
pg_isready || systemctl restart postgresql

# 4. Start with safe defaults
export ARAGORA_MAX_CONCURRENT_DEBATES=3
export ARAGORA_RATE_LIMIT_DISABLED=1  # Temporarily
systemctl start aragora

# 5. Verify
sleep 5
curl http://localhost:8080/api/health
```

---

## Health Checks

### Endpoint: `/api/health`

```bash
curl http://localhost:8080/api/health | jq
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.8.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": {"status": "ok", "latency_ms": 5},
    "redis": {"status": "ok", "latency_ms": 2},
    "circuit_breakers": {"open": 0, "half_open": 1},
    "memory_mb": 512,
    "active_debates": 3
  }
}
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

HEALTH=$(curl -sf http://localhost:8080/api/health)
if [ $? -ne 0 ]; then
  echo "CRITICAL: Health endpoint unreachable"
  exit 2
fi

STATUS=$(echo $HEALTH | jq -r '.status')
if [ "$STATUS" != "healthy" ]; then
  echo "WARNING: Service degraded - $STATUS"
  exit 1
fi

# Check component health
DB_STATUS=$(echo $HEALTH | jq -r '.checks.database.status')
if [ "$DB_STATUS" != "ok" ]; then
  echo "WARNING: Database check failed"
  exit 1
fi

echo "OK: All checks passed"
exit 0
```

---

## Alerts and Responses

### Alert: `AragoraDown`
**Severity:** Critical

**Response:**
1. Check server process: `systemctl status aragora`
2. Check logs: `journalctl -u aragora -n 50`
3. Attempt restart: `systemctl restart aragora`
4. If restart fails, check dependencies (Redis, PostgreSQL)
5. Escalate if not resolved in 10 minutes

### Alert: `HighErrorRate`
**Severity:** Critical

**Response:**
1. Check error logs: `grep ERROR /var/log/aragora/api.log | tail -50`
2. Identify common error pattern
3. Check dependent services (AI providers, database)
4. Consider rate limiting if DDoS suspected
5. Rollback if recent deployment

### Alert: `CircuitBreakerOpen`
**Severity:** Warning

**Response:**
1. Identify which agent: Check Grafana dashboard
2. Test agent directly: `curl http://localhost:8080/api/debug/agent-test -d '{"agent":"claude"}'`
3. Check AI provider status pages
4. Wait for half-open state (auto-recovery)
5. Manual reset if needed: `curl -X POST http://localhost:8080/api/agents/claude/reset-circuit`

### Alert: `HighMemoryUsage`
**Severity:** Warning

**Response:**
1. Check memory: `ps aux | grep aragora`
2. Trigger cleanup: `curl -X POST http://localhost:8080/api/admin/cleanup-caches`
3. Check for memory leaks in recent changes
4. Reduce concurrent debates if needed
5. Schedule restart during low traffic

### Alert: `RedisDown`
**Severity:** Critical

**Response:**
1. Check Redis: `redis-cli ping`
2. Check Redis logs: `journalctl -u redis -n 50`
3. Restart Redis: `systemctl restart redis`
4. Verify Aragora fallback to in-memory (rate limiting)
5. Monitor for rate limit issues

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.26+)
- kubectl configured
- Helm 3.x (optional, for Helm charts)
- Container registry access

### Deployment Files

```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secret.yaml
├── deployment.yaml
├── service.yaml
├── ingress.yaml
├── hpa.yaml
└── pdb.yaml
```

### Basic Deployment

```bash
# Create namespace
kubectl create namespace aragora

# Create secrets (from environment)
kubectl create secret generic aragora-secrets \
  --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --from-literal=ARAGORA_JWT_SECRET="${ARAGORA_JWT_SECRET}" \
  --from-literal=SUPABASE_KEY="${SUPABASE_KEY}" \
  -n aragora

# Apply configuration
kubectl apply -f k8s/configmap.yaml -n aragora
kubectl apply -f k8s/deployment.yaml -n aragora
kubectl apply -f k8s/service.yaml -n aragora
kubectl apply -f k8s/ingress.yaml -n aragora

# Verify deployment
kubectl get pods -n aragora
kubectl logs -f deployment/aragora -n aragora
```

### Deployment Manifest Example

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora
  namespace: aragora
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aragora
  template:
    metadata:
      labels:
        app: aragora
    spec:
      containers:
      - name: aragora
        image: ghcr.io/aragora/aragora:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8766
          name: websocket
        envFrom:
        - configMapRef:
            name: aragora-config
        - secretRef:
            name: aragora-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aragora-hpa
  namespace: aragora
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aragora
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Rolling Update Strategy

```bash
# Update image with rolling deployment
kubectl set image deployment/aragora aragora=ghcr.io/aragora/aragora:v0.9.0 -n aragora

# Watch rollout status
kubectl rollout status deployment/aragora -n aragora

# Rollback if needed
kubectl rollout undo deployment/aragora -n aragora

# View rollout history
kubectl rollout history deployment/aragora -n aragora
```

### Pod Disruption Budget

```yaml
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: aragora-pdb
  namespace: aragora
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: aragora
```

### Monitoring in Kubernetes

```bash
# Port-forward to access Prometheus metrics
kubectl port-forward svc/aragora 8080:8080 -n aragora

# Check metrics
curl http://localhost:8080/metrics

# View pod logs
kubectl logs -f -l app=aragora -n aragora --all-containers

# Check resource usage
kubectl top pods -n aragora
```

### Troubleshooting Kubernetes

```bash
# Check pod status
kubectl describe pod -l app=aragora -n aragora

# Check events
kubectl get events -n aragora --sort-by='.lastTimestamp'

# Shell into pod for debugging
kubectl exec -it deployment/aragora -n aragora -- /bin/sh

# Check service endpoints
kubectl get endpoints aragora -n aragora
```

---

## Helm Chart Deployment

Aragora provides a Helm chart for simplified Kubernetes deployment.

### Installation

```bash
# Add repository (if published)
helm repo add aragora https://charts.aragora.ai
helm repo update

# Or install from local chart
helm install aragora ./deploy/helm/aragora \
  --namespace aragora \
  --create-namespace \
  -f values-production.yaml
```

### Configuration Values

```yaml
# values-production.yaml
replicaCount: 3

image:
  repository: ghcr.io/aragora/aragora
  tag: "v1.0.0"
  pullPolicy: IfNotPresent

# Enable autoscaling for production
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1

# Prometheus Operator integration
serviceMonitor:
  enabled: true
  interval: 15s
  labels:
    release: prometheus  # Match your Prometheus selector

# Resource limits
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

# External secrets (recommended for production)
existingSecrets:
  apiKeys: aragora-api-keys
  database: aragora-db-credentials
```

### Upgrading

```bash
# Check current release
helm list -n aragora

# Upgrade with new values
helm upgrade aragora ./deploy/helm/aragora \
  --namespace aragora \
  -f values-production.yaml

# Rollback if needed
helm rollback aragora 1 -n aragora
```

---

## Backup and Restore Procedures

### Automated Backups

**PostgreSQL Backups:**
```bash
# Daily backup script (add to cron)
#!/bin/bash
BACKUP_DIR=/backups/postgres
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -Fc aragora > ${BACKUP_DIR}/aragora_${DATE}.dump

# Retain last 30 days
find ${BACKUP_DIR} -name "*.dump" -mtime +30 -delete
```

**SQLite Backups (local mode):**
```bash
# Backup local databases
cp /var/lib/aragora/agent_elo.db /backups/agent_elo_$(date +%Y%m%d).db
cp /var/lib/aragora/continuum.db /backups/continuum_$(date +%Y%m%d).db
```

**Redis Backup:**
```bash
# Trigger RDB snapshot
redis-cli BGSAVE

# Copy snapshot
cp /var/lib/redis/dump.rdb /backups/redis_$(date +%Y%m%d).rdb
```

### Restore Procedures

**Full PostgreSQL Restore:**
```bash
# Stop service
systemctl stop aragora

# Drop and recreate database
psql -c "DROP DATABASE IF EXISTS aragora;"
psql -c "CREATE DATABASE aragora;"

# Restore from backup
pg_restore -d aragora /backups/aragora_20240115.dump

# Restart service
systemctl start aragora
```

**Selective Table Restore:**
```bash
# Restore specific table
pg_restore -d aragora -t debates /backups/aragora_20240115.dump
```

**Kubernetes Backup with Velero:**
```bash
# Create backup
velero backup create aragora-backup \
  --include-namespaces aragora \
  --include-cluster-resources=true

# List backups
velero backup get

# Restore from backup
velero restore create --from-backup aragora-backup
```

---

## Security Incident Response

### Incident Classification

| Severity | Response Time | Examples |
|----------|--------------|----------|
| P1 - Critical | Immediate | Data breach, service compromise |
| P2 - High | 1 hour | Auth bypass, rate limit bypass |
| P3 - Medium | 4 hours | API key leak, suspicious activity |
| P4 - Low | 24 hours | Minor vulnerability, audit finding |

### Immediate Response Checklist

**1. Containment (first 15 minutes):**
```bash
# Rotate compromised API keys
curl -X POST http://localhost:8080/api/admin/rotate-keys

# Block suspicious IPs
iptables -A INPUT -s SUSPICIOUS_IP -j DROP

# Enable emergency rate limiting
export ARAGORA_RATE_LIMIT_EMERGENCY=1
systemctl restart aragora
```

**2. Preserve Evidence:**
```bash
# Capture logs before rotation
tar -czf /evidence/logs_$(date +%Y%m%d_%H%M%S).tar.gz /var/log/aragora/

# Export database audit log
pg_dump -t audit_log aragora > /evidence/audit_$(date +%Y%m%d).sql

# Capture current connections
netstat -tupn > /evidence/connections_$(date +%Y%m%d).txt
```

**3. Revoke Access:**
```bash
# Revoke all active tokens
curl -X POST http://localhost:8080/api/admin/revoke-all-tokens

# Force password reset
curl -X POST http://localhost:8080/api/admin/force-password-reset
```

### Token Revocation

The audit logging and token revocation middleware track all sensitive operations:

```bash
# View recent audit events
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8080/api/admin/audit-log?limit=100

# Revoke specific token
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8080/api/admin/revoke-token \
  -d '{"token_id": "tok_xyz123"}'

# Check revocation status
curl http://localhost:8080/api/admin/revoked-tokens
```

### Post-Incident

1. Document timeline and actions taken
2. Notify affected users (if data breach)
3. Update security controls
4. Conduct post-mortem
5. File CVE if applicable

---

## Nomic Loop Operations

### Overview

The Nomic Loop is Aragora's autonomous self-improvement system. It runs debates to propose code improvements, designs solutions, implements changes, and verifies them.

### Starting the Nomic Loop

```bash
# Run with human approval required (safer)
python scripts/nomic_loop.py run --cycles 3 --path /path/to/aragora

# Run autonomously (for trusted environments)
python scripts/nomic_loop.py run --cycles 5 --auto

# Run with streaming output
python scripts/run_nomic_with_stream.py run --cycles 3
```

### Pre-Flight Checks

```bash
# Run health checks before starting
python scripts/nomic_loop.py preflight

# Expected output:
# [PASS] Primary API Key
# [PASS] Disk Space
# [PASS] Protected Files
# [PASS] Git Repository
# [PASS] Backup Directory
```

### Safety Features

The Nomic Loop includes multiple safety mechanisms:

1. **Constitution Verification**: Cryptographically signed safety rules
2. **Protected File Checksums**: Cannot modify critical files
3. **Automatic Backups**: Created before each cycle
4. **Pre-flight Checks**: Validates environment before running
5. **Rollback on Failure**: Automatic rollback if tests fail

### Monitoring the Loop

```bash
# Watch loop status
tail -f .nomic/sessions/*/transcript.md

# Check backup status
ls -la .nomic/backups/

# View cycle outcomes
cat .nomic/outcomes.db | sqlite3 -header -csv
```

### Rollback Procedures

```bash
# Manual rollback to backup
python scripts/nomic_loop.py rollback --backup .nomic/backups/cycle_5_20240115/

# Git-based rollback
git checkout HEAD~1  # Revert last cycle

# Full reset to clean state
git stash
git checkout main
git pull
```

### Troubleshooting

**Loop Stuck:**
```bash
# Check for hung processes
ps aux | grep nomic_loop

# Check for lock files
ls -la .nomic/*.lock

# Remove stale locks
rm .nomic/*.lock
```

**Constitution Signature Invalid:**
```bash
# Re-sign constitution (requires private key)
python scripts/sign_constitution.py sign

# Verify signature
python scripts/sign_constitution.py verify
```

---

## Production Readiness Checklist

### Pre-Launch Checklist

- [ ] **Secrets Management**
  - [ ] API keys stored in secure secret manager (not env vars)
  - [ ] JWT secret is strong and rotated periodically
  - [ ] Database credentials use least-privilege accounts

- [ ] **High Availability**
  - [ ] Minimum 2 replicas running
  - [ ] Pod Disruption Budget configured
  - [ ] Horizontal Pod Autoscaler enabled
  - [ ] Multi-AZ deployment (if cloud)

- [ ] **Monitoring**
  - [ ] ServiceMonitor created for Prometheus
  - [ ] Grafana dashboards imported
  - [ ] Alerts configured and tested
  - [ ] On-call rotation established

- [ ] **Security**
  - [ ] Rate limiting enabled
  - [ ] CORS configured correctly
  - [ ] TLS termination enabled
  - [ ] Network policies applied

- [ ] **Backup**
  - [ ] Automated database backups
  - [ ] Backup retention policy defined
  - [ ] Restore procedure tested

- [ ] **Performance**
  - [ ] Load testing completed
  - [ ] Resource limits tuned
  - [ ] Connection pooling configured

### Go-Live Checklist

```bash
# Verify all services healthy
kubectl get pods -n aragora
curl https://aragora.example.com/api/health

# Check metrics endpoint
curl https://aragora.example.com/metrics | head

# Test debate creation
curl -X POST https://aragora.example.com/api/debates \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"topic": "test", "agents": ["claude", "gpt-4"]}'

# Verify WebSocket connectivity
websocat wss://aragora.example.com/ws/debates/test-123
```

---

## Scaling Guidelines

### When to Scale

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Utilization | >70% sustained | Add replicas |
| Memory Usage | >80% | Add replicas or increase limits |
| p95 Latency | >500ms | Investigate + scale |
| Active Debates | >10 per pod | Add replicas |
| Error Rate | >1% | Investigate before scaling |

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment aragora --replicas=5 -n aragora

# Or update HPA min
kubectl patch hpa aragora -n aragora \
  -p '{"spec":{"minReplicas":3}}'
```

### Vertical vs Horizontal

- **Horizontal (add replicas)**: Preferred for most workloads
- **Vertical (increase resources)**: For memory-bound operations or large debates

---

## Contact Information

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-Call Engineer | PagerDuty | Immediate |
| Platform Team | #aragora-platform | 15 minutes |
| Security Team | security@aragora.ai | Security issues |

---

*Last updated: 2026-01-13*
