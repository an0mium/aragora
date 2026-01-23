# Control Plane Deployment Guide

This guide covers deploying the Aragora Control Plane in production environments.

## Architecture Overview

```
                                    ┌─────────────────────┐
                                    │   Load Balancer     │
                                    │   (HTTPS/WSS)       │
                                    └─────────┬───────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
    ┌─────────▼──────────┐      ┌────────────▼───────────┐     ┌─────────────▼────────────┐
    │ Control Plane API  │      │  Control Plane API     │     │  Control Plane API       │
    │ (Region: us-east)  │      │  (Region: us-west)     │     │  (Region: eu-west)       │
    └─────────┬──────────┘      └────────────┬───────────┘     └─────────────┬────────────┘
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                    ┌─────────▼─────┐  ┌──────▼─────┐  ┌──────▼─────┐
                    │ Redis Cluster │  │ PostgreSQL │  │ Prometheus │
                    │ (Primary)     │  │ (Primary)  │  │ (Metrics)  │
                    └───────────────┘  └────────────┘  └────────────┘
```

## Prerequisites

### Infrastructure Requirements

| Component | Minimum | Recommended | Purpose |
|-----------|---------|-------------|---------|
| Control Plane API | 2 vCPU, 4GB RAM | 4 vCPU, 8GB RAM | API servers |
| Redis | 2 vCPU, 4GB RAM | 4 vCPU, 8GB RAM | State & coordination |
| PostgreSQL | 2 vCPU, 8GB RAM | 4 vCPU, 16GB RAM | Persistence |
| Prometheus | 1 vCPU, 2GB RAM | 2 vCPU, 4GB RAM | Metrics |

### Network Requirements

- **Internal**: TCP 6379 (Redis), TCP 5432 (PostgreSQL), TCP 9090 (Prometheus)
- **External**: TCP 443 (HTTPS), TCP 8080 (HTTP), TCP 443 (WebSocket)
- **Agent**: TCP 8080 (Agent HTTP endpoints)

## Deployment Methods

### Docker Compose (Development/Staging)

```yaml
# docker-compose.control-plane.yml
version: '3.8'

services:
  control-plane:
    image: aragora/control-plane:latest
    ports:
      - "8080:8080"
    environment:
      ARAGORA_REDIS_URL: redis://redis:6379
      ARAGORA_DATABASE_URL: postgresql://user:pass@postgres:5432/aragora
      ARAGORA_ENABLE_METRICS: "true"
      ARAGORA_PROMETHEUS_PORT: "9090"
      ARAGORA_LOG_LEVEL: info
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: aragora
    volumes:
      - postgres-data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

volumes:
  redis-data:
  postgres-data:
```

### Kubernetes (Production)

```yaml
# control-plane-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora-control-plane
  labels:
    app: aragora-control-plane
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aragora-control-plane
  template:
    metadata:
      labels:
        app: aragora-control-plane
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
      - name: control-plane
        image: aragora/control-plane:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: ARAGORA_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: aragora-secrets
              key: redis-url
        - name: ARAGORA_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aragora-secrets
              key: database-url
        - name: ARAGORA_ENABLE_METRICS
          value: "true"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: aragora-control-plane
spec:
  selector:
    app: aragora-control-plane
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aragora-control-plane
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/websocket-services: "aragora-control-plane"
spec:
  tls:
  - hosts:
    - control-plane.yourdomain.com
    secretName: aragora-tls
  rules:
  - host: control-plane.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aragora-control-plane
            port:
              number: 8080
```

## Configuration

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ARAGORA_REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `ARAGORA_DATABASE_URL` | PostgreSQL connection URL | `postgresql://user:pass@host:5432/db` |

#### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_HEARTBEAT_TIMEOUT` | `30` | Agent heartbeat timeout (seconds) |
| `ARAGORA_TASK_TIMEOUT` | `300` | Default task timeout (seconds) |
| `ARAGORA_MAX_TASK_RETRIES` | `3` | Maximum task retry attempts |
| `ARAGORA_CLEANUP_INTERVAL` | `60` | Cleanup interval (seconds) |
| `ARAGORA_ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `ARAGORA_PROMETHEUS_PORT` | `9090` | Prometheus metrics port |
| `ARAGORA_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `ARAGORA_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `ARAGORA_HTTP_POOL_SIZE` | `20` | HTTP connection pool size per provider |
| `ARAGORA_HTTP_TIMEOUT` | `60` | HTTP request timeout (seconds) |

### Redis Configuration

For production Redis clusters:

```bash
# High availability configuration
ARAGORA_REDIS_URL=redis://redis-cluster:6379
ARAGORA_REDIS_SENTINEL_MASTER=mymaster
ARAGORA_REDIS_SENTINEL_HOSTS=sentinel1:26379,sentinel2:26379,sentinel3:26379

# Connection pool settings
ARAGORA_POOL_MIN_CONNECTIONS=5
ARAGORA_POOL_MAX_CONNECTIONS=50
ARAGORA_POOL_IDLE_TIMEOUT=300
```

### TLS Configuration

```bash
# Enable TLS
ARAGORA_TLS_ENABLED=true
ARAGORA_TLS_CERT_FILE=/etc/aragora/tls/cert.pem
ARAGORA_TLS_KEY_FILE=/etc/aragora/tls/key.pem
ARAGORA_TLS_CA_FILE=/etc/aragora/tls/ca.pem
```

## Multi-Region Deployment

### Regional Configuration

```yaml
# Region: us-east-1
ARAGORA_REGION=us-east-1
ARAGORA_REGION_PRIMARY=true
ARAGORA_REPLICATION_TARGETS=us-west-2,eu-west-1

# Region: us-west-2
ARAGORA_REGION=us-west-2
ARAGORA_REGION_PRIMARY=false
ARAGORA_PRIMARY_REGION=us-east-1

# Region: eu-west-1
ARAGORA_REGION=eu-west-1
ARAGORA_REGION_PRIMARY=false
ARAGORA_PRIMARY_REGION=us-east-1
ARAGORA_DATA_RESIDENCY=eu
```

### Failover Strategy

1. **Automatic Failover**: Secondary regions detect primary failure via heartbeat timeout
2. **Leader Election**: Raft-based consensus for new primary selection
3. **State Sync**: Event-based state synchronization on recovery

```python
# Configure failover settings
ARAGORA_LEADER_ELECTION_TIMEOUT=10.0  # seconds
ARAGORA_STALE_LEADER_THRESHOLD=60.0   # seconds
ARAGORA_SYNC_INTERVAL=5.0             # seconds
```

## Monitoring

### Prometheus Metrics

Key metrics to monitor:

| Metric | Type | Description |
|--------|------|-------------|
| `aragora_cp_agents_active` | Gauge | Active agents by capability |
| `aragora_cp_tasks_submitted_total` | Counter | Total tasks submitted |
| `aragora_cp_task_duration_seconds` | Histogram | Task execution duration |
| `aragora_cp_deliberation_duration_seconds` | Histogram | Deliberation duration |
| `aragora_cp_deliberation_sla_total` | Counter | SLA compliance counts |
| `aragora_http_requests_total` | Counter | HTTP requests by endpoint |
| `aragora_http_request_duration_seconds` | Histogram | Request latency |

### Alerting Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: aragora-control-plane
  rules:
  - alert: ControlPlaneDown
    expr: up{job="aragora-control-plane"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Control Plane instance down"

  - alert: HighTaskLatency
    expr: histogram_quantile(0.99, sum(rate(aragora_cp_task_duration_seconds_bucket[5m])) by (le)) > 300
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Task P99 latency > 5 minutes"

  - alert: AgentUnhealthy
    expr: aragora_cp_agent_health_checks_total{status="unhealthy"} > 0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Agent health check failures detected"

  - alert: SLAViolation
    expr: increase(aragora_cp_deliberation_sla_total{level="violated"}[1h]) > 5
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "More than 5 SLA violations in the last hour"
```

### Grafana Dashboard

Import the control plane dashboard from `docs/dashboards/control-plane.json` or use dashboard ID `aragora-cp-001`.

## Security Considerations

### Authentication

```bash
# API authentication
ARAGORA_AUTH_ENABLED=true
ARAGORA_JWT_SECRET=your-256-bit-secret
ARAGORA_JWT_EXPIRY=3600  # 1 hour

# OAuth integration
ARAGORA_OAUTH_ENABLED=true
ARAGORA_OAUTH_PROVIDER=google
ARAGORA_OAUTH_CLIENT_ID=your-client-id
ARAGORA_OAUTH_CLIENT_SECRET=your-client-secret
```

### RBAC Configuration

```bash
# Role-based access control
ARAGORA_RBAC_ENABLED=true
ARAGORA_ADMIN_USERS=admin@company.com
ARAGORA_RBAC_DEFAULT_ROLE=viewer
```

### Network Security

1. **TLS Everywhere**: Enable TLS for all external connections
2. **Private Networking**: Keep Redis and PostgreSQL on private networks
3. **Firewall Rules**: Restrict ingress to known IP ranges
4. **mTLS for Agents**: Use mutual TLS for agent registration

## Scaling

### Horizontal Scaling

```bash
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aragora-control-plane-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aragora-control-plane
  minReplicas: 3
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

### Performance Tuning

```bash
# Connection pools
ARAGORA_HTTP_POOL_SIZE=30          # HTTP connections per provider
ARAGORA_POOL_MAX_CONNECTIONS=100   # Redis connections

# Worker configuration
ARAGORA_WORKER_CONCURRENCY=10      # Concurrent task processing
ARAGORA_BATCH_SIZE=50              # Batch processing size
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Agents not registering | Network connectivity | Check firewall rules and DNS resolution |
| Tasks stuck in pending | No available agents | Scale up agents or check agent health |
| High latency | Resource constraints | Scale up resources or optimize queries |
| State inconsistency | Network partition | Check regional connectivity and logs |

### Debug Mode

```bash
# Enable debug logging
ARAGORA_LOG_LEVEL=debug
ARAGORA_DEBUG_SQL=true  # Log SQL queries
ARAGORA_DEBUG_REDIS=true  # Log Redis commands
```

### Health Endpoints

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed component health
- `GET /metrics` - Prometheus metrics

## Backup and Recovery

### Redis Backup

```bash
# Trigger RDB snapshot
redis-cli BGSAVE

# Continuous backup with AOF
redis-cli CONFIG SET appendonly yes
```

### PostgreSQL Backup

```bash
# pg_dump for logical backup
pg_dump -h localhost -U user -d aragora > backup.sql

# Point-in-time recovery with pg_basebackup
pg_basebackup -h localhost -D /backup -U replication -P
```

### Disaster Recovery

1. **RTO Target**: < 15 minutes for failover to secondary region
2. **RPO Target**: < 1 minute with synchronous replication
3. **Backup Retention**: 30 days for compliance

## Upgrade Procedures

### Rolling Update (Zero Downtime)

```bash
# Kubernetes rolling update
kubectl set image deployment/aragora-control-plane \
  control-plane=aragora/control-plane:v2.0.0

# Monitor rollout
kubectl rollout status deployment/aragora-control-plane
```

### Blue-Green Deployment

1. Deploy new version to green environment
2. Run smoke tests on green
3. Switch traffic from blue to green
4. Monitor for issues
5. Rollback by switching back to blue if needed

## Support

- **Documentation**: https://docs.aragora.ai/control-plane
- **Issues**: https://github.com/aragora/aragora/issues
- **Enterprise Support**: support@aragora.ai
