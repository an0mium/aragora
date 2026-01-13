# Redis High Availability Setup

Guide for configuring Redis with high availability for Aragora production deployments.

## Overview

Aragora uses Redis for:
- **Rate limiting** - Distributed token bucket across instances
- **Token blacklist** - JWT revocation across replicas
- **Session storage** - Shared session state
- **Circuit breaker state** - Agent failure tracking

Redis availability directly impacts these features. This guide covers HA options.

---

## Option 1: Redis Sentinel (Self-Managed)

Redis Sentinel provides automatic failover for Redis master-replica setups.

### Architecture

```
                    ┌──────────────┐
                    │   Sentinel   │
                    │   (quorum)   │
                    │  3 instances │
                    └──────┬───────┘
                           │ monitors
           ┌───────────────┼───────────────┐
           │               │               │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │   Redis   │   │   Redis   │   │   Redis   │
     │  Master   │──▶│  Replica  │   │  Replica  │
     │  :6379    │   │  :6380    │   │  :6381    │
     └───────────┘   └───────────┘   └───────────┘
           │
           │ writes
           ▼
     ┌───────────┐
     │  Aragora  │
     │  Server   │
     └───────────┘
```

### Kubernetes Deployment

```yaml
# deploy/k8s/redis/sentinel-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: aragora
spec:
  serviceName: redis
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      initContainers:
        - name: config
          image: redis:7-alpine
          command: ["sh", "-c"]
          args:
            - |
              if [ "$(hostname)" = "redis-0" ]; then
                cp /mnt/config/master.conf /etc/redis/redis.conf
              else
                cp /mnt/config/replica.conf /etc/redis/redis.conf
              fi
          volumeMounts:
            - name: config
              mountPath: /mnt/config
            - name: redis-config
              mountPath: /etc/redis
      containers:
        - name: redis
          image: redis:7-alpine
          command: ["redis-server", "/etc/redis/redis.conf"]
          ports:
            - containerPort: 6379
              name: redis
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          volumeMounts:
            - name: redis-config
              mountPath: /etc/redis
            - name: data
              mountPath: /data
          livenessProbe:
            exec:
              command: ["redis-cli", "ping"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["redis-cli", "ping"]
            initialDelaySeconds: 5
            periodSeconds: 5
        - name: sentinel
          image: redis:7-alpine
          command: ["redis-sentinel", "/etc/sentinel/sentinel.conf"]
          ports:
            - containerPort: 26379
              name: sentinel
          volumeMounts:
            - name: sentinel-config
              mountPath: /etc/sentinel
      volumes:
        - name: config
          configMap:
            name: redis-config
        - name: redis-config
          emptyDir: {}
        - name: sentinel-config
          configMap:
            name: sentinel-config
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: aragora
spec:
  clusterIP: None
  ports:
    - port: 6379
      name: redis
    - port: 26379
      name: sentinel
  selector:
    app: redis
---
apiVersion: v1
kind: Service
metadata:
  name: redis-master
  namespace: aragora
spec:
  ports:
    - port: 6379
      name: redis
  selector:
    app: redis
    role: master
```

### Redis Configuration

```ini
# redis-master.conf
bind 0.0.0.0
port 6379
dir /data
appendonly yes
appendfsync everysec

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru

# Security
requirepass ${REDIS_PASSWORD}
masterauth ${REDIS_PASSWORD}
```

```ini
# redis-replica.conf
bind 0.0.0.0
port 6379
dir /data
appendonly yes
appendfsync everysec

# Replication
replicaof redis-0.redis.aragora.svc.cluster.local 6379
masterauth ${REDIS_PASSWORD}
requirepass ${REDIS_PASSWORD}

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru
```

```ini
# sentinel.conf
port 26379
sentinel monitor aragora-master redis-0.redis.aragora.svc.cluster.local 6379 2
sentinel auth-pass aragora-master ${REDIS_PASSWORD}
sentinel down-after-milliseconds aragora-master 5000
sentinel failover-timeout aragora-master 60000
sentinel parallel-syncs aragora-master 1
```

### Aragora Configuration

```bash
# Connect via Sentinel
ARAGORA_REDIS_SENTINEL_MASTER=aragora-master
ARAGORA_REDIS_SENTINEL_HOSTS=redis-0.redis:26379,redis-1.redis:26379,redis-2.redis:26379
ARAGORA_REDIS_PASSWORD=${REDIS_PASSWORD}

# Or connect directly (not recommended for HA)
ARAGORA_REDIS_URL=redis://:${REDIS_PASSWORD}@redis-master.aragora.svc:6379/0
```

### Failover Testing

```bash
# Check current master
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# Simulate master failure
kubectl exec -it redis-0 -n aragora -- redis-cli DEBUG SLEEP 30

# Watch failover
kubectl exec -it redis-1 -n aragora -- redis-cli -p 26379 sentinel master aragora-master

# Verify new master
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master
```

---

## Option 2: Managed Redis (Recommended for Production)

Cloud-managed Redis services provide automatic HA, backups, and scaling.

### AWS ElastiCache

**Setup:**
1. Create ElastiCache cluster with Multi-AZ enabled
2. Enable automatic failover
3. Configure security groups for VPC access

```bash
# Configuration
ARAGORA_REDIS_URL=rediss://master.aragora-redis.abcd.use1.cache.amazonaws.com:6379

# With auth token
ARAGORA_REDIS_URL=rediss://:${ELASTICACHE_AUTH_TOKEN}@master.aragora-redis.abcd.use1.cache.amazonaws.com:6379
```

**Terraform:**
```hcl
resource "aws_elasticache_replication_group" "aragora" {
  replication_group_id       = "aragora-redis"
  description                = "Aragora Redis cluster"
  node_type                  = "cache.t3.micro"
  num_cache_clusters         = 2
  port                       = 6379

  automatic_failover_enabled = true
  multi_az_enabled           = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token

  subnet_group_name          = aws_elasticache_subnet_group.aragora.name
  security_group_ids         = [aws_security_group.redis.id]

  snapshot_retention_limit   = 7
  snapshot_window            = "05:00-09:00"
  maintenance_window         = "sun:05:00-sun:09:00"

  tags = {
    Environment = "production"
    Service     = "aragora"
  }
}
```

### GCP Memorystore

**Setup:**
1. Create Memorystore for Redis with Standard tier (HA)
2. Configure VPC connector for access

```bash
# Configuration
ARAGORA_REDIS_URL=redis://10.0.0.3:6379

# In Private Services Access VPC
```

**Terraform:**
```hcl
resource "google_redis_instance" "aragora" {
  name           = "aragora-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 1
  region         = "us-central1"

  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version     = "REDIS_7_0"
  display_name      = "Aragora Redis HA"

  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWELVE_HOURS"
  }

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 5
        minutes = 0
      }
    }
  }
}
```

### Azure Cache for Redis

**Setup:**
1. Create Azure Cache for Redis with Premium tier
2. Enable geo-replication for DR

```bash
# Configuration
ARAGORA_REDIS_URL=rediss://:${AZURE_REDIS_KEY}@aragora.redis.cache.windows.net:6380
```

---

## Option 3: Redis Cluster (High Scale)

For very high throughput requirements, Redis Cluster provides horizontal scaling.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Redis Cluster                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │              │
│  │ slots    │  │ slots    │  │ slots    │              │
│  │ 0-5460   │  │ 5461-10922│ │10923-16383│             │
│  │          │  │          │  │          │              │
│  │ Master   │  │ Master   │  │ Master   │              │
│  │  + 1     │  │  + 1     │  │  + 1     │              │
│  │ Replica  │  │ Replica  │  │ Replica  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Configuration

```bash
# Cluster mode
ARAGORA_REDIS_CLUSTER_NODES=redis-0:6379,redis-1:6379,redis-2:6379
ARAGORA_REDIS_CLUSTER_MODE=true
```

**Note:** Aragora's rate limiting uses simple key-value operations compatible with Redis Cluster. However, Lua scripts used for atomic operations must ensure all keys hash to the same slot (use `{prefix}` notation).

---

## Fail-Open Configuration

When Redis is unavailable, Aragora can optionally allow requests through:

```bash
# Allow requests when Redis is down (fail-open)
ARAGORA_RATE_LIMIT_FAIL_OPEN=true

# Log Redis failures
ARAGORA_LOG_LEVEL=WARNING
```

**Behavior:**
- Rate limiting falls back to in-memory (per-instance)
- Token blacklist may not sync across instances
- Circuit breaker state may diverge

**Warning:** Fail-open mode reduces security guarantees. Only enable if availability is critical.

---

## Monitoring Redis HA

### Prometheus Metrics

```yaml
# ServiceMonitor for Redis exporter
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: redis
  namespace: aragora
spec:
  selector:
    matchLabels:
      app: redis-exporter
  endpoints:
    - port: metrics
      interval: 15s
```

### Key Metrics

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `redis_connected_clients` | >100 | Connection pressure |
| `redis_memory_used_bytes` | >80% of max | Memory pressure |
| `redis_replication_lag` | >1s | Replica falling behind |
| `redis_master_link_status` | 0 | Replication broken |
| `redis_sentinel_master_status` | != 1 | Sentinel failover issue |

### Health Check Script

```bash
#!/bin/bash
# redis_health.sh

SENTINEL_HOST=${1:-redis-0.redis:26379}
MASTER_NAME=${2:-aragora-master}

# Check Sentinel
sentinel_status=$(redis-cli -h $SENTINEL_HOST -p 26379 ping 2>&1)
if [ "$sentinel_status" != "PONG" ]; then
  echo "CRITICAL: Sentinel not responding"
  exit 2
fi

# Check Master
master_info=$(redis-cli -h $SENTINEL_HOST -p 26379 sentinel master $MASTER_NAME 2>&1)
if echo "$master_info" | grep -q "ODOWN"; then
  echo "CRITICAL: Master is down"
  exit 2
fi

# Check replication
num_slaves=$(redis-cli -h $SENTINEL_HOST -p 26379 sentinel master $MASTER_NAME | grep num-slaves | awk '{print $2}')
if [ "$num_slaves" -lt 1 ]; then
  echo "WARNING: No replicas available"
  exit 1
fi

echo "OK: Redis HA healthy (master + $num_slaves replicas)"
exit 0
```

---

## Troubleshooting

### Failover Not Happening

```bash
# Check Sentinel logs
kubectl logs -l app=redis -c sentinel -n aragora

# Check quorum
redis-cli -p 26379 sentinel ckquorum aragora-master

# Force failover
redis-cli -p 26379 sentinel failover aragora-master
```

### Split-Brain Prevention

Ensure:
1. Sentinel quorum > replicas/2 (e.g., 2 out of 3)
2. Network partitions detected via `down-after-milliseconds`
3. `min-replicas-to-write 1` on master prevents writes without replicas

### Connection Issues

```bash
# Test connectivity from Aragora pod
kubectl exec -it deployment/aragora -n aragora -- redis-cli -h redis-master ping

# Check DNS resolution
kubectl exec -it deployment/aragora -n aragora -- nslookup redis-master

# Verify password
kubectl exec -it deployment/aragora -n aragora -- redis-cli -h redis-master -a $REDIS_PASSWORD ping
```

---

## Migration from Single Redis

1. **Add replica** to existing Redis
2. **Deploy Sentinel** pointing to current master
3. **Update Aragora config** to use Sentinel
4. **Test failover** in staging
5. **Promote replica** and retire old master

```bash
# Step 1: Add replica
kubectl apply -f redis-replica.yaml

# Step 2: Wait for sync
redis-cli -h redis-replica INFO replication | grep master_sync_in_progress

# Step 3: Deploy Sentinel
kubectl apply -f sentinel-deployment.yaml

# Step 4: Update Aragora
kubectl set env deployment/aragora \
  ARAGORA_REDIS_SENTINEL_MASTER=aragora-master \
  ARAGORA_REDIS_SENTINEL_HOSTS=sentinel:26379

# Step 5: Verify
kubectl exec -it deployment/aragora -- redis-cli -h redis-master ping
```

---

## Related Documentation

- [RUNBOOK.md](RUNBOOK.md#redis-cluster-failover) - Redis recovery procedures
- [DEPLOYMENT.md](DEPLOYMENT.md#redis-for-shared-state) - Basic Redis setup
- [ENVIRONMENT.md](ENVIRONMENT.md#rate-limiting) - Redis configuration variables
