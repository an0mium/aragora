# Redis High Availability Guide

This document covers Redis high availability configurations for Aragora production deployments.

## Table of Contents

- [Overview](#overview)
- [Option 1: Redis Sentinel](#option-1-redis-sentinel)
- [Option 2: Managed Redis](#option-2-managed-redis)
- [Failover Testing](#failover-testing)
- [Aragora Configuration](#aragora-configuration)

---

## Overview

### Why Redis HA?

Aragora uses Redis for:
- Rate limiting (token bucket)
- Session caching
- Real-time pub/sub for WebSocket events
- Temporary debate state

Without HA, a Redis failure causes:
- Rate limiting falls back to in-memory (less accurate)
- Session tokens may require re-authentication
- WebSocket events may be delayed

### Choosing an HA Strategy

| Strategy | Complexity | Failover Time | Best For |
|----------|------------|---------------|----------|
| Sentinel | Medium | 10-30 seconds | Self-managed deployments |
| Managed | Low | Provider-dependent | Cloud deployments (recommended) |

---

## Option 1: Redis Sentinel

Redis Sentinel provides automatic failover with a master-replica setup.

### Docker Compose Setup

```yaml
# docker-compose-redis-sentinel.yml
version: '3.8'

services:
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis-master-data:/data

  redis-replica-1:
    image: redis:7-alpine
    command: redis-server --replicaof redis-master 6379 --appendonly yes
    depends_on:
      - redis-master

  redis-replica-2:
    image: redis:7-alpine
    command: redis-server --replicaof redis-master 6379 --appendonly yes
    depends_on:
      - redis-master

  sentinel-1:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26379:26379"
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

  sentinel-2:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26380:26379"
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

  sentinel-3:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26381:26379"
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

volumes:
  redis-master-data:
```

### Sentinel Configuration

```conf
# sentinel.conf
port 26379
sentinel monitor aragora-master redis-master 6379 2
sentinel down-after-milliseconds aragora-master 5000
sentinel failover-timeout aragora-master 60000
sentinel parallel-syncs aragora-master 1
```

### Kubernetes Deployment

```yaml
# redis-sentinel-k8s.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-master
  namespace: aragora
spec:
  serviceName: redis-master
  replicas: 1
  selector:
    matchLabels:
      app: redis
      role: master
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes"]
        ports:
        - containerPort: 6379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-replica
  namespace: aragora
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--replicaof", "redis-master", "6379"]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-sentinel
  namespace: aragora
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: sentinel
        image: redis:7-alpine
        command: ["redis-sentinel", "/etc/redis/sentinel.conf"]
```

---

## Option 2: Managed Redis

Recommended for cloud deployments.

### AWS ElastiCache

```bash
export ARAGORA_REDIS_URL=rediss://master.aragora.xxxxx.use1.cache.amazonaws.com:6379
export ARAGORA_REDIS_PASSWORD=your-auth-token
```

### Google Cloud Memorystore

```bash
export ARAGORA_REDIS_URL=redis://10.0.0.5:6379
```

### Azure Cache for Redis

```bash
export ARAGORA_REDIS_URL=rediss://aragora.redis.cache.windows.net:6380
export ARAGORA_REDIS_PASSWORD=your-access-key
```

---

## Failover Testing

### Test Sentinel Failover

```bash
# Check current master
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# Force failover
redis-cli -p 26379 sentinel failover aragora-master

# Verify new master (wait 10-30 seconds)
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master
```

### Test Aragora During Failover

```bash
# Start a debate
curl -X POST http://localhost:8080/api/debates -d '{"topic": "test"}'

# Trigger failover
redis-cli -p 26379 sentinel failover aragora-master

# Verify debate continues
curl http://localhost:8080/api/debates/$DEBATE_ID
```

---

## Aragora Configuration

### Sentinel Connection

```bash
export ARAGORA_REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
export ARAGORA_REDIS_SENTINEL_MASTER=aragora-master
```

### Fail-Open Mode

If Redis fails completely, enable fail-open to continue operating:

```bash
export ARAGORA_RATE_LIMIT_FAIL_OPEN=true
```

---

## Monitoring

Key metrics to monitor:
- `redis_connected_slaves` - Should be >= 1
- `redis_master_link_status` - Should be "up"
- `redis_memory_used_bytes` - < 90% of max

Alerts are pre-configured in `deploy/observability/alerts.rules`.

---

*Last updated: 2026-01-13*
