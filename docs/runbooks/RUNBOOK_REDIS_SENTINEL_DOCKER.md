# Redis Sentinel Docker Setup Runbook

**Purpose:** Deploy Redis with Sentinel for high availability in Docker environments.
**Audience:** DevOps, SRE, Platform Engineers
**Last Updated:** January 2026

---

## Overview

Redis Sentinel provides:
- Automatic failover when master fails
- Monitoring of Redis instances
- Configuration provider for clients
- Notification system for failures

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Redis Sentinel Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│    │ Sentinel 1  │    │ Sentinel 2  │    │ Sentinel 3  │       │
│    │  (Quorum)   │────│  (Quorum)   │────│  (Quorum)   │       │
│    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘       │
│           │                  │                  │               │
│           └─────────────┬────┴──────────────────┘               │
│                         │                                        │
│    ┌────────────────────┼────────────────────┐                  │
│    ▼                    ▼                    ▼                  │
│ ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│ │   Redis     │   │   Redis     │   │   Redis     │            │
│ │   Master    │──▶│  Replica 1  │   │  Replica 2  │            │
│ │             │   │             │◀──│             │            │
│ └─────────────┘   └─────────────┘   └─────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    ┌─────────────────┐
                    │    Aragora      │
                    │    Server       │
                    └─────────────────┘
```

---

## Prerequisites

| Requirement | Specification |
|-------------|---------------|
| Docker | 20.10+ |
| Docker Compose | v2.0+ |
| Memory | 2GB minimum per Redis node |
| Network | Internal network between containers |

---

## Phase 1: Basic Setup

### 1.1 Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Redis Master
  redis-master:
    image: redis:7-alpine
    container_name: redis-master
    command: redis-server /etc/redis/redis.conf
    ports:
      - "6379:6379"
    volumes:
      - ./redis/master.conf:/etc/redis/redis.conf:ro
      - redis-master-data:/data
    networks:
      - redis-net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Replica 1
  redis-replica-1:
    image: redis:7-alpine
    container_name: redis-replica-1
    command: redis-server /etc/redis/redis.conf
    ports:
      - "6380:6379"
    volumes:
      - ./redis/replica.conf:/etc/redis/redis.conf:ro
      - redis-replica-1-data:/data
    networks:
      - redis-net
    depends_on:
      - redis-master
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Replica 2
  redis-replica-2:
    image: redis:7-alpine
    container_name: redis-replica-2
    command: redis-server /etc/redis/redis.conf
    ports:
      - "6381:6379"
    volumes:
      - ./redis/replica.conf:/etc/redis/redis.conf:ro
      - redis-replica-2-data:/data
    networks:
      - redis-net
    depends_on:
      - redis-master
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Sentinel 1
  sentinel-1:
    image: redis:7-alpine
    container_name: sentinel-1
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26379:26379"
    volumes:
      - ./redis/sentinel.conf:/etc/redis/sentinel.conf
    networks:
      - redis-net
    depends_on:
      - redis-master
      - redis-replica-1
      - redis-replica-2
    restart: unless-stopped

  # Sentinel 2
  sentinel-2:
    image: redis:7-alpine
    container_name: sentinel-2
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26380:26379"
    volumes:
      - ./redis/sentinel-2.conf:/etc/redis/sentinel.conf
    networks:
      - redis-net
    depends_on:
      - redis-master
      - redis-replica-1
      - redis-replica-2
    restart: unless-stopped

  # Sentinel 3
  sentinel-3:
    image: redis:7-alpine
    container_name: sentinel-3
    command: redis-sentinel /etc/redis/sentinel.conf
    ports:
      - "26381:26379"
    volumes:
      - ./redis/sentinel-3.conf:/etc/redis/sentinel.conf
    networks:
      - redis-net
    depends_on:
      - redis-master
      - redis-replica-1
      - redis-replica-2
    restart: unless-stopped

  # Aragora Server
  aragora:
    image: aragora/server:latest
    environment:
      - REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
      - REDIS_SENTINEL_MASTER=aragora-master
      - REDIS_PASSWORD=your_redis_password
    networks:
      - redis-net
    depends_on:
      - sentinel-1
      - sentinel-2
      - sentinel-3
    restart: unless-stopped

networks:
  redis-net:
    driver: bridge

volumes:
  redis-master-data:
  redis-replica-1-data:
  redis-replica-2-data:
```

### 1.2 Redis Master Configuration

```bash
# redis/master.conf

# Network
bind 0.0.0.0
port 6379
protected-mode yes

# Authentication
requirepass your_redis_password
masterauth your_redis_password

# Persistence
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Performance
tcp-keepalive 300
timeout 0

# Replication
min-replicas-to-write 1
min-replicas-max-lag 10

# Logging
loglevel notice
logfile ""
```

### 1.3 Redis Replica Configuration

```bash
# redis/replica.conf

# Network
bind 0.0.0.0
port 6379
protected-mode yes

# Authentication
requirepass your_redis_password
masterauth your_redis_password

# Replication
replicaof redis-master 6379
replica-read-only yes
replica-serve-stale-data yes

# Persistence
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Performance
tcp-keepalive 300
timeout 0

# Logging
loglevel notice
logfile ""
```

### 1.4 Sentinel Configuration

```bash
# redis/sentinel.conf

port 26379
sentinel monitor aragora-master redis-master 6379 2
sentinel auth-pass aragora-master your_redis_password
sentinel down-after-milliseconds aragora-master 5000
sentinel failover-timeout aragora-master 60000
sentinel parallel-syncs aragora-master 1

# Notification script (optional)
# sentinel notification-script aragora-master /scripts/notify.sh

# Reconfig script (optional)
# sentinel client-reconfig-script aragora-master /scripts/reconfig.sh

# Logging
logfile ""
```

---

## Phase 2: Deployment

### 2.1 Initialize Configuration

```bash
#!/bin/bash
# scripts/init_redis_sentinel.sh

set -euo pipefail

# Create directories
mkdir -p redis

# Generate master config
cat > redis/master.conf << 'EOF'
bind 0.0.0.0
port 6379
protected-mode yes
requirepass ${REDIS_PASSWORD:-your_redis_password}
masterauth ${REDIS_PASSWORD:-your_redis_password}
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
maxmemory 1gb
maxmemory-policy allkeys-lru
min-replicas-to-write 1
min-replicas-max-lag 10
EOF

# Generate replica config
cat > redis/replica.conf << 'EOF'
bind 0.0.0.0
port 6379
protected-mode yes
requirepass ${REDIS_PASSWORD:-your_redis_password}
masterauth ${REDIS_PASSWORD:-your_redis_password}
replicaof redis-master 6379
replica-read-only yes
replica-serve-stale-data yes
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
maxmemory 1gb
maxmemory-policy allkeys-lru
EOF

# Generate sentinel configs (each needs unique file for persistence)
for i in 1 2 3; do
    if [ "$i" == "1" ]; then
        filename="sentinel.conf"
    else
        filename="sentinel-${i}.conf"
    fi

    cat > "redis/${filename}" << EOF
port 26379
sentinel monitor aragora-master redis-master 6379 2
sentinel auth-pass aragora-master ${REDIS_PASSWORD:-your_redis_password}
sentinel down-after-milliseconds aragora-master 5000
sentinel failover-timeout aragora-master 60000
sentinel parallel-syncs aragora-master 1
EOF
done

echo "Redis Sentinel configuration initialized"
```

### 2.2 Start Cluster

```bash
#!/bin/bash
# scripts/start_redis_cluster.sh

set -euo pipefail

echo "Starting Redis Sentinel cluster..."

# Start Redis nodes first
docker compose up -d redis-master
sleep 5

docker compose up -d redis-replica-1 redis-replica-2
sleep 5

# Start Sentinels
docker compose up -d sentinel-1 sentinel-2 sentinel-3
sleep 5

# Verify cluster health
./scripts/check_redis_health.sh

echo "Redis Sentinel cluster started successfully"
```

### 2.3 Health Check Script

```bash
#!/bin/bash
# scripts/check_redis_health.sh

set -euo pipefail

REDIS_PASSWORD="${REDIS_PASSWORD:-your_redis_password}"

echo "Checking Redis cluster health..."

# Check master
echo -n "Master: "
docker exec redis-master redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning ping

# Check replicas
for replica in redis-replica-1 redis-replica-2; do
    echo -n "${replica}: "
    docker exec "${replica}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning ping
done

# Check replication status
echo ""
echo "Replication status:"
docker exec redis-master redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning info replication | grep -E "role|connected_slaves|slave[0-9]"

# Check Sentinel status
echo ""
echo "Sentinel status:"
docker exec sentinel-1 redis-cli -p 26379 sentinel master aragora-master | grep -E "name|ip|port|flags|num-slaves|num-other-sentinels|quorum"

# Check if quorum is met
echo ""
echo "Sentinel quorum check:"
docker exec sentinel-1 redis-cli -p 26379 sentinel ckquorum aragora-master
```

---

## Phase 3: Failover Testing

### 3.1 Simulate Master Failure

```bash
#!/bin/bash
# scripts/test_failover.sh

set -euo pipefail

REDIS_PASSWORD="${REDIS_PASSWORD:-your_redis_password}"

echo "Testing automatic failover..."

# Get current master
echo "Current master:"
docker exec sentinel-1 redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# Write test data
echo "Writing test data..."
docker exec redis-master redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning SET failover_test "before_failover"

# Stop master
echo "Stopping master..."
docker stop redis-master

# Wait for failover
echo "Waiting for failover (30 seconds)..."
sleep 30

# Check new master
echo "New master:"
NEW_MASTER=$(docker exec sentinel-1 redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master | head -1)
echo "${NEW_MASTER}"

# Verify data survived
echo "Verifying data on new master..."
if [ "${NEW_MASTER}" == "redis-replica-1" ]; then
    docker exec redis-replica-1 redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning GET failover_test
else
    docker exec redis-replica-2 redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning GET failover_test
fi

# Restart old master as replica
echo "Restarting old master..."
docker start redis-master

sleep 10

echo "Failover test complete"
./scripts/check_redis_health.sh
```

### 3.2 Manual Failover

```bash
#!/bin/bash
# scripts/manual_failover.sh

set -euo pipefail

echo "Initiating manual failover..."

# Trigger failover via Sentinel
docker exec sentinel-1 redis-cli -p 26379 sentinel failover aragora-master

echo "Failover initiated. Waiting for completion..."
sleep 15

# Verify new master
echo "New master:"
docker exec sentinel-1 redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

./scripts/check_redis_health.sh
```

---

## Phase 4: Application Integration

### 4.1 Python Client Configuration

```python
# aragora/cache/sentinel_client.py

import redis
from redis.sentinel import Sentinel
import os
import logging

logger = logging.getLogger(__name__)


class RedisSentinelClient:
    """Redis client with Sentinel support for HA."""

    def __init__(self):
        sentinel_hosts = os.getenv("REDIS_SENTINEL_HOSTS", "localhost:26379")
        self.master_name = os.getenv("REDIS_SENTINEL_MASTER", "aragora-master")
        self.password = os.getenv("REDIS_PASSWORD")

        # Parse sentinel hosts
        sentinels = []
        for host in sentinel_hosts.split(","):
            addr, port = host.strip().split(":")
            sentinels.append((addr, int(port)))

        self.sentinel = Sentinel(
            sentinels,
            socket_timeout=5.0,
            password=self.password,
        )

    def get_master(self):
        """Get connection to current master."""
        return self.sentinel.master_for(
            self.master_name,
            socket_timeout=5.0,
            password=self.password,
        )

    def get_slave(self):
        """Get connection to a replica for reads."""
        return self.sentinel.slave_for(
            self.master_name,
            socket_timeout=5.0,
            password=self.password,
        )

    async def get(self, key: str) -> str | None:
        """Read from replica."""
        try:
            client = self.get_slave()
            return client.get(key)
        except redis.ReadOnlyError:
            # Failover in progress, try master
            logger.warning("Replica read failed, falling back to master")
            client = self.get_master()
            return client.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Write to master."""
        client = self.get_master()
        return client.set(key, value, ex=ex)

    async def delete(self, key: str) -> int:
        """Delete from master."""
        client = self.get_master()
        return client.delete(key)


# Singleton instance
_client = None


def get_redis_client() -> RedisSentinelClient:
    global _client
    if _client is None:
        _client = RedisSentinelClient()
    return _client
```

### 4.2 Connection String Format

```bash
# Environment variables for Aragora

# Sentinel-based connection
REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
REDIS_SENTINEL_MASTER=aragora-master
REDIS_PASSWORD=your_redis_password

# Alternative: Direct connection (for single instance)
# REDIS_URL=redis://:your_redis_password@redis-master:6379/0
```

---

## Phase 5: Monitoring

### 5.1 Prometheus Metrics

```yaml
# prometheus/redis-exporter.yml

  redis-exporter:
    image: oliver006/redis_exporter:latest
    environment:
      - REDIS_ADDR=redis://redis-master:6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    ports:
      - "9121:9121"
    networks:
      - redis-net
```

### 5.2 Alert Rules

```yaml
# prometheus/rules/redis.yml

groups:
  - name: redis_sentinel
    rules:
      - alert: RedisSentinelDown
        expr: redis_sentinel_master_status != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis Sentinel reports master down"

      - alert: RedisReplicaLag
        expr: redis_connected_slaves < 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis has fewer than 2 connected replicas"

      - alert: RedisMasterChanged
        expr: changes(redis_sentinel_master_address[5m]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Redis master address changed (failover occurred)"

      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage above 90%"

      - alert: RedisSentinelQuorumLost
        expr: redis_sentinel_sentinels < 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis Sentinel quorum not met"
```

### 5.3 Grafana Dashboard

Key panels:
- Connected clients over time
- Memory usage per instance
- Commands per second
- Replication lag
- Sentinel failover events
- Key evictions

---

## Troubleshooting

### Sentinel Not Detecting Master

```bash
# Check Sentinel logs
docker logs sentinel-1

# Verify Sentinel can reach master
docker exec sentinel-1 redis-cli -p 26379 sentinel master aragora-master

# Check network connectivity
docker exec sentinel-1 ping redis-master

# Reset Sentinel state
docker exec sentinel-1 redis-cli -p 26379 sentinel reset aragora-master
```

### Split Brain Prevention

```bash
# If multiple masters detected:

# 1. Stop all Sentinels
docker stop sentinel-1 sentinel-2 sentinel-3

# 2. Determine true master (highest replication offset)
for node in redis-master redis-replica-1 redis-replica-2; do
    echo "=== ${node} ==="
    docker exec "${node}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning info replication
done

# 3. Configure others as replicas of true master
docker exec redis-replica-1 redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning REPLICAOF redis-master 6379

# 4. Restart Sentinels
docker start sentinel-1 sentinel-2 sentinel-3
```

### Replica Not Syncing

```bash
# Check replica status
docker exec redis-replica-1 redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning info replication

# Common issues:
# - Wrong masterauth password
# - Network connectivity
# - Disk full

# Force full resync
docker exec redis-replica-1 redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning DEBUG sleep 0
```

### Failover Not Happening

```bash
# Check Sentinel quorum
docker exec sentinel-1 redis-cli -p 26379 sentinel ckquorum aragora-master

# Check down-after-milliseconds setting
docker exec sentinel-1 redis-cli -p 26379 sentinel master aragora-master | grep down-after

# Manually trigger failover
docker exec sentinel-1 redis-cli -p 26379 sentinel failover aragora-master
```

---

## Maintenance

### Rolling Restart

```bash
#!/bin/bash
# scripts/rolling_restart.sh

set -euo pipefail

# Restart replicas first
for replica in redis-replica-2 redis-replica-1; do
    echo "Restarting ${replica}..."
    docker restart "${replica}"
    sleep 10
    docker exec "${replica}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning ping
done

# Failover to a replica
echo "Triggering failover..."
docker exec sentinel-1 redis-cli -p 26379 sentinel failover aragora-master
sleep 15

# Restart old master (now replica)
echo "Restarting old master..."
docker restart redis-master
sleep 10

# Restart Sentinels one by one
for sentinel in sentinel-3 sentinel-2 sentinel-1; do
    echo "Restarting ${sentinel}..."
    docker restart "${sentinel}"
    sleep 5
done

echo "Rolling restart complete"
./scripts/check_redis_health.sh
```

### Backup Procedure

```bash
#!/bin/bash
# scripts/backup_redis.sh

BACKUP_DIR="/backup/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "${BACKUP_DIR}"

# Trigger RDB save on master
docker exec redis-master redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning BGSAVE

# Wait for save to complete
sleep 5

# Copy RDB file
docker cp redis-master:/data/dump.rdb "${BACKUP_DIR}/dump_${DATE}.rdb"

# Verify backup
redis-check-rdb "${BACKUP_DIR}/dump_${DATE}.rdb"

echo "Backup saved to ${BACKUP_DIR}/dump_${DATE}.rdb"
```

---

## Reference

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `down-after-milliseconds` | 5000 | Time before marking node as down |
| `failover-timeout` | 60000 | Max time for failover |
| `parallel-syncs` | 1 | Replicas syncing simultaneously |
| `quorum` | 2 | Sentinels needed for failover |

### Useful Commands

```bash
# Get master address
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# List all replicas
redis-cli -p 26379 sentinel replicas aragora-master

# List all sentinels
redis-cli -p 26379 sentinel sentinels aragora-master

# Force failover
redis-cli -p 26379 sentinel failover aragora-master

# Check quorum
redis-cli -p 26379 sentinel ckquorum aragora-master
```

---

**Document Owner:** Platform Team
**Review Cycle:** Quarterly
