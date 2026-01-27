# Multi-Region Deployment Setup Runbook

**Purpose:** Deploy Aragora across multiple geographic regions for high availability and low latency.
**Audience:** DevOps, SRE, Platform Engineers
**Last Updated:** January 2026

---

## Overview

Multi-region deployment provides:
- Geographic redundancy (survive regional outages)
- Lower latency for global users
- Compliance with data residency requirements
- Horizontal scaling across regions

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │   Global Load Balancer      │
                    │  (CloudFlare/Route53 GLB)   │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   US-EAST-1     │     │   EU-WEST-1     │     │   AP-SOUTH-1    │
│                 │     │                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Aragora   │  │     │  │ Aragora   │  │     │  │ Aragora   │  │
│  │ Cluster   │  │     │  │ Cluster   │  │     │  │ Cluster   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │     │        │        │
│  ┌─────┴─────┐  │     │  ┌─────┴─────┐  │     │  ┌─────┴─────┐  │
│  │PostgreSQL │◀─┼─────┼──│PostgreSQL │──┼─────┼─▶│PostgreSQL │  │
│  │ (Primary) │  │     │  │ (Replica) │  │     │  │ (Replica) │  │
│  └───────────┘  │     │  └───────────┘  │     │  └───────────┘  │
│                 │     │                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │   Redis   │◀─┼─────┼──│   Redis   │──┼─────┼─▶│   Redis   │  │
│  │ (Primary) │  │     │  │ (Replica) │  │     │  │ (Replica) │  │
│  └───────────┘  │     │  └───────────┘  │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Prerequisites

| Requirement | Specification |
|-------------|---------------|
| Regions | 2+ AWS/GCP/Azure regions |
| Network | VPC peering or Transit Gateway between regions |
| Latency | < 100ms between regions |
| DNS | Route53/CloudFlare with health checks |

---

## Phase 1: Network Setup

### 1.1 VPC Peering (AWS)

```bash
# Create VPC peering between us-east-1 and eu-west-1
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-us-east-xxxxx \
    --peer-vpc-id vpc-eu-west-xxxxx \
    --peer-region eu-west-1

# Accept peering in eu-west-1
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-xxxxx \
    --region eu-west-1

# Update route tables in both VPCs
aws ec2 create-route \
    --route-table-id rtb-us-east-xxxxx \
    --destination-cidr-block 10.1.0.0/16 \
    --vpc-peering-connection-id pcx-xxxxx

aws ec2 create-route \
    --route-table-id rtb-eu-west-xxxxx \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-xxxxx \
    --region eu-west-1
```

### 1.2 Security Groups

```bash
# Allow PostgreSQL replication between regions
aws ec2 authorize-security-group-ingress \
    --group-id sg-postgres-xxxxx \
    --protocol tcp \
    --port 5432 \
    --cidr 10.1.0.0/16  # EU VPC CIDR

# Allow Redis replication
aws ec2 authorize-security-group-ingress \
    --group-id sg-redis-xxxxx \
    --protocol tcp \
    --port 6379 \
    --cidr 10.1.0.0/16
```

---

## Phase 2: Database Replication

### 2.1 PostgreSQL Cross-Region Replication

```bash
# On primary (us-east-1)
# /etc/postgresql/15/main/postgresql.conf

wal_level = replica
max_wal_senders = 10
wal_keep_size = 2GB
max_replication_slots = 10

# pg_hba.conf - allow replication from secondary regions
host replication replicator 10.1.0.0/16 scram-sha-256  # EU
host replication replicator 10.2.0.0/16 scram-sha-256  # APAC
```

```bash
# On replica (eu-west-1)
# Create base backup from primary
pg_basebackup -h primary.us-east-1.internal \
    -U replicator \
    -D /var/lib/postgresql/15/main \
    -Fp -Xs -P -R -S eu_west_slot

# postgresql.auto.conf (created by pg_basebackup -R)
primary_conninfo = 'host=primary.us-east-1.internal port=5432 user=replicator'
primary_slot_name = 'eu_west_slot'
```

### 2.2 Monitor Replication Lag

```sql
-- On primary, check all replicas
SELECT
    client_addr,
    application_name,
    state,
    pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes,
    replay_lag
FROM pg_stat_replication;

-- Alert if lag > 1MB
SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) > 1048576;
```

---

## Phase 3: Redis Cross-Region Replication

### 3.1 Redis Replica Configuration

```bash
# On replica (eu-west-1)
# /etc/redis/redis.conf

replicaof primary.us-east-1.internal 6379
masterauth your_redis_password
replica-read-only yes
replica-serve-stale-data yes  # Serve stale data during sync

# For cross-region, increase timeout
repl-timeout 120
repl-ping-replica-period 10
```

### 3.2 Redis Sentinel for Automatic Failover

```bash
# /etc/redis/sentinel.conf (on each region)

sentinel monitor aragora-redis primary.us-east-1.internal 6379 2
sentinel auth-pass aragora-redis your_redis_password
sentinel down-after-milliseconds aragora-redis 30000
sentinel failover-timeout aragora-redis 180000
sentinel parallel-syncs aragora-redis 1
```

---

## Phase 4: Application Deployment

### 4.1 Kubernetes Deployment per Region

```yaml
# deploy/k8s/multi-region/us-east-1.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora
  labels:
    region: us-east-1
spec:
  replicas: 3
  template:
    spec:
      nodeSelector:
        topology.kubernetes.io/region: us-east-1
      containers:
        - name: aragora
          image: aragora/server:latest
          env:
            - name: ARAGORA_REGION
              value: "us-east-1"
            - name: ARAGORA_PRIMARY_REGION
              value: "true"
            - name: DATABASE_URL
              value: "postgresql://app:xxx@postgres.us-east-1.internal:5432/aragora"
            - name: REDIS_URL
              value: "redis://redis.us-east-1.internal:6379"
            - name: ARAGORA_PEER_REGIONS
              value: "eu-west-1,ap-south-1"
```

### 4.2 Read Replica Routing

```python
# aragora/db/routing.py

class MultiRegionRouter:
    def __init__(self):
        self.primary_region = os.getenv("ARAGORA_PRIMARY_REGION", "us-east-1")
        self.current_region = os.getenv("ARAGORA_REGION", "us-east-1")

    def get_connection(self, operation: str) -> str:
        """Route reads to local replica, writes to primary."""
        if operation in ("SELECT", "read"):
            return self.local_replica_url
        return self.primary_url
```

---

## Phase 5: Global Load Balancer

### 5.1 CloudFlare Load Balancer

```bash
# Create origin pools for each region
curl -X POST "https://api.cloudflare.com/client/v4/user/load_balancers/pools" \
    -H "Authorization: Bearer $CF_TOKEN" \
    -d '{
        "name": "aragora-us-east",
        "origins": [{"name": "us-east-1", "address": "aragora.us-east-1.example.com"}],
        "origin_steering": {"policy": "random"}
    }'

# Create load balancer with geo routing
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/load_balancers" \
    -H "Authorization: Bearer $CF_TOKEN" \
    -d '{
        "name": "aragora.example.com",
        "fallback_pool": "aragora-us-east",
        "default_pools": ["aragora-us-east", "aragora-eu-west", "aragora-ap-south"],
        "region_pools": {
            "WNAM": ["aragora-us-east"],
            "ENAM": ["aragora-us-east"],
            "WEU": ["aragora-eu-west"],
            "EEU": ["aragora-eu-west"],
            "SEAS": ["aragora-ap-south"],
            "NEAS": ["aragora-ap-south"]
        },
        "steering_policy": "geo"
    }'
```

### 5.2 Route53 Latency-Based Routing

```bash
# Create latency record for each region
aws route53 change-resource-record-sets \
    --hosted-zone-id $ZONE_ID \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "api.aragora.example.com",
                "Type": "A",
                "SetIdentifier": "us-east-1",
                "Region": "us-east-1",
                "AliasTarget": {
                    "HostedZoneId": "Z3AADJGX6KTTL2",
                    "DNSName": "aragora-us-east-alb.us-east-1.elb.amazonaws.com",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'
```

---

## Phase 6: Failover Testing

### 6.1 Regional Failover Test

```bash
#!/bin/bash
# test_regional_failover.sh

echo "Starting regional failover test..."

# 1. Verify current primary
PRIMARY=$(psql -h primary.internal -U monitor -c "SELECT pg_is_in_recovery();" -t)
if [ "$PRIMARY" != " f" ]; then
    echo "ERROR: Primary is not in primary mode"
    exit 1
fi

# 2. Simulate primary failure (stop primary temporarily)
echo "Simulating primary failure..."
kubectl scale deployment aragora --replicas=0 -n us-east-1

# 3. Wait for failover (Sentinel should promote replica)
sleep 60

# 4. Verify new primary in eu-west-1
NEW_PRIMARY=$(psql -h replica.eu-west-1.internal -U monitor -c "SELECT pg_is_in_recovery();" -t)
if [ "$NEW_PRIMARY" == " f" ]; then
    echo "SUCCESS: EU replica promoted to primary"
else
    echo "ERROR: Failover did not occur"
fi

# 5. Restore original primary as replica
kubectl scale deployment aragora --replicas=3 -n us-east-1
```

### 6.2 Validate Data Consistency

```bash
#!/bin/bash
# validate_cross_region.sh

# Compare row counts across regions
for region in us-east-1 eu-west-1 ap-south-1; do
    count=$(psql -h postgres.$region.internal -t -c "SELECT COUNT(*) FROM debates;")
    echo "$region debates: $count"
done

# Compare checksums of critical tables
for region in us-east-1 eu-west-1 ap-south-1; do
    checksum=$(psql -h postgres.$region.internal -t -c "
        SELECT md5(string_agg(id::text, ',' ORDER BY id))
        FROM debates WHERE created_at < now() - interval '1 hour';
    ")
    echo "$region checksum: $checksum"
done
```

---

## Monitoring

### Multi-Region Metrics

```yaml
# prometheus rules

groups:
  - name: multi_region
    rules:
      - alert: CrossRegionReplicationLag
        expr: pg_replication_lag_seconds > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cross-region replication lag > 60s"

      - alert: RegionDown
        expr: up{job="aragora"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Region {{ $labels.region }} is down"
```

### Grafana Dashboard

Key panels for multi-region:
- Request latency by region
- Cross-region replication lag
- Regional traffic distribution
- Failover events timeline

---

## Troubleshooting

### High Replication Lag

```bash
# Check network latency between regions
ping -c 10 primary.us-east-1.internal

# Check WAL sender status
psql -c "SELECT * FROM pg_stat_replication;"

# Check if WAL is being archived
ls -la /var/lib/postgresql/15/main/pg_wal/

# Increase wal_keep_size if replicas fall behind
```

### Split Brain Prevention

```bash
# If both regions think they're primary:

# 1. Stop all writes immediately
kubectl scale deployment aragora --replicas=0 --all-namespaces

# 2. Determine which has more recent data
for region in us-east-1 eu-west-1; do
    echo "=== $region ==="
    psql -h postgres.$region.internal -c "SELECT pg_last_wal_receive_lsn();"
done

# 3. Keep region with higher LSN as primary
# 4. Reconfigure other as replica
```

### DNS Failover Not Working

```bash
# Check CloudFlare/Route53 health checks
curl -s "https://api.cloudflare.com/client/v4/user/load_balancers/pools" \
    -H "Authorization: Bearer $CF_TOKEN" | jq '.result[].origins[].healthy'

# Manually check endpoint health
for region in us-east-1 eu-west-1 ap-south-1; do
    curl -s -o /dev/null -w "%{http_code}" https://aragora.$region.example.com/health
done
```

---

## Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARAGORA_REGION` | Current region identifier |
| `ARAGORA_PRIMARY_REGION` | Whether this is the write primary |
| `ARAGORA_PEER_REGIONS` | Comma-separated list of peer regions |
| `DATABASE_URL` | Local PostgreSQL connection |
| `DATABASE_URL_PRIMARY` | Primary region database (for writes) |

### Related Documentation

- [RUNBOOK_POSTGRESQL_REPLICATION.md](./RUNBOOK_POSTGRESQL_REPLICATION.md)
- [redis-failover.md](./redis-failover.md)
- [DISASTER_RECOVERY.md](../DISASTER_RECOVERY.md)

---

**Document Owner:** Platform Team
**Review Cycle:** Quarterly
