# Performance Baseline and Scaling Guide

Performance characteristics, benchmarks, and scaling recommendations for Aragora.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Baseline Metrics](#baseline-metrics)
- [Resource Requirements](#resource-requirements)
- [Throughput Benchmarks](#throughput-benchmarks)
- [Latency Profiles](#latency-profiles)
- [Scaling Guidelines](#scaling-guidelines)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Load Testing](#load-testing)

---

## Performance Overview

### System Characteristics

| Component | Bottleneck | Mitigation |
|-----------|------------|------------|
| Debate engine | LLM API latency | Parallel agent calls |
| Memory system | SQLite write lock | WAL mode, batching |
| WebSocket | Connection limits | Horizontal scaling |
| API handlers | CPU (JSON parsing) | Connection pooling |

### Performance Targets (SLOs)

| Metric | Target | Critical |
|--------|--------|----------|
| API p50 latency | < 100ms | < 200ms |
| API p99 latency | < 500ms | < 2000ms |
| WebSocket connect | < 200ms | < 500ms |
| Debate round time | < 30s | < 60s |
| Availability | 99.9% | 99.5% |
| Error rate | < 0.1% | < 1% |

---

## Baseline Metrics

### Single Instance Performance

Tested on: 4 vCPU, 8GB RAM, SSD storage

| Operation | Ops/sec | p50 (ms) | p99 (ms) |
|-----------|---------|----------|----------|
| Health check | 5,000 | 2 | 10 |
| List debates | 1,200 | 15 | 50 |
| Get debate | 2,500 | 8 | 30 |
| Create debate | 200 | 45 | 150 |
| Cast vote | 800 | 20 | 80 |
| WebSocket message | 3,000 | 5 | 25 |

### Debate Performance

| Agents | Rounds | Avg Duration | Memory |
|--------|--------|--------------|--------|
| 2 | 3 | 45s | 150MB |
| 3 | 3 | 60s | 200MB |
| 4 | 3 | 90s | 280MB |
| 5 | 5 | 180s | 400MB |

### Memory Usage Patterns

| Component | Idle | Per Debate | Max |
|-----------|------|------------|-----|
| Base process | 120MB | - | - |
| Debate context | - | 50MB | 200MB |
| Agent state | - | 30MB/agent | 150MB |
| Cache (fast tier) | 20MB | - | 100MB |
| Cache (medium tier) | 50MB | - | 500MB |
| WebSocket connections | - | 1MB/100 conn | - |

---

## Resource Requirements

### Minimum Requirements

| Environment | CPU | Memory | Storage | Network |
|-------------|-----|--------|---------|---------|
| Development | 2 cores | 4GB | 10GB | 10 Mbps |
| Production (small) | 4 cores | 8GB | 50GB | 100 Mbps |
| Production (medium) | 8 cores | 16GB | 100GB | 1 Gbps |
| Production (large) | 16 cores | 32GB | 500GB | 10 Gbps |

### Capacity Planning

| Concurrent Users | Instances | CPU (total) | Memory (total) |
|------------------|-----------|-------------|----------------|
| 100 | 1 | 4 cores | 8GB |
| 500 | 2 | 8 cores | 16GB |
| 2,000 | 4 | 16 cores | 32GB |
| 10,000 | 8 | 32 cores | 64GB |
| 50,000 | 16 | 64 cores | 128GB |

### Database Sizing

| Debates | Users | Database Size | Recommended Config |
|---------|-------|---------------|-------------------|
| 1,000 | 100 | 50MB | SQLite |
| 10,000 | 1,000 | 500MB | SQLite + WAL |
| 100,000 | 10,000 | 5GB | PostgreSQL |
| 1,000,000 | 100,000 | 50GB | PostgreSQL + replicas |

---

## Throughput Benchmarks

### API Throughput

```
Benchmark: API Handler Throughput
Hardware: 4 vCPU, 8GB RAM, 100 concurrent connections

Endpoint                    RPS     p50(ms)  p99(ms)  Errors
────────────────────────────────────────────────────────────
GET /api/health            5,247      1.8      8.2     0.00%
GET /api/debates           1,156     14.2     52.1     0.01%
GET /api/debates/:id       2,489      7.9     31.4     0.00%
POST /api/debates            187     48.2    162.3     0.02%
POST /api/debates/:id/vote   823     19.1     78.6     0.01%
GET /api/leaderboard         892     18.4     65.2     0.00%
GET /api/agents              645     24.8     89.1     0.00%
WS /ws (connect)           1,024      9.2     42.1     0.05%
WS /ws (message)           3,156      4.1     18.9     0.00%
```

### Debate Throughput

```
Benchmark: Concurrent Debates
Hardware: 8 vCPU, 16GB RAM

Concurrent Debates    Duration(avg)    Memory    CPU    Errors
────────────────────────────────────────────────────────────────
1                     45s              280MB     15%    0.0%
5                     52s              650MB     45%    0.0%
10                    68s             1.2GB     72%    0.1%
20                    95s             2.1GB     89%    0.5%
50                   145s             4.8GB     98%    2.1%
```

### LLM Provider Latency

| Provider | Model | p50 (ms) | p99 (ms) | Rate Limit |
|----------|-------|----------|----------|------------|
| Anthropic | claude-sonnet | 2,500 | 8,000 | 60 RPM |
| Anthropic | claude-haiku | 1,200 | 4,000 | 100 RPM |
| OpenAI | gpt-4o | 3,000 | 12,000 | 60 RPM |
| OpenAI | gpt-4o-mini | 800 | 3,000 | 200 RPM |
| Google | gemini-pro | 2,000 | 7,000 | 60 RPM |
| Mistral | mistral-large | 2,200 | 8,500 | 50 RPM |

---

## Latency Profiles

### Request Latency Breakdown

```
POST /api/debates (create debate) - 48ms average

Component              Duration    %
──────────────────────────────────────
Auth middleware         2ms       4%
Request parsing         3ms       6%
Validation              2ms       4%
Database read           5ms      10%
Business logic          8ms      17%
Database write         15ms      31%
Response serialize      3ms       6%
Network overhead       10ms      21%
```

### Debate Round Latency

```
Single debate round - 15s average (3 agents)

Component              Duration    %
──────────────────────────────────────
Context preparation    500ms      3%
Agent 1 LLM call      4,500ms    30%
Agent 2 LLM call      4,200ms    28%
Agent 3 LLM call      4,800ms    32%
Consensus check        200ms      1%
Memory update          300ms      2%
Event broadcast        100ms      1%
Overhead               400ms      3%
```

### WebSocket Latency

| Event Type | Publish (ms) | Delivery (ms) | Total (ms) |
|------------|--------------|---------------|------------|
| debate_start | 2 | 5 | 7 |
| agent_message | 3 | 8 | 11 |
| round_complete | 2 | 6 | 8 |
| vote_cast | 1 | 4 | 5 |
| debate_end | 2 | 5 | 7 |

---

## Scaling Guidelines

### Horizontal Scaling

```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aragora-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aragora
  minReplicas: 2
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### Database Scaling

#### SQLite (Single Instance)

```bash
# Optimize for concurrent reads
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;  # 64MB cache
PRAGMA busy_timeout=5000;
```

#### PostgreSQL (Multi-Instance)

```yaml
# Connection pooling with PgBouncer
[databases]
aragora = host=postgres port=5432 dbname=aragora

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
min_pool_size = 10
reserve_pool_size = 10
```

### Redis Scaling

```yaml
# Redis Cluster for high availability
# Minimum 3 masters + 3 replicas
redis-cli --cluster create \
  redis-1:6379 redis-2:6379 redis-3:6379 \
  redis-4:6379 redis-5:6379 redis-6:6379 \
  --cluster-replicas 1
```

### Load Balancer Configuration

```nginx
# NGINX configuration for Aragora
upstream aragora_api {
    least_conn;
    server api-1:8080 weight=1;
    server api-2:8080 weight=1;
    server api-3:8080 weight=1;
    keepalive 32;
}

upstream aragora_ws {
    ip_hash;  # Sticky sessions for WebSocket
    server ws-1:8765;
    server ws-2:8765;
}

server {
    listen 443 ssl http2;
    server_name aragora.example.com;

    location /api {
        proxy_pass http://aragora_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    location /ws {
        proxy_pass http://aragora_ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }
}
```

---

## Performance Tuning

### Application Settings

```bash
# Performance-optimized configuration
# .env

# Connection limits
ARAGORA_MAX_CONNECTIONS=1000
ARAGORA_CONNECTION_TIMEOUT=30

# Worker configuration
ARAGORA_WORKERS=4  # Generally num_cpus
ARAGORA_THREAD_POOL_SIZE=20

# Cache settings
ARAGORA_CACHE_MAX_ENTRIES=10000
ARAGORA_CACHE_TTL_FAST=60
ARAGORA_CACHE_TTL_MEDIUM=3600

# Database settings
ARAGORA_DB_POOL_SIZE=20
ARAGORA_DB_MAX_OVERFLOW=10
ARAGORA_DB_POOL_TIMEOUT=30

# Rate limiting (per minute)
ARAGORA_RATE_LIMIT_DEFAULT=100
ARAGORA_RATE_LIMIT_AUTH=10
ARAGORA_RATE_LIMIT_DEBATES=30
```

### Memory Optimization

```python
# Reduce memory footprint
import gc

# Configure garbage collection for high-throughput
gc.set_threshold(50000, 500, 100)

# Limit debate context size
MAX_CONTEXT_TOKENS = 8000
MAX_HISTORY_ROUNDS = 5

# Compress stored data
import zlib
compressed = zlib.compress(json.dumps(data).encode())
```

### Database Optimization

```sql
-- Essential indexes
CREATE INDEX idx_debates_created_at ON debates(created_at DESC);
CREATE INDEX idx_debates_status ON debates(status);
CREATE INDEX idx_votes_debate_id ON votes(debate_id);
CREATE INDEX idx_messages_debate_round ON messages(debate_id, round);
CREATE INDEX idx_users_email ON users(email);

-- Analyze tables periodically
ANALYZE debates;
ANALYZE votes;
ANALYZE messages;

-- Vacuum to reclaim space
VACUUM ANALYZE;
```

### Network Optimization

```bash
# Linux kernel tuning for high connections
# /etc/sysctl.conf

# Increase connection tracking
net.netfilter.nf_conntrack_max = 1000000

# Increase socket buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

# Increase backlog
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# Enable TCP fastopen
net.ipv4.tcp_fastopen = 3

# Reuse TIME_WAIT connections
net.ipv4.tcp_tw_reuse = 1
```

---

## Monitoring

### Key Metrics

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| `aragora_request_duration_seconds` | Histogram | p99 > 2s |
| `aragora_request_total` | Counter | Error rate > 1% |
| `aragora_active_debates` | Gauge | > 100 (capacity) |
| `aragora_db_connections` | Gauge | > 80% pool |
| `aragora_memory_bytes` | Gauge | > 80% limit |
| `aragora_cpu_usage` | Gauge | > 80% sustained |
| `aragora_ws_connections` | Gauge | > 1000/instance |

### Prometheus Queries

```promql
# Request rate
rate(aragora_request_total[5m])

# Error rate
sum(rate(aragora_request_total{status=~"5.."}[5m]))
  / sum(rate(aragora_request_total[5m]))

# p99 latency
histogram_quantile(0.99, rate(aragora_request_duration_seconds_bucket[5m]))

# Active debates
aragora_active_debates

# Memory usage percentage
aragora_memory_bytes / aragora_memory_limit_bytes * 100
```

### Grafana Dashboard

```json
{
  "title": "Aragora Performance",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [{"expr": "rate(aragora_request_total[1m])"}]
    },
    {
      "title": "Latency p50/p99",
      "type": "graph",
      "targets": [
        {"expr": "histogram_quantile(0.5, rate(aragora_request_duration_seconds_bucket[5m]))"},
        {"expr": "histogram_quantile(0.99, rate(aragora_request_duration_seconds_bucket[5m]))"}
      ]
    },
    {
      "title": "Error Rate",
      "type": "stat",
      "targets": [{"expr": "sum(rate(aragora_request_total{status=~\"5..\"}[5m])) / sum(rate(aragora_request_total[5m])) * 100"}]
    },
    {
      "title": "Active Debates",
      "type": "gauge",
      "targets": [{"expr": "aragora_active_debates"}]
    }
  ]
}
```

---

## Load Testing

### Locust Configuration

```python
# locustfile.py
from locust import HttpUser, task, between

class AragoraUser(HttpUser):
    wait_time = between(1, 3)

    @task(10)
    def health_check(self):
        self.client.get("/api/health")

    @task(5)
    def list_debates(self):
        self.client.get("/api/debates?limit=20")

    @task(3)
    def get_debate(self):
        self.client.get("/api/debates/test-debate-id")

    @task(1)
    def create_debate(self):
        self.client.post("/api/debates", json={
            "topic": "Test debate",
            "agents": ["demo", "demo"],
            "rounds": 2
        })
```

### Running Load Tests

```bash
# Install locust
pip install locust

# Run load test
locust -f locustfile.py --host=http://localhost:8080

# Headless mode for CI
locust -f locustfile.py \
  --host=http://localhost:8080 \
  --users=100 \
  --spawn-rate=10 \
  --run-time=5m \
  --headless \
  --csv=results
```

### k6 Alternative

```javascript
// k6-script.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up
    { duration: '5m', target: 100 },  // Sustain
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const res = http.get('http://localhost:8080/api/health');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  sleep(1);
}
```

```bash
# Run k6 test
k6 run k6-script.js
```

### CI Integration

```yaml
# .github/workflows/load-test.yml
name: Load Tests

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start Aragora
        run: |
          docker-compose up -d
          sleep 10

      - name: Run load test
        run: |
          pip install locust
          locust -f tests/load/locustfile.py \
            --host=http://localhost:8080 \
            --users=50 \
            --spawn-rate=5 \
            --run-time=2m \
            --headless \
            --csv=results

      - name: Check SLOs
        run: |
          # Verify p99 < 500ms
          python -c "
          import csv
          with open('results_stats.csv') as f:
              reader = csv.DictReader(f)
              for row in reader:
                  if float(row['99%']) > 500:
                      exit(1)
          "

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: load-test-results
          path: results_*.csv
```

---

## Bottleneck Analysis

### Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High CPU, low latency | Good utilization | Scale horizontally if needed |
| High CPU, high latency | CPU bound | Optimize code, add instances |
| Low CPU, high latency | I/O bound | Check database, LLM APIs |
| Memory growth | Memory leak | Profile with memray |
| Connection errors | Pool exhausted | Increase pool size |
| Timeouts | Slow LLM responses | Add circuit breaker, timeout |

### Profiling

```bash
# CPU profiling with py-spy
pip install py-spy
py-spy record -o profile.svg -- aragora serve --api-port 8080 --ws-port 8765

# Memory profiling with memray
pip install memray
memray run aragora serve --api-port 8080 --ws-port 8765
memray flamegraph memray-*.bin

# Database query analysis
sqlite3 .nomic/aragora_debates.db "EXPLAIN QUERY PLAN SELECT * FROM debates WHERE status='active'"
```

---

## Related Documentation

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Performance troubleshooting
- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - Capacity planning for DR
- [RUNBOOK.md](RUNBOOK.md) - Operational procedures
- [DATABASE.md](DATABASE.md) - Database optimization
