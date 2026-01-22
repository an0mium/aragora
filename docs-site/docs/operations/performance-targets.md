---
title: Performance Targets & SLO Definitions
description: Performance Targets & SLO Definitions
---

# Performance Targets & SLO Definitions

This document defines Service Level Objectives (SLOs), performance targets, and benchmarks for Aragora.

## Service Level Objectives (SLOs)

### Availability

| Service | Target | Measurement Window |
|---------|--------|-------------------|
| API Endpoints | 99.9% | Monthly |
| WebSocket Connections | 99.5% | Monthly |
| Debate Orchestration | 99.0% | Monthly |
| Background Jobs | 98.0% | Monthly |

**Calculation:**
```
Availability = (Total Minutes - Downtime Minutes) / Total Minutes × 100
```

### Latency Targets

| Operation | P50 | P95 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Health Check | 5ms | 20ms | 50ms | 1s |
| Authentication | 50ms | 150ms | 300ms | 2s |
| Simple API Call | 100ms | 300ms | 500ms | 5s |
| Debate Start | 500ms | 1.5s | 3s | 10s |
| Debate Round | 2s | 5s | 10s | 30s |
| Full Debate | 30s | 60s | 120s | 300s |
| Search/Query | 200ms | 500ms | 1s | 5s |
| Export (small) | 500ms | 2s | 5s | 30s |
| Export (large) | 5s | 15s | 30s | 120s |

### Throughput Targets

| Metric | Target | Burst Capacity |
|--------|--------|----------------|
| API Requests/sec | 1,000 | 5,000 |
| WebSocket Connections | 10,000 | 25,000 |
| Concurrent Debates | 100 | 500 |
| Messages/sec (streaming) | 5,000 | 20,000 |

### Error Rate Targets

| Category | Target | Alert Threshold |
|----------|--------|-----------------|
| 5xx Errors | < 0.1% | > 0.5% |
| 4xx Errors | < 5% | > 10% |
| Timeout Errors | < 0.5% | > 2% |
| Connection Errors | < 0.1% | > 1% |

## Resource Utilization Targets

### Compute Resources

| Resource | Normal | Warning | Critical |
|----------|--------|---------|----------|
| CPU Usage | < 60% | > 75% | > 90% |
| Memory Usage | < 70% | > 80% | > 95% |
| Disk I/O | < 50% | > 70% | > 90% |
| Network I/O | < 60% | > 80% | > 95% |

### Database Performance

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Query Time (P95) | < 100ms | > 500ms |
| Connection Pool Usage | < 70% | > 90% |
| Replication Lag | < 1s | > 5s |
| Cache Hit Ratio | > 90% | < 80% |

### Redis Performance

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Command Latency (P99) | < 5ms | > 20ms |
| Memory Usage | < 80% | > 90% |
| Connected Clients | < 1000 | > 5000 |
| Eviction Rate | 0 | > 100/s |

## Performance Benchmarks

### API Endpoint Benchmarks

```bash
# Run API benchmarks
python scripts/benchmark_api.py --endpoint /api/health --requests 10000
python scripts/benchmark_api.py --endpoint /api/debates --requests 1000
```

**Expected Results:**

| Endpoint | RPS | P50 | P99 |
|----------|-----|-----|-----|
| GET /api/health | 5000 | 2ms | 10ms |
| GET /api/debates | 500 | 50ms | 200ms |
| POST /api/debates | 100 | 500ms | 2s |
| GET /api/consensus | 300 | 100ms | 400ms |
| GET /api/leaderboard | 200 | 150ms | 500ms |

### Debate Orchestration Benchmarks

```bash
# Run debate benchmarks
python scripts/benchmark_debate.py --rounds 3 --agents 4 --iterations 10
```

**Expected Results:**

| Configuration | Avg Time | P95 Time | Success Rate |
|---------------|----------|----------|--------------|
| 3 rounds, 4 agents | 45s | 75s | 99% |
| 5 rounds, 4 agents | 75s | 120s | 98% |
| 3 rounds, 8 agents | 90s | 150s | 97% |
| 5 rounds, 8 agents | 150s | 240s | 95% |

### WebSocket Benchmarks

```bash
# Run WebSocket benchmarks
python tests/load/websocket_load.py --connections 1000 --duration 60
```

**Expected Results:**

| Connections | Message Rate | Latency P99 | Memory/Conn |
|-------------|--------------|-------------|-------------|
| 100 | 1000 msg/s | 50ms | 50KB |
| 1000 | 5000 msg/s | 100ms | 45KB |
| 5000 | 10000 msg/s | 200ms | 40KB |
| 10000 | 15000 msg/s | 500ms | 35KB |

## Capacity Planning

### Scaling Thresholds

| Metric | Scale Up | Scale Down | Cooldown |
|--------|----------|------------|----------|
| CPU > 70% | +2 pods | CPU < 30% | 5 min |
| Memory > 80% | +2 pods | Memory < 40% | 5 min |
| RPS > 800 | +1 pod | RPS < 400 | 10 min |
| Queue Depth > 100 | +2 pods | Queue < 20 | 5 min |

### Resource Requirements

| Component | Min CPU | Min Memory | Recommended CPU | Recommended Memory |
|-----------|---------|------------|-----------------|-------------------|
| API Server | 0.5 | 512MB | 2 | 2GB |
| Worker | 0.25 | 256MB | 1 | 1GB |
| Redis | 0.5 | 1GB | 2 | 4GB |
| PostgreSQL | 1 | 2GB | 4 | 8GB |

### Traffic Projections

| User Count | API RPS | WS Connections | Debates/hour | Infrastructure |
|------------|---------|----------------|--------------|----------------|
| 100 | 50 | 100 | 10 | 2 pods |
| 1,000 | 300 | 500 | 50 | 4 pods |
| 10,000 | 1,500 | 3,000 | 200 | 10 pods |
| 100,000 | 5,000 | 15,000 | 500 | 25 pods |

## Performance Monitoring

### Key Metrics Dashboard

```yaml
# Grafana dashboard panels
panels:
  - title: Request Latency (P99)
    query: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

  - title: Error Rate
    query: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

  - title: Active Debates
    query: aragora_active_debates

  - title: WebSocket Connections
    query: aragora_websocket_connections

  - title: Agent Response Time
    query: histogram_quantile(0.95, rate(agent_response_duration_seconds_bucket[5m]))
```

### Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: performance
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"

      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 1%"

      - alert: DebateTimeout
        expr: rate(debate_timeouts_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Debate timeouts increasing"
```

## Performance Testing

### Load Test Suite

```bash
# Run full load test suite
pytest tests/load/ -v --tb=short

# Specific tests
pytest tests/load/test_concurrent_debates.py -v
pytest tests/load/gauntlet_load.py -v
pytest tests/load/websocket_load.py -v
```

### Stress Test Procedure

1. **Baseline Measurement**
   ```bash
   python scripts/benchmark_api.py --baseline
   ```

2. **Gradual Load Increase**
   ```bash
   for load in 100 500 1000 2000 5000; do
     python scripts/load_test.py --rps $load --duration 60
     sleep 30
   done
   ```

3. **Spike Test**
   ```bash
   python scripts/load_test.py --spike --peak-rps 10000 --duration 30
   ```

4. **Endurance Test**
   ```bash
   python scripts/load_test.py --rps 500 --duration 3600
   ```

### Results Documentation

After each load test, document:
- Test configuration
- Peak metrics achieved
- Bottlenecks identified
- Recommendations

## Performance Optimization Checklist

### Application Level

- [ ] Enable response caching for static data
- [ ] Use connection pooling for databases
- [ ] Implement request batching where applicable
- [ ] Enable gzip compression
- [ ] Use async I/O for external calls

### Database Level

- [ ] Index optimization (explain analyze)
- [ ] Query optimization (N+1 prevention)
- [ ] Connection pool tuning
- [ ] Read replica utilization
- [ ] Cache frequently accessed data

### Infrastructure Level

- [ ] Horizontal pod autoscaling configured
- [ ] Resource requests/limits set
- [ ] CDN for static assets
- [ ] Redis cluster for high availability
- [ ] Database connection pooler (PgBouncer)

## Performance Review Schedule

| Review Type | Frequency | Participants |
|-------------|-----------|--------------|
| Metrics Review | Daily | On-call engineer |
| Trend Analysis | Weekly | Engineering team |
| Capacity Planning | Monthly | Engineering + Ops |
| Load Testing | Quarterly | Full team |
| Architecture Review | Annually | Engineering leads |
