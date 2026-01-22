---
title: Capacity Planning Guide
description: Capacity Planning Guide
---

# Capacity Planning Guide

Resource requirements and scaling guidelines for Aragora deployments.

## Table of Contents

- [Resource Requirements](#resource-requirements)
- [Scaling Guidelines](#scaling-guidelines)
- [Database Sizing](#database-sizing)
- [Monitoring Thresholds](#monitoring-thresholds)
- [Load Testing](#load-testing)

---

## Resource Requirements

### Per-Debate Resource Consumption

| Resource | Per Active Debate | Notes |
|----------|------------------|-------|
| Memory | 50-200 MB | Depends on agent count, context size |
| CPU | 0.1-0.5 cores | Spikes during LLM calls |
| Network | 1-5 MB | LLM API traffic, WebSocket streaming |
| Database I/O | 10-50 ops/min | Checkpoints, critiques, votes |

### Agent Resource Multipliers

| Agent Type | Memory Multiplier | Notes |
|------------|------------------|-------|
| API agents | 1.0x baseline | Stateless, minimal memory |
| CLI agents | 1.5x | Subprocess overhead |
| Local models | 5-10x | Model weights in memory |

### Server Baseline Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| HTTP Server | 256 MB, 0.5 CPU | 512 MB, 1 CPU | 1 GB, 2 CPU |
| WebSocket Server | 256 MB, 0.5 CPU | 512 MB, 1 CPU | 1 GB, 2 CPU |
| Worker Process | 512 MB, 1 CPU | 1 GB, 2 CPU | 2 GB, 4 CPU |

---

## Scaling Guidelines

### Horizontal Scaling

#### Multi-Worker Deployment

```bash
# Start with multiple workers (recommended: 2-4x CPU cores)
aragora serve --workers 4 --host 127.0.0.1
```

Workers are assigned consecutive ports:
- HTTP: 8080, 8081, 8082, 8083
- WebSocket: 8765, 8766, 8767, 8768

#### Nginx Load Balancing

```nginx
upstream aragora_api {
    least_conn;  # Best for API traffic
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
    server 127.0.0.1:8083;
}

upstream aragora_ws {
    ip_hash;  # Session affinity for WebSockets
    server 127.0.0.1:8765;
    server 127.0.0.1:8766;
    server 127.0.0.1:8767;
    server 127.0.0.1:8768;
}
```

### Capacity Planning Matrix

| Concurrent Debates | Workers | RAM | CPU Cores | Database |
|-------------------|---------|-----|-----------|----------|
| 1-10 | 1 | 2 GB | 2 | SQLite |
| 10-50 | 2-4 | 4-8 GB | 4-8 | SQLite |
| 50-100 | 4-8 | 8-16 GB | 8-16 | SQLite + Redis |
| 100-500 | 8-16 | 32-64 GB | 16-32 | PostgreSQL + Redis |
| 500+ | 16+ | 64+ GB | 32+ | PostgreSQL cluster |

### Vertical vs Horizontal Scaling

| Scenario | Recommendation |
|----------|---------------|
| Memory pressure | Add workers (horizontal) |
| CPU bottleneck | Add workers or upgrade CPU |
| Database contention | Migrate to PostgreSQL |
| WebSocket connections | Add workers with ip_hash |
| LLM API rate limits | Add API key rotation |

---

## Database Sizing

### SQLite Limits

| Metric | Soft Limit | Hard Limit | Mitigation |
|--------|------------|------------|------------|
| Database size | 10 GB | 140 TB | Archive old data |
| Concurrent writes | 1 | 1 | Use WAL mode |
| Concurrent reads | Unlimited | Unlimited | - |
| Connections | 50 | 2000 | Connection pooling |

### Growth Projections

| Data Type | Growth Rate | 1 Month | 6 Months | 1 Year |
|-----------|-------------|---------|----------|--------|
| Debates | 100/day | 3K | 18K | 36K |
| Critiques | 1K/day | 30K | 180K | 360K |
| Messages | 5K/day | 150K | 900K | 1.8M |
| Evidence | 500/day | 15K | 90K | 180K |
| Memory entries | 2K/day | 60K | 360K | 720K |

### Database Size Estimates

| Database | Per 1K Debates | Per 10K Debates | Per 100K Debates |
|----------|---------------|-----------------|------------------|
| Consensus | ~5 MB | ~50 MB | ~500 MB |
| Continuum | ~10 MB | ~100 MB | ~1 GB |
| ELO | ~2 MB | ~20 MB | ~200 MB |
| Critiques | ~20 MB | ~200 MB | ~2 GB |
| Evidence | ~50 MB | ~500 MB | ~5 GB |

### Retention Policies

| Data Type | Recommended Retention | Archive Strategy |
|-----------|----------------------|------------------|
| Active debates | Indefinite | - |
| Completed debates | 1 year | Archive to cold storage |
| Critiques/patterns | 6 months | Archive high-value only |
| Evidence | 90 days | TTL cleanup |
| Memory (fast tier) | 1 hour | Auto-expire |
| Memory (glacial) | 30 days | Promote or archive |
| Audit logs | 2 years | Compliance requirement |

---

## Monitoring Thresholds

### Resource Alerts

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU utilization | 70% | 90% | Add workers |
| Memory usage | 80% | 95% | Increase RAM or workers |
| Disk usage | 70% | 85% | Clean up or expand |
| Database size | 5 GB | 10 GB | Archive data |
| Open connections | 100 | 200 | Check for leaks |

### Performance Alerts

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| API latency p99 | 500ms | 2000ms | Profile/optimize |
| WebSocket lag | 100ms | 500ms | Check network/load |
| Debate duration | 5 min | 15 min | Check LLM latency |
| Query time | 100ms | 1000ms | Add indexes |
| Queue depth | 50 | 200 | Add workers |

### Circuit Breaker Thresholds

| Agent Type | Failure Threshold | Recovery Time |
|------------|------------------|---------------|
| API agents | 5 failures | 30 seconds |
| External APIs | 3 failures | 60 seconds |
| Database | 2 failures | 10 seconds |

---

## Load Testing

### Benchmark Commands

```bash
# Single debate benchmark
aragora benchmark --debates 1 --rounds 3 --agents 3

# Concurrent debate stress test
aragora benchmark --debates 10 --concurrent 5 --rounds 2

# Sustained load test
aragora benchmark --debates 100 --concurrent 10 --duration 3600
```

### Expected Performance

| Scenario | Debates/Hour | Latency p50 | Latency p99 |
|----------|-------------|-------------|-------------|
| Light (1 worker) | 20-50 | 30s | 120s |
| Medium (4 workers) | 100-200 | 25s | 90s |
| Heavy (8 workers) | 300-500 | 20s | 60s |

### Bottleneck Identification

| Symptom | Likely Cause | Diagnosis |
|---------|-------------|-----------|
| High CPU, low throughput | LLM rate limits | Check API quotas |
| High memory, OOM | Context accumulation | Enable RLM compression |
| Slow queries | Missing indexes | Run EXPLAIN ANALYZE |
| Connection timeouts | Thread exhaustion | Increase workers |
| Intermittent failures | Circuit breaker trips | Check external APIs |

---

## Infrastructure Recommendations

### Development

```yaml
# docker-compose.dev.yml
services:
  aragora:
    image: aragora:latest
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    environment:
      - ARAGORA_WORKERS=1
```

### Staging

```yaml
# docker-compose.staging.yml
services:
  aragora:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '4'
    environment:
      - ARAGORA_WORKERS=2
```

### Production

```yaml
# kubernetes/deployment.yaml
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: aragora
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: ARAGORA_WORKERS
          value: "4"
```

---

## Cost Optimization

### LLM API Costs

| Provider | Model | Cost/1K tokens | Debates/$ |
|----------|-------|---------------|-----------|
| Anthropic | Claude 3.5 | $0.003-$0.015 | ~20-100 |
| OpenAI | GPT-4 | $0.03-$0.06 | ~5-20 |
| OpenRouter | Various | $0.001-$0.01 | ~50-500 |

### Cost Reduction Strategies

1. **RLM Compression**: Reduce context by 40-60%
2. **Agent caching**: Cache similar debate contexts
3. **Tiered models**: Use cheaper models for critiques
4. **Batch processing**: Combine small debates
5. **Evidence deduplication**: Share evidence across debates

---

## Capacity Planning Checklist

### Before Launch

- [ ] Baseline resource measurement complete
- [ ] Database indexes optimized
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set
- [ ] Backup strategy tested
- [ ] Failover procedures documented

### Monthly Review

- [ ] Review resource utilization trends
- [ ] Check database growth rate
- [ ] Analyze query performance
- [ ] Review circuit breaker metrics
- [ ] Update capacity projections
- [ ] Plan infrastructure changes

### Quarterly Review

- [ ] Load test with projected traffic
- [ ] Review cost optimization opportunities
- [ ] Update scaling thresholds
- [ ] Test disaster recovery procedures
- [ ] Review and update this guide
