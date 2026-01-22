---
title: Performance Benchmark Results
description: Performance Benchmark Results
---

# Performance Benchmark Results

This document outlines the performance benchmarks and SLO targets for Aragora's API endpoints.

## SLO Targets

Based on `PERFORMANCE_TARGETS.md`, the following latency targets apply:

| Endpoint Category | P50 Target | P95 Target | P99 Target |
|-------------------|------------|------------|------------|
| Health Check      | < 5ms      | < 20ms     | < 50ms     |
| Authentication    | < 50ms     | < 150ms    | < 300ms    |
| Simple API Call   | < 100ms    | < 300ms    | < 500ms    |
| Search/Query      | < 200ms    | < 500ms    | < 1000ms   |

## Running Benchmarks

### API Endpoint Benchmarks

```bash
# Run with pytest
pytest benchmarks/api_endpoints.py -v

# Run as module
python -m benchmarks.api_endpoints
```

### Memory Tier Benchmarks

```bash
python -m benchmarks.memory_tiers
```

### Debate Throughput Benchmarks

```bash
python -m benchmarks.debate_throughput
```

### Full Evaluation Suite

```bash
python -m benchmarks.gauntlet_evaluation
```

## Benchmark Methodology

### Latency Measurements

- **Samples**: 10,000 iterations per endpoint (default)
- **Warmup**: 1,000 iterations discarded to allow JIT compilation
- **Metrics**: P50, P95, P99, min, max, mean, stddev
- **Environment**: Single-threaded, no network latency (mocked handlers)

### Throughput Measurements

- **Duration**: 60-second sustained load
- **Concurrency**: Variable (1, 10, 50, 100 concurrent requests)
- **Metrics**: Requests/second, errors/second, success rate

## Expected Results

The following are target results on a modern development machine (M1/M2 Mac, 16GB RAM):

### Health Check (`/api/health`)

| Metric | Target | Notes |
|--------|--------|-------|
| P50    | 0.01ms | In-memory status check |
| P95    | 0.05ms | No database access |
| P99    | 0.10ms | Constant time operation |
| OPS    | 500,000+ | CPU-bound only |

### Authentication (`/api/auth/validate`)

| Metric | Target | Notes |
|--------|--------|-------|
| P50    | 1-5ms  | JWT decode + cache check |
| P95    | 10ms   | Cache miss scenario |
| P99    | 25ms   | Database lookup |
| OPS    | 50,000+ | With token caching |

### Debate API (`/api/debates/:id`)

| Metric | Target | Notes |
|--------|--------|-------|
| P50    | 10-20ms | Database fetch |
| P95    | 50ms    | Cold cache |
| P99    | 100ms   | Complex debate with messages |
| OPS    | 5,000+  | With connection pooling |

### Search (`/api/debates/search`)

| Metric | Target | Notes |
|--------|--------|-------|
| P50    | 50-100ms | Indexed search |
| P95    | 200ms    | Complex query |
| P99    | 400ms    | Large result set |
| OPS    | 1,000+   | Depends on index size |

## Production Monitoring

### Metrics Collection

All endpoints emit Prometheus-compatible metrics:

```
aragora_http_request_duration_seconds{endpoint="/api/health", method="GET"}
aragora_http_requests_total{endpoint="/api/debates", method="GET", status="200"}
```

### Alerting Thresholds

| Condition | Severity | Action |
|-----------|----------|--------|
| P99 > 2x SLO | Warning | Investigate |
| P99 > 5x SLO | Critical | Page on-call |
| Error rate > 1% | Warning | Investigate |
| Error rate > 5% | Critical | Page on-call |

## Performance Tuning

### Database Connection Pool

```python
# Recommended settings
POOL_SIZE = 20
MAX_OVERFLOW = 10
POOL_TIMEOUT = 30
POOL_RECYCLE = 1800
```

### Redis Cache

```python
# Recommended TTLs
AUTH_TOKEN_CACHE_TTL = 300  # 5 minutes
DEBATE_CACHE_TTL = 60       # 1 minute
SEARCH_CACHE_TTL = 30       # 30 seconds
```

### Rate Limiting

Default rate limits (per endpoint):

| Category | Limit |
|----------|-------|
| Health | Unlimited |
| Auth | 60/min |
| API (authenticated) | 120/min |
| Search | 30/min |
| Debates | 60/min |

## Continuous Benchmarking

Benchmarks run automatically in CI on:
- Pull requests (baseline comparison)
- Main branch merges (regression detection)
- Nightly builds (comprehensive suite)

Results are stored in `.benchmarks/` and compared against baseline.

## Hot Path Profiling Results

*Latest run: 2026-01-21*

Core operations profiled with `scripts/profile_hot_paths.py`:

| Operation | Avg (ms) | Min (ms) | Max (ms) | StdDev (ms) | Iterations |
|-----------|----------|----------|----------|-------------|------------|
| Cosine similarity (1536d) | 0.227 | 0.073 | 62.484 | 2.463 | 1000 |
| Text similarity (word-based) | 0.002 | 0.002 | 0.070 | 0.003 | 1000 |
| SQLite SELECT (indexed) | 0.017 | 0.015 | 0.085 | 0.003 | 500 |
| SQLite GROUP BY | 1.006 | 0.263 | 41.123 | 4.380 | 100 |
| CritiqueStore.get_stats | 0.011 | 0.002 | 0.199 | 0.027 | 100 |
| CritiqueStore.retrieve_patterns | 0.014 | 0.003 | 0.687 | 0.071 | 100 |
| MetaCritiqueAnalyzer.analyze | 0.610 | 0.287 | 7.094 | 0.979 | 100 |

**Recommendations:**
- SQLite GROUP BY aggregation (1.01ms avg) - consider caching for repeated queries
- MetaCritiqueAnalyzer.analyze (0.61ms avg) - candidate for result caching

## Related Documentation

- [Performance Targets](../operations/performance-targets)
- [API Rate Limits](../api/rate-limits)
- [Monitoring](../deployment/observability-setup)
