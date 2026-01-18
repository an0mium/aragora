# Aragora Metrics Runbook

Operational guide for monitoring and responding to Aragora platform metrics and alerts.

## Table of Contents

1. [Overview](#overview)
2. [Key Metrics](#key-metrics)
3. [Alert Response](#alert-response)
4. [Troubleshooting](#troubleshooting)
5. [Dashboards](#dashboards)

---

## Overview

Aragora exposes Prometheus metrics for three main subsystems:

| Subsystem | Prefix | Purpose |
|-----------|--------|---------|
| RLM (Recursive Language Models) | `aragora_rlm_*` | Context compression efficiency |
| Enterprise Connectors | `aragora_connector_*` | Data sync operations |
| Platform Core | `aragora_*` | Requests, agents, debates |

### Metrics Endpoint

```
GET /metrics
```

Served on port 9090 by default, configurable via `METRICS_PORT`.

---

## Key Metrics

### RLM Compression

| Metric | Type | Description | Target |
|--------|------|-------------|--------|
| `aragora_rlm_compression_ratio` | Histogram | Ratio of compressed/original tokens | p50 < 0.5 |
| `aragora_rlm_tokens_saved_total` | Counter | Total tokens saved | Increasing |
| `aragora_rlm_compression_duration_seconds` | Histogram | Time for compression ops | p99 < 5s |
| `aragora_rlm_compressions_total` | Counter | Compression operations | By status |
| `aragora_rlm_cache_hits_total` | Counter | Cache hits | >50% hit rate |
| `aragora_rlm_memory_bytes` | Gauge | Context cache memory | < 500MB |

**Key Query: Compression Efficiency**
```promql
# Median compression ratio (lower is better)
histogram_quantile(0.5, sum(rate(aragora_rlm_compression_ratio_bucket[15m])) by (le))
```

### Enterprise Connectors

| Metric | Type | Description | Target |
|--------|------|-------------|--------|
| `aragora_connector_syncs_total` | Counter | Sync operations | >99% success |
| `aragora_connector_sync_duration_seconds` | Histogram | Sync duration | p99 < 5min |
| `aragora_connector_sync_items_total` | Counter | Items synced | Increasing |
| `aragora_connector_health` | Gauge | Health status (0/1) | = 1 |
| `aragora_connector_sync_errors_total` | Counter | Errors by type | Minimal |
| `aragora_connector_rate_limits_total` | Counter | Rate limit hits | < 10/15min |

**Key Query: Connector Success Rate**
```promql
# Success rate per connector
sum(rate(aragora_connector_syncs_total{status="success"}[1h])) by (connector_type)
/ sum(rate(aragora_connector_syncs_total[1h])) by (connector_type)
```

### Platform Core

| Metric | Type | Description | Target |
|--------|------|-------------|--------|
| `aragora_requests_total` | Counter | HTTP requests | By status |
| `aragora_request_latency_seconds` | Histogram | Request latency | p99 < 2s |
| `aragora_agent_calls_total` | Counter | Agent API calls | >95% success |
| `aragora_active_debates` | Gauge | Concurrent debates | Stable |

---

## Alert Response

### RLMCompressionRatioHigh

**Severity:** Warning
**Condition:** Median compression ratio > 70%

**Investigation:**
1. Check source type distribution:
   ```promql
   sum(rate(aragora_rlm_compressions_total[15m])) by (source_type)
   ```
2. Review large content being compressed
3. Check if RLM agent_call is returning quality summaries

**Resolution:**
- Increase `max_levels` for deep hierarchies
- Review compression prompts for quality
- Consider pre-filtering large content

### RLMCompressionFailures

**Severity:** Critical
**Condition:** >5% failure rate

**Investigation:**
1. Check error logs:
   ```bash
   kubectl logs -l app=aragora | grep "RLM.*error" | tail -50
   ```
2. Check agent availability:
   ```promql
   sum(rate(aragora_agent_calls_total{status="failure"}[5m])) by (agent)
   ```

**Resolution:**
- Check API key validity
- Verify network connectivity to LLM providers
- Review rate limits on underlying APIs

### ConnectorSyncFailureRate

**Severity:** Warning
**Condition:** >10% failure rate for a connector type

**Investigation:**
1. Identify failing connector:
   ```promql
   topk(5, sum(rate(aragora_connector_sync_errors_total[15m])) by (connector_type, error_type))
   ```
2. Check error types (auth, network, rate_limit, parse)
3. Review connector logs

**Resolution:**
- **auth**: Refresh OAuth tokens, check credentials
- **network**: Verify external service availability
- **rate_limit**: Reduce sync frequency, implement backoff
- **parse**: Check for API changes, update connector

### ConnectorUnhealthy

**Severity:** Critical
**Condition:** Health gauge = 0 for 5 minutes

**Investigation:**
1. Check connector status in admin UI
2. Verify credentials are valid
3. Test connectivity to external service

**Resolution:**
1. Re-authenticate connector via admin UI
2. Check external service status page
3. Restart connector if credentials are valid

### HighRequestLatency

**Severity:** Warning
**Condition:** p99 latency > 2 seconds

**Investigation:**
1. Identify slow endpoints:
   ```promql
   topk(5, histogram_quantile(0.99, sum(rate(aragora_request_latency_seconds_bucket[5m])) by (le, endpoint)))
   ```
2. Check database query performance
3. Review RLM compression timing

**Resolution:**
- Scale horizontally if CPU-bound
- Optimize slow database queries
- Review caching strategies
- Check for network latency to external services

---

## Troubleshooting

### No Metrics Appearing

1. Verify metrics are enabled:
   ```bash
   echo $METRICS_ENABLED  # Should be "true"
   ```

2. Check prometheus-client is installed:
   ```bash
   pip show prometheus-client
   ```

3. Verify metrics endpoint responds:
   ```bash
   curl http://localhost:9090/metrics | head -20
   ```

### Metrics Lag

1. Check Prometheus scrape interval (default 30s)
2. Verify network connectivity to Prometheus
3. Check for clock skew between nodes

### High Cardinality

Watch for label explosion on these metrics:
- `connector_id` on sync metrics
- `endpoint` on request metrics

**Mitigation:**
- Use recording rules for high-traffic queries
- Limit unique connector IDs per tenant
- Aggregate endpoints by pattern

---

## Dashboards

### Importing the Dashboard

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `k8s/monitoring/aragora-dashboard.json`
4. Select Prometheus data source
5. Click Import

### Dashboard Panels

| Panel | Description | Key Insight |
|-------|-------------|-------------|
| RLM Compression Ratio (p50) | Median compression efficiency | Should be < 50% |
| Tokens Saved (24h) | Daily token savings | ROI indicator |
| Connector Sync Success Rate | Hourly sync success | Should be > 99% |
| Request Latency (p99) | API performance | Should be < 2s |
| Compression Operations Rate | RLM throughput | By source type |
| Sync Operations by Connector | Connector activity | Identify busy connectors |
| Connector Health Status | Live health | Red = action needed |
| Sync Errors by Type | Error breakdown | Identify patterns |

### Custom Queries

**Token Savings ROI (estimated cost savings):**
```promql
# Assuming $0.01 per 1K tokens
sum(increase(aragora_rlm_tokens_saved_total[30d])) / 1000 * 0.01
```

**Busiest Connectors:**
```promql
topk(5, sum(increase(aragora_connector_sync_items_total[24h])) by (connector_type))
```

**Debate Completion Rate:**
```promql
sum(rate(aragora_consensus_rate[1h]))
```

---

## SLO Targets

| SLO | Target | Measurement |
|-----|--------|-------------|
| Debate API Latency | 99% < 2s | `aragora_request_latency_seconds{endpoint=~"/api/debates.*"}` |
| Connector Availability | 99% success | `aragora_connector_syncs_total{status="success"}` |
| RLM Efficiency | p50 ratio < 50% | `aragora_rlm_compression_ratio` |
| Platform Availability | 99.9% uptime | `aragora_requests_total{status!~"5.."}` |

### Error Budget Calculation

```promql
# 30-day error budget remaining for connector availability (99% SLO)
1 - (
  (1 - sum(increase(aragora_connector_syncs_total{status="success"}[30d]))
       / sum(increase(aragora_connector_syncs_total[30d])))
  / 0.01
)
```

---

## Contact

- **Platform Team:** #platform-oncall
- **Integrations Team:** #integrations-oncall
- **Escalation:** See RUNBOOK_INCIDENT.md
