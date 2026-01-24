# Aragora Service Level Objectives (SLOs)

This document defines the Service Level Objectives for the Aragora platform.

## Overview

SLOs define the target reliability levels for Aragora services. These objectives guide engineering decisions, incident response, and capacity planning.

## Primary SLOs

### Availability

| Service | SLO | Measurement | Error Budget (30d) |
|---------|-----|-------------|-------------------|
| API Gateway | 99.5% | HTTP 5xx rate | 3.6 hours |
| WebSocket Service | 99.0% | Connection success rate | 7.2 hours |
| Database | 99.9% | Query success rate | 43 minutes |

**Calculation:** `(1 - error_requests / total_requests) * 100`

### Latency

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| `POST /api/v1/debates` | 50ms | 200ms | 500ms |
| `GET /api/v1/debates/:id` | 10ms | 50ms | 100ms |
| `POST /api/v1/documents` | 100ms | 300ms | 1000ms |
| WebSocket message delivery | 5ms | 20ms | 50ms |

**Note:** Debate completion latency is excluded as it depends on agent response times (typically 2-30 seconds).

### Error Rate

| Service | Target | Threshold |
|---------|--------|-----------|
| API Error Rate | < 0.5% | < 1.0% |
| WebSocket Error Rate | < 1.0% | < 2.0% |
| Background Job Failure | < 0.1% | < 0.5% |

## Secondary SLOs

### Throughput

| Metric | Target | Minimum |
|--------|--------|---------|
| Debates per minute | 100 | 50 |
| Documents ingested per hour | 1000 | 500 |
| Concurrent WebSocket connections | 10,000 | 5,000 |

### Data Durability

| Data Type | RPO | RTO |
|-----------|-----|-----|
| Debate results | 0 (synchronous write) | 15 min |
| Knowledge Mound | 1 hour | 4 hours |
| User data | 0 | 1 hour |

## Alerting Thresholds

### P1 (Page immediately)
- Availability drops below 99.0%
- P95 latency exceeds 2x target for 5+ minutes
- Error rate exceeds 5%

### P2 (Alert within 15 minutes)
- Availability drops below 99.5%
- P95 latency exceeds target for 10+ minutes
- Error rate exceeds 1%

### P3 (Alert within 1 hour)
- Error budget consumption > 50% in 7 days
- P95 latency trending toward threshold
- Capacity utilization > 70%

## Prometheus Metrics

```yaml
# Availability SLI
- record: slo:api:availability:ratio
  expr: |
    sum(rate(http_requests_total{status!~"5.."}[5m]))
    /
    sum(rate(http_requests_total[5m]))

# Latency SLI
- record: slo:api:latency:p95
  expr: |
    histogram_quantile(0.95,
      sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
    )

# Error Rate SLI
- record: slo:api:error_rate:ratio
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m]))
    /
    sum(rate(http_requests_total[5m]))
```

## Error Budget Policy

### When error budget is consumed > 50%:
1. Freeze non-critical deployments
2. Focus engineering on reliability improvements
3. Review recent changes for regression

### When error budget is consumed > 80%:
1. Halt all feature deployments
2. Mandatory post-mortems for all incidents
3. Escalate to engineering leadership

### When error budget is exhausted:
1. Emergency freeze on all changes
2. All hands on reliability
3. Executive notification

## SLO Review Cadence

| Review Type | Frequency | Attendees |
|-------------|-----------|-----------|
| Weekly SLO review | Weekly | SRE + On-call |
| Monthly reliability review | Monthly | Engineering leads |
| Quarterly SLO adjustment | Quarterly | Product + Engineering |

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-24 | Initial SLO definitions | Platform Team |
