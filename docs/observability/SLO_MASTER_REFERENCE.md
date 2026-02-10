# SLO Master Reference

This document is the single source of truth mapping customer-facing SLA commitments to internal SLO targets, monitoring infrastructure, and operational code.

## Quick Reference: SLA Tier to Internal SLO Mapping

| Customer SLA Target | Internal SLO | Code Module | Prometheus Metric |
|---------------------|-------------|-------------|-------------------|
| 99.95% uptime (Enterprise) | API Availability 99.9% | `observability/slo.py` | `slo:api:availability:ratio` |
| 99.9% uptime (Professional) | API Availability 99.9% | `observability/slo.py` | `slo:api:availability:ratio` |
| 99.5% uptime (Standard) | API Availability 99.5% | `observability/slo.py` | `slo:api:availability:ratio` |
| P99 API < 500ms | p99 Latency SLO | `observability/slo.py` | `slo:api:latency:p95` |
| Debate completion P95 | Debate Success 95% | `observability/slo.py` | `slo:debate:completion:p95` |
| WebSocket 99.0% | WebSocket availability | `observability/slo.py` | Connection success rate |
| Webhook P99 < 5s | Webhook delivery | `observability/alerting.py` | `slo:webhook:delivery:latency:p99` |

## 1. Primary SLOs

### 1.1 API Availability

| Attribute | Value |
|-----------|-------|
| **Target** | 99.9% (configurable via `SLO_AVAILABILITY_TARGET`) |
| **Calculation** | `(1 - error_requests / total_requests) * 100` |
| **Error budget (30d)** | 43 minutes |
| **Code** | `aragora/observability/slo.py` — `check_availability_slo()` |
| **Prometheus** | `slo:api:availability:ratio` |
| **Alert P1** | Drops below 99.0% |
| **Alert P2** | Drops below 99.5% |
| **Maps to SLA** | Enterprise 99.95%, Professional 99.9%, Standard 99.5% |

### 1.2 API Latency (p99)

| Attribute | Value |
|-----------|-------|
| **Target** | < 500ms (configurable via `SLO_LATENCY_P99_TARGET_MS`) |
| **Code** | `aragora/observability/slo.py` — `check_latency_slo()` |
| **Prometheus** | `slo:api:latency:p95`, `slo:api:latency:p99` |
| **Alert P1** | P95 exceeds 2x target for 5+ minutes |
| **Alert P2** | P95 exceeds target for 10+ minutes |
| **Maps to SLA** | API Latency targets (SLA Section 4.1) |

**Endpoint-specific targets** (from `config/performance_slos.py`):

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| Debate creation | 50ms | 200ms | 500ms |
| Debate retrieval | 10ms | 50ms | 100ms |
| Document processing | 100ms | 300ms | 1000ms |
| WebSocket messages | 5ms | 20ms | 50ms |
| Webhook delivery | 500ms | 2s | 5s |

### 1.3 Debate Success Rate

| Attribute | Value |
|-----------|-------|
| **Target** | 95% (configurable via `SLO_DEBATE_SUCCESS_TARGET`) |
| **Code** | `aragora/observability/slo.py` — `check_debate_success_slo()` |
| **Alert** | 8 built-in rules in `observability/alerting.py` |
| **Maps to SLA** | Debate completion targets (SLA Section 4.2) |

**Debate completion targets:**

| Type | P50 | P95 |
|------|-----|-----|
| Standard (3 agents, 3 rounds) | 8s | 20s |
| Extended (5+ agents) | 15s | 45s |
| Complex (10+ rounds) | 30s | 90s |

## 2. Secondary SLOs

### 2.1 Error Rate

| Service | Target | Max | Code |
|---------|--------|-----|------|
| API | < 0.5% | 1.0% | `slo:api:error_rate:ratio` |
| WebSocket | < 1.0% | 2.0% | Connection metrics |
| Background jobs | < 0.1% | 0.5% | Job failure metrics |

### 2.2 Throughput

| Metric | Target | Minimum |
|--------|--------|---------|
| Debates/min | 100 | 50 |
| Documents ingested/hr | 1000 | 500 |
| Concurrent WS connections | 10,000 | 5,000 |

### 2.3 Data Durability

| Data | RPO | RTO |
|------|-----|-----|
| Debate results | 0 (sync write) | 15 min |
| Knowledge Mound | 1 hour | 4 hours |
| User data | 0 | 1 hour |

## 3. Operational SLO Modules

### 3.1 Performance SLOs (`config/performance_slos.py`)

30+ operation-specific SLO classes with P50/P90/P99 targets:

| Domain | Operations Covered |
|--------|--------------------|
| Knowledge Mound | Query, ingestion, checkpoint |
| Consensus | Detection, validation |
| Adapters | Sync, forward, reverse, search, validation |
| Memory | Store, recall |
| Debate | Full lifecycle |
| RLM | Compression, streaming, query |
| Workflow | Execution, checkpoint, recovery |
| Control Plane | Leader election, config sync, health |
| WebSocket | Connection, message, broadcast |
| Bot Platforms | Response, webhook |
| API Endpoints | Per-endpoint latency |

### 3.2 SLO Runtime (`observability/slo.py`)

- `SLOTarget`, `SLOResult`, `SLOStatus` dataclasses
- Error budget tracking with burn rate calculation
- `SLOAlertMonitor` for background monitoring
- Env var overrides for all targets
- Prometheus metrics integration

### 3.3 Alert Bridge (`observability/slo_alert_bridge.py`)

Routes SLO violations to external systems:

| Channel | Capability |
|---------|-----------|
| PagerDuty | Incident creation with severity mapping |
| Slack | Block kit notifications |
| Teams | Webhook integration |

Features: incident deduplication, auto-recovery, cooldown, business-hour escalation.

### 3.4 Alerting Rules (`observability/alerting.py`)

8 critical built-in alert rules:

| # | Rule | Trigger |
|---|------|---------|
| 1 | Agent cascade failure | 2+ providers fail in 5 min |
| 2 | Debate stalling | >5 min without phase transition |
| 3 | Queue saturation | >90% capacity |
| 4 | Consensus failure | >10% failure rate/hour |
| 5 | Memory pressure | >5% eviction rate |
| 6 | Rate limit exceeded | Sustained limit hits |
| 7 | Circuit breaker open | Service circuit opened |
| 8 | API latency spike | p99 > 2x baseline |

## 4. Error Budget Policy

| Budget consumed | Action |
|-----------------|--------|
| > 50% in 7 days | Freeze non-critical deployments, focus on reliability |
| > 80% | Halt all feature deployments, mandatory post-mortems |
| Exhausted | Emergency freeze, all hands on reliability, executive notification |

**Alert priority thresholds:**

| Priority | Trigger |
|----------|---------|
| P1 (page) | Availability < 99.0%, P95 > 2x target for 5 min, error rate > 5% |
| P2 (15 min) | Availability < 99.5%, P95 > target for 10 min, error rate > 1% |
| P3 (1 hour) | Budget > 50% in 7 days, latency trending, capacity > 70% |

## 5. SLO Review Cadence

| Review | Frequency | Attendees |
|--------|-----------|-----------|
| Weekly SLO review | Weekly | SRE + On-call |
| Monthly reliability review | Monthly | Engineering leads |
| Quarterly SLO adjustment | Quarterly | Product + Engineering |

## 6. Related Documents

| Document | Purpose |
|----------|---------|
| `docs/SLA.md` | Customer-facing SLA with tiers and credits |
| `docs/enterprise/SLA.md` | Enterprise SLA with DR objectives |
| `docs/BREACH_NOTIFICATION_SLA.md` | Security breach notification timelines |
| `docs/SLO_DEFINITIONS.md` | Internal SLO targets and Prometheus rules |
| `docs/TEST_COVERAGE_SLOS.md` | Test coverage targets by tier |
| `docs/runbooks/RUNBOOK_SLOW_DEBATES.md` | Slow debate diagnostic runbook |

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2026-02-02 | Initial master reference | Platform Team |
