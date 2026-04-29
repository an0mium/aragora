# P4 (Permissioned Memory) First Deliverable: KM Connection Health → Metrics Surface

**Status:** spec only — text proposal for operator review.
**Author:** droid (Factory) overnight 2026-04-28.
**Scope:** wire the existing `ConnectionHealthMonitor` into the existing `KMMetrics` observability surface.
**Non-scope:** no implementation overnight, no PR. Stop at spec.

## Why this exists

The 2026-04-21..28 reassessment surfaced that Pillar 4 (Permissioned Memory and Large Context) received **1.1%** of merged PRs in the 7-day window — the second-lowest of any pillar. Three real PRs landed in scope:

- #6429 fix(mound): narrow Connection/Redis handles + truthy-function checks
- #6431 [DIC-16] Receipt + KM provenance for claim/crux IDs
- #6635 [DIC-16] CruxReceiptAdapter — KM ingestion for crux-finder receipts

A grep of the codebase reveals a structural gap: `ConnectionHealthMonitor`
in `aragora/knowledge/mound/resilience/health.py` is **exported but
orphaned** — it has no caller. The monitor was added with the resilience
hardening tranche; it has unit tests (existing sync paths in
`tests/knowledge/mound/test_resilience.py`, plus the async paths just
added in PR #6784); but no production module instantiates it and feeds
its `HealthStatus` into the observability surface.

Meanwhile `aragora/knowledge/mound/metrics.py` (559 lines) provides
`KMMetrics` — operation latency tracking, cache hit/miss counters, an
`OperationType` taxonomy. It was designed to surface KM health to
operators. But it does **not** consume `ConnectionHealthMonitor`'s
output. The result: KM connection health is invisible to the metrics
surface even though both halves of the equation exist.

This is a single-deliverable bridging gap. Worker C's rescue-class
audit identified the broader pattern (orphan modules accumulating);
this is one concrete instance, picked because:

1. The bridge is **<100 LOC** of additive code.
2. It exercises the code paths we just added tests for in PR #6784.
3. It produces immediately-observable operator value (KM health
   visible in Prometheus / Grafana / dashboard).
4. It has zero conflict surface with active automation lanes.

## Surface in scope

Today:
- `aragora/knowledge/mound/resilience/health.py` (151 LOC)
  - `HealthStatus` dataclass with `to_dict()`
  - `ConnectionHealthMonitor` with `start`, `stop`, `check_health`,
    `record_success`, `record_failure`, `is_healthy`, `get_status`
  - **No production caller.** Confirmed by grep.

- `aragora/knowledge/mound/metrics.py` (559 LOC)
  - `KMMetrics` class with `measure_operation`, `get_health`,
    `get_stats`
  - `OperationType` enum
  - Has its own internal `HealthStatus` import from
    `aragora.resilience.health` (NOT the mound's resilience module —
    different code path)
  - **No reference to ConnectionHealthMonitor.**

- Test coverage for the orphan: PR #6784 just landed 17 async-path
  tests for `ConnectionHealthMonitor`. The module is now well-tested
  but still orphan.

## The first deliverable: KMMetricsHealthBridge

Single new module: `aragora/knowledge/mound/metrics_health_bridge.py`
(~80 LOC). One small function added to `KMMetrics` (~20 LOC). One test
file (~80 LOC). Total: ~180 LOC.

### Module: `metrics_health_bridge.py`

```python
"""
Bridge from ConnectionHealthMonitor → KMMetrics observability.

Periodically polls the resilience-layer ConnectionHealthMonitor
and surfaces its HealthStatus into the KMMetrics observability
surface so operators see KM connection health in the same place
they see operation latency and cache hit rate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from aragora.knowledge.mound.resilience import ConnectionHealthMonitor
from aragora.knowledge.mound.metrics import KMMetrics

@dataclass
class HealthBridgeConfig:
    poll_interval_seconds: float = 5.0
    surface_latency: bool = True

class KMMetricsHealthBridge:
    """Polls ConnectionHealthMonitor and feeds KMMetrics."""

    def __init__(
        self,
        monitor: ConnectionHealthMonitor,
        metrics: KMMetrics,
        config: HealthBridgeConfig | None = None,
    ) -> None: ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
```

Behavior:
- On `start()`, spawns a task that every `poll_interval_seconds` calls
  `monitor.get_status()` and posts the result into `metrics` via a new
  `KMMetrics.record_connection_health(status: HealthStatus)` method.
- On `stop()`, cancels the polling task cleanly (mirrors
  `ConnectionHealthMonitor`'s own start/stop pattern).
- `snapshot()` returns a JSON-friendly dict combining the latest
  health and a derived sub-summary from `metrics.get_stats()`.

### Hook into `KMMetrics`

Add one method:

```python
def record_connection_health(self, status: HealthStatus) -> None:
    """Record connection health state into the metrics surface."""
    with self._lock:
        self._connection_healthy = status.healthy
        self._connection_consecutive_failures = status.consecutive_failures
        if status.latency_ms is not None:
            self._connection_latency_ms = status.latency_ms
```

These three new fields surface in `get_health()` and `get_stats()` so
existing dashboards pick them up automatically.

### Tests

`tests/knowledge/mound/test_metrics_health_bridge.py`:

- bridge writes monitor's `HealthStatus` into metrics on each tick
- `start()` is idempotent
- `stop()` cancels cleanly
- `snapshot()` returns a coherent dict combining both halves
- failure-threshold transition propagates from monitor to metrics

All async, mocked pool, parallel to the test patterns established in
PR #6784.

## What this gives operators

After landing:
1. `metrics.get_health()` exposes `connection_healthy`,
   `connection_consecutive_failures`, `connection_latency_ms`.
2. Existing Prometheus exporters (in
   `aragora/observability/metrics.py`) pick these up automatically
   if the dashboard wires `KMMetrics` as a source.
3. A `aragora knowledge-mound health-status` CLI command (out of
   scope for this spec, but a small follow-on PR) can dump the
   snapshot for SRE-style debugging.

## Why this is the right scope

- ~180 LOC additive, no production code modified except a single
  new method on `KMMetrics`.
- Closes the orphan-module gap that Worker C's rescue-class audit
  flagged as a recurrent shape.
- Exercises the code paths PR #6784 just gave test coverage to —
  every call path the bridge uses has tests.
- Roadmap-aligned: Pillar 4 (Memory) + Pillar 5 (Receipts /
  Auditability — operator visibility into substrate health is part
  of the trust layer).
- No conflict with: Codex's automation lane (different module),
  Claude's docs/red-CI lane (different files), the four red main
  workflows (no workflow touches), or PR #6784 (additive on top of
  the same module, no test conflict).

## Sequencing suggestion

If the operator approves implementation:
1. PR 1 (this spec): bridge module + KMMetrics method + tests. ~180
   LOC total. No auto-merge. Operator review.
2. PR 2 (later, optional): `aragora knowledge-mound health-status`
   CLI command. ~80 LOC.
3. PR 3 (later, optional): wire into Prometheus exporter if not
   automatic. ~50 LOC.

Each PR remains narrow.

## What this spec does NOT solve

- **Distributed KM health** (cross-region, cross-tenant): out of
  scope; the resilience layer's monitor is per-pool.
- **Adapter-level health** (per-adapter health rolled up): the
  adapter mixin already does this in
  `aragora/knowledge/mound/resilience/adapter_mixin.py`; bridging
  that into `KMMetrics` is a separate, larger spec.
- **SLO breach alerting**: the SLO module
  (`aragora/knowledge/mound/resilience/slo.py`) handles its own
  alerting; this bridge just surfaces health for operator
  inspection.
- **Wiring into the Bridge Run Inspector** (P3 first-deliverable):
  separate spec; the inspector consumes the agent bridge run feed,
  not KM health.

## Stop conditions

Spec only. No implementation initiative requested. If rejected, the
orphan module continues to sit unused (no harm); operators who need
KM health checks today already have the SLO surface and the
adapter-mixin's combined status. This is incremental observability
improvement, not critical-path work.

---

*End of spec. No PR is filed by this document.*
