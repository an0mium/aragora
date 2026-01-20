"""
Gauntlet and workflow metrics for Aragora.

Provides metrics for:
- Gauntlet export operations
- Workflow template execution
- Consensus ingestion
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

# Module-level initialization state
_initialized = False

# Gauntlet metrics
GAUNTLET_EXPORTS_TOTAL: Any = None
GAUNTLET_EXPORT_LATENCY: Any = None
GAUNTLET_EXPORT_SIZE: Any = None

# Workflow metrics
WORKFLOW_TEMPLATES_CREATED: Any = None
WORKFLOW_TEMPLATE_EXECUTIONS: Any = None
WORKFLOW_TEMPLATE_EXECUTION_LATENCY: Any = None

# Consensus ingestion metrics
CONSENSUS_INGESTION_TOTAL: Any = None
CONSENSUS_INGESTION_LATENCY: Any = None
CONSENSUS_INGESTION_CLAIMS: Any = None

# Enhanced consensus metrics
CONSENSUS_DISSENT_INGESTED: Any = None
CONSENSUS_EVOLUTION_TRACKED: Any = None
CONSENSUS_EVIDENCE_LINKED: Any = None
CONSENSUS_AGREEMENT_RATIO: Any = None


def init_gauntlet_metrics() -> bool:
    """Initialize gauntlet and workflow Prometheus metrics."""
    global _initialized
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Histogram

        # Gauntlet metrics
        GAUNTLET_EXPORTS_TOTAL = Counter(
            "aragora_gauntlet_exports_total",
            "Total Gauntlet export operations",
            ["format", "type", "status"],
        )
        GAUNTLET_EXPORT_LATENCY = Histogram(
            "aragora_gauntlet_export_latency_seconds",
            "Gauntlet export operation latency",
            ["format", "type"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        GAUNTLET_EXPORT_SIZE = Histogram(
            "aragora_gauntlet_export_size_bytes",
            "Gauntlet export output size in bytes",
            ["format", "type"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
        )

        # Workflow metrics
        WORKFLOW_TEMPLATES_CREATED = Counter(
            "aragora_workflow_templates_created_total",
            "Total workflow templates created",
            ["pattern", "template_id"],
        )
        WORKFLOW_TEMPLATE_EXECUTIONS = Counter(
            "aragora_workflow_template_executions_total",
            "Total workflow template executions",
            ["pattern", "status"],
        )
        WORKFLOW_TEMPLATE_EXECUTION_LATENCY = Histogram(
            "aragora_workflow_template_execution_latency_seconds",
            "Workflow template execution latency",
            ["pattern"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
        )

        # Consensus ingestion metrics
        CONSENSUS_INGESTION_TOTAL = Counter(
            "aragora_consensus_ingestion_total",
            "Total consensus ingestion events",
            ["strength", "tier", "status"],
        )
        CONSENSUS_INGESTION_LATENCY = Histogram(
            "aragora_consensus_ingestion_latency_seconds",
            "Consensus ingestion latency",
            ["strength"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        CONSENSUS_INGESTION_CLAIMS = Counter(
            "aragora_consensus_ingestion_claims_total",
            "Total key claims ingested from consensus",
            ["tier"],
        )

        # Enhanced consensus metrics
        CONSENSUS_DISSENT_INGESTED = Counter(
            "aragora_consensus_dissent_ingested_total",
            "Total dissenting views ingested from consensus debates",
            ["dissent_type", "acknowledged"],
        )
        CONSENSUS_EVOLUTION_TRACKED = Counter(
            "aragora_consensus_evolution_tracked_total",
            "Total consensus evolution events (supersedes relationships)",
            ["evolution_type"],
        )
        CONSENSUS_EVIDENCE_LINKED = Counter(
            "aragora_consensus_evidence_linked_total",
            "Total evidence items linked to consensus nodes",
            ["tier"],
        )
        CONSENSUS_AGREEMENT_RATIO = Histogram(
            "aragora_consensus_agreement_ratio",
            "Distribution of agreement ratios in ingested consensus",
            ["strength"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        _initialized = True
        logger.debug("Gauntlet metrics initialized")
        return True

    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    noop = NoOpMetric()
    GAUNTLET_EXPORTS_TOTAL = noop
    GAUNTLET_EXPORT_LATENCY = noop
    GAUNTLET_EXPORT_SIZE = noop
    WORKFLOW_TEMPLATES_CREATED = noop
    WORKFLOW_TEMPLATE_EXECUTIONS = noop
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = noop
    CONSENSUS_INGESTION_TOTAL = noop
    CONSENSUS_INGESTION_LATENCY = noop
    CONSENSUS_INGESTION_CLAIMS = noop
    CONSENSUS_DISSENT_INGESTED = noop
    CONSENSUS_EVOLUTION_TRACKED = noop
    CONSENSUS_EVIDENCE_LINKED = noop
    CONSENSUS_AGREEMENT_RATIO = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_gauntlet_metrics()


# =============================================================================
# Gauntlet Export Functions
# =============================================================================


def record_gauntlet_export(
    format: str,
    export_type: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a Gauntlet export operation."""
    _ensure_init()
    status = "success" if success else "error"
    GAUNTLET_EXPORTS_TOTAL.labels(format=format, type=export_type, status=status).inc()
    GAUNTLET_EXPORT_LATENCY.labels(format=format, type=export_type).observe(latency_seconds)
    if size_bytes > 0:
        GAUNTLET_EXPORT_SIZE.labels(format=format, type=export_type).observe(size_bytes)


@contextmanager
def track_gauntlet_export(format: str, export_type: str) -> Generator[dict, None, None]:
    """Context manager to track Gauntlet export operations."""
    _ensure_init()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"size_bytes": 0}
    success = True
    try:
        yield ctx
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_gauntlet_export(format, export_type, success, latency, ctx.get("size_bytes", 0))


# =============================================================================
# Workflow Template Functions
# =============================================================================


def record_workflow_template_created(pattern: str, template_id: str) -> None:
    """Record a workflow template creation."""
    _ensure_init()
    WORKFLOW_TEMPLATES_CREATED.labels(pattern=pattern, template_id=template_id).inc()


def record_workflow_template_execution(pattern: str, success: bool, latency_seconds: float) -> None:
    """Record a workflow template execution."""
    _ensure_init()
    status = "success" if success else "error"
    WORKFLOW_TEMPLATE_EXECUTIONS.labels(pattern=pattern, status=status).inc()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY.labels(pattern=pattern).observe(latency_seconds)


@contextmanager
def track_workflow_template_execution(pattern: str) -> Generator[None, None, None]:
    """Context manager to track workflow template execution."""
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_workflow_template_execution(pattern, success, latency)


# =============================================================================
# Consensus Ingestion Functions
# =============================================================================


def record_consensus_ingestion(
    strength: str,
    tier: str,
    success: bool,
    latency_seconds: float,
    claims_count: int = 0,
) -> None:
    """Record a consensus ingestion event."""
    _ensure_init()
    status = "success" if success else "error"
    CONSENSUS_INGESTION_TOTAL.labels(strength=strength, tier=tier, status=status).inc()
    CONSENSUS_INGESTION_LATENCY.labels(strength=strength).observe(latency_seconds)
    if claims_count > 0:
        CONSENSUS_INGESTION_CLAIMS.labels(tier=tier).inc(claims_count)


def record_consensus_dissent(dissent_type: str, acknowledged: bool = False, count: int = 1) -> None:
    """Record ingestion of dissenting views from consensus."""
    _ensure_init()
    CONSENSUS_DISSENT_INGESTED.labels(
        dissent_type=dissent_type,
        acknowledged="true" if acknowledged else "false",
    ).inc(count)


def record_consensus_evolution(evolution_type: str) -> None:
    """Record consensus evolution tracking event."""
    _ensure_init()
    CONSENSUS_EVOLUTION_TRACKED.labels(evolution_type=evolution_type).inc()


def record_consensus_evidence_linked(tier: str, count: int = 1) -> None:
    """Record evidence items linked to consensus."""
    _ensure_init()
    CONSENSUS_EVIDENCE_LINKED.labels(tier=tier).inc(count)


def record_consensus_agreement_ratio(strength: str, agreement_ratio: float) -> None:
    """Record the agreement ratio for a consensus."""
    _ensure_init()
    CONSENSUS_AGREEMENT_RATIO.labels(strength=strength).observe(agreement_ratio)


def record_km_inbound_event(event_type: str, source: str, success: bool = True) -> None:
    """Record an inbound event to Knowledge Mound."""
    _ensure_init()
    from aragora.observability.metrics.km import record_km_operation, record_km_event_emitted

    record_km_operation("ingest", success, 0.0)
    record_km_event_emitted(f"inbound_{event_type}")
