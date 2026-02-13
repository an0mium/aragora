"""
Consensus ingestion metrics.

Provides Prometheus metrics for tracking consensus ingestion events,
including dissent, evolution tracking, and evidence linking.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Basic consensus ingestion metrics
CONSENSUS_INGESTION_TOTAL: Any = None
CONSENSUS_INGESTION_LATENCY: Any = None
CONSENSUS_INGESTION_CLAIMS: Any = None

# Enhanced consensus metrics
CONSENSUS_DISSENT_INGESTED: Any = None
CONSENSUS_EVOLUTION_TRACKED: Any = None
CONSENSUS_EVIDENCE_LINKED: Any = None
CONSENSUS_AGREEMENT_RATIO: Any = None

_initialized = False
_enhanced_initialized = False


def init_consensus_metrics() -> None:
    """Initialize basic consensus ingestion metrics."""
    global _initialized
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

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

        _initialized = True
        logger.debug("Consensus ingestion metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def init_enhanced_consensus_metrics() -> None:
    """Initialize enhanced consensus metrics for dissent, evolution, and linking."""
    global _enhanced_initialized
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    if _enhanced_initialized:
        return

    if not get_metrics_enabled():
        _init_enhanced_noop_metrics()
        _enhanced_initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

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

        _enhanced_initialized = True
        logger.debug("Enhanced consensus metrics initialized")

    except (ImportError, ValueError):
        _init_enhanced_noop_metrics()
        _enhanced_initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op basic consensus metrics."""
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS

    CONSENSUS_INGESTION_TOTAL = NoOpMetric()
    CONSENSUS_INGESTION_LATENCY = NoOpMetric()
    CONSENSUS_INGESTION_CLAIMS = NoOpMetric()


def _init_enhanced_noop_metrics() -> None:
    """Initialize no-op enhanced consensus metrics."""
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    CONSENSUS_DISSENT_INGESTED = NoOpMetric()
    CONSENSUS_EVOLUTION_TRACKED = NoOpMetric()
    CONSENSUS_EVIDENCE_LINKED = NoOpMetric()
    CONSENSUS_AGREEMENT_RATIO = NoOpMetric()


def _ensure_init() -> None:
    """Ensure basic metrics are initialized."""
    if not _initialized:
        init_consensus_metrics()


def _ensure_enhanced_init() -> None:
    """Ensure enhanced metrics are initialized."""
    if not _enhanced_initialized:
        init_enhanced_consensus_metrics()


# =============================================================================
# Basic Recording Functions
# =============================================================================


def record_consensus_ingestion(
    strength: str,
    tier: str,
    success: bool,
    latency_seconds: float,
    claims_count: int = 0,
) -> None:
    """Record a consensus ingestion event.

    Args:
        strength: Consensus strength (unanimous, strong, moderate, weak, split, contested)
        tier: KM tier used (glacial, slow, medium, fast)
        success: Whether the ingestion succeeded
        latency_seconds: Ingestion latency in seconds
        claims_count: Number of key claims ingested
    """
    _ensure_init()
    status = "success" if success else "error"
    CONSENSUS_INGESTION_TOTAL.labels(strength=strength, tier=tier, status=status).inc()
    CONSENSUS_INGESTION_LATENCY.labels(strength=strength).observe(latency_seconds)
    if claims_count > 0:
        CONSENSUS_INGESTION_CLAIMS.labels(tier=tier).inc(claims_count)


# =============================================================================
# Enhanced Recording Functions (Dissent, Evolution, Linking)
# =============================================================================


def record_consensus_dissent(
    dissent_type: str,
    acknowledged: bool = False,
    count: int = 1,
) -> None:
    """Record ingestion of dissenting views from consensus.

    Args:
        dissent_type: Type of dissent (risk_warning, fundamental_disagreement, etc.)
        acknowledged: Whether the dissent was acknowledged by majority
        count: Number of dissents ingested (default: 1)
    """
    _ensure_enhanced_init()
    CONSENSUS_DISSENT_INGESTED.labels(
        dissent_type=dissent_type,
        acknowledged="true" if acknowledged else "false",
    ).inc(count)


def record_consensus_evolution(evolution_type: str) -> None:
    """Record consensus evolution tracking event.

    Args:
        evolution_type: Type of evolution:
            - new_supersedes: New consensus supersedes an existing one
            - found_similar: Found similar prior consensus (potential supersedes)
            - no_evolution: No evolution relationship detected
    """
    _ensure_enhanced_init()
    CONSENSUS_EVOLUTION_TRACKED.labels(evolution_type=evolution_type).inc()


def record_consensus_evidence_linked(tier: str, count: int = 1) -> None:
    """Record evidence items linked to consensus.

    Args:
        tier: KM tier for the evidence
        count: Number of evidence items linked
    """
    _ensure_enhanced_init()
    CONSENSUS_EVIDENCE_LINKED.labels(tier=tier).inc(count)


def record_consensus_agreement_ratio(strength: str, agreement_ratio: float) -> None:
    """Record the agreement ratio for a consensus.

    Args:
        strength: Consensus strength
        agreement_ratio: Ratio of agreeing agents to total participants (0.0-1.0)
    """
    _ensure_enhanced_init()
    CONSENSUS_AGREEMENT_RATIO.labels(strength=strength).observe(agreement_ratio)


__all__ = [
    # Basic Metrics
    "CONSENSUS_INGESTION_TOTAL",
    "CONSENSUS_INGESTION_LATENCY",
    "CONSENSUS_INGESTION_CLAIMS",
    # Enhanced Metrics
    "CONSENSUS_DISSENT_INGESTED",
    "CONSENSUS_EVOLUTION_TRACKED",
    "CONSENSUS_EVIDENCE_LINKED",
    "CONSENSUS_AGREEMENT_RATIO",
    # Init Functions
    "init_consensus_metrics",
    "init_enhanced_consensus_metrics",
    # Basic Recording Functions
    "record_consensus_ingestion",
    # Enhanced Recording Functions
    "record_consensus_dissent",
    "record_consensus_evolution",
    "record_consensus_evidence_linked",
    "record_consensus_agreement_ratio",
]
