"""
Evidence and culture metrics.

Provides Prometheus metrics for tracking evidence collection and
organizational culture pattern extraction from debates including:
- Evidence items stored in Knowledge Mound
- Evidence citation vote bonuses
- Culture patterns extracted from debates
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Evidence metrics
EVIDENCE_STORED: Any = None
EVIDENCE_CITATION_BONUSES: Any = None

# Culture metrics
CULTURE_PATTERNS: Any = None

_initialized = False


def init_evidence_metrics() -> None:
    """Initialize evidence and culture metrics."""
    global _initialized
    global EVIDENCE_STORED, EVIDENCE_CITATION_BONUSES, CULTURE_PATTERNS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter

        EVIDENCE_STORED = Counter(
            "aragora_evidence_stored_total",
            "Evidence items stored in knowledge mound",
        )

        EVIDENCE_CITATION_BONUSES = Counter(
            "aragora_evidence_citation_bonuses_total",
            "Evidence citation vote bonuses applied",
            ["agent"],
        )

        CULTURE_PATTERNS = Counter(
            "aragora_culture_patterns_total",
            "Culture patterns extracted from debates",
        )

        _initialized = True
        logger.debug("Evidence metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize evidence metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global EVIDENCE_STORED, EVIDENCE_CITATION_BONUSES, CULTURE_PATTERNS

    noop = NoOpMetric()
    EVIDENCE_STORED = noop
    EVIDENCE_CITATION_BONUSES = noop
    CULTURE_PATTERNS = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_evidence_metrics()


# =============================================================================
# Evidence Recording Functions
# =============================================================================


def record_evidence_stored(count: int = 1) -> None:
    """Record evidence items stored in Knowledge Mound.

    Evidence items are factual claims, citations, or supporting
    data extracted from debate arguments and stored for future
    reference and cross-debate learning.

    Args:
        count: Number of evidence items stored (default: 1)
    """
    _ensure_init()
    EVIDENCE_STORED.inc(count)


def record_evidence_citation_bonus(agent: str) -> None:
    """Record an evidence citation vote bonus.

    Applied when an agent provides well-cited evidence in their
    arguments, rewarding agents who support claims with references.

    Args:
        agent: Agent identifier that received the bonus
    """
    _ensure_init()
    EVIDENCE_CITATION_BONUSES.labels(agent=agent).inc()


# =============================================================================
# Culture Recording Functions
# =============================================================================


def record_culture_patterns(count: int = 1) -> None:
    """Record culture patterns extracted from debates.

    Culture patterns represent organizational decision-making styles,
    communication preferences, and recurring argument patterns
    that can inform future debates and team selection.

    Args:
        count: Number of patterns extracted (default: 1)
    """
    _ensure_init()
    CULTURE_PATTERNS.inc(count)


__all__ = [
    # Metrics
    "EVIDENCE_STORED",
    "EVIDENCE_CITATION_BONUSES",
    "CULTURE_PATTERNS",
    # Functions
    "init_evidence_metrics",
    "record_evidence_stored",
    "record_evidence_citation_bonus",
    "record_culture_patterns",
]
