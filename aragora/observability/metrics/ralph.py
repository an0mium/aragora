"""Ralph campaign supervisor metrics.

Prometheus counters and gauges for campaign lifecycle, blocker classification,
repair tracking, budget burn, and PR merge gate status.

Usage:
    from aragora.observability.metrics.ralph import (
        record_campaign_step,
        record_blocker_classified,
        record_repair_attempt,
        record_budget_spent,
        record_pr_gate_disposition,
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_initialized = False

RALPH_CAMPAIGN_STEPS: Any = None
RALPH_CAMPAIGN_STATUS: Any = None
RALPH_BLOCKER_CLASSIFIED: Any = None
RALPH_REPAIR_ATTEMPTS: Any = None
RALPH_REPAIR_OUTCOMES: Any = None
RALPH_BUDGET_SPENT: Any = None
RALPH_PR_GATE_DISPOSITION: Any = None


class _NoOpMetric:
    """Fallback when Prometheus is unavailable."""

    def labels(self, **_: Any) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass


def _init_ralph_metrics() -> bool:
    """Initialize ralph metrics lazily."""
    global _initialized
    global RALPH_CAMPAIGN_STEPS, RALPH_CAMPAIGN_STATUS
    global RALPH_BLOCKER_CLASSIFIED, RALPH_REPAIR_ATTEMPTS, RALPH_REPAIR_OUTCOMES
    global RALPH_BUDGET_SPENT, RALPH_PR_GATE_DISPOSITION

    if _initialized:
        return True

    try:
        from prometheus_client import Counter, Gauge

        RALPH_CAMPAIGN_STEPS = Counter(
            "ralph_campaign_steps_total",
            "Total campaign supervisor steps",
            ["campaign_id", "action"],
        )
        RALPH_CAMPAIGN_STATUS = Gauge(
            "ralph_campaign_status",
            "Current campaign status (1=active)",
            ["campaign_id", "status"],
        )
        RALPH_BLOCKER_CLASSIFIED = Counter(
            "ralph_blocker_classified_total",
            "Blockers classified by kind",
            ["kind", "is_deterministic"],
        )
        RALPH_REPAIR_ATTEMPTS = Counter(
            "ralph_repair_attempts_total",
            "Repair attempts by blocker kind",
            ["blocker_kind"],
        )
        RALPH_REPAIR_OUTCOMES = Counter(
            "ralph_repair_outcomes_total",
            "Repair outcomes",
            ["blocker_kind", "outcome"],
        )
        RALPH_BUDGET_SPENT = Gauge(
            "ralph_budget_spent_usd",
            "Budget spent in USD",
            ["campaign_id"],
        )
        RALPH_PR_GATE_DISPOSITION = Counter(
            "ralph_pr_gate_disposition_total",
            "PR merge gate dispositions",
            ["disposition"],
        )
        _initialized = True
        return True
    except ImportError:
        logger.debug("prometheus_client not available, using no-op ralph metrics")
        noop = _NoOpMetric()
        RALPH_CAMPAIGN_STEPS = noop
        RALPH_CAMPAIGN_STATUS = noop
        RALPH_BLOCKER_CLASSIFIED = noop
        RALPH_REPAIR_ATTEMPTS = noop
        RALPH_REPAIR_OUTCOMES = noop
        RALPH_BUDGET_SPENT = noop
        RALPH_PR_GATE_DISPOSITION = noop
        _initialized = True
        return False


def record_campaign_step(campaign_id: str, action: str) -> None:
    """Record a campaign supervisor step."""
    _init_ralph_metrics()
    RALPH_CAMPAIGN_STEPS.labels(campaign_id=campaign_id, action=action).inc()


def record_campaign_status(campaign_id: str, status: str) -> None:
    """Update the campaign status gauge."""
    _init_ralph_metrics()
    RALPH_CAMPAIGN_STATUS.labels(campaign_id=campaign_id, status=status).set(1)


def record_blocker_classified(kind: str, is_deterministic: bool) -> None:
    """Record a blocker classification."""
    _init_ralph_metrics()
    RALPH_BLOCKER_CLASSIFIED.labels(kind=kind, is_deterministic=str(is_deterministic).lower()).inc()


def record_repair_attempt(blocker_kind: str) -> None:
    """Record a repair attempt."""
    _init_ralph_metrics()
    RALPH_REPAIR_ATTEMPTS.labels(blocker_kind=blocker_kind).inc()


def record_repair_outcome(blocker_kind: str, outcome: str) -> None:
    """Record a repair outcome (success/failure)."""
    _init_ralph_metrics()
    RALPH_REPAIR_OUTCOMES.labels(blocker_kind=blocker_kind, outcome=outcome).inc()


def record_budget_spent(campaign_id: str, amount_usd: float) -> None:
    """Update the budget spent gauge."""
    _init_ralph_metrics()
    RALPH_BUDGET_SPENT.labels(campaign_id=campaign_id).set(amount_usd)


def record_pr_gate_disposition(disposition: str) -> None:
    """Record a PR merge gate disposition."""
    _init_ralph_metrics()
    RALPH_PR_GATE_DISPOSITION.labels(disposition=disposition).inc()
