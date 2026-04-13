"""
Governance observability metric helpers.

Provides optional metric recording for governance operations.
All functions gracefully degrade if the observability module is not available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    _RecordFunc = Callable[[str, str], None]


def _record_metric(metric_name: str, first_value: str, second_value: str) -> None:
    """Record a governance metric if the observability helper is available."""
    try:
        from aragora.observability import metrics as _metrics
    except ImportError:
        return

    record_func = getattr(_metrics, metric_name, None)
    if callable(record_func):
        record_func(first_value, second_value)


def record_governance_verification(verification_type: str, result: str) -> None:
    """Record governance verification metric if available."""
    _record_metric("record_governance_verification", verification_type, result)


def record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record governance decision metric if available."""
    _record_metric("record_governance_decision", decision_type, outcome)


def record_governance_approval(approval_type: str, status: str) -> None:
    """Record governance approval metric if available."""
    _record_metric("record_governance_approval", approval_type, status)


__all__ = [
    "record_governance_verification",
    "record_governance_decision",
    "record_governance_approval",
]
