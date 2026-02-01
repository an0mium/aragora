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
    from typing import Callable

    _RecordFunc = Callable[[str, str], None]


def record_governance_verification(verification_type: str, result: str) -> None:
    """Record governance verification metric if available."""
    try:
        from aragora.observability.metrics import (
            record_governance_verification as _record,
        )

        _record(verification_type, result)
    except ImportError:
        pass


def record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record governance decision metric if available."""
    try:
        from aragora.observability.metrics import (
            record_governance_decision as _record,
        )

        _record(decision_type, outcome)
    except ImportError:
        pass


def record_governance_approval(approval_type: str, status: str) -> None:
    """Record governance approval metric if available."""
    try:
        from aragora.observability.metrics import (
            record_governance_approval as _record,
        )

        _record(approval_type, status)
    except ImportError:
        pass


__all__ = [
    "record_governance_verification",
    "record_governance_decision",
    "record_governance_approval",
]
