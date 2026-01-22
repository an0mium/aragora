"""
Deliberation helpers for the control plane.

Provides a thin wrapper around DecisionRouter to run deliberations,
persist results, and standardize stored payloads.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aragora.core.decision import DecisionRequest, DecisionResult, get_decision_router
from aragora.core.decision_results import save_decision_result

logger = logging.getLogger(__name__)


def build_decision_record(
    request_id: str,
    result: Optional[DecisionResult] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a DecisionResultStore record."""
    resolved_status = status or ("completed" if result and result.success else "failed")
    return {
        "request_id": request_id,
        "status": resolved_status,
        "result": result.to_dict() if result else {},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


async def run_deliberation(
    request: DecisionRequest,
    router: Optional[Any] = None,
) -> DecisionResult:
    """Run a deliberation and persist the result."""
    decision_router = router or get_decision_router()
    result = await decision_router.route(request)
    save_decision_result(request.request_id, build_decision_record(request.request_id, result))
    return result


def record_deliberation_error(request_id: str, error: str, status: str = "failed") -> None:
    """Persist a deliberation error result."""
    save_decision_result(
        request_id,
        build_decision_record(
            request_id=request_id,
            result=None,
            status=status,
            error=error,
        ),
    )
    logger.warning("deliberation_failed", request_id=request_id, error=error)


__all__ = [
    "build_decision_record",
    "run_deliberation",
    "record_deliberation_error",
]
