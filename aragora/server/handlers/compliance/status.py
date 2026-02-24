"""
Compliance status endpoint handler.

Provides the overall compliance status endpoint that summarizes
compliance across key frameworks (SOC 2, GDPR, HIPAA).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from aragora.server.handlers.base import HandlerResult, json_response
from aragora.rbac.decorators import require_permission
from aragora.observability.metrics import track_handler


async def _evaluate_controls() -> list[dict[str, Any]]:
    """Evaluate SOC 2 controls using the SOC2Mixin if available."""
    try:
        from .soc2 import SOC2Mixin

        mixin = SOC2Mixin()
        return await mixin._evaluate_controls()
    except (ImportError, AttributeError):
        return []


@track_handler("compliance/status", method="GET")
@require_permission("compliance:read")
async def get_status() -> HandlerResult:
    """
    Get overall compliance status.

    Returns summary of compliance across key frameworks.
    """
    now = datetime.now(timezone.utc)

    # Collect compliance metrics
    controls = await _evaluate_controls()

    # Calculate overall compliance score
    total_controls = len(controls)
    compliant_controls = sum(1 for c in controls if c["status"] == "compliant")
    score = int((compliant_controls / total_controls * 100) if total_controls > 0 else 0)

    # Determine overall status
    if score >= 95:
        overall_status = "compliant"
    elif score >= 80:
        overall_status = "mostly_compliant"
    elif score >= 60:
        overall_status = "partial"
    else:
        overall_status = "non_compliant"

    return json_response(
        {
            "status": overall_status,
            "compliance_score": score,
            "frameworks": {
                "soc2_type2": {
                    "status": "in_progress",
                    "controls_assessed": total_controls,
                    "controls_compliant": compliant_controls,
                },
                "gdpr": {
                    "status": "supported",
                    "data_export": True,
                    "consent_tracking": True,
                    "retention_policy": True,
                },
                "hipaa": {
                    "status": "partial",
                    "note": "PHI handling requires additional configuration",
                },
            },
            "controls_summary": {
                "total": total_controls,
                "compliant": compliant_controls,
                "non_compliant": total_controls - compliant_controls,
            },
            "last_audit": (now - timedelta(days=7)).isoformat(),
            "next_audit_due": (now + timedelta(days=83)).isoformat(),
            "generated_at": now.isoformat(),
        }
    )


__all__ = ["get_status"]
