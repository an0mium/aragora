"""
Codebase Audit Reporting.

Contains dashboard data generation, risk score calculation,
and demo report generation logic.
"""

from __future__ import annotations

from typing import Any

from .rules import (
    Finding,
    FindingStatus,
    ScanType,
)
from .scanning import (
    _get_mock_bug_findings,
    _get_mock_dependency_findings,
    _get_mock_metrics,
    _get_mock_sast_findings,
    _get_mock_secrets_findings,
    _get_tenant_findings,
    _get_tenant_scans,
)


def calculate_risk_score(findings: list[Finding]) -> float:
    """Calculate overall risk score (0-100)."""
    if not findings:
        return 0.0

    weights = {"critical": 10, "high": 5, "medium": 2, "low": 1, "info": 0.1}
    total_weight = sum(weights.get(f.severity.value, 1) * f.confidence for f in findings)

    # Normalize to 0-100 (cap at 100)
    return min(100.0, total_weight)


def build_dashboard_data(tenant_id: str) -> dict[str, Any]:
    """Build dashboard summary data for a tenant.

    Args:
        tenant_id: The tenant to build dashboard data for.

    Returns:
        Dashboard data dict ready for API response.
    """
    scans = list(_get_tenant_scans(tenant_id).values())
    findings = list(_get_tenant_findings(tenant_id).values())

    # Get open findings only
    open_findings = [f for f in findings if f.status == FindingStatus.OPEN]

    # Count by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in open_findings:
        severity_counts[finding.severity.value] += 1

    # Count by type
    type_counts = {"sast": 0, "bugs": 0, "secrets": 0, "dependencies": 0}
    for finding in open_findings:
        if finding.scan_type.value in type_counts:
            type_counts[finding.scan_type.value] += 1

    # Get latest metrics
    latest_metrics = {}
    metrics_scans = [s for s in scans if s.scan_type == ScanType.METRICS]
    if metrics_scans:
        latest_metrics = metrics_scans[-1].metrics

    # Get recent scans
    recent_scans = sorted(scans, key=lambda s: s.started_at, reverse=True)[:5]

    # Sort open findings by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

    return {
        "summary": {
            "total_findings": len(open_findings),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "total_scans": len(scans),
            "risk_score": calculate_risk_score(open_findings),
        },
        "metrics": latest_metrics,
        "recent_scans": [s.to_dict() for s in recent_scans],
        "top_findings": [
            f.to_dict()
            for f in sorted(
                open_findings,
                key=lambda f: severity_order.get(f.severity.value, 5),
            )[:10]
        ],
    }


def build_demo_data() -> dict[str, Any]:
    """Build demo dashboard data with mock findings and metrics.

    Returns:
        Demo data dict ready for API response.
    """
    scan_id = "demo_scan"
    findings = (
        _get_mock_sast_findings(scan_id)
        + _get_mock_bug_findings(scan_id)
        + _get_mock_secrets_findings(scan_id)
        + _get_mock_dependency_findings(scan_id)
    )
    metrics = _get_mock_metrics()

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in findings:
        severity_counts[finding.severity.value] += 1

    return {
        "is_demo": True,
        "summary": {
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "type_counts": {
                "sast": 3,
                "bugs": 2,
                "secrets": 1,
                "dependencies": 2,
            },
            "total_scans": 5,
            "risk_score": 45.5,
        },
        "metrics": metrics,
        "findings": [f.to_dict() for f in findings],
    }
