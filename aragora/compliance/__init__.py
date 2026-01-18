"""
Compliance Framework Manager for Enterprise Multi-Agent Control Plane.

Provides compliance rule checking for industry verticals:
- Software: OWASP, SOC2, ISO 27001
- Legal: ABA Ethics, GDPR, CCPA
- Healthcare: HIPAA, FDA 21 CFR Part 11
- Accounting: GAAP, IFRS, SOX

Usage:
    from aragora.compliance import (
        ComplianceFramework,
        ComplianceFrameworkManager,
        check_compliance,
    )

    # Check content against frameworks
    manager = ComplianceFrameworkManager()
    result = await manager.check(
        content="Patient data will be stored in plaintext...",
        frameworks=["hipaa", "gdpr"],
    )

    if not result.compliant:
        for issue in result.issues:
            print(f"[{issue.severity}] {issue.framework}: {issue.description}")
"""

from aragora.compliance.framework import (
    ComplianceSeverity,
    ComplianceIssue,
    ComplianceCheckResult,
    ComplianceRule,
    ComplianceFramework,
    ComplianceFrameworkManager,
    COMPLIANCE_FRAMEWORKS,
    check_compliance,
)

__all__ = [
    "ComplianceSeverity",
    "ComplianceIssue",
    "ComplianceCheckResult",
    "ComplianceRule",
    "ComplianceFramework",
    "ComplianceFrameworkManager",
    "COMPLIANCE_FRAMEWORKS",
    "check_compliance",
]
