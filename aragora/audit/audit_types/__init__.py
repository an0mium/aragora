"""
Audit type handlers for document analysis.

Each handler specializes in detecting specific categories of issues:
- SecurityAuditor: Credentials, injection, exposure risks
- ComplianceAuditor: GDPR, HIPAA, SOC2, contractual violations
- ConsistencyAuditor: Cross-document contradictions
- QualityAuditor: Ambiguity, completeness, documentation quality

Domain-specific auditors:
- LegalAuditor: Contract analysis, obligations, risk clauses
- AccountingAuditor: Financial irregularities, SOX, reconciliation
- SoftwareAuditor: SAST patterns, secrets, licenses, dependencies
"""

from aragora.audit.audit_types.security import SecurityAuditor
from aragora.audit.audit_types.compliance import ComplianceAuditor
from aragora.audit.audit_types.consistency import ConsistencyAuditor
from aragora.audit.audit_types.quality import QualityAuditor
from aragora.audit.audit_types.legal import LegalAuditor
from aragora.audit.audit_types.accounting import AccountingAuditor
from aragora.audit.audit_types.software import SoftwareAuditor


def register_all_auditors() -> None:
    """Register all built-in auditors with the global registry."""
    from aragora.audit.registry import audit_registry

    # Core auditors
    audit_registry.register(SecurityAuditor())
    audit_registry.register(ComplianceAuditor())
    audit_registry.register(ConsistencyAuditor())
    audit_registry.register(QualityAuditor())

    # Domain-specific auditors
    audit_registry.register(LegalAuditor())
    audit_registry.register(AccountingAuditor())
    audit_registry.register(SoftwareAuditor())


__all__ = [
    # Core auditors
    "SecurityAuditor",
    "ComplianceAuditor",
    "ConsistencyAuditor",
    "QualityAuditor",
    # Domain-specific auditors
    "LegalAuditor",
    "AccountingAuditor",
    "SoftwareAuditor",
    # Registration
    "register_all_auditors",
]
