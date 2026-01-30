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
- AISystemsAuditor: Prompt injection, guardrails, hallucination risks

Enterprise vertical auditors:
- HealthcareAuditor: HIPAA, PHI detection, clinical documentation
- RegulatoryAuditor: SOX, GDPR, PCI-DSS, industry compliance
- AcademicAuditor: Citation verification, plagiarism detection
"""

from aragora.audit.audit_types.security import SecurityAuditor
from aragora.audit.audit_types.compliance import ComplianceAuditor
from aragora.audit.audit_types.consistency import ConsistencyAuditor
from aragora.audit.audit_types.quality import QualityAuditor
from aragora.audit.audit_types.legal import LegalAuditor
from aragora.audit.audit_types.accounting import AccountingAuditor
from aragora.audit.audit_types.software import SoftwareAuditor
from aragora.audit.audit_types.ai_systems import AISystemsAuditor, AIRiskCategory

# Enterprise vertical auditors
from aragora.audit.audit_types.healthcare import HealthcareAuditor, PHIDetector
from aragora.audit.audit_types.regulatory import (
    RegulatoryAuditor,
    RegulatoryFramework,
    GDPRDataMapper,
)
from aragora.audit.audit_types.academic import AcademicAuditor, CitationExtractor, CitationStyle


def register_all_auditors() -> None:
    """Register all built-in auditors with the global registry."""
    from aragora.audit.registry import audit_registry

    # Core auditors (legacy - don't inherit from BaseAuditor)
    audit_registry.register_legacy(
        "security",
        SecurityAuditor(),
        display_name="Security Analysis",
        description="Detects credentials, injection vulnerabilities, and security risks",
    )
    audit_registry.register_legacy(
        "compliance",
        ComplianceAuditor(),
        display_name="Compliance Check",
        description="Checks GDPR, HIPAA, SOC2, and contractual compliance",
    )
    audit_registry.register_legacy(
        "consistency",
        ConsistencyAuditor(),
        display_name="Consistency Analysis",
        description="Finds cross-document contradictions and inconsistencies",
    )
    audit_registry.register_legacy(
        "quality",
        QualityAuditor(),
        display_name="Quality Assessment",
        description="Evaluates ambiguity, completeness, and documentation quality",
    )

    # Domain-specific auditors
    audit_registry.register(LegalAuditor())
    audit_registry.register(AccountingAuditor())
    audit_registry.register(SoftwareAuditor())
    audit_registry.register(AISystemsAuditor())

    # Enterprise vertical auditors
    audit_registry.register(HealthcareAuditor())
    audit_registry.register(RegulatoryAuditor())
    audit_registry.register(AcademicAuditor())


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
    "AISystemsAuditor",
    "AIRiskCategory",
    # Enterprise vertical auditors
    "HealthcareAuditor",
    "PHIDetector",
    "RegulatoryAuditor",
    "RegulatoryFramework",
    "GDPRDataMapper",
    "AcademicAuditor",
    "CitationExtractor",
    "CitationStyle",
    # Registration
    "register_all_auditors",
]
