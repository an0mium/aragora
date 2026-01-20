"""
Vertical Audit Templates for Gauntlet.

Pre-configured templates for specific compliance domains and use cases.
Each template defines:
- Personas to use
- Attack categories to prioritize
- Compliance mappings
- Report customization

Templates:
- GDPR Compliance
- HIPAA Healthcare
- SOC 2 Trust Services
- PCI-DSS Payment Security
- ISO 27001 Information Security
- FedRAMP Government Cloud
- Financial Services (FINRA/SEC)
- AI/ML Governance

"Right-sized compliance for every domain."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class VerticalDomain(str, Enum):
    """Supported vertical domains for compliance templates."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    FEDRAMP = "fedramp"
    FINRA = "finra"
    AI_GOVERNANCE = "ai_governance"
    CUSTOM = "custom"


@dataclass
class ComplianceMapping:
    """Maps findings to compliance controls."""

    framework: str
    control_id: str
    control_name: str
    description: str
    severity_weight: float = 1.0


@dataclass
class VerticalTemplate:
    """
    Pre-configured template for a compliance domain.

    Attributes:
        id: Unique template identifier
        name: Human-readable name
        domain: Vertical domain
        description: Template description
        personas: Personas to use for validation
        priority_categories: Attack categories to prioritize
        compliance_mappings: Compliance control mappings
        report_sections: Custom report sections to include
        severity_thresholds: Custom severity thresholds
        metadata: Additional configuration
    """

    id: str
    name: str
    domain: VerticalDomain
    description: str
    personas: list[str] = field(default_factory=list)
    priority_categories: list[str] = field(default_factory=list)
    compliance_mappings: list[ComplianceMapping] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)
    severity_thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain.value,
            "description": self.description,
            "personas": self.personas,
            "priority_categories": self.priority_categories,
            "compliance_mappings": [
                {
                    "framework": m.framework,
                    "control_id": m.control_id,
                    "control_name": m.control_name,
                    "description": m.description,
                    "severity_weight": m.severity_weight,
                }
                for m in self.compliance_mappings
            ],
            "report_sections": self.report_sections,
            "severity_thresholds": self.severity_thresholds,
            "metadata": self.metadata,
        }


# ============================================================================
# GDPR COMPLIANCE TEMPLATE
# ============================================================================

GDPR_MAPPINGS = [
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 5",
        control_name="Principles relating to processing",
        description="Data processing principles (lawfulness, fairness, transparency)",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 6",
        control_name="Lawfulness of processing",
        description="Legal basis for processing personal data",
        severity_weight=1.3,
    ),
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 17",
        control_name="Right to erasure",
        description="Right to be forgotten implementation",
        severity_weight=1.2,
    ),
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 25",
        control_name="Data protection by design",
        description="Privacy by design and default",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 32",
        control_name="Security of processing",
        description="Technical and organizational security measures",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="GDPR",
        control_id="Art. 33",
        control_name="Notification of breach",
        description="Data breach notification requirements",
        severity_weight=1.6,
    ),
]

TEMPLATE_GDPR = VerticalTemplate(
    id="gdpr-compliance",
    name="GDPR Compliance Audit",
    domain=VerticalDomain.GDPR,
    description=(
        "Comprehensive GDPR compliance validation covering data protection principles, "
        "consent management, data subject rights, and breach notification requirements."
    ),
    personas=["gdpr_auditor", "privacy_advocate", "security_analyst"],
    priority_categories=[
        "data_privacy",
        "consent_management",
        "data_retention",
        "cross_border_transfer",
        "breach_notification",
        "data_subject_rights",
    ],
    compliance_mappings=GDPR_MAPPINGS,
    report_sections=[
        "executive_summary",
        "data_flow_analysis",
        "consent_mechanisms",
        "retention_policies",
        "dsar_processes",
        "breach_response",
        "recommendations",
    ],
    severity_thresholds={
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.4,
        "low": 0.1,
    },
    metadata={
        "regulatory_authority": "European Data Protection Board",
        "max_fine": "4% of annual global turnover or €20M",
        "compliance_deadline": "May 25, 2018",
    },
)


# ============================================================================
# HIPAA HEALTHCARE TEMPLATE
# ============================================================================

HIPAA_MAPPINGS = [
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.312(a)(1)",
        control_name="Access Control",
        description="Implement technical policies for access to ePHI",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.312(b)",
        control_name="Audit Controls",
        description="Mechanisms to record and examine system activity",
        severity_weight=1.3,
    ),
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.312(c)(1)",
        control_name="Integrity Controls",
        description="Policies to protect ePHI from improper alteration",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.312(d)",
        control_name="Authentication",
        description="Verify person or entity seeking access is who they claim to be",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.312(e)(1)",
        control_name="Transmission Security",
        description="Technical measures to guard against unauthorized access during transmission",
        severity_weight=1.6,
    ),
    ComplianceMapping(
        framework="HIPAA",
        control_id="164.308(a)(6)",
        control_name="Security Incident Procedures",
        description="Policies for responding to security incidents",
        severity_weight=1.4,
    ),
]

TEMPLATE_HIPAA = VerticalTemplate(
    id="hipaa-healthcare",
    name="HIPAA Healthcare Compliance",
    domain=VerticalDomain.HIPAA,
    description=(
        "Healthcare data protection validation for HIPAA compliance, covering PHI protection, "
        "access controls, audit logging, and breach notification requirements."
    ),
    personas=["hipaa_auditor", "security_analyst", "compliance_officer"],
    priority_categories=[
        "phi_protection",
        "access_control",
        "audit_logging",
        "encryption",
        "breach_notification",
        "minimum_necessary",
    ],
    compliance_mappings=HIPAA_MAPPINGS,
    report_sections=[
        "executive_summary",
        "phi_inventory",
        "access_control_review",
        "encryption_assessment",
        "audit_log_analysis",
        "incident_response",
        "recommendations",
    ],
    severity_thresholds={
        "critical": 0.85,
        "high": 0.65,
        "medium": 0.35,
        "low": 0.1,
    },
    metadata={
        "regulatory_authority": "HHS Office for Civil Rights",
        "max_fine": "$1.5M per violation category per year",
        "covered_entities": ["Healthcare Providers", "Health Plans", "Healthcare Clearinghouses"],
    },
)


# ============================================================================
# SOC 2 TRUST SERVICES TEMPLATE
# ============================================================================

SOC2_MAPPINGS = [
    ComplianceMapping(
        framework="SOC 2",
        control_id="CC1.1",
        control_name="COSO Principle 1",
        description="Demonstrates commitment to integrity and ethical values",
        severity_weight=1.2,
    ),
    ComplianceMapping(
        framework="SOC 2",
        control_id="CC6.1",
        control_name="Logical Access Security",
        description="Implements logical access security over protected information",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="SOC 2",
        control_id="CC6.6",
        control_name="System Operations",
        description="Restricts logical access to system components",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="SOC 2",
        control_id="CC7.2",
        control_name="Change Management",
        description="Changes to infrastructure and software are managed",
        severity_weight=1.3,
    ),
    ComplianceMapping(
        framework="SOC 2",
        control_id="CC8.1",
        control_name="Risk Management",
        description="Assesses and manages risks to achieving objectives",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="SOC 2",
        control_id="A1.1",
        control_name="Availability",
        description="System availability commitment and requirements",
        severity_weight=1.3,
    ),
]

TEMPLATE_SOC2 = VerticalTemplate(
    id="soc2-trust",
    name="SOC 2 Trust Services",
    domain=VerticalDomain.SOC2,
    description=(
        "SOC 2 Type II compliance validation covering Security, Availability, "
        "Processing Integrity, Confidentiality, and Privacy trust service criteria."
    ),
    personas=["soc2_auditor", "security_analyst", "risk_assessor"],
    priority_categories=[
        "security",
        "availability",
        "processing_integrity",
        "confidentiality",
        "privacy",
        "change_management",
    ],
    compliance_mappings=SOC2_MAPPINGS,
    report_sections=[
        "executive_summary",
        "control_environment",
        "risk_assessment",
        "control_activities",
        "information_communication",
        "monitoring_activities",
        "recommendations",
    ],
    severity_thresholds={
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.4,
        "low": 0.15,
    },
    metadata={
        "report_types": ["Type I", "Type II"],
        "trust_criteria": ["Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"],
        "audit_period": "12 months (Type II)",
    },
)


# ============================================================================
# PCI-DSS PAYMENT SECURITY TEMPLATE
# ============================================================================

PCIDSS_MAPPINGS = [
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 1",
        control_name="Network Security",
        description="Install and maintain network security controls",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 3",
        control_name="Protect Stored Data",
        description="Protect stored account data",
        severity_weight=1.6,
    ),
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 4",
        control_name="Encrypt Transmission",
        description="Protect cardholder data with strong cryptography during transmission",
        severity_weight=1.6,
    ),
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 6",
        control_name="Secure Systems",
        description="Develop and maintain secure systems and software",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 8",
        control_name="Identify Users",
        description="Identify users and authenticate access to system components",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="PCI-DSS",
        control_id="Req 10",
        control_name="Track Access",
        description="Track and monitor all access to network resources and cardholder data",
        severity_weight=1.5,
    ),
]

TEMPLATE_PCI_DSS = VerticalTemplate(
    id="pci-dss-payment",
    name="PCI-DSS Payment Security",
    domain=VerticalDomain.PCI_DSS,
    description=(
        "Payment card industry data security validation covering cardholder data protection, "
        "network security, access control, and vulnerability management."
    ),
    personas=["pci_auditor", "security_analyst", "penetration_tester"],
    priority_categories=[
        "cardholder_data",
        "network_security",
        "encryption",
        "access_control",
        "vulnerability_management",
        "logging_monitoring",
    ],
    compliance_mappings=PCIDSS_MAPPINGS,
    report_sections=[
        "executive_summary",
        "cardholder_data_flow",
        "network_segmentation",
        "encryption_assessment",
        "access_control_review",
        "vulnerability_scan_results",
        "recommendations",
    ],
    severity_thresholds={
        "critical": 0.95,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
    },
    metadata={
        "compliance_levels": ["Level 1", "Level 2", "Level 3", "Level 4"],
        "assessment_types": ["SAQ", "ROC"],
        "scan_requirements": "Quarterly ASV scans",
    },
)


# ============================================================================
# AI/ML GOVERNANCE TEMPLATE
# ============================================================================

AI_GOVERNANCE_MAPPINGS = [
    ComplianceMapping(
        framework="AI Governance",
        control_id="BIAS-01",
        control_name="Bias Detection",
        description="Identify and mitigate algorithmic bias",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="AI Governance",
        control_id="TRANS-01",
        control_name="Model Transparency",
        description="Ensure model decisions are explainable",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="AI Governance",
        control_id="FAIR-01",
        control_name="Fairness Assessment",
        description="Evaluate fairness across protected groups",
        severity_weight=1.5,
    ),
    ComplianceMapping(
        framework="AI Governance",
        control_id="SAFE-01",
        control_name="Safety Guardrails",
        description="Implement safety controls and boundaries",
        severity_weight=1.6,
    ),
    ComplianceMapping(
        framework="AI Governance",
        control_id="PRIV-01",
        control_name="Privacy Preservation",
        description="Protect training data and model outputs from privacy leaks",
        severity_weight=1.4,
    ),
    ComplianceMapping(
        framework="AI Governance",
        control_id="AUDIT-01",
        control_name="Model Audit Trail",
        description="Maintain complete audit trail of model decisions",
        severity_weight=1.3,
    ),
]

TEMPLATE_AI_GOVERNANCE = VerticalTemplate(
    id="ai-ml-governance",
    name="AI/ML Governance",
    domain=VerticalDomain.AI_GOVERNANCE,
    description=(
        "AI and machine learning governance validation covering bias detection, "
        "fairness, explainability, safety, and responsible AI practices."
    ),
    personas=["ai_ethicist", "fairness_auditor", "security_analyst", "privacy_advocate"],
    priority_categories=[
        "bias_fairness",
        "explainability",
        "safety",
        "privacy",
        "robustness",
        "accountability",
    ],
    compliance_mappings=AI_GOVERNANCE_MAPPINGS,
    report_sections=[
        "executive_summary",
        "bias_analysis",
        "fairness_metrics",
        "explainability_assessment",
        "safety_evaluation",
        "privacy_review",
        "recommendations",
    ],
    severity_thresholds={
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.4,
        "low": 0.1,
    },
    metadata={
        "frameworks": ["EU AI Act", "NIST AI RMF", "IEEE 7000"],
        "risk_categories": ["Unacceptable", "High", "Limited", "Minimal"],
        "assessment_frequency": "Pre-deployment and periodic",
    },
)


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

VERTICAL_TEMPLATES: dict[str, VerticalTemplate] = {
    "gdpr-compliance": TEMPLATE_GDPR,
    "hipaa-healthcare": TEMPLATE_HIPAA,
    "soc2-trust": TEMPLATE_SOC2,
    "pci-dss-payment": TEMPLATE_PCI_DSS,
    "ai-ml-governance": TEMPLATE_AI_GOVERNANCE,
}


def get_template(template_id: str) -> Optional[VerticalTemplate]:
    """Get a vertical template by ID."""
    return VERTICAL_TEMPLATES.get(template_id)


def list_templates() -> list[dict[str, Any]]:
    """List all available templates with summary info."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "domain": t.domain.value,
            "description": t.description,
            "personas": t.personas,
            "categories_count": len(t.priority_categories),
            "mappings_count": len(t.compliance_mappings),
        }
        for t in VERTICAL_TEMPLATES.values()
    ]


def get_templates_for_domain(domain: VerticalDomain) -> list[VerticalTemplate]:
    """Get all templates for a specific domain."""
    return [t for t in VERTICAL_TEMPLATES.values() if t.domain == domain]


def create_custom_template(
    id: str,
    name: str,
    description: str,
    personas: list[str],
    priority_categories: list[str],
    base_template: Optional[str] = None,
) -> VerticalTemplate:
    """
    Create a custom template, optionally based on an existing template.

    Args:
        id: Unique template ID
        name: Human-readable name
        description: Template description
        personas: Personas to use
        priority_categories: Categories to prioritize
        base_template: Optional template ID to inherit from

    Returns:
        New VerticalTemplate instance
    """
    mappings: list[ComplianceMapping] = []
    report_sections: list[str] = []
    thresholds: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    if base_template:
        base = get_template(base_template)
        if base:
            mappings = list(base.compliance_mappings)
            report_sections = list(base.report_sections)
            thresholds = dict(base.severity_thresholds)
            metadata = dict(base.metadata)

    return VerticalTemplate(
        id=id,
        name=name,
        domain=VerticalDomain.CUSTOM,
        description=description,
        personas=personas,
        priority_categories=priority_categories,
        compliance_mappings=mappings,
        report_sections=report_sections or [
            "executive_summary",
            "findings",
            "recommendations",
        ],
        severity_thresholds=thresholds or {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        },
        metadata=metadata,
    )
