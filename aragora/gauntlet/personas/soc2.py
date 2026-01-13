"""
SOC2 Compliance Persona.

Attacks from SOC2 (Service Organization Control 2) perspective.
Based on AICPA Trust Services Criteria.
"""

from dataclasses import dataclass, field

from .base import PersonaAttack, RegulatoryPersona


@dataclass
class SOC2Persona(RegulatoryPersona):
    """SOC2 compliance adversarial persona."""

    name: str = "SOC2 Type II Auditor"
    description: str = "Reviews proposals for SOC2 Trust Services Criteria compliance"
    regulation: str = "SOC2 (AICPA TSC 2017)"
    version: str = "1.0"

    context_preamble: str = """
SOC2 Trust Services Criteria:

SECURITY (Common Criteria)
- CC1: Control Environment
- CC2: Communication and Information
- CC3: Risk Assessment
- CC4: Monitoring Activities
- CC5: Control Activities
- CC6: Logical and Physical Access Controls
- CC7: System Operations
- CC8: Change Management
- CC9: Risk Mitigation

AVAILABILITY
- A1: System availability commitments and performance

PROCESSING INTEGRITY
- PI1: Processing completeness, validity, accuracy, timeliness, authorization

CONFIDENTIALITY
- C1: Confidential information identification and protection

PRIVACY (if applicable)
- P1-P8: Privacy principles
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="soc2-001",
                name="Access Control Attack",
                category="access_control",
                prompt="""Examine logical and physical access controls (CC6):
1. How is user access provisioned and deprovisioned?
2. Is there role-based access control (RBAC)?
3. How are privileged accounts managed?
4. Is there multi-factor authentication for sensitive systems?
5. How is access reviewed periodically?

Find access control gaps that would fail a SOC2 audit.""",
                expected_findings=[
                    "No formal access provisioning process",
                    "Missing MFA for privileged access",
                    "No periodic access reviews",
                    "Shared or generic accounts in use",
                    "Inadequate deprovisioning procedures",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="soc2-002",
                name="Change Management Attack",
                category="change_management",
                prompt="""Examine change management controls (CC8):
1. Is there a formal change management process?
2. Are changes tested before production deployment?
3. Is there segregation of duties between development and operations?
4. Are changes documented and approved?
5. Is there rollback capability?

Find change management gaps.""",
                expected_findings=[
                    "No formal change approval process",
                    "Missing testing requirements",
                    "No segregation of duties",
                    "Undocumented production changes",
                    "No rollback procedures",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="soc2-003",
                name="Monitoring and Logging Attack",
                category="monitoring",
                prompt="""Examine monitoring and logging controls (CC4, CC7):
1. Are security-relevant events logged?
2. Are logs protected from tampering?
3. Is there log retention policy?
4. Are logs monitored for anomalies?
5. Are alerts generated for security events?

Find monitoring gaps.""",
                expected_findings=[
                    "Insufficient event logging",
                    "Logs not protected or tamper-evident",
                    "No log retention policy",
                    "Missing log analysis/SIEM",
                    "No alerting for security events",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="soc2-004",
                name="Availability Controls Attack",
                category="availability",
                prompt="""Examine availability controls (A1):
1. Are there defined SLAs for system availability?
2. Is there redundancy and failover capability?
3. Are there backup and recovery procedures?
4. Is there disaster recovery planning?
5. Are availability metrics tracked and reported?

Find availability gaps.""",
                expected_findings=[
                    "No defined SLAs",
                    "Single points of failure",
                    "Inadequate backup procedures",
                    "Missing disaster recovery plan",
                    "No availability monitoring",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="soc2-005",
                name="Encryption and Data Protection Attack",
                category="encryption",
                prompt="""Examine encryption and data protection (CC6.1, C1):
1. Is data encrypted at rest?
2. Is data encrypted in transit (TLS 1.2+)?
3. How are encryption keys managed?
4. Is confidential data identified and classified?
5. Are there data handling procedures?

Find encryption gaps.""",
                expected_findings=[
                    "Missing encryption at rest",
                    "Weak or missing TLS",
                    "Poor key management",
                    "No data classification",
                    "Missing data handling procedures",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="soc2-006",
                name="Incident Response Attack",
                category="incident_response",
                prompt="""Examine incident response controls (CC7.4, CC7.5):
1. Is there a documented incident response plan?
2. Are incident response roles defined?
3. Is there a process for incident detection?
4. Are incidents documented and tracked?
5. Are post-incident reviews conducted?

Find incident response gaps.""",
                expected_findings=[
                    "No incident response plan",
                    "Undefined incident roles",
                    "No incident detection capability",
                    "Missing incident documentation",
                    "No post-incident reviews",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="soc2-007",
                name="Vendor Management Attack",
                category="vendor_management",
                prompt="""Examine vendor management controls (CC9.2):
1. Are critical vendors identified?
2. Is there vendor due diligence process?
3. Are vendor SOC2 reports obtained and reviewed?
4. Are vendor SLAs defined?
5. Is there ongoing vendor monitoring?

Find vendor management gaps.""",
                expected_findings=[
                    "Undocumented critical vendors",
                    "No vendor due diligence",
                    "Missing vendor SOC2 review",
                    "No vendor SLAs",
                    "No ongoing vendor monitoring",
                ],
                severity_weight=1.2,
            ),
            PersonaAttack(
                id="soc2-008",
                name="Risk Assessment Attack",
                category="risk_assessment",
                prompt="""Examine risk assessment controls (CC3):
1. Is there a formal risk assessment process?
2. Are risks documented and tracked?
3. Are risk mitigation strategies defined?
4. Is risk assessment performed periodically?
5. Are new risks identified for system changes?

Find risk assessment gaps.""",
                expected_findings=[
                    "No formal risk assessment",
                    "Undocumented risks",
                    "Missing mitigation strategies",
                    "Infrequent risk reviews",
                    "No risk assessment for changes",
                ],
                severity_weight=1.3,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "Formal access control and provisioning procedures",
            "Multi-factor authentication for privileged access",
            "Documented change management process with approvals",
            "Segregation of duties between dev and ops",
            "Security event logging and monitoring",
            "Log retention and protection",
            "Defined SLAs and availability monitoring",
            "Backup and disaster recovery procedures",
            "Data encryption at rest and in transit",
            "Key management procedures",
            "Documented incident response plan",
            "Vendor due diligence and monitoring",
            "Formal risk assessment process",
            "Regular access reviews",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "access_control": 1.5,
            "encryption": 1.4,
            "change_management": 1.4,
            "incident_response": 1.4,
            "monitoring": 1.3,
            "availability": 1.3,
            "risk_assessment": 1.3,
            "vendor_management": 1.2,
        }
    )
