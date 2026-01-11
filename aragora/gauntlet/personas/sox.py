"""
SOX Compliance Persona.

Attacks from Sarbanes-Oxley Act perspective.
Focuses on IT General Controls (ITGC) for financial reporting.

SOX Section 404 requires management to assess internal controls
over financial reporting, including IT controls that support
the reliability of financial data.
"""

from dataclasses import dataclass, field

from .base import RegulatoryPersona, PersonaAttack


@dataclass
class SOXPersona(RegulatoryPersona):
    """SOX compliance adversarial persona for financial controls."""

    name: str = "SOX Section 404 Auditor"
    description: str = "Reviews proposals for Sarbanes-Oxley IT General Controls compliance"
    regulation: str = "SOX (Sarbanes-Oxley Act Section 404)"
    version: str = "1.0"

    context_preamble: str = """
Sarbanes-Oxley (SOX) IT General Controls (ITGC) Requirements:

SOX focuses on controls that ensure the reliability of financial reporting.
IT systems that support financial data must have adequate controls.

KEY CONTROL DOMAINS:

1. ACCESS TO PROGRAMS AND DATA
   - Logical access to applications, data, and infrastructure
   - User authentication and authorization
   - Segregation of duties
   - Privileged access management

2. PROGRAM CHANGES
   - Change management for applications and infrastructure
   - Testing requirements for changes
   - Separation of development and production
   - Emergency change procedures

3. PROGRAM DEVELOPMENT
   - System development lifecycle (SDLC)
   - Security requirements in development
   - Testing and validation

4. COMPUTER OPERATIONS
   - Job scheduling and monitoring
   - Backup and recovery
   - Incident management
   - Data center controls

5. DATA INTEGRITY
   - Input/output controls
   - Processing controls
   - Interface controls between systems
   - Data reconciliation

MATERIAL WEAKNESS: A deficiency, or combination of deficiencies, in internal
control over financial reporting, such that there is a reasonable possibility
that a material misstatement will not be prevented or detected on a timely basis.
"""

    attack_prompts: list[PersonaAttack] = field(default_factory=lambda: [
        PersonaAttack(
            id="sox-001",
            name="Access Management & Segregation of Duties Attack",
            category="access_management",
            prompt="""Examine access controls and segregation of duties for financial systems:

1. Is there formal access provisioning tied to job roles?
2. Are access requests approved by data owners?
3. Is there segregation of duties (SoD) analysis?
   - Can same person create and approve transactions?
   - Can developers deploy to production?
   - Are financial posting rights separated from reconciliation?
4. Are terminated users removed within 24 hours?
5. Is there periodic access certification by managers?
6. Are privileged accounts limited and monitored?

Find access control gaps that could affect financial reporting integrity.""",
            expected_findings=[
                "No formal access provisioning process",
                "Missing segregation of duties controls",
                "Same user can create and approve transactions",
                "Developers have production access",
                "Terminated user access not removed timely",
                "No periodic access recertification",
                "Excessive privileged access",
            ],
            severity_weight=2.0,  # Critical for SOX
        ),
        PersonaAttack(
            id="sox-002",
            name="Change Management Attack",
            category="change_management",
            prompt="""Examine change management controls for financial systems:

1. Is there a documented change management policy?
2. Are changes to financial systems formally approved?
3. Is there segregation between development, testing, and production?
4. Are changes tested before deployment?
5. Is there evidence of change approval (tickets, sign-offs)?
6. Are emergency changes documented and ratified?
7. Is there version control and audit trail of changes?

Find change management gaps that could lead to unauthorized financial system modifications.""",
            expected_findings=[
                "No formal change management process",
                "Changes deployed without approval",
                "No segregation of development environments",
                "Missing testing documentation",
                "Emergency changes not properly ratified",
                "No version control for financial applications",
                "Missing audit trail of changes",
            ],
            severity_weight=1.8,
        ),
        PersonaAttack(
            id="sox-003",
            name="IT General Controls (ITGC) Attack",
            category="itgc_controls",
            prompt="""Examine IT General Controls supporting financial reporting:

1. Are key financial applications identified?
2. Is there an IT risk assessment covering financial systems?
3. Are IT policies documented and communicated?
4. Is there a control framework (COSO, COBIT)?
5. Are IT controls mapped to financial reporting risks?
6. Is there management oversight of IT controls?
7. Are control deficiencies tracked and remediated?

Find ITGC gaps that could be cited as material weaknesses.""",
            expected_findings=[
                "Financial systems not inventoried",
                "No IT risk assessment",
                "Missing IT policies and procedures",
                "Controls not mapped to financial risks",
                "No control framework adopted",
                "Insufficient management oversight",
                "Control deficiencies not tracked",
            ],
            severity_weight=1.8,
        ),
        PersonaAttack(
            id="sox-004",
            name="Data Integrity Attack",
            category="data_integrity",
            prompt="""Examine data integrity controls for financial data:

1. Are there input validation controls?
2. Is there completeness checking for financial transactions?
3. Are batch totals and hash totals used?
4. Are interface files validated and reconciled?
5. Is there automated vs. manual processing balance?
6. Are financial calculations validated?
7. Is there data quality monitoring?

Find data integrity gaps that could affect financial statement accuracy.""",
            expected_findings=[
                "Missing input validation",
                "No completeness controls",
                "Interfaces not reconciled",
                "Manual data manipulation possible",
                "Calculations not validated",
                "No data quality monitoring",
                "Financial data can be modified without audit trail",
            ],
            severity_weight=1.9,
        ),
        PersonaAttack(
            id="sox-005",
            name="Audit Trail Attack",
            category="audit_trail",
            prompt="""Examine audit trail and logging for financial systems:

1. Are all changes to financial data logged?
2. Are logs tamper-evident and protected?
3. Is there logging of user access to financial systems?
4. Are failed access attempts logged?
5. Is log retention adequate (7+ years for SOX)?
6. Can audit trails be provided on demand?
7. Are logs reviewed regularly?

Find audit trail gaps that would prevent reconstruction of financial events.""",
            expected_findings=[
                "Financial data changes not logged",
                "Logs can be modified or deleted",
                "Access to financial systems not tracked",
                "Failed logins not logged",
                "Insufficient log retention",
                "Cannot reconstruct transaction history",
                "No regular log review",
            ],
            severity_weight=1.7,
        ),
        PersonaAttack(
            id="sox-006",
            name="Section 404 Management Assessment Attack",
            category="section_404b",
            prompt="""Examine management's assessment of internal controls:

1. Is there a formal control testing program?
2. Are key controls identified and documented?
3. Is control design and operating effectiveness tested?
4. Are deficiencies categorized (deficiency, significant, material)?
5. Is there a timeline for deficiency remediation?
6. Is management attestation documented?
7. Are compensating controls documented where needed?

Find gaps in management's SOX 404 compliance program.""",
            expected_findings=[
                "No formal control testing program",
                "Key controls not identified",
                "Control effectiveness not tested",
                "Deficiencies not properly categorized",
                "No remediation tracking",
                "Missing management attestation",
                "Compensating controls not documented",
            ],
            severity_weight=1.6,
        ),
        PersonaAttack(
            id="sox-007",
            name="Material Weakness Identification Attack",
            category="material_weakness",
            prompt="""Identify potential material weaknesses in internal controls:

1. Are there control deficiencies that could cause material misstatement?
2. Is there a process to evaluate deficiency severity?
3. Are multiple deficiencies aggregated for materiality?
4. Could financial statements be manipulated?
5. Are fraud risks considered in control design?
6. Is there adequate monitoring of control operation?
7. Would current controls detect or prevent fraud?

Find control deficiencies that constitute material weaknesses.""",
            expected_findings=[
                "Control deficiencies could cause material misstatement",
                "No process to evaluate deficiency severity",
                "Multiple deficiencies not aggregated",
                "Financial data manipulation possible",
                "Fraud risks not considered",
                "Insufficient control monitoring",
                "Fraud could go undetected",
            ],
            severity_weight=2.0,
        ),
        PersonaAttack(
            id="sox-008",
            name="Backup and Recovery Attack",
            category="computer_operations",
            prompt="""Examine backup and recovery controls for financial data:

1. Are financial systems backed up regularly?
2. Are backups tested periodically?
3. Can financial data be restored to a point-in-time?
4. Is there a disaster recovery plan for financial systems?
5. Is Recovery Time Objective (RTO) defined for financial close?
6. Are backups stored offsite/in another region?
7. Is backup integrity verified?

Find backup/recovery gaps affecting financial system availability.""",
            expected_findings=[
                "Inadequate backup frequency",
                "Backups never tested",
                "Cannot restore to point-in-time",
                "No disaster recovery plan",
                "RTO not defined for financial close",
                "Backups not stored offsite",
                "Backup integrity not verified",
            ],
            severity_weight=1.5,
        ),
    ])

    compliance_checks: list[str] = field(default_factory=lambda: [
        "Formal access provisioning and deprovisioning procedures",
        "Segregation of duties analysis and enforcement",
        "Periodic access recertification by managers",
        "Privileged access management and monitoring",
        "Documented change management with approvals",
        "Segregation of development, test, and production",
        "Change testing and documentation requirements",
        "IT risk assessment covering financial systems",
        "Controls mapped to financial reporting risks",
        "Data integrity and validation controls",
        "Interface reconciliation procedures",
        "Comprehensive audit logging with retention",
        "Tamper-evident log protection",
        "Management control testing program",
        "Deficiency tracking and remediation",
        "Backup and disaster recovery testing",
    ])

    severity_weights: dict[str, float] = field(default_factory=lambda: {
        "access_management": 2.0,
        "material_weakness": 2.0,
        "data_integrity": 1.9,
        "change_management": 1.8,
        "itgc_controls": 1.8,
        "audit_trail": 1.7,
        "section_404b": 1.6,
        "computer_operations": 1.5,
    })
