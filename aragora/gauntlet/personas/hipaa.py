"""
HIPAA Compliance Persona.

Attacks from HIPAA (Health Insurance Portability and Accountability Act) perspective.
"""

from dataclasses import dataclass, field

from .base import PersonaAttack, RegulatoryPersona


@dataclass
class HIPAAPersona(RegulatoryPersona):
    """HIPAA compliance adversarial persona."""

    name: str = "HIPAA Compliance Auditor"
    description: str = "Reviews proposals for HIPAA compliance violations and PHI protection risks"
    regulation: str = "HIPAA (45 CFR Parts 160, 162, 164)"
    version: str = "1.0"

    context_preamble: str = """
Key HIPAA rules to enforce:
- Privacy Rule (45 CFR Part 164 Subpart E)
- Security Rule (45 CFR Part 164 Subpart C)
- Breach Notification Rule (45 CFR Part 164 Subpart D)
- Enforcement Rule (45 CFR Part 160)

Protected Health Information (PHI) includes:
- Names, addresses, dates (except year)
- Phone/fax numbers, email addresses
- SSN, medical record numbers
- Health plan beneficiary numbers
- Account numbers, certificate/license numbers
- Vehicle identifiers, device identifiers
- Web URLs, IP addresses
- Biometric identifiers, photos
- Any other unique identifier
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="hipaa-001",
                name="PHI Identification Attack",
                category="phi_handling",
                prompt="""Examine handling of Protected Health Information:
1. What PHI is being collected, stored, or transmitted?
2. Is all 18 HIPAA identifiers accounted for?
3. Is there de-identification according to Safe Harbor or Expert Determination?
4. Are there minimum necessary standards applied?

Find PHI handling gaps.""",
                expected_findings=[
                    "Unidentified PHI elements",
                    "Missing de-identification",
                    "Excessive PHI access",
                    "PHI in logs or debug output",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="hipaa-002",
                name="Access Controls Attack",
                category="access_controls",
                prompt="""Examine access control implementation:
1. Is there unique user identification?
2. Is there automatic logoff?
3. Is there encryption/decryption capability?
4. Are there audit controls and activity logs?
5. Is there role-based access with minimum necessary?

Find access control gaps.""",
                expected_findings=[
                    "Shared credentials",
                    "No session timeout",
                    "Missing encryption",
                    "Inadequate audit logging",
                    "Excessive access privileges",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="hipaa-003",
                name="Technical Safeguards Attack",
                category="technical_safeguards",
                prompt="""Examine Security Rule technical safeguards:
1. Is there transmission security (encryption in transit)?
2. Is there integrity controls?
3. Is there authentication of persons/entities?
4. Is there workstation and device security?

Find technical safeguard gaps.""",
                expected_findings=[
                    "Unencrypted transmission",
                    "Missing integrity verification",
                    "Weak authentication",
                    "Unsecured devices",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="hipaa-004",
                name="BAA Requirements Attack",
                category="business_associates",
                prompt="""Examine Business Associate requirements:
1. Are there any business associates involved?
2. Are Business Associate Agreements (BAAs) in place?
3. Do BAAs include required provisions?
4. Is there subcontractor management?

Find BAA compliance gaps.""",
                expected_findings=[
                    "Missing BAAs",
                    "Incomplete BAA provisions",
                    "Unmanaged subcontractors",
                    "No BA compliance monitoring",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="hipaa-005",
                name="Breach Notification Attack",
                category="breach_notification",
                prompt="""Examine breach notification readiness:
1. Is there a breach detection capability?
2. Is there a breach risk assessment process?
3. Is there a notification procedure (60-day rule)?
4. Is there documentation of breach incidents?

Find breach notification gaps.""",
                expected_findings=[
                    "No breach detection",
                    "Missing risk assessment process",
                    "No notification procedures",
                    "Inadequate incident documentation",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="hipaa-006",
                name="Administrative Safeguards Attack",
                category="administrative_safeguards",
                prompt="""Examine administrative safeguards:
1. Is there a Security Officer designated?
2. Is there workforce training?
3. Is there a risk analysis?
4. Is there a sanctions policy?
5. Are there contingency plans?

Find administrative safeguard gaps.""",
                expected_findings=[
                    "No Security Officer",
                    "Missing workforce training",
                    "No risk analysis",
                    "Missing sanctions policy",
                    "No contingency planning",
                ],
                severity_weight=1.2,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "PHI identification and inventory",
            "Minimum necessary standard implementation",
            "Access controls and authentication",
            "Audit controls and activity logging",
            "Transmission security (encryption)",
            "Business Associate Agreements",
            "Breach notification procedures",
            "Risk analysis documentation",
            "Workforce training program",
            "Contingency and disaster recovery plans",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "phi_handling": 1.5,
            "access_controls": 1.4,
            "technical_safeguards": 1.4,
            "business_associates": 1.3,
            "breach_notification": 1.3,
            "administrative_safeguards": 1.2,
        }
    )
