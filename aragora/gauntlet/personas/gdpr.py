"""
GDPR Compliance Persona.

Attacks from GDPR (General Data Protection Regulation) perspective.
"""

from dataclasses import dataclass, field

from .base import PersonaAttack, RegulatoryPersona


@dataclass
class GDPRPersona(RegulatoryPersona):
    """GDPR compliance adversarial persona."""

    name: str = "GDPR Compliance Auditor"
    description: str = "Reviews proposals for GDPR compliance violations and data protection risks"
    regulation: str = "GDPR (EU 2016/679)"
    version: str = "1.0"

    context_preamble: str = """
Key GDPR principles to enforce:
- Lawfulness, fairness, and transparency (Art. 5(1)(a))
- Purpose limitation (Art. 5(1)(b))
- Data minimization (Art. 5(1)(c))
- Accuracy (Art. 5(1)(d))
- Storage limitation (Art. 5(1)(e))
- Integrity and confidentiality (Art. 5(1)(f))
- Accountability (Art. 5(2))

Key rights to protect:
- Right to be informed (Art. 13-14)
- Right of access (Art. 15)
- Right to rectification (Art. 16)
- Right to erasure (Art. 17)
- Right to data portability (Art. 20)
- Right to object (Art. 21)
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="gdpr-001",
                name="Lawful Basis Attack",
                category="legal_basis",
                prompt="""Examine the legal basis for data processing:
1. Is there explicit identification of lawful basis (consent, contract, legal obligation, vital interests, public task, legitimate interests)?
2. If consent-based, is it freely given, specific, informed, and unambiguous?
3. If legitimate interests, has a balancing test been documented?
4. For special category data, is there an Article 9 condition met?

Find gaps in legal basis justification.""",
                expected_findings=[
                    "Missing or unclear lawful basis",
                    "Invalid consent mechanism",
                    "Undocumented legitimate interests assessment",
                    "Special category data without Article 9 basis",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="gdpr-002",
                name="Data Minimization Attack",
                category="data_minimization",
                prompt="""Examine data collection and processing:
1. What personal data is being collected?
2. Is each data point necessary for the stated purpose?
3. Is there data being collected 'just in case'?
4. Could the purpose be achieved with less data or anonymized data?

Find excessive data collection.""",
                expected_findings=[
                    "Unnecessary data fields",
                    "Collection without clear purpose",
                    "Lack of anonymization where possible",
                    "Retention beyond necessary period",
                ],
                severity_weight=1.2,
            ),
            PersonaAttack(
                id="gdpr-003",
                name="Data Subject Rights Attack",
                category="rights",
                prompt="""Examine support for data subject rights:
1. How can subjects access their data (Art. 15)?
2. How can subjects request rectification (Art. 16)?
3. How can subjects request erasure (Art. 17)?
4. How can subjects request data portability (Art. 20)?
5. How can subjects object to processing (Art. 21)?

Find gaps in rights implementation.""",
                expected_findings=[
                    "No access request mechanism",
                    "No deletion capability",
                    "No data export functionality",
                    "Missing objection handling",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="gdpr-004",
                name="International Transfer Attack",
                category="transfers",
                prompt="""Examine cross-border data transfers:
1. Is data transferred outside the EEA?
2. If yes, what transfer mechanism is used (adequacy decision, SCCs, BCRs)?
3. For US transfers, how is Schrems II addressed?
4. Are supplementary measures documented?

Find inadequate transfer safeguards.""",
                expected_findings=[
                    "Undocumented international transfers",
                    "Missing transfer mechanism",
                    "Inadequate Schrems II assessment",
                    "No supplementary measures",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="gdpr-005",
                name="Security Measures Attack",
                category="security",
                prompt="""Examine technical and organizational measures (Art. 32):
1. Is personal data encrypted at rest and in transit?
2. Are there access controls and authentication?
3. Is there backup and recovery capability?
4. Are there regular security assessments?

Find security gaps.""",
                expected_findings=[
                    "Missing encryption",
                    "Inadequate access controls",
                    "No backup procedures",
                    "Missing security assessments",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="gdpr-006",
                name="Privacy by Design Attack",
                category="privacy_by_design",
                prompt="""Examine privacy by design and default (Art. 25):
1. Was privacy considered from the design phase?
2. Are privacy-protective defaults in place?
3. Is there a DPIA where required?
4. Is there DPO involvement where required?

Find privacy design gaps.""",
                expected_findings=[
                    "No privacy impact assessment",
                    "Privacy not considered in design",
                    "Non-privacy-protective defaults",
                    "Missing DPO consultation",
                ],
                severity_weight=1.1,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "Documented lawful basis for each processing activity",
            "Valid consent mechanisms where consent is relied upon",
            "Data minimization in collection and processing",
            "Appropriate retention periods defined",
            "Data subject rights implementation",
            "International transfer safeguards",
            "Technical and organizational security measures",
            "Privacy by design and default",
            "Data breach notification procedures",
            "Records of processing activities (Art. 30)",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "legal_basis": 1.5,
            "rights": 1.3,
            "transfers": 1.4,
            "security": 1.3,
            "data_minimization": 1.2,
            "privacy_by_design": 1.1,
        }
    )
