"""
NIST Cybersecurity Framework Persona.

Attacks from NIST CSF (Cybersecurity Framework) perspective.
Based on NIST CSF 2.0.
"""

from dataclasses import dataclass, field

from .base import PersonaAttack, RegulatoryPersona


@dataclass
class NISTCSFPersona(RegulatoryPersona):
    """NIST Cybersecurity Framework adversarial persona."""

    name: str = "NIST CSF Security Assessor"
    description: str = "Reviews proposals against NIST Cybersecurity Framework functions"
    regulation: str = "NIST CSF 2.0"
    version: str = "1.0"

    context_preamble: str = """
NIST Cybersecurity Framework 2.0 Functions:

GOVERN (GV) - New in 2.0
- Organizational context, risk strategy, policies, oversight, supply chain

IDENTIFY (ID)
- Asset management, risk assessment, improvement

PROTECT (PR)
- Identity management, awareness, data security, platform security, resilience

DETECT (DE)
- Continuous monitoring, adverse event analysis

RESPOND (RS)
- Incident management, analysis, mitigation, reporting, communications

RECOVER (RC)
- Recovery planning, execution, communications
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="nist-001",
                name="Govern Function Attack",
                category="govern",
                prompt="""Examine governance and risk strategy (GV):
1. Is there a documented cybersecurity policy?
2. Are roles and responsibilities defined?
3. Is there executive oversight of cybersecurity?
4. Is there a risk management strategy?
5. Is supply chain risk addressed?

Find governance gaps.""",
                expected_findings=[
                    "No cybersecurity policy",
                    "Undefined roles and responsibilities",
                    "Lack of executive oversight",
                    "No risk management strategy",
                    "Supply chain risks not addressed",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="nist-002",
                name="Identify Function Attack",
                category="identify",
                prompt="""Examine asset and risk identification (ID):
1. Is there a complete asset inventory?
2. Are assets classified by criticality?
3. Is there a risk assessment process?
4. Are vulnerabilities identified and tracked?
5. Are business environment and dependencies understood?

Find identification gaps.""",
                expected_findings=[
                    "Incomplete asset inventory",
                    "No asset classification",
                    "No risk assessment",
                    "Unknown vulnerabilities",
                    "Dependencies not mapped",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="nist-003",
                name="Protect - Identity Management Attack",
                category="protect_identity",
                prompt="""Examine identity and access management (PR.AA):
1. Are identities managed and authenticated?
2. Is there role-based access control?
3. Is MFA implemented for privileged access?
4. Are credentials protected?
5. Is access reviewed periodically?

Find identity management gaps.""",
                expected_findings=[
                    "Weak identity management",
                    "No RBAC implementation",
                    "Missing MFA",
                    "Poor credential management",
                    "No access reviews",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="nist-004",
                name="Protect - Data Security Attack",
                category="protect_data",
                prompt="""Examine data security controls (PR.DS):
1. Is data-at-rest protected (encryption)?
2. Is data-in-transit protected (TLS)?
3. Are data integrity controls in place?
4. Is there data loss prevention?
5. Are backups performed and tested?

Find data security gaps.""",
                expected_findings=[
                    "Unencrypted data at rest",
                    "Unencrypted data in transit",
                    "No data integrity verification",
                    "No DLP controls",
                    "Untested backups",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="nist-005",
                name="Protect - Platform Security Attack",
                category="protect_platform",
                prompt="""Examine platform and infrastructure security (PR.PS):
1. Is there configuration management?
2. Are systems hardened to baselines?
3. Is software integrity verified?
4. Are patches applied timely?
5. Is there change management?

Find platform security gaps.""",
                expected_findings=[
                    "No configuration management",
                    "Unhardened systems",
                    "No software integrity checks",
                    "Delayed patching",
                    "No change management",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="nist-006",
                name="Detect Function Attack",
                category="detect",
                prompt="""Examine detection capabilities (DE):
1. Is there continuous security monitoring?
2. Are anomalies detected and analyzed?
3. Is there a SIEM or log aggregation?
4. Are security events correlated?
5. Is there threat intelligence integration?

Find detection gaps.""",
                expected_findings=[
                    "No continuous monitoring",
                    "No anomaly detection",
                    "No SIEM implementation",
                    "Events not correlated",
                    "No threat intelligence",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="nist-007",
                name="Respond Function Attack",
                category="respond",
                prompt="""Examine incident response capabilities (RS):
1. Is there an incident response plan?
2. Are incidents analyzed and categorized?
3. Is there containment and mitigation capability?
4. Are incidents reported to stakeholders?
5. Is there post-incident analysis?

Find response gaps.""",
                expected_findings=[
                    "No incident response plan",
                    "No incident categorization",
                    "No containment capability",
                    "Missing incident reporting",
                    "No post-incident reviews",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="nist-008",
                name="Recover Function Attack",
                category="recover",
                prompt="""Examine recovery capabilities (RC):
1. Is there a recovery plan?
2. Are recovery procedures tested?
3. Are recovery priorities defined?
4. Is there communication during recovery?
5. Are lessons learned incorporated?

Find recovery gaps.""",
                expected_findings=[
                    "No recovery plan",
                    "Untested recovery procedures",
                    "Undefined recovery priorities",
                    "No recovery communications",
                    "No lessons learned process",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="nist-009",
                name="Resilience Attack",
                category="resilience",
                prompt="""Examine overall resilience (PR.IR):
1. Is there redundancy for critical systems?
2. Is there failover capability?
3. Are recovery time objectives (RTO) defined?
4. Are recovery point objectives (RPO) defined?
5. Is there disaster recovery capability?

Find resilience gaps.""",
                expected_findings=[
                    "No redundancy",
                    "No failover capability",
                    "Undefined RTO",
                    "Undefined RPO",
                    "No DR capability",
                ],
                severity_weight=1.3,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "Documented cybersecurity policy and strategy",
            "Defined roles and responsibilities",
            "Executive oversight and risk governance",
            "Complete asset inventory with classification",
            "Formal risk assessment process",
            "Identity and access management program",
            "Multi-factor authentication for privileged access",
            "Data encryption at rest and in transit",
            "System hardening and configuration management",
            "Timely patch management",
            "Continuous security monitoring",
            "SIEM or equivalent log management",
            "Documented incident response plan",
            "Tested recovery procedures",
            "Defined RTO and RPO objectives",
            "Supply chain risk management",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "protect_identity": 1.5,
            "protect_data": 1.5,
            "govern": 1.4,
            "protect_platform": 1.4,
            "detect": 1.4,
            "respond": 1.4,
            "identify": 1.3,
            "recover": 1.3,
            "resilience": 1.3,
        }
    )
