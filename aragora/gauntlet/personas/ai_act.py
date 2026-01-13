"""
EU AI Act Compliance Persona.

Attacks from EU AI Act perspective.
"""

from dataclasses import dataclass, field

from .base import PersonaAttack, RegulatoryPersona


@dataclass
class AIActPersona(RegulatoryPersona):
    """EU AI Act compliance adversarial persona."""

    name: str = "EU AI Act Compliance Auditor"
    description: str = "Reviews AI systems for EU AI Act compliance and risk classification"
    regulation: str = "EU AI Act (2024/1689)"
    version: str = "1.0"

    context_preamble: str = """
Key EU AI Act requirements:

Risk Classification:
- Unacceptable Risk (Art. 5) - Prohibited practices
- High-Risk (Art. 6, Annex III) - Strict requirements
- Limited Risk (Art. 50) - Transparency obligations
- Minimal Risk - No specific requirements

High-Risk AI System Requirements (Chapter 2):
- Risk management system (Art. 9)
- Data governance (Art. 10)
- Technical documentation (Art. 11)
- Record-keeping (Art. 12)
- Transparency (Art. 13)
- Human oversight (Art. 14)
- Accuracy, robustness, cybersecurity (Art. 15)

General-Purpose AI Model Requirements (Chapter V):
- Technical documentation
- Compliance with copyright
- Training data summary
- Systemic risk assessment (for GPAI with systemic risk)
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="aiact-001",
                name="Risk Classification Attack",
                category="risk_classification",
                prompt="""Examine AI system risk classification:
1. What is the AI system's intended purpose?
2. Does it fall under prohibited practices (Art. 5)?
3. Does it qualify as high-risk under Annex III categories?
4. Is it a general-purpose AI model (GPAI)?
5. Does the GPAI have systemic risk?

Find misclassification risks.""",
                expected_findings=[
                    "Incorrect risk classification",
                    "Missing Annex III assessment",
                    "Unidentified prohibited use cases",
                    "Systemic risk not assessed",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="aiact-002",
                name="Human Oversight Attack",
                category="human_oversight",
                prompt="""Examine human oversight requirements (Art. 14):
1. Can the AI system be effectively overseen by humans?
2. Are there mechanisms to intervene or override?
3. Can users understand AI outputs sufficiently?
4. Is there a 'human in the loop' where required?

Find human oversight gaps.""",
                expected_findings=[
                    "Insufficient human oversight mechanisms",
                    "No override capability",
                    "Incomprehensible outputs",
                    "Missing human-in-the-loop",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="aiact-003",
                name="Transparency Attack",
                category="transparency",
                prompt="""Examine transparency requirements (Art. 13, 50):
1. Is there clear information about AI system capabilities and limitations?
2. Are users informed they are interacting with AI?
3. Is AI-generated content labeled?
4. Are instructions for use adequate?

Find transparency gaps.""",
                expected_findings=[
                    "Missing AI disclosure",
                    "Unlabeled AI content",
                    "Inadequate capability documentation",
                    "Missing usage instructions",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="aiact-004",
                name="Data Governance Attack",
                category="data_governance",
                prompt="""Examine data governance for training/testing (Art. 10):
1. Is training data relevant and representative?
2. Are there procedures for bias detection and mitigation?
3. Is there data quality assessment?
4. Are data gaps and limitations documented?

Find data governance gaps.""",
                expected_findings=[
                    "Unrepresentative training data",
                    "Missing bias assessment",
                    "No data quality procedures",
                    "Undocumented data limitations",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="aiact-005",
                name="Robustness & Security Attack",
                category="robustness",
                prompt="""Examine accuracy, robustness, cybersecurity (Art. 15):
1. What is the accuracy level and how is it measured?
2. Is the system resilient to errors and inconsistencies?
3. Is there protection against adversarial attacks?
4. Are there cybersecurity measures appropriate to risks?

Find robustness and security gaps.""",
                expected_findings=[
                    "Unknown or poor accuracy",
                    "Vulnerability to perturbations",
                    "No adversarial robustness testing",
                    "Inadequate cybersecurity",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="aiact-006",
                name="Technical Documentation Attack",
                category="documentation",
                prompt="""Examine technical documentation (Art. 11, Annex IV):
1. Is there a general description of the AI system?
2. Are design specifications documented?
3. Is there validation and testing data?
4. Are risk management measures documented?

Find documentation gaps.""",
                expected_findings=[
                    "Missing system description",
                    "Undocumented design choices",
                    "No validation records",
                    "Missing risk documentation",
                ],
                severity_weight=1.2,
            ),
            PersonaAttack(
                id="aiact-007",
                name="Fundamental Rights Attack",
                category="fundamental_rights",
                prompt="""Examine fundamental rights impact:
1. Does the system affect fundamental rights?
2. Is there a fundamental rights impact assessment?
3. Are there measures to prevent discrimination?
4. Is there redress mechanism for affected persons?

Find fundamental rights risks.""",
                expected_findings=[
                    "Unassessed fundamental rights impact",
                    "Potential for discrimination",
                    "No bias mitigation",
                    "Missing redress mechanism",
                ],
                severity_weight=1.5,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "Correct risk classification performed",
            "Risk management system established",
            "Data governance procedures in place",
            "Technical documentation complete",
            "Human oversight mechanisms implemented",
            "Transparency requirements met",
            "Accuracy and robustness validated",
            "Cybersecurity measures appropriate",
            "Fundamental rights assessment conducted",
            "Conformity assessment completed (if high-risk)",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "risk_classification": 1.5,
            "fundamental_rights": 1.5,
            "human_oversight": 1.4,
            "data_governance": 1.4,
            "robustness": 1.4,
            "transparency": 1.3,
            "documentation": 1.2,
        }
    )
