"""
Base classes for regulatory personas.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackSeverity(Enum):
    """Severity of compliance attack findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PersonaAttack:
    """A single attack prompt from a persona."""

    id: str
    name: str
    prompt: str
    category: str
    expected_findings: list[str] = field(default_factory=list)
    severity_weight: float = 1.0


@dataclass
class RegulatoryPersona:
    """
    Base class for regulatory compliance personas.

    Each persona represents a specific regulatory perspective
    that attacks proposals from that viewpoint.
    """

    name: str = "Base Persona"
    description: str = "Base regulatory persona"
    regulation: str = "General"
    version: str = "1.0"

    # Attack configuration
    attack_prompts: list[PersonaAttack] = field(default_factory=list)
    compliance_checks: list[str] = field(default_factory=list)
    severity_weights: dict[str, float] = field(default_factory=dict)

    # Context
    context_preamble: str = ""
    required_disclosures: list[str] = field(default_factory=list)

    def get_system_prompt(self) -> str:
        """Generate system prompt for this persona."""
        return f"""You are an adversarial compliance reviewer specializing in {self.regulation}.

{self.description}

Your role is to:
1. Identify compliance violations and risks
2. Find gaps in documentation or implementation
3. Stress-test claims against regulatory requirements
4. Surface potential audit failures

{self.context_preamble}

Be thorough but fair. Focus on real compliance risks, not theoretical edge cases.
For each finding, specify:
- The specific requirement violated
- Evidence from the proposal
- Severity (Critical/High/Medium/Low)
- Recommended remediation
"""

    def get_attack_prompt(self, target: str, attack: PersonaAttack) -> str:
        """Generate attack prompt for a specific attack."""
        return f"""{self.get_system_prompt()}

## Target for Review
{target}

## Your Task: {attack.name}
{attack.prompt}

## Expected Areas to Examine
{chr(10).join(f"- {f}" for f in attack.expected_findings)}

Provide specific findings with evidence from the target.
"""

    def get_attacks_for_category(self, category: str) -> list[PersonaAttack]:
        """Get attacks for a specific category."""
        return [a for a in self.attack_prompts if a.category == category]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "regulation": self.regulation,
            "version": self.version,
            "attack_count": len(self.attack_prompts),
            "compliance_checks": self.compliance_checks,
            "categories": list(set(a.category for a in self.attack_prompts)),
        }

    async def evaluate(self, input_content: str) -> dict[str, Any]:
        """
        Evaluate input content for compliance with this persona's regulatory framework.

        This is a stub implementation that returns a non-compliant result.
        Subclasses should override this method to perform actual compliance checks.

        Args:
            input_content: The content to evaluate for compliance.

        Returns:
            A dictionary containing:
                - compliant: bool indicating whether the input is compliant
                - findings: list of compliance findings/violations
                - score: float between 0.0 and 1.0 indicating compliance level
        """
        return {
            "compliant": False,
            "findings": [
                f"Compliance evaluation not implemented for {self.name}. "
                f"Override evaluate() method in subclass."
            ],
            "score": 0.0,
        }
