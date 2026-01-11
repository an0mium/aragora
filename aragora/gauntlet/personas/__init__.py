"""
Regulatory Personas - Pre-built adversarial personas for compliance stress-testing.

Provides domain-specific attack prompts and compliance checks for:
- GDPR (Data Protection)
- HIPAA (Healthcare)
- EU AI Act (AI Regulation)
- Security (Red Team)
"""

from .base import RegulatoryPersona, PersonaAttack
from .gdpr import GDPRPersona
from .hipaa import HIPAAPersona
from .ai_act import AIActPersona
from .security import SecurityPersona

__all__ = [
    "RegulatoryPersona",
    "PersonaAttack",
    "GDPRPersona",
    "HIPAAPersona",
    "AIActPersona",
    "SecurityPersona",
    "get_persona",
    "list_personas",
]

PERSONAS = {
    "gdpr": GDPRPersona,
    "hipaa": HIPAAPersona,
    "ai_act": AIActPersona,
    "security": SecurityPersona,
}


def get_persona(name: str) -> RegulatoryPersona:
    """Get persona by name."""
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona: {name}. Available: {list(PERSONAS.keys())}")
    return PERSONAS[name]()


def list_personas() -> list[str]:
    """List available personas."""
    return list(PERSONAS.keys())
