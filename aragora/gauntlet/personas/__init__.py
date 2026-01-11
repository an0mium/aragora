"""
Regulatory Personas - Pre-built adversarial personas for compliance stress-testing.

Provides domain-specific attack prompts and compliance checks for:
- GDPR (Data Protection)
- HIPAA (Healthcare)
- EU AI Act (AI Regulation)
- Security (Red Team)
- SOC2 (Trust Services)
- PCI-DSS (Payment Card Industry)
- NIST CSF (Cybersecurity Framework)
"""

from .base import RegulatoryPersona, PersonaAttack
from .gdpr import GDPRPersona
from .hipaa import HIPAAPersona
from .ai_act import AIActPersona
from .security import SecurityPersona
from .soc2 import SOC2Persona
from .pci_dss import PCIDSSPersona
from .nist_csf import NISTCSFPersona

__all__ = [
    "RegulatoryPersona",
    "PersonaAttack",
    "GDPRPersona",
    "HIPAAPersona",
    "AIActPersona",
    "SecurityPersona",
    "SOC2Persona",
    "PCIDSSPersona",
    "NISTCSFPersona",
    "get_persona",
    "list_personas",
]

PERSONAS = {
    "gdpr": GDPRPersona,
    "hipaa": HIPAAPersona,
    "ai_act": AIActPersona,
    "security": SecurityPersona,
    "soc2": SOC2Persona,
    "pci_dss": PCIDSSPersona,
    "pci-dss": PCIDSSPersona,  # Alias with hyphen
    "nist_csf": NISTCSFPersona,
    "nist-csf": NISTCSFPersona,  # Alias with hyphen
}


def get_persona(name: str) -> RegulatoryPersona:
    """Get persona by name."""
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona: {name}. Available: {list(PERSONAS.keys())}")
    return PERSONAS[name]()


def list_personas() -> list[str]:
    """List available personas."""
    return list(PERSONAS.keys())
