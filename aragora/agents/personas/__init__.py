"""
Agent Personas with evolving specialization.

Inspired by Project Sid's emergent specialization, this module provides:
- Defined personality traits and expertise areas
- Specialization scores that evolve based on performance
- Persona-aware prompting for more focused critiques

Modules:
- core: Persona dataclass, PersonaManager, domain/trait constants
- defaults: DEFAULT_PERSONAS dictionary (60 predefined personas)
- helpers: get_or_create_persona, apply_persona_to_agent, get_persona_prompt
"""

from aragora.agents.personas.core import (
    EXPERTISE_DOMAINS,
    PERSONA_SCHEMA_VERSION,
    PERSONALITY_TRAITS,
    Persona,
    PersonaManager,
)
from aragora.agents.personas.defaults import DEFAULT_PERSONAS
from aragora.agents.personas.helpers import (
    apply_persona_to_agent,
    get_or_create_persona,
    get_persona_prompt,
)

__all__ = [
    "EXPERTISE_DOMAINS",
    "PERSONA_SCHEMA_VERSION",
    "PERSONALITY_TRAITS",
    "Persona",
    "PersonaManager",
    "DEFAULT_PERSONAS",
    "get_or_create_persona",
    "apply_persona_to_agent",
    "get_persona_prompt",
]
