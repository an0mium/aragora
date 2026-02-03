"""
Persona helper functions for retrieval and application.
"""

from __future__ import annotations

import logging

from aragora.agents.personas.core import Persona, PersonaManager
from aragora.agents.personas.defaults import DEFAULT_PERSONAS

logger = logging.getLogger(__name__)


def get_or_create_persona(manager: PersonaManager, agent_name: str) -> Persona:
    """Get existing persona or create from defaults."""
    persona = manager.get_persona(agent_name)

    if persona:
        return persona

    # Check for default
    base_name = agent_name.split("_")[0].lower()  # e.g., "claude_critic" -> "claude"
    if base_name in DEFAULT_PERSONAS:
        default = DEFAULT_PERSONAS[base_name]
        return manager.create_persona(
            agent_name=agent_name,
            description=default.description,
            traits=default.traits.copy(),
            expertise=default.expertise.copy(),
        )

    # Create empty persona
    return manager.create_persona(agent_name=agent_name)


def apply_persona_to_agent(agent, persona_name: str, manager: PersonaManager | None = None) -> bool:
    """
    Apply a persona to an agent instance.

    This is the unified method for applying personas across CLI and server.
    It sets the system prompt and generation parameters from the persona.

    Args:
        agent: Agent instance to modify
        persona_name: Name of the persona to apply
        manager: Optional PersonaManager for database personas

    Returns:
        True if persona was applied, False if persona not found
    """
    import logging

    logger = logging.getLogger(__name__)

    persona: Persona | None = None

    # Try default personas first
    if persona_name in DEFAULT_PERSONAS:
        persona = DEFAULT_PERSONAS[persona_name]
    elif manager:
        # Try database persona
        persona = manager.get_persona(persona_name)

    if not persona:
        logger.debug(f"Persona '{persona_name}' not found")
        return False

    # Build system prompt from persona
    persona_prompt = persona.to_prompt_context()

    if not persona_prompt:
        # Generate a simple prompt from traits and expertise
        parts = []
        if persona.traits:
            traits_str = ", ".join(persona.traits)
            parts.append(f"You are a {traits_str} agent.")
        if persona.description:
            parts.append(persona.description)
        if persona.top_expertise:
            top_domains = [d for d, _ in persona.top_expertise]
            parts.append(f"Your key areas of expertise: {', '.join(top_domains)}.")
        persona_prompt = " ".join(parts)

    # Apply system prompt
    if persona_prompt and hasattr(agent, "system_prompt"):
        existing = getattr(agent, "system_prompt", "") or ""
        agent.system_prompt = f"{persona_prompt}\n\n{existing}".strip()

    # Apply generation parameters
    if hasattr(agent, "set_generation_params"):
        agent.set_generation_params(
            temperature=persona.temperature,
            top_p=persona.top_p,
            frequency_penalty=persona.frequency_penalty,
        )
    else:
        # Try setting individual attributes
        if hasattr(agent, "temperature"):
            agent.temperature = persona.temperature
        if hasattr(agent, "top_p"):
            agent.top_p = persona.top_p
        if hasattr(agent, "frequency_penalty"):
            agent.frequency_penalty = persona.frequency_penalty

    logger.debug(
        f"Applied persona '{persona_name}' to agent: "
        f"temp={persona.temperature}, traits={persona.traits[:2] if persona.traits else []}"
    )
    return True


def get_persona_prompt(persona_name: str, manager: PersonaManager | None = None) -> str:
    """
    Get the system prompt for a persona.

    Args:
        persona_name: Name of the persona
        manager: Optional PersonaManager for database personas

    Returns:
        System prompt string, or empty string if persona not found
    """
    persona: Persona | None = None

    # Try default personas first
    if persona_name in DEFAULT_PERSONAS:
        persona = DEFAULT_PERSONAS[persona_name]
    elif manager:
        persona = manager.get_persona(persona_name)

    if not persona:
        return ""

    prompt = persona.to_prompt_context()

    if not prompt:
        # Generate a simple prompt
        parts = []
        if persona.traits:
            traits_str = ", ".join(persona.traits)
            parts.append(f"You are a {traits_str} agent.")
        if persona.description:
            parts.append(persona.description)
        if persona.top_expertise:
            top_domains = [d for d, _ in persona.top_expertise]
            parts.append(f"Your key areas of expertise: {', '.join(top_domains)}.")
        prompt = " ".join(parts)

    return prompt
