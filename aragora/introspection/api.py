"""
Core API for Agent Introspection.

Provides functions to aggregate introspection data from various sources
(reputation, personas, etc.) into a unified snapshot for prompt injection.
"""

import logging
from typing import Optional, TYPE_CHECKING

from .types import IntrospectionSnapshot

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.memory.store import CritiqueStore
    from aragora.agents.personas import PersonaManager


def get_agent_introspection(
    agent_name: str,
    memory: Optional["CritiqueStore"] = None,
    persona_manager: Optional["PersonaManager"] = None,
) -> IntrospectionSnapshot:
    """
    Aggregate introspection data from all available sources.

    Collects agent reputation, performance history, and persona traits
    into a unified snapshot for prompt injection.

    Data sources (in priority order):
    1. CritiqueStore.get_reputation() - reputation metrics
    2. PersonaManager.get_persona() - traits and expertise

    Args:
        agent_name: Name of the agent to get introspection for
        memory: Optional CritiqueStore for reputation data
        persona_manager: Optional PersonaManager for traits/expertise

    Returns:
        IntrospectionSnapshot with aggregated data (fields may be defaults
        if sources are unavailable)
    """
    snapshot = IntrospectionSnapshot(agent_name=agent_name)

    # 1. Reputation data (primary source)
    if memory is not None:
        try:
            rep = memory.get_reputation(agent_name)
            if rep is not None:
                snapshot.reputation_score = rep.score
                snapshot.vote_weight = rep.vote_weight
                snapshot.proposals_made = rep.proposals_made
                snapshot.proposals_accepted = rep.proposals_accepted
                snapshot.critiques_given = rep.critiques_given
                snapshot.critiques_valuable = rep.critiques_valuable
                snapshot.calibration_score = rep.calibration_score
        except Exception as e:
            # Graceful degradation - continue with defaults
            logger.debug(f"Could not get reputation for {agent_name}: {type(e).__name__}: {e}")

    # 2. Persona data (enrichment)
    if persona_manager is not None:
        try:
            persona = persona_manager.get_persona(agent_name)
            if persona is not None:
                # Get top expertise domains
                if hasattr(persona, "top_expertise"):
                    snapshot.top_expertise = [
                        domain for domain, _ in persona.top_expertise[:3]
                    ]
                elif hasattr(persona, "expertise") and persona.expertise:
                    # Fallback: sort expertise dict by score
                    sorted_expertise = sorted(
                        persona.expertise.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    snapshot.top_expertise = [
                        domain for domain, _ in sorted_expertise[:3]
                    ]

                # Get personality traits
                if hasattr(persona, "traits") and persona.traits:
                    snapshot.traits = persona.traits[:3]
        except Exception as e:
            # Graceful degradation - continue with defaults
            logger.debug(f"Could not get persona for {agent_name}: {type(e).__name__}: {e}")

    return snapshot


def format_introspection_section(
    snapshot: IntrospectionSnapshot,
    max_chars: int = 600,
) -> str:
    """
    Format introspection data for prompt injection.

    Convenience wrapper around snapshot.to_prompt_section().

    Args:
        snapshot: IntrospectionSnapshot to format
        max_chars: Maximum characters (default 600)

    Returns:
        Formatted prompt section string
    """
    return snapshot.to_prompt_section(max_chars=max_chars)
