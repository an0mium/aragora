"""
Agent selection for automated debate team composition.

This module provides intelligent agent selection using:
- QuestionClassifier for matching question types to personas
- AgentSelector for team composition with ELO and diversity optimization
"""

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from aragora.config import ALLOWED_AGENT_TYPES
from aragora.server.initialization import (
    ROUTING_AVAILABLE,
    AgentProfile,
    AgentSelector,
    TaskRequirements,
)

if TYPE_CHECKING:
    from aragora.agents.personas import PersonaManager
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


def auto_select_agents(
    question: str,
    config: dict,
    elo_system: Optional["EloSystem"] = None,
    persona_manager: Optional["PersonaManager"] = None,
) -> str:
    """Select optimal agents using question classification and AgentSelector.

    First tries the QuestionClassifier to match question type to appropriate
    personas (e.g., ethical/theological -> philosopher, humanist).
    Falls back to AgentSelector if classifier returns no recommendations.

    Args:
        question: The debate question/topic
        config: Optional configuration with:
            - primary_domain: Main domain (default: 'general')
            - secondary_domains: Additional domains
            - min_agents: Minimum team size (default: 2)
            - max_agents: Maximum team size (default: 4)
            - quality_priority: 0-1 scale (default: 0.7)
            - diversity_preference: 0-1 scale (default: 0.5)
        elo_system: Optional EloSystem for agent ratings
        persona_manager: Optional PersonaManager for agent specialization

    Returns:
        Comma-separated string of agent types with optional roles
    """
    # Try question classifier first for better persona matching
    try:
        from aragora.server.question_classifier import classify_and_assign_agents_sync

        agent_string, classification = classify_and_assign_agents_sync(question)

        if classification.recommended_personas and len(classification.recommended_personas) >= 2:
            logger.info(
                f"[auto_select] Classified as '{classification.category}' "
                f"(confidence={classification.confidence:.2f}), "
                f"personas={classification.recommended_personas}"
            )
            return agent_string
        else:
            logger.info(
                f"[auto_select] Classifier returned insufficient personas for "
                f"'{classification.category}', falling back to AgentSelector"
            )
    except Exception as e:
        logger.warning(f"[auto_select] Question classification failed: {e}, using AgentSelector")

    # Fall back to AgentSelector
    if not ROUTING_AVAILABLE:
        return "gemini,anthropic-api"  # Fallback

    try:
        # Build task requirements from question and config
        requirements = TaskRequirements(
            task_id=f"debate-{uuid.uuid4().hex[:8]}",
            description=question[:500],  # Truncate for safety
            primary_domain=config.get("primary_domain", "general"),
            secondary_domains=config.get("secondary_domains", []),
            required_traits=config.get("required_traits", []),
            min_agents=min(max(config.get("min_agents", 2), 2), 5),
            max_agents=min(max(config.get("max_agents", 4), 2), 8),
            quality_priority=min(max(config.get("quality_priority", 0.7), 0), 1),
            diversity_preference=min(max(config.get("diversity_preference", 0.5), 0), 1),
        )

        # Create selector with ELO system and persona manager
        selector = AgentSelector(
            elo_system=elo_system,
            persona_manager=persona_manager,
        )

        # Populate agent pool from allowed types
        for agent_type in ALLOWED_AGENT_TYPES:
            selector.register_agent(
                AgentProfile(
                    name=agent_type,
                    agent_type=agent_type,
                )
            )

        # Select optimal team
        team = selector.select_team(requirements)

        # Build agent string with roles if available
        agent_specs = []
        for agent in team.agents:
            role = team.roles.get(agent.name, "")
            if role:
                agent_specs.append(f"{agent.agent_type}:{role}")
            else:
                agent_specs.append(agent.agent_type)

        logger.info(
            f"[auto_select] Selected team: {agent_specs} (rationale: {team.rationale[:100]})"
        )
        return ",".join(agent_specs)

    except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
        logger.warning(f"[auto_select] Failed: {e}, using fallback")
        return "gemini,anthropic-api"  # Fallback on error


__all__ = ["auto_select_agents"]
