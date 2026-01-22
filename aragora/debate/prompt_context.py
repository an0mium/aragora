"""
Prompt context building for debate agents.

Extracted from Arena to improve code organization. Handles:
- Building persona context for agent specialization
- Building flip/consistency context for position awareness
- Preparing audience context from suggestions
- Building proposal and revision prompts
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.audience.suggestions import cluster_suggestions, format_for_prompt

if TYPE_CHECKING:
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent, Critique
    from aragora.insights.flip_detector import FlipDetector
    from aragora.debate.audience_manager import AudienceManager
    from aragora.debate.prompt_builder import PromptBuilder
    from aragora.debate.protocol import DebateProtocol
    from aragora.spectate.stream import SpectatorStream

logger = logging.getLogger(__name__)


class PromptContextBuilder:
    """Builds context strings for debate agent prompts.

    Extracted from Arena to centralize prompt context building operations.
    Handles persona, flip detection, audience suggestions, and proposal/revision
    prompt construction.

    Usage:
        builder = PromptContextBuilder(
            persona_manager=personas,
            flip_detector=flip,
            protocol=protocol,
            prompt_builder=prompts,
            audience_manager=audience,
        )
        context = builder.get_persona_context(agent)
        flip_ctx = builder.get_flip_context(agent)
        prompt = builder.build_proposal_prompt(agent)
    """

    def __init__(
        self,
        persona_manager: Optional["PersonaManager"] = None,
        flip_detector: Optional["FlipDetector"] = None,
        protocol: Optional["DebateProtocol"] = None,
        prompt_builder: Optional["PromptBuilder"] = None,
        audience_manager: Optional["AudienceManager"] = None,
        spectator: Optional["SpectatorStream"] = None,
        notify_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize prompt context builder.

        Args:
            persona_manager: Optional manager for agent personas
            flip_detector: Optional detector for position consistency
            protocol: Debate protocol with audience injection settings
            prompt_builder: Core prompt builder for prompt construction
            audience_manager: Manager for audience events and suggestions
            spectator: Optional spectator stream for notifications
            notify_callback: Optional callback for spectator notifications
        """
        self.persona_manager = persona_manager
        self.flip_detector = flip_detector
        self.protocol = protocol
        self.prompt_builder = prompt_builder
        self.audience_manager = audience_manager
        self.spectator = spectator
        self._notify_callback = notify_callback

    def get_persona_context(self, agent: "Agent") -> str:
        """Get persona context for agent specialization.

        Args:
            agent: The agent to get persona context for

        Returns:
            Formatted persona context string, or empty string if not available
        """
        if not self.persona_manager:
            return ""

        # Try to get persona from database
        persona = self.persona_manager.get_persona(agent.name)
        if not persona:
            # Try default persona based on agent type (e.g., "claude_proposer" -> "claude")
            agent_type = agent.name.split("_")[0].lower()
            from aragora.agents.personas import DEFAULT_PERSONAS

            if agent_type in DEFAULT_PERSONAS:
                # DEFAULT_PERSONAS contains Persona objects directly
                persona = DEFAULT_PERSONAS[agent_type]
            else:
                return ""

        return persona.to_prompt_context()

    def get_flip_context(self, agent: "Agent") -> str:
        """Get flip/consistency context for agent self-awareness.

        This helps agents be aware of their position history and avoid
        unnecessary flip-flopping while still allowing genuine position changes.

        Args:
            agent: The agent to get flip context for

        Returns:
            Formatted flip context string, or empty string if not applicable
        """
        if not self.flip_detector:
            return ""

        try:
            consistency = self.flip_detector.get_agent_consistency(agent.name)

            # Skip if no position history yet
            if consistency.total_positions == 0:
                return ""

            # Only inject context if there are notable flips
            if consistency.total_flips == 0:
                return ""

            # Build context based on flip patterns
            lines = ["## Position Consistency Note"]

            # Warn about contradictions specifically
            if consistency.contradictions > 0:
                lines.append(
                    f"You have {consistency.contradictions} prior position contradiction(s) on record. "
                    "Consider your stance carefully before arguing against positions you previously held."
                )

            # Note retractions
            if consistency.retractions > 0:
                lines.append(
                    f"You have retracted {consistency.retractions} previous position(s). "
                    "If changing positions again, clearly explain your reasoning."
                )

            # Add overall consistency score
            score = consistency.consistency_score
            if score < 0.7:
                lines.append(
                    f"Your consistency score is {score:.0%}. Prioritize coherent positions."
                )

            # Note domains with instability
            if consistency.domains_with_flips:
                domains = ", ".join(consistency.domains_with_flips[:3])
                lines.append(f"Domains with position changes: {domains}")

            return "\n".join(lines) if len(lines) > 1 else ""

        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Flip context error for {agent.name}: {e}")
            return ""

    def _notify_spectator(self, event_type: str, **kwargs) -> None:
        """Notify spectator of an event."""
        if self._notify_callback:
            self._notify_callback(event_type, **kwargs)
        elif self.spectator:
            try:
                self.spectator.emit(event_type, **kwargs)
            except (TypeError, AttributeError) as e:
                logger.debug(f"Spectator notification error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected spectator notification error: {e}")

    def prepare_audience_context(self, emit_event: bool = False) -> str:
        """Prepare audience context for prompt building.

        Handles the shared pre-processing for prompt building:
        1. Drain pending audience events
        2. Compute audience section from suggestions

        Args:
            emit_event: Whether to emit spectator event for dashboard

        Returns:
            Formatted audience section string (empty if no suggestions)
        """
        # Drain pending audience events
        if self.audience_manager:
            self.audience_manager.drain_events()

        # Get user suggestions
        user_suggestions: deque = (
            self.audience_manager._suggestions if self.audience_manager else deque()
        )

        # Compute audience section if enabled and suggestions exist
        if not self.protocol:
            return ""

        audience_injection = getattr(self.protocol, "audience_injection", None)
        if not (audience_injection in ("summary", "inject") and user_suggestions):
            return ""

        clusters = cluster_suggestions(list(user_suggestions))
        audience_section = format_for_prompt(clusters)

        # Emit stream event for dashboard if requested
        if emit_event and self.spectator and clusters:
            self._notify_spectator(
                "audience_summary",
                details=f"{sum(c.count for c in clusters)} suggestions in {len(clusters)} clusters",
                metric=len(clusters),
            )

        return audience_section

    def build_proposal_prompt(self, agent: "Agent") -> str:
        """Build the initial proposal prompt.

        Args:
            agent: The agent to build the prompt for

        Returns:
            The formatted proposal prompt string
        """
        if not self.prompt_builder:
            raise ValueError("PromptBuilder is required for building prompts")

        audience_section = self.prepare_audience_context(emit_event=True)
        return self.prompt_builder.build_proposal_prompt(agent, audience_section)

    def build_revision_prompt(
        self,
        agent: "Agent",
        original: str,
        critiques: list["Critique"],
        round_number: int = 0,
    ) -> str:
        """Build the revision prompt including critiques and round-specific phase context.

        Args:
            agent: The agent to build the prompt for
            original: The original proposal text
            critiques: List of critiques to address
            round_number: Current round number for phase context

        Returns:
            The formatted revision prompt string
        """
        if not self.prompt_builder:
            raise ValueError("PromptBuilder is required for building prompts")

        audience_section = self.prepare_audience_context(emit_event=False)
        return self.prompt_builder.build_revision_prompt(
            agent, original, critiques, audience_section, round_number=round_number
        )


__all__ = ["PromptContextBuilder"]
