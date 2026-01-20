"""
Persona feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles persona performance updates and trait emergence detection.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.type_protocols import EventEmitterProtocol, PersonaManagerProtocol

logger = logging.getLogger(__name__)


class PersonaFeedback:
    """Handles persona-related feedback operations."""

    def __init__(
        self,
        persona_manager: Optional["PersonaManagerProtocol"] = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
        loop_id: Optional[str] = None,
    ):
        self.persona_manager = persona_manager
        self.event_emitter = event_emitter
        self.loop_id = loop_id

    def update_persona_performance(self, ctx: "DebateContext") -> None:
        """Update PersonaManager with performance feedback."""
        if not self.persona_manager:
            return

        try:
            from aragora.agents.errors import _build_error_action

            result = ctx.result
            for agent in ctx.agents:
                success = (agent.name == result.winner) or (
                    result.consensus_reached and result.confidence > 0.7
                )
                self.persona_manager.record_performance(
                    agent_name=agent.name,
                    domain=ctx.domain,
                    success=success,
                )

            # Check for trait emergence after performance updates
            self.check_trait_emergence(ctx)
        except Exception as e:
            from aragora.agents.errors import _build_error_action

            _, msg, exc_info = _build_error_action(e, "persona")
            logger.warning("Persona update failed: %s", msg, exc_info=exc_info)

    def check_trait_emergence(self, ctx: "DebateContext") -> None:
        """Check if any new agent traits emerged from performance patterns.

        Traits emerge when an agent demonstrates consistent behavior patterns:
        - High win rates in specific domains
        - Consistent prediction accuracy
        - Distinct communication styles
        """
        if not self.persona_manager or not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            for agent in ctx.agents:
                # Get agent's current traits
                persona = self.persona_manager.get_persona(agent.name)
                if not persona:
                    continue

                # Check for newly emerged traits
                new_traits = getattr(persona, "emerging_traits", [])
                if not new_traits:
                    # Try to detect traits from performance history
                    new_traits = self.detect_emerging_traits(agent.name, ctx)

                for trait in new_traits:
                    self.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.TRAIT_EMERGED,
                            loop_id=self.loop_id,
                            data={
                                "agent": agent.name,
                                "trait": trait.get("name", "unknown"),
                                "description": trait.get("description", ""),
                                "confidence": trait.get("confidence", 0.5),
                                "domain": ctx.domain,
                                "debate_id": ctx.debate_id,
                            },
                        )
                    )
                    logger.info(
                        "[persona] Trait emerged for %s: %s",
                        agent.name,
                        trait.get("name", "unknown"),
                    )

        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug(f"Trait emergence check error: {e}")

    def detect_emerging_traits(
        self, agent_name: str, ctx: "DebateContext"
    ) -> List[Dict[str, Any]]:
        """Detect traits based on agent performance patterns.

        Returns list of trait dicts with name, description, confidence.
        """
        traits: List[Dict[str, Any]] = []

        try:
            # Get performance stats if available
            if not hasattr(self.persona_manager, "get_performance_stats"):
                return traits

            stats = self.persona_manager.get_performance_stats(agent_name)
            if not stats:
                return traits

            # Domain specialist: High win rate in specific domain
            domain_wins = stats.get("domain_wins", {})
            if ctx.domain in domain_wins and domain_wins[ctx.domain] >= 3:
                traits.append(
                    {
                        "name": f"{ctx.domain}_specialist",
                        "description": f"Demonstrated expertise in {ctx.domain} domain",
                        "confidence": min(0.9, 0.5 + (domain_wins[ctx.domain] * 0.1)),
                    }
                )

            # High calibration: Consistent accurate predictions
            accuracy = stats.get("prediction_accuracy", 0.0)
            if accuracy >= 0.8 and stats.get("total_predictions", 0) >= 5:
                traits.append(
                    {
                        "name": "well_calibrated",
                        "description": f"Highly accurate predictions ({accuracy:.0%})",
                        "confidence": accuracy,
                    }
                )

            # Consistent winner: High overall win rate
            win_rate = stats.get("win_rate", 0.0)
            if win_rate >= 0.7 and stats.get("total_debates", 0) >= 5:
                traits.append(
                    {
                        "name": "consistent_winner",
                        "description": f"Wins {win_rate:.0%} of debates",
                        "confidence": win_rate,
                    }
                )

        except (TypeError, ValueError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.debug(f"Trait detection error for {agent_name}: {e}")

        return traits


__all__ = ["PersonaFeedback"]
