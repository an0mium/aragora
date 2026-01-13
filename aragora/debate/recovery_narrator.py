"""
Recovery Narrator - Template-based commentary on resilience events.

Generates entertaining commentary for system failures and recoveries:
- Template-based narrative generation
- Mood-aware messaging
- ImmuneSystem integration
- Audience-friendly explanations

Inspired by nomic loop debate consensus on transparent resilience.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecoveryNarrative:
    """A narrative for a recovery event."""

    event_type: str
    agent: str
    headline: str
    narrative: str
    mood: str  # "tense", "triumphant", "cautionary", "neutral"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "agent": self.agent,
            "headline": self.headline,
            "narrative": self.narrative,
            "mood": self.mood,
            "timestamp": self.timestamp,
        }


class RecoveryNarrator:
    """
    Template-based narrator for system recovery events.

    Transforms technical failure/recovery events into engaging
    audience commentary. Integrates with TransparentImmuneSystem
    to receive health events and generate narratives.

    Usage:
        narrator = RecoveryNarrator()

        # Subscribe to immune system events
        immune = get_immune_system()
        immune.set_broadcast_callback(narrator.handle_health_event)

        # Or generate manually
        narrative = narrator.narrate("agent_timeout", "claude")
        print(narrative.headline)
        print(narrative.narrative)
    """

    # Template bank organized by event type
    TEMPLATES = {
        "agent_started": {
            "headlines": [
                "{agent} enters the arena",
                "{agent} is powering up",
                "{agent} joins the debate",
            ],
            "narratives": [
                "{agent} has entered the discussion. Let's see what insights they bring!",
                "Here comes {agent}, ready to contribute to the conversation.",
                "{agent} is warming up - expect great things!",
            ],
            "mood": "neutral",
        },
        "agent_timeout": {
            "headlines": [
                "{agent} is taking a moment...",
                "Timeout alert for {agent}!",
                "{agent} needs more thinking time",
            ],
            "narratives": [
                "{agent} is taking the scenic route through this problem. The audience holds its breath!",
                "Timeout alert! {agent} needs a moment to think deeply about this one.",
                "{agent} has hit a processing wall - but don't worry, we're on it!",
                "Looks like {agent} bit off more than they could chew. Recovery in progress...",
            ],
            "mood": "tense",
        },
        "agent_failed": {
            "headlines": [
                "{agent} stumbles!",
                "Technical difficulties for {agent}",
                "{agent} hits a snag",
            ],
            "narratives": [
                "{agent} has encountered an unexpected obstacle! Our recovery systems are kicking in.",
                "Oops! {agent} hit a snag - but fear not, we've got backup plans!",
                "{agent} is having a rough moment. Activating contingency protocols...",
            ],
            "mood": "tense",
        },
        "agent_recovered": {
            "headlines": [
                "{agent} bounces back!",
                "Recovery complete for {agent}!",
                "{agent} is back in action!",
            ],
            "narratives": [
                "{agent} bounces back! The show goes on!",
                "Recovery complete! {agent} rejoins the fray with renewed vigor.",
                "Like a phoenix from the ashes, {agent} has returned!",
                "{agent} has overcome the challenge. Full steam ahead!",
            ],
            "mood": "triumphant",
        },
        "agent_completed": {
            "headlines": [
                "{agent} delivers!",
                "Response from {agent}",
                "{agent} weighs in",
            ],
            "narratives": [
                "{agent} has spoken! Another quality contribution to the debate.",
                "And there we have it - {agent}'s perspective is now on the table.",
                "{agent} completes their turn. What will the other agents think?",
            ],
            "mood": "neutral",
        },
        "circuit_opened": {
            "headlines": [
                "{agent} takes a mandatory break",
                "Circuit breaker activated for {agent}",
                "{agent} benched temporarily",
            ],
            "narratives": [
                "{agent} takes a mandatory rest - circuit breaker activated! Too many hiccups in a row.",
                "The safety systems have kicked in - {agent} needs to cool off before returning.",
                "Circuit breaker open! {agent} will sit this one out while we stabilize.",
            ],
            "mood": "cautionary",
        },
        "circuit_closed": {
            "headlines": [
                "{agent} ready to return!",
                "Circuit restored for {agent}",
                "{agent} back online",
            ],
            "narratives": [
                "{agent} is ready to rejoin! Circuit breaker has closed after successful recovery.",
                "All systems green - {agent} is cleared for action!",
                "Welcome back, {agent}! The circuit has been restored.",
            ],
            "mood": "triumphant",
        },
        "fallback_used": {
            "headlines": [
                "Plan B activated!",
                "Backup response engaged",
                "Alternative path taken",
            ],
            "narratives": [
                "Plan B activated! Our backup response keeps the debate moving forward.",
                "Switching to contingency mode - the show must go on!",
                "Engaging fallback protocols. We won't let a timeout stop us!",
            ],
            "mood": "cautionary",
        },
        "consensus_reached": {
            "headlines": [
                "Consensus achieved!",
                "The agents agree!",
                "Meeting of minds",
            ],
            "narratives": [
                "Remarkable! The agents have found common ground. Consensus has been reached!",
                "After spirited debate, a unified vision emerges. This is consensus in action!",
                "The wisdom of the collective prevails - we have agreement!",
            ],
            "mood": "triumphant",
        },
        "debate_stalled": {
            "headlines": [
                "Debate in gridlock",
                "Impasse detected",
                "The agents are stuck",
            ],
            "narratives": [
                "We've hit a deadlock. The agents are circling but not converging.",
                "Stalemate alert! Different perspectives aren't finding middle ground.",
                "The debate has stalled - time for intervention or alternative approaches.",
            ],
            "mood": "cautionary",
        },
    }

    def __init__(
        self,
        broadcast_callback: Optional[Callable[[dict], Any]] = None,
    ):
        """
        Initialize the narrator.

        Args:
            broadcast_callback: Optional callback for broadcasting narratives
        """
        self.broadcast_callback = broadcast_callback
        self.recent_narratives: list[RecoveryNarrative] = []
        self.used_templates: dict[str, set] = {}  # Track used templates to avoid repetition

    def narrate(
        self,
        event_type: str,
        agent: str = "System",
        details: Optional[dict] = None,
    ) -> RecoveryNarrative:
        """
        Generate a narrative for an event.

        Args:
            event_type: Type of event (agent_timeout, agent_recovered, etc.)
            agent: Agent name involved
            details: Optional additional details

        Returns:
            RecoveryNarrative with headline and narrative
        """
        templates = self.TEMPLATES.get(event_type, self.TEMPLATES.get("agent_completed"))

        if not templates:
            # Fallback for unknown events
            return RecoveryNarrative(
                event_type=event_type,
                agent=agent,
                headline=f"Event: {event_type}",
                narrative=f"An event occurred involving {agent}.",
                mood="neutral",
            )

        # Pick templates (avoiding recent repeats)
        headline = self._pick_template(event_type, "headlines", templates["headlines"], agent)
        narrative = self._pick_template(event_type, "narratives", templates["narratives"], agent)
        mood = templates.get("mood", "neutral")

        result = RecoveryNarrative(
            event_type=event_type,
            agent=agent,
            headline=headline.format(agent=agent),
            narrative=narrative.format(agent=agent, **(details or {})),
            mood=mood,
        )

        # Track this narrative
        self.recent_narratives.append(result)
        if len(self.recent_narratives) > 100:
            self.recent_narratives = self.recent_narratives[-50:]

        return result

    def _pick_template(
        self,
        event_type: str,
        template_type: str,
        options: list[str],
        agent: str,
    ) -> str:
        """Pick a template, avoiding recent repeats."""
        key = f"{event_type}:{template_type}"

        if key not in self.used_templates:
            self.used_templates[key] = set()

        # Filter out recently used
        unused = [t for t in options if t not in self.used_templates[key]]

        if not unused:
            # Reset if all used
            self.used_templates[key] = set()
            unused = options

        # Pick and track
        choice = random.choice(unused)
        self.used_templates[key].add(choice)

        return choice

    def handle_health_event(self, event: dict) -> Optional[RecoveryNarrative]:
        """
        Handle a health event from the immune system.

        Args:
            event: Health event dict with 'type' and 'data'

        Returns:
            Generated narrative if applicable
        """
        if event.get("type") != "health_event":
            return None

        data = event.get("data", {})
        event_type = data.get("event_type", "")
        component = data.get("component", "System")

        # Skip non-narrative events
        if event_type not in self.TEMPLATES:
            return None

        narrative = self.narrate(event_type, component, data.get("details"))

        # Broadcast if callback set
        if self.broadcast_callback:
            try:
                self.broadcast_callback(
                    {
                        "type": "recovery_narrative",
                        "data": narrative.to_dict(),
                    }
                )
            except Exception as e:
                logger.warning(f"narrator_broadcast_failed error={e}")

        logger.debug(f"narrator_generated event={event_type} agent={component}")
        return narrative

    def get_recent_narratives(self, limit: int = 10) -> list[dict]:
        """Get recent narratives."""
        return [n.to_dict() for n in self.recent_narratives[-limit:]]

    def get_mood_summary(self) -> dict:
        """Get summary of recent mood distribution."""
        if not self.recent_narratives:
            return {"mood": "neutral", "distribution": {}}

        moods = [n.mood for n in self.recent_narratives[-20:]]
        distribution = {}
        for mood in moods:
            distribution[mood] = distribution.get(mood, 0) + 1

        # Determine overall mood
        overall = max(distribution.items(), key=lambda x: x[1])[0] if distribution else "neutral"

        return {
            "mood": overall,
            "distribution": distribution,
        }


# Global narrator instance
_narrator: Optional[RecoveryNarrator] = None


def get_narrator() -> RecoveryNarrator:
    """Get the global narrator instance."""
    global _narrator
    if _narrator is None:
        _narrator = RecoveryNarrator()
    return _narrator


def reset_narrator() -> None:
    """Reset the global narrator (for testing)."""
    global _narrator
    _narrator = None


def setup_narrator_with_immune_system() -> RecoveryNarrator:
    """Set up narrator integrated with the immune system."""
    narrator = get_narrator()

    try:
        from aragora.debate.immune_system import get_immune_system

        immune = get_immune_system()

        # Chain the callbacks if there's an existing one
        existing_callback = immune.broadcast_callback

        def combined_callback(event: dict) -> None:
            # Call existing callback
            if existing_callback:
                existing_callback(event)
            # Generate narrative
            narrator.handle_health_event(event)

        immune.set_broadcast_callback(combined_callback)
        logger.info("narrator_integrated_with_immune_system")

    except ImportError:
        logger.warning("narrator_immune_integration_failed immune_system_not_available")

    return narrator
