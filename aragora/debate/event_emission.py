"""
Event emission and agent preview handling for Arena.

Extracted from Arena to reduce orchestrator size. Handles:
- Spectator notification
- Moment event emission
- Health event broadcasting
- Agent preview emission for UI feedback
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent
    from aragora.debate.event_bridge import EventEmitterBridge
    from aragora.debate.event_bus import EventBus

logger = logging.getLogger(__name__)


class EventEmitter:
    """Handles event emission for Arena debates.

    Extracted from Arena to centralize event-related operations:
    - Spectator/websocket notifications
    - Moment event emission
    - Health event broadcasting
    - Agent preview emission

    Usage:
        emitter = EventEmitter(
            event_bus=bus,
            event_bridge=bridge,
            hooks=hooks,
            persona_manager=personas,
        )
        emitter.notify_spectator("event_type", data=data)
        emitter.emit_agent_preview(agents, role_assignments)
    """

    def __init__(
        self,
        event_bus: Optional["EventBus"] = None,
        event_bridge: Optional["EventEmitterBridge"] = None,
        hooks: Optional[dict] = None,
        persona_manager: Optional["PersonaManager"] = None,
    ) -> None:
        """Initialize event emitter.

        Args:
            event_bus: Optional EventBus for pub/sub events
            event_bridge: Optional bridge for spectator/websocket events
            hooks: Optional dict of event callbacks
            persona_manager: Optional manager for agent personas
        """
        self.event_bus = event_bus
        self.event_bridge = event_bridge
        self.hooks = hooks or {}
        self.persona_manager = persona_manager
        self._current_debate_id = ""

    def set_debate_id(self, debate_id: str) -> None:
        """Set the current debate ID for event context."""
        self._current_debate_id = debate_id

    def notify_spectator(self, event_type: str, **kwargs) -> None:
        """Emit event via EventBus or fallback to event_bridge.

        Args:
            event_type: Type of event to emit
            **kwargs: Event data
        """
        if self.event_bus:
            debate_id = kwargs.pop("debate_id", self._current_debate_id)
            self.event_bus.emit_sync(event_type, debate_id=debate_id, **kwargs)
        elif self.event_bridge:
            self.event_bridge.notify(event_type, **kwargs)

    def emit_moment(self, moment: Any) -> None:
        """Emit a moment event.

        Args:
            moment: Moment object with to_dict() method
        """
        if self.event_bus and hasattr(moment, "to_dict"):
            moment_data = moment.to_dict()
            self.event_bus.emit_sync(
                event_type="moment",
                debate_id=self._current_debate_id,
                moment_type=getattr(moment, "moment_type", "unknown"),
                agent=getattr(moment, "agent_name", None),
                **moment_data,
            )
        elif self.event_bridge:
            self.event_bridge.emit_moment(moment)

    def _extract_health_event_data(self, event: dict) -> dict:
        """Extract and filter health event data for broadcasting.

        Removes event_type and debate_id keys to avoid duplicate kwargs.
        """
        data = event.get("data", event)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in ("event_type", "debate_id")}
        return data if isinstance(data, dict) else {}

    def broadcast_health_event(self, event: dict) -> None:
        """Broadcast health events to WebSocket clients via EventBus.

        Args:
            event: Health event dictionary
        """
        try:
            data = self._extract_health_event_data(event)
            self._emit_health_event(data)
        except (KeyError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug(f"health_broadcast_failed error={e}")

    def _emit_health_event(self, data: dict) -> None:
        """Emit health event via EventBus or fallback to event_bridge."""
        if self.event_bus:
            self.event_bus.emit_sync(
                event_type="health_event",
                debate_id=self._current_debate_id,
                **data,
            )
        elif self.event_bridge:
            self.event_bridge.notify(event_type="health_event", **data)

    def should_emit_preview(self) -> bool:
        """Check if agent preview hook is registered."""
        return "on_agent_preview" in self.hooks

    def get_agent_role(self, agent: "Agent", role_assignments: dict) -> str:
        """Get the role string for an agent from role assignments.

        Args:
            agent: The agent
            role_assignments: Dict mapping agent names to role data

        Returns:
            Role string (defaults to "proposer")
        """
        role_data = role_assignments.get(agent.name, {})
        return str(role_data.get("role", "proposer"))

    def get_agent_stance(self, agent: "Agent", role_assignments: dict) -> str:
        """Get the stance string for an agent from role assignments.

        Args:
            agent: The agent
            role_assignments: Dict mapping agent names to role data

        Returns:
            Stance string (defaults to "neutral")
        """
        role_data = role_assignments.get(agent.name, {})
        return role_data.get("stance", "neutral")

    def get_agent_description(self, agent: "Agent") -> str:
        """Get the persona description for an agent.

        Args:
            agent: The agent

        Returns:
            Brief description or empty string
        """
        if not self.persona_manager:
            return ""
        persona = self.persona_manager.get_persona(agent.name)
        return getattr(persona, "brief_description", "")

    def build_agent_preview(self, agent: "Agent", role_assignments: dict) -> dict:
        """Build preview data for a single agent.

        Args:
            agent: The agent
            role_assignments: Dict mapping agent names to role data

        Returns:
            Dict with agent preview data
        """
        return {
            "name": agent.name,
            "role": self.get_agent_role(agent, role_assignments),
            "stance": self.get_agent_stance(agent, role_assignments),
            "description": self.get_agent_description(agent),
            "strengths": [],
        }

    def emit_agent_preview(self, agents: list["Agent"], role_assignments: dict) -> None:
        """Emit agent preview for quick UI feedback.

        Shows agent roles and stances while proposals are being generated.

        Args:
            agents: List of agents to preview
            role_assignments: Dict mapping agent names to role data
        """
        if not self.should_emit_preview():
            return
        try:
            previews = [self.build_agent_preview(a, role_assignments) for a in agents]
            self.hooks["on_agent_preview"](previews)
        except Exception as e:
            logger.debug(f"Agent preview emission failed: {e}")


__all__ = ["EventEmitter"]
