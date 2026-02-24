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
from typing import TYPE_CHECKING, Any

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
        event_bus: EventBus | None = None,
        event_bridge: EventEmitterBridge | None = None,
        hooks: dict | None = None,
        persona_manager: PersonaManager | None = None,
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

    def notify_spectator(self, event_type: str, **kwargs: Any) -> None:
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
            logger.debug("health_broadcast_failed error=%s", e)

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

    def get_agent_role(self, agent: Agent, role_assignments: dict) -> str:
        """Get the role string for an agent from role assignments.

        Args:
            agent: The agent
            role_assignments: Dict mapping agent names to role data

        Returns:
            Role string (defaults to "proposer")
        """
        role_data = role_assignments.get(agent.name, {})
        return str(role_data.get("role", "proposer"))

    def get_agent_stance(self, agent: Agent, role_assignments: dict) -> str:
        """Get the stance string for an agent from role assignments.

        Args:
            agent: The agent
            role_assignments: Dict mapping agent names to role data

        Returns:
            Stance string (defaults to "neutral")
        """
        role_data = role_assignments.get(agent.name, {})
        return role_data.get("stance", "neutral")

    def get_agent_description(self, agent: Agent) -> str:
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

    def build_agent_preview(self, agent: Agent, role_assignments: dict) -> dict:
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

    def emit_agent_preview(self, agents: list[Agent], role_assignments: dict) -> None:
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
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError) as e:
            logger.debug("Agent preview emission failed: %s", e)

    # === Feature Integration Events ===
    # These events connect backend subsystems to frontend panels

    def emit_calibration_update(
        self,
        agent_name: str,
        brier_score: float,
        prediction_count: int,
        accuracy: float,
    ) -> None:
        """Emit calibration update event when an agent's calibration changes.

        Args:
            agent_name: Name of the agent
            brier_score: Updated Brier score (0-1, lower is better)
            prediction_count: Total predictions made
            accuracy: Prediction accuracy percentage
        """
        self.notify_spectator(
            "calibration_update",
            agent=agent_name,
            brier_score=brier_score,
            prediction_count=prediction_count,
            accuracy=accuracy,
        )

    def emit_evidence_found(
        self,
        claim: str,
        evidence_type: str,
        source: str,
        confidence: float,
        excerpt: str = "",
    ) -> None:
        """Emit evidence found event when supporting evidence is collected.

        Args:
            claim: The claim being supported
            evidence_type: Type of evidence (citation, fact, quote, etc.)
            source: Source of the evidence
            confidence: Confidence score (0-1)
            excerpt: Relevant excerpt from the evidence
        """
        self.notify_spectator(
            "evidence_found",
            claim=claim,
            evidence_type=evidence_type,
            source=source,
            confidence=confidence,
            excerpt=excerpt[:500] if excerpt else "",
        )

    def emit_trait_emerged(
        self,
        agent_name: str,
        trait_name: str,
        trait_description: str,
        emergence_round: int,
    ) -> None:
        """Emit trait emerged event when PersonaLaboratory detects a new trait.

        Args:
            agent_name: Name of the agent
            trait_name: Name of the emerged trait
            trait_description: Description of the trait
            emergence_round: Round in which trait emerged
        """
        self.notify_spectator(
            "trait_emerged",
            agent=agent_name,
            trait_name=trait_name,
            trait_description=trait_description,
            emergence_round=emergence_round,
        )

    def emit_risk_warning(
        self,
        risk_type: str,
        severity: str,
        description: str,
        affected_claims: list[str] | None = None,
    ) -> None:
        """Emit risk warning event when domain risk is identified.

        Args:
            risk_type: Type of risk (factual, logical, ethical, etc.)
            severity: Severity level (low, medium, high, critical)
            description: Description of the risk
            affected_claims: List of claims affected by this risk
        """
        self.notify_spectator(
            "risk_warning",
            risk_type=risk_type,
            severity=severity,
            description=description,
            affected_claims=affected_claims or [],
        )

    def emit_rhetorical_observation(
        self,
        agent_name: str,
        technique: str,
        description: str,
        quote: str = "",
        effectiveness: float = 0.5,
    ) -> None:
        """Emit rhetorical observation when a rhetorical pattern is detected.

        Args:
            agent_name: Agent using the technique
            technique: Name of the rhetorical technique
            description: Explanation of how it was used
            quote: Example quote demonstrating the technique
            effectiveness: Estimated effectiveness (0-1)
        """
        self.notify_spectator(
            "rhetorical_observation",
            agent=agent_name,
            technique=technique,
            description=description,
            quote=quote[:200] if quote else "",
            effectiveness=effectiveness,
        )

    def emit_hollow_consensus(
        self,
        confidence: float,
        indicators: list[str],
        recommendation: str,
    ) -> None:
        """Emit hollow consensus warning from Trickster.

        Args:
            confidence: Confidence that consensus is hollow (0-1)
            indicators: List of indicators suggesting hollow consensus
            recommendation: Suggested action
        """
        self.notify_spectator(
            "hollow_consensus",
            confidence=confidence,
            indicators=indicators,
            recommendation=recommendation,
        )

    def emit_trickster_intervention(
        self,
        intervention_type: str,
        challenge: str,
        target_claim: str = "",
    ) -> None:
        """Emit trickster intervention when Trickster injects a challenge.

        Args:
            intervention_type: Type of intervention (devils_advocate, contrarian, etc.)
            challenge: The challenge or question injected
            target_claim: The claim being challenged
        """
        self.notify_spectator(
            "trickster_intervention",
            intervention_type=intervention_type,
            challenge=challenge,
            target_claim=target_claim,
        )

    def emit_agent_message(
        self,
        agent_name: str,
        content: str,
        role: str = "proposer",
        round_num: int = 0,
        enable_tts: bool = True,
    ) -> None:
        """Emit agent message event for UI and TTS integration.

        This event is consumed by:
        - WebSocket clients for live updates
        - Voice sessions for TTS synthesis (if enabled)
        - Chat handlers (Telegram, Discord, etc.) for relaying responses

        Args:
            agent_name: Name of the agent producing the message
            content: The message content
            role: Agent's role (proposer, critic, judge, etc.)
            round_num: Current debate round
            enable_tts: Whether to trigger TTS synthesis for voice sessions
        """
        self.notify_spectator(
            "agent_message",
            agent=agent_name,
            content=content,
            role=role,
            round_num=round_num,
            enable_tts=enable_tts,
        )

    def emit_phase_start(self, phase_name: str, round_number: int) -> None:
        """Emit phase start event.

        Args:
            phase_name: Name of the phase starting
            round_number: Current debate round
        """
        self.notify_spectator(
            "phase_start",
            phase=phase_name,
            round=round_number,
        )

    def emit_phase_end(
        self,
        phase_name: str,
        round_number: int,
        duration_ms: int,
        success: bool = True,
    ) -> None:
        """Emit phase end event.

        Args:
            phase_name: Name of the phase ending
            round_number: Current debate round
            duration_ms: Duration of the phase in milliseconds
            success: Whether the phase completed successfully
        """
        self.notify_spectator(
            "phase_end",
            phase=phase_name,
            round=round_number,
            duration_ms=duration_ms,
            success=success,
        )

    def emit_round_start(self, round_number: int, total_rounds: int) -> None:
        """Emit round start event.

        Args:
            round_number: Current round number
            total_rounds: Total number of rounds
        """
        self.notify_spectator(
            "round_start",
            round=round_number,
            total_rounds=total_rounds,
        )

    def emit_round_end(
        self,
        round_number: int,
        total_rounds: int,
        duration_ms: int,
    ) -> None:
        """Emit round end event.

        Args:
            round_number: Current round number
            total_rounds: Total number of rounds
            duration_ms: Duration of the round in milliseconds
        """
        self.notify_spectator(
            "round_end",
            round=round_number,
            total_rounds=total_rounds,
            duration_ms=duration_ms,
        )

    def emit_agent_thinking(
        self,
        agent_name: str,
        step: str,
        phase: str = "reasoning",
        round_num: int = 0,
    ) -> None:
        """Emit agent_thinking event when an agent starts processing.

        Signals to the UI that an agent is actively formulating a response.

        Args:
            agent_name: Name of the agent
            step: Description of current thinking step
            phase: Phase of reasoning (proposal, critique, revision, etc.)
            round_num: Current debate round
        """
        self.notify_spectator(
            "agent_thinking",
            agent=agent_name,
            step=step[:500],
            phase=phase,
            round_num=round_num,
        )

    def emit_agent_reasoning(
        self,
        agent_name: str,
        reasoning_chunk: str,
        chain_position: int = 0,
        total_steps: int = 0,
        round_num: int = 0,
    ) -> None:
        """Emit agent_reasoning event to stream partial chain-of-thought.

        Allows UI to display incremental reasoning as it generates.

        Args:
            agent_name: Name of the agent
            reasoning_chunk: Partial reasoning text
            chain_position: Position in the reasoning chain (0-indexed)
            total_steps: Estimated total reasoning steps (0 if unknown)
            round_num: Current debate round
        """
        self.notify_spectator(
            "agent_reasoning",
            agent=agent_name,
            reasoning_chunk=reasoning_chunk[:1000],
            chain_position=chain_position,
            total_steps=total_steps,
            round_num=round_num,
        )

    def emit_argument_strength(
        self,
        agent_name: str,
        argument_summary: str,
        strength_score: float,
        factors: dict | None = None,
        round_num: int = 0,
    ) -> None:
        """Emit argument_strength event with real-time quality scores.

        Args:
            agent_name: Name of the agent whose argument is scored
            argument_summary: Brief summary of the argument
            strength_score: Quality score (0.0-1.0)
            factors: Optional dict of contributing factors (e.g., evidence, logic, novelty)
            round_num: Current debate round
        """
        self.notify_spectator(
            "argument_strength",
            agent=agent_name,
            argument_summary=argument_summary[:300],
            strength_score=max(0.0, min(1.0, strength_score)),
            factors=factors or {},
            round_num=round_num,
        )

    def emit_crux_identified(
        self,
        crux_description: str,
        agents_disagreeing: list[str],
        positions: dict[str, str] | None = None,
        severity: float = 0.5,
        round_num: int = 0,
    ) -> None:
        """Emit crux_identified event when agents identify a key disagreement.

        Args:
            crux_description: Description of the crux/key disagreement
            agents_disagreeing: Names of agents on opposing sides
            positions: Optional dict of agent_name -> position summary
            severity: How significant the disagreement is (0.0-1.0)
            round_num: Current debate round
        """
        self.notify_spectator(
            "crux_identified",
            crux_description=crux_description[:500],
            agents_disagreeing=agents_disagreeing,
            positions=positions or {},
            severity=max(0.0, min(1.0, severity)),
            round_num=round_num,
        )

    def emit_intervention_window(
        self,
        debate_id: str,
        round_num: int,
        window_type: str = "between_rounds",
        expires_in_seconds: float = 30.0,
        context_summary: str = "",
    ) -> None:
        """Emit intervention_window event signaling the user can interject.

        Args:
            debate_id: ID of the current debate
            round_num: Current debate round
            window_type: Type of intervention window (between_rounds, post_critique, etc.)
            expires_in_seconds: How long the window stays open
            context_summary: Brief summary of current debate state
        """
        self.notify_spectator(
            "intervention_window",
            debate_id=debate_id,
            round_num=round_num,
            window_type=window_type,
            expires_in_seconds=expires_in_seconds,
            context_summary=context_summary[:300],
        )

    def emit_intervention_applied(
        self,
        intervention_id: str,
        intervention_type: str,
        content_summary: str,
        applied_at_round: int,
    ) -> None:
        """Emit intervention_applied event when a user intervention takes effect.

        Args:
            intervention_id: Unique ID of the intervention
            intervention_type: Type (redirect, constraint, challenge, evidence_request)
            content_summary: Brief summary of the intervention content
            applied_at_round: Round at which the intervention was applied
        """
        self.notify_spectator(
            "intervention_applied",
            intervention_id=intervention_id,
            intervention_type=intervention_type,
            content_summary=content_summary[:300],
            applied_at_round=applied_at_round,
        )


__all__ = ["EventEmitter"]
