"""
Event bridge for coordinating spectator and WebSocket event emission.

Extracts event emission logic from Arena orchestrator for better modularity.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventEmitterBridge:
    """
    Bridge between SpectatorStream and WebSocket event emission.

    Coordinates event emission to:
    - SpectatorStream (console/file output)
    - SyncEventEmitter (WebSocket clients)
    - ArgumentCartographer (graph visualization)
    """

    # Map spectator event types to StreamEventType
    EVENT_TYPE_MAPPING = {
        "debate_start": "DEBATE_START",
        "debate_end": "DEBATE_END",
        "round": "ROUND_START",
        "round_start": "ROUND_START",
        "propose": "AGENT_MESSAGE",
        "proposal": "AGENT_MESSAGE",
        "critique": "CRITIQUE",
        "vote": "VOTE",
        "consensus": "CONSENSUS",
        "convergence": "CONSENSUS",
        "judge": "AGENT_MESSAGE",
        "memory_recall": "MEMORY_RECALL",
        "audience_drain": "AUDIENCE_DRAIN",
        "audience_summary": "AUDIENCE_SUMMARY",
        "insight_extracted": "INSIGHT_EXTRACTED",
        "token_start": "TOKEN_START",
        "token_delta": "TOKEN_DELTA",
        "token_end": "TOKEN_END",
        # New event mappings for feedback loop events
        "claim_verification": "CLAIM_VERIFICATION_RESULT",
        "memory_tier_promotion": "MEMORY_TIER_PROMOTION",
        "memory_tier_demotion": "MEMORY_TIER_DEMOTION",
        "agent_elo_updated": "AGENT_ELO_UPDATED",
    }

    def __init__(
        self,
        spectator: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        cartographer: Optional[Any] = None,
        loop_id: str = "",
    ):
        """
        Initialize the event bridge.

        Args:
            spectator: SpectatorStream for console/file output
            event_emitter: SyncEventEmitter for WebSocket clients
            cartographer: ArgumentCartographer for graph updates
            loop_id: Debate/loop identifier
        """
        self.spectator = spectator
        self.event_emitter = event_emitter
        self.cartographer = cartographer
        self.loop_id = loop_id

    def notify(self, event_type: str, **kwargs) -> None:
        """
        Emit event to all registered listeners.

        Emits to SpectatorStream, WebSocket, and Cartographer.

        Args:
            event_type: Type of event (e.g., "proposal", "critique", "vote")
            **kwargs: Event data (agent, details, round_number, etc.)
        """
        # Emit to spectator (console/file) - only pass supported params
        if self.spectator:
            spectator_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ("agent", "details", "metric", "round_number")
            }
            self.spectator.emit(event_type, **spectator_kwargs)

        # Emit to WebSocket clients
        if self.event_emitter:
            self._emit_to_websocket(event_type, **kwargs)

    def _emit_to_websocket(self, event_type: str, **kwargs) -> None:
        """Convert spectator event to StreamEvent and emit to WebSocket."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            stream_type_name = self.EVENT_TYPE_MAPPING.get(event_type)
            if not stream_type_name:
                return  # Skip unmapped event types

            stream_type = getattr(StreamEventType, stream_type_name, None)
            if not stream_type:
                return

            stream_event = StreamEvent(
                type=stream_type,
                data={
                    "details": kwargs.get("details", ""),
                    "metric": kwargs.get("metric"),
                    "event_source": "spectator",
                },
                round=kwargs.get("round_number", 0),
                agent=kwargs.get("agent", ""),
                loop_id=self.loop_id,
            )
            self.event_emitter.emit(stream_event)
        except (TypeError, ValueError) as e:
            # Serialization issues with event data
            logger.warning(f"Event emission serialization error (non-fatal): {e}")
        except (ConnectionError, RuntimeError) as e:
            # Network/event system issues
            logger.warning(f"Event emission connection error (non-fatal): {e}")
        except Exception as e:
            # Unexpected - log with type info
            logger.warning(f"Event emission unexpected error (non-fatal): {type(e).__name__}: {e}")

        # Update cartographer with this event
        self._update_cartographer(event_type, **kwargs)

    def _update_cartographer(self, event_type: str, **kwargs) -> None:
        """Update the ArgumentCartographer graph with debate events."""
        if not self.cartographer:
            return

        try:
            agent = kwargs.get("agent", "")
            details = kwargs.get("details", "")
            round_num = kwargs.get("round_number", 0)

            if event_type in ("propose", "proposal"):
                self.cartographer.update_from_message(
                    agent=agent,
                    content=details,
                    role="proposer",
                    round_num=round_num,
                )
            elif event_type == "critique":
                target = self._extract_critique_target(details)
                severity = kwargs.get("metric", 0.5)
                self.cartographer.update_from_critique(
                    critic_agent=agent,
                    target_agent=target,
                    severity=severity if isinstance(severity, (int, float)) else 0.5,
                    round_num=round_num,
                    critique_text=details,
                )
            elif event_type == "vote":
                vote_value = details.split(":")[-1].strip() if ":" in details else details
                self.cartographer.update_from_vote(
                    agent=agent,
                    vote_value=vote_value,
                    round_num=round_num,
                )
            elif event_type == "consensus":
                result = details.split(":")[-1].strip() if ":" in details else details
                self.cartographer.update_from_consensus(
                    result=result,
                    round_num=round_num,
                )
        except Exception as e:
            logger.warning(f"Cartographer error (non-fatal): {e}")

    @staticmethod
    def _extract_critique_target(details: str) -> str:
        """Extract target agent from critique details string."""
        if "Critiqued " in details:
            return details.split("Critiqued ")[1].split(":")[0]
        return ""

    def emit_moment(self, moment: Any) -> None:
        """Emit a significant moment event to WebSocket clients."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.MOMENT_DETECTED,
                    data=moment.to_dict(),
                    debate_id=self.loop_id or "unknown",
                )
            )
            logger.debug("Emitted moment event: %s for %s", moment.moment_type, moment.agent_name)
        except Exception as e:
            logger.warning("Failed to emit moment event: %s", e)
