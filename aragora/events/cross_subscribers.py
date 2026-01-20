"""
Cross-Subsystem Event Subscribers.

Handles event propagation between subsystems to enable cross-pollination:
- Memory → RLM: Retrieval patterns inform compression strategies
- Agent ELO → Debate: Performance updates team selection weights
- Knowledge → Memory: Index updates sync to memory insights

This module bridges the event system with subsystem-specific actions,
enabling loose coupling while maintaining data flow.

Usage:
    from aragora.events.cross_subscribers import (
        CrossSubscriberManager,
        get_cross_subscriber_manager,
    )

    # Initialize and connect to event stream
    manager = get_cross_subscriber_manager()
    manager.connect(event_emitter)

    # Subscribers automatically process relevant events
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from aragora.events.types import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


@dataclass
class SubscriberStats:
    """Statistics for a cross-subsystem subscriber."""

    name: str
    events_processed: int = 0
    events_failed: int = 0
    last_event_time: Optional[datetime] = None
    enabled: bool = True


class CrossSubscriberManager:
    """
    Manages cross-subsystem event subscribers.

    Provides a central point for registering and managing subscribers
    that react to events from different subsystems.

    Example:
        manager = CrossSubscriberManager()

        # Register custom subscriber
        @manager.subscribe(StreamEventType.MEMORY_STORED)
        def on_memory_stored(event: StreamEvent):
            # Handle memory storage event
            pass

        # Connect to event stream
        manager.connect(event_emitter)
    """

    def __init__(self):
        """Initialize the cross-subscriber manager."""
        self._subscribers: dict[StreamEventType, list[tuple[str, Callable[[StreamEvent], None]]]] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self._connected = False

        # Register built-in cross-subsystem handlers
        self._register_builtin_subscribers()

    def _register_builtin_subscribers(self) -> None:
        """Register built-in cross-subsystem event handlers."""
        # Memory → RLM feedback
        self.register(
            "memory_to_rlm",
            StreamEventType.MEMORY_RETRIEVED,
            self._handle_memory_to_rlm,
        )

        # Agent ELO → Debate team selection
        self.register(
            "elo_to_debate",
            StreamEventType.AGENT_ELO_UPDATED,
            self._handle_elo_to_debate,
        )

        # Knowledge → Memory sync
        self.register(
            "knowledge_to_memory",
            StreamEventType.KNOWLEDGE_INDEXED,
            self._handle_knowledge_to_memory,
        )

        # Calibration → Agent weights
        self.register(
            "calibration_to_agent",
            StreamEventType.CALIBRATION_UPDATE,
            self._handle_calibration_to_agent,
        )

        # Evidence → Insight extraction
        self.register(
            "evidence_to_insight",
            StreamEventType.EVIDENCE_FOUND,
            self._handle_evidence_to_insight,
        )

        logger.debug("Registered built-in cross-subsystem subscribers")

    def register(
        self,
        name: str,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None],
    ) -> None:
        """
        Register a cross-subsystem subscriber.

        Args:
            name: Unique name for the subscriber
            event_type: Event type to subscribe to
            handler: Handler function called with StreamEvent
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append((name, handler))
        self._stats[name] = SubscriberStats(name=name)

        logger.debug(f"Registered subscriber '{name}' for {event_type.value}")

    def subscribe(
        self,
        event_type: StreamEventType,
    ) -> Callable[[Callable[[StreamEvent], None]], Callable[[StreamEvent], None]]:
        """
        Decorator for registering subscribers.

        Usage:
            @manager.subscribe(StreamEventType.MEMORY_STORED)
            def on_memory_stored(event):
                pass
        """
        def decorator(func: Callable[[StreamEvent], None]) -> Callable[[StreamEvent], None]:
            self.register(func.__name__, event_type, func)
            return func
        return decorator

    def connect(self, event_emitter) -> None:
        """
        Connect to an event emitter to receive events.

        Args:
            event_emitter: EventEmitter instance to subscribe to
        """
        if self._connected:
            logger.warning("CrossSubscriberManager already connected")
            return

        def on_event(event: StreamEvent) -> None:
            self._dispatch_event(event)

        event_emitter.subscribe(on_event)
        self._connected = True
        logger.info("CrossSubscriberManager connected to event stream")

    def _dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch event to registered subscribers."""
        handlers = self._subscribers.get(event.type, [])

        for name, handler in handlers:
            try:
                handler(event)

                # Update stats
                if name in self._stats:
                    self._stats[name].events_processed += 1
                    self._stats[name].last_event_time = datetime.now()

            except Exception as e:
                logger.error(f"Cross-subscriber error in {name}: {e}")
                if name in self._stats:
                    self._stats[name].events_failed += 1

    # =========================================================================
    # Built-in Cross-Subsystem Handlers
    # =========================================================================

    def _handle_memory_to_rlm(self, event: StreamEvent) -> None:
        """
        Memory retrieval → RLM feedback.

        When memory is retrieved, inform RLM about retrieval patterns
        to optimize compression strategies.
        """
        data = event.data
        tier = data.get("tier", "unknown")
        hit = data.get("cache_hit", False)

        # Log retrieval pattern for RLM analysis
        logger.debug(f"Memory retrieval: tier={tier}, cache_hit={hit}")

        # Future: Update RLM compression hints based on access patterns
        # from aragora.rlm import get_compressor
        # compressor = get_compressor()
        # compressor.record_access_pattern(tier, hit)

    def _handle_elo_to_debate(self, event: StreamEvent) -> None:
        """
        ELO update → Debate team selection weights.

        When agent ELO changes, update team selection weights
        for future debates.
        """
        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)

        if abs(delta) > 50:
            logger.info(f"Significant ELO change: {agent_name} -> {new_elo} (Δ{delta:+.0f})")

        # Future: Update TeamSelector weights
        # from aragora.debate.team_selector import get_team_selector
        # selector = get_team_selector()
        # selector.update_weight(agent_name, new_elo)

    def _handle_knowledge_to_memory(self, event: StreamEvent) -> None:
        """
        Knowledge indexed → Memory sync.

        When new knowledge is indexed, create corresponding
        memory entries for cross-referencing.
        """
        data = event.data
        node_id = data.get("node_id", "")
        content = data.get("content", "")[:200]
        node_type = data.get("node_type", "fact")

        logger.debug(f"Knowledge indexed: {node_type} {node_id}")

        # Future: Create memory entry referencing knowledge node
        # from aragora.memory import get_continuum_memory
        # memory = get_continuum_memory()
        # memory.store_reference(node_id, content)

    def _handle_calibration_to_agent(self, event: StreamEvent) -> None:
        """
        Calibration update → Agent confidence weights.

        When calibration data changes, update agent confidence
        weights for vote weighting.
        """
        data = event.data
        agent_name = data.get("agent", "")
        calibration_score = data.get("score", 0.5)

        logger.debug(f"Calibration update: {agent_name} -> {calibration_score:.2f}")

        # Future: Update vote weights based on calibration
        # from aragora.debate.voting_engine import get_vote_weighter
        # weighter = get_vote_weighter()
        # weighter.set_agent_calibration(agent_name, calibration_score)

    def _handle_evidence_to_insight(self, event: StreamEvent) -> None:
        """
        Evidence found → Insight extraction.

        When new evidence is collected, attempt to extract
        insights that can be stored in memory.
        """
        data = event.data
        evidence_id = data.get("evidence_id", "")
        source = data.get("source", "")

        logger.debug(f"Evidence collected: {evidence_id} from {source}")

        # Future: Extract and store insights
        # from aragora.insights import extract_insight
        # insight = extract_insight(data.get("content", ""))
        # if insight:
        #     memory.store_insight(insight)

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all subscribers."""
        return {
            name: {
                "events_processed": stats.events_processed,
                "events_failed": stats.events_failed,
                "last_event": stats.last_event_time.isoformat() if stats.last_event_time else None,
                "enabled": stats.enabled,
            }
            for name, stats in self._stats.items()
        }

    def enable_subscriber(self, name: str) -> bool:
        """Enable a subscriber by name."""
        if name in self._stats:
            self._stats[name].enabled = True
            return True
        return False

    def disable_subscriber(self, name: str) -> bool:
        """Disable a subscriber by name."""
        if name in self._stats:
            self._stats[name].enabled = False
            return True
        return False

    def reset_stats(self) -> None:
        """Reset all subscriber statistics."""
        for stats in self._stats.values():
            stats.events_processed = 0
            stats.events_failed = 0
            stats.last_event_time = None


# Global manager instance
_global_manager: Optional[CrossSubscriberManager] = None


def get_cross_subscriber_manager() -> CrossSubscriberManager:
    """Get or create the global cross-subscriber manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CrossSubscriberManager()
    return _global_manager


def reset_cross_subscriber_manager() -> None:
    """Reset the global manager (for testing)."""
    global _global_manager
    _global_manager = None


__all__ = [
    "CrossSubscriberManager",
    "SubscriberStats",
    "get_cross_subscriber_manager",
    "reset_cross_subscriber_manager",
]
