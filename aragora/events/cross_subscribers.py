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
from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class SubscriberStats:
    """Statistics for a cross-subsystem subscriber."""

    name: str
    events_processed: int = 0
    events_failed: int = 0
    events_skipped: int = 0  # Skipped due to sampling/filtering
    last_event_time: Optional[datetime] = None
    enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = 100% of events, 0.1 = 10% sampling
    # Latency metrics (in milliseconds)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.events_processed == 0:
            return 0.0
        return self.total_latency_ms / self.events_processed


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

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        """Initialize the cross-subscriber manager.

        Args:
            failure_threshold: Consecutive failures before circuit opens (default: 5)
            cooldown_seconds: Seconds before attempting recovery (default: 60)
        """
        self._subscribers: dict[StreamEventType, list[tuple[str, Callable[[StreamEvent], None]]]] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self._filters: dict[str, Callable[[StreamEvent], bool]] = {}
        self._connected = False

        # Circuit breaker for handler failure protection
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )

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

        # Mound structure → Memory/Debate sync
        self.register(
            "mound_to_memory",
            StreamEventType.MOUND_UPDATED,
            self._handle_mound_to_memory,
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
        """Dispatch event to registered subscribers with sampling and circuit breaker support."""
        import random
        import time

        handlers = self._subscribers.get(event.type, [])

        for name, handler in handlers:
            # Check if handler is enabled
            if name in self._stats and not self._stats[name].enabled:
                continue

            # Check circuit breaker - skip if handler circuit is open
            if not self._circuit_breaker.is_available(name):
                if name in self._stats:
                    self._stats[name].events_skipped += 1
                logger.debug(f"Circuit open for handler {name}, skipping event")
                continue

            # Apply sampling - skip based on sample_rate
            if name in self._stats:
                sample_rate = self._stats[name].sample_rate
                if sample_rate < 1.0 and random.random() > sample_rate:
                    self._stats[name].events_skipped += 1
                    continue

            # Apply custom filter if set
            if name in self._filters:
                filter_fn = self._filters[name]
                try:
                    if not filter_fn(event):
                        self._stats[name].events_skipped += 1
                        continue
                except Exception as e:
                    logger.debug(f"Filter error for {name}: {e}")

            start_time = time.perf_counter()
            try:
                handler(event)

                # Record success with circuit breaker
                self._circuit_breaker.record_success(name)

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Update stats with latency
                if name in self._stats:
                    stats = self._stats[name]
                    stats.events_processed += 1
                    stats.last_event_time = datetime.now()
                    stats.total_latency_ms += latency_ms
                    stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
                    stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)

            except Exception as e:
                # Record failure with circuit breaker
                self._circuit_breaker.record_failure(name)

                logger.error(f"Cross-subscriber error in {name}: {e}")
                if name in self._stats:
                    self._stats[name].events_failed += 1

                # Check if circuit just opened
                if not self._circuit_breaker.is_available(name):
                    logger.warning(f"Circuit breaker opened for handler {name} after repeated failures")

    async def _dispatch_event_async(self, event: StreamEvent) -> None:
        """Dispatch event to registered subscribers asynchronously.

        Runs all handlers concurrently using asyncio.gather for better
        performance when handlers involve I/O operations.
        """
        import asyncio
        import time

        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            return

        async def run_handler(name: str, handler: Callable) -> None:
            """Run a single handler and track stats."""
            start_time = time.perf_counter()
            try:
                # Check if handler is async
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handler in executor to not block
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, event)

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Update stats
                if name in self._stats:
                    stats = self._stats[name]
                    stats.events_processed += 1
                    stats.last_event_time = datetime.now()
                    stats.total_latency_ms += latency_ms
                    stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
                    stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)

            except Exception as e:
                logger.error(f"Cross-subscriber error in {name}: {e}")
                if name in self._stats:
                    self._stats[name].events_failed += 1

        # Run all handlers concurrently
        await asyncio.gather(
            *[run_handler(name, handler) for name, handler in handlers],
            return_exceptions=True,
        )

    def dispatch_async(self, event: StreamEvent) -> None:
        """Schedule async dispatch (fire-and-forget).

        Use this for non-blocking event dispatch when you don't need
        to wait for handler completion.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._dispatch_event_async(event))
        except RuntimeError:
            # No running loop - fall back to sync dispatch
            self._dispatch_event(event)

    # =========================================================================
    # Built-in Cross-Subsystem Handlers
    # =========================================================================

    def _handle_memory_to_rlm(self, event: StreamEvent) -> None:
        """
        Memory retrieval → RLM feedback.

        When memory is retrieved, inform RLM about retrieval patterns
        to optimize compression strategies. Tracks access patterns
        for adaptive compression.
        """
        data = event.data
        tier = data.get("tier", "unknown")
        hit = data.get("cache_hit", False)
        importance = data.get("importance", 0.5)

        # Track access pattern for RLM optimization
        logger.debug(f"Memory retrieval: tier={tier}, cache_hit={hit}")

        # Update RLM compression hints based on access patterns
        try:
            from aragora.rlm.compressor import get_compressor

            compressor = get_compressor()
            if compressor and hasattr(compressor, "record_access_pattern"):
                compressor.record_access_pattern(
                    tier=tier,
                    cache_hit=hit,
                    importance=importance,
                )
        except ImportError:
            pass  # RLM not available
        except Exception as e:
            logger.debug(f"RLM pattern recording failed: {e}")

    def _handle_elo_to_debate(self, event: StreamEvent) -> None:
        """
        ELO update → Debate team selection weights.

        When agent ELO changes, update team selection weights
        for future debates. Significant changes are logged.
        """
        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)
        debate_id = data.get("debate_id", "")

        # Log significant ELO changes
        if abs(delta) > 50:
            logger.info(
                f"Significant ELO change: {agent_name} -> {new_elo} "
                f"(Δ{delta:+.0f}) in debate {debate_id}"
            )

        # Update agent pool weights for future team selection
        try:
            from aragora.debate.agent_pool import get_agent_pool

            pool = get_agent_pool()
            if pool and hasattr(pool, "update_elo_weight"):
                pool.update_elo_weight(agent_name, new_elo)
        except ImportError:
            pass  # AgentPool not available
        except Exception as e:
            logger.debug(f"AgentPool weight update failed: {e}")

    def _handle_knowledge_to_memory(self, event: StreamEvent) -> None:
        """
        Knowledge indexed → Memory sync.

        When new knowledge is indexed, create corresponding
        memory entries for cross-referencing in debates.
        """
        data = event.data
        node_id = data.get("node_id", "")
        content = data.get("content", "")
        node_type = data.get("node_type", "fact")
        workspace_id = data.get("workspace_id", "default")

        logger.debug(f"Knowledge indexed: {node_type} {node_id}")

        # Create memory entry referencing knowledge node
        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory:
                # Store a reference to the knowledge node in memory
                memory_content = f"[Knowledge:{node_type}] {content[:500]}"
                metadata = {
                    "source": "knowledge_mound",
                    "node_id": node_id,
                    "node_type": node_type,
                    "workspace_id": workspace_id,
                }
                memory.store(
                    content=memory_content,
                    importance=0.6,  # Default importance for knowledge references
                    metadata=metadata,
                )
                logger.debug(f"Created memory reference for knowledge node {node_id}")
        except ImportError:
            pass  # ContinuumMemory not available
        except Exception as e:
            logger.debug(f"Memory sync for knowledge failed: {e}")

    def _handle_calibration_to_agent(self, event: StreamEvent) -> None:
        """
        Calibration update → Agent confidence weights.

        When calibration data changes, update agent confidence
        weights for vote weighting and team selection.
        """
        data = event.data
        agent_name = data.get("agent", "")
        calibration_score = data.get("score", 0.5)
        brier_score = data.get("brier_score", None)
        prediction_count = data.get("prediction_count", 0)

        logger.debug(
            f"Calibration update: {agent_name} -> {calibration_score:.2f} "
            f"(predictions: {prediction_count})"
        )

        # Update agent pool with calibration data
        try:
            from aragora.debate.agent_pool import get_agent_pool

            pool = get_agent_pool()
            if pool and hasattr(pool, "update_calibration"):
                pool.update_calibration(
                    agent_name=agent_name,
                    score=calibration_score,
                    brier_score=brier_score,
                )
        except ImportError:
            pass  # AgentPool not available
        except Exception as e:
            logger.debug(f"AgentPool calibration update failed: {e}")

    def _handle_evidence_to_insight(self, event: StreamEvent) -> None:
        """
        Evidence found → Insight extraction.

        When new evidence is collected, attempt to extract
        insights that can be stored in memory for future debates.
        """
        data = event.data
        evidence_id = data.get("evidence_id", "")
        source = data.get("source", "")
        content = data.get("content", "")
        claim = data.get("claim", "")
        confidence = data.get("confidence", 0.5)

        logger.debug(f"Evidence collected: {evidence_id} from {source}")

        # Skip if no meaningful content
        if not content or len(content) < 50:
            return

        # Store evidence-backed insight in memory
        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory and confidence >= 0.7:  # Only store high-confidence evidence
                insight_content = (
                    f"[Evidence from {source}] "
                    f"Claim: {claim[:200] if claim else 'N/A'} | "
                    f"Evidence: {content[:300]}"
                )
                metadata = {
                    "source": source,
                    "evidence_id": evidence_id,
                    "confidence": confidence,
                    "type": "evidence_insight",
                }
                memory.store(
                    content=insight_content,
                    importance=confidence,
                    metadata=metadata,
                )
                logger.debug(f"Stored evidence insight from {source}")
        except ImportError:
            pass  # ContinuumMemory not available
        except Exception as e:
            logger.debug(f"Evidence insight storage failed: {e}")

    def _handle_mound_to_memory(self, event: StreamEvent) -> None:
        """
        Mound structure update → Memory/Debate sync.

        When the Knowledge Mound structure changes significantly,
        notify memory and debate systems to refresh their context.
        """
        data = event.data
        update_type = data.get("update_type", "unknown")
        workspace_id = data.get("workspace_id", "")

        logger.debug(f"Mound updated: type={update_type}, workspace={workspace_id}")

        # Handle culture pattern updates
        if update_type == "culture_patterns":
            patterns_count = data.get("patterns_count", 0)
            debate_id = data.get("debate_id", "")
            logger.info(
                f"Culture patterns updated: {patterns_count} patterns "
                f"from debate {debate_id}"
            )

        # Handle node deletions
        elif update_type == "node_deleted":
            node_id = data.get("node_id", "")
            archived = data.get("archived", False)
            logger.debug(
                f"Knowledge node removed: {node_id} (archived={archived})"
            )

            # Clear any cached references to this node
            try:
                from aragora.memory import get_continuum_memory

                memory = get_continuum_memory()
                if memory and hasattr(memory, "invalidate_reference"):
                    memory.invalidate_reference(node_id)
            except (ImportError, AttributeError):
                pass

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all subscribers including latency, sampling, and circuit breaker metrics."""
        result = {}
        for name, stats in self._stats.items():
            # Get circuit breaker status for this handler
            cb_state = self._circuit_breaker.get_state(name) if hasattr(self._circuit_breaker, 'get_state') else "unknown"
            cb_available = self._circuit_breaker.is_available(name)

            result[name] = {
                "events_processed": stats.events_processed,
                "events_failed": stats.events_failed,
                "events_skipped": stats.events_skipped,
                "last_event": stats.last_event_time.isoformat() if stats.last_event_time else None,
                "enabled": stats.enabled,
                "sample_rate": stats.sample_rate,
                "has_filter": name in self._filters,
                "latency_ms": {
                    "avg": round(stats.avg_latency_ms, 3),
                    "min": round(stats.min_latency_ms, 3) if stats.min_latency_ms != float("inf") else None,
                    "max": round(stats.max_latency_ms, 3),
                    "total": round(stats.total_latency_ms, 3),
                },
                "circuit_breaker": {
                    "available": cb_available,
                    "state": str(cb_state),
                },
            }
        return result

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
        """Reset all subscriber statistics including latency."""
        for stats in self._stats.values():
            stats.events_processed = 0
            stats.events_failed = 0
            stats.events_skipped = 0
            stats.last_event_time = None
            stats.total_latency_ms = 0.0
            stats.min_latency_ms = float("inf")
            stats.max_latency_ms = 0.0

    def reset_circuit_breaker(self, name: str) -> bool:
        """Reset circuit breaker for a specific handler.

        Args:
            name: Handler name to reset

        Returns:
            True if reset successful, False if handler not found
        """
        if name not in self._stats:
            return False
        self._circuit_breaker.reset(name)
        logger.info(f"Reset circuit breaker for handler '{name}'")
        return True

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for name in self._stats:
            self._circuit_breaker.reset(name)
        logger.info("Reset all circuit breakers")

    def set_sample_rate(self, name: str, rate: float) -> bool:
        """Set sampling rate for a subscriber.

        Args:
            name: Subscriber name
            rate: Sample rate (0.0 to 1.0). 1.0 = process all events,
                  0.1 = process ~10% of events randomly.

        Returns:
            True if subscriber found and updated, False otherwise.
        """
        if name not in self._stats:
            return False
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {rate}")
        self._stats[name].sample_rate = rate
        logger.info(f"Set sample rate for '{name}' to {rate:.0%}")
        return True

    def set_filter(
        self,
        name: str,
        filter_fn: Optional[Callable[[StreamEvent], bool]],
    ) -> bool:
        """Set a custom filter function for a subscriber.

        The filter is called before the handler. If it returns False,
        the event is skipped (counted as skipped, not processed).

        Args:
            name: Subscriber name
            filter_fn: Function that takes StreamEvent and returns bool.
                       Pass None to remove an existing filter.

        Returns:
            True if subscriber found and updated, False otherwise.
        """
        if name not in self._stats:
            return False

        if filter_fn is None:
            self._filters.pop(name, None)
            logger.info(f"Removed filter for '{name}'")
        else:
            self._filters[name] = filter_fn
            logger.info(f"Set filter for '{name}'")
        return True

    def get_filter(self, name: str) -> Optional[Callable[[StreamEvent], bool]]:
        """Get the filter function for a subscriber, if any."""
        return self._filters.get(name)


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
