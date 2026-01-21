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
from datetime import datetime
from typing import Any, Callable, Optional

from aragora.events.subscribers.config import (
    RetryConfig,
    SubscriberStats,
    AsyncDispatchConfig,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.resilience import CircuitBreaker

# Import settings for feature flags
try:
    from aragora.config.settings import get_settings

    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

    def get_settings():
        return None


# Import metrics (optional - graceful fallback if not available)
try:
    from aragora.server.prometheus_cross_pollination import (
        record_event_dispatched,
        record_handler_call,
        set_circuit_breaker_state,
        # KM bidirectional flow metrics
        record_km_inbound_event,
        record_km_outbound_event,
        record_km_adapter_sync,
        record_km_staleness_check,
        update_km_nodes_by_source,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    # Stub functions when metrics not available
    def record_km_inbound_event(source: str, event_type: str) -> None:
        pass

    def record_km_outbound_event(target: str, event_type: str) -> None:
        pass

    def record_km_adapter_sync(
        adapter: str, direction: str, status: str, duration: float = None
    ) -> None:
        pass

    def record_km_staleness_check(workspace: str, status: str, stale_count: int = 0) -> None:
        pass

    def update_km_nodes_by_source(source: str, count: int) -> None:
        pass


# Import SLO metrics (optional - graceful fallback)
try:
    from aragora.observability.metrics.slo import check_and_record_slo

    SLO_METRICS_AVAILABLE = True
except ImportError:
    SLO_METRICS_AVAILABLE = False

    def check_and_record_slo(operation: str, latency_ms: float, percentile: str = "p99"):
        return True, f"SLO metrics not available for {operation}"


logger = logging.getLogger(__name__)


# Re-export dataclasses for backward compatibility
__all__ = [
    "RetryConfig",
    "SubscriberStats",
    "AsyncDispatchConfig",
    "CrossSubscriberManager",
    "get_cross_subscriber_manager",
    "reset_cross_subscriber_manager",
]


def _default_async_event_types() -> set:
    """Default event types for async dispatch."""
    return {
        StreamEventType.MEMORY_STORED,
        StreamEventType.MEMORY_RETRIEVED,
        StreamEventType.KNOWLEDGE_QUERIED,
        StreamEventType.RLM_COMPRESSION_COMPLETE,
    }


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

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        default_retry_config: Optional[RetryConfig] = None,
        async_config: Optional[AsyncDispatchConfig] = None,
    ):
        """Initialize the cross-subscriber manager.

        Args:
            failure_threshold: Consecutive failures before circuit opens (default: 5)
            cooldown_seconds: Seconds before attempting recovery (default: 60)
            default_retry_config: Default retry configuration for handlers (default: 3 retries)
            async_config: Configuration for async/batched event dispatch
        """
        self._subscribers: dict[
            StreamEventType, list[tuple[str, Callable[[StreamEvent], None]]]
        ] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self._filters: dict[str, Callable[[StreamEvent], bool]] = {}
        self._connected = False

        # Default retry configuration
        self._default_retry_config = default_retry_config or RetryConfig()

        # Cache settings reference for feature flags and batch config
        self._settings = get_settings() if SETTINGS_AVAILABLE else None

        # Async dispatch configuration (use settings if available)
        if async_config is not None:
            self._async_config = async_config
        else:
            self._async_config = self._create_async_config_from_settings()

        # Event batch queue for high-volume events
        self._event_batch: dict[StreamEventType, list[StreamEvent]] = {}
        self._batch_last_flush: float = 0.0

        # Circuit breaker for handler failure protection
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )

        # Register built-in cross-subsystem handlers
        self._register_builtin_subscribers()

    def _create_async_config_from_settings(self) -> AsyncDispatchConfig:
        """Create AsyncDispatchConfig from settings or use defaults.

        Returns:
            Configured AsyncDispatchConfig
        """
        if self._settings is None:
            return AsyncDispatchConfig()

        try:
            integration = self._settings.integration
            return AsyncDispatchConfig(
                batch_size=integration.event_batch_size,
                batch_timeout_seconds=integration.event_batch_timeout_seconds,
                enable_batching=integration.event_batching_enabled,
            )
        except (AttributeError, TypeError):
            return AsyncDispatchConfig()

    def _is_km_handler_enabled(self, handler_name: str) -> bool:
        """Check if a KM handler is enabled via feature flags.

        Args:
            handler_name: The handler name (e.g., 'memory_to_mound')

        Returns:
            True if enabled (default) or settings not available
        """
        if self._settings is None:
            return True  # Default to enabled if settings unavailable
        try:
            integration = self._settings.integration
            return integration.is_km_handler_enabled(handler_name)
        except (AttributeError, TypeError):
            return True  # Default to enabled on error

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

        # =====================================================================
        # Bidirectional Knowledge Mound Handlers (NEW)
        # =====================================================================

        # Phase 1: Memory → KM (bidirectional completion)
        self.register(
            "memory_to_mound",
            StreamEventType.MEMORY_STORED,
            self._handle_memory_to_mound,
        )

        # Phase 1: KM → Memory pre-warm
        self.register(
            "mound_to_memory_retrieval",
            StreamEventType.KNOWLEDGE_QUERIED,
            self._handle_mound_to_memory_retrieval,
        )

        # Phase 2: Belief → KM
        self.register(
            "belief_to_mound",
            StreamEventType.BELIEF_CONVERGED,
            self._handle_belief_to_mound,
        )

        # Phase 2: KM → Belief (on debate start)
        self.register(
            "mound_to_belief",
            StreamEventType.DEBATE_START,
            self._handle_mound_to_belief,
        )

        # Phase 3: RLM → KM
        self.register(
            "rlm_to_mound",
            StreamEventType.RLM_COMPRESSION_COMPLETE,
            self._handle_rlm_to_mound,
        )

        # Phase 3: KM → RLM (on knowledge query)
        self.register(
            "mound_to_rlm",
            StreamEventType.KNOWLEDGE_QUERIED,
            self._handle_mound_to_rlm,
        )

        # Phase 4: ELO → KM
        self.register(
            "elo_to_mound",
            StreamEventType.AGENT_ELO_UPDATED,
            self._handle_elo_to_mound,
        )

        # Phase 4: KM → Team Selection (on debate start)
        self.register(
            "mound_to_team_selection",
            StreamEventType.DEBATE_START,
            self._handle_mound_to_team_selection,
        )

        # Phase 5: Insight → KM
        self.register(
            "insight_to_mound",
            StreamEventType.INSIGHT_EXTRACTED,
            self._handle_insight_to_mound,
        )

        # Phase 5: Flip → KM
        self.register(
            "flip_to_mound",
            StreamEventType.FLIP_DETECTED,
            self._handle_flip_to_mound,
        )

        # Phase 5: KM → Trickster (on debate start)
        self.register(
            "mound_to_trickster",
            StreamEventType.DEBATE_START,
            self._handle_mound_to_trickster,
        )

        # Phase 6: Culture → Debate (pattern updates)
        self.register(
            "culture_to_debate",
            StreamEventType.MOUND_UPDATED,
            self._handle_culture_to_debate,
        )

        # Phase 6b: Debate Start → Load Culture (active retrieval)
        self.register(
            "mound_to_culture",
            StreamEventType.DEBATE_START,
            self._handle_mound_to_culture,
        )

        # Phase 7: Staleness → Debate
        self.register(
            "staleness_to_debate",
            StreamEventType.KNOWLEDGE_STALE,
            self._handle_staleness_to_debate,
        )

        # Phase 8: Provenance → KM
        self.register(
            "provenance_to_mound",
            StreamEventType.CONSENSUS,
            self._handle_provenance_to_mound,
        )

        # Phase 8: KM → Provenance
        self.register(
            "mound_to_provenance",
            StreamEventType.CLAIM_VERIFICATION_RESULT,
            self._handle_mound_to_provenance,
        )

        # Phase 9: Consensus → KM (direct content ingestion)
        self.register(
            "consensus_to_mound",
            StreamEventType.CONSENSUS,
            self._handle_consensus_to_mound,
        )

        # Phase 10: KM Validation Feedback (reverse flow quality improvement)
        self.register(
            "km_validation_feedback",
            StreamEventType.CONSENSUS,
            self._handle_km_validation_feedback,
        )

        # Register webhook delivery for all cross-pollination events
        webhook_event_types = [
            StreamEventType.MEMORY_STORED,
            StreamEventType.MEMORY_RETRIEVED,
            StreamEventType.AGENT_ELO_UPDATED,
            StreamEventType.KNOWLEDGE_INDEXED,
            StreamEventType.KNOWLEDGE_QUERIED,
            StreamEventType.MOUND_UPDATED,
            StreamEventType.CALIBRATION_UPDATE,
            StreamEventType.EVIDENCE_FOUND,
        ]

        for event_type in webhook_event_types:
            self.register(
                f"webhook_{event_type.value.lower()}",
                event_type,
                self._handle_webhook_delivery,
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

        # Record event dispatch metric
        if METRICS_AVAILABLE:
            record_event_dispatched(event.type.value)

        handlers = self._subscribers.get(event.type, [])

        for name, handler in handlers:
            # Check if handler is enabled
            if name in self._stats and not self._stats[name].enabled:
                continue

            # Check circuit breaker - skip if handler circuit is open
            if not self._circuit_breaker.is_available(name):
                if name in self._stats:
                    self._stats[name].events_skipped += 1
                if METRICS_AVAILABLE:
                    record_handler_call(name, "skipped")
                    set_circuit_breaker_state(name, is_open=True)
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

            # Get retry config for this handler
            retry_config = (
                self._stats[name].retry_config
                if name in self._stats and self._stats[name].retry_config
                else self._default_retry_config
            )

            start_time = time.perf_counter()
            last_error: Optional[Exception] = None
            retried = False

            # Retry loop
            for attempt in range(retry_config.max_retries + 1):
                try:
                    if attempt > 0:
                        # Wait before retry with exponential backoff
                        delay_ms = retry_config.get_delay(attempt - 1)
                        time.sleep(delay_ms / 1000.0)
                        retried = True
                        logger.debug(f"Retrying handler {name}, attempt {attempt + 1}")

                    handler(event)

                    # Record success with circuit breaker
                    self._circuit_breaker.record_success(name)

                    # Calculate latency (includes retry delays)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Update stats with latency
                    if name in self._stats:
                        stats = self._stats[name]
                        stats.events_processed += 1
                        stats.last_event_time = datetime.now()
                        stats.total_latency_ms += latency_ms
                        stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
                        stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
                        stats.record_latency(latency_ms)  # For percentile tracking
                        if retried:
                            stats.events_retried += 1

                    # Record success metric
                    if METRICS_AVAILABLE:
                        record_handler_call(name, "success", duration=latency_ms / 1000.0)
                        set_circuit_breaker_state(name, is_open=False)

                    # Check handler execution SLO
                    if SLO_METRICS_AVAILABLE:
                        check_and_record_slo("handler_execution", latency_ms)

                    last_error = None
                    break  # Success - exit retry loop

                except Exception as e:
                    last_error = e
                    if attempt < retry_config.max_retries:
                        logger.debug(
                            f"Handler {name} failed (attempt {attempt + 1}), will retry: {e}"
                        )
                    continue  # Try again

            # If all retries exhausted and still failing
            if last_error is not None:
                # Record failure with circuit breaker
                self._circuit_breaker.record_failure(name)

                logger.error(
                    f"Cross-subscriber error in {name} after {retry_config.max_retries + 1} attempts: {last_error}"
                )
                if name in self._stats:
                    self._stats[name].events_failed += 1

                # Record failure metric
                if METRICS_AVAILABLE:
                    record_handler_call(name, "failure")

                # Check if circuit just opened
                if not self._circuit_breaker.is_available(name):
                    logger.warning(
                        f"Circuit breaker opened for handler {name} after repeated failures"
                    )
                    if METRICS_AVAILABLE:
                        set_circuit_breaker_state(name, is_open=True)

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

                # Check handler execution SLO
                if SLO_METRICS_AVAILABLE:
                    check_and_record_slo("handler_execution", latency_ms)

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

    def dispatch(self, event: StreamEvent) -> None:
        """Smart dispatch that routes events based on configuration.

        For high-volume event types (configured in async_config), uses async
        dispatch or batching. For other events, uses synchronous dispatch.

        Args:
            event: The event to dispatch
        """

        # Check if this event type should use async dispatch
        if event.type in self._async_config.async_event_types:
            # Check if batching is enabled
            if self._async_config.enable_batching and self._async_config.batch_size > 0:
                self._add_to_batch(event)
            else:
                # Direct async dispatch
                self.dispatch_async(event)
        else:
            # Synchronous dispatch for normal events
            self._dispatch_event(event)

    def _add_to_batch(self, event: StreamEvent) -> None:
        """Add event to batch queue, flushing if thresholds are met."""
        import time

        event_type = event.type
        if event_type not in self._event_batch:
            self._event_batch[event_type] = []

        self._event_batch[event_type].append(event)

        # Check if we should flush (batch size or timeout)
        should_flush = (
            len(self._event_batch[event_type]) >= self._async_config.batch_size
            or (time.time() - self._batch_last_flush) >= self._async_config.batch_timeout_seconds
        )

        if should_flush:
            self._flush_batch(event_type)

    def _flush_batch(self, event_type: StreamEventType) -> None:
        """Flush batched events for a specific type."""
        import time

        events = self._event_batch.get(event_type, [])
        if not events:
            return

        # Clear batch before processing
        self._event_batch[event_type] = []
        self._batch_last_flush = time.time()

        logger.debug(f"Flushing batch of {len(events)} {event_type.value} events")

        # Dispatch all events asynchronously
        for event in events:
            self.dispatch_async(event)

    def flush_all_batches(self) -> int:
        """Flush all pending batched events.

        Returns:
            Total number of events flushed
        """
        total = 0
        for event_type in list(self._event_batch.keys()):
            count = len(self._event_batch.get(event_type, []))
            if count > 0:
                self._flush_batch(event_type)
                total += count
        return total

    def get_batch_stats(self) -> dict:
        """Get statistics about pending batched events."""
        return {
            "pending_batches": {
                event_type.value: len(events)
                for event_type, events in self._event_batch.items()
                if events
            },
            "async_event_types": [et.value for et in self._async_config.async_event_types],
            "batch_size": self._async_config.batch_size,
            "batch_timeout_seconds": self._async_config.batch_timeout_seconds,
            "batching_enabled": self._async_config.enable_batching,
        }

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

    def _handle_webhook_delivery(self, event: StreamEvent) -> None:
        """
        Event → Webhook delivery.

        When any subscribable event occurs, deliver to registered webhooks.
        This enables external systems to receive real-time notifications.
        """
        try:
            from aragora.server.handlers.webhooks import get_webhook_store
            from aragora.events.dispatcher import dispatch_webhook_with_retry

            # Get registered webhooks for this event type
            store = get_webhook_store()
            event_type_str = event.type.value.lower()  # Convert enum to string
            webhooks = store.get_for_event(event_type_str)

            if not webhooks:
                return  # No webhooks registered for this event

            # Build payload
            import time
            import uuid

            payload = {
                "event": event_type_str,
                "delivery_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "data": event.data or {},
            }

            # Deliver to each matching webhook
            for webhook in webhooks:
                try:
                    result = dispatch_webhook_with_retry(webhook, payload)
                    if not result.success:
                        logger.warning(f"Webhook delivery failed for {webhook.id}: {result.error}")
                except Exception as e:
                    logger.error(f"Webhook dispatch error for {webhook.id}: {e}")

        except ImportError:
            logger.debug("Webhook modules not available for event delivery")
        except Exception as e:
            logger.debug(f"Webhook delivery handler error: {e}")

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
                f"Culture patterns updated: {patterns_count} patterns " f"from debate {debate_id}"
            )

        # Handle node deletions
        elif update_type == "node_deleted":
            node_id = data.get("node_id", "")
            archived = data.get("archived", False)
            logger.debug(f"Knowledge node removed: {node_id} (archived={archived})")

            # Clear any cached references to this node
            try:
                from aragora.memory import get_continuum_memory

                memory = get_continuum_memory()
                if memory and hasattr(memory, "invalidate_reference"):
                    memory.invalidate_reference(node_id)
            except (ImportError, AttributeError):
                pass

    # =========================================================================
    # Bidirectional Knowledge Mound Handlers (NEW)
    # =========================================================================

    def _handle_memory_to_mound(self, event: StreamEvent) -> None:
        """
        Memory stored → Knowledge Mound.

        Sync high-importance memories to Knowledge Mound for cross-debate access.
        Only syncs memories with importance ≥ 0.7 to avoid noise.
        """
        if not self._is_km_handler_enabled("memory_to_mound"):
            return

        data = event.data
        importance = data.get("importance", 0.0)
        content = data.get("content", "")
        tier = data.get("tier", "unknown")

        # Only sync significant memories
        if importance < 0.7:
            return

        logger.debug(
            f"Syncing high-importance memory to KM: importance={importance:.2f}, tier={tier}"
        )

        # Record KM inbound metric
        record_km_inbound_event("memory", event.type.value)

        try:
            from aragora.knowledge.mound import KnowledgeMound
            from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

            # Get or create mound instance
            mound = KnowledgeMound.get_instance()
            if mound is None:
                return

            # Use ContinuumAdapter to convert and store
            adapter = ContinuumAdapter()
            adapter.sync_memory_to_mound(
                content=content,
                importance=importance,
                tier=tier,
                metadata=data.get("metadata", {}),
            )
            logger.info(f"Synced memory to Knowledge Mound (importance={importance:.2f})")

        except ImportError:
            pass  # KnowledgeMound not available
        except Exception as e:
            logger.debug(f"Memory→KM sync failed: {e}")

    def _handle_mound_to_memory_retrieval(self, event: StreamEvent) -> None:
        """
        Knowledge Mound queried → Memory pre-warm.

        When KM is queried, check for related memories and pre-warm the cache.
        """
        if not self._is_km_handler_enabled("mound_to_memory"):
            return

        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        workspace_id = data.get("workspace_id", "")

        if not query or results_count == 0:
            return

        logger.debug(f"KM queried, pre-warming memory cache: query='{query[:50]}...'")

        # Record KM outbound metric
        record_km_outbound_event("memory", event.type.value)

        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory and hasattr(memory, "prewarm_for_query"):
                memory.prewarm_for_query(query, workspace_id=workspace_id)
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            logger.debug(f"KM→Memory pre-warm failed: {e}")

    def _handle_belief_to_mound(self, event: StreamEvent) -> None:
        """
        Belief network converged → Knowledge Mound.

        Store high-confidence beliefs and cruxes in KM for cross-debate learning.
        """
        if not self._is_km_handler_enabled("belief_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        beliefs_count = data.get("beliefs_count", 0)
        cruxes = data.get("cruxes", [])

        logger.debug(f"Belief network converged: {beliefs_count} beliefs, {len(cruxes)} cruxes")

        # Record KM inbound metric
        record_km_inbound_event("belief", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Store converged beliefs
            for belief_data in data.get("beliefs", []):
                if belief_data.get("confidence", 0) >= 0.8:
                    adapter.store_converged_belief(
                        node=belief_data,
                        debate_id=debate_id,
                    )

            # Store cruxes
            for crux_data in cruxes:
                adapter.store_crux(
                    crux=crux_data,
                    debate_id=debate_id,
                    topics=crux_data.get("topics", []),
                )

            logger.info(f"Stored beliefs/cruxes from debate {debate_id}")

        except ImportError:
            pass  # BeliefAdapter not available
        except Exception as e:
            logger.debug(f"Belief→KM storage failed: {e}")

    def _handle_mound_to_belief(self, event: StreamEvent) -> None:
        """
        Debate start → Initialize belief priors from KM.

        Retrieve historical cruxes and beliefs to initialize priors for new debate.
        """
        if not self._is_km_handler_enabled("mound_to_belief"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Initializing belief priors from KM for debate {debate_id}")

        # Record KM outbound metric
        record_km_outbound_event("belief", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Search for similar historical cruxes
            similar_cruxes = adapter.search_similar_cruxes(
                query=question,
                limit=10,
                min_score=0.3,
            )

            if similar_cruxes:
                logger.info(
                    f"Found {len(similar_cruxes)} historical cruxes relevant to debate {debate_id}"
                )
                # Store in event data for debate to pick up
                # (Actual initialization happens in debate orchestrator)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Belief initialization failed: {e}")

    def _handle_rlm_to_mound(self, event: StreamEvent) -> None:
        """
        RLM compression complete → Knowledge Mound.

        Store compression patterns that worked well for future retrieval optimization.
        """
        if not self._is_km_handler_enabled("rlm_to_mound"):
            return

        data = event.data
        compression_ratio = data.get("compression_ratio", 0.0)
        value_score = data.get("value_score", 0.0)
        content_markers = data.get("content_markers", [])

        # Only store high-value compression patterns
        if value_score < 0.7:
            return

        logger.debug(
            f"Storing RLM compression pattern: ratio={compression_ratio:.2f}, value={value_score:.2f}"
        )

        # Record KM inbound metric
        record_km_inbound_event("rlm", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

            adapter = RlmAdapter()
            adapter.store_compression_pattern(
                compression_ratio=compression_ratio,
                value_score=value_score,
                content_markers=content_markers,
                metadata=data.get("metadata", {}),
            )

        except ImportError:
            pass  # RlmAdapter not available yet
        except Exception as e:
            logger.debug(f"RLM→KM storage failed: {e}")

    def _handle_mound_to_rlm(self, event: StreamEvent) -> None:
        """
        Knowledge Mound queried → RLM priority update.

        Inform RLM about access patterns to optimize compression priorities.
        """
        if not self._is_km_handler_enabled("mound_to_rlm"):
            return

        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        node_ids = data.get("node_ids", [])

        if not node_ids:
            return

        logger.debug(f"Updating RLM priorities based on KM query: {results_count} results")

        # Record KM outbound metric
        record_km_outbound_event("rlm", event.type.value)

        try:
            from aragora.rlm.compressor import get_compressor

            compressor = get_compressor()
            if compressor and hasattr(compressor, "update_priority_hints"):
                compressor.update_priority_hints(
                    accessed_ids=node_ids,
                    query=query,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→RLM priority update failed: {e}")

    def _handle_elo_to_mound(self, event: StreamEvent) -> None:
        """
        ELO updated → Knowledge Mound.

        Store agent expertise profiles for cross-debate team selection.
        Only stores significant ELO changes (|delta| > 25).
        """
        if not self._is_km_handler_enabled("elo_to_mound"):
            return

        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)
        debate_id = data.get("debate_id", "")
        domain = data.get("domain", "general")

        # Only store significant changes
        if abs(delta) < 25:
            return

        logger.debug(
            f"Storing agent expertise: {agent_name} -> {new_elo} (Δ{delta:+.0f}) in {domain}"
        )

        # Record KM inbound metric
        record_km_inbound_event("ranking", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

            adapter = RankingAdapter()
            adapter.store_agent_expertise(
                agent_name=agent_name,
                domain=domain,
                elo=new_elo,
                delta=delta,
                debate_id=debate_id,
            )

        except ImportError:
            pass  # RankingAdapter not available yet
        except Exception as e:
            logger.debug(f"ELO→KM storage failed: {e}")

    def _handle_mound_to_team_selection(self, event: StreamEvent) -> None:
        """
        Debate start → Query KM for domain experts.

        Retrieve agent expertise profiles to inform team selection.
        """
        if not self._is_km_handler_enabled("mound_to_team"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Querying KM for domain experts for debate {debate_id}")

        # Record KM outbound metric
        record_km_outbound_event("team_selection", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

            adapter = RankingAdapter()
            # Detect domain from question
            domain = adapter.detect_domain(question)
            experts = adapter.get_domain_experts(domain=domain, limit=10)

            if experts:
                logger.info(f"Found {len(experts)} domain experts for '{domain}'")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Team selection query failed: {e}")

    def _handle_insight_to_mound(self, event: StreamEvent) -> None:
        """
        Insight extracted → Knowledge Mound.

        Store high-confidence insights (≥0.7) for organizational learning.
        """
        if not self._is_km_handler_enabled("insight_to_mound"):
            return

        data = event.data
        confidence = data.get("confidence", 0.0)
        insight_type = data.get("type", "")
        data.get("debate_id", "")

        # Only store high-confidence insights
        if confidence < 0.7:
            return

        logger.debug(f"Storing insight: type={insight_type}, confidence={confidence:.2f}")

        # Record KM inbound metric
        record_km_inbound_event("insights", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()
            adapter.store_insight(
                insight=data,
                min_confidence=0.7,
            )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Insight→KM storage failed: {e}")

    def _handle_flip_to_mound(self, event: StreamEvent) -> None:
        """
        Flip detected → Knowledge Mound.

        Store ALL flip events for meta-learning and consistency tracking.
        """
        if not self._is_km_handler_enabled("flip_to_mound"):
            return

        data = event.data
        agent_name = data.get("agent_name", "")
        flip_type = data.get("flip_type", "")

        logger.debug(f"Storing flip event: agent={agent_name}, type={flip_type}")

        # Record KM inbound metric
        record_km_inbound_event("trickster", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()
            adapter.store_flip(flip=data)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Flip→KM storage failed: {e}")

    def _handle_mound_to_trickster(self, event: StreamEvent) -> None:
        """
        Debate start → Query KM for flip history.

        Retrieve agent flip history for consistency prediction.
        """
        if not self._is_km_handler_enabled("mound_to_trickster"):
            return

        data = event.data
        data.get("debate_id", "")
        agents = data.get("agents", [])

        if not agents:
            return

        logger.debug(f"Querying KM for flip history: {len(agents)} agents")

        # Record KM outbound metric
        record_km_outbound_event("trickster", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()

            for agent_name in agents:
                flip_history = adapter.get_agent_flip_history(
                    agent_name=agent_name,
                    limit=20,
                )
                if flip_history:
                    logger.debug(f"Found {len(flip_history)} historical flips for {agent_name}")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Trickster query failed: {e}")

    def _handle_culture_to_debate(self, event: StreamEvent) -> None:
        """
        Culture patterns updated → Debate protocol.

        When culture patterns emerge, inform debate protocol selection.
        Only handles MOUND_UPDATED events with type=culture_patterns.
        """
        if not self._is_km_handler_enabled("culture_to_debate"):
            return

        data = event.data
        update_type = data.get("update_type", "")

        if update_type != "culture_patterns":
            return

        patterns_count = data.get("patterns_count", 0)
        workspace_id = data.get("workspace_id", "")

        logger.debug(
            f"Culture patterns available: {patterns_count} patterns in workspace {workspace_id}"
        )

        # Culture patterns are used passively during debate initialization
        # by querying the CultureAccumulator

    def _handle_mound_to_culture(self, event: StreamEvent) -> None:
        """
        Debate start → Load culture patterns from KM.

        Retrieve relevant culture patterns when a debate starts to inform
        protocol selection and agent behavior. Patterns include:
        - Decision style preferences (consensus vs majority)
        - Risk tolerance (conservative vs aggressive)
        - Domain expertise distribution
        - Debate dynamics (rounds to consensus, critique patterns)
        """
        if not self._is_km_handler_enabled("mound_to_culture"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        domain = data.get("domain", "")
        data.get("protocol", {})

        logger.debug(f"Loading culture patterns for debate {debate_id}, domain={domain}")

        # Record KM outbound metric
        record_km_outbound_event("culture", event.type.value)

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for culture retrieval")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping culture retrieval")
                return

            # Retrieve culture profile from mound
            import asyncio

            async def retrieve_culture():
                if hasattr(mound, "get_culture_profile"):
                    profile = await mound.get_culture_profile()
                    return profile
                return None

            # Run async retrieval
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task for later execution
                    asyncio.create_task(retrieve_culture())
                else:
                    profile = loop.run_until_complete(retrieve_culture())
                    if profile:
                        self._store_debate_culture(debate_id, profile, domain)
            except RuntimeError:
                profile = asyncio.run(retrieve_culture())
                if profile:
                    self._store_debate_culture(debate_id, profile, domain)

        except ImportError as e:
            logger.debug(f"Culture retrieval import failed: {e}")
        except Exception as e:
            logger.debug(f"Culture→Debate retrieval failed: {e}")

    def _store_debate_culture(
        self,
        debate_id: str,
        profile: Any,
        domain: str,
    ) -> None:
        """Store culture profile for a debate to inform protocol behavior.

        Args:
            debate_id: Debate identifier
            profile: CultureProfile from Knowledge Mound
            domain: Detected debate domain
        """
        try:
            # Store culture context for this debate
            # This can be accessed by the orchestrator during debate execution
            if not hasattr(self, "_debate_cultures"):
                self._debate_cultures: dict = {}

            # Extract relevant protocol hints from culture
            protocol_hints = {}

            if hasattr(profile, "dominant_pattern"):
                dominant = profile.dominant_pattern
                if dominant:
                    # Map decision style to protocol recommendations
                    if hasattr(dominant, "pattern_type"):
                        if str(dominant.pattern_type) == "decision_style":
                            protocol_hints["recommended_consensus"] = dominant.value

                    # Map risk tolerance to critique depth
                    if hasattr(dominant, "pattern_type"):
                        if str(dominant.pattern_type) == "risk_tolerance":
                            if dominant.value == "conservative":
                                protocol_hints["extra_critique_rounds"] = 1
                            elif dominant.value == "aggressive":
                                protocol_hints["early_consensus_threshold"] = 0.7

            # Extract domain-specific patterns
            if hasattr(profile, "patterns"):
                domain_patterns = [
                    p for p in profile.patterns if hasattr(p, "domain") and p.domain == domain
                ]
                if domain_patterns:
                    protocol_hints["domain_patterns"] = [
                        {"type": str(p.pattern_type), "value": p.value, "confidence": p.confidence}
                        for p in domain_patterns
                    ]

            self._debate_cultures[debate_id] = {
                "profile": profile,
                "protocol_hints": protocol_hints,
                "domain": domain,
            }

            logger.info(
                f"Stored culture context for debate {debate_id}: " f"{len(protocol_hints)} hints"
            )

        except Exception as e:
            logger.debug(f"Failed to store debate culture: {e}")

    def get_debate_culture_hints(self, debate_id: str) -> dict:
        """Get protocol hints from culture for a debate.

        Args:
            debate_id: Debate identifier

        Returns:
            Dict of protocol hints derived from organizational culture
        """
        if not hasattr(self, "_debate_cultures"):
            return {}

        culture_ctx = self._debate_cultures.get(debate_id, {})
        return culture_ctx.get("protocol_hints", {})

    def _handle_staleness_to_debate(self, event: StreamEvent) -> None:
        """
        Knowledge stale → Debate warning.

        When knowledge becomes stale, check if any active debate cites it.
        """
        if not self._is_km_handler_enabled("staleness_to_debate"):
            return

        data = event.data
        node_id = data.get("node_id", "")
        staleness_reason = data.get("reason", "")
        data.get("last_verified", "")

        logger.debug(f"Knowledge stale: {node_id} - {staleness_reason}")

        # Record KM outbound metric (staleness warning to debate)
        record_km_outbound_event("debate", event.type.value)

        try:
            from aragora.server.stream.state_manager import get_active_debates

            active_debates = get_active_debates()

            # Check if any active debate references this node
            for debate_id, debate_state in active_debates.items():
                cited_nodes = debate_state.get("cited_knowledge", [])
                if node_id in cited_nodes:
                    logger.warning(f"Active debate {debate_id} cites stale knowledge: {node_id}")
                    # Could emit a warning event to the debate here

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Staleness→Debate check failed: {e}")

    def _handle_provenance_to_mound(self, event: StreamEvent) -> None:
        """
        Consensus reached → Store verified provenance chains.

        After debate consensus, store verified provenance chains in KM.
        """
        if not self._is_km_handler_enabled("provenance_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)

        if not consensus_reached:
            return

        logger.debug(f"Storing provenance chains from consensus in debate {debate_id}")

        # Record KM inbound metric
        record_km_inbound_event("provenance", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Store verified provenance chains
            chains = data.get("provenance_chains", [])
            for chain in chains:
                if chain.get("verified", False):
                    adapter.store_provenance(
                        chain_id=chain.get("id", ""),
                        source_id=chain.get("source_id", ""),
                        claim_ids=chain.get("claim_ids", []),
                        verified=True,
                        verification_method=chain.get("method", "consensus"),
                        debate_id=debate_id,
                    )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Provenance→KM storage failed: {e}")

    def _handle_mound_to_provenance(self, event: StreamEvent) -> None:
        """
        Claim verification → Query KM for verification history.

        When verifying claims, check KM for related verified chains.
        """
        if not self._is_km_handler_enabled("mound_to_provenance"):
            return

        data = event.data
        claim_id = data.get("claim_id", "")
        claim_text = data.get("claim", "")

        if not claim_text:
            return

        logger.debug(f"Querying KM for verification history: claim {claim_id}")

        # Record KM outbound metric
        record_km_outbound_event("provenance", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Search for related verified claims
            related = adapter.search_similar_cruxes(
                query=claim_text,
                limit=5,
            )

            if related:
                logger.debug(f"Found {len(related)} related verified claims")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Provenance query failed: {e}")

    def _handle_consensus_to_mound(self, event: StreamEvent) -> None:
        """
        Consensus reached → Ingest consensus content to Knowledge Mound.

        After debate consensus, store the consensus conclusion, key claims,
        and dissenting views as knowledge nodes for organizational learning.

        Enhanced features:
        - Dissent tracking: Store dissenting views as separate nodes linked to consensus
        - Evolution tracking: Detect similar prior consensus and create supersedes links
        - Linking: Connect consensus to claims, evidence, and related knowledge
        """
        if not self._is_km_handler_enabled("consensus_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)

        if not consensus_reached:
            return

        topic = data.get("topic", "")
        conclusion = data.get("conclusion", "")
        confidence = data.get("confidence", 0.5)
        strength = data.get("strength", "moderate")
        key_claims = data.get("key_claims", [])
        supporting_evidence = data.get("supporting_evidence", [])
        domain = data.get("domain", "general")
        tags = data.get("tags", [])

        # Dissent data
        dissents = data.get("dissents", [])
        dissenting_agents = data.get("dissenting_agents", [])
        _dissent_ids = data.get("dissent_ids", [])  # Preserved for future linking

        # Evolution data
        supersedes = data.get("supersedes", None)
        agreeing_agents = data.get("agreeing_agents", [])
        participating_agents = data.get("participating_agents", [])

        if not topic and not conclusion:
            return

        logger.info(
            f"Ingesting consensus from debate {debate_id} to Knowledge Mound "
            f"(dissents={len(dissents)}, evolution={supersedes is not None})"
        )

        # Record KM inbound metric
        record_km_inbound_event("consensus", event.type.value)

        try:
            from aragora.knowledge.mound import get_knowledge_mound
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for consensus ingestion")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping consensus ingestion")
                return

            # Build content from topic and conclusion
            content = f"{topic}: {conclusion}" if conclusion else topic

            # Map strength to tier
            strength_to_tier = {
                "unanimous": "glacial",  # Highly stable
                "strong": "slow",
                "moderate": "slow",
                "weak": "medium",
                "split": "medium",
                "contested": "fast",  # May change
            }
            tier = strength_to_tier.get(strength, "slow")

            # Calculate agreement ratio
            agreement_ratio = (
                len(agreeing_agents) / len(participating_agents) if participating_agents else 0.0
            )

            import asyncio

            async def ingest_consensus_with_enhancements():
                # ============================================================
                # EVOLUTION TRACKING: Check for similar prior consensus
                # ============================================================
                supersedes_node_id = None
                if supersedes:
                    # Direct supersedes reference provided
                    supersedes_node_id = f"cs_{supersedes}"
                else:
                    # Search for similar prior consensus on same topic
                    try:
                        similar_results = await mound.search(
                            query=topic,
                            node_types=["consensus"],
                            limit=3,
                            min_score=0.85,  # High threshold for "same topic"
                        )
                        if similar_results:
                            # Found similar prior consensus - this new one supersedes it
                            prior = similar_results[0]
                            prior_debate_id = prior.metadata.get("debate_id", "")
                            if prior_debate_id != debate_id:
                                supersedes_node_id = prior.id
                                logger.info(
                                    f"Consensus {debate_id} supersedes prior "
                                    f"consensus {prior_debate_id} on topic '{topic[:50]}...'"
                                )
                    except Exception as e:
                        logger.debug(f"Evolution tracking search failed: {e}")

                # ============================================================
                # MAIN CONSENSUS INGESTION
                # ============================================================
                request = IngestionRequest(
                    content=content,
                    workspace_id=mound.workspace_id,
                    source_type=KnowledgeSource.CONSENSUS,
                    debate_id=debate_id,
                    node_type="consensus",
                    confidence=confidence,
                    tier=tier,
                    supersedes=supersedes_node_id,
                    metadata={
                        "debate_id": debate_id,
                        "strength": strength,
                        "topic": topic,
                        "conclusion": conclusion,
                        "domain": domain,
                        "tags": tags,
                        "key_claims_count": len(key_claims),
                        "dissent_count": len(dissents),
                        "agreement_ratio": agreement_ratio,
                        "agreeing_agents": agreeing_agents,
                        "dissenting_agents": dissenting_agents,
                        "participating_agents": participating_agents,
                        "has_dissent": len(dissents) > 0 or len(dissenting_agents) > 0,
                        "ingested_at": datetime.now().isoformat(),
                    },
                )

                result = await mound.store(request)
                consensus_node_id = result.node_id

                logger.debug(
                    f"Ingested consensus {debate_id}: node_id={consensus_node_id}, "
                    f"deduplicated={result.deduplicated}, supersedes={supersedes_node_id}"
                )

                # ============================================================
                # DISSENT TRACKING: Store dissenting views
                # ============================================================
                dissent_node_ids = []
                for i, dissent in enumerate(dissents[:10]):  # Limit to 10 dissents
                    if isinstance(dissent, dict):
                        dissent_content = dissent.get("content", "")
                        dissent_type = dissent.get(
                            "type", dissent.get("dissent_type", "alternative_approach")
                        )
                        dissent_agent = dissent.get("agent_id", dissent.get("agent", "unknown"))
                        dissent_reasoning = dissent.get("reasoning", "")
                        dissent_confidence = dissent.get("confidence", 0.5)
                        acknowledged = dissent.get("acknowledged", False)
                        rebuttal = dissent.get("rebuttal", "")
                    elif isinstance(dissent, str):
                        dissent_content = dissent
                        dissent_type = "alternative_approach"
                        dissent_agent = (
                            dissenting_agents[i] if i < len(dissenting_agents) else "unknown"
                        )
                        dissent_reasoning = ""
                        dissent_confidence = 0.5
                        acknowledged = False
                        rebuttal = ""
                    else:
                        continue

                    if not dissent_content.strip():
                        continue

                    # Determine dissent importance based on type
                    dissent_importance = 0.5
                    if dissent_type == "risk_warning":
                        dissent_importance = 0.7  # Risk warnings are valuable
                    elif dissent_type == "fundamental_disagreement":
                        dissent_importance = 0.6  # Strong dissent worth preserving
                    elif dissent_type == "edge_case_concern":
                        dissent_importance = 0.55  # Edge cases inform future debates

                    dissent_request = IngestionRequest(
                        content=f"[DISSENT from {dissent_agent}] {dissent_content}",
                        workspace_id=mound.workspace_id,
                        source_type=KnowledgeSource.CONSENSUS,
                        debate_id=debate_id,
                        node_type="dissent",
                        confidence=dissent_confidence,
                        tier="medium",  # Dissents may be reconsidered
                        derived_from=[consensus_node_id] if consensus_node_id else None,
                        metadata={
                            "debate_id": debate_id,
                            "dissent_type": dissent_type,
                            "agent_id": dissent_agent,
                            "reasoning": dissent_reasoning,
                            "acknowledged": acknowledged,
                            "rebuttal": rebuttal,
                            "parent_consensus_id": consensus_node_id,
                            "dissent_index": i,
                            "topic": topic,
                            "is_risk_warning": dissent_type == "risk_warning",
                            "importance": dissent_importance,
                        },
                    )

                    dissent_result = await mound.store(dissent_request)
                    if dissent_result.node_id:
                        dissent_node_ids.append(dissent_result.node_id)
                        logger.debug(
                            f"Stored dissent from {dissent_agent}: "
                            f"type={dissent_type}, node_id={dissent_result.node_id}"
                        )

                if dissent_node_ids:
                    logger.info(
                        f"Stored {len(dissent_node_ids)} dissenting views "
                        f"for consensus {debate_id}"
                    )

                # ============================================================
                # CLAIM LINKING: Store key claims linked to consensus
                # ============================================================
                claim_node_ids = []
                for i, claim in enumerate(key_claims[:10]):  # Limit to 10 claims
                    if isinstance(claim, str) and claim.strip():
                        claim_request = IngestionRequest(
                            content=claim,
                            workspace_id=mound.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=debate_id,
                            node_type="claim",
                            confidence=confidence * 0.9,  # Slightly lower than main consensus
                            tier=tier,
                            derived_from=[consensus_node_id] if consensus_node_id else None,
                            metadata={
                                "debate_id": debate_id,
                                "claim_index": i,
                                "parent_consensus_id": consensus_node_id,
                                "domain": domain,
                            },
                        )
                        claim_result = await mound.store(claim_request)
                        if claim_result.node_id:
                            claim_node_ids.append(claim_result.node_id)

                # ============================================================
                # EVIDENCE LINKING: Store supporting evidence references
                # ============================================================
                for i, evidence in enumerate(supporting_evidence[:5]):  # Limit evidence
                    if isinstance(evidence, str) and evidence.strip():
                        evidence_request = IngestionRequest(
                            content=evidence,
                            workspace_id=mound.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=debate_id,
                            node_type="evidence",
                            confidence=confidence * 0.85,
                            tier=tier,
                            derived_from=[consensus_node_id] if consensus_node_id else None,
                            metadata={
                                "debate_id": debate_id,
                                "evidence_index": i,
                                "parent_consensus_id": consensus_node_id,
                                "supports_conclusion": True,
                            },
                        )
                        await mound.store(evidence_request)

                # ============================================================
                # UPDATE SUPERSEDED NODE (if applicable)
                # ============================================================
                if supersedes_node_id and hasattr(mound, "update_metadata"):
                    try:
                        await mound.update_metadata(
                            node_id=supersedes_node_id,
                            updates={"superseded_by": consensus_node_id},
                        )
                        logger.debug(
                            f"Marked {supersedes_node_id} as superseded by {consensus_node_id}"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update superseded node: {e}")

                # Log summary
                logger.info(
                    f"Consensus ingestion complete for debate {debate_id}: "
                    f"consensus={consensus_node_id}, claims={len(claim_node_ids)}, "
                    f"dissents={len(dissent_node_ids)}, "
                    f"supersedes={'yes' if supersedes_node_id else 'no'}"
                )

            # Run async ingestion
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(ingest_consensus_with_enhancements())
                else:
                    loop.run_until_complete(ingest_consensus_with_enhancements())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(ingest_consensus_with_enhancements())

        except ImportError as e:
            logger.debug(f"Consensus→KM ingestion import failed: {e}")
        except Exception as e:
            logger.warning(f"Consensus→KM ingestion failed: {e}")

    def _handle_km_validation_feedback(self, event: StreamEvent) -> None:
        """
        KM Validation Feedback: Improve source system quality based on debate outcomes.

        When consensus is reached, this handler:
        1. Queries KM for items that may have contributed to the debate
        2. For items from ContinuumMemory or ConsensusMemory that match the topic:
           - If consensus was reached with high confidence → positive validation
           - If consensus contradicts prior knowledge → negative validation
        3. Feeds validation back to source adapters to improve quality scores

        This creates a learning loop where KM data that proves useful in debates
        gets promoted (higher tiers, higher importance), while contradicted data
        gets demoted or flagged for review.
        """
        if not self._is_km_handler_enabled("km_validation_feedback"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)
        confidence = data.get("confidence", 0.5)
        topic = data.get("topic", "")

        # Only process debates with clear outcomes
        if not consensus_reached or confidence < 0.5 or not topic:
            return

        logger.debug(
            f"Processing KM validation feedback for debate {debate_id}: "
            f"confidence={confidence:.2f}, topic={topic[:50]}..."
        )

        try:
            import asyncio
            from aragora.knowledge.mound import get_knowledge_mound
            from aragora.knowledge.mound.adapters.continuum_adapter import (
                ContinuumAdapter,
                KMValidationResult,
            )
            from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter  # noqa: F401

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for validation feedback")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping validation feedback")
                return

            async def process_validation_feedback():
                # Query KM for items that may have contributed to this debate
                try:
                    # Search for related knowledge by topic
                    results = await mound.search(
                        query=topic,
                        limit=20,
                        min_score=0.6,  # Moderate threshold for potential contributors
                    )

                    if not results:
                        logger.debug(f"No KM items found for validation feedback: {topic[:50]}")
                        return

                    continuum_validations = 0
                    consensus_validations = 0

                    for result in results:
                        node_id = (
                            result.node_id
                            if hasattr(result, "node_id")
                            else result.get("node_id", "")
                        )
                        score = (
                            result.score if hasattr(result, "score") else result.get("score", 0.0)
                        )
                        source = (
                            result.source if hasattr(result, "source") else result.get("source", "")
                        )

                        # Determine validation recommendation based on outcome
                        # High confidence + high similarity = item was useful
                        cross_debate_utility = score * confidence

                        if confidence >= 0.8 and score >= 0.7:
                            recommendation = "promote"
                        elif confidence >= 0.6 and score >= 0.5:
                            recommendation = "keep"
                        elif confidence < 0.5:
                            recommendation = "review"
                        else:
                            recommendation = "keep"

                        # Create validation result
                        validation = KMValidationResult(
                            memory_id=node_id,
                            km_confidence=confidence,
                            cross_debate_utility=cross_debate_utility,
                            validation_count=1,
                            was_supported=consensus_reached and confidence >= 0.7,
                            was_contradicted=False,  # Would need contradiction detection
                            recommendation=recommendation,
                            metadata={
                                "debate_id": debate_id,
                                "topic": topic[:100],
                                "similarity_score": score,
                                "source_type": source,
                            },
                        )

                        # Route validation to appropriate adapter
                        if node_id.startswith("cm_"):
                            # ContinuumMemory item
                            try:
                                from aragora.memory.continuum import get_continuum_memory

                                continuum = get_continuum_memory()
                                if continuum and hasattr(continuum, "_km_adapter"):
                                    adapter = continuum._km_adapter
                                    if adapter and isinstance(adapter, ContinuumAdapter):
                                        updated = await adapter.update_continuum_from_km(
                                            memory_id=node_id,
                                            km_validation=validation,
                                        )
                                        if updated:
                                            continuum_validations += 1
                            except ImportError:
                                pass
                            except Exception as e:
                                logger.debug(f"Continuum validation failed: {e}")

                        elif node_id.startswith("cs_"):
                            # Consensus item - track but consensus records are immutable
                            # Instead, update the confidence tracking for the adapter
                            consensus_validations += 1

                    if continuum_validations > 0 or consensus_validations > 0:
                        logger.info(
                            f"KM validation feedback for debate {debate_id}: "
                            f"continuum={continuum_validations}, consensus={consensus_validations}"
                        )

                        # Emit validation event for dashboard
                        try:
                            from aragora.events.types import StreamEvent, StreamEventType

                            validation_event = StreamEvent(
                                type=StreamEventType.KM_ADAPTER_VALIDATION,
                                data={
                                    "debate_id": debate_id,
                                    "topic_preview": topic[:50],
                                    "confidence": confidence,
                                    "continuum_validations": continuum_validations,
                                    "consensus_validations": consensus_validations,
                                    "total_items_reviewed": len(results),
                                },
                            )
                            # Don't dispatch to avoid recursion - just log for now
                            logger.debug(f"Validation event: {validation_event.data}")
                        except Exception as e:
                            logger.debug(f"Failed to create validation event: {e}")

                except Exception as e:
                    logger.warning(f"KM validation feedback query failed: {e}")

            # Run async validation
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(process_validation_feedback())
                else:
                    loop.run_until_complete(process_validation_feedback())
            except RuntimeError:
                asyncio.run(process_validation_feedback())

        except ImportError as e:
            logger.debug(f"KM validation feedback import failed: {e}")
        except Exception as e:
            logger.warning(f"KM validation feedback failed: {e}")

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all subscribers including latency, sampling, retry, and circuit breaker metrics."""
        result = {}
        for name, stats in self._stats.items():
            # Get circuit breaker status for this handler
            cb_state = (
                self._circuit_breaker.get_state(name)
                if hasattr(self._circuit_breaker, "get_state")
                else "unknown"
            )
            cb_available = self._circuit_breaker.is_available(name)

            # Get retry config
            retry_cfg = stats.retry_config or self._default_retry_config

            result[name] = {
                "events_processed": stats.events_processed,
                "events_failed": stats.events_failed,
                "events_skipped": stats.events_skipped,
                "events_retried": stats.events_retried,
                "last_event": stats.last_event_time.isoformat() if stats.last_event_time else None,
                "enabled": stats.enabled,
                "sample_rate": stats.sample_rate,
                "has_filter": name in self._filters,
                "latency_ms": {
                    "avg": round(stats.avg_latency_ms, 3),
                    "min": (
                        round(stats.min_latency_ms, 3)
                        if stats.min_latency_ms != float("inf")
                        else None
                    ),
                    "max": round(stats.max_latency_ms, 3),
                    "total": round(stats.total_latency_ms, 3),
                    "p50": (
                        round(stats.p50_latency_ms, 3) if stats.p50_latency_ms is not None else None
                    ),
                    "p90": (
                        round(stats.p90_latency_ms, 3) if stats.p90_latency_ms is not None else None
                    ),
                    "p99": (
                        round(stats.p99_latency_ms, 3) if stats.p99_latency_ms is not None else None
                    ),
                    "sample_count": len(stats.latency_samples),
                },
                "retry": {
                    "max_retries": retry_cfg.max_retries,
                    "base_delay_ms": retry_cfg.base_delay_ms,
                    "max_delay_ms": retry_cfg.max_delay_ms,
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
        """Reset all subscriber statistics including latency and retry counts."""
        for stats in self._stats.values():
            stats.events_processed = 0
            stats.events_failed = 0
            stats.events_skipped = 0
            stats.events_retried = 0
            stats.last_event_time = None
            stats.total_latency_ms = 0.0
            stats.min_latency_ms = float("inf")
            stats.max_latency_ms = 0.0
            stats.latency_samples = []

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

    def set_retry_config(
        self,
        name: str,
        max_retries: Optional[int] = None,
        base_delay_ms: Optional[float] = None,
        max_delay_ms: Optional[float] = None,
    ) -> bool:
        """Set retry configuration for a specific handler.

        Args:
            name: Handler name
            max_retries: Maximum retry attempts (None to keep current)
            base_delay_ms: Base delay between retries (None to keep current)
            max_delay_ms: Maximum delay cap (None to keep current)

        Returns:
            True if handler found and updated, False otherwise.
        """
        if name not in self._stats:
            return False

        stats = self._stats[name]
        if stats.retry_config is None:
            stats.retry_config = RetryConfig()

        if max_retries is not None:
            stats.retry_config.max_retries = max_retries
        if base_delay_ms is not None:
            stats.retry_config.base_delay_ms = base_delay_ms
        if max_delay_ms is not None:
            stats.retry_config.max_delay_ms = max_delay_ms

        logger.info(
            f"Updated retry config for '{name}': "
            f"max_retries={stats.retry_config.max_retries}, "
            f"base_delay={stats.retry_config.base_delay_ms}ms"
        )
        return True

    def disable_retry(self, name: str) -> bool:
        """Disable retry for a specific handler.

        Args:
            name: Handler name

        Returns:
            True if handler found and updated, False otherwise.
        """
        if name not in self._stats:
            return False

        # Set max_retries to 0 to disable retry
        self._stats[name].retry_config = RetryConfig(max_retries=0)
        logger.info(f"Disabled retry for handler '{name}'")
        return True

    def get_performance_report(self) -> dict:
        """Get a comprehensive performance profiling report.

        Returns a summary of dispatch performance including:
        - Total events processed across all handlers
        - Latency percentiles (P50, P90, P99) per handler
        - Error rates per handler
        - Slowest and fastest handlers
        - Circuit breaker status summary
        """
        stats = self.get_stats()

        total_processed = sum(s.get("events_processed", 0) for s in stats.values())
        total_failed = sum(s.get("events_failed", 0) for s in stats.values())
        total_skipped = sum(s.get("events_skipped", 0) for s in stats.values())
        total_retried = sum(s.get("events_retried", 0) for s in stats.values())

        # Find slowest handlers by P90 latency
        handlers_by_p90 = []
        for name, s in stats.items():
            p90 = s.get("latency_ms", {}).get("p90")
            if p90 is not None:
                handlers_by_p90.append((name, p90))
        handlers_by_p90.sort(key=lambda x: x[1], reverse=True)

        # Find handlers with highest error rates
        handlers_by_error_rate = []
        for name, s in stats.items():
            processed = s.get("events_processed", 0)
            failed = s.get("events_failed", 0)
            if processed > 0:
                error_rate = failed / (processed + failed)
                handlers_by_error_rate.append((name, error_rate))
        handlers_by_error_rate.sort(key=lambda x: x[1], reverse=True)

        # Circuit breaker summary
        circuits_open = sum(
            1 for s in stats.values() if not s.get("circuit_breaker", {}).get("available", True)
        )

        return {
            "summary": {
                "total_handlers": len(stats),
                "total_events_processed": total_processed,
                "total_events_failed": total_failed,
                "total_events_skipped": total_skipped,
                "total_events_retried": total_retried,
                "overall_error_rate": round(
                    total_failed / max(total_processed + total_failed, 1), 4
                ),
                "circuits_open": circuits_open,
            },
            "slowest_handlers": [
                {"name": name, "p90_latency_ms": lat} for name, lat in handlers_by_p90[:5]
            ],
            "highest_error_handlers": [
                {"name": name, "error_rate": round(rate, 4)}
                for name, rate in handlers_by_error_rate[:5]
                if rate > 0
            ],
            "per_handler": stats,
        }


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
    "RetryConfig",
    "get_cross_subscriber_manager",
    "reset_cross_subscriber_manager",
]
