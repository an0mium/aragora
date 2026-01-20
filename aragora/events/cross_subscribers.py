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

# Import metrics (optional - graceful fallback if not available)
try:
    from aragora.server.prometheus_cross_pollination import (
        record_event_dispatched,
        record_handler_call,
        set_circuit_breaker_state,
        update_subscriber_count,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior on handler failures."""

    max_retries: int = 3
    base_delay_ms: float = 100.0  # Base delay between retries
    max_delay_ms: float = 5000.0  # Maximum delay cap
    exponential_base: float = 2.0  # Exponential backoff multiplier

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt (0-indexed).

        Uses exponential backoff with jitter: delay = min(base * exp^attempt + jitter, max)
        """
        import random

        delay = self.base_delay_ms * (self.exponential_base ** attempt)
        # Add jitter (±20%)
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        delay += jitter
        return min(delay, self.max_delay_ms)


@dataclass
class SubscriberStats:
    """Statistics for a cross-subsystem subscriber."""

    name: str
    events_processed: int = 0
    events_failed: int = 0
    events_skipped: int = 0  # Skipped due to sampling/filtering
    events_retried: int = 0  # Events that required retry
    last_event_time: Optional[datetime] = None
    enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = 100% of events, 0.1 = 10% sampling
    retry_config: Optional[RetryConfig] = None  # Per-handler retry config
    # Latency metrics (in milliseconds)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    # Latency histogram buckets (for P50, P90, P99 calculation)
    latency_samples: list = field(default_factory=list)
    max_samples: int = 1000  # Keep last N samples for percentile calculation

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.events_processed == 0:
            return 0.0
        return self.total_latency_ms / self.events_processed

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample for percentile calculation."""
        self.latency_samples.append(latency_ms)
        # Maintain bounded sample size
        if len(self.latency_samples) > self.max_samples:
            self.latency_samples = self.latency_samples[-self.max_samples:]

    def get_percentile(self, p: float) -> Optional[float]:
        """Get latency at given percentile (0-100)."""
        if not self.latency_samples:
            return None
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50_latency_ms(self) -> Optional[float]:
        """50th percentile (median) latency."""
        return self.get_percentile(50)

    @property
    def p90_latency_ms(self) -> Optional[float]:
        """90th percentile latency."""
        return self.get_percentile(90)

    @property
    def p99_latency_ms(self) -> Optional[float]:
        """99th percentile latency."""
        return self.get_percentile(99)


@dataclass
class AsyncDispatchConfig:
    """Configuration for async/batched event dispatch."""

    # Event types that should always use async dispatch
    async_event_types: set = field(default_factory=lambda: {
        StreamEventType.MEMORY_STORED,
        StreamEventType.MEMORY_RETRIEVED,
        StreamEventType.KNOWLEDGE_QUERIED,
        StreamEventType.RLM_COMPRESSION_COMPLETE,
    })

    # Batch size before auto-flush (0 = no batching)
    batch_size: int = 10

    # Maximum time (seconds) to hold events before flush
    batch_timeout_seconds: float = 1.0

    # Enable batching (groups similar events)
    enable_batching: bool = True


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
        self._subscribers: dict[StreamEventType, list[tuple[str, Callable[[StreamEvent], None]]]] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self._filters: dict[str, Callable[[StreamEvent], bool]] = {}
        self._connected = False

        # Default retry configuration
        self._default_retry_config = default_retry_config or RetryConfig()

        # Async dispatch configuration
        self._async_config = async_config or AsyncDispatchConfig()

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

        # Phase 6: Culture → Debate
        self.register(
            "culture_to_debate",
            StreamEventType.MOUND_UPDATED,
            self._handle_culture_to_debate,
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
                self._stats[name].retry_config if name in self._stats and self._stats[name].retry_config
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

                    last_error = None
                    break  # Success - exit retry loop

                except Exception as e:
                    last_error = e
                    if attempt < retry_config.max_retries:
                        logger.debug(f"Handler {name} failed (attempt {attempt + 1}), will retry: {e}")
                    continue  # Try again

            # If all retries exhausted and still failing
            if last_error is not None:
                # Record failure with circuit breaker
                self._circuit_breaker.record_failure(name)

                logger.error(f"Cross-subscriber error in {name} after {retry_config.max_retries + 1} attempts: {last_error}")
                if name in self._stats:
                    self._stats[name].events_failed += 1

                # Record failure metric
                if METRICS_AVAILABLE:
                    record_handler_call(name, "failure")

                # Check if circuit just opened
                if not self._circuit_breaker.is_available(name):
                    logger.warning(f"Circuit breaker opened for handler {name} after repeated failures")
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
        import time

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
                        logger.warning(
                            f"Webhook delivery failed for {webhook.id}: {result.error}"
                        )
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
    # Bidirectional Knowledge Mound Handlers (NEW)
    # =========================================================================

    def _handle_memory_to_mound(self, event: StreamEvent) -> None:
        """
        Memory stored → Knowledge Mound.

        Sync high-importance memories to Knowledge Mound for cross-debate access.
        Only syncs memories with importance ≥ 0.7 to avoid noise.
        """
        data = event.data
        importance = data.get("importance", 0.0)
        content = data.get("content", "")
        tier = data.get("tier", "unknown")

        # Only sync significant memories
        if importance < 0.7:
            return

        logger.debug(f"Syncing high-importance memory to KM: importance={importance:.2f}, tier={tier}")

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
        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        workspace_id = data.get("workspace_id", "")

        if not query or results_count == 0:
            return

        logger.debug(f"KM queried, pre-warming memory cache: query='{query[:50]}...'")

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
        data = event.data
        debate_id = data.get("debate_id", "")
        beliefs_count = data.get("beliefs_count", 0)
        cruxes = data.get("cruxes", [])

        logger.debug(f"Belief network converged: {beliefs_count} beliefs, {len(cruxes)} cruxes")

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
        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Initializing belief priors from KM for debate {debate_id}")

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
        data = event.data
        compression_ratio = data.get("compression_ratio", 0.0)
        value_score = data.get("value_score", 0.0)
        content_markers = data.get("content_markers", [])

        # Only store high-value compression patterns
        if value_score < 0.7:
            return

        logger.debug(f"Storing RLM compression pattern: ratio={compression_ratio:.2f}, value={value_score:.2f}")

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
        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        node_ids = data.get("node_ids", [])

        if not node_ids:
            return

        logger.debug(f"Updating RLM priorities based on KM query: {results_count} results")

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
        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)
        debate_id = data.get("debate_id", "")
        domain = data.get("domain", "general")

        # Only store significant changes
        if abs(delta) < 25:
            return

        logger.debug(f"Storing agent expertise: {agent_name} -> {new_elo} (Δ{delta:+.0f}) in {domain}")

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
        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Querying KM for domain experts for debate {debate_id}")

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
        data = event.data
        confidence = data.get("confidence", 0.0)
        insight_type = data.get("type", "")
        debate_id = data.get("debate_id", "")

        # Only store high-confidence insights
        if confidence < 0.7:
            return

        logger.debug(f"Storing insight: type={insight_type}, confidence={confidence:.2f}")

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
        data = event.data
        agent_name = data.get("agent_name", "")
        flip_type = data.get("flip_type", "")

        logger.debug(f"Storing flip event: agent={agent_name}, type={flip_type}")

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
        data = event.data
        debate_id = data.get("debate_id", "")
        agents = data.get("agents", [])

        if not agents:
            return

        logger.debug(f"Querying KM for flip history: {len(agents)} agents")

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
        data = event.data
        update_type = data.get("update_type", "")

        if update_type != "culture_patterns":
            return

        patterns_count = data.get("patterns_count", 0)
        workspace_id = data.get("workspace_id", "")

        logger.debug(f"Culture patterns available: {patterns_count} patterns in workspace {workspace_id}")

        # Culture patterns are used passively during debate initialization
        # by querying the CultureAccumulator

    def _handle_staleness_to_debate(self, event: StreamEvent) -> None:
        """
        Knowledge stale → Debate warning.

        When knowledge becomes stale, check if any active debate cites it.
        """
        data = event.data
        node_id = data.get("node_id", "")
        staleness_reason = data.get("reason", "")
        last_verified = data.get("last_verified", "")

        logger.debug(f"Knowledge stale: {node_id} - {staleness_reason}")

        try:
            from aragora.server.stream.state_manager import get_active_debates

            active_debates = get_active_debates()

            # Check if any active debate references this node
            for debate_id, debate_state in active_debates.items():
                cited_nodes = debate_state.get("cited_knowledge", [])
                if node_id in cited_nodes:
                    logger.warning(
                        f"Active debate {debate_id} cites stale knowledge: {node_id}"
                    )
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
        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)

        if not consensus_reached:
            return

        logger.debug(f"Storing provenance chains from consensus in debate {debate_id}")

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
        data = event.data
        claim_id = data.get("claim_id", "")
        claim_text = data.get("claim", "")

        if not claim_text:
            return

        logger.debug(f"Querying KM for verification history: claim {claim_id}")

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

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all subscribers including latency, sampling, retry, and circuit breaker metrics."""
        result = {}
        for name, stats in self._stats.items():
            # Get circuit breaker status for this handler
            cb_state = self._circuit_breaker.get_state(name) if hasattr(self._circuit_breaker, 'get_state') else "unknown"
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
                    "min": round(stats.min_latency_ms, 3) if stats.min_latency_ms != float("inf") else None,
                    "max": round(stats.max_latency_ms, 3),
                    "total": round(stats.total_latency_ms, 3),
                    "p50": round(stats.p50_latency_ms, 3) if stats.p50_latency_ms is not None else None,
                    "p90": round(stats.p90_latency_ms, 3) if stats.p90_latency_ms is not None else None,
                    "p99": round(stats.p99_latency_ms, 3) if stats.p99_latency_ms is not None else None,
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
            1 for s in stats.values()
            if not s.get("circuit_breaker", {}).get("available", True)
        )

        return {
            "summary": {
                "total_handlers": len(stats),
                "total_events_processed": total_processed,
                "total_events_failed": total_failed,
                "total_events_skipped": total_skipped,
                "total_events_retried": total_retried,
                "overall_error_rate": round(total_failed / max(total_processed + total_failed, 1), 4),
                "circuits_open": circuits_open,
            },
            "slowest_handlers": [
                {"name": name, "p90_latency_ms": lat} for name, lat in handlers_by_p90[:5]
            ],
            "highest_error_handlers": [
                {"name": name, "error_rate": round(rate, 4)} for name, rate in handlers_by_error_rate[:5]
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
