"""
Cross-Subscriber Manager.

Core CrossSubscriberManager class that orchestrates event dispatch
and subscriber lifecycle management.
"""

import logging
import random
import time
from datetime import datetime
from typing import Any, Callable, Optional

from aragora.events.subscribers.config import (
    AsyncDispatchConfig,
    RetryConfig,
    SubscriberStats,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.resilience import CircuitBreaker

from .handlers.basic import BasicHandlersMixin
from .handlers.culture import CultureHandlersMixin
from .handlers.knowledge_mound import KnowledgeMoundHandlersMixin
from .handlers.validation import ValidationHandlersMixin

# Import settings for feature flags
try:
    from aragora.config.settings import get_settings

    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

    def get_settings() -> None:  # type: ignore[misc]
        return None


# Import metrics (optional - graceful fallback if not available)
try:
    from aragora.server.prometheus_cross_pollination import (
        record_event_dispatched,
        record_handler_call,
        set_circuit_breaker_state,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_event_dispatched(event_type: str) -> None:
        pass

    def record_handler_call(handler: str, status: str, duration: float) -> None:
        pass

    def set_circuit_breaker_state(handler: str, state: str) -> None:
        pass


# Import SLO metrics (optional - graceful fallback)
try:
    from aragora.observability.metrics.slo import check_and_record_slo

    SLO_METRICS_AVAILABLE = True
except ImportError:
    SLO_METRICS_AVAILABLE = False

    def check_and_record_slo(  # type: ignore[misc]
        operation: str, latency_ms: float, percentile: str = "p99"
    ) -> tuple[bool, str]:
        return True, f"SLO metrics not available for {operation}"


logger = logging.getLogger(__name__)


def _default_async_event_types() -> set:
    """Default event types for async dispatch."""
    return {
        StreamEventType.MEMORY_STORED,
        StreamEventType.MEMORY_RETRIEVED,
        StreamEventType.KNOWLEDGE_QUERIED,
        StreamEventType.RLM_COMPRESSION_COMPLETE,
    }


class CrossSubscriberManager(
    BasicHandlersMixin,
    KnowledgeMoundHandlersMixin,
    CultureHandlersMixin,
    ValidationHandlersMixin,
):
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

        # Culture storage dict for CultureHandlersMixin
        self._debate_cultures: dict = {}

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
        # Bidirectional Knowledge Mound Handlers
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

    def connect(self, event_emitter: Any) -> None:
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
                continue

            # Check filter
            if name in self._filters:
                filter_func = self._filters[name]
                try:
                    if not filter_func(event):
                        if name in self._stats:
                            self._stats[name].events_skipped += 1
                        continue
                except Exception as e:
                    logger.warning(f"Filter error for {name}: {e}")

            # Apply sampling
            sample_rate = self._stats[name].sample_rate if name in self._stats else 1.0
            if sample_rate < 1.0 and random.random() > sample_rate:
                if name in self._stats:
                    self._stats[name].events_skipped += 1
                continue

            # Get retry config
            retry_config = (
                self._stats[name].retry_config
                if name in self._stats and self._stats[name].retry_config
                else self._default_retry_config
            )

            # Execute handler with timing, retry, and metrics
            start_time = time.time()
            success = False
            retries = 0

            while retries <= retry_config.max_retries:
                try:
                    handler(event)
                    success = True
                    break
                except Exception as e:
                    retries += 1
                    if retries <= retry_config.max_retries:
                        # Calculate backoff delay
                        delay_ms = min(
                            retry_config.base_delay_ms * (2 ** (retries - 1)),
                            retry_config.max_delay_ms,
                        )
                        time.sleep(delay_ms / 1000.0)
                        logger.debug(
                            f"Retrying handler {name} (attempt {retries}/{retry_config.max_retries})"
                        )
                    else:
                        logger.warning(
                            f"Handler {name} failed after {retry_config.max_retries} retries: {e}"
                        )
                        self._circuit_breaker.record_failure(name)
                        if METRICS_AVAILABLE:
                            set_circuit_breaker_state(name, self._circuit_breaker.get_state(name))

            elapsed_ms = (time.time() - start_time) * 1000

            # Update stats
            if name in self._stats:
                stats = self._stats[name]
                if success:
                    stats.events_processed += 1
                    self._circuit_breaker.record_success(name)
                else:
                    stats.events_failed += 1

                if retries > 0:
                    stats.events_retried += retries

                stats.last_event_time = datetime.now()

                # Update latency stats
                stats.total_latency_ms += elapsed_ms
                stats.min_latency_ms = min(stats.min_latency_ms, elapsed_ms)
                stats.max_latency_ms = max(stats.max_latency_ms, elapsed_ms)

                # Record latency for percentile calculation
                stats.record_latency(elapsed_ms)

            # Record handler metrics
            if METRICS_AVAILABLE:
                record_handler_call(
                    name,
                    "success" if success else "failure",
                    elapsed_ms,
                )

            # Check SLO
            if SLO_METRICS_AVAILABLE:
                check_and_record_slo(
                    f"cross_subscriber_{name}",
                    elapsed_ms,
                    percentile="p99",
                )

    def dispatch_async(self, event: StreamEvent) -> None:
        """Dispatch event asynchronously (adds to batch if batching enabled).

        Args:
            event: The event to dispatch
        """
        if self._async_config.enable_batching and event.type in _default_async_event_types():
            self._add_to_batch(event)
        else:
            self._dispatch_event(event)

    def dispatch(self, event: StreamEvent) -> None:
        """Dispatch event synchronously.

        Args:
            event: The event to dispatch
        """
        self._dispatch_event(event)

    def _add_to_batch(self, event: StreamEvent) -> None:
        """Add event to batch queue for later processing.

        Args:
            event: The event to batch
        """
        if event.type not in self._event_batch:
            self._event_batch[event.type] = []

        self._event_batch[event.type].append(event)

        # Check if batch should be flushed
        batch = self._event_batch[event.type]
        if len(batch) >= self._async_config.batch_size:
            self._flush_batch(event.type)
        elif time.time() - self._batch_last_flush >= self._async_config.batch_timeout_seconds:
            self._flush_batch(event.type)

    def _flush_batch(self, event_type: StreamEventType) -> None:
        """Flush a batch of events for processing.

        Args:
            event_type: The type of events to flush
        """
        if event_type not in self._event_batch:
            return

        batch = self._event_batch[event_type]
        if not batch:
            return

        # Process all events in batch
        for event in batch:
            self._dispatch_event(event)

        self._event_batch[event_type] = []
        self._batch_last_flush = time.time()

    def flush_all_batches(self) -> int:
        """Flush all pending batches.

        Returns:
            Number of events flushed
        """
        total = 0
        for event_type in list(self._event_batch.keys()):
            batch_size = len(self._event_batch.get(event_type, []))
            self._flush_batch(event_type)
            total += batch_size
        return total

    def get_batch_stats(self) -> dict:
        """Get statistics about current batch queues.

        Returns:
            Dict with batch queue statistics
        """
        return {
            "queues": {
                event_type.value: len(events) for event_type, events in self._event_batch.items()
            },
            "total_pending": sum(len(e) for e in self._event_batch.values()),
            "last_flush": self._batch_last_flush,
            "batch_size": self._async_config.batch_size,
            "timeout_seconds": self._async_config.batch_timeout_seconds,
            "batching_enabled": self._async_config.enable_batching,
        }

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
                "last_event": (
                    stats.last_event_time.isoformat() if stats.last_event_time else None
                ),
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
                    "state": cb_state,
                    "available": cb_available,
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
        """Reset all subscriber statistics."""
        for stats in self._stats.values():
            stats.events_processed = 0
            stats.events_failed = 0
            stats.events_skipped = 0
            stats.events_retried = 0
            stats.last_event_time = None
            stats.total_latency_ms = 0.0
            stats.avg_latency_ms = 0.0
            stats.min_latency_ms = float("inf")
            stats.max_latency_ms = 0.0
            stats.p50_latency_ms = None
            stats.p90_latency_ms = None
            stats.p99_latency_ms = None
            stats.latency_samples.clear()

    def reset_circuit_breaker(self, name: str) -> bool:
        """Reset circuit breaker for a specific handler.

        Args:
            name: Handler name

        Returns:
            True if reset successful
        """
        try:
            self._circuit_breaker.reset(name)
            if METRICS_AVAILABLE:
                set_circuit_breaker_state(name, "closed")
            return True
        except Exception:
            return False

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for name in self._stats:
            self.reset_circuit_breaker(name)

    def set_sample_rate(self, name: str, rate: float) -> bool:
        """Set sampling rate for a subscriber.

        Args:
            name: Subscriber name
            rate: Sample rate (0.0 to 1.0)

        Returns:
            True if set successfully
        """
        if name not in self._stats:
            return False
        if not 0.0 <= rate <= 1.0:
            return False
        self._stats[name].sample_rate = rate
        return True

    def set_filter(
        self,
        name: str,
        filter_func: Callable[[StreamEvent], bool],
    ) -> bool:
        """Set a filter function for a subscriber.

        Args:
            name: Subscriber name
            filter_func: Function that returns True if event should be processed

        Returns:
            True if set successfully
        """
        if name not in self._stats:
            return False
        self._filters[name] = filter_func
        return True

    def get_filter(self, name: str) -> Optional[Callable[[StreamEvent], bool]]:
        """Get the filter function for a subscriber."""
        return self._filters.get(name)

    def set_retry_config(
        self,
        name: str,
        max_retries: Optional[int] = None,
        base_delay_ms: Optional[int] = None,
        max_delay_ms: Optional[int] = None,
    ) -> bool:
        """Set retry configuration for a specific subscriber.

        Args:
            name: Subscriber name
            max_retries: Maximum retry attempts
            base_delay_ms: Base delay between retries
            max_delay_ms: Maximum delay between retries

        Returns:
            True if configuration was set
        """
        if name not in self._stats:
            return False

        stats = self._stats[name]

        # Create or update retry config
        if stats.retry_config is None:
            stats.retry_config = RetryConfig()

        if max_retries is not None:
            stats.retry_config.max_retries = max_retries
        if base_delay_ms is not None:
            stats.retry_config.base_delay_ms = base_delay_ms
        if max_delay_ms is not None:
            stats.retry_config.max_delay_ms = max_delay_ms

        return True

    def disable_retry(self, name: str) -> bool:
        """Disable retries for a specific subscriber.

        Args:
            name: Subscriber name

        Returns:
            True if disabled successfully
        """
        if name not in self._stats:
            return False

        self._stats[name].retry_config = RetryConfig(max_retries=0)
        return True

    def get_performance_report(self) -> dict:
        """Get a performance report summarizing all handlers.

        Returns:
            Dict with summary statistics
        """
        stats = self.get_stats()

        # Calculate totals
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
