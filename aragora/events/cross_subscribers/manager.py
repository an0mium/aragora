"""
Cross-Subscriber Manager.

Core CrossSubscriberManager class that orchestrates event dispatch
and subscriber lifecycle management.

The heavy lifting is delegated to specialized mixins:
- DispatchMixin: Event dispatch, batching, retry, circuit breaker, metrics
- AdminMixin: Stats reporting, enable/disable, sampling, filtering, retry config
- BasicHandlersMixin: Core subsystem event handlers
- KnowledgeMoundHandlersMixin: Bidirectional KM event handlers
- CultureHandlersMixin: Culture pattern handlers
- ValidationHandlersMixin: Consensus and validation handlers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.events.subscribers.config import (
    AsyncDispatchConfig,
    RetryConfig,
    SubscriberStats,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.resilience import CircuitBreaker

from .admin import AdminMixin
from .dispatch import DispatchMixin
from .handlers.basic import BasicHandlersMixin
from .handlers.culture import CultureHandlersMixin
from .handlers.knowledge_mound import KnowledgeMoundHandlersMixin
from .handlers.validation import ValidationHandlersMixin

if TYPE_CHECKING:
    from aragora.config.settings import Settings

# Import settings for feature flags
try:
    from aragora.config.settings import get_settings as _get_settings

    SETTINGS_AVAILABLE = True

    def get_settings() -> Settings | None:
        """Get settings instance (wrapper for type safety)."""
        return _get_settings()

except ImportError:
    SETTINGS_AVAILABLE = False

    def get_settings() -> Settings | None:
        """Fallback when settings module not available."""
        return None



logger = logging.getLogger(__name__)


class CrossSubscriberManager(
    DispatchMixin,
    AdminMixin,
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
        default_retry_config: RetryConfig | None = None,
        async_config: AsyncDispatchConfig | None = None,
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
