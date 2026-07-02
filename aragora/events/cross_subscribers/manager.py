"""
Cross-Subscriber Manager.

Core CrossSubscriberManager class that orchestrates event dispatch
and subscriber lifecycle management.

The heavy lifting is delegated to specialized mixins:
- DispatchMixin: Event dispatch, batching, retry, circuit breaker, metrics
- AdminMixin: Stats reporting, enable/disable, sampling, filtering, retry config
- BasicHandlersMixin: Core subsystem event handlers
- CultureHandlersMixin: Culture pattern handlers
- StrategicHandlersMixin: Strategic feedback loop handlers (risk, genesis, budget, alerts)
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
from .handlers.strategic import StrategicHandlersMixin
from .registry import get_registered_subscribers

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
    CultureHandlersMixin,
    StrategicHandlersMixin,
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

        # Registry subscribers already wired into this manager instance
        self._applied_subscribers: set[str] = set()

        # Register built-in cross-subsystem handlers
        self._register_builtin_subscribers()

        # Registry subscribers are wired only by explicit layered bootstraps.
        # Direct construction stays infrastructure-only, so relocated domain
        # reactions cannot appear or disappear based on prior import order.

    def apply_registered_subscribers(self) -> int:
        """Wire registry subscribers not yet applied into this manager.

        Home modules (domain/application/interface) self-register their
        subscribers via ``aragora.events.cross_subscribers.register_subscriber``;
        this applies them by delegating to each subscriber's ``register`` method,
        reusing the existing per-event dispatch/stats/retry machinery. Idempotent:
        each subscriber is applied at most once per manager instance.

        Returns:
            The number of subscribers newly applied by this call.
        """
        applied = 0
        for name, subscriber in get_registered_subscribers().items():
            if name in self._applied_subscribers:
                continue
            subscriber.register(self)
            self._applied_subscribers.add(name)
            applied += 1
        return applied

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

        # Bidirectional Knowledge Mound reactions relocated to their domain home
        # (aragora.knowledge.event_subscribers, P4a E2 relocate-UP): they self-register
        # via that module and are wired by apply_registered_subscribers at bootstrap,
        # so events/ no longer imports knowledge here.

        # Phase 6: Culture → Debate (pattern updates)
        self.register(
            "culture_to_debate",
            StreamEventType.MOUND_UPDATED,
            self._handle_culture_to_debate,
        )

        # Phase 6b: Debate Start → Load Culture (active retrieval) relocated to
        # aragora.knowledge.event_subscribers (P4a Batch E2c); wired at bootstrap
        # via apply_registered_subscribers, not registered here.

        # Phase 7: Staleness → Debate
        self.register(
            "staleness_to_debate",
            StreamEventType.KNOWLEDGE_STALE,
            self._handle_staleness_to_debate,
        )

        # Explainability: Debate End → Explanation auto-trigger
        self.register(
            "debate_end_to_explainability",
            StreamEventType.DEBATE_END,
            self._handle_debate_end_to_explainability,
        )

        # Knowledge: Debate End → Outcome persistence relocated to
        # aragora.knowledge.event_subscribers (P4a Batch E2c); wired at bootstrap
        # via apply_registered_subscribers, not registered here.

        # =====================================================================
        # Phase 3: Cross-Subsystem Event Bridges
        # =====================================================================

        # Gauntlet Complete → Notification
        self.register(
            "gauntlet_to_notification",
            StreamEventType.GAUNTLET_COMPLETE,
            self._handle_gauntlet_complete_to_notification,
        )

        # Debate End → Cost Tracking
        self.register(
            "debate_end_to_cost_tracking",
            StreamEventType.DEBATE_END,
            self._handle_debate_end_to_cost_tracking,
        )

        # Consensus → Selection Learning
        self.register(
            "consensus_to_learning",
            StreamEventType.CONSENSUS,
            self._handle_consensus_to_learning,
        )

        # Agent Message → Rhetorical Analysis
        self.register(
            "agent_message_to_rhetorical",
            StreamEventType.AGENT_MESSAGE,
            self._handle_agent_message_to_rhetorical,
        )

        # Vote → Belief Network
        self.register(
            "vote_to_belief",
            StreamEventType.VOTE,
            self._handle_vote_to_belief,
        )

        # Workflow Complete/Failed → Supermemory and Memory Tier
        # Demotion/Promotion → Knowledge Mound all relocated to
        # aragora.knowledge.event_subscribers (P4a Batch E2c); wired at bootstrap
        # via apply_registered_subscribers, not registered here.

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

        # =====================================================================
        # Strategic Feedback Loops (Tier 5)
        # =====================================================================

        # Risk Warning → Health Registry degradation
        self.register(
            "risk_warning_to_health",
            StreamEventType.RISK_WARNING,
            self._handle_risk_warning_to_health,
        )

        # Agent Birth → Control Plane Registry
        self.register(
            "agent_birth_to_control_plane",
            StreamEventType.AGENT_BIRTH,
            self._handle_genesis_to_control_plane,
        )

        # Agent Death → Control Plane Registry
        self.register(
            "agent_death_to_control_plane",
            StreamEventType.AGENT_DEATH,
            self._handle_genesis_to_control_plane,
        )

        # Agent Evolution → Control Plane Registry
        self.register(
            "agent_evolution_to_control_plane",
            StreamEventType.AGENT_EVOLUTION,
            self._handle_genesis_to_control_plane,
        )

        # Approval Approved → KM Reinforcement relocated to
        # aragora.knowledge.event_subscribers (P4a Batch E2c); wired at bootstrap
        # via apply_registered_subscribers, not registered here.

        # Budget Alert → Team Selection Constraint
        self.register(
            "budget_alert_to_team_selection",
            StreamEventType.BUDGET_ALERT,
            self._handle_budget_alert_to_team_selection,
        )

        # Alert Escalated → Workflow Emergency Brake
        self.register(
            "alert_escalated_to_workflow_brake",
            StreamEventType.ALERT_ESCALATED,
            self._handle_alert_escalated_to_workflow_brake,
        )

        # Meta-Learning Adjusted → Team Selection Recalibration
        self.register(
            "meta_learning_to_team_selection",
            StreamEventType.META_LEARNING_ADJUSTED,
            self._handle_meta_learning_to_team_selection,
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

        logger.debug("Registered subscriber '%s' for %s", name, event_type.value)

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
