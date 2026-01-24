"""
Hook handler registry for wiring subsystems to HookManager.

This module provides default handlers that connect various Aragora subsystems
to the hook lifecycle system, enabling automatic data flow between components.

Usage:
    from aragora.debate.hook_handlers import HookHandlerRegistry

    registry = HookHandlerRegistry(hook_manager, subsystems={
        "analytics": analytics_coordinator,
        "continuum_memory": continuum,
        "calibration_tracker": calibration,
        "outcome_tracker": outcome_tracker,
    })
    registry.register_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.debate.hooks import HookManager, HookPriority

logger = logging.getLogger(__name__)


@dataclass
class HookHandlerRegistry:
    """Registry of default hook handlers that wire subsystems to HookManager.

    This registry provides automatic wiring between the hook lifecycle
    and various subsystems like analytics, memory, calibration, and outcomes.

    The registry follows these principles:
    1. **Graceful degradation**: Missing subsystems are skipped without error
    2. **Priority ordering**: Critical handlers run before logging handlers
    3. **Error isolation**: Failed handlers don't affect other hooks

    Subsystems supported:
    - analytics: AnalyticsCoordinator for debate metrics
    - continuum_memory: ContinuumMemory for cross-debate learning
    - consensus_memory: ConsensusMemory for outcome storage
    - calibration_tracker: CalibrationTracker for prediction accuracy
    - outcome_tracker: OutcomeTracker for debate outcomes
    - performance_monitor: AgentPerformanceMonitor for timing metrics
    - elo_system: EloSystem for agent rankings
    - trickster: Trickster for hollow consensus detection
    - selection_feedback: SelectionFeedbackLoop for agent selection weights
    - knowledge_mound: Knowledge Mound for organizational knowledge
    - km_coordinator: BidirectionalCoordinator for KM sync
    - webhook_delivery: WebhookDeliveryManager for outbound webhook dispatch
    """

    hook_manager: "HookManager"
    subsystems: Dict[str, Any] = field(default_factory=dict)

    # Track registered handlers for cleanup
    _unregister_fns: List[Callable[[], None]] = field(default_factory=list, repr=False)
    _registered: bool = field(default=False, repr=False)

    def register_all(self) -> int:
        """Register all default handlers for available subsystems.

        Returns:
            Number of handlers registered
        """
        if self._registered:
            logger.debug("HookHandlerRegistry already registered, skipping")
            return 0

        count = 0
        count += self._register_analytics_handlers()
        count += self._register_memory_handlers()
        count += self._register_calibration_handlers()
        count += self._register_outcome_handlers()
        count += self._register_performance_handlers()
        count += self._register_selection_handlers()
        count += self._register_detection_handlers()
        count += self._register_km_handlers()
        count += self._register_webhook_handlers()
        count += self._register_receipt_handlers()

        self._registered = True
        logger.info(f"HookHandlerRegistry registered {count} handlers")
        return count

    def unregister_all(self) -> int:
        """Unregister all handlers that were registered by this registry.

        Returns:
            Number of handlers unregistered
        """
        count = 0
        for unregister_fn in self._unregister_fns:
            try:
                unregister_fn()
                count += 1
            except Exception as e:
                logger.debug(f"Error unregistering handler: {e}")

        self._unregister_fns.clear()
        self._registered = False
        logger.debug(f"Unregistered {count} hook handlers")
        return count

    def _register(
        self,
        hook_type: str,
        callback: Callable[..., Any],
        name: str,
        priority: Optional["HookPriority"] = None,
    ) -> bool:
        """Register a handler and track for later cleanup.

        Returns:
            True if registration succeeded
        """
        try:
            from aragora.debate.hooks import HookPriority as HP

            unregister = self.hook_manager.register(
                hook_type,
                callback,
                name=name,
                priority=priority or HP.NORMAL,
            )
            self._unregister_fns.append(unregister)
            return True
        except Exception as e:
            logger.debug(f"Failed to register {name}: {e}")
            return False

    # =========================================================================
    # Analytics Handlers
    # =========================================================================

    def _register_analytics_handlers(self) -> int:
        """Wire hooks to AnalyticsCoordinator.

        Tracks:
        - Round completion metrics
        - Agent response timings
        - Debate completion stats
        """
        analytics = self.subsystems.get("analytics")
        if not analytics:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Round completion
        if hasattr(analytics, "on_round_complete"):

            def handle_round_complete(ctx: Any = None, round_num: int = 0, **kwargs: Any) -> None:
                try:
                    analytics.on_round_complete(ctx, round_num)
                except Exception as e:
                    logger.debug(f"Analytics round_complete handler failed: {e}")

            if self._register(
                HookType.POST_ROUND.value,
                handle_round_complete,
                "analytics_round_complete",
                HookPriority.NORMAL,
            ):
                count += 1

        # Agent response
        if hasattr(analytics, "on_agent_response"):

            def handle_agent_response(agent: Any = None, response: str = "", **kwargs: Any) -> None:
                try:
                    analytics.on_agent_response(agent, response)
                except Exception as e:
                    logger.debug(f"Analytics agent_response handler failed: {e}")

            if self._register(
                HookType.POST_GENERATE.value,
                handle_agent_response,
                "analytics_agent_response",
                HookPriority.LOW,
            ):
                count += 1

        # Debate completion
        if hasattr(analytics, "on_debate_complete"):

            def handle_debate_complete(ctx: Any = None, result: Any = None, **kwargs: Any) -> None:
                try:
                    analytics.on_debate_complete(ctx, result)
                except Exception as e:
                    logger.debug(f"Analytics debate_complete handler failed: {e}")

            if self._register(
                HookType.POST_DEBATE.value,
                handle_debate_complete,
                "analytics_debate_complete",
                HookPriority.NORMAL,
            ):
                count += 1

        return count

    # =========================================================================
    # Memory Handlers
    # =========================================================================

    def _register_memory_handlers(self) -> int:
        """Wire hooks to memory systems.

        Handles:
        - Continuum memory updates on debate end
        - Consensus memory on consensus reached
        """
        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Continuum memory
        continuum = self.subsystems.get("continuum_memory")
        if continuum:
            if hasattr(continuum, "on_debate_end"):

                def handle_continuum_debate_end(
                    ctx: Any = None, result: Any = None, **kwargs: Any
                ) -> None:
                    try:
                        continuum.on_debate_end(ctx, result)
                    except Exception as e:
                        logger.debug(f"Continuum on_debate_end failed: {e}")

                if self._register(
                    HookType.POST_DEBATE.value,
                    handle_continuum_debate_end,
                    "continuum_memory_debate_end",
                    HookPriority.HIGH,  # Memory writes are high priority
                ):
                    count += 1

        # Consensus memory
        consensus = self.subsystems.get("consensus_memory")
        if consensus:
            if hasattr(consensus, "on_consensus_reached"):

                def handle_consensus(
                    ctx: Any = None,
                    consensus_text: str = "",
                    confidence: float = 0.0,
                    **kwargs: Any,
                ) -> None:
                    try:
                        consensus.on_consensus_reached(ctx, consensus_text, confidence)
                    except Exception as e:
                        logger.debug(f"Consensus memory on_consensus failed: {e}")

                if self._register(
                    HookType.POST_CONSENSUS.value,
                    handle_consensus,
                    "consensus_memory_on_consensus",
                    HookPriority.HIGH,
                ):
                    count += 1

        return count

    # =========================================================================
    # Calibration Handlers
    # =========================================================================

    def _register_calibration_handlers(self) -> int:
        """Wire hooks to CalibrationTracker.

        Handles:
        - Recording predictions after votes
        - Updating calibration after debate outcomes
        """
        calibration = self.subsystems.get("calibration_tracker")
        if not calibration:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Record predictions after votes
        if hasattr(calibration, "on_vote"):

            def handle_vote(ctx: Any = None, vote: Any = None, **kwargs: Any) -> None:
                try:
                    calibration.on_vote(ctx, vote)
                except Exception as e:
                    logger.debug(f"Calibration on_vote failed: {e}")

            if self._register(
                HookType.POST_VOTE.value,
                handle_vote,
                "calibration_on_vote",
                HookPriority.NORMAL,
            ):
                count += 1

        # Update calibration after debate
        if hasattr(calibration, "on_debate_outcome"):

            def handle_outcome(ctx: Any = None, result: Any = None, **kwargs: Any) -> None:
                try:
                    calibration.on_debate_outcome(ctx, result)
                except Exception as e:
                    logger.debug(f"Calibration on_debate_outcome failed: {e}")

            if self._register(
                HookType.POST_DEBATE.value,
                handle_outcome,
                "calibration_on_outcome",
                HookPriority.NORMAL,
            ):
                count += 1

        return count

    # =========================================================================
    # Outcome Handlers
    # =========================================================================

    def _register_outcome_handlers(self) -> int:
        """Wire hooks to OutcomeTracker.

        Handles:
        - Recording outcomes at debate end
        - Tracking consensus convergence
        """
        outcome_tracker = self.subsystems.get("outcome_tracker")
        if not outcome_tracker:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Record debate outcome
        if hasattr(outcome_tracker, "record_outcome"):

            def handle_debate_outcome(ctx: Any = None, result: Any = None, **kwargs: Any) -> None:
                try:
                    outcome_tracker.record_outcome(ctx, result)
                except Exception as e:
                    logger.debug(f"OutcomeTracker record_outcome failed: {e}")

            if self._register(
                HookType.POST_DEBATE.value,
                handle_debate_outcome,
                "outcome_tracker_record",
                HookPriority.HIGH,  # Outcome recording is high priority
            ):
                count += 1

        # Track convergence
        if hasattr(outcome_tracker, "on_convergence"):

            def handle_convergence(ctx: Any = None, **kwargs: Any) -> None:
                try:
                    outcome_tracker.on_convergence(ctx)
                except Exception as e:
                    logger.debug(f"OutcomeTracker on_convergence failed: {e}")

            if self._register(
                HookType.ON_CONVERGENCE.value,
                handle_convergence,
                "outcome_tracker_convergence",
                HookPriority.NORMAL,
            ):
                count += 1

        return count

    # =========================================================================
    # Performance Handlers
    # =========================================================================

    def _register_performance_handlers(self) -> int:
        """Wire hooks to performance monitoring systems.

        Handles:
        - Agent response timing
        - Round timing metrics
        """
        performance = self.subsystems.get("performance_monitor")
        if not performance:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Track agent responses
        if hasattr(performance, "record_response"):

            def handle_response(
                agent: Any = None,
                response: str = "",
                latency_ms: float = 0.0,
                **kwargs: Any,
            ) -> None:
                try:
                    performance.record_response(agent, response, latency_ms)
                except Exception as e:
                    logger.debug(f"Performance record_response failed: {e}")

            if self._register(
                HookType.POST_GENERATE.value,
                handle_response,
                "performance_record_response",
                HookPriority.LOW,  # Telemetry is low priority
            ):
                count += 1

        # Track round timing
        if hasattr(performance, "record_round"):

            def handle_round(
                ctx: Any = None,
                round_num: int = 0,
                duration_ms: float = 0.0,
                **kwargs: Any,
            ) -> None:
                try:
                    performance.record_round(ctx, round_num, duration_ms)
                except Exception as e:
                    logger.debug(f"Performance record_round failed: {e}")

            if self._register(
                HookType.POST_ROUND.value,
                handle_round,
                "performance_record_round",
                HookPriority.LOW,
            ):
                count += 1

        return count

    # =========================================================================
    # Selection Feedback Handlers
    # =========================================================================

    def _register_selection_handlers(self) -> int:
        """Wire hooks to SelectionFeedbackLoop.

        Handles:
        - Recording debate outcomes for selection adjustment
        - Updating agent selection weights
        """
        feedback_loop = self.subsystems.get("selection_feedback")
        if not feedback_loop:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Update selection weights after debate
        if hasattr(feedback_loop, "record_debate_outcome"):

            def handle_feedback(ctx: Any = None, result: Any = None, **kwargs: Any) -> None:
                try:
                    feedback_loop.record_debate_outcome(ctx, result)
                except Exception as e:
                    logger.debug(f"SelectionFeedback record_outcome failed: {e}")

            if self._register(
                HookType.POST_DEBATE.value,
                handle_feedback,
                "selection_feedback_outcome",
                HookPriority.NORMAL,
            ):
                count += 1

        return count

    # =========================================================================
    # Detection Handlers
    # =========================================================================

    def _register_detection_handlers(self) -> int:
        """Wire hooks to detection systems (Trickster, FlipDetector, etc.).

        Handles:
        - Trickster hollow consensus detection at consensus
        - Flip detection at round boundaries
        """
        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Trickster
        trickster = self.subsystems.get("trickster")
        if trickster:
            if hasattr(trickster, "check_consensus"):

                def handle_consensus_check(
                    ctx: Any = None,
                    votes: List[Any] = None,
                    **kwargs: Any,
                ) -> None:
                    try:
                        votes = votes or []
                        trickster.check_consensus(ctx, votes)
                    except Exception as e:
                        logger.debug(f"Trickster check_consensus failed: {e}")

                if self._register(
                    HookType.PRE_CONSENSUS.value,
                    handle_consensus_check,
                    "trickster_consensus_check",
                    HookPriority.CRITICAL,  # Safety check is critical
                ):
                    count += 1

        # Flip detector
        flip_detector = self.subsystems.get("flip_detector")
        if flip_detector:
            if hasattr(flip_detector, "check_positions"):

                def handle_flip_check(
                    ctx: Any = None,
                    round_num: int = 0,
                    positions: Dict[str, str] = None,
                    **kwargs: Any,
                ) -> None:
                    try:
                        positions = positions or {}
                        flip_detector.check_positions(ctx, round_num, positions)
                    except Exception as e:
                        logger.debug(f"FlipDetector check_positions failed: {e}")

                if self._register(
                    HookType.POST_ROUND.value,
                    handle_flip_check,
                    "flip_detector_check",
                    HookPriority.NORMAL,
                ):
                    count += 1

        return count

    # =========================================================================
    # Knowledge Mound Handlers
    # =========================================================================

    def _register_km_handlers(self) -> int:
        """Wire hooks to Knowledge Mound bidirectional sync.

        Handles:
        - Storing debate knowledge at debate end
        - Triggering validation on consensus
        - Recording outcomes for KM patterns
        - Running bidirectional sync periodically
        """
        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Knowledge Mound
        km = self.subsystems.get("knowledge_mound")
        if km:
            # Store debate knowledge at debate end
            if hasattr(km, "on_debate_end"):

                def handle_km_debate_end(
                    ctx: Any = None,
                    result: Any = None,
                    **kwargs: Any,
                ) -> None:
                    try:
                        km.on_debate_end(ctx, result)
                    except Exception as e:
                        logger.debug(f"KM on_debate_end failed: {e}")

                if self._register(
                    HookType.POST_DEBATE.value,
                    handle_km_debate_end,
                    "km_debate_end",
                    HookPriority.NORMAL,
                ):
                    count += 1

            # Trigger validation on consensus
            if hasattr(km, "on_consensus_reached"):

                def handle_km_consensus(
                    ctx: Any = None,
                    consensus_text: str = "",
                    confidence: float = 0.0,
                    **kwargs: Any,
                ) -> None:
                    try:
                        km.on_consensus_reached(ctx, consensus_text, confidence)
                    except Exception as e:
                        logger.debug(f"KM on_consensus_reached failed: {e}")

                if self._register(
                    HookType.POST_CONSENSUS.value,
                    handle_km_consensus,
                    "km_consensus",
                    HookPriority.NORMAL,
                ):
                    count += 1

            # Record outcome for KM pattern tracking
            if hasattr(km, "on_outcome_tracked"):

                def handle_km_outcome(
                    ctx: Any = None,
                    outcome: Any = None,
                    **kwargs: Any,
                ) -> None:
                    try:
                        km.on_outcome_tracked(ctx, outcome)
                    except Exception as e:
                        logger.debug(f"KM on_outcome_tracked failed: {e}")

                if self._register(
                    HookType.POST_DEBATE.value,
                    handle_km_outcome,
                    "km_outcome_tracked",
                    HookPriority.LOW,  # After main outcome recording
                ):
                    count += 1

        # Bidirectional Coordinator
        km_coordinator = self.subsystems.get("km_coordinator")
        if km_coordinator:
            # Trigger bidirectional sync after debate
            if hasattr(km_coordinator, "on_debate_complete"):

                def handle_coord_sync(
                    ctx: Any = None,
                    result: Any = None,
                    **kwargs: Any,
                ) -> None:
                    try:
                        km_coordinator.on_debate_complete(ctx, result)
                    except Exception as e:
                        logger.debug(f"KM coordinator on_debate_complete failed: {e}")

                if self._register(
                    HookType.POST_DEBATE.value,
                    handle_coord_sync,
                    "km_coordinator_sync",
                    HookPriority.LOW,  # Sync is low priority
                ):
                    count += 1

            # Run validation sync after consensus
            if hasattr(km_coordinator, "on_consensus_reached"):

                def handle_coord_consensus(
                    ctx: Any = None,
                    consensus_text: str = "",
                    confidence: float = 0.0,
                    **kwargs: Any,
                ) -> None:
                    try:
                        km_coordinator.on_consensus_reached(ctx, consensus_text, confidence)
                    except Exception as e:
                        logger.debug(f"KM coordinator on_consensus failed: {e}")

                if self._register(
                    HookType.POST_CONSENSUS.value,
                    handle_coord_consensus,
                    "km_coordinator_consensus",
                    HookPriority.LOW,
                ):
                    count += 1

        return count

    # =========================================================================
    # Webhook Delivery Handlers
    # =========================================================================

    def _register_webhook_handlers(self) -> int:
        """Wire hooks to webhook delivery system.

        Handles:
        - Delivering webhooks on debate end
        - Delivering webhooks on consensus reached
        - Delivering webhooks on round completion
        """
        webhook_delivery = self.subsystems.get("webhook_delivery")
        if not webhook_delivery:
            return 0

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        # Debate end webhook
        def handle_debate_end_webhook(
            ctx: Any = None,
            result: Any = None,
            **kwargs: Any,
        ) -> None:
            try:
                import asyncio
                from aragora.storage.webhook_config_store import get_webhook_config_store

                # Get debate info from context
                debate_id = getattr(ctx, "debate_id", None) if ctx else None
                task = getattr(ctx, "task", "") if ctx else ""

                # Build payload
                payload = {
                    "event": "debate_end",
                    "debate_id": debate_id,
                    "task": task[:200] if task else None,
                    "timestamp": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .isoformat(),
                }

                # Add result info if available
                if result:
                    if hasattr(result, "to_dict"):
                        payload["result"] = result.to_dict()
                    elif hasattr(result, "consensus"):
                        payload["consensus"] = result.consensus
                        payload["verdict"] = getattr(result, "verdict", None)
                    elif isinstance(result, dict):
                        payload["result"] = result

                # Get webhooks configured for this event
                store = get_webhook_config_store()
                webhooks = store.get_for_event("debate_end")

                # Deliver to each webhook
                async def deliver_all():
                    from aragora.server.webhook_delivery import get_delivery_manager

                    manager = await get_delivery_manager()

                    for webhook in webhooks:
                        try:
                            await manager.deliver(
                                webhook_id=webhook.id,
                                event_type="debate_end",
                                payload=payload,
                                url=webhook.url,
                                secret=webhook.secret,
                            )
                            # Record delivery in config store
                            store.record_delivery(webhook.id, 200, success=True)
                        except Exception as e:
                            logger.debug(f"Webhook delivery failed for {webhook.id}: {e}")
                            store.record_delivery(webhook.id, 500, success=False)

                # Run async delivery
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(deliver_all())
                    else:
                        loop.run_until_complete(deliver_all())
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(deliver_all())

            except Exception as e:
                logger.debug(f"Webhook debate_end handler failed: {e}")

        if self._register(
            HookType.POST_DEBATE.value,
            handle_debate_end_webhook,
            "webhook_debate_end",
            HookPriority.LOW,  # Webhooks are lower priority than core functionality
        ):
            count += 1

        # Consensus webhook
        def handle_consensus_webhook(
            ctx: Any = None,
            consensus_text: str = "",
            confidence: float = 0.0,
            **kwargs: Any,
        ) -> None:
            try:
                import asyncio
                from aragora.storage.webhook_config_store import get_webhook_config_store

                debate_id = getattr(ctx, "debate_id", None) if ctx else None

                payload = {
                    "event": "consensus",
                    "debate_id": debate_id,
                    "consensus_text": consensus_text[:500] if consensus_text else None,
                    "confidence": confidence,
                    "timestamp": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .isoformat(),
                }

                store = get_webhook_config_store()
                webhooks = store.get_for_event("consensus")

                async def deliver_all():
                    from aragora.server.webhook_delivery import get_delivery_manager

                    manager = await get_delivery_manager()

                    for webhook in webhooks:
                        try:
                            await manager.deliver(
                                webhook_id=webhook.id,
                                event_type="consensus",
                                payload=payload,
                                url=webhook.url,
                                secret=webhook.secret,
                            )
                            store.record_delivery(webhook.id, 200, success=True)
                        except Exception as e:
                            logger.debug(f"Webhook delivery failed for {webhook.id}: {e}")
                            store.record_delivery(webhook.id, 500, success=False)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(deliver_all())
                    else:
                        loop.run_until_complete(deliver_all())
                except RuntimeError:
                    asyncio.run(deliver_all())

            except Exception as e:
                logger.debug(f"Webhook consensus handler failed: {e}")

        if self._register(
            HookType.POST_CONSENSUS.value,
            handle_consensus_webhook,
            "webhook_consensus",
            HookPriority.LOW,
        ):
            count += 1

        logger.debug(f"Registered {count} webhook handlers")
        return count

    # =========================================================================
    # Decision Receipt Handlers
    # =========================================================================

    def _register_receipt_handlers(self) -> int:
        """Wire hooks to decision receipt generation.

        Handles:
        - Auto-generating decision receipts after high-confidence debates
        - Persisting receipts to configured store
        - Optional auto-signing with HMAC-SHA256
        """
        # Check if receipt generation is enabled in config
        arena_config = self.subsystems.get("arena_config")
        if not arena_config:
            return 0

        enable_receipt = getattr(arena_config, "enable_receipt_generation", False)
        if not enable_receipt:
            return 0

        min_confidence = getattr(arena_config, "receipt_min_confidence", 0.6)
        auto_sign = getattr(arena_config, "receipt_auto_sign", False)
        receipt_store = getattr(arena_config, "receipt_store", None) or self.subsystems.get(
            "receipt_store"
        )

        count = 0
        from aragora.debate.hooks import HookType, HookPriority

        def handle_receipt_generation(
            ctx: Any = None,
            result: Any = None,
            **kwargs: Any,
        ) -> None:
            """Generate decision receipt after debate completion."""
            try:
                # Skip if no result or low confidence
                if not result:
                    return

                confidence = getattr(result, "confidence", 0.0)
                if confidence < min_confidence:
                    logger.debug(
                        f"Skipping receipt generation: confidence {confidence:.2f} < {min_confidence}"
                    )
                    return

                # Import receipt module
                from aragora.gauntlet.receipt import DecisionReceipt

                # Generate receipt from debate result
                receipt = DecisionReceipt.from_debate_result(result)
                logger.info(
                    f"Generated decision receipt {receipt.receipt_id} for debate "
                    f"{receipt.gauntlet_id} (confidence: {confidence:.2f})"
                )

                # Optional signing
                if auto_sign:
                    try:
                        from aragora.gauntlet.signing import sign_receipt

                        signed = sign_receipt(receipt.to_dict())
                        logger.debug(f"Auto-signed receipt {receipt.receipt_id}")

                        # If we have a store, save the signed version
                        if receipt_store and hasattr(receipt_store, "save_signed"):
                            receipt_store.save_signed(signed)
                            return
                    except (ImportError, ValueError) as sign_err:
                        logger.warning(f"Auto-signing failed: {sign_err}")

                # Persist to store if available
                if receipt_store:
                    try:
                        if hasattr(receipt_store, "save"):
                            receipt_store.save(receipt)
                            logger.debug(f"Persisted receipt {receipt.receipt_id} to store")
                    except Exception as store_err:
                        logger.warning(f"Failed to persist receipt: {store_err}")

            except ImportError as e:
                logger.debug(f"Receipt generation unavailable: {e}")
            except Exception as e:
                logger.warning(f"Receipt generation failed: {e}")

        if self._register(
            HookType.POST_DEBATE.value,
            handle_receipt_generation,
            "receipt_generation",
            HookPriority.LOW,  # Run after other post-debate handlers
        ):
            count += 1
            logger.debug("Registered decision receipt generation handler")

        return count

    @property
    def registered_count(self) -> int:
        """Number of handlers currently registered."""
        return len(self._unregister_fns)

    @property
    def is_registered(self) -> bool:
        """Whether handlers have been registered."""
        return self._registered


def create_hook_handler_registry(
    hook_manager: "HookManager",
    *,
    analytics: Any = None,
    continuum_memory: Any = None,
    consensus_memory: Any = None,
    calibration_tracker: Any = None,
    outcome_tracker: Any = None,
    performance_monitor: Any = None,
    selection_feedback: Any = None,
    trickster: Any = None,
    flip_detector: Any = None,
    knowledge_mound: Any = None,
    km_coordinator: Any = None,
    webhook_delivery: Any = None,
    arena_config: Any = None,
    receipt_store: Any = None,
    auto_register: bool = True,
) -> HookHandlerRegistry:
    """Create and optionally register a HookHandlerRegistry.

    Args:
        hook_manager: The HookManager to wire handlers to
        analytics: AnalyticsCoordinator for debate metrics
        continuum_memory: ContinuumMemory for cross-debate learning
        consensus_memory: ConsensusMemory for outcome storage
        calibration_tracker: CalibrationTracker for prediction accuracy
        outcome_tracker: OutcomeTracker for debate outcomes
        performance_monitor: AgentPerformanceMonitor for timing
        selection_feedback: SelectionFeedbackLoop for selection weights
        trickster: Trickster for hollow consensus detection
        flip_detector: FlipDetector for position reversals
        knowledge_mound: Knowledge Mound for organizational knowledge
        km_coordinator: BidirectionalCoordinator for KM sync
        webhook_delivery: WebhookDeliveryManager for webhook dispatch
        arena_config: ArenaConfig for receipt generation settings
        receipt_store: Receipt store for persisting decision receipts
        auto_register: If True, automatically call register_all()

    Returns:
        Configured HookHandlerRegistry instance
    """
    subsystems: Dict[str, Any] = {}

    if analytics is not None:
        subsystems["analytics"] = analytics
    if continuum_memory is not None:
        subsystems["continuum_memory"] = continuum_memory
    if consensus_memory is not None:
        subsystems["consensus_memory"] = consensus_memory
    if calibration_tracker is not None:
        subsystems["calibration_tracker"] = calibration_tracker
    if outcome_tracker is not None:
        subsystems["outcome_tracker"] = outcome_tracker
    if performance_monitor is not None:
        subsystems["performance_monitor"] = performance_monitor
    if selection_feedback is not None:
        subsystems["selection_feedback"] = selection_feedback
    if trickster is not None:
        subsystems["trickster"] = trickster
    if flip_detector is not None:
        subsystems["flip_detector"] = flip_detector
    if knowledge_mound is not None:
        subsystems["knowledge_mound"] = knowledge_mound
    if km_coordinator is not None:
        subsystems["km_coordinator"] = km_coordinator
    if webhook_delivery is not None:
        subsystems["webhook_delivery"] = webhook_delivery
    if arena_config is not None:
        subsystems["arena_config"] = arena_config
    if receipt_store is not None:
        subsystems["receipt_store"] = receipt_store

    registry = HookHandlerRegistry(hook_manager=hook_manager, subsystems=subsystems)

    if auto_register:
        registry.register_all()

    return registry


__all__ = ["HookHandlerRegistry", "create_hook_handler_registry"]
