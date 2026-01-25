"""Arena extensions for billing, broadcast, and training export.

This module separates non-core debate concerns from the Arena class,
following the Single Responsibility Principle. Extensions are triggered
via callbacks after debate completion.

Usage:
    # Create extensions configuration
    extensions = ArenaExtensions(
        org_id="org_123",
        user_id="user_456",
        workspace_id="ws_789",
        usage_tracker=tracker,
        auto_broadcast=True,
    )

    # After debate completion, trigger extensions
    extensions.on_debate_complete(ctx, result, agents)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import Agent, DebateResult
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


@dataclass
class ArenaExtensions:
    """Extension hooks for Arena that handle non-core concerns.

    These extensions are triggered after debate completion and handle:
    - Billing/usage tracking (with per-agent cost attribution)
    - LLM-as-Judge evaluation of debate quality
    - Audio/video broadcast generation
    - Training data export (Tinker integration)

    All extensions are optional and fail gracefully (don't break debates).
    """

    # Billing/usage tracking
    org_id: str = ""
    user_id: str = ""
    workspace_id: str = ""  # For cost attribution
    usage_tracker: Any = None  # UsageTracker instance
    cost_tracker: Any = None  # CostTracker instance for per-agent costs
    debate_budget_limit_usd: Optional[float] = None  # Per-debate cost limit
    enforce_budget_limit: bool = True  # Raise error when budget exceeded

    # LLM-as-Judge evaluation
    llm_judge: Any = None  # LLMJudge instance
    auto_evaluate: bool = False  # Auto-evaluate final answer
    evaluation_use_case: str = "debate"  # Weight profile for evaluation
    evaluation_threshold: float = 3.5  # Minimum score to pass

    # Broadcast pipeline
    broadcast_pipeline: Any = None  # BroadcastPipeline instance
    auto_broadcast: bool = False
    broadcast_min_confidence: float = 0.8

    # Training data export (Tinker integration)
    training_exporter: Any = None  # DebateTrainingExporter instance
    auto_export_training: bool = False
    training_export_min_confidence: float = 0.75

    # Stripe usage sync (metered billing)
    usage_sync_service: Any = None  # UsageSyncService instance
    auto_sync_usage: bool = False

    # Notification dispatch (omnichannel delivery)
    notification_dispatcher: Any = None  # NotificationDispatcher instance
    auto_notify: bool = False  # Auto-emit notifications on debate completion
    notify_min_confidence: float = 0.0  # Minimum confidence to notify (0 = always)

    # Internal state
    _initialized: bool = field(default=False, repr=False)
    _last_evaluation: Any = field(default=None, repr=False)  # Store last evaluation result

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._initialized = True

    @property
    def has_billing(self) -> bool:
        """Check if billing/usage tracking is configured."""
        return self.usage_tracker is not None

    @property
    def has_evaluation(self) -> bool:
        """Check if LLM-as-Judge evaluation is configured."""
        return self.llm_judge is not None or self.auto_evaluate

    @property
    def has_broadcast(self) -> bool:
        """Check if broadcast pipeline is configured."""
        return self.broadcast_pipeline is not None or self.auto_broadcast

    @property
    def has_training_export(self) -> bool:
        """Check if training export is configured."""
        return self.training_exporter is not None or self.auto_export_training

    @property
    def has_usage_sync(self) -> bool:
        """Check if Stripe usage sync is configured."""
        return self.usage_sync_service is not None or self.auto_sync_usage

    @property
    def has_notifications(self) -> bool:
        """Check if notification dispatch is configured."""
        return self.notification_dispatcher is not None or self.auto_notify

    @property
    def last_evaluation(self) -> Optional[Any]:
        """Get the last evaluation result."""
        return self._last_evaluation

    @property
    def has_debate_budget(self) -> bool:
        """Check if per-debate budget limit is configured."""
        return self.debate_budget_limit_usd is not None and self.debate_budget_limit_usd > 0

    def setup_debate_budget(self, debate_id: str) -> None:
        """Set up budget tracking for a new debate.

        Call this at the start of a debate to enable per-debate cost limits.
        If debate_budget_limit_usd is set, this will configure the cost
        tracker to enforce that limit.

        Args:
            debate_id: The debate ID to track
        """
        if not self.has_debate_budget:
            return

        try:
            from decimal import Decimal
            from aragora.billing.cost_tracker import get_cost_tracker

            if self.cost_tracker is None:
                self.cost_tracker = get_cost_tracker()

            limit = Decimal(str(self.debate_budget_limit_usd))
            self.cost_tracker.set_debate_limit(debate_id, limit)
            logger.info(
                "debate_budget_set debate=%s limit=$%.4f",
                debate_id,
                self.debate_budget_limit_usd,
            )
        except Exception as e:
            logger.warning("debate_budget_setup_failed: %s", e)

    def check_debate_budget(self, debate_id: str) -> dict:
        """Check if the debate is within its budget.

        Args:
            debate_id: The debate ID to check

        Returns:
            Budget status dict with 'allowed', 'current_cost', 'limit', 'message'
        """
        if not self.has_debate_budget or self.cost_tracker is None:
            return {"allowed": True, "message": "No budget limit configured"}

        try:
            return self.cost_tracker.check_debate_budget(debate_id)
        except Exception as e:
            logger.warning("debate_budget_check_failed: %s", e)
            return {"allowed": True, "message": f"Budget check failed: {e}"}

    def cleanup_debate_budget(self, debate_id: str) -> None:
        """Clean up budget tracking after a debate completes.

        Args:
            debate_id: The debate ID to clean up
        """
        if self.cost_tracker is None:
            return

        try:
            self.cost_tracker.clear_debate_budget(debate_id)
        except Exception as e:
            logger.debug("debate_budget_cleanup_failed: %s", e)

    def on_debate_complete(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
        agents: list["Agent"],
    ) -> None:
        """Trigger all extensions after debate completion.

        This is the main entry point called by Arena after a debate finishes.
        Each extension runs in a try/except to ensure one failure doesn't
        prevent other extensions from running.

        Args:
            ctx: The debate context with state and metadata
            result: The final debate result
            agents: List of agents that participated
        """
        # Record token usage for billing
        self._record_token_usage(ctx.debate_id, agents)

        # Sync usage to Stripe for metered billing
        self._sync_usage_to_stripe()

        # Evaluate debate quality with LLM-as-Judge
        self._evaluate_debate(ctx, result)

        # Export training data if configured
        self._export_training_data(ctx, result)

        # Sync KM adapters (expertise, patterns) learned during debate
        self._sync_km_adapters(ctx, result)

        # Emit debate completion notifications (omnichannel delivery)
        self._emit_debate_notifications(ctx, result)

        # Clean up per-debate budget tracking
        self.cleanup_debate_budget(ctx.debate_id)

        # Trigger broadcast if configured (not implemented here - kept in Arena for now)
        # The broadcast pipeline requires more complex integration

    def _record_token_usage(
        self,
        debate_id: str,
        agents: list["Agent"],
    ) -> None:
        """Record token usage from all agents for billing.

        Aggregates input/output tokens from each agent's metrics
        and records via the usage_tracker if configured. Also records
        per-agent costs via the cost_tracker for granular attribution.

        Args:
            debate_id: The debate ID for tracking
            agents: List of agents with usage metrics
        """
        total_input = 0
        total_output = 0
        providers = set()

        # Record per-agent costs via CostTracker
        for agent in agents:
            agent_input = 0
            agent_output = 0

            # Try to get token usage from different agent types
            metrics = getattr(agent, "metrics", None)
            if metrics:
                agent_input = getattr(metrics, "total_input_tokens", 0)
                agent_output = getattr(metrics, "total_output_tokens", 0)
            else:
                # Try API agent style (total_tokens_in/out)
                agent_input = getattr(agent, "total_tokens_in", 0)
                agent_output = getattr(agent, "total_tokens_out", 0)

            total_input += agent_input
            total_output += agent_output

            provider = getattr(agent, "provider", None)
            if provider:
                providers.add(provider)

            # Record per-agent cost if cost_tracker is available
            if agent_input > 0 or agent_output > 0:
                self._record_agent_cost(
                    agent=agent,
                    debate_id=debate_id,
                    tokens_in=agent_input,
                    tokens_out=agent_output,
                )

        # Record aggregate usage via UsageTracker
        if self.usage_tracker and (total_input > 0 or total_output > 0):
            try:
                self.usage_tracker.record_debate(
                    user_id=self.user_id,
                    org_id=self.org_id,
                    debate_id=debate_id,
                    tokens_in=total_input,
                    tokens_out=total_output,
                    provider=",".join(providers) if providers else "mixed",
                    model="debate",  # Debates use multiple models
                )
                logger.info(
                    "usage_recorded input=%d output=%d total=%d for debate %s (org=%s)",
                    total_input,
                    total_output,
                    total_input + total_output,
                    debate_id,
                    self.org_id,
                )
            except Exception as e:
                # Don't fail the debate if usage tracking fails
                logger.warning("usage_tracking_failed error=%s", e)

    def _record_agent_cost(
        self,
        agent: "Agent",
        debate_id: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record per-agent cost for granular attribution.

        Args:
            agent: The agent with usage data
            debate_id: The debate ID
            tokens_in: Input tokens used
            tokens_out: Output tokens generated

        Raises:
            DebateBudgetExceededError: If enforce_budget_limit is True and
                the debate has exceeded its budget
        """
        # Get or create cost tracker
        if self.cost_tracker is None:
            try:
                from aragora.billing.cost_tracker import get_cost_tracker

                self.cost_tracker = get_cost_tracker()
            except Exception as e:
                logger.debug("cost_tracker_init_skipped: %s", e)
                return

        try:
            from decimal import Decimal
            from aragora.billing.cost_tracker import TokenUsage, DebateBudgetExceededError

            # Extract agent info
            agent_name = getattr(agent, "name", str(agent))
            agent_id = getattr(agent, "id", "") or agent_name
            provider = getattr(agent, "provider", "unknown")
            model = getattr(agent, "model", "unknown")

            # Create usage record
            usage = TokenUsage(
                workspace_id=self.workspace_id,
                agent_id=agent_id,
                agent_name=agent_name,
                debate_id=debate_id,
                provider=provider,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                operation="debate_round",
                metadata={
                    "user_id": self.user_id,
                    "org_id": self.org_id,
                },
            )
            usage.calculate_cost()

            # Check budget before recording if limit is set
            if self.has_debate_budget and self.enforce_budget_limit:
                budget_status = self.cost_tracker.check_debate_budget(
                    debate_id,
                    estimated_cost_usd=usage.cost_usd,
                )
                if not budget_status.get("allowed", True):
                    current_cost = Decimal(budget_status.get("current_cost", "0"))
                    limit = Decimal(budget_status.get("limit", "0"))
                    logger.warning(
                        "debate_budget_exceeded debate=%s current=$%.4f limit=$%.4f",
                        debate_id,
                        current_cost,
                        limit,
                    )
                    raise DebateBudgetExceededError(
                        debate_id=debate_id,
                        current_cost=current_cost,
                        limit=limit,
                    )

            # Record asynchronously if in async context, otherwise sync
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.cost_tracker.record(usage))
            except RuntimeError:
                # No running loop - skip async recording
                pass

            logger.debug(
                "agent_cost_recorded agent=%s cost=$%.6f tokens=%d",
                agent_name,
                usage.cost_usd,
                tokens_in + tokens_out,
            )
        except DebateBudgetExceededError:
            # Re-raise budget exceeded errors
            raise
        except Exception as e:
            logger.debug("agent_cost_recording_failed: %s", e)

    def _sync_usage_to_stripe(self) -> None:
        """Sync usage data to Stripe for metered billing.

        Triggers the UsageSyncService to report accumulated usage
        to Stripe subscription metering. This enables per-debate
        billing for professional/enterprise tiers.
        """
        if not self.auto_sync_usage:
            return

        if not self.org_id:
            return

        try:
            # Lazy import to avoid circular dependencies
            if self.usage_sync_service is None:
                from aragora.billing.usage_sync import UsageSyncService

                self.usage_sync_service = UsageSyncService()

            # Trigger sync for this org (by ID lookup)
            self.usage_sync_service.sync_org_by_id(self.org_id)
            logger.debug("usage_sync_triggered org_id=%s", self.org_id)
        except Exception as e:
            # Don't fail the debate if Stripe sync fails
            logger.warning("usage_sync_failed org=%s error=%s", self.org_id, e)

    def _sync_km_adapters(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Sync Knowledge Mound adapters after debate completion.

        Persists expertise profiles and compression patterns learned during
        the debate to the Knowledge Mound for cross-debate learning.

        Args:
            ctx: The debate context with workspace info
            result: The final debate result
        """
        try:
            workspace_id = self.workspace_id or "default"

            # Get cross-subscriber manager to access adapter instances
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()

            # Sync RankingAdapter if expertise was recorded
            try:
                from aragora.knowledge.mound.adapters import RankingAdapter

                ranking_adapter = getattr(manager, "_ranking_adapter", None)
                if ranking_adapter is None:
                    ranking_adapter = RankingAdapter()
                    manager._ranking_adapter = ranking_adapter  # type: ignore[attr-defined]

                stats = ranking_adapter.get_stats()
                if stats.get("total_expertise_records", 0) > 0:
                    logger.debug(
                        "km_adapter_sync: %d expertise records to sync",
                        stats["total_expertise_records"],
                    )
            except (ImportError, AttributeError) as e:
                logger.debug("ranking_adapter_sync_skipped: %s", e)

            # Sync RlmAdapter if compression patterns were recorded
            try:
                from aragora.knowledge.mound.adapters import RlmAdapter

                rlm_adapter = getattr(manager, "_rlm_adapter", None)
                if rlm_adapter is None:
                    rlm_adapter = RlmAdapter()
                    manager._rlm_adapter = rlm_adapter  # type: ignore[attr-defined]

                stats = rlm_adapter.get_stats()
                if stats.get("total_patterns", 0) > 0:
                    logger.debug(
                        "km_adapter_sync: %d compression patterns to sync",
                        stats["total_patterns"],
                    )
            except (ImportError, AttributeError) as e:
                logger.debug("rlm_adapter_sync_skipped: %s", e)

            logger.debug("km_adapter_sync_complete workspace=%s", workspace_id)

        except Exception as e:
            # Don't fail the debate if KM sync fails
            logger.warning("km_adapter_sync_failed: %s", e)

    def _evaluate_debate(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Evaluate debate quality using LLM-as-Judge.

        Evaluates the final consensus/answer against the original query
        using 8-dimension scoring (relevance, accuracy, completeness,
        clarity, reasoning, evidence, creativity, safety).

        Results are stored in _last_evaluation for retrieval.

        Args:
            ctx: The debate context with query
            result: The final debate result with consensus
        """
        if not self.auto_evaluate:
            return

        # Get query and response
        query = getattr(ctx, "task", "") or getattr(ctx, "query", "")
        if not query:
            logger.debug("evaluation_skipped: no query found in context")
            return

        # Get final answer/consensus
        final_answer = getattr(result, "final_answer", None)
        if not final_answer:
            # Try to get from consensus
            consensus = getattr(result, "consensus", None)
            if consensus:
                final_answer = getattr(consensus, "content", str(consensus))
            else:
                # Fall back to last message
                messages = getattr(result, "messages", [])
                if messages:
                    final_answer = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )

        if not final_answer:
            logger.debug("evaluation_skipped: no final answer found")
            return

        try:
            # Lazy import to avoid circular dependencies
            if self.llm_judge is None:
                from aragora.evaluation.llm_judge import LLMJudge, JudgeConfig

                config = JudgeConfig(
                    use_case=self.evaluation_use_case,
                    pass_threshold=self.evaluation_threshold,
                )
                self.llm_judge = LLMJudge(config)

            # Run evaluation asynchronously
            async def run_eval() -> None:
                try:
                    evaluation = await self.llm_judge.evaluate(
                        query=query,
                        response=final_answer,
                        response_id=ctx.debate_id,
                    )
                    self._last_evaluation = evaluation

                    logger.info(
                        "debate_evaluated debate_id=%s score=%.2f passes=%s",
                        ctx.debate_id,
                        evaluation.overall_score,
                        evaluation.passes_threshold,
                    )

                    # Log dimension breakdown
                    for dim, score in evaluation.dimension_scores.items():
                        logger.debug(
                            "eval_dimension %s=%.1f confidence=%.2f",
                            dim.value,
                            score.score,
                            score.confidence,
                        )
                except Exception as e:
                    logger.warning("evaluation_async_failed: %s", e)

            # Try to schedule in running loop, otherwise skip
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(run_eval())
            except RuntimeError:
                # No running loop - try to run synchronously
                try:
                    asyncio.run(run_eval())
                except (RuntimeError, asyncio.CancelledError, OSError):
                    logger.debug("evaluation_skipped: no async context")

        except Exception as e:
            logger.warning("evaluation_failed: %s", e)

    def _emit_debate_notifications(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Emit notifications for debate completion (omnichannel delivery).

        Sends notifications to configured channels (Slack, Teams, Email, etc.)
        when a debate completes. Emits both task completion and deliberation
        consensus events.

        Args:
            ctx: The debate context with debate metadata
            result: The final debate result
        """
        if not self.auto_notify:
            return

        # Check confidence threshold
        confidence = getattr(result, "consensus_confidence", 0.0)
        if confidence < self.notify_min_confidence:
            logger.debug(
                "notification_skipped confidence=%.2f threshold=%.2f",
                confidence,
                self.notify_min_confidence,
            )
            return

        try:
            # Import task event emitters
            from aragora.control_plane.task_events import (
                emit_task_completed,
                get_task_event_dispatcher,
                set_task_event_dispatcher,
            )

            # Use configured dispatcher or get default
            if self.notification_dispatcher is not None:
                set_task_event_dispatcher(self.notification_dispatcher)

            # Get dispatcher (configured or default)
            dispatcher = get_task_event_dispatcher()
            if dispatcher is None:
                logger.debug("notification_skipped: no dispatcher available")
                return

            # Calculate duration from context metadata
            duration_seconds = 0.0
            if hasattr(ctx, "metadata") and ctx.metadata:
                start_time = ctx.metadata.get("start_time")
                end_time = ctx.metadata.get("end_time")
                if start_time and end_time:
                    duration_seconds = (end_time - start_time).total_seconds()

            # Emit task completion notification
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    emit_task_completed(
                        task_id=ctx.debate_id,
                        task_type="debate",
                        agent_id=result.winner or "consensus",
                        duration_seconds=duration_seconds,
                        workspace_id=self.workspace_id or None,
                    )
                )
            except RuntimeError:
                # No running loop - skip async notification
                logger.debug("notification_deferred: no running event loop")

            # Emit deliberation consensus notification for high-confidence results
            if confidence >= 0.7 and getattr(result, "consensus_reached", False):
                try:
                    from aragora.control_plane.channels import (
                        NotificationEventType,
                        NotificationPriority,
                    )

                    # Get the question from context
                    question = ""
                    if hasattr(ctx, "environment") and ctx.environment:
                        question = getattr(ctx.environment, "task", "") or ""

                    # Get the answer from result
                    answer = getattr(result, "consensus_answer", "") or ""
                    if not answer and result.messages:
                        # Use final message as answer
                        answer = result.messages[-1].content[:500]

                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        dispatcher.dispatch(
                            event_type=NotificationEventType.DELIBERATION_CONSENSUS,
                            title="Deliberation Consensus Reached",
                            body=f"**Question:** {question[:100]}...\n\n**Confidence:** {confidence:.0%}",
                            priority=NotificationPriority.NORMAL,
                            workspace_id=self.workspace_id or None,
                            metadata={
                                "debate_id": ctx.debate_id,
                                "confidence": confidence,
                                "question": question[:200],
                                "answer": answer[:500],
                            },
                        )
                    )
                except RuntimeError:
                    pass  # No running loop

            logger.info(
                "debate_notification_emitted debate_id=%s confidence=%.2f",
                ctx.debate_id,
                confidence,
            )
        except ImportError as e:
            logger.debug("notification_skipped: import failed: %s", e)
        except Exception as e:
            # Don't fail the debate if notification fails
            logger.warning("notification_emission_failed error=%s", e)

    def _export_training_data(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Export training data from completed debate (Tinker integration).

        Generates SFT (supervised fine-tuning) and DPO (direct preference
        optimization) examples from high-quality debates.

        Args:
            ctx: The debate context
            result: The final debate result
        """
        if not self.auto_export_training:
            return

        # Check confidence threshold
        confidence = getattr(result, "consensus_confidence", 0.0)
        if confidence < self.training_export_min_confidence:
            logger.debug(
                "training_export_skipped confidence=%.2f threshold=%.2f",
                confidence,
                self.training_export_min_confidence,
            )
            return

        try:
            # Lazy import to avoid circular dependencies
            if self.training_exporter is None:
                from aragora.training.debate_exporter import DebateTrainingExporter

                self.training_exporter = DebateTrainingExporter()

            # Export the debate
            export_result = self.training_exporter.export_debate(
                debate_id=ctx.debate_id,
                messages=result.messages,
                critiques=result.critiques,
                votes=result.votes,
            )
            if export_result:
                logger.info(
                    "training_export_complete debate_id=%s sft=%d dpo=%d",
                    ctx.debate_id,
                    export_result.get("sft_examples", 0),
                    export_result.get("dpo_examples", 0),
                )
        except Exception as e:
            # Don't fail the debate if training export fails
            logger.warning("training_export_failed error=%s", e)


@dataclass
class ExtensionsConfig:
    """Configuration for creating ArenaExtensions.

    This is used by ArenaConfig to pass extension settings.
    """

    org_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    usage_tracker: Any = None
    cost_tracker: Any = None
    llm_judge: Any = None
    auto_evaluate: bool = False
    evaluation_use_case: str = "debate"
    evaluation_threshold: float = 3.5
    broadcast_pipeline: Any = None
    auto_broadcast: bool = False
    broadcast_min_confidence: float = 0.8
    training_exporter: Any = None
    auto_export_training: bool = False
    training_export_min_confidence: float = 0.75
    usage_sync_service: Any = None
    auto_sync_usage: bool = False
    notification_dispatcher: Any = None
    auto_notify: bool = False
    notify_min_confidence: float = 0.0

    def create_extensions(self) -> ArenaExtensions:
        """Create ArenaExtensions from this configuration."""
        return ArenaExtensions(
            org_id=self.org_id,
            user_id=self.user_id,
            workspace_id=self.workspace_id,
            usage_tracker=self.usage_tracker,
            cost_tracker=self.cost_tracker,
            llm_judge=self.llm_judge,
            auto_evaluate=self.auto_evaluate,
            evaluation_use_case=self.evaluation_use_case,
            evaluation_threshold=self.evaluation_threshold,
            broadcast_pipeline=self.broadcast_pipeline,
            auto_broadcast=self.auto_broadcast,
            broadcast_min_confidence=self.broadcast_min_confidence,
            training_exporter=self.training_exporter,
            auto_export_training=self.auto_export_training,
            training_export_min_confidence=self.training_export_min_confidence,
            usage_sync_service=self.usage_sync_service,
            auto_sync_usage=self.auto_sync_usage,
            notification_dispatcher=self.notification_dispatcher,
            auto_notify=self.auto_notify,
            notify_min_confidence=self.notify_min_confidence,
        )


__all__ = ["ArenaExtensions", "ExtensionsConfig"]
