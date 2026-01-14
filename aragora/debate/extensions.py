"""Arena extensions for billing, broadcast, and training export.

This module separates non-core debate concerns from the Arena class,
following the Single Responsibility Principle. Extensions are triggered
via callbacks after debate completion.

Usage:
    # Create extensions configuration
    extensions = ArenaExtensions(
        org_id="org_123",
        user_id="user_456",
        usage_tracker=tracker,
        auto_broadcast=True,
    )

    # After debate completion, trigger extensions
    extensions.on_debate_complete(ctx, result, agents)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.core import Agent, DebateResult
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


@dataclass
class ArenaExtensions:
    """Extension hooks for Arena that handle non-core concerns.

    These extensions are triggered after debate completion and handle:
    - Billing/usage tracking
    - Audio/video broadcast generation
    - Training data export (Tinker integration)

    All extensions are optional and fail gracefully (don't break debates).
    """

    # Billing/usage tracking
    org_id: str = ""
    user_id: str = ""
    usage_tracker: Any = None  # UsageTracker instance

    # Broadcast pipeline
    broadcast_pipeline: Any = None  # BroadcastPipeline instance
    auto_broadcast: bool = False
    broadcast_min_confidence: float = 0.8

    # Training data export (Tinker integration)
    training_exporter: Any = None  # DebateTrainingExporter instance
    auto_export_training: bool = False
    training_export_min_confidence: float = 0.75

    # Internal state
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._initialized = True

    @property
    def has_billing(self) -> bool:
        """Check if billing/usage tracking is configured."""
        return self.usage_tracker is not None

    @property
    def has_broadcast(self) -> bool:
        """Check if broadcast pipeline is configured."""
        return self.broadcast_pipeline is not None or self.auto_broadcast

    @property
    def has_training_export(self) -> bool:
        """Check if training export is configured."""
        return self.training_exporter is not None or self.auto_export_training

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

        # Export training data if configured
        self._export_training_data(ctx, result)

        # Trigger broadcast if configured (not implemented here - kept in Arena for now)
        # The broadcast pipeline requires more complex integration

    def _record_token_usage(
        self,
        debate_id: str,
        agents: list["Agent"],
    ) -> None:
        """Record token usage from all agents for billing.

        Aggregates input/output tokens from each agent's metrics
        and records via the usage_tracker if configured.

        Args:
            debate_id: The debate ID for tracking
            agents: List of agents with usage metrics
        """
        if not self.usage_tracker:
            return

        try:
            total_input = 0
            total_output = 0

            for agent in agents:
                metrics = getattr(agent, "metrics", None)
                if metrics:
                    total_input += getattr(metrics, "total_input_tokens", 0)
                    total_output += getattr(metrics, "total_output_tokens", 0)

            if total_input > 0 or total_output > 0:
                # Collect provider info from agents
                providers = set()
                for agent in agents:
                    provider = getattr(agent, "provider", None)
                    if provider:
                        providers.add(provider)

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
                    "usage_recorded input=%d output=%d total=%d " "for debate %s (org=%s)",
                    total_input,
                    total_output,
                    total_input + total_output,
                    debate_id,
                    self.org_id,
                )
        except Exception as e:
            # Don't fail the debate if usage tracking fails
            logger.warning("usage_tracking_failed error=%s", e)

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
    usage_tracker: Any = None
    broadcast_pipeline: Any = None
    auto_broadcast: bool = False
    broadcast_min_confidence: float = 0.8
    training_exporter: Any = None
    auto_export_training: bool = False
    training_export_min_confidence: float = 0.75

    def create_extensions(self) -> ArenaExtensions:
        """Create ArenaExtensions from this configuration."""
        return ArenaExtensions(
            org_id=self.org_id,
            user_id=self.user_id,
            usage_tracker=self.usage_tracker,
            broadcast_pipeline=self.broadcast_pipeline,
            auto_broadcast=self.auto_broadcast,
            broadcast_min_confidence=self.broadcast_min_confidence,
            training_exporter=self.training_exporter,
            auto_export_training=self.auto_export_training,
            training_export_min_confidence=self.training_export_min_confidence,
        )


__all__ = ["ArenaExtensions", "ExtensionsConfig"]
