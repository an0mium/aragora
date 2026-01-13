"""
Autotuner - Budget-aware debate optimization.

Dynamically adjusts debate parameters based on:
- Cost budget (token limits, API costs)
- Quality metrics (support score variance, verification density)
- Early-stop conditions (convergence, consensus reached)

Key features:
- Budget tracking across rounds
- Quality-based early stopping
- Model tier selection (cheap vs expensive)
- Round count optimization
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional


class CostTier(Enum):
    """Model cost tiers."""

    FREE = "free"  # Open source, local
    CHEAP = "cheap"  # GPT-3.5, Claude Haiku, Gemini Flash
    STANDARD = "standard"  # GPT-4o, Claude Sonnet
    EXPENSIVE = "expensive"  # GPT-4, Claude Opus, Gemini Pro


@dataclass
class AutotuneConfig:
    """Configuration for autotuner behavior."""

    # Budget limits
    max_cost_dollars: float = 1.0  # Max spend per debate
    max_tokens: int = 100000  # Max tokens used
    max_rounds: int = 5  # Hard limit on rounds
    max_duration_seconds: float = 300  # 5 minute timeout

    # Early-stop thresholds
    early_stop_support_variance: float = 0.1  # Stop if variance < this
    early_stop_verification_density: float = 0.7  # Stop if density > this
    early_stop_consensus_confidence: float = 0.85  # Stop if confidence > this

    # Quality targets
    min_evidence_per_claim: int = 1
    min_rounds_before_stop: int = 1

    # Cost weights ($ per 1K tokens, approximate)
    cost_per_1k_tokens: dict[CostTier, float] = field(
        default_factory=lambda: {
            CostTier.FREE: 0.0,
            CostTier.CHEAP: 0.0005,  # ~$0.50 per 1M
            CostTier.STANDARD: 0.003,  # ~$3 per 1M
            CostTier.EXPENSIVE: 0.03,  # ~$30 per 1M
        }
    )

    # Default model tiers
    default_tier: CostTier = CostTier.STANDARD


@dataclass
class RunMetrics:
    """Metrics collected during a debate run."""

    # Counts
    rounds_completed: int = 0
    messages_sent: int = 0
    tokens_used: int = 0
    critiques_made: int = 0
    claims_made: int = 0
    evidence_cited: int = 0

    # Quality metrics
    avg_support_score: float = 0.0
    support_score_variance: float = 1.0
    verification_density: float = 0.0  # Verified claims / total claims
    consensus_confidence: float = 0.0

    # Cost metrics
    estimated_cost: float = 0.0
    duration_seconds: float = 0.0

    # Timestamps
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    # Per-round metrics
    round_metrics: list[dict] = field(default_factory=list)

    def add_round_metrics(
        self,
        round_num: int,
        tokens: int,
        messages: int,
        support_scores: list[float],
    ):
        """Record metrics for a completed round."""
        self.rounds_completed = round_num + 1
        self.tokens_used += tokens
        self.messages_sent += messages

        # Update support score stats
        if support_scores:
            avg = sum(support_scores) / len(support_scores)
            variance = sum((s - avg) ** 2 for s in support_scores) / len(support_scores)

            self.avg_support_score = avg
            self.support_score_variance = variance

        self.round_metrics.append(
            {
                "round": round_num,
                "tokens": tokens,
                "messages": messages,
                "avg_support_score": self.avg_support_score,
                "variance": self.support_score_variance,
            }
        )

    def to_dict(self) -> dict:
        return {
            "rounds_completed": self.rounds_completed,
            "messages_sent": self.messages_sent,
            "tokens_used": self.tokens_used,
            "critiques_made": self.critiques_made,
            "claims_made": self.claims_made,
            "evidence_cited": self.evidence_cited,
            "avg_support_score": self.avg_support_score,
            "support_score_variance": self.support_score_variance,
            "verification_density": self.verification_density,
            "consensus_confidence": self.consensus_confidence,
            "estimated_cost": self.estimated_cost,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "round_metrics": self.round_metrics,
        }


class StopReason(Enum):
    """Why the autotuner decided to stop."""

    MAX_ROUNDS = "max_rounds"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    MAX_DURATION = "max_duration"
    CONSENSUS_REACHED = "consensus_reached"
    QUALITY_THRESHOLD = "quality_threshold"
    USER_REQUESTED = "user_requested"
    ERROR = "error"


@dataclass
class AutotuneDecision:
    """Decision from the autotuner."""

    should_continue: bool
    stop_reason: Optional[StopReason] = None
    recommended_tier: CostTier = CostTier.STANDARD
    suggested_action: str = ""
    metrics_summary: dict = field(default_factory=dict)


class Autotuner:
    """
    Budget-aware debate optimizer.

    Monitors debate progress and decides when to:
    - Continue to next round
    - Stop early (quality reached)
    - Downgrade model tier (budget pressure)
    - Stop immediately (budget exceeded)
    """

    def __init__(self, config: Optional[AutotuneConfig] = None):
        self.config = config or AutotuneConfig()
        self.metrics = RunMetrics()
        self._start_time: Optional[datetime] = None
        self._current_tier = self.config.default_tier

    def start(self):
        """Mark debate start."""
        self._start_time = datetime.now()
        self.metrics.started_at = self._start_time.isoformat()

    def end(self):
        """Mark debate end."""
        if self._start_time:
            self.metrics.ended_at = datetime.now().isoformat()
            self.metrics.duration_seconds = (datetime.now() - self._start_time).total_seconds()

    def record_round(
        self,
        round_num: int,
        tokens: int,
        messages: int,
        support_scores: list[float],
        verified_claims: int = 0,
        total_claims: int = 0,
    ):
        """Record metrics after a round completes."""
        self.metrics.add_round_metrics(round_num, tokens, messages, support_scores)

        # Update verification density
        if total_claims > 0:
            self.metrics.verification_density = verified_claims / total_claims
            self.metrics.claims_made = total_claims

        # Update cost estimate
        cost_rate = self.config.cost_per_1k_tokens.get(self._current_tier, 0.003)
        self.metrics.estimated_cost = (self.metrics.tokens_used / 1000) * cost_rate

    def record_consensus(self, confidence: float, reached: bool):
        """Record consensus check result."""
        self.metrics.consensus_confidence = confidence

    def should_continue(self) -> AutotuneDecision:
        """
        Decide whether to continue the debate.

        Checks budget limits, quality thresholds, and early-stop conditions.
        """
        # Check hard limits first
        if self.metrics.rounds_completed >= self.config.max_rounds:
            return AutotuneDecision(
                should_continue=False,
                stop_reason=StopReason.MAX_ROUNDS,
                metrics_summary=self.metrics.to_dict(),
            )

        if self.metrics.estimated_cost >= self.config.max_cost_dollars:
            return AutotuneDecision(
                should_continue=False,
                stop_reason=StopReason.MAX_COST,
                suggested_action="Consider increasing budget or using cheaper models",
                metrics_summary=self.metrics.to_dict(),
            )

        if self.metrics.tokens_used >= self.config.max_tokens:
            return AutotuneDecision(
                should_continue=False,
                stop_reason=StopReason.MAX_TOKENS,
                metrics_summary=self.metrics.to_dict(),
            )

        # Check duration
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            if elapsed >= self.config.max_duration_seconds:
                return AutotuneDecision(
                    should_continue=False,
                    stop_reason=StopReason.MAX_DURATION,
                    metrics_summary=self.metrics.to_dict(),
                )

        # Check early-stop conditions (only after min rounds)
        if self.metrics.rounds_completed >= self.config.min_rounds_before_stop:

            # High consensus confidence
            if self.metrics.consensus_confidence >= self.config.early_stop_consensus_confidence:
                return AutotuneDecision(
                    should_continue=False,
                    stop_reason=StopReason.CONSENSUS_REACHED,
                    suggested_action="Strong consensus achieved",
                    metrics_summary=self.metrics.to_dict(),
                )

            # Quality threshold reached
            if (
                self.metrics.support_score_variance < self.config.early_stop_support_variance
                and self.metrics.verification_density >= self.config.early_stop_verification_density
            ):
                return AutotuneDecision(
                    should_continue=False,
                    stop_reason=StopReason.QUALITY_THRESHOLD,
                    suggested_action="Quality metrics indicate convergence",
                    metrics_summary=self.metrics.to_dict(),
                )

        # Budget pressure - consider downgrading tier
        budget_used = self.metrics.estimated_cost / self.config.max_cost_dollars
        recommended_tier = self._recommend_tier(budget_used)

        return AutotuneDecision(
            should_continue=True,
            recommended_tier=recommended_tier,
            suggested_action=f"Continue with {recommended_tier.value} tier models",
            metrics_summary=self.metrics.to_dict(),
        )

    def _recommend_tier(self, budget_used: float) -> CostTier:
        """Recommend model tier based on budget usage."""
        if budget_used < 0.3:
            # Plenty of budget - use expensive for quality
            return CostTier.EXPENSIVE
        elif budget_used < 0.6:
            # Normal usage
            return CostTier.STANDARD
        elif budget_used < 0.85:
            # Budget pressure - downgrade
            return CostTier.CHEAP
        else:
            # Critical - use free/cheapest
            return CostTier.FREE

    def get_budget_remaining(self) -> dict:
        """Get remaining budget info."""
        return {
            "cost_remaining": self.config.max_cost_dollars - self.metrics.estimated_cost,
            "tokens_remaining": self.config.max_tokens - self.metrics.tokens_used,
            "rounds_remaining": self.config.max_rounds - self.metrics.rounds_completed,
            "budget_used_percent": (self.metrics.estimated_cost / self.config.max_cost_dollars)
            * 100,
        }

    def suggest_rounds(self) -> int:
        """Suggest optimal number of remaining rounds based on budget."""
        budget = self.get_budget_remaining()

        # Estimate tokens per round from history
        if self.metrics.rounds_completed > 0:
            tokens_per_round = self.metrics.tokens_used / self.metrics.rounds_completed
        else:
            tokens_per_round = 5000  # Default estimate

        # How many rounds can we afford?
        affordable_rounds = int(budget["tokens_remaining"] / tokens_per_round)

        # Cap at remaining rounds limit
        max_remaining = budget["rounds_remaining"]

        return min(affordable_rounds, max_remaining, 3)  # Max 3 more rounds suggested


class AutotunedDebateRunner:
    """
    Wrapper to run debates with autotuning.

    Integrates with Arena to provide automatic optimization.
    """

    def __init__(
        self,
        arena,  # Arena instance
        config: Optional[AutotuneConfig] = None,
    ):
        self.arena = arena
        self.tuner = Autotuner(config)

    async def run(
        self,
        on_round_complete: Optional[Callable[[int, RunMetrics], None]] = None,
    ):
        """
        Run debate with autotuning.

        Monitors each round and decides whether to continue.
        """
        self.tuner.start()

        try:
            # Run debate with round callbacks
            result = await self.arena.run(
                on_round_complete=lambda round_num, stats: self._on_round(
                    round_num, stats, on_round_complete
                )
            )

            return result, self.tuner.metrics

        finally:
            self.tuner.end()

    def _on_round(
        self,
        round_num: int,
        stats: dict,
        user_callback: Optional[Callable],
    ):
        """Handle round completion."""
        # Record metrics
        self.tuner.record_round(
            round_num=round_num,
            tokens=stats.get("tokens", 0),
            messages=stats.get("messages", 0),
            support_scores=stats.get("support_scores", []),
            verified_claims=stats.get("verified_claims", 0),
            total_claims=stats.get("total_claims", 0),
        )

        # Check if we should continue
        decision = self.tuner.should_continue()

        if user_callback:
            user_callback(round_num, self.tuner.metrics)

        if not decision.should_continue:
            # Signal arena to stop
            # This would require Arena to support early termination
            pass

        return decision
