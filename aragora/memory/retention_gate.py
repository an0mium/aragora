"""
Titans/MIRAS-Inspired Retention Gate.

Bridges SurpriseScorer to ConfidenceDecay pipeline to drive what gets
remembered and forgotten across memory systems.

Key concepts:
- High surprise -> consolidate/promote (novel, important)
- Low surprise -> eligible for forgetting (predictable, redundant)
- Red-line protection -> never forget critical entries
- Adaptive decay rate -> surprise modulates how fast confidence decays
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aragora.memory.surprise import (
    ContentSurpriseScorer,
    SurpriseScorer,
    DEFAULT_SURPRISE_ALPHA,
)

logger = logging.getLogger(__name__)


@dataclass
class RetentionGateConfig:
    """Configuration for the retention gate."""

    enable_surprise_driven_decay: bool = False  # Opt-in
    surprise_alpha: float = DEFAULT_SURPRISE_ALPHA  # EMA smoothing
    forget_threshold: float = 0.15  # Below -> eligible for forget
    consolidate_threshold: float = 0.7  # Above -> consolidate/promote
    adaptive_decay_enabled: bool = True
    max_decay_rate: float = 0.01
    min_decay_rate: float = 0.0001
    red_line_protection: bool = True  # Never forget red_line entries


@dataclass
class RetentionDecision:
    """Result of retention gate evaluation."""

    item_id: str
    source_system: str  # "continuum", "km", "supermemory", "claude_mem"
    surprise_score: float
    retention_score: float
    action: str  # "retain", "demote", "forget", "consolidate"
    decay_rate_override: float | None = None
    reason: str = ""


class RetentionGate:
    """Connects SurpriseScorer -> ConfidenceDecay pipeline.

    Evaluates memory items and produces RetentionDecisions that can be
    applied to the ConfidenceDecayManager (for KM items) or TierManager
    (for ContinuumMemory entries).
    """

    def __init__(
        self,
        config: RetentionGateConfig | None = None,
        scorer: SurpriseScorer | None = None,
        content_scorer: ContentSurpriseScorer | None = None,
    ):
        self.config = config or RetentionGateConfig()
        self.scorer = scorer or SurpriseScorer(alpha=self.config.surprise_alpha)
        self._content_scorer = content_scorer
        self._decisions: list[RetentionDecision] = []

    def evaluate(
        self,
        item_id: str,
        source: str,
        content: str,
        outcome_surprise: float,
        current_confidence: float,
        access_count: int = 0,
        is_red_line: bool = False,
    ) -> RetentionDecision:
        """Evaluate a single item and produce a retention decision."""
        # Red-line protection
        if is_red_line and self.config.red_line_protection:
            decision = RetentionDecision(
                item_id=item_id,
                source_system=source,
                surprise_score=outcome_surprise,
                retention_score=1.0,
                action="retain",
                reason="Red-line protected entry",
            )
            self._decisions.append(decision)
            return decision

        # Compute retention score from surprise + confidence + access
        # Higher surprise = more novel = retain/consolidate
        # Lower surprise = more predictable = consider forgetting
        access_bonus = min(0.2, access_count * 0.02)
        retention_score = outcome_surprise * 0.5 + current_confidence * 0.3 + access_bonus
        retention_score = min(1.0, max(0.0, retention_score))

        # Determine action
        if outcome_surprise >= self.config.consolidate_threshold:
            action = "consolidate"
            reason = (
                f"High surprise ({outcome_surprise:.2f}) >= "
                f"threshold ({self.config.consolidate_threshold})"
            )
        elif outcome_surprise <= self.config.forget_threshold and current_confidence < 0.3:
            action = "forget"
            reason = (
                f"Low surprise ({outcome_surprise:.2f}) and "
                f"low confidence ({current_confidence:.2f})"
            )
        elif outcome_surprise <= self.config.forget_threshold:
            action = "demote"
            reason = (
                f"Low surprise ({outcome_surprise:.2f}) <= "
                f"threshold ({self.config.forget_threshold})"
            )
        else:
            action = "retain"
            reason = f"Surprise ({outcome_surprise:.2f}) in normal range"

        # Compute adaptive decay rate override
        decay_rate_override = None
        if self.config.adaptive_decay_enabled:
            decay_rate_override = self.compute_adaptive_decay_rate(
                outcome_surprise, current_confidence
            )

        decision = RetentionDecision(
            item_id=item_id,
            source_system=source,
            surprise_score=outcome_surprise,
            retention_score=retention_score,
            action=action,
            decay_rate_override=decay_rate_override,
            reason=reason,
        )
        self._decisions.append(decision)
        return decision

    def compute_adaptive_decay_rate(self, surprise: float, current_confidence: float) -> float:
        """Compute adaptive decay rate based on surprise.

        High surprise -> slow decay (preserve novel items longer)
        Low surprise -> fast decay (forget predictable items faster)
        """
        # Invert surprise: high surprise = low decay rate
        # Linear interpolation between min and max decay rates
        surprise_factor = 1.0 - min(1.0, max(0.0, surprise))
        decay_rate = self.config.min_decay_rate + surprise_factor * (
            self.config.max_decay_rate - self.config.min_decay_rate
        )
        return decay_rate

    def score_content_surprise(
        self, content: str, source: str, existing_context: str = ""
    ) -> float:
        """Score content novelty using the ContentSurpriseScorer if available.

        Falls back to 0.5 (neutral) if no content scorer is configured.
        This enables the MemoryCoordinator to get per-item content-aware
        surprise scores rather than using the debate-level score for all items.
        """
        if self._content_scorer is None:
            return 0.5
        try:
            score = self._content_scorer.score(content, source, existing_context)
            return score.combined
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning("Content surprise scoring failed: %s", e)
            return 0.5

    def batch_evaluate(self, items: list[dict[str, Any]]) -> list[RetentionDecision]:
        """Evaluate a batch of items.

        Each item dict should have keys:
        - item_id, source, content, outcome_surprise, current_confidence
        - Optional: access_count, is_red_line
        """
        decisions = []
        for item in items:
            decision = self.evaluate(
                item_id=item["item_id"],
                source=item["source"],
                content=item.get("content", ""),
                outcome_surprise=item["outcome_surprise"],
                current_confidence=item["current_confidence"],
                access_count=item.get("access_count", 0),
                is_red_line=item.get("is_red_line", False),
            )
            decisions.append(decision)
        return decisions

    def get_decisions(self) -> list[RetentionDecision]:
        """Get all decisions made so far."""
        return list(self._decisions)

    def clear_decisions(self) -> None:
        """Clear decision history."""
        self._decisions.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get retention gate statistics."""
        actions: dict[str, int] = {}
        for d in self._decisions:
            actions[d.action] = actions.get(d.action, 0) + 1
        return {
            "total_decisions": len(self._decisions),
            "by_action": actions,
            "config": {
                "forget_threshold": self.config.forget_threshold,
                "consolidate_threshold": self.config.consolidate_threshold,
                "adaptive_decay_enabled": self.config.adaptive_decay_enabled,
                "red_line_protection": self.config.red_line_protection,
            },
        }
