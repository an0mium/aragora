"""Business context layer for intelligent goal prioritization.

Scores goals based on business impact: user-facing visibility,
revenue relevance, unblocking potential, and tech debt importance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TECH_DEBT_KEYWORDS = frozenset({
    "refactor",
    "cleanup",
    "tech debt",
    "technical debt",
    "legacy",
    "deprecated",
    "migration",
    "upgrade",
    "modernize",
    "decouple",
    "simplify",
    "dead code",
    "lint",
    "type annotation",
    "type hint",
})

_UNBLOCKING_KEYWORDS = frozenset({
    "block",
    "blocking",
    "unblock",
    "dependency",
    "prerequisite",
    "required by",
    "critical path",
    "waiting on",
})


@dataclass
class BusinessContextConfig:
    """Weights and path patterns for business context scoring."""

    user_facing_weight: float = 0.3
    revenue_weight: float = 0.25
    unblocking_weight: float = 0.25
    tech_debt_weight: float = 0.2

    user_facing_paths: list[str] = field(default_factory=lambda: [
        "aragora/live/",
        "aragora/cli/",
        "aragora/server/handlers/",
        "aragora/server/stream/",
        "sdk/",
    ])
    revenue_paths: list[str] = field(default_factory=lambda: [
        "aragora/billing/",
        "aragora/tenancy/",
        "aragora/auth/",
        "aragora/marketplace/",
    ])
    infrastructure_paths: list[str] = field(default_factory=lambda: [
        "aragora/storage/",
        "aragora/resilience/",
        "aragora/observability/",
    ])


@dataclass
class GoalScore:
    """Scored result for a single goal."""

    total: float
    user_facing: float
    revenue: float
    unblocking: float
    tech_debt: float
    breakdown: dict[str, float] = field(default_factory=dict)


class BusinessContext:
    """Score goals based on business impact dimensions."""

    def __init__(self, config: BusinessContextConfig | None = None) -> None:
        self.config = config or BusinessContextConfig()

    def score_goal(
        self,
        goal: str,
        file_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GoalScore:
        """Compute a weighted composite score for a goal."""
        files = file_paths or []
        meta = metadata or {}

        uf = self._user_facing_score(files)
        rv = self._revenue_relevance(files)
        ub = self._unblocking_score(goal, meta)
        td = self._tech_debt_score(goal)

        cfg = self.config
        total = (
            cfg.user_facing_weight * uf
            + cfg.revenue_weight * rv
            + cfg.unblocking_weight * ub
            + cfg.tech_debt_weight * td
        )

        return GoalScore(
            total=round(total, 4),
            user_facing=round(uf, 4),
            revenue=round(rv, 4),
            unblocking=round(ub, 4),
            tech_debt=round(td, 4),
            breakdown={
                "user_facing_weighted": round(cfg.user_facing_weight * uf, 4),
                "revenue_weighted": round(cfg.revenue_weight * rv, 4),
                "unblocking_weighted": round(cfg.unblocking_weight * ub, 4),
                "tech_debt_weighted": round(cfg.tech_debt_weight * td, 4),
            },
        )

    def rank_goals(self, goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort goals by weighted business score (highest first).

        Each dict should have at least a ``goal`` key (str). Optional keys:
        ``file_paths`` (list[str]) and ``metadata`` (dict).
        """
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, g in enumerate(goals):
            score = self.score_goal(
                goal=g.get("goal", ""),
                file_paths=g.get("file_paths"),
                metadata=g.get("metadata"),
            )
            enriched = {**g, "score": score}
            scored.append((score.total, idx, enriched))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [t[2] for t in scored]

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _user_facing_score(self, file_paths: list[str]) -> float:
        """0-1: fraction of files in user-facing paths."""
        if not file_paths:
            return 0.0
        matches = sum(
            1 for f in file_paths
            if any(f.startswith(p) or f"/{p}" in f for p in self.config.user_facing_paths)
        )
        return matches / len(file_paths)

    def _revenue_relevance(self, file_paths: list[str]) -> float:
        """0-1: fraction of files in revenue-related paths."""
        if not file_paths:
            return 0.0
        matches = sum(
            1 for f in file_paths
            if any(f.startswith(p) or f"/{p}" in f for p in self.config.revenue_paths)
        )
        return matches / len(file_paths)

    def _unblocking_score(self, goal: str, metadata: dict[str, Any]) -> float:
        """0-1: based on dependency count, blocking count, and keywords."""
        score = 0.0
        goal_lower = goal.lower()

        # Keyword presence
        if any(kw in goal_lower for kw in _UNBLOCKING_KEYWORDS):
            score += 0.4

        # Metadata signals
        blocks_count = metadata.get("blocks_count", 0)
        if blocks_count > 0:
            score += min(0.3, blocks_count * 0.1)

        dep_count = metadata.get("dependency_count", 0)
        if dep_count > 0:
            score += min(0.3, dep_count * 0.1)

        return min(1.0, score)

    def _tech_debt_score(self, goal: str) -> float:
        """0-1: keyword matching for tech debt indicators."""
        goal_lower = goal.lower()
        matches = sum(1 for kw in _TECH_DEBT_KEYWORDS if kw in goal_lower)
        if matches == 0:
            return 0.0
        # Diminishing returns: 1 keyword = 0.4, 2 = 0.6, 3+ = 0.8, 5+ = 1.0
        return min(1.0, 0.2 + matches * 0.2)
