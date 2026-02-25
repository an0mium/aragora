"""
Compute Budget Manager -- budget compute resources based on epistemic track record.

Allocates compute tokens to agents proportional to their reputation and stake,
rewards accurate outputs, and penalizes inaccurate ones. This creates the core
economic incentive: truth-production is the cheapest path to compute-acquisition.

Design:
    - allocation = base_tokens * (1 + reputation_factor) * task_complexity
    - accuracy rewards scale with epistemic_score (0.0 to 1.0)
    - inaccuracy penalties scale with (1.0 - epistemic_score)
    - agents with no stake or reputation get a baseline allocation

All operations are synchronous and work in-memory. The optional
StakingRegistry and ReputationRegistryContract provide on-chain backing
when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Baseline tokens every agent gets, regardless of stake/reputation
BASELINE_ALLOCATION = 100

# Maximum allocation multiplier from reputation
MAX_REPUTATION_MULTIPLIER = 3.0

# Scaling factors for rewards and penalties
ACCURACY_REWARD_SCALE = 100  # max tokens per perfect score
INACCURACY_PENALTY_SCALE = 50  # max tokens deducted per zero score

# Reputation normalization -- scores above this are max multiplier
REPUTATION_SCORE_CAP = 1000


@dataclass
class ComputeBudget:
    """Snapshot of an agent's compute budget.

    Attributes:
        agent_id: The agent's identifier.
        total_tokens: Total tokens ever allocated to this agent.
        used_tokens: Tokens consumed by compute tasks.
        earned_tokens: Bonus tokens earned from accurate outputs.
        penalty_tokens: Tokens deducted for inaccurate outputs.
    """

    agent_id: str
    total_tokens: int = 0
    used_tokens: int = 0
    earned_tokens: int = 0
    penalty_tokens: int = 0

    @property
    def available_tokens(self) -> int:
        """Tokens available for use."""
        return max(
            0, self.total_tokens + self.earned_tokens - self.used_tokens - self.penalty_tokens
        )


class ComputeBudgetManager:
    """Budget compute resources based on agent epistemic track record.

    Usage:
        manager = ComputeBudgetManager()
        tokens = manager.allocate("agent_1", task_complexity=0.8)
        manager.charge("agent_1", tokens_used=50)
        bonus = manager.reward_accuracy("agent_1", epistemic_score=0.95)
        penalty = manager.penalize_inaccuracy("agent_1", epistemic_score=0.2)
        budget = manager.get_budget("agent_1")

    With optional backing registries:
        manager = ComputeBudgetManager(
            staking_registry=staking_reg,
            reputation_registry=reputation_reg,
        )
    """

    def __init__(
        self,
        staking_registry: Any | None = None,
        reputation_registry: Any | None = None,
    ) -> None:
        self._staking = staking_registry
        self._reputation = reputation_registry
        self._budgets: dict[str, ComputeBudget] = {}

    def _get_or_create_budget(self, agent_id: str) -> ComputeBudget:
        """Get existing budget or create a new one."""
        if agent_id not in self._budgets:
            self._budgets[agent_id] = ComputeBudget(agent_id=agent_id)
        return self._budgets[agent_id]

    def _get_reputation_factor(self, agent_id: str) -> float:
        """Get a reputation multiplier for the agent.

        Returns a value between 0.0 and MAX_REPUTATION_MULTIPLIER.
        If no reputation registry is configured, returns 1.0 (neutral).
        """
        if self._reputation is None:
            return 1.0

        try:
            # Try to get a reputation summary; the registry may use int IDs
            summary = self._reputation.get_summary(agent_id=int(agent_id))
            raw_score = summary.normalized_value
            # Normalize to [0, MAX_REPUTATION_MULTIPLIER]
            factor = min(
                MAX_REPUTATION_MULTIPLIER,
                max(0.0, raw_score / REPUTATION_SCORE_CAP * MAX_REPUTATION_MULTIPLIER),
            )
            return max(factor, 0.1)  # Floor at 0.1 so agents always get something
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            logger.debug("Could not fetch reputation for agent %s: %s", agent_id, exc)
            return 1.0

    def allocate(self, agent_id: str, task_complexity: float = 1.0) -> int:
        """Allocate compute tokens for a task.

        Args:
            agent_id: The agent receiving the allocation.
            task_complexity: Complexity factor (0.0 to 1.0+). Higher = more tokens.

        Returns:
            Number of tokens allocated.
        """
        complexity = max(0.1, task_complexity)
        rep_factor = self._get_reputation_factor(agent_id)
        allocation = int(BASELINE_ALLOCATION * rep_factor * complexity)
        allocation = max(1, allocation)  # Always allocate at least 1 token

        budget = self._get_or_create_budget(agent_id)
        budget.total_tokens += allocation

        logger.debug(
            "Allocated %d tokens to agent %s (complexity=%.2f, rep_factor=%.2f)",
            allocation,
            agent_id,
            complexity,
            rep_factor,
        )
        return allocation

    def charge(self, agent_id: str, tokens_used: int) -> None:
        """Charge an agent for compute usage.

        Args:
            agent_id: The agent being charged.
            tokens_used: Number of tokens consumed.

        Raises:
            ValueError: If tokens_used is negative.
        """
        if tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")

        budget = self._get_or_create_budget(agent_id)
        budget.used_tokens += tokens_used

        logger.debug(
            "Charged agent %s for %d tokens (available: %d)",
            agent_id,
            tokens_used,
            budget.available_tokens,
        )

    def reward_accuracy(self, agent_id: str, epistemic_score: float) -> int:
        """Reward an agent for accurate output.

        Args:
            agent_id: The agent being rewarded.
            epistemic_score: Accuracy score between 0.0 and 1.0.

        Returns:
            Number of bonus tokens earned.
        """
        score = max(0.0, min(1.0, epistemic_score))
        bonus = int(score * ACCURACY_REWARD_SCALE)

        if bonus > 0:
            budget = self._get_or_create_budget(agent_id)
            budget.earned_tokens += bonus
            logger.debug(
                "Rewarded agent %s with %d tokens (epistemic_score=%.2f)",
                agent_id,
                bonus,
                score,
            )
        return bonus

    def penalize_inaccuracy(self, agent_id: str, epistemic_score: float) -> int:
        """Penalize an agent for inaccurate output.

        The penalty is proportional to (1.0 - epistemic_score). An agent
        with a perfect score (1.0) receives no penalty.

        Args:
            agent_id: The agent being penalized.
            epistemic_score: Accuracy score between 0.0 and 1.0.

        Returns:
            Number of tokens deducted.
        """
        score = max(0.0, min(1.0, epistemic_score))
        penalty = int((1.0 - score) * INACCURACY_PENALTY_SCALE)

        if penalty > 0:
            budget = self._get_or_create_budget(agent_id)
            budget.penalty_tokens += penalty
            logger.debug(
                "Penalized agent %s for %d tokens (epistemic_score=%.2f)",
                agent_id,
                penalty,
                score,
            )
        return penalty

    def get_budget(self, agent_id: str) -> ComputeBudget:
        """Get the current compute budget for an agent.

        Args:
            agent_id: The agent's identifier.

        Returns:
            ComputeBudget snapshot.
        """
        return self._get_or_create_budget(agent_id)

    def has_budget(self, agent_id: str, tokens_needed: int) -> bool:
        """Check whether an agent can afford a task.

        Args:
            agent_id: The agent's identifier.
            tokens_needed: Tokens the task requires.

        Returns:
            True if the agent has sufficient available tokens.
        """
        budget = self._get_or_create_budget(agent_id)
        return budget.available_tokens >= tokens_needed

    def reset(self, agent_id: str) -> None:
        """Reset an agent's budget to zero. Primarily for testing.

        Args:
            agent_id: The agent whose budget to reset.
        """
        self._budgets.pop(agent_id, None)


__all__ = [
    "BASELINE_ALLOCATION",
    "ComputeBudget",
    "ComputeBudgetManager",
]
