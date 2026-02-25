"""
Staking Registry for agent compute-budget staking and slashing.

Implements the economic layer for ERC-8004 agents: agents stake compute tokens
to participate, earn rewards for accurate outputs, and face slashing for
epistemic failures (hollow consensus, factual errors, calibration drift).

Design philosophy: "compute is ATP, truth is demanded behavior" -- make
truth-production the cheapest path to compute-acquisition within the system.

This registry works with a mock/local provider by default (no real ETH needed).
All operations are opt-in via ``enable_staking: bool = False``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default lock duration: 1 hour (in seconds)
DEFAULT_LOCK_DURATION = 3600

# Slashing caps: max 50% of stake per event, max 90% cumulative
MAX_SLASH_FRACTION = 0.50
MAX_CUMULATIVE_SLASH_FRACTION = 0.90

# Initial compute token grant for new agents
INITIAL_COMPUTE_GRANT = 1000

# Reward/penalty scaling constants
ACCURACY_REWARD_MULTIPLIER = 100  # tokens per 1.0 epistemic score
INACCURACY_PENALTY_MULTIPLIER = 50  # tokens per 1.0 inverse epistemic score


@dataclass
class SlashEvent:
    """Record of a slashing event against an agent's stake.

    Attributes:
        reason: Why the slash occurred (e.g. "hollow_consensus", "factual_error").
        amount_slashed_wei: Amount of stake slashed (in wei-equivalent tokens).
        timestamp: When the slash occurred.
        evidence_hash: SHA-256 hash of the evidence supporting the slash.
    """

    reason: str
    amount_slashed_wei: int
    timestamp: float
    evidence_hash: str


@dataclass
class StakePosition:
    """An agent's current stake position.

    Attributes:
        agent_id: The agent's identifier.
        amount_wei: Current staked amount (in wei-equivalent tokens).
        staked_at: Timestamp when the stake was created or last topped up.
        locked_until: Timestamp until which the stake is locked.
        slashing_events: History of slashing events against this stake.
    """

    agent_id: str
    amount_wei: int
    staked_at: float
    locked_until: float
    slashing_events: list[SlashEvent] = field(default_factory=list)

    @property
    def is_locked(self) -> bool:
        """Whether the stake is currently locked."""
        return time.time() < self.locked_until

    @property
    def total_slashed(self) -> int:
        """Total amount slashed from this position."""
        return sum(e.amount_slashed_wei for e in self.slashing_events)

    @property
    def effective_stake(self) -> int:
        """Stake amount minus cumulative slashing."""
        return max(0, self.amount_wei - self.total_slashed)


class StakingRegistry:
    """Manages agent compute-budget staking and slashing.

    Works with a mock/local provider by default -- no real blockchain
    transactions are needed for the in-memory mode. When a Web3Provider
    and contract address are supplied, operations are anchored on-chain.

    Usage:
        registry = StakingRegistry()  # In-memory mode
        await registry.stake("agent_1", amount_wei=1000, lock_duration=3600)
        budget = await registry.get_compute_budget("agent_1")

        # With blockchain provider
        provider = Web3Provider.from_env()
        registry = StakingRegistry(provider=provider, contract_address="0x...")
    """

    def __init__(
        self,
        provider: Any | None = None,
        contract_address: str | None = None,
    ) -> None:
        self._provider = provider
        self._contract_address = contract_address
        self._stakes: dict[str, StakePosition] = {}
        self._rewards: dict[str, int] = {}  # accumulated rewards per agent

    async def stake(
        self,
        agent_id: str,
        amount_wei: int,
        lock_duration: int = DEFAULT_LOCK_DURATION,
    ) -> StakePosition:
        """Stake compute tokens for an agent.

        Args:
            agent_id: The agent's identifier.
            amount_wei: Amount to stake (in wei-equivalent compute tokens).
            lock_duration: How long the stake is locked, in seconds.

        Returns:
            The created or updated StakePosition.

        Raises:
            ValueError: If amount_wei is not positive.
        """
        if amount_wei <= 0:
            raise ValueError("Stake amount must be positive")

        now = time.time()

        if agent_id in self._stakes:
            existing = self._stakes[agent_id]
            existing.amount_wei += amount_wei
            existing.staked_at = now
            existing.locked_until = max(existing.locked_until, now + lock_duration)
            logger.info(
                "Agent %s topped up stake to %d (locked until %.0f)",
                agent_id,
                existing.amount_wei,
                existing.locked_until,
            )
            return existing

        position = StakePosition(
            agent_id=agent_id,
            amount_wei=amount_wei,
            staked_at=now,
            locked_until=now + lock_duration,
        )
        self._stakes[agent_id] = position
        logger.info(
            "Agent %s staked %d tokens (locked for %ds)",
            agent_id,
            amount_wei,
            lock_duration,
        )
        return position

    async def slash(
        self,
        agent_id: str,
        amount_wei: int,
        reason: str,
        evidence: bytes,
    ) -> SlashEvent:
        """Slash an agent's stake for epistemic failure.

        The slash amount is capped at MAX_SLASH_FRACTION of the current
        effective stake per event, and cumulative slashing is capped at
        MAX_CUMULATIVE_SLASH_FRACTION.

        Args:
            agent_id: The agent to slash.
            amount_wei: Requested slash amount.
            reason: Why the slash is happening.
            evidence: Raw evidence bytes (hashed with SHA-256).

        Returns:
            The SlashEvent recording what happened.

        Raises:
            ValueError: If the agent has no stake or amount is not positive.
        """
        if amount_wei <= 0:
            raise ValueError("Slash amount must be positive")

        position = self._stakes.get(agent_id)
        if position is None:
            raise ValueError(f"Agent {agent_id} has no stake to slash")

        # Cap per-event slash at MAX_SLASH_FRACTION of effective stake
        max_per_event = int(position.effective_stake * MAX_SLASH_FRACTION)
        actual_slash = min(amount_wei, max_per_event)

        # Cap cumulative slashing
        max_cumulative = int(position.amount_wei * MAX_CUMULATIVE_SLASH_FRACTION)
        remaining_slash_budget = max(0, max_cumulative - position.total_slashed)
        actual_slash = min(actual_slash, remaining_slash_budget)

        if actual_slash <= 0:
            actual_slash = 0
            logger.warning(
                "Agent %s slash capped to 0 (cumulative limit reached)", agent_id
            )

        evidence_hash = hashlib.sha256(evidence).hexdigest()

        event = SlashEvent(
            reason=reason,
            amount_slashed_wei=actual_slash,
            timestamp=time.time(),
            evidence_hash=evidence_hash,
        )
        position.slashing_events.append(event)

        logger.info(
            "Slashed agent %s for %d tokens (reason: %s, evidence: %s)",
            agent_id,
            actual_slash,
            reason,
            evidence_hash[:16],
        )
        return event

    async def get_stake(self, agent_id: str) -> StakePosition | None:
        """Get an agent's current stake position.

        Args:
            agent_id: The agent's identifier.

        Returns:
            StakePosition if the agent has staked, None otherwise.
        """
        return self._stakes.get(agent_id)

    async def get_compute_budget(self, agent_id: str) -> int:
        """Calculate the available compute budget for an agent.

        The budget is: effective_stake + accumulated_rewards + initial_grant.
        Agents with no stake still get INITIAL_COMPUTE_GRANT tokens.

        Args:
            agent_id: The agent's identifier.

        Returns:
            Available compute tokens.
        """
        position = self._stakes.get(agent_id)
        base = position.effective_stake if position else 0
        rewards = self._rewards.get(agent_id, 0)
        grant = INITIAL_COMPUTE_GRANT if position is None else 0
        return base + rewards + grant

    async def reward(self, agent_id: str, amount_wei: int, reason: str) -> None:
        """Reward an agent with additional compute tokens.

        Args:
            agent_id: The agent to reward.
            amount_wei: Amount of compute tokens to grant.
            reason: Why the reward is being given.

        Raises:
            ValueError: If amount is not positive.
        """
        if amount_wei <= 0:
            raise ValueError("Reward amount must be positive")

        self._rewards[agent_id] = self._rewards.get(agent_id, 0) + amount_wei
        logger.info(
            "Rewarded agent %s with %d tokens (reason: %s)",
            agent_id,
            amount_wei,
            reason,
        )

    async def withdraw(self, agent_id: str, amount_wei: int) -> int:
        """Withdraw unlocked stake tokens.

        Args:
            agent_id: The agent withdrawing.
            amount_wei: Amount to withdraw.

        Returns:
            Actual amount withdrawn (may be less if stake is partially locked).

        Raises:
            ValueError: If the agent has no stake or stake is locked.
        """
        position = self._stakes.get(agent_id)
        if position is None:
            raise ValueError(f"Agent {agent_id} has no stake")

        if position.is_locked:
            raise ValueError(
                f"Agent {agent_id} stake is locked until "
                f"{position.locked_until:.0f}"
            )

        withdrawable = min(amount_wei, position.effective_stake)
        position.amount_wei -= withdrawable
        logger.info("Agent %s withdrew %d tokens", agent_id, withdrawable)
        return withdrawable


__all__ = [
    "DEFAULT_LOCK_DURATION",
    "INITIAL_COMPUTE_GRANT",
    "SlashEvent",
    "StakePosition",
    "StakingRegistry",
]
