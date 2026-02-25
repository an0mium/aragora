"""
Tests for StakingRegistry -- agent compute-budget staking and slashing.

All tests use in-memory mode (no Web3Provider needed).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.blockchain.contracts.staking import (
    DEFAULT_LOCK_DURATION,
    INITIAL_COMPUTE_GRANT,
    MAX_CUMULATIVE_SLASH_FRACTION,
    MAX_SLASH_FRACTION,
    SlashEvent,
    StakePosition,
    StakingRegistry,
)


# =============================================================================
# StakePosition dataclass tests
# =============================================================================


class TestStakePosition:
    """Tests for the StakePosition dataclass."""

    def test_is_locked_when_future(self):
        """Stake is locked when locked_until is in the future."""
        pos = StakePosition(
            agent_id="a1",
            amount_wei=100,
            staked_at=time.time(),
            locked_until=time.time() + 3600,
        )
        assert pos.is_locked is True

    def test_is_unlocked_when_past(self):
        """Stake is unlocked when locked_until is in the past."""
        pos = StakePosition(
            agent_id="a1",
            amount_wei=100,
            staked_at=time.time() - 7200,
            locked_until=time.time() - 3600,
        )
        assert pos.is_locked is False

    def test_total_slashed_empty(self):
        """Total slashed is 0 with no slashing events."""
        pos = StakePosition(
            agent_id="a1",
            amount_wei=100,
            staked_at=time.time(),
            locked_until=time.time() + 100,
        )
        assert pos.total_slashed == 0

    def test_total_slashed_with_events(self):
        """Total slashed sums all slash event amounts."""
        pos = StakePosition(
            agent_id="a1",
            amount_wei=1000,
            staked_at=time.time(),
            locked_until=time.time() + 100,
            slashing_events=[
                SlashEvent(reason="r1", amount_slashed_wei=100, timestamp=time.time(), evidence_hash="abc"),
                SlashEvent(reason="r2", amount_slashed_wei=200, timestamp=time.time(), evidence_hash="def"),
            ],
        )
        assert pos.total_slashed == 300
        assert pos.effective_stake == 700

    def test_effective_stake_never_negative(self):
        """Effective stake is floored at 0."""
        pos = StakePosition(
            agent_id="a1",
            amount_wei=100,
            staked_at=time.time(),
            locked_until=time.time() + 100,
            slashing_events=[
                SlashEvent(reason="r1", amount_slashed_wei=200, timestamp=time.time(), evidence_hash="abc"),
            ],
        )
        assert pos.effective_stake == 0


# =============================================================================
# StakingRegistry tests
# =============================================================================


class TestStakingRegistry:
    """Tests for StakingRegistry."""

    @pytest.fixture
    def registry(self) -> StakingRegistry:
        """Create a fresh in-memory staking registry."""
        return StakingRegistry()

    @pytest.mark.asyncio
    async def test_stake_creates_position(self, registry: StakingRegistry):
        """Staking creates a new position."""
        pos = await registry.stake("agent_1", amount_wei=500, lock_duration=3600)
        assert pos.agent_id == "agent_1"
        assert pos.amount_wei == 500
        assert pos.is_locked is True

    @pytest.mark.asyncio
    async def test_stake_rejects_zero(self, registry: StakingRegistry):
        """Staking with zero or negative amount raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            await registry.stake("agent_1", amount_wei=0)
        with pytest.raises(ValueError, match="positive"):
            await registry.stake("agent_1", amount_wei=-100)

    @pytest.mark.asyncio
    async def test_stake_topup(self, registry: StakingRegistry):
        """Staking again tops up the existing position."""
        await registry.stake("agent_1", amount_wei=500)
        pos = await registry.stake("agent_1", amount_wei=300)
        assert pos.amount_wei == 800

    @pytest.mark.asyncio
    async def test_slash_basic(self, registry: StakingRegistry):
        """Slashing deducts from effective stake."""
        await registry.stake("agent_1", amount_wei=1000)
        event = await registry.slash("agent_1", amount_wei=200, reason="factual_error", evidence=b"evidence")
        assert event.amount_slashed_wei == 200
        assert event.reason == "factual_error"
        assert len(event.evidence_hash) == 64  # SHA-256 hex

        pos = await registry.get_stake("agent_1")
        assert pos is not None
        assert pos.effective_stake == 800

    @pytest.mark.asyncio
    async def test_slash_no_stake_raises(self, registry: StakingRegistry):
        """Slashing an agent with no stake raises ValueError."""
        with pytest.raises(ValueError, match="no stake"):
            await registry.slash("nonexistent", amount_wei=100, reason="test", evidence=b"x")

    @pytest.mark.asyncio
    async def test_slash_negative_raises(self, registry: StakingRegistry):
        """Slashing with zero or negative amount raises ValueError."""
        await registry.stake("agent_1", amount_wei=1000)
        with pytest.raises(ValueError, match="positive"):
            await registry.slash("agent_1", amount_wei=0, reason="test", evidence=b"x")

    @pytest.mark.asyncio
    async def test_slash_capped_per_event(self, registry: StakingRegistry):
        """Single slash is capped at MAX_SLASH_FRACTION of effective stake."""
        await registry.stake("agent_1", amount_wei=1000)
        event = await registry.slash("agent_1", amount_wei=900, reason="test", evidence=b"x")
        max_expected = int(1000 * MAX_SLASH_FRACTION)
        assert event.amount_slashed_wei == max_expected

    @pytest.mark.asyncio
    async def test_slash_capped_cumulative(self, registry: StakingRegistry):
        """Cumulative slashing is capped at MAX_CUMULATIVE_SLASH_FRACTION."""
        await registry.stake("agent_1", amount_wei=1000)
        total_slashed = 0
        for _ in range(10):
            event = await registry.slash("agent_1", amount_wei=200, reason="test", evidence=b"x")
            total_slashed += event.amount_slashed_wei

        max_cumulative = int(1000 * MAX_CUMULATIVE_SLASH_FRACTION)
        assert total_slashed <= max_cumulative

    @pytest.mark.asyncio
    async def test_get_stake_returns_none_for_unknown(self, registry: StakingRegistry):
        """Getting stake for unknown agent returns None."""
        result = await registry.get_stake("unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_compute_budget_with_stake(self, registry: StakingRegistry):
        """Compute budget reflects effective stake plus rewards."""
        await registry.stake("agent_1", amount_wei=500)
        await registry.reward("agent_1", amount_wei=100, reason="accuracy")
        budget = await registry.get_compute_budget("agent_1")
        assert budget == 600  # 500 stake + 100 reward

    @pytest.mark.asyncio
    async def test_get_compute_budget_no_stake(self, registry: StakingRegistry):
        """Agents without a stake get the initial compute grant."""
        budget = await registry.get_compute_budget("new_agent")
        assert budget == INITIAL_COMPUTE_GRANT

    @pytest.mark.asyncio
    async def test_reward_basic(self, registry: StakingRegistry):
        """Rewarding adds to compute budget."""
        await registry.reward("agent_1", amount_wei=250, reason="good_output")
        budget = await registry.get_compute_budget("agent_1")
        # No stake, so: 0 (no stake, but also no initial grant since rewards exist)
        # Actually: no stake means INITIAL_COMPUTE_GRANT + rewards
        assert budget == INITIAL_COMPUTE_GRANT + 250

    @pytest.mark.asyncio
    async def test_reward_rejects_nonpositive(self, registry: StakingRegistry):
        """Rewarding with zero or negative amount raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            await registry.reward("agent_1", amount_wei=0, reason="test")

    @pytest.mark.asyncio
    async def test_withdraw_unlocked(self, registry: StakingRegistry):
        """Withdrawal works when stake is unlocked."""
        await registry.stake("agent_1", amount_wei=1000, lock_duration=0)
        # Wait a tiny bit for the lock to expire (lock_duration=0 means locked_until=now)
        with patch("aragora.blockchain.contracts.staking.time") as mock_time:
            mock_time.time.return_value = time.time() + 1
            withdrawn = await registry.withdraw("agent_1", amount_wei=500)
        assert withdrawn == 500

    @pytest.mark.asyncio
    async def test_withdraw_locked_raises(self, registry: StakingRegistry):
        """Withdrawal raises when stake is locked."""
        await registry.stake("agent_1", amount_wei=1000, lock_duration=3600)
        with pytest.raises(ValueError, match="locked"):
            await registry.withdraw("agent_1", amount_wei=500)

    @pytest.mark.asyncio
    async def test_withdraw_no_stake_raises(self, registry: StakingRegistry):
        """Withdrawal raises when agent has no stake."""
        with pytest.raises(ValueError, match="no stake"):
            await registry.withdraw("unknown", amount_wei=100)
