"""
Tests for the ChannelBudgetEnforcer.

Verifies:
- Budget check allows debates under limit
- Budget check blocks debates over limit
- Warning at 80% threshold
- Per-channel spend tracking
- Per-workspace spend tracking
- Custom channel budgets
- Budget reset operations
- External budget manager integration
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.integrations.channel_budget import (
    BudgetCheckResult,
    ChannelBudgetEnforcer,
    ChannelSpendRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enforcer() -> ChannelBudgetEnforcer:
    return ChannelBudgetEnforcer(
        workspace_budget_usd=100.0,
        channel_budget_usd=20.0,
        warning_threshold=0.80,
    )


@pytest.fixture
def small_budget_enforcer() -> ChannelBudgetEnforcer:
    return ChannelBudgetEnforcer(
        workspace_budget_usd=10.0,
        channel_budget_usd=5.0,
    )


# ---------------------------------------------------------------------------
# Basic budget check tests
# ---------------------------------------------------------------------------


class TestBudgetCheckAllow:
    """Tests for budget checks that should allow debates."""

    @pytest.mark.asyncio
    async def test_allows_when_under_budget(self, enforcer: ChannelBudgetEnforcer) -> None:
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.allowed is True
        assert result.blocked is False

    @pytest.mark.asyncio
    async def test_no_warning_when_low_utilization(self, enforcer: ChannelBudgetEnforcer) -> None:
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.warning is False

    @pytest.mark.asyncio
    async def test_allows_with_estimated_cost_under_limit(self, enforcer: ChannelBudgetEnforcer) -> None:
        result = await enforcer.check_budget("slack", "C01", "T01", estimated_cost_usd=1.0)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Budget blocking tests
# ---------------------------------------------------------------------------


class TestBudgetCheckBlock:
    """Tests for budget checks that should block debates."""

    @pytest.mark.asyncio
    async def test_blocks_when_channel_budget_exceeded(self, small_budget_enforcer: ChannelBudgetEnforcer) -> None:
        # Spend up to the limit
        await small_budget_enforcer.record_spend("slack", "C01", "T01", cost_usd=5.5)

        result = await small_budget_enforcer.check_budget("slack", "C01", "T01")
        assert result.blocked is True
        assert result.allowed is False
        assert "exceeded" in result.message.lower()

    @pytest.mark.asyncio
    async def test_blocks_when_workspace_budget_exceeded(self, small_budget_enforcer: ChannelBudgetEnforcer) -> None:
        # Spend across multiple channels to exceed workspace
        await small_budget_enforcer.record_spend("slack", "C01", "T01", cost_usd=4.0)
        await small_budget_enforcer.record_spend("slack", "C02", "T01", cost_usd=4.0)
        await small_budget_enforcer.record_spend("slack", "C03", "T01", cost_usd=3.0)

        result = await small_budget_enforcer.check_budget("slack", "C04", "T01")
        assert result.blocked is True

    @pytest.mark.asyncio
    async def test_blocks_with_estimated_cost_exceeding_limit(self, small_budget_enforcer: ChannelBudgetEnforcer) -> None:
        await small_budget_enforcer.record_spend("slack", "C01", "T01", cost_usd=4.0)
        result = await small_budget_enforcer.check_budget("slack", "C01", "T01", estimated_cost_usd=2.0)
        assert result.blocked is True


# ---------------------------------------------------------------------------
# Warning threshold tests
# ---------------------------------------------------------------------------


class TestWarningThreshold:
    """Tests for budget warning when approaching limit."""

    @pytest.mark.asyncio
    async def test_warning_at_80_percent(self, enforcer: ChannelBudgetEnforcer) -> None:
        # Spend 85% of channel budget (17 of 20)
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=17.0)
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.warning is True
        assert result.allowed is True
        assert "approaching" in result.message.lower() or "remaining" in result.message.lower()

    @pytest.mark.asyncio
    async def test_no_warning_under_threshold(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=10.0)  # 50%
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.warning is False


# ---------------------------------------------------------------------------
# Spend tracking tests
# ---------------------------------------------------------------------------


class TestSpendTracking:
    """Tests for per-channel and per-workspace spend recording."""

    @pytest.mark.asyncio
    async def test_record_spend_updates_channel(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=5.0)
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is not None
        assert record.total_usd == 5.0
        assert record.debate_count == 1

    @pytest.mark.asyncio
    async def test_cumulative_spend(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=3.0)
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=2.0)
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is not None
        assert record.total_usd == 5.0
        assert record.debate_count == 2

    @pytest.mark.asyncio
    async def test_workspace_spend_aggregates(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=3.0)
        await enforcer.record_spend("slack", "C02", "T01", cost_usd=4.0)
        ws_total = enforcer.get_workspace_spend("slack", "T01")
        assert ws_total == 7.0

    @pytest.mark.asyncio
    async def test_zero_spend_ignored(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=0.0)
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is None

    @pytest.mark.asyncio
    async def test_negative_spend_ignored(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=-1.0)
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is None


# ---------------------------------------------------------------------------
# Custom budget tests
# ---------------------------------------------------------------------------


class TestCustomBudgets:
    """Tests for custom per-channel budgets."""

    @pytest.mark.asyncio
    async def test_set_custom_channel_budget(self, enforcer: ChannelBudgetEnforcer) -> None:
        enforcer.set_channel_budget("slack", "C01", "T01", budget_usd=50.0)
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=25.0)
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.allowed is True  # 50% of 50.0

    @pytest.mark.asyncio
    async def test_custom_budget_enforced(self, enforcer: ChannelBudgetEnforcer) -> None:
        enforcer.set_channel_budget("slack", "C01", "T01", budget_usd=5.0)
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=6.0)
        result = await enforcer.check_budget("slack", "C01", "T01")
        assert result.blocked is True


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestBudgetReset:
    """Tests for budget reset operations."""

    @pytest.mark.asyncio
    async def test_reset_channel_clears_spend(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=15.0)
        enforcer.reset_channel("slack", "C01", "T01")
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is not None
        assert record.total_usd == 0.0

    @pytest.mark.asyncio
    async def test_reset_preserves_custom_budget(self, enforcer: ChannelBudgetEnforcer) -> None:
        enforcer.set_channel_budget("slack", "C01", "T01", budget_usd=50.0)
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=15.0)
        enforcer.reset_channel("slack", "C01", "T01")
        record = enforcer.get_channel_spend("slack", "C01", "T01")
        assert record is not None
        assert record.budget_limit_usd == 50.0

    @pytest.mark.asyncio
    async def test_reset_workspace(self, enforcer: ChannelBudgetEnforcer) -> None:
        await enforcer.record_spend("slack", "C01", "T01", cost_usd=10.0)
        enforcer.reset_workspace("slack", "T01")
        ws_total = enforcer.get_workspace_spend("slack", "T01")
        assert ws_total == 0.0


# ---------------------------------------------------------------------------
# BudgetCheckResult tests
# ---------------------------------------------------------------------------


class TestBudgetCheckResult:
    """Tests for BudgetCheckResult dataclass."""

    def test_utilization_pct_format(self) -> None:
        result = BudgetCheckResult(utilization=0.85)
        assert result.utilization_pct == "85%"

    def test_default_allowed(self) -> None:
        result = BudgetCheckResult()
        assert result.allowed is True
        assert result.blocked is False
        assert result.warning is False
