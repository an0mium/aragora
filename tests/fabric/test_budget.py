"""Tests for Agent Fabric BudgetManager."""

from __future__ import annotations

import pytest
from datetime import datetime

from aragora.fabric.budget import BudgetManager
from aragora.fabric.models import BudgetConfig, Usage


@pytest.fixture
def manager():
    return BudgetManager()


def make_usage(
    agent_id: str = "a1",
    tokens_in: int = 100,
    tokens_out: int = 50,
    cost: float = 0.01,
    model: str = "claude-3-opus",
) -> Usage:
    return Usage(
        agent_id=agent_id,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=cost,
        model=model,
    )


class TestSetBudget:
    @pytest.mark.asyncio
    async def test_set_budget(self, manager):
        config = BudgetConfig(max_tokens_per_day=10000)
        await manager.set_budget("a1", config)
        stored = await manager.get_budget("a1")
        assert stored is not None
        assert stored.max_tokens_per_day == 10000

    @pytest.mark.asyncio
    async def test_get_budget_nonexistent(self, manager):
        result = await manager.get_budget("nonexistent")
        assert result is None


class TestTrackUsage:
    @pytest.mark.asyncio
    async def test_track_basic(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=10000))
        status = await manager.track(make_usage())
        assert status.tokens_used == 150  # 100 + 50
        assert not status.over_limit

    @pytest.mark.asyncio
    async def test_track_accumulates(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=10000))
        await manager.track(make_usage(tokens_in=100, tokens_out=50))
        status = await manager.track(make_usage(tokens_in=200, tokens_out=100))
        assert status.tokens_used == 450  # (100+50) + (200+100)

    @pytest.mark.asyncio
    async def test_track_without_budget(self, manager):
        status = await manager.track(make_usage())
        assert status.tokens_used == 150
        assert not status.over_limit  # No budget = no limit


class TestCheckBudget:
    @pytest.mark.asyncio
    async def test_within_budget(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=10000))
        allowed, status = await manager.check_budget("a1", estimated_tokens=100)
        assert allowed

    @pytest.mark.asyncio
    async def test_exceeds_budget(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=100, hard_limit=True))
        # Use up tokens
        await manager.track(make_usage(tokens_in=80, tokens_out=30))
        # Checking more tokens should fail
        allowed, status = await manager.check_budget("a1", estimated_tokens=50)
        assert not allowed

    @pytest.mark.asyncio
    async def test_no_budget_always_allowed(self, manager):
        allowed, status = await manager.check_budget("a1", estimated_tokens=999999)
        assert allowed

    @pytest.mark.asyncio
    async def test_cost_limit(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_cost_per_day_usd=1.0, hard_limit=True))
        await manager.track(make_usage(cost=0.90))
        allowed, status = await manager.check_budget("a1", estimated_cost_usd=0.20)
        assert not allowed

    @pytest.mark.asyncio
    async def test_soft_limit_allows(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=100, hard_limit=False))
        await manager.track(make_usage(tokens_in=80, tokens_out=30))
        # Soft limit: still allowed even though over
        allowed, status = await manager.check_budget("a1", estimated_tokens=50)
        assert allowed


class TestAlerts:
    @pytest.mark.asyncio
    async def test_alert_triggered(self, manager):
        await manager.set_budget(
            "a1",
            BudgetConfig(max_tokens_per_day=1000, alert_threshold_percent=80.0),
        )
        # Use 85% of budget
        status = await manager.track(make_usage(tokens_in=600, tokens_out=250))
        assert status.alert_triggered
        assert status.usage_percent >= 80.0

    @pytest.mark.asyncio
    async def test_alert_not_triggered(self, manager):
        await manager.set_budget(
            "a1",
            BudgetConfig(max_tokens_per_day=10000, alert_threshold_percent=80.0),
        )
        status = await manager.track(make_usage(tokens_in=100, tokens_out=50))
        assert not status.alert_triggered

    @pytest.mark.asyncio
    async def test_alert_callback(self):
        callback_calls = []

        async def alert_cb(entity_id, status):
            callback_calls.append((entity_id, status.usage_percent))

        manager = BudgetManager(alert_callback=alert_cb)
        await manager.set_budget(
            "a1",
            BudgetConfig(max_tokens_per_day=100, alert_threshold_percent=50.0),
        )
        await manager.track(make_usage(tokens_in=60, tokens_out=10))
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "a1"


class TestUsageReport:
    @pytest.mark.asyncio
    async def test_report(self, manager):
        await manager.set_budget("a1", BudgetConfig())
        await manager.track(make_usage(tokens_in=100, tokens_out=50, cost=0.01, model="opus"))
        await manager.track(make_usage(tokens_in=200, tokens_out=100, cost=0.02, model="gpt-4"))

        report = await manager.get_usage("a1")
        assert report.total_tokens == 450
        assert abs(report.total_cost_usd - 0.03) < 0.001
        assert "opus" in report.by_model
        assert "gpt-4" in report.by_model

    @pytest.mark.asyncio
    async def test_report_empty(self, manager):
        report = await manager.get_usage("nonexistent")
        assert report.total_tokens == 0
        assert report.total_cost_usd == 0.0


class TestResetPeriod:
    @pytest.mark.asyncio
    async def test_reset(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=1000))
        await manager.track(make_usage(tokens_in=500, tokens_out=200))
        await manager.reset_period("a1")
        # Alert flag should be reset
        assert not manager._alert_triggered.get("a1", False)


class TestStats:
    @pytest.mark.asyncio
    async def test_stats(self, manager):
        await manager.set_budget("a1", BudgetConfig(max_tokens_per_day=1000))
        await manager.track(make_usage())

        stats = await manager.get_stats()
        assert stats["entities_tracked"] == 1
        assert stats["total_usage_records"] >= 1
