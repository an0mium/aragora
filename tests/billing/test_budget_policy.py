"""
Tests for the workspace budget policy engine.

Tests cover:
- Budget check allows when under limit
- Budget check denies when over hard limit
- Budget check warns when over soft threshold but allows
- Usage summary calculation
- Default policy allows everything
- Policy CRUD operations
- Usage recording and reset
"""

from __future__ import annotations

import pytest

from aragora.billing.budget_policy import (
    BudgetDecision,
    BudgetPolicy,
    BudgetPolicyEngine,
    UsageSummary,
    get_budget_policy_engine,
)


@pytest.fixture
def engine():
    """Create a fresh BudgetPolicyEngine for each test."""
    return BudgetPolicyEngine()


class TestBudgetCheckAllowsUnderLimit:
    """Budget check allows when under limit."""

    @pytest.mark.asyncio
    async def test_under_monthly_limit(self, engine):
        """Operations under the monthly limit are allowed."""
        policy = BudgetPolicy(monthly_limit=100.0, hard_limit=True)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 30.0)

        decision = await engine.check_budget("ws-1", estimated_cost=10.0)

        assert decision.allowed is True
        assert decision.usage_pct == pytest.approx(30.0)
        assert decision.remaining == pytest.approx(70.0)
        assert "Within budget" in decision.reason

    @pytest.mark.asyncio
    async def test_zero_usage(self, engine):
        """First operation on a fresh workspace is allowed."""
        policy = BudgetPolicy(monthly_limit=500.0, hard_limit=True)
        await engine.set_policy("ws-new", policy)

        decision = await engine.check_budget("ws-new", estimated_cost=25.0)

        assert decision.allowed is True
        assert decision.usage_pct == pytest.approx(0.0)
        assert decision.remaining == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_exactly_at_limit_allowed(self, engine):
        """Operation that would exactly reach the limit is allowed."""
        policy = BudgetPolicy(monthly_limit=100.0, hard_limit=True)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 90.0)

        decision = await engine.check_budget("ws-1", estimated_cost=10.0)

        # 90 + 10 = 100, which is exactly the limit (not exceeding)
        assert decision.allowed is True


class TestBudgetCheckDeniesOverHardLimit:
    """Budget check denies when over hard limit."""

    @pytest.mark.asyncio
    async def test_over_hard_limit_denied(self, engine):
        """Operations exceeding hard limit are denied."""
        policy = BudgetPolicy(monthly_limit=100.0, hard_limit=True)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 95.0)

        decision = await engine.check_budget("ws-1", estimated_cost=10.0)

        assert decision.allowed is False
        assert "exceeded" in decision.reason.lower()
        assert decision.usage_pct == pytest.approx(95.0)
        assert decision.remaining == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_already_over_limit_denied(self, engine):
        """Operations denied when already over limit."""
        policy = BudgetPolicy(monthly_limit=50.0, hard_limit=True)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 60.0)

        decision = await engine.check_budget("ws-1", estimated_cost=1.0)

        assert decision.allowed is False
        assert decision.usage_pct == pytest.approx(120.0)
        assert decision.remaining == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_hard_limit_false_allows_over(self, engine):
        """When hard_limit=False, exceeding limit triggers warning but allows."""
        policy = BudgetPolicy(
            monthly_limit=100.0,
            hard_limit=False,
            alert_threshold_pct=80.0,
        )
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 95.0)

        decision = await engine.check_budget("ws-1", estimated_cost=10.0)

        # Soft limit: should still be allowed (warning only)
        assert decision.allowed is True
        assert "warning" in decision.reason.lower()


class TestBudgetCheckWarnsOverSoftThreshold:
    """Budget check warns when over soft threshold but allows."""

    @pytest.mark.asyncio
    async def test_over_alert_threshold_warns(self, engine):
        """Crossing the alert threshold emits a warning but allows."""
        policy = BudgetPolicy(
            monthly_limit=100.0,
            alert_threshold_pct=80.0,
            hard_limit=True,
        )
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 85.0)

        decision = await engine.check_budget("ws-1", estimated_cost=5.0)

        assert decision.allowed is True
        assert "warning" in decision.reason.lower()
        assert decision.usage_pct == pytest.approx(85.0)

    @pytest.mark.asyncio
    async def test_exactly_at_threshold(self, engine):
        """Exactly at the alert threshold triggers warning."""
        policy = BudgetPolicy(
            monthly_limit=100.0,
            alert_threshold_pct=80.0,
            hard_limit=True,
        )
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 80.0)

        decision = await engine.check_budget("ws-1", estimated_cost=1.0)

        assert decision.allowed is True
        assert "warning" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_below_threshold_no_warning(self, engine):
        """Below the alert threshold, no warning is returned."""
        policy = BudgetPolicy(
            monthly_limit=100.0,
            alert_threshold_pct=80.0,
            hard_limit=True,
        )
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 50.0)

        decision = await engine.check_budget("ws-1", estimated_cost=5.0)

        assert decision.allowed is True
        assert "Within budget" in decision.reason


class TestUsageSummaryCalculation:
    """Usage summary calculation."""

    @pytest.mark.asyncio
    async def test_usage_summary_with_policy(self, engine):
        """Usage summary reflects recorded costs and policy limits."""
        policy = BudgetPolicy(monthly_limit=200.0)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 50.0, debate_id="d1")
        await engine.record_cost("ws-1", 30.0, debate_id="d2")

        summary = await engine.get_usage_summary("ws-1")

        assert summary.workspace_id == "ws-1"
        assert summary.period == "monthly"
        assert summary.total_cost == pytest.approx(80.0)
        assert summary.limit == pytest.approx(200.0)
        assert summary.usage_pct == pytest.approx(40.0)
        assert summary.debates_count == 2

    @pytest.mark.asyncio
    async def test_usage_summary_no_policy(self, engine):
        """Usage summary with no policy shows zero limit."""
        await engine.record_cost("ws-orphan", 25.0)

        summary = await engine.get_usage_summary("ws-orphan")

        assert summary.workspace_id == "ws-orphan"
        assert summary.total_cost == pytest.approx(25.0)
        assert summary.limit == pytest.approx(0.0)
        assert summary.usage_pct == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_usage_summary_no_usage(self, engine):
        """Usage summary with no recorded costs."""
        policy = BudgetPolicy(monthly_limit=100.0)
        await engine.set_policy("ws-empty", policy)

        summary = await engine.get_usage_summary("ws-empty")

        assert summary.total_cost == pytest.approx(0.0)
        assert summary.debates_count == 0
        assert summary.usage_pct == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_usage_summary_to_dict(self, engine):
        """UsageSummary.to_dict returns correct structure."""
        policy = BudgetPolicy(monthly_limit=100.0)
        await engine.set_policy("ws-1", policy)
        await engine.record_cost("ws-1", 40.0, debate_id="d1")

        summary = await engine.get_usage_summary("ws-1")
        d = summary.to_dict()

        assert d["workspace_id"] == "ws-1"
        assert d["period"] == "monthly"
        assert d["total_cost"] == pytest.approx(40.0)
        assert d["limit"] == pytest.approx(100.0)
        assert d["usage_pct"] == pytest.approx(40.0)
        assert d["debates_count"] == 1


class TestDefaultPolicyAllowsEverything:
    """Default policy allows everything."""

    @pytest.mark.asyncio
    async def test_no_policy_allows(self, engine):
        """Without any policy set, all operations are allowed."""
        decision = await engine.check_budget("ws-unknown", estimated_cost=1000.0)

        assert decision.allowed is True
        assert "No budget policy configured" in decision.reason
        assert decision.usage_pct == 0.0
        assert decision.remaining == 0.0

    @pytest.mark.asyncio
    async def test_unlimited_policy_allows(self, engine):
        """A policy with limit=0 (unlimited) allows everything."""
        policy = BudgetPolicy(monthly_limit=0.0, hard_limit=True)
        await engine.set_policy("ws-unlimited", policy)

        decision = await engine.check_budget("ws-unlimited", estimated_cost=99999.0)

        assert decision.allowed is True
        assert "No monthly limit" in decision.reason


class TestPolicyCRUD:
    """Policy set and get operations."""

    @pytest.mark.asyncio
    async def test_set_and_get_policy(self, engine):
        """Can set and retrieve a policy."""
        policy = BudgetPolicy(monthly_limit=500.0, daily_limit=25.0, hard_limit=True)
        await engine.set_policy("ws-1", policy)

        retrieved = engine.get_policy("ws-1")

        assert retrieved is not None
        assert retrieved.monthly_limit == 500.0
        assert retrieved.daily_limit == 25.0
        assert retrieved.hard_limit is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_policy(self, engine):
        """Getting a policy for unknown workspace returns None."""
        assert engine.get_policy("ws-nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_policy(self, engine):
        """Updating a policy replaces the old one."""
        await engine.set_policy("ws-1", BudgetPolicy(monthly_limit=100.0))
        await engine.set_policy("ws-1", BudgetPolicy(monthly_limit=200.0))

        retrieved = engine.get_policy("ws-1")
        assert retrieved is not None
        assert retrieved.monthly_limit == 200.0


class TestUsageRecordingAndReset:
    """Usage recording and reset operations."""

    @pytest.mark.asyncio
    async def test_record_cost_accumulates(self, engine):
        """Recording costs accumulates usage."""
        await engine.record_cost("ws-1", 10.0)
        await engine.record_cost("ws-1", 20.0)
        await engine.record_cost("ws-1", 5.0)

        assert engine._usage["ws-1"] == pytest.approx(35.0)

    @pytest.mark.asyncio
    async def test_record_cost_with_debate(self, engine):
        """Recording costs with debate_id increments debate counter."""
        await engine.record_cost("ws-1", 10.0, debate_id="d1")
        await engine.record_cost("ws-1", 20.0, debate_id="d2")
        await engine.record_cost("ws-1", 5.0)  # No debate_id

        assert engine._debate_counts["ws-1"] == 2

    @pytest.mark.asyncio
    async def test_reset_usage(self, engine):
        """Resetting clears usage and debate counts."""
        await engine.record_cost("ws-1", 100.0, debate_id="d1")
        engine.reset_usage("ws-1")

        assert engine._usage.get("ws-1") is None
        assert engine._debate_counts.get("ws-1") is None

    @pytest.mark.asyncio
    async def test_reset_all_usage(self, engine):
        """Resetting all clears all workspace counters."""
        await engine.record_cost("ws-1", 50.0)
        await engine.record_cost("ws-2", 75.0)
        engine.reset_all_usage()

        assert len(engine._usage) == 0
        assert len(engine._debate_counts) == 0

    @pytest.mark.asyncio
    async def test_reset_nonexistent_workspace(self, engine):
        """Resetting a nonexistent workspace does not raise."""
        engine.reset_usage("ws-nonexistent")  # Should not raise


class TestBudgetDecisionSerialization:
    """BudgetDecision data class operations."""

    def test_to_dict(self):
        """BudgetDecision.to_dict returns correct structure."""
        decision = BudgetDecision(
            allowed=True,
            reason="Within budget",
            usage_pct=45.0,
            remaining=55.0,
        )
        d = decision.to_dict()

        assert d["allowed"] is True
        assert d["reason"] == "Within budget"
        assert d["usage_pct"] == pytest.approx(45.0)
        assert d["remaining"] == pytest.approx(55.0)


class TestSingleton:
    """Singleton getter."""

    def test_get_budget_policy_engine(self):
        """get_budget_policy_engine returns a BudgetPolicyEngine instance."""
        import aragora.billing.budget_policy as mod

        # Reset singleton for isolated test
        old = mod._budget_policy_engine
        mod._budget_policy_engine = None
        try:
            engine = get_budget_policy_engine()
            assert isinstance(engine, BudgetPolicyEngine)

            # Calling again returns the same instance
            engine2 = get_budget_policy_engine()
            assert engine is engine2
        finally:
            mod._budget_policy_engine = old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
