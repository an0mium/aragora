"""
Tests for Cost Tracking.

Tests cover:
- Token usage recording and cost calculation
- Budget management and alerts
- Workspace and organization cost tracking
- Debate-level budget enforcement
- Cost reporting and aggregation
"""

from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aragora.billing.cost_tracker import (
    Budget,
    BudgetAlert,
    BudgetAlertLevel,
    CostGranularity,
    CostReport,
    CostTracker,
    DebateBudgetExceededError,
    TokenUsage,
    get_cost_tracker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tracker():
    """Fresh CostTracker instance for each test."""
    return CostTracker()


@pytest.fixture
def sample_usage():
    """Sample TokenUsage for testing."""
    return TokenUsage(
        workspace_id="ws-123",
        agent_id="agent-456",
        agent_name="claude",
        debate_id="debate-789",
        provider="anthropic",
        model="claude-3-opus",
        tokens_in=1000,
        tokens_out=500,
        latency_ms=250.0,
        operation="debate_round",
    )


@pytest.fixture
def sample_budget():
    """Sample Budget for testing."""
    return Budget(
        id="budget-001",
        name="Test Budget",
        workspace_id="ws-123",
        monthly_limit_usd=Decimal("100.00"),
        daily_limit_usd=Decimal("10.00"),
        per_debate_limit_usd=Decimal("1.00"),
    )


# =============================================================================
# BudgetAlertLevel Tests
# =============================================================================


class TestBudgetAlertLevel:
    """Tests for BudgetAlertLevel enum."""

    def test_alert_levels(self):
        """Test all alert levels exist."""
        assert BudgetAlertLevel.INFO.value == "info"
        assert BudgetAlertLevel.WARNING.value == "warning"
        assert BudgetAlertLevel.CRITICAL.value == "critical"
        assert BudgetAlertLevel.EXCEEDED.value == "exceeded"


# =============================================================================
# DebateBudgetExceededError Tests
# =============================================================================


class TestDebateBudgetExceededError:
    """Tests for DebateBudgetExceededError exception."""

    def test_exception_attributes(self):
        """Test exception stores attributes."""
        error = DebateBudgetExceededError(
            debate_id="debate-123",
            current_cost=Decimal("1.50"),
            limit=Decimal("1.00"),
        )

        assert error.debate_id == "debate-123"
        assert error.current_cost == Decimal("1.50")
        assert error.limit == Decimal("1.00")

    def test_exception_message(self):
        """Test exception message formatting."""
        error = DebateBudgetExceededError(
            debate_id="debate-123",
            current_cost=Decimal("1.50"),
            limit=Decimal("1.00"),
        )

        assert "debate-123" in str(error)
        assert "1.50" in str(error)
        assert "1.00" in str(error)

    def test_custom_message(self):
        """Test custom exception message."""
        error = DebateBudgetExceededError(
            debate_id="debate-123",
            current_cost=Decimal("1.50"),
            limit=Decimal("1.00"),
            message="Custom error message",
        )

        assert str(error) == "Custom error message"


# =============================================================================
# TokenUsage Tests
# =============================================================================


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_create_usage(self, sample_usage):
        """Test creating token usage."""
        assert sample_usage.workspace_id == "ws-123"
        assert sample_usage.agent_name == "claude"
        assert sample_usage.tokens_in == 1000
        assert sample_usage.tokens_out == 500

    def test_calculate_cost(self, sample_usage):
        """Test cost calculation."""
        cost = sample_usage.calculate_cost()
        assert isinstance(cost, Decimal)
        assert sample_usage.cost_usd == cost

    def test_to_dict(self, sample_usage):
        """Test to_dict conversion."""
        data = sample_usage.to_dict()

        assert data["workspace_id"] == "ws-123"
        assert data["agent_name"] == "claude"
        assert data["tokens_in"] == 1000
        assert data["tokens_out"] == 500
        assert "timestamp" in data
        assert "id" in data

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "id": "usage-001",
            "workspace_id": "ws-123",
            "agent_name": "gemini",
            "provider": "google",
            "model": "gemini-pro",
            "tokens_in": 500,
            "tokens_out": 200,
            "cost_usd": "0.005",
            "timestamp": "2024-01-01T12:00:00+00:00",
        }

        usage = TokenUsage.from_dict(data)

        assert usage.id == "usage-001"
        assert usage.workspace_id == "ws-123"
        assert usage.agent_name == "gemini"
        assert usage.tokens_in == 500
        assert usage.cost_usd == Decimal("0.005")

    def test_default_values(self):
        """Test default field values."""
        usage = TokenUsage()

        assert usage.workspace_id == ""
        assert usage.tokens_in == 0
        assert usage.cost_usd == Decimal("0")
        assert usage.metadata == {}
        assert usage.id  # Should have auto-generated UUID


# =============================================================================
# Budget Tests
# =============================================================================


class TestBudget:
    """Tests for Budget dataclass."""

    def test_create_budget(self, sample_budget):
        """Test creating a budget."""
        assert sample_budget.name == "Test Budget"
        assert sample_budget.monthly_limit_usd == Decimal("100.00")
        assert sample_budget.daily_limit_usd == Decimal("10.00")

    def test_check_alert_level_no_limit(self):
        """Test no alert when no limit set."""
        budget = Budget(name="No Limit")
        assert budget.check_alert_level() is None

    def test_check_alert_level_info(self, sample_budget):
        """Test INFO alert at 50%."""
        sample_budget.current_monthly_spend = Decimal("50.00")
        assert sample_budget.check_alert_level() == BudgetAlertLevel.INFO

    def test_check_alert_level_warning(self, sample_budget):
        """Test WARNING alert at 75%."""
        sample_budget.current_monthly_spend = Decimal("75.00")
        assert sample_budget.check_alert_level() == BudgetAlertLevel.WARNING

    def test_check_alert_level_critical(self, sample_budget):
        """Test CRITICAL alert at 90%."""
        sample_budget.current_monthly_spend = Decimal("90.00")
        assert sample_budget.check_alert_level() == BudgetAlertLevel.CRITICAL

    def test_check_alert_level_exceeded(self, sample_budget):
        """Test EXCEEDED alert over 100%."""
        sample_budget.current_monthly_spend = Decimal("110.00")
        assert sample_budget.check_alert_level() == BudgetAlertLevel.EXCEEDED

    def test_check_alert_level_disabled_thresholds(self):
        """Test disabled alert thresholds."""
        budget = Budget(
            monthly_limit_usd=Decimal("100.00"),
            alert_threshold_50=False,
            alert_threshold_75=False,
            alert_threshold_90=False,
            current_monthly_spend=Decimal("50.00"),
        )

        # 50% but threshold disabled
        assert budget.check_alert_level() is None

    def test_to_dict(self, sample_budget):
        """Test to_dict conversion."""
        data = sample_budget.to_dict()

        assert data["name"] == "Test Budget"
        assert data["monthly_limit_usd"] == "100.00"
        assert data["workspace_id"] == "ws-123"


# =============================================================================
# BudgetAlert Tests
# =============================================================================


class TestBudgetAlert:
    """Tests for BudgetAlert dataclass."""

    def test_create_alert(self):
        """Test creating a budget alert."""
        alert = BudgetAlert(
            budget_id="budget-001",
            workspace_id="ws-123",
            level=BudgetAlertLevel.WARNING,
            message="75% of budget used",
            current_spend=Decimal("75.00"),
            limit=Decimal("100.00"),
            percentage=75.0,
        )

        assert alert.level == BudgetAlertLevel.WARNING
        assert alert.percentage == 75.0
        assert alert.acknowledged is False


# =============================================================================
# CostReport Tests
# =============================================================================


class TestCostReport:
    """Tests for CostReport dataclass."""

    def test_create_report(self):
        """Test creating a cost report."""
        report = CostReport(
            workspace_id="ws-123",
            total_cost_usd=Decimal("50.00"),
            total_tokens_in=100000,
            total_tokens_out=50000,
            total_api_calls=100,
        )

        assert report.total_cost_usd == Decimal("50.00")
        assert report.total_api_calls == 100

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = CostReport(
            workspace_id="ws-123",
            total_cost_usd=Decimal("50.00"),
            cost_by_agent={"claude": Decimal("30.00"), "gemini": Decimal("20.00")},
        )

        data = report.to_dict()

        assert data["workspace_id"] == "ws-123"
        assert data["total_cost_usd"] == "50.00"
        assert data["cost_by_agent"]["claude"] == "30.00"


# =============================================================================
# CostTracker Recording Tests
# =============================================================================


@pytest.mark.asyncio
class TestRecording:
    """Tests for recording token usage."""

    async def test_record_usage(self, tracker, sample_usage):
        """Test recording token usage."""
        await tracker.record(sample_usage)

        stats = tracker.get_workspace_stats("ws-123")
        assert Decimal(stats["total_cost_usd"]) >= Decimal("0")
        assert stats["total_tokens_in"] == 1000
        assert stats["total_tokens_out"] == 500
        assert stats["total_api_calls"] == 1

    async def test_record_multiple_usages(self, tracker):
        """Test recording multiple usages."""
        for i in range(5):
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name=f"agent-{i}",
                provider="anthropic",
                model="claude-3",
                tokens_in=100,
                tokens_out=50,
            )
            await tracker.record(usage)

        stats = tracker.get_workspace_stats("ws-123")
        assert stats["total_api_calls"] == 5
        assert stats["total_tokens_in"] == 500

    async def test_record_batch(self, tracker):
        """Test batch recording."""
        usages = [
            TokenUsage(
                workspace_id="ws-123",
                agent_name="claude",
                provider="anthropic",
                model="claude-3",
                tokens_in=100,
                tokens_out=50,
            )
            for _ in range(3)
        ]

        await tracker.record_batch(usages)

        stats = tracker.get_workspace_stats("ws-123")
        assert stats["total_api_calls"] == 3

    async def test_record_updates_debate_cost(self, tracker, sample_usage):
        """Test recording updates debate cost tracking."""
        await tracker.record(sample_usage)

        # Check debate cost is tracked
        assert "debate-789" in tracker._debate_costs

    async def test_record_by_agent_breakdown(self, tracker):
        """Test cost breakdown by agent."""
        for agent in ["claude", "gemini", "claude"]:
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name=agent,
                provider="test",
                model="test",
                tokens_in=100,
                tokens_out=50,
            )
            await tracker.record(usage)

        stats = tracker.get_workspace_stats("ws-123")
        assert "claude" in stats["cost_by_agent"]
        assert "gemini" in stats["cost_by_agent"]


# =============================================================================
# CostTracker Budget Tests
# =============================================================================


@pytest.mark.asyncio
class TestBudgetManagement:
    """Tests for budget management."""

    async def test_set_and_get_budget(self, tracker, sample_budget):
        """Test setting and getting a budget."""
        tracker.set_budget(sample_budget)

        budget = tracker.get_budget(workspace_id="ws-123")
        assert budget is not None
        assert budget.monthly_limit_usd == Decimal("100.00")

    async def test_get_budget_by_org(self, tracker):
        """Test getting budget by organization."""
        budget = Budget(
            id="org-budget",
            name="Org Budget",
            org_id="org-456",
            monthly_limit_usd=Decimal("1000.00"),
        )
        tracker.set_budget(budget)

        retrieved = tracker.get_budget(org_id="org-456")
        assert retrieved is not None
        assert retrieved.id == "org-budget"

    async def test_budget_no_match(self, tracker):
        """Test getting budget with no match."""
        budget = tracker.get_budget(workspace_id="nonexistent")
        assert budget is None

    async def test_budget_alert_callback(self, tracker, sample_budget):
        """Test budget alert callbacks."""
        alerts_received = []

        def callback(alert: BudgetAlert):
            alerts_received.append(alert)

        tracker.add_alert_callback(callback)
        tracker.set_budget(sample_budget)

        # Record enough to trigger 50% alert
        for _ in range(50):
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name="claude",
                provider="anthropic",
                model="claude-3",
                tokens_in=10000,
                tokens_out=5000,
                cost_usd=Decimal("1.00"),
            )
            await tracker.record(usage)

        # Should have received at least one alert
        assert len(alerts_received) > 0
        assert alerts_received[0].budget_id == "budget-001"

    async def test_remove_alert_callback(self, tracker):
        """Test removing alert callback."""
        callback = Mock()
        tracker.add_alert_callback(callback)
        tracker.remove_alert_callback(callback)

        assert callback not in tracker._alert_callbacks


# =============================================================================
# CostTracker Debate Budget Tests
# =============================================================================


@pytest.mark.asyncio
class TestDebateBudget:
    """Tests for per-debate budget enforcement."""

    async def test_set_debate_limit(self, tracker):
        """Test setting debate cost limit."""
        tracker.set_debate_limit("debate-001", Decimal("5.00"))

        status = tracker.check_debate_budget("debate-001")
        assert status["allowed"] is True
        assert status["limit"] == "5.00"
        assert status["remaining"] == "5.00"

    async def test_check_debate_budget_no_limit(self, tracker):
        """Test checking debate with no limit."""
        status = tracker.check_debate_budget("debate-no-limit")

        assert status["allowed"] is True
        assert status["limit"] == "unlimited"

    async def test_record_debate_cost(self, tracker):
        """Test recording cost against debate budget."""
        tracker.set_debate_limit("debate-001", Decimal("5.00"))

        status = tracker.record_debate_cost("debate-001", Decimal("2.00"))

        assert status["current_cost"] == "2.00"
        assert status["remaining"] == "3.00"
        assert status["allowed"] is True

    async def test_debate_budget_exceeded(self, tracker):
        """Test debate budget exceeded."""
        tracker.set_debate_limit("debate-001", Decimal("1.00"))
        tracker.record_debate_cost("debate-001", Decimal("1.50"))

        status = tracker.check_debate_budget("debate-001")

        assert status["allowed"] is False
        assert "exceeded" in status["message"].lower()

    async def test_debate_budget_with_estimate(self, tracker):
        """Test debate budget check with estimated cost."""
        tracker.set_debate_limit("debate-001", Decimal("5.00"))
        tracker.record_debate_cost("debate-001", Decimal("4.00"))

        # Check if $1.50 more is allowed
        status = tracker.check_debate_budget("debate-001", Decimal("1.50"))

        assert status["allowed"] is False

        # Check if $0.50 more is allowed
        status = tracker.check_debate_budget("debate-001", Decimal("0.50"))

        assert status["allowed"] is True

    async def test_clear_debate_budget(self, tracker):
        """Test clearing debate budget tracking."""
        tracker.set_debate_limit("debate-001", Decimal("5.00"))
        tracker.record_debate_cost("debate-001", Decimal("2.00"))

        tracker.clear_debate_budget("debate-001")

        # Should be cleared
        assert "debate-001" not in tracker._debate_costs
        assert "debate-001" not in tracker._debate_limits

    async def test_get_debate_cost(self, tracker):
        """Test getting debate cost breakdown."""
        usage = TokenUsage(
            workspace_id="ws-123",
            agent_name="claude",
            debate_id="debate-001",
            provider="anthropic",
            model="claude-3",
            tokens_in=1000,
            tokens_out=500,
        )
        await tracker.record(usage)

        cost_data = await tracker.get_debate_cost("debate-001")

        assert cost_data["debate_id"] == "debate-001"
        assert cost_data["total_tokens_in"] == 1000
        assert cost_data["total_tokens_out"] == 500


# =============================================================================
# CostTracker Report Tests
# =============================================================================


@pytest.mark.asyncio
class TestReporting:
    """Tests for cost reporting."""

    async def test_generate_report_empty(self, tracker):
        """Test generating report with no data."""
        report = await tracker.generate_report(workspace_id="ws-123")

        assert report.workspace_id == "ws-123"
        assert report.total_cost_usd == Decimal("0")

    async def test_generate_report_with_data(self, tracker):
        """Test generating report with data."""
        for agent in ["claude", "gemini", "claude"]:
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name=agent,
                provider="test",
                model="test-model",
                tokens_in=1000,
                tokens_out=500,
                cost_usd=Decimal("0.01"),
            )
            await tracker.record(usage)

        report = await tracker.generate_report(workspace_id="ws-123")

        assert report.total_api_calls == 3
        assert report.total_tokens_in == 3000
        assert "claude" in report.cost_by_agent

    async def test_generate_report_projections(self, tracker):
        """Test report includes projections."""
        for _ in range(10):
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name="claude",
                provider="test",
                model="test",
                tokens_in=1000,
                tokens_out=500,
                cost_usd=Decimal("1.00"),
            )
            await tracker.record(usage)

        report = await tracker.generate_report(workspace_id="ws-123")

        assert report.projected_daily_rate is not None
        assert report.projected_monthly_cost is not None

    async def test_get_agent_costs(self, tracker):
        """Test getting agent cost breakdown."""
        for agent, count in [("claude", 3), ("gemini", 2)]:
            for _ in range(count):
                usage = TokenUsage(
                    workspace_id="ws-123",
                    agent_name=agent,
                    provider="test",
                    model="test",
                    tokens_in=1000,
                    tokens_out=500,
                    cost_usd=Decimal("1.00"),
                )
                await tracker.record(usage)

        costs = await tracker.get_agent_costs("ws-123")

        assert "claude" in costs
        assert "gemini" in costs
        assert costs["claude"]["percentage"] > costs["gemini"]["percentage"]


# =============================================================================
# CostTracker Reset Tests
# =============================================================================


class TestBudgetReset:
    """Tests for budget reset functionality."""

    def test_reset_daily_budgets(self, tracker, sample_budget):
        """Test resetting daily budgets."""
        sample_budget.current_daily_spend = Decimal("5.00")
        tracker.set_budget(sample_budget)

        tracker.reset_daily_budgets()

        budget = tracker.get_budget(workspace_id="ws-123")
        assert budget.current_daily_spend == Decimal("0")

    def test_reset_monthly_budgets(self, tracker, sample_budget):
        """Test resetting monthly budgets."""
        sample_budget.current_monthly_spend = Decimal("50.00")
        sample_budget.current_daily_spend = Decimal("5.00")
        tracker.set_budget(sample_budget)

        tracker.reset_monthly_budgets()

        budget = tracker.get_budget(workspace_id="ws-123")
        assert budget.current_monthly_spend == Decimal("0")
        assert budget.current_daily_spend == Decimal("0")

    def test_reset_clears_alert_dedup(self, tracker, sample_budget):
        """Test reset clears alert deduplication."""
        tracker.set_budget(sample_budget)
        tracker._sent_alerts.add("test-alert-key")

        tracker.reset_monthly_budgets()

        assert len(tracker._sent_alerts) == 0


# =============================================================================
# CostTracker KM Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestKMIntegration:
    """Tests for Knowledge Mound integration."""

    async def test_set_km_adapter(self, tracker):
        """Test setting KM adapter."""
        mock_adapter = Mock()
        tracker.set_km_adapter(mock_adapter)

        assert tracker._km_adapter == mock_adapter

    async def test_query_cost_patterns_no_adapter(self, tracker):
        """Test querying cost patterns without adapter."""
        result = tracker.query_km_cost_patterns("ws-123")
        assert result == {}

    async def test_query_cost_patterns_with_adapter(self, tracker):
        """Test querying cost patterns with adapter."""
        mock_adapter = Mock()
        mock_adapter.get_cost_patterns.return_value = {
            "avg_cost": 0.05,
            "stddev": 0.01,
        }
        tracker.set_km_adapter(mock_adapter)

        result = tracker.query_km_cost_patterns("ws-123")

        assert result["avg_cost"] == 0.05
        mock_adapter.get_cost_patterns.assert_called_once_with("ws-123", None)

    async def test_query_workspace_alerts_no_adapter(self, tracker):
        """Test querying alerts without adapter."""
        result = tracker.query_km_workspace_alerts("ws-123")
        assert result == []

    async def test_detect_anomalies_no_adapter(self, tracker):
        """Test anomaly detection without adapter."""
        result = await tracker.detect_and_store_anomalies("ws-123")
        assert result == []


# =============================================================================
# CostGranularity Tests
# =============================================================================


class TestCostGranularity:
    """Tests for CostGranularity enum."""

    def test_granularity_values(self):
        """Test all granularity values."""
        assert CostGranularity.HOURLY.value == "hourly"
        assert CostGranularity.DAILY.value == "daily"
        assert CostGranularity.WEEKLY.value == "weekly"
        assert CostGranularity.MONTHLY.value == "monthly"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for global cost tracker singleton."""

    def test_get_cost_tracker_returns_instance(self):
        """Test getting singleton instance."""
        # Reset singleton for test
        import aragora.billing.cost_tracker as ct

        ct._cost_tracker = None

        with patch.object(ct, "UsageTracker", side_effect=ImportError):
            tracker = get_cost_tracker()
            assert isinstance(tracker, CostTracker)

            # Same instance returned
            tracker2 = get_cost_tracker()
            assert tracker is tracker2


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for cost tracking workflow."""

    async def test_full_workspace_tracking(self, tracker, sample_budget):
        """Test complete workspace cost tracking workflow."""
        # 1. Set budget
        tracker.set_budget(sample_budget)

        # 2. Record usage
        for i in range(10):
            usage = TokenUsage(
                workspace_id="ws-123",
                agent_name="claude" if i % 2 == 0 else "gemini",
                provider="anthropic" if i % 2 == 0 else "google",
                model="claude-3" if i % 2 == 0 else "gemini-pro",
                tokens_in=1000,
                tokens_out=500,
                debate_id=f"debate-{i % 3}",
                operation="debate_round",
            )
            await tracker.record(usage)

        # 3. Check stats
        stats = tracker.get_workspace_stats("ws-123")
        assert stats["total_api_calls"] == 10
        assert "claude" in stats["cost_by_agent"]
        assert "gemini" in stats["cost_by_agent"]

        # 4. Generate report
        report = await tracker.generate_report(workspace_id="ws-123")
        assert report.total_api_calls == 10

        # 5. Check agent costs
        agent_costs = await tracker.get_agent_costs("ws-123")
        assert len(agent_costs) == 2

    async def test_debate_budget_enforcement(self, tracker):
        """Test debate budget enforcement workflow."""
        # 1. Set debate limit
        tracker.set_debate_limit("debate-001", Decimal("0.05"))

        # 2. Record some costs
        tracker.record_debate_cost("debate-001", Decimal("0.02"))

        # 3. Check still within budget
        status = tracker.check_debate_budget("debate-001")
        assert status["allowed"] is True

        # 4. Record more, exceeding budget
        tracker.record_debate_cost("debate-001", Decimal("0.04"))

        # 5. Check now exceeded
        status = tracker.check_debate_budget("debate-001")
        assert status["allowed"] is False

        # 6. Clean up
        tracker.clear_debate_budget("debate-001")

    async def test_multiple_workspaces(self, tracker):
        """Test tracking multiple workspaces."""
        for ws in ["ws-1", "ws-2", "ws-3"]:
            for i in range(3):
                usage = TokenUsage(
                    workspace_id=ws,
                    agent_name="claude",
                    provider="anthropic",
                    model="claude-3",
                    tokens_in=1000 * (i + 1),
                    tokens_out=500,
                )
                await tracker.record(usage)

        # Each workspace tracked separately
        for ws in ["ws-1", "ws-2", "ws-3"]:
            stats = tracker.get_workspace_stats(ws)
            assert stats["total_api_calls"] == 3
