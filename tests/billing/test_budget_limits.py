"""
Tests for per-debate budget limit enforcement.

Tests cover:
- Setting per-debate cost limits
- Checking budget status
- Recording costs against debate budget
- Budget exceeded behavior
- Cleanup after debate completion
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

from aragora.billing.cost_tracker import (
    CostTracker,
    DebateBudgetExceededError,
    TokenUsage,
)


@pytest.fixture
def cost_tracker():
    """Create a cost tracker for testing."""
    return CostTracker()


class TestPerDebateBudgetLimits:
    """Tests for per-debate cost limit functionality."""

    def test_set_debate_limit(self, cost_tracker):
        """Can set a cost limit for a debate."""
        cost_tracker.set_debate_limit("debate_123", Decimal("5.00"))

        assert "debate_123" in cost_tracker._debate_limits
        assert cost_tracker._debate_limits["debate_123"] == Decimal("5.00")
        assert "debate_123" in cost_tracker._debate_costs
        assert cost_tracker._debate_costs["debate_123"] == Decimal("0")

    def test_check_budget_no_limit(self, cost_tracker):
        """Check budget returns allowed when no limit set."""
        result = cost_tracker.check_debate_budget("debate_456")

        assert result["allowed"] is True
        assert result["limit"] == "unlimited"
        assert result["remaining"] == "unlimited"

    def test_check_budget_within_limit(self, cost_tracker):
        """Check budget returns allowed when within limit."""
        cost_tracker.set_debate_limit("debate_123", Decimal("10.00"))
        cost_tracker._debate_costs["debate_123"] = Decimal("3.00")

        result = cost_tracker.check_debate_budget("debate_123")

        assert result["allowed"] is True
        assert result["current_cost"] == "3.00"
        assert result["limit"] == "10.00"
        assert result["remaining"] == "7.00"

    def test_check_budget_exceeded(self, cost_tracker):
        """Check budget returns not allowed when limit exceeded."""
        cost_tracker.set_debate_limit("debate_123", Decimal("5.00"))
        cost_tracker._debate_costs["debate_123"] = Decimal("5.00")

        result = cost_tracker.check_debate_budget(
            "debate_123",
            estimated_cost_usd=Decimal("1.00"),
        )

        assert result["allowed"] is False
        assert "exceeded" in result["message"].lower()

    def test_check_budget_with_estimate(self, cost_tracker):
        """Check budget considers estimated cost."""
        cost_tracker.set_debate_limit("debate_123", Decimal("10.00"))
        cost_tracker._debate_costs["debate_123"] = Decimal("9.00")

        # Small estimate - should pass
        result = cost_tracker.check_debate_budget(
            "debate_123",
            estimated_cost_usd=Decimal("0.50"),
        )
        assert result["allowed"] is True

        # Large estimate - should fail
        result = cost_tracker.check_debate_budget(
            "debate_123",
            estimated_cost_usd=Decimal("2.00"),
        )
        assert result["allowed"] is False

    def test_record_debate_cost(self, cost_tracker):
        """Recording cost updates debate budget."""
        cost_tracker.set_debate_limit("debate_123", Decimal("10.00"))

        result = cost_tracker.record_debate_cost("debate_123", Decimal("2.50"))

        assert cost_tracker._debate_costs["debate_123"] == Decimal("2.50")
        assert result["current_cost"] == "2.50"

    def test_record_debate_cost_cumulative(self, cost_tracker):
        """Multiple recordings accumulate cost."""
        cost_tracker.set_debate_limit("debate_123", Decimal("10.00"))

        cost_tracker.record_debate_cost("debate_123", Decimal("2.00"))
        cost_tracker.record_debate_cost("debate_123", Decimal("3.00"))
        result = cost_tracker.record_debate_cost("debate_123", Decimal("1.50"))

        assert cost_tracker._debate_costs["debate_123"] == Decimal("6.50")
        assert result["current_cost"] == "6.50"

    def test_clear_debate_budget(self, cost_tracker):
        """Clearing budget removes tracking data."""
        cost_tracker.set_debate_limit("debate_123", Decimal("10.00"))
        cost_tracker._debate_costs["debate_123"] = Decimal("5.00")

        cost_tracker.clear_debate_budget("debate_123")

        assert "debate_123" not in cost_tracker._debate_limits
        assert "debate_123" not in cost_tracker._debate_costs

    def test_clear_nonexistent_budget(self, cost_tracker):
        """Clearing nonexistent budget doesn't raise."""
        cost_tracker.clear_debate_budget("nonexistent_debate")
        # Should not raise


class TestDebateBudgetExceededError:
    """Tests for the budget exceeded exception."""

    def test_exception_attributes(self):
        """Exception stores debate info."""
        exc = DebateBudgetExceededError(
            debate_id="debate_123",
            current_cost=Decimal("12.50"),
            limit=Decimal("10.00"),
        )

        assert exc.debate_id == "debate_123"
        assert exc.current_cost == Decimal("12.50")
        assert exc.limit == Decimal("10.00")
        assert "debate_123" in str(exc)
        assert "12.50" in str(exc)
        assert "10.00" in str(exc)

    def test_exception_custom_message(self):
        """Exception can have custom message."""
        exc = DebateBudgetExceededError(
            debate_id="debate_123",
            current_cost=Decimal("15.00"),
            limit=Decimal("10.00"),
            message="Custom budget error message",
        )

        assert str(exc) == "Custom budget error message"


class TestArenaExtensionsBudget:
    """Tests for budget integration in ArenaExtensions."""

    def test_setup_debate_budget(self):
        """ArenaExtensions sets up debate budget."""
        from aragora.debate.extensions import ArenaExtensions

        extensions = ArenaExtensions(
            debate_budget_limit_usd=5.00,
        )

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_get.return_value = mock_tracker

            extensions.setup_debate_budget("debate_123")

            mock_tracker.set_debate_limit.assert_called_once()
            call_args = mock_tracker.set_debate_limit.call_args
            assert call_args[0][0] == "debate_123"
            assert call_args[0][1] == Decimal("5.00")

    def test_setup_debate_budget_no_limit(self):
        """Setup does nothing when no limit configured."""
        from aragora.debate.extensions import ArenaExtensions

        extensions = ArenaExtensions(
            debate_budget_limit_usd=None,
        )

        with patch("aragora.billing.cost_tracker.get_cost_tracker") as mock_get:
            extensions.setup_debate_budget("debate_123")
            mock_get.assert_not_called()

    def test_check_debate_budget(self):
        """ArenaExtensions checks debate budget."""
        from aragora.debate.extensions import ArenaExtensions

        extensions = ArenaExtensions(
            debate_budget_limit_usd=10.00,
        )

        mock_tracker = MagicMock()
        mock_tracker.check_debate_budget.return_value = {
            "allowed": True,
            "current_cost": "5.00",
            "limit": "10.00",
        }
        extensions.cost_tracker = mock_tracker

        result = extensions.check_debate_budget("debate_123")

        assert result["allowed"] is True
        mock_tracker.check_debate_budget.assert_called_once_with("debate_123")

    def test_cleanup_debate_budget(self):
        """ArenaExtensions cleans up debate budget."""
        from aragora.debate.extensions import ArenaExtensions

        extensions = ArenaExtensions()
        mock_tracker = MagicMock()
        extensions.cost_tracker = mock_tracker

        extensions.cleanup_debate_budget("debate_123")

        mock_tracker.clear_debate_budget.assert_called_once_with("debate_123")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
