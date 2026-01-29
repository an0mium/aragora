"""Tests for Budget Coordinator module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.budget_coordinator import BudgetCoordinator


class TestBudgetCoordinatorInit:
    """Test BudgetCoordinator initialization."""

    def test_init_with_org_id(self):
        """Test initialization with organization ID."""
        coordinator = BudgetCoordinator(org_id="org-123")
        assert coordinator.org_id == "org-123"
        assert coordinator.user_id is None

    def test_init_with_user_id(self):
        """Test initialization with user ID."""
        coordinator = BudgetCoordinator(org_id="org-123", user_id="user-456")
        assert coordinator.org_id == "org-123"
        assert coordinator.user_id == "user-456"

    def test_init_without_org_id(self):
        """Test initialization without organization ID."""
        coordinator = BudgetCoordinator()
        assert coordinator.org_id is None
        assert coordinator.user_id is None

    def test_cost_constants_defined(self):
        """Test cost estimation constants are defined."""
        assert BudgetCoordinator.ESTIMATED_DEBATE_COST_USD > 0
        assert BudgetCoordinator.ESTIMATED_ROUND_COST_USD > 0
        assert BudgetCoordinator.ESTIMATED_MESSAGE_COST_USD > 0


class TestCheckBudgetBeforeDebate:
    """Test pre-debate budget checks."""

    def test_skips_check_without_org_id(self):
        """Test budget check is skipped without org_id."""
        coordinator = BudgetCoordinator()
        # Should not raise
        coordinator.check_budget_before_debate("debate-123")

    def test_allows_when_budget_available(self):
        """Test allows debate when budget is available."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "", None)

        # Create a mock module with BudgetAction enum
        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()
        mock_budget_module.BudgetAction.SOFT_LIMIT = "SOFT_LIMIT"

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            # Should not raise
            coordinator.check_budget_before_debate("debate-123")

        mock_manager.check_budget.assert_called_once()
        call_args = mock_manager.check_budget.call_args
        assert call_args.kwargs["org_id"] == "org-123"
        assert call_args.kwargs["estimated_cost_usd"] == BudgetCoordinator.ESTIMATED_DEBATE_COST_USD

    def test_raises_when_budget_exceeded(self):
        """Test raises BudgetExceededError when budget is exhausted."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (False, "Monthly limit reached", None)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            from aragora.exceptions import BudgetExceededError

            with pytest.raises(BudgetExceededError, match="Budget limit reached"):
                coordinator.check_budget_before_debate("debate-123")

    def test_warns_on_soft_limit(self):
        """Test warns but allows on soft limit."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        soft_limit = MagicMock()
        mock_manager.check_budget.return_value = (True, "Approaching limit", soft_limit)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()
        mock_budget_module.BudgetAction.SOFT_LIMIT = soft_limit

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            # Should not raise
            coordinator.check_budget_before_debate("debate-123")

    def test_handles_import_error_gracefully(self):
        """Test handles missing budget manager gracefully."""
        coordinator = BudgetCoordinator(org_id="org-123")

        # Remove the module from cache if it exists
        for key in list(sys.modules.keys()):
            if "aragora.billing.budget_manager" in key:
                del sys.modules[key]

        with patch.dict(
            sys.modules,
            {"aragora.billing.budget_manager": None},
        ):
            # Should not raise - handles ImportError internally
            coordinator.check_budget_before_debate("debate-123")

    def test_includes_user_id_when_provided(self):
        """Test includes user_id in budget check when provided."""
        coordinator = BudgetCoordinator(org_id="org-123", user_id="user-456")

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "", None)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.check_budget_before_debate("debate-123")

        call_args = mock_manager.check_budget.call_args
        assert call_args.kwargs["user_id"] == "user-456"


class TestCheckBudgetMidDebate:
    """Test mid-debate budget checks."""

    def test_returns_true_without_org_id(self):
        """Test returns True without org_id."""
        coordinator = BudgetCoordinator()
        allowed, reason = coordinator.check_budget_mid_debate("debate-123", round_num=2)
        assert allowed is True
        assert reason == ""

    def test_returns_true_when_budget_available(self):
        """Test returns True when budget is available."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "", None)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            allowed, reason = coordinator.check_budget_mid_debate("debate-123", round_num=2)

        assert allowed is True
        assert reason == ""
        call_args = mock_manager.check_budget.call_args
        assert call_args.kwargs["estimated_cost_usd"] == BudgetCoordinator.ESTIMATED_ROUND_COST_USD

    def test_returns_false_when_budget_exceeded(self):
        """Test returns False when budget is exceeded."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (False, "Budget exhausted", None)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            allowed, reason = coordinator.check_budget_mid_debate("debate-123", round_num=3)

        assert allowed is False
        assert reason == "Budget exhausted"

    def test_continues_on_soft_limit(self):
        """Test continues with soft limit warning."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        soft_limit = MagicMock()
        mock_manager.check_budget.return_value = (True, "Approaching limit", soft_limit)

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()
        mock_budget_module.BudgetAction.SOFT_LIMIT = soft_limit

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            allowed, reason = coordinator.check_budget_mid_debate("debate-123", round_num=2)

        assert allowed is True

    def test_handles_connection_errors(self):
        """Test handles connection errors gracefully (fail open)."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = ConnectionError("Network error")

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager
        mock_budget_module.BudgetAction = MagicMock()

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            allowed, reason = coordinator.check_budget_mid_debate("debate-123", round_num=2)

        # Should fail open for availability
        assert allowed is True


class TestRecordDebateCost:
    """Test post-debate cost recording."""

    def test_skips_recording_without_org_id(self):
        """Test skips recording without org_id."""
        coordinator = BudgetCoordinator()
        mock_result = MagicMock()

        # Should not call budget manager
        coordinator.record_debate_cost("debate-123", mock_result)

    def test_records_cost_from_extensions(self):
        """Test records cost from extensions object."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.task = "Test debate task"

        mock_extensions = MagicMock()
        mock_extensions.total_cost_usd = 0.15

        mock_manager = MagicMock()

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.record_debate_cost("debate-123", mock_result, mock_extensions)

        mock_manager.record_spend.assert_called_once()
        call_args = mock_manager.record_spend.call_args
        assert call_args.kwargs["amount_usd"] == 0.15
        assert call_args.kwargs["org_id"] == "org-123"
        assert call_args.kwargs["debate_id"] == "debate-123"

    def test_records_cost_from_result_metadata(self):
        """Test records cost from result metadata."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.task = "Test debate task"
        mock_result.metadata = {"total_cost_usd": 0.12}

        mock_manager = MagicMock()

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.record_debate_cost("debate-123", mock_result)

        call_args = mock_manager.record_spend.call_args
        assert call_args.kwargs["amount_usd"] == 0.12

    def test_fallback_to_message_count_estimate(self):
        """Test fallback to message count estimate."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.task = "Test debate task"
        mock_result.metadata = {}
        mock_result.messages = ["msg1", "msg2", "msg3"]  # 3 messages
        mock_result.critiques = ["crit1", "crit2"]  # 2 critiques

        mock_manager = MagicMock()

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.record_debate_cost("debate-123", mock_result)

        call_args = mock_manager.record_spend.call_args
        expected_cost = 5 * BudgetCoordinator.ESTIMATED_MESSAGE_COST_USD
        assert call_args.kwargs["amount_usd"] == expected_cost

    def test_includes_user_id_when_provided(self):
        """Test includes user_id when provided."""
        coordinator = BudgetCoordinator(org_id="org-123", user_id="user-456")

        mock_result = MagicMock()
        mock_result.task = "Test"
        mock_result.metadata = {"total_cost_usd": 0.10}

        mock_manager = MagicMock()

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.record_debate_cost("debate-123", mock_result)

        call_args = mock_manager.record_spend.call_args
        assert call_args.kwargs["user_id"] == "user-456"

    def test_skips_zero_cost(self):
        """Test skips recording when cost is zero."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.task = "Test"
        mock_result.metadata = {}
        mock_result.messages = []
        mock_result.critiques = []

        mock_manager = MagicMock()

        mock_budget_module = MagicMock()
        mock_budget_module.get_budget_manager.return_value = mock_manager

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_budget_module}):
            coordinator.record_debate_cost("debate-123", mock_result)

        # Should not call record_spend for zero cost
        mock_manager.record_spend.assert_not_called()


class TestCalculateActualCost:
    """Test actual cost calculation."""

    def test_prioritizes_extensions_cost(self):
        """Test extensions cost takes priority."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.metadata = {"total_cost_usd": 0.05}
        mock_result.messages = ["msg1"] * 100  # Would be $1.00 fallback

        mock_extensions = MagicMock()
        mock_extensions.total_cost_usd = 0.20

        cost = coordinator._calculate_actual_cost(mock_result, mock_extensions)
        assert cost == 0.20

    def test_uses_metadata_when_no_extensions(self):
        """Test uses metadata when no extensions provided."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.metadata = {"total_cost_usd": 0.08}
        mock_result.messages = ["msg1"] * 100

        cost = coordinator._calculate_actual_cost(mock_result, None)
        assert cost == 0.08

    def test_uses_message_count_fallback(self):
        """Test uses message count as fallback."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.metadata = {}
        mock_result.messages = ["msg1", "msg2"]
        mock_result.critiques = ["crit1"]

        cost = coordinator._calculate_actual_cost(mock_result, None)
        expected = 3 * BudgetCoordinator.ESTIMATED_MESSAGE_COST_USD
        assert cost == expected

    def test_handles_missing_messages(self):
        """Test handles missing messages list."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.metadata = {}
        mock_result.messages = None
        mock_result.critiques = None

        cost = coordinator._calculate_actual_cost(mock_result, None)
        assert cost == 0.0

    def test_extensions_with_zero_cost_falls_back(self):
        """Test extensions with zero cost falls back to metadata."""
        coordinator = BudgetCoordinator(org_id="org-123")

        mock_result = MagicMock()
        mock_result.metadata = {"total_cost_usd": 0.05}
        mock_result.messages = []
        mock_result.critiques = []

        mock_extensions = MagicMock()
        mock_extensions.total_cost_usd = 0.0

        cost = coordinator._calculate_actual_cost(mock_result, mock_extensions)
        assert cost == 0.05  # Falls back to metadata
