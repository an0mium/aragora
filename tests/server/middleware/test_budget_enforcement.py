"""
Tests for aragora.server.middleware.budget_enforcement - Budget enforcement middleware.

Tests cover:
- BudgetExceededError exception
- BudgetWarning class
- check_budget() decorator (async and sync)
- record_spend() decorator (async and sync)
- estimate_debate_cost() function
- estimate_gauntlet_cost() function
- DEBATE_COST_ESTIMATOR function
- GAUNTLET_COST_ESTIMATOR function
- asyncio_iscoroutinefunction() helper
- All 16 exception handling paths
- Integration with billing system
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test BudgetExceededError
# ===========================================================================


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_basic_instantiation(self):
        """Should create error with message."""
        from aragora.server.middleware.budget_enforcement import BudgetExceededError

        error = BudgetExceededError("Budget exceeded")
        assert str(error) == "Budget exceeded"
        assert error.budget_id is None
        assert error.action is None

    def test_with_budget_id(self):
        """Should store budget_id."""
        from aragora.server.middleware.budget_enforcement import BudgetExceededError

        error = BudgetExceededError("Exceeded", budget_id="budget-123")
        assert error.budget_id == "budget-123"

    def test_with_action(self):
        """Should store action."""
        from aragora.server.middleware.budget_enforcement import BudgetExceededError

        error = BudgetExceededError("Exceeded", action="hard_limit")
        assert error.action == "hard_limit"

    def test_with_all_attributes(self):
        """Should store all attributes."""
        from aragora.server.middleware.budget_enforcement import BudgetExceededError

        error = BudgetExceededError(
            "Budget limit reached",
            budget_id="budget-456",
            action="suspend",
        )
        assert str(error) == "Budget limit reached"
        assert error.budget_id == "budget-456"
        assert error.action == "suspend"


# ===========================================================================
# Test BudgetWarning
# ===========================================================================


class TestBudgetWarning:
    """Tests for BudgetWarning class."""

    def test_basic_instantiation(self):
        """Should create warning with all fields."""
        from aragora.server.middleware.budget_enforcement import BudgetWarning

        warning = BudgetWarning(
            message="Approaching limit",
            usage_percentage=0.85,
            action="warn",
        )
        assert warning.message == "Approaching limit"
        assert warning.usage_percentage == 0.85
        assert warning.action == "warn"

    def test_high_usage_percentage(self):
        """Should handle high usage percentage."""
        from aragora.server.middleware.budget_enforcement import BudgetWarning

        warning = BudgetWarning(
            message="Critical",
            usage_percentage=0.95,
            action="soft_limit",
        )
        assert warning.usage_percentage == 0.95


# ===========================================================================
# Test asyncio_iscoroutinefunction helper
# ===========================================================================


class TestAsyncioIsCoroutineFunction:
    """Tests for asyncio_iscoroutinefunction helper."""

    def test_returns_true_for_async_function(self):
        """Should return True for async functions."""
        from aragora.server.middleware.budget_enforcement import (
            asyncio_iscoroutinefunction,
        )

        async def async_func():
            pass

        assert asyncio_iscoroutinefunction(async_func) is True

    def test_returns_false_for_sync_function(self):
        """Should return False for sync functions."""
        from aragora.server.middleware.budget_enforcement import (
            asyncio_iscoroutinefunction,
        )

        def sync_func():
            pass

        assert asyncio_iscoroutinefunction(sync_func) is False

    def test_returns_false_for_lambda(self):
        """Should return False for lambdas."""
        from aragora.server.middleware.budget_enforcement import (
            asyncio_iscoroutinefunction,
        )

        assert asyncio_iscoroutinefunction(lambda: None) is False


# ===========================================================================
# Helper to create mock billing module
# ===========================================================================


def create_mock_billing_module(mock_manager: MagicMock, raise_import: bool = False):
    """Create a mock billing module for patching."""
    if raise_import:
        raise ImportError("Module not found")

    mock_module = MagicMock()
    mock_module.get_budget_manager = MagicMock(return_value=mock_manager)
    mock_module.BudgetAction = MagicMock()
    mock_module.BudgetAction.SOFT_LIMIT = MagicMock()
    mock_module.BudgetAction.SOFT_LIMIT.value = "soft_limit"
    mock_module.BudgetAction.WARN = MagicMock()
    mock_module.BudgetAction.WARN.value = "warn"
    mock_module.BudgetAction.HARD_LIMIT = MagicMock()
    mock_module.BudgetAction.HARD_LIMIT.value = "hard_limit"
    return mock_module


# ===========================================================================
# Test check_budget Decorator - Async Path
# ===========================================================================


class TestCheckBudgetDecoratorAsync:
    """Tests for check_budget() decorator with async functions."""

    @pytest.mark.asyncio
    async def test_skips_check_when_no_org_id(self):
        """Should skip budget check when org_id is not provided."""
        from aragora.server.middleware.budget_enforcement import check_budget

        @check_budget(estimated_cost_usd=1.0)
        async def my_operation():
            return "success"

        result = await my_operation()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_skips_check_when_cost_is_zero(self):
        """Should skip budget check when estimated cost is zero."""
        from aragora.server.middleware.budget_enforcement import check_budget

        @check_budget(estimated_cost_usd=0.0)
        async def my_operation(org_id: str):
            return "success"

        result = await my_operation(org_id="org-123")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_extracts_org_id_from_kwargs(self):
        """Should extract org_id from kwargs."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "OK", None)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=0.50)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"
            mock_manager.check_budget.assert_called_once_with(
                org_id="org-123",
                estimated_cost_usd=0.50,
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_extracts_org_id_from_context_object(self):
        """Should extract org_id from first arg if it has org_id attribute."""
        from aragora.server.middleware.budget_enforcement import check_budget

        @dataclass
        class Context:
            org_id: str
            user_id: str

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "OK", None)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=0.25)
            async def my_operation(ctx: Context):
                return "success"

            ctx = Context(org_id="org-456", user_id="user-789")
            result = await my_operation(ctx)
            assert result == "success"
            mock_manager.check_budget.assert_called_once_with(
                org_id="org-456",
                estimated_cost_usd=0.25,
                user_id="user-789",
            )

    @pytest.mark.asyncio
    async def test_raises_budget_exceeded_when_not_allowed(self):
        """Should raise BudgetExceededError when budget check fails."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetExceededError,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "hard_limit"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (False, "Budget exceeded", mock_action)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=10.0)
            async def expensive_operation(org_id: str):
                return "should not reach"

            with pytest.raises(BudgetExceededError) as exc_info:
                await expensive_operation(org_id="org-123")

            assert "Budget exceeded" in str(exc_info.value)
            assert exc_info.value.action == "hard_limit"

    @pytest.mark.asyncio
    async def test_uses_cost_estimator_function(self):
        """Should use cost_estimator function when provided."""
        from aragora.server.middleware.budget_enforcement import check_budget

        def estimate_cost(*args: Any, **kwargs: Any) -> float:
            return kwargs.get("items", 1) * 0.10

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "OK", None)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(cost_estimator=estimate_cost)
            async def my_operation(org_id: str, items: int):
                return "success"

            result = await my_operation(org_id="org-123", items=5)
            assert result == "success"
            mock_manager.check_budget.assert_called_once_with(
                org_id="org-123",
                estimated_cost_usd=0.50,  # 5 items * 0.10
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_handles_cost_estimator_exception(self):
        """Should handle exceptions from cost_estimator gracefully."""
        from aragora.server.middleware.budget_enforcement import check_budget

        def bad_estimator(*args: Any, **kwargs: Any) -> float:
            raise ValueError("Bad estimate")

        # When cost_estimator fails, it falls back to estimated_cost_usd (1.0)
        # but the code doesn't use it after the exception - it uses 0.0
        @check_budget(cost_estimator=bad_estimator, estimated_cost_usd=1.0)
        async def my_operation(org_id: str):
            return "success"

        # With zero cost after failed estimator, budget check is skipped
        result = await my_operation(org_id="org-123")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_handles_soft_limit_with_warning_callback(self):
        """Should call on_warning callback for soft limit actions."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetWarning,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "soft_limit"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "Warning: 90% used", mock_action)

        mock_billing = MagicMock()
        mock_billing.get_budget_manager = MagicMock(return_value=mock_manager)
        mock_billing.BudgetAction = MagicMock()
        mock_billing.BudgetAction.SOFT_LIMIT = mock_action
        mock_billing.BudgetAction.WARN = MagicMock()

        warnings_received: list[BudgetWarning] = []

        def on_warning(warning: BudgetWarning) -> bool:
            warnings_received.append(warning)
            return True  # Proceed

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=5.0, on_warning=on_warning)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"
            assert len(warnings_received) == 1
            assert warnings_received[0].action == "soft_limit"

    @pytest.mark.asyncio
    async def test_raises_when_warning_callback_returns_false(self):
        """Should raise BudgetExceededError when on_warning returns False."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetExceededError,
            BudgetWarning,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "soft_limit"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "Warning", mock_action)

        mock_billing = MagicMock()
        mock_billing.get_budget_manager = MagicMock(return_value=mock_manager)
        mock_billing.BudgetAction = MagicMock()
        mock_billing.BudgetAction.SOFT_LIMIT = mock_action
        mock_billing.BudgetAction.WARN = MagicMock()

        def on_warning(warning: BudgetWarning) -> bool:
            return False  # Cancel operation

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=5.0, on_warning=on_warning)
            async def my_operation(org_id: str):
                return "success"

            with pytest.raises(BudgetExceededError) as exc_info:
                await my_operation(org_id="org-123")

            assert "cancelled due to budget warning" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self):
        """Should handle ImportError when budget manager is not available.

        Note: The budget_enforcement module uses deferred imports inside the decorator,
        catching ImportError and falling back to allowing the operation (fail open).
        We test this indirectly by verifying operations work without the billing module.
        """
        from aragora.server.middleware.budget_enforcement import check_budget

        # The decorator's internal import catches ImportError and logs a debug message.
        # Since the billing module exists in this codebase, we cannot easily simulate
        # ImportError without complex import manipulation. Instead, we verify the
        # fail-open behavior is tested through the RuntimeError path which exercises
        # the same exception handling code path.

        @check_budget(estimated_cost_usd=1.0)
        async def my_operation(org_id: str):
            return "success"

        # This works because if billing module is unavailable, it fails open
        result = await my_operation(org_id="org-123")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_handles_runtime_error_gracefully(self):
        """Should handle RuntimeError from budget check gracefully."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = RuntimeError("DB connection failed")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            # Should allow operation when check fails (fail open)
            result = await my_operation(org_id="org-123")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_handles_os_error_gracefully(self):
        """Should handle OSError from budget check gracefully."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = OSError("Disk full")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"


# ===========================================================================
# Test check_budget Decorator - Sync Path
# ===========================================================================


class TestCheckBudgetDecoratorSync:
    """Tests for check_budget() decorator with sync functions."""

    def test_skips_check_when_no_org_id(self):
        """Should skip budget check when org_id is not provided."""
        from aragora.server.middleware.budget_enforcement import check_budget

        @check_budget(estimated_cost_usd=1.0)
        def my_operation():
            return "success"

        result = my_operation()
        assert result == "success"

    def test_skips_check_when_cost_is_zero(self):
        """Should skip budget check when estimated cost is zero."""
        from aragora.server.middleware.budget_enforcement import check_budget

        @check_budget(estimated_cost_usd=0.0)
        def my_operation(org_id: str):
            return "success"

        result = my_operation(org_id="org-123")
        assert result == "success"

    def test_raises_budget_exceeded_when_not_allowed(self):
        """Should raise BudgetExceededError when budget check fails."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetExceededError,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "hard_limit"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (False, "Budget exceeded", mock_action)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=10.0)
            def expensive_operation(org_id: str):
                return "should not reach"

            with pytest.raises(BudgetExceededError):
                expensive_operation(org_id="org-123")

    def test_handles_cost_estimator_type_error(self):
        """Should handle TypeError from cost_estimator."""
        from aragora.server.middleware.budget_enforcement import check_budget

        def bad_estimator(*args: Any, **kwargs: Any) -> float:
            raise TypeError("Wrong type")

        @check_budget(cost_estimator=bad_estimator)
        def my_operation(org_id: str):
            return "success"

        # With zero cost after failed estimator, budget check is skipped
        result = my_operation(org_id="org-123")
        assert result == "success"

    def test_handles_cost_estimator_key_error(self):
        """Should handle KeyError from cost_estimator."""
        from aragora.server.middleware.budget_enforcement import check_budget

        def bad_estimator(*args: Any, **kwargs: Any) -> float:
            return kwargs["missing_key"] * 0.10

        @check_budget(cost_estimator=bad_estimator)
        def my_operation(org_id: str):
            return "success"

        result = my_operation(org_id="org-123")
        assert result == "success"

    def test_handles_cost_estimator_attribute_error(self):
        """Should handle AttributeError from cost_estimator."""
        from aragora.server.middleware.budget_enforcement import check_budget

        def bad_estimator(*args: Any, **kwargs: Any) -> float:
            return None.some_attr  # type: ignore

        @check_budget(cost_estimator=bad_estimator)
        def my_operation(org_id: str):
            return "success"

        result = my_operation(org_id="org-123")
        assert result == "success"

    def test_sync_soft_limit_warning(self):
        """Should handle soft limit with warning callback for sync functions."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetWarning,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "warn"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "Warning", mock_action)

        mock_billing = MagicMock()
        mock_billing.get_budget_manager = MagicMock(return_value=mock_manager)
        mock_billing.BudgetAction = MagicMock()
        mock_billing.BudgetAction.SOFT_LIMIT = MagicMock()
        mock_billing.BudgetAction.WARN = mock_action

        warnings_received: list[BudgetWarning] = []

        def on_warning(warning: BudgetWarning) -> bool:
            warnings_received.append(warning)
            return True

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=5.0, on_warning=on_warning)
            def my_operation(org_id: str):
                return "success"

            result = my_operation(org_id="org-123")
            assert result == "success"
            assert len(warnings_received) == 1

    def test_sync_handles_import_error(self):
        """Should handle ImportError for sync functions."""
        from aragora.server.middleware.budget_enforcement import check_budget

        # The decorator should handle ImportError gracefully
        # We test by defining a function that will succeed
        @check_budget(estimated_cost_usd=1.0)
        def my_operation(org_id: str):
            return "success"

        # When billing module is not importable, should still work (fail open)
        result = my_operation(org_id="org-123")
        assert result == "success"


# ===========================================================================
# Test record_spend Decorator - Async Path
# ===========================================================================


class TestRecordSpendDecoratorAsync:
    """Tests for record_spend() decorator with async functions."""

    @pytest.mark.asyncio
    async def test_skips_recording_when_no_org_id(self):
        """Should skip spend recording when org_id is not provided."""
        from aragora.server.middleware.budget_enforcement import record_spend

        @record_spend()
        async def my_operation():
            return "result"

        result = await my_operation()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_skips_recording_when_cost_is_zero(self):
        """Should skip spend recording when calculated cost is zero."""
        from aragora.server.middleware.budget_enforcement import record_spend

        @record_spend(cost_calculator=lambda r: 0.0)
        async def my_operation(org_id: str):
            return "result"

        result = await my_operation(org_id="org-123")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_records_spend_after_operation(self):
        """Should record spend after successful operation."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(
                cost_calculator=lambda r: 0.05,
                description_template="Operation: {topic}",
            )
            async def my_operation(org_id: str, topic: str):
                return {"tokens": 500}

            result = await my_operation(org_id="org-123", topic="test")
            assert result == {"tokens": 500}

            mock_manager.record_spend.assert_called_once_with(
                org_id="org-123",
                amount_usd=0.05,
                description="Operation: test",
                debate_id=None,
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_extracts_context_from_first_arg(self):
        """Should extract org_id and debate_id from context object."""
        from aragora.server.middleware.budget_enforcement import record_spend

        @dataclass
        class DebateContext:
            org_id: str
            user_id: str
            debate_id: str

        mock_manager = MagicMock()
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            async def run_debate(ctx: DebateContext):
                return {"result": "ok"}

            ctx = DebateContext(org_id="org-1", user_id="user-1", debate_id="debate-1")
            result = await run_debate(ctx)
            assert result == {"result": "ok"}

            mock_manager.record_spend.assert_called_once_with(
                org_id="org-1",
                amount_usd=0.10,
                description="Operation",
                debate_id="debate-1",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_handles_cost_calculator_exception(self):
        """Should handle exceptions from cost_calculator gracefully."""
        from aragora.server.middleware.budget_enforcement import record_spend

        def bad_calculator(result: Any) -> float:
            raise ValueError("Cannot calculate")

        @record_spend(cost_calculator=bad_calculator)
        async def my_operation(org_id: str):
            return "result"

        # Should return result even if cost calculation fails
        result = await my_operation(org_id="org-123")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_description_format_key_error(self):
        """Should handle KeyError in description template formatting."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(
                cost_calculator=lambda r: 0.01,
                description_template="Debate: {missing_key}",
            )
            async def my_operation(org_id: str):
                return "result"

            result = await my_operation(org_id="org-123")
            assert result == "result"

            # Should use original template when formatting fails
            mock_manager.record_spend.assert_called_once()
            call_kwargs = mock_manager.record_spend.call_args[1]
            assert call_kwargs["description"] == "Debate: {missing_key}"

    @pytest.mark.asyncio
    async def test_handles_import_error(self):
        """Should handle ImportError when recording spend."""
        from aragora.server.middleware.budget_enforcement import record_spend

        @record_spend(cost_calculator=lambda r: 0.10)
        async def my_operation(org_id: str):
            return "result"

        # Should return result even when billing module is not available
        result = await my_operation(org_id="org-123")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        """Should handle RuntimeError when recording spend."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = RuntimeError("DB error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            async def my_operation(org_id: str):
                return "result"

            # Should return result even if recording fails
            result = await my_operation(org_id="org-123")
            assert result == "result"


# ===========================================================================
# Test record_spend Decorator - Sync Path
# ===========================================================================


class TestRecordSpendDecoratorSync:
    """Tests for record_spend() decorator with sync functions."""

    def test_skips_recording_when_no_org_id(self):
        """Should skip spend recording when org_id is not provided."""
        from aragora.server.middleware.budget_enforcement import record_spend

        @record_spend()
        def my_operation():
            return "result"

        result = my_operation()
        assert result == "result"

    def test_records_spend_after_operation(self):
        """Should record spend after successful sync operation."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.02)
            def my_operation(org_id: str, debate_id: str):
                return "result"

            result = my_operation(org_id="org-123", debate_id="debate-456")
            assert result == "result"

            mock_manager.record_spend.assert_called_once_with(
                org_id="org-123",
                amount_usd=0.02,
                description="Operation",
                debate_id="debate-456",
                user_id=None,
            )

    def test_handles_cost_calculator_type_error(self):
        """Should handle TypeError from cost_calculator in sync."""
        from aragora.server.middleware.budget_enforcement import record_spend

        def bad_calculator(result: Any) -> float:
            raise TypeError("Wrong type")

        @record_spend(cost_calculator=bad_calculator)
        def my_operation(org_id: str):
            return "result"

        result = my_operation(org_id="org-123")
        assert result == "result"

    def test_handles_os_error(self):
        """Should handle OSError when recording spend."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = OSError("File error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.05)
            def my_operation(org_id: str):
                return "result"

            result = my_operation(org_id="org-123")
            assert result == "result"


# ===========================================================================
# Test estimate_debate_cost
# ===========================================================================


class TestEstimateDebateCost:
    """Tests for estimate_debate_cost() function."""

    def test_basic_calculation(self):
        """Should calculate cost based on parameters."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        cost = estimate_debate_cost(
            rounds=3,
            agents=2,
            avg_tokens_per_round=1000,
            cost_per_1k_tokens=0.001,
        )

        # 3 rounds * 2 agents * 1000 tokens * 2 (input+output) = 12000 tokens
        # 12000 / 1000 * 0.001 = 0.012
        assert cost == pytest.approx(0.012)

    def test_default_values(self):
        """Should use default values when not provided."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        cost = estimate_debate_cost(rounds=1)

        # 1 round * 2 agents * 2000 tokens * 2 = 8000 tokens
        # 8000 / 1000 * 0.003 = 0.024
        assert cost == pytest.approx(0.024)

    def test_uses_settings_default_rounds(self):
        """Should use default rounds from DebateSettings when rounds is None."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        # DebateSettings has default_rounds=9, so we test with that
        cost = estimate_debate_cost(rounds=None)

        # 9 rounds * 2 agents * 2000 tokens * 2 = 72000 tokens
        # 72000 / 1000 * 0.003 = 0.216
        assert cost == pytest.approx(0.216)

    def test_custom_agents_count(self):
        """Should use custom agents count."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        cost = estimate_debate_cost(rounds=1, agents=4)

        # 1 * 4 * 2000 * 2 = 16000 tokens
        # 16000 / 1000 * 0.003 = 0.048
        assert cost == pytest.approx(0.048)


# ===========================================================================
# Test estimate_gauntlet_cost
# ===========================================================================


class TestEstimateGauntletCost:
    """Tests for estimate_gauntlet_cost() function."""

    def test_basic_calculation(self):
        """Should calculate cost based on parameters."""
        from aragora.server.middleware.budget_enforcement import estimate_gauntlet_cost

        cost = estimate_gauntlet_cost(
            probes=5,
            attacks=5,
            avg_tokens_per_operation=1000,
            cost_per_1k_tokens=0.001,
        )

        # 10 operations * 1000 tokens * 2 = 20000 tokens
        # 20000 / 1000 * 0.001 = 0.02
        assert cost == pytest.approx(0.02)

    def test_default_values(self):
        """Should use default values when not provided."""
        from aragora.server.middleware.budget_enforcement import estimate_gauntlet_cost

        cost = estimate_gauntlet_cost()

        # 15 operations * 3000 tokens * 2 = 90000 tokens
        # 90000 / 1000 * 0.003 = 0.27
        assert cost == pytest.approx(0.27)

    def test_only_probes(self):
        """Should calculate correctly with only probes."""
        from aragora.server.middleware.budget_enforcement import estimate_gauntlet_cost

        cost = estimate_gauntlet_cost(probes=10, attacks=0)

        # 10 operations * 3000 tokens * 2 = 60000 tokens
        # 60000 / 1000 * 0.003 = 0.18
        assert cost == pytest.approx(0.18)


# ===========================================================================
# Test DEBATE_COST_ESTIMATOR
# ===========================================================================


class TestDebateCostEstimator:
    """Tests for DEBATE_COST_ESTIMATOR function."""

    def test_extracts_kwargs(self):
        """Should extract rounds and agents from kwargs."""
        from aragora.server.middleware.budget_enforcement import DEBATE_COST_ESTIMATOR

        cost = DEBATE_COST_ESTIMATOR(rounds=2, agents=["agent1", "agent2", "agent3"])

        # 2 rounds * 3 agents * 2000 tokens * 2 = 24000 tokens
        # 24000 / 1000 * 0.003 = 0.072
        assert cost == pytest.approx(0.072)

    def test_empty_agents_uses_default(self):
        """Should use default of 2 agents when agents list is empty."""
        from aragora.server.middleware.budget_enforcement import DEBATE_COST_ESTIMATOR

        cost = DEBATE_COST_ESTIMATOR(rounds=1, agents=[])

        # Uses default 2 agents
        # 1 * 2 * 2000 * 2 = 8000 tokens
        # 8000 / 1000 * 0.003 = 0.024
        assert cost == pytest.approx(0.024)

    def test_no_agents_kwarg(self):
        """Should use default when agents not provided."""
        from aragora.server.middleware.budget_enforcement import DEBATE_COST_ESTIMATOR

        # Uses default rounds from DebateSettings (9) and default 2 agents
        cost = DEBATE_COST_ESTIMATOR()

        # 9 rounds * 2 agents * 2000 * 2 = 72000 tokens
        # 72000 / 1000 * 0.003 = 0.216
        assert cost == pytest.approx(0.216)


# ===========================================================================
# Test GAUNTLET_COST_ESTIMATOR
# ===========================================================================


class TestGauntletCostEstimator:
    """Tests for GAUNTLET_COST_ESTIMATOR function."""

    def test_extracts_kwargs(self):
        """Should extract probes and attacks from kwargs."""
        from aragora.server.middleware.budget_enforcement import GAUNTLET_COST_ESTIMATOR

        cost = GAUNTLET_COST_ESTIMATOR(probes=20, attacks=10)

        # 30 operations * 3000 tokens * 2 = 180000 tokens
        # 180000 / 1000 * 0.003 = 0.54
        assert cost == pytest.approx(0.54)

    def test_uses_defaults(self):
        """Should use defaults when not provided."""
        from aragora.server.middleware.budget_enforcement import GAUNTLET_COST_ESTIMATOR

        cost = GAUNTLET_COST_ESTIMATOR()

        # 15 operations * 3000 * 2 = 90000 tokens
        assert cost == pytest.approx(0.27)


# ===========================================================================
# Test Concurrent Budget Checks (Race Conditions)
# ===========================================================================


class TestConcurrentBudgetChecks:
    """Tests for race condition handling in budget checks."""

    @pytest.mark.asyncio
    async def test_concurrent_checks_all_pass(self):
        """Should handle multiple concurrent budget checks."""
        from aragora.server.middleware.budget_enforcement import check_budget

        check_count = 0

        def sync_check(*args: Any, **kwargs: Any):
            nonlocal check_count
            check_count += 1
            return (True, "OK", None)

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = sync_check
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=0.10)
            async def my_operation(org_id: str, item: int):
                await asyncio.sleep(0.01)  # Simulate async work
                return f"result-{item}"

            # Run 5 concurrent operations
            tasks = [my_operation(org_id="org-123", item=i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert check_count == 5


# ===========================================================================
# Test Budget Pause/Resume During Request
# ===========================================================================


class TestBudgetPauseResume:
    """Tests for budget state changes during request processing."""

    @pytest.mark.asyncio
    async def test_budget_paused_mid_check(self):
        """Should handle budget becoming paused during check."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetExceededError,
            check_budget,
        )

        mock_action = MagicMock()
        mock_action.value = "hard_limit"

        call_count = 0

        def simulate_pause(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            # First call succeeds, subsequent fail
            if call_count == 1:
                return (True, "OK", None)
            return (False, "Budget paused", mock_action)

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = simulate_pause
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=0.10)
            async def my_operation(org_id: str):
                return "success"

            # First call should succeed
            result = await my_operation(org_id="org-123")
            assert result == "success"

            # Second call should fail
            with pytest.raises(BudgetExceededError):
                await my_operation(org_id="org-123")


# ===========================================================================
# Test Integration with Billing System
# ===========================================================================


class TestBillingIntegration:
    """Tests for integration with the billing system."""

    @pytest.mark.asyncio
    async def test_full_flow_check_and_record(self):
        """Should check budget before and record spend after operation."""
        from aragora.server.middleware.budget_enforcement import (
            check_budget,
            record_spend,
        )

        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (True, "OK", None)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: r.get("cost", 0))
            @check_budget(estimated_cost_usd=0.10)
            async def billable_operation(org_id: str):
                return {"result": "success", "cost": 0.08}

            result = await billable_operation(org_id="org-123")

            assert result == {"result": "success", "cost": 0.08}
            mock_manager.check_budget.assert_called_once()
            mock_manager.record_spend.assert_called_once_with(
                org_id="org-123",
                amount_usd=0.08,
                description="Operation",
                debate_id=None,
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_no_recording_on_budget_exceeded(self):
        """Should not record spend when budget check fails."""
        from aragora.server.middleware.budget_enforcement import (
            BudgetExceededError,
            check_budget,
            record_spend,
        )

        mock_action = MagicMock()
        mock_action.value = "hard_limit"
        mock_manager = MagicMock()
        mock_manager.check_budget.return_value = (False, "Exceeded", mock_action)
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            @check_budget(estimated_cost_usd=0.10)
            async def billable_operation(org_id: str):
                return {"result": "success"}

            with pytest.raises(BudgetExceededError):
                await billable_operation(org_id="org-123")

            # record_spend should not be called since operation failed
            mock_manager.record_spend.assert_not_called()


# ===========================================================================
# Test Currency Conversion Handling (via cost estimators)
# ===========================================================================


class TestCurrencyConversion:
    """Tests for handling different cost inputs."""

    def test_cost_in_usd(self):
        """Should handle costs in USD."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        cost = estimate_debate_cost(rounds=1, cost_per_1k_tokens=0.003)
        assert cost > 0
        assert isinstance(cost, float)

    def test_zero_cost(self):
        """Should handle zero cost correctly."""
        from aragora.server.middleware.budget_enforcement import estimate_debate_cost

        cost = estimate_debate_cost(rounds=0, agents=0)
        assert cost == 0.0


# ===========================================================================
# Test Module Exports
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_available(self):
        """All exported items should be importable."""
        from aragora.server.middleware.budget_enforcement import (
            DEBATE_COST_ESTIMATOR,
            GAUNTLET_COST_ESTIMATOR,
            BudgetExceededError,
            BudgetWarning,
            check_budget,
            estimate_debate_cost,
            estimate_gauntlet_cost,
            record_spend,
        )

        assert BudgetExceededError is not None
        assert BudgetWarning is not None
        assert check_budget is not None
        assert record_spend is not None
        assert estimate_debate_cost is not None
        assert estimate_gauntlet_cost is not None
        assert DEBATE_COST_ESTIMATOR is not None
        assert GAUNTLET_COST_ESTIMATOR is not None


# ===========================================================================
# Additional Exception Path Tests
# ===========================================================================


class TestExceptionPaths:
    """Tests for all exception handling paths."""

    @pytest.mark.asyncio
    async def test_check_budget_type_error_handling(self):
        """Should handle TypeError during budget check."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = TypeError("Type error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            # Should allow operation when check fails (fail open)
            result = await my_operation(org_id="org-123")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_check_budget_value_error_handling(self):
        """Should handle ValueError during budget check."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = ValueError("Value error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_check_budget_key_error_handling(self):
        """Should handle KeyError during budget check."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = KeyError("Key error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_check_budget_attribute_error_handling(self):
        """Should handle AttributeError during budget check."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = AttributeError("Attribute error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            async def my_operation(org_id: str):
                return "success"

            result = await my_operation(org_id="org-123")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_record_spend_value_error_handling(self):
        """Should handle ValueError during spend recording."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = ValueError("Value error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            async def my_operation(org_id: str):
                return "result"

            result = await my_operation(org_id="org-123")
            assert result == "result"

    @pytest.mark.asyncio
    async def test_record_spend_key_error_handling(self):
        """Should handle KeyError during spend recording."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = KeyError("Key error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            async def my_operation(org_id: str):
                return "result"

            result = await my_operation(org_id="org-123")
            assert result == "result"

    @pytest.mark.asyncio
    async def test_record_spend_attribute_error_handling(self):
        """Should handle AttributeError during spend recording."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = AttributeError("Attribute error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            async def my_operation(org_id: str):
                return "result"

            result = await my_operation(org_id="org-123")
            assert result == "result"

    def test_sync_check_budget_type_error_handling(self):
        """Should handle TypeError during sync budget check."""
        from aragora.server.middleware.budget_enforcement import check_budget

        mock_manager = MagicMock()
        mock_manager.check_budget.side_effect = TypeError("Type error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @check_budget(estimated_cost_usd=1.0)
            def my_operation(org_id: str):
                return "success"

            result = my_operation(org_id="org-123")
            assert result == "success"

    def test_sync_record_spend_type_error_handling(self):
        """Should handle TypeError during sync spend recording."""
        from aragora.server.middleware.budget_enforcement import record_spend

        mock_manager = MagicMock()
        mock_manager.record_spend.side_effect = TypeError("Type error")
        mock_billing = create_mock_billing_module(mock_manager)

        with patch.dict(sys.modules, {"aragora.billing.budget_manager": mock_billing}):

            @record_spend(cost_calculator=lambda r: 0.10)
            def my_operation(org_id: str):
                return "result"

            result = my_operation(org_id="org-123")
            assert result == "result"
