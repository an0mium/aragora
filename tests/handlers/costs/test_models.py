"""Comprehensive tests for cost visibility data models and CostTracker integration.

Tests cover aragora/server/handlers/costs/models.py (328 lines):

  TestCostEntry             - CostEntry dataclass creation and fields
  TestBudgetAlert           - BudgetAlert dataclass creation and defaults
  TestCostSummary           - CostSummary dataclass with default factories
  TestIsDemoMode            - _is_demo_mode() feature flag check
  TestGetCostTracker        - _get_cost_tracker() singleton init and fallback
  TestRecordCost            - record_cost() async/sync recording paths
  TestGetActiveAlerts       - _get_active_alerts() budget alert retrieval
  TestEmptyCostSummary      - _empty_cost_summary() default values
  TestGetCostSummary        - get_cost_summary() with tracker/demo/empty paths
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.costs.models import (
    BudgetAlert,
    CostEntry,
    CostSummary,
    _empty_cost_summary,
    _get_active_alerts,
    _get_cost_tracker,
    _is_demo_mode,
    get_cost_summary,
    record_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _make_tracker(**overrides):
    """Create a mock CostTracker with sensible defaults."""
    tracker = MagicMock()
    tracker.record = AsyncMock()
    tracker.get_budget = MagicMock(return_value=None)
    tracker.generate_report = AsyncMock(
        return_value=MagicMock(
            total_cost_usd=Decimal("0"),
            total_tokens_in=0,
            total_tokens_out=0,
            total_api_calls=0,
            cost_by_provider={},
            cost_by_operation={},
            cost_over_time=[],
        )
    )
    for k, v in overrides.items():
        setattr(tracker, k, v)
    return tracker


def _make_report(**overrides):
    """Create a mock cost report."""
    defaults = dict(
        total_cost_usd=Decimal("125.50"),
        total_tokens_in=2_000_000,
        total_tokens_out=1_125_000,
        total_api_calls=12_550,
        cost_by_provider={"Anthropic": Decimal("77.31"), "OpenAI": Decimal("34.64")},
        cost_by_operation={"Debates": Decimal("54.22"), "Code Review": Decimal("22.46")},
        cost_over_time=[
            {"date": "2026-02-20", "cost": 18.50},
            {"date": "2026-02-21", "cost": 21.00},
        ],
    )
    defaults.update(overrides)
    report = MagicMock()
    for k, v in defaults.items():
        setattr(report, k, v)
    return report


def _make_budget(**overrides):
    """Create a mock budget object."""
    defaults = dict(
        monthly_limit_usd=Decimal("500.00"),
        current_monthly_spend=Decimal("125.50"),
    )
    defaults.update(overrides)
    budget = MagicMock()
    for k, v in defaults.items():
        setattr(budget, k, v)
    return budget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cost_tracker():
    """Reset the module-level _cost_tracker singleton before each test."""
    import aragora.server.handlers.costs.models as models_mod

    original = models_mod._cost_tracker
    models_mod._cost_tracker = None
    yield
    models_mod._cost_tracker = original


# ============================================================================
# TestCostEntry
# ============================================================================


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_create_with_all_fields(self):
        now = _now()
        entry = CostEntry(
            timestamp=now,
            provider="anthropic",
            feature="debate",
            tokens_input=1000,
            tokens_output=500,
            cost=0.0045,
            model="claude-3-opus",
            workspace_id="ws-1",
            user_id="user-1",
        )
        assert entry.timestamp == now
        assert entry.provider == "anthropic"
        assert entry.feature == "debate"
        assert entry.tokens_input == 1000
        assert entry.tokens_output == 500
        assert entry.cost == 0.0045
        assert entry.model == "claude-3-opus"
        assert entry.workspace_id == "ws-1"
        assert entry.user_id == "user-1"

    def test_user_id_defaults_to_none(self):
        entry = CostEntry(
            timestamp=_now(),
            provider="openai",
            feature="code_review",
            tokens_input=500,
            tokens_output=200,
            cost=0.002,
            model="gpt-4o",
            workspace_id="ws-default",
        )
        assert entry.user_id is None

    def test_zero_cost_entry(self):
        entry = CostEntry(
            timestamp=_now(),
            provider="local",
            feature="cache_hit",
            tokens_input=0,
            tokens_output=0,
            cost=0.0,
            model="cached",
            workspace_id="ws-1",
        )
        assert entry.cost == 0.0
        assert entry.tokens_input == 0

    def test_large_token_counts(self):
        entry = CostEntry(
            timestamp=_now(),
            provider="anthropic",
            feature="large_debate",
            tokens_input=10_000_000,
            tokens_output=5_000_000,
            cost=150.00,
            model="claude-3-opus",
            workspace_id="ws-enterprise",
        )
        assert entry.tokens_input == 10_000_000
        assert entry.tokens_output == 5_000_000

    def test_different_providers(self):
        providers = ["anthropic", "openai", "mistral", "openrouter", "grok"]
        for provider in providers:
            entry = CostEntry(
                timestamp=_now(),
                provider=provider,
                feature="test",
                tokens_input=100,
                tokens_output=50,
                cost=0.001,
                model="test-model",
                workspace_id="ws-1",
            )
            assert entry.provider == provider


# ============================================================================
# TestBudgetAlert
# ============================================================================


class TestBudgetAlert:
    """Tests for BudgetAlert dataclass."""

    def test_create_warning_alert(self):
        now = _now()
        alert = BudgetAlert(
            id="alert-1",
            type="budget_warning",
            message="Budget usage at 80%",
            severity="warning",
            timestamp=now,
        )
        assert alert.id == "alert-1"
        assert alert.type == "budget_warning"
        assert alert.severity == "warning"
        assert alert.acknowledged is False

    def test_create_critical_alert(self):
        alert = BudgetAlert(
            id="alert-2",
            type="limit_reached",
            message="Monthly budget limit exceeded",
            severity="critical",
            timestamp=_now(),
        )
        assert alert.severity == "critical"
        assert alert.type == "limit_reached"

    def test_create_info_alert(self):
        alert = BudgetAlert(
            id="alert-3",
            type="spike_detected",
            message="Cost spike detected",
            severity="info",
            timestamp=_now(),
        )
        assert alert.severity == "info"

    def test_acknowledged_default_false(self):
        alert = BudgetAlert(
            id="a",
            type="budget_warning",
            message="test",
            severity="warning",
            timestamp=_now(),
        )
        assert alert.acknowledged is False

    def test_acknowledged_can_be_set_true(self):
        alert = BudgetAlert(
            id="a",
            type="budget_warning",
            message="test",
            severity="warning",
            timestamp=_now(),
            acknowledged=True,
        )
        assert alert.acknowledged is True

    def test_all_alert_types(self):
        alert_types = ["budget_warning", "spike_detected", "limit_reached"]
        for atype in alert_types:
            alert = BudgetAlert(
                id=f"alert-{atype}",
                type=atype,
                message=f"Alert: {atype}",
                severity="warning",
                timestamp=_now(),
            )
            assert alert.type == atype


# ============================================================================
# TestCostSummary
# ============================================================================


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_create_with_all_fields(self):
        now = _now()
        summary = CostSummary(
            total_cost=100.0,
            budget=500.0,
            tokens_used=2_500_000,
            api_calls=10_000,
            last_updated=now,
            cost_by_provider=[{"name": "Anthropic", "cost": 60.0, "percentage": 60.0}],
            cost_by_feature=[{"name": "Debates", "cost": 40.0, "percentage": 40.0}],
            daily_costs=[{"date": "2026-02-20", "cost": 20.0, "tokens": 500000}],
            alerts=[{"id": "1", "type": "warning", "message": "test"}],
        )
        assert summary.total_cost == 100.0
        assert summary.budget == 500.0
        assert summary.tokens_used == 2_500_000
        assert summary.api_calls == 10_000
        assert len(summary.cost_by_provider) == 1
        assert len(summary.cost_by_feature) == 1
        assert len(summary.daily_costs) == 1
        assert len(summary.alerts) == 1

    def test_default_factory_lists_are_empty(self):
        summary = CostSummary(
            total_cost=0.0,
            budget=0.0,
            tokens_used=0,
            api_calls=0,
            last_updated=_now(),
        )
        assert summary.cost_by_provider == []
        assert summary.cost_by_feature == []
        assert summary.daily_costs == []
        assert summary.alerts == []

    def test_default_factory_lists_are_independent(self):
        s1 = CostSummary(
            total_cost=0.0, budget=0.0, tokens_used=0, api_calls=0, last_updated=_now()
        )
        s2 = CostSummary(
            total_cost=0.0, budget=0.0, tokens_used=0, api_calls=0, last_updated=_now()
        )
        s1.cost_by_provider.append({"name": "test"})
        assert len(s2.cost_by_provider) == 0

    def test_multiple_providers(self):
        providers = [
            {"name": "Anthropic", "cost": 60.0, "percentage": 60.0},
            {"name": "OpenAI", "cost": 25.0, "percentage": 25.0},
            {"name": "Mistral", "cost": 10.0, "percentage": 10.0},
            {"name": "OpenRouter", "cost": 5.0, "percentage": 5.0},
        ]
        summary = CostSummary(
            total_cost=100.0,
            budget=500.0,
            tokens_used=2_500_000,
            api_calls=10_000,
            last_updated=_now(),
            cost_by_provider=providers,
        )
        assert len(summary.cost_by_provider) == 4

    def test_zero_cost_summary(self):
        summary = CostSummary(
            total_cost=0.0,
            budget=0.0,
            tokens_used=0,
            api_calls=0,
            last_updated=_now(),
        )
        assert summary.total_cost == 0.0
        assert summary.tokens_used == 0


# ============================================================================
# TestIsDemoMode
# ============================================================================


class TestIsDemoMode:
    """Tests for _is_demo_mode()."""

    def test_demo_mode_enabled(self):
        mock_settings = MagicMock()
        mock_settings.features.demo_mode = True
        with patch(
            "aragora.server.handlers.costs.models.get_settings",
            return_value=mock_settings,
            create=True,
        ):
            # Need to patch the import inside the function
            with patch.dict("sys.modules", {}):
                pass
            # The function does a local import, so we patch the module it imports from
            with patch(
                "aragora.config.settings.get_settings",
                return_value=mock_settings,
            ):
                assert _is_demo_mode() is True

    def test_demo_mode_disabled(self):
        mock_settings = MagicMock()
        mock_settings.features.demo_mode = False
        with patch(
            "aragora.config.settings.get_settings",
            return_value=mock_settings,
        ):
            assert _is_demo_mode() is False

    def test_import_error_returns_false(self):
        with patch(
            "aragora.config.settings.get_settings",
            side_effect=ImportError("no module"),
        ):
            assert _is_demo_mode() is False

    def test_attribute_error_returns_false(self):
        mock_settings = MagicMock(spec=[])  # No attributes
        mock_settings.features = MagicMock(spec=[])  # features has no demo_mode
        with patch(
            "aragora.config.settings.get_settings",
            side_effect=AttributeError("no demo_mode"),
        ):
            assert _is_demo_mode() is False


# ============================================================================
# TestGetCostTracker
# ============================================================================


class TestGetCostTracker:
    """Tests for _get_cost_tracker() singleton."""

    def test_returns_none_on_import_error(self):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no billing"),
        ):
            result = _get_cost_tracker()
            assert result is None

    def test_returns_none_on_runtime_error(self):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=RuntimeError("init failed"),
        ):
            result = _get_cost_tracker()
            assert result is None

    def test_returns_none_on_os_error(self):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=OSError("disk error"),
        ):
            result = _get_cost_tracker()
            assert result is None

    def test_returns_none_on_value_error(self):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ValueError("bad config"),
        ):
            result = _get_cost_tracker()
            assert result is None

    def test_returns_none_on_attribute_error(self):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=AttributeError("missing attr"),
        ):
            result = _get_cost_tracker()
            assert result is None

    def test_returns_tracker_on_success(self):
        mock_tracker = MagicMock()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = _get_cost_tracker()
            assert result is mock_tracker

    def test_singleton_caches_tracker(self):
        """After successful init, subsequent calls return cached tracker."""
        mock_tracker = MagicMock()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ) as mock_get:
            result1 = _get_cost_tracker()
            result2 = _get_cost_tracker()
            assert result1 is mock_tracker
            assert result2 is mock_tracker
            # Should only be called once since it caches
            mock_get.assert_called_once()

    def test_singleton_retries_after_none(self):
        """When tracker init returns None-equivalent failure, retries on next call."""
        import aragora.server.handlers.costs.models as models_mod

        # First call fails
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no module"),
        ):
            result = _get_cost_tracker()
            assert result is None

        # _cost_tracker stays None after failure, so next call retries
        mock_tracker = MagicMock()
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = _get_cost_tracker()
            assert result is mock_tracker


# ============================================================================
# TestRecordCost
# ============================================================================


class TestRecordCost:
    """Tests for record_cost() function."""

    def test_no_tracker_does_not_raise(self):
        """When no tracker is available, record_cost just logs and returns."""
        with patch(
            "aragora.server.handlers.costs.models._get_cost_tracker",
            return_value=None,
        ):
            # Should not raise
            record_cost(
                provider="anthropic",
                feature="debate",
                tokens_input=1000,
                tokens_output=500,
                cost=0.005,
                model="claude-3-opus",
            )

    def test_record_with_tracker_in_sync_context(self):
        """When no event loop is running, uses asyncio.run()."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()

        mock_token_usage_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
            patch(
                "asyncio.run",
            ) as mock_run,
        ):
            record_cost(
                provider="openai",
                feature="code_review",
                tokens_input=500,
                tokens_output=200,
                cost=0.002,
                model="gpt-4o",
                workspace_id="ws-test",
                user_id="user-42",
            )
            mock_token_usage_cls.assert_called_once()
            call_kwargs = mock_token_usage_cls.call_args
            assert call_kwargs.kwargs["provider"] == "openai"
            assert call_kwargs.kwargs["model"] == "gpt-4o"
            assert call_kwargs.kwargs["tokens_in"] == 500
            assert call_kwargs.kwargs["tokens_out"] == 200
            assert call_kwargs.kwargs["cost_usd"] == Decimal("0.002")
            assert call_kwargs.kwargs["operation"] == "code_review"
            assert call_kwargs.kwargs["workspace_id"] == "ws-test"
            assert call_kwargs.kwargs["metadata"] == {"user_id": "user-42"}
            mock_run.assert_called_once()

    def test_record_with_tracker_in_async_context(self):
        """When an event loop is running, uses create_task."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()

        mock_token_usage_cls = MagicMock()
        mock_loop = MagicMock()
        mock_task = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "asyncio.create_task",
                return_value=mock_task,
            ) as mock_create_task,
        ):
            record_cost(
                provider="anthropic",
                feature="debate",
                tokens_input=1000,
                tokens_output=500,
                cost=0.005,
                model="claude-3-opus",
            )
            mock_create_task.assert_called_once()
            mock_task.add_done_callback.assert_called_once()

    def test_record_default_workspace_id(self):
        """Default workspace_id is 'default'."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()

        mock_token_usage_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                side_effect=RuntimeError("no loop"),
            ),
            patch(
                "asyncio.run",
            ),
        ):
            record_cost(
                provider="mistral",
                feature="summary",
                tokens_input=200,
                tokens_output=100,
                cost=0.001,
                model="mistral-large",
            )
            call_kwargs = mock_token_usage_cls.call_args.kwargs
            assert call_kwargs["workspace_id"] == "default"

    def test_record_no_user_id_metadata_empty(self):
        """When user_id is None, metadata is empty dict."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()

        mock_token_usage_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                side_effect=RuntimeError("no loop"),
            ),
            patch(
                "asyncio.run",
            ),
        ):
            record_cost(
                provider="openai",
                feature="test",
                tokens_input=100,
                tokens_output=50,
                cost=0.001,
                model="gpt-4",
            )
            call_kwargs = mock_token_usage_cls.call_args.kwargs
            assert call_kwargs["metadata"] == {}

    def test_record_handles_import_error(self):
        """record_cost gracefully handles ImportError from TokenUsage import."""
        mock_tracker = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                side_effect=ImportError("no TokenUsage"),
            ),
        ):
            # Should not raise
            record_cost(
                provider="anthropic",
                feature="debate",
                tokens_input=1000,
                tokens_output=500,
                cost=0.005,
                model="claude-3-opus",
            )

    def test_record_handles_runtime_error_in_tracker(self):
        """record_cost gracefully handles RuntimeError."""
        mock_tracker = MagicMock()
        mock_token_usage_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                side_effect=RuntimeError("no loop"),
            ),
            patch(
                "asyncio.run",
                side_effect=RuntimeError("async failed"),
            ),
        ):
            # Should not raise
            record_cost(
                provider="openai",
                feature="test",
                tokens_input=100,
                tokens_output=50,
                cost=0.001,
                model="gpt-4",
            )

    def test_record_handles_value_error(self):
        """record_cost gracefully handles ValueError from Decimal conversion."""
        mock_tracker = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                side_effect=ValueError("invalid decimal"),
            ),
        ):
            record_cost(
                provider="test",
                feature="test",
                tokens_input=0,
                tokens_output=0,
                cost=float("nan"),
                model="test",
            )

    def test_record_handles_type_error(self):
        """record_cost gracefully handles TypeError."""
        mock_tracker = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                side_effect=TypeError("bad type"),
            ),
        ):
            record_cost(
                provider="test",
                feature="test",
                tokens_input=0,
                tokens_output=0,
                cost=0.0,
                model="test",
            )

    def test_record_handles_os_error(self):
        """record_cost gracefully handles OSError."""
        mock_tracker = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                side_effect=OSError("disk failure"),
            ),
        ):
            record_cost(
                provider="test",
                feature="test",
                tokens_input=0,
                tokens_output=0,
                cost=0.0,
                model="test",
            )

    def test_record_cost_decimal_precision(self):
        """Verify cost is converted to Decimal with string representation."""
        mock_tracker = MagicMock()
        mock_tracker.record = AsyncMock()
        mock_token_usage_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.billing.cost_tracker.TokenUsage",
                mock_token_usage_cls,
            ),
            patch(
                "asyncio.get_running_loop",
                side_effect=RuntimeError("no loop"),
            ),
            patch(
                "asyncio.run",
            ),
        ):
            record_cost(
                provider="anthropic",
                feature="debate",
                tokens_input=1000,
                tokens_output=500,
                cost=0.123456,
                model="claude-3-opus",
            )
            call_kwargs = mock_token_usage_cls.call_args.kwargs
            assert call_kwargs["cost_usd"] == Decimal("0.123456")


# ============================================================================
# TestGetActiveAlerts
# ============================================================================


class TestGetActiveAlerts:
    """Tests for _get_active_alerts()."""

    def test_no_budget_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.return_value = None
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_budget_no_alert_level_returns_empty(self):
        budget = _make_budget()
        budget.check_alert_level.return_value = None
        tracker = MagicMock()
        tracker.get_budget.return_value = budget
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_budget_warning_alert(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("400.00"),
        )
        # Create a mock alert level
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        # We need to mock BudgetAlertLevel for the severity map
        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = mock_alert_level
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            assert len(alerts) == 1
            assert alerts[0]["id"] == "budget_ws-1"
            assert alerts[0]["type"] == "budget_warning"
            assert alerts[0]["severity"] == "warning"
            assert "80.0%" in alerts[0]["message"]
            assert "$400.00" in alerts[0]["message"]
            assert "$500.00" in alerts[0]["message"]

    def test_budget_critical_alert(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("475.00"),
        )
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = "warning_level"
        mock_levels.CRITICAL = mock_alert_level
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-enterprise")
            assert len(alerts) == 1
            assert alerts[0]["severity"] == "critical"
            assert "95.0%" in alerts[0]["message"]

    def test_budget_exceeded_alert(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("550.00"),
        )
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = "warning_level"
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = mock_alert_level

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            assert len(alerts) == 1
            assert alerts[0]["severity"] == "critical"
            assert "110.0%" in alerts[0]["message"]

    def test_budget_info_alert(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("250.00"),
        )
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        mock_levels = MagicMock()
        mock_levels.INFO = mock_alert_level
        mock_levels.WARNING = "warning_level"
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            assert len(alerts) == 1
            assert alerts[0]["severity"] == "info"

    def test_zero_monthly_limit_percentage_is_zero(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("0"),
            current_monthly_spend=Decimal("10.00"),
        )
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = mock_alert_level
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            assert len(alerts) == 1
            # With zero limit, percentage should be 0
            assert "0.0%" in alerts[0]["message"]

    def test_check_alert_level_attribute_error_returns_empty(self):
        """When budget.check_alert_level raises AttributeError, returns empty."""
        tracker = MagicMock()
        budget = MagicMock()
        budget.check_alert_level.side_effect = AttributeError("no check_alert_level")
        tracker.get_budget.return_value = budget
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_runtime_error_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.side_effect = RuntimeError("tracker broken")
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_value_error_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.side_effect = ValueError("bad workspace")
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_type_error_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.side_effect = TypeError("wrong type")
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_attribute_error_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.side_effect = AttributeError("missing attr")
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_key_error_returns_empty(self):
        tracker = MagicMock()
        tracker.get_budget.side_effect = KeyError("missing key")
        alerts = _get_active_alerts(tracker, "ws-1")
        assert alerts == []

    def test_unmatched_alert_level_defaults_to_warning(self):
        """When alert level isn't in severity_map, defaults to 'warning'."""
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("300.00"),
        )
        unknown_alert_level = MagicMock()
        budget.check_alert_level.return_value = unknown_alert_level

        # None of the levels match
        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = "warning_level"
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            assert len(alerts) == 1
            assert alerts[0]["severity"] == "warning"

    def test_alert_timestamp_is_isoformat(self):
        budget = _make_budget(
            monthly_limit_usd=Decimal("500.00"),
            current_monthly_spend=Decimal("400.00"),
        )
        mock_alert_level = MagicMock()
        budget.check_alert_level.return_value = mock_alert_level

        mock_levels = MagicMock()
        mock_levels.INFO = "info_level"
        mock_levels.WARNING = mock_alert_level
        mock_levels.CRITICAL = "critical_level"
        mock_levels.EXCEEDED = "exceeded_level"

        tracker = MagicMock()
        tracker.get_budget.return_value = budget

        with patch(
            "aragora.billing.cost_tracker.BudgetAlertLevel",
            mock_levels,
        ):
            alerts = _get_active_alerts(tracker, "ws-1")
            # Verify timestamp is a valid ISO format string
            ts = alerts[0]["timestamp"]
            parsed = datetime.fromisoformat(ts)
            assert parsed.tzinfo is not None  # Should be UTC


# ============================================================================
# TestEmptyCostSummary
# ============================================================================


class TestEmptyCostSummary:
    """Tests for _empty_cost_summary()."""

    def test_returns_cost_summary_instance(self):
        result = _empty_cost_summary()
        assert isinstance(result, CostSummary)

    def test_all_costs_zero(self):
        result = _empty_cost_summary()
        assert result.total_cost == 0.0
        assert result.budget == 0.0
        assert result.tokens_used == 0
        assert result.api_calls == 0

    def test_all_lists_empty(self):
        result = _empty_cost_summary()
        assert result.cost_by_provider == []
        assert result.cost_by_feature == []
        assert result.daily_costs == []
        assert result.alerts == []

    def test_last_updated_is_recent(self):
        before = _now()
        result = _empty_cost_summary()
        after = _now()
        assert before <= result.last_updated <= after

    def test_last_updated_is_utc(self):
        result = _empty_cost_summary()
        assert result.last_updated.tzinfo == timezone.utc


# ============================================================================
# TestGetCostSummary
# ============================================================================


class TestGetCostSummary:
    """Tests for get_cost_summary() async function."""

    @pytest.mark.asyncio
    async def test_no_tracker_demo_mode_returns_mock(self):
        """When no tracker and demo mode, returns mock summary."""
        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=True,
            ),
        ):
            result = await get_cost_summary()
            assert isinstance(result, CostSummary)
            assert result.total_cost > 0
            assert len(result.cost_by_provider) > 0

    @pytest.mark.asyncio
    async def test_no_tracker_no_demo_returns_empty(self):
        """When no tracker and no demo mode, returns empty summary."""
        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert isinstance(result, CostSummary)
            assert result.total_cost == 0.0
            assert result.cost_by_provider == []

    @pytest.mark.asyncio
    async def test_tracker_with_real_data(self):
        """When tracker has real data, returns real summary."""
        report = _make_report()
        budget_obj = _make_budget()

        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=budget_obj)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert isinstance(result, CostSummary)
            assert result.total_cost == 125.50
            assert result.budget == 500.0
            assert result.tokens_used == 2_000_000 + 1_125_000
            assert result.api_calls == 12_550
            assert len(result.cost_by_provider) == 2
            assert len(result.cost_by_feature) == 2

    @pytest.mark.asyncio
    async def test_tracker_cost_by_provider_sorted_descending(self):
        """Providers are sorted by cost descending."""
        report = _make_report(
            cost_by_provider={
                "Mistral": Decimal("10.00"),
                "Anthropic": Decimal("80.00"),
                "OpenAI": Decimal("35.00"),
            },
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            providers = [p["name"] for p in result.cost_by_provider]
            assert providers == ["Anthropic", "OpenAI", "Mistral"]

    @pytest.mark.asyncio
    async def test_tracker_cost_by_feature_sorted_descending(self):
        """Features/operations are sorted by cost descending."""
        report = _make_report(
            cost_by_operation={
                "Code Review": Decimal("20.00"),
                "Debates": Decimal("60.00"),
                "Email": Decimal("5.00"),
            },
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            features = [f["name"] for f in result.cost_by_feature]
            assert features == ["Debates", "Code Review", "Email"]

    @pytest.mark.asyncio
    async def test_tracker_percentage_calculation(self):
        """Percentages are calculated correctly."""
        report = _make_report(
            total_cost_usd=Decimal("100.00"),
            cost_by_provider={"A": Decimal("75.00"), "B": Decimal("25.00")},
            cost_by_operation={"X": Decimal("60.00"), "Y": Decimal("40.00")},
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.cost_by_provider[0]["percentage"] == 75.0
            assert result.cost_by_provider[1]["percentage"] == 25.0
            assert result.cost_by_feature[0]["percentage"] == 60.0
            assert result.cost_by_feature[1]["percentage"] == 40.0

    @pytest.mark.asyncio
    async def test_tracker_zero_total_cost_percentage_is_zero(self):
        """When total cost is 0, percentages should be 0."""
        report = _make_report(
            total_cost_usd=Decimal("0"),
            cost_by_provider={"A": Decimal("0")},
            cost_by_operation={"X": Decimal("0")},
            cost_over_time=[{"date": "2026-02-20", "cost": 0.0}],
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            # Zero cost with cost_over_time present - should not go to empty/mock path
            result = await get_cost_summary()
            assert result.cost_by_provider[0]["percentage"] == 0

    @pytest.mark.asyncio
    async def test_tracker_no_budget_defaults_to_500(self):
        """When no budget is set, defaults to 500.0."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.budget == 500.0

    @pytest.mark.asyncio
    async def test_tracker_custom_budget(self):
        """When budget is set, uses it."""
        report = _make_report()
        budget_obj = _make_budget(monthly_limit_usd=Decimal("1000.00"))
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=budget_obj)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.budget == 1000.0

    @pytest.mark.asyncio
    async def test_tracker_budget_with_zero_limit_defaults_500(self):
        """When budget limit is 0, defaults to 500.0."""
        report = _make_report()
        budget_obj = _make_budget(monthly_limit_usd=Decimal("0"))
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=budget_obj)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.budget == 500.0

    @pytest.mark.asyncio
    async def test_time_range_24h(self):
        """24h time range maps to 1 day."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(time_range="24h")
            call_args = tracker.generate_report.call_args
            period_start = call_args.kwargs["period_start"]
            period_end = call_args.kwargs["period_end"]
            diff = period_end - period_start
            assert 0.9 < diff.total_seconds() / 86400 < 1.1

    @pytest.mark.asyncio
    async def test_time_range_7d(self):
        """7d time range maps to 7 days."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(time_range="7d")
            call_args = tracker.generate_report.call_args
            period_start = call_args.kwargs["period_start"]
            period_end = call_args.kwargs["period_end"]
            diff = period_end - period_start
            assert 6.9 < diff.total_seconds() / 86400 < 7.1

    @pytest.mark.asyncio
    async def test_time_range_30d(self):
        """30d time range maps to 30 days."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(time_range="30d")
            call_args = tracker.generate_report.call_args
            period_start = call_args.kwargs["period_start"]
            period_end = call_args.kwargs["period_end"]
            diff = period_end - period_start
            assert 29.9 < diff.total_seconds() / 86400 < 30.1

    @pytest.mark.asyncio
    async def test_time_range_90d(self):
        """90d time range maps to 90 days."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(time_range="90d")
            call_args = tracker.generate_report.call_args
            period_start = call_args.kwargs["period_start"]
            period_end = call_args.kwargs["period_end"]
            diff = period_end - period_start
            assert 89.9 < diff.total_seconds() / 86400 < 90.1

    @pytest.mark.asyncio
    async def test_unknown_time_range_defaults_to_7d(self):
        """Unknown time range defaults to 7 days."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(time_range="unknown")
            call_args = tracker.generate_report.call_args
            period_start = call_args.kwargs["period_start"]
            period_end = call_args.kwargs["period_end"]
            diff = period_end - period_start
            assert 6.9 < diff.total_seconds() / 86400 < 7.1

    @pytest.mark.asyncio
    async def test_tracker_zero_cost_no_data_demo_returns_mock(self):
        """Zero cost with no cost_over_time in demo mode returns mock."""
        report = _make_report(
            total_cost_usd=Decimal("0"),
            cost_by_provider={},
            cost_by_operation={},
            cost_over_time=[],
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=True,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost > 0  # Mock data has positive costs

    @pytest.mark.asyncio
    async def test_tracker_zero_cost_no_data_no_demo_returns_empty(self):
        """Zero cost with no cost_over_time without demo mode returns empty."""
        report = _make_report(
            total_cost_usd=Decimal("0"),
            cost_by_provider={},
            cost_by_operation={},
            cost_over_time=[],
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0
            assert result.cost_by_provider == []

    @pytest.mark.asyncio
    async def test_tracker_exception_demo_returns_mock(self):
        """When tracker raises and demo mode, returns mock."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=RuntimeError("db down"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=True,
            ),
        ):
            result = await get_cost_summary()
            assert isinstance(result, CostSummary)
            assert result.total_cost > 0

    @pytest.mark.asyncio
    async def test_tracker_exception_no_demo_returns_empty(self):
        """When tracker raises without demo mode, returns empty."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=ValueError("bad input"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_tracker_import_error_demo_returns_mock(self):
        """ImportError from CostGranularity in demo returns mock."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=ImportError("no granularity"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=True,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost > 0

    @pytest.mark.asyncio
    async def test_tracker_os_error_returns_empty(self):
        """OSError from tracker returns empty summary."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=OSError("disk failure"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_tracker_type_error_returns_empty(self):
        """TypeError from tracker returns empty summary."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=TypeError("bad arg"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_tracker_key_error_returns_empty(self):
        """KeyError from tracker returns empty summary."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=KeyError("missing"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_tracker_attribute_error_returns_empty(self):
        """AttributeError from tracker returns empty summary."""
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(side_effect=AttributeError("no attr"))

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
        ):
            result = await get_cost_summary()
            assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_default_workspace_and_time_range(self):
        """Default workspace is 'default' and time_range is '7d'."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary()
            call_args = tracker.generate_report.call_args
            assert call_args.kwargs["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_custom_workspace_id(self):
        """Custom workspace_id is passed to tracker."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            await get_cost_summary(workspace_id="ws-enterprise")
            call_args = tracker.generate_report.call_args
            assert call_args.kwargs["workspace_id"] == "ws-enterprise"

    @pytest.mark.asyncio
    async def test_empty_provider_and_operation_maps(self):
        """Empty provider/operation maps produce empty lists."""
        report = _make_report(
            cost_by_provider={},
            cost_by_operation={},
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.cost_by_provider == []
            assert result.cost_by_feature == []

    @pytest.mark.asyncio
    async def test_none_provider_and_operation_maps(self):
        """None provider/operation maps produce empty lists."""
        report = _make_report(
            cost_by_provider=None,
            cost_by_operation=None,
        )
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.cost_by_provider == []
            assert result.cost_by_feature == []

    @pytest.mark.asyncio
    async def test_cost_over_time_passed_through(self):
        """cost_over_time from report is passed through as daily_costs."""
        timeline = [
            {"date": "2026-02-20", "cost": 18.50},
            {"date": "2026-02-21", "cost": 21.00},
            {"date": "2026-02-22", "cost": 16.00},
        ]
        report = _make_report(cost_over_time=timeline)
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.daily_costs == timeline

    @pytest.mark.asyncio
    async def test_none_cost_over_time_becomes_empty_list(self):
        """None cost_over_time becomes empty daily_costs list."""
        report = _make_report(cost_over_time=None)
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.daily_costs == []

    @pytest.mark.asyncio
    async def test_alerts_from_active_alerts(self):
        """Alerts are sourced from _get_active_alerts."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        mock_alerts = [{"id": "alert-1", "type": "budget_warning", "severity": "warning"}]

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=mock_alerts,
            ),
        ):
            result = await get_cost_summary()
            assert result.alerts == mock_alerts

    @pytest.mark.asyncio
    async def test_generates_report_with_daily_granularity(self):
        """generate_report is called with DAILY granularity."""
        report = _make_report()
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=None)

        mock_granularity = MagicMock()
        mock_granularity.DAILY = "daily"

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
            patch(
                "aragora.billing.cost_tracker.CostGranularity",
                mock_granularity,
            ),
        ):
            await get_cost_summary()
            call_args = tracker.generate_report.call_args
            assert call_args.kwargs["granularity"] == "daily"

    @pytest.mark.asyncio
    async def test_budget_object_with_none_monthly_limit(self):
        """Budget object where monthly_limit_usd is None defaults to 500."""
        report = _make_report()
        budget_obj = MagicMock()
        budget_obj.monthly_limit_usd = None
        tracker = _make_tracker()
        tracker.generate_report = AsyncMock(return_value=report)
        tracker.get_budget = MagicMock(return_value=budget_obj)

        with (
            patch(
                "aragora.server.handlers.costs.models._get_cost_tracker",
                return_value=tracker,
            ),
            patch(
                "aragora.server.handlers.costs.models._is_demo_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.costs.models._get_active_alerts",
                return_value=[],
            ),
        ):
            result = await get_cost_summary()
            assert result.budget == 500.0
