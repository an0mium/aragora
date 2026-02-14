"""Tests for budget notification wiring at server startup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.initialization import init_budget_notifications


class TestInitBudgetNotifications:
    """Test init_budget_notifications wires callback to BudgetManager."""

    @patch("aragora.server.initialization.logger")
    def test_wires_callback_successfully(self, mock_logger):
        """setup_budget_notifications is called with the BudgetManager singleton."""
        mock_manager = MagicMock()
        mock_notifier = MagicMock()

        with (
            patch(
                "aragora.billing.budget_manager.get_budget_manager",
                return_value=mock_manager,
            ),
            patch(
                "aragora.billing.budget_alert_notifier.setup_budget_notifications",
                return_value=mock_notifier,
            ) as mock_setup,
        ):
            result = init_budget_notifications()

        assert result is True
        mock_setup.assert_called_once_with(mock_manager)

    @patch("aragora.server.initialization.logger")
    def test_returns_false_on_import_error(self, mock_logger):
        """Returns False when billing modules are not importable."""
        with patch(
            "builtins.__import__",
            side_effect=_make_import_blocker("aragora.billing.budget_manager"),
        ):
            result = init_budget_notifications()

        assert result is False

    @patch("aragora.server.initialization.logger")
    def test_returns_false_on_runtime_error(self, mock_logger):
        """Returns False when setup raises a RuntimeError."""
        with (
            patch(
                "aragora.billing.budget_manager.get_budget_manager",
                side_effect=RuntimeError("db locked"),
            ),
        ):
            result = init_budget_notifications()

        assert result is False

    @patch("aragora.server.initialization.logger")
    def test_returns_false_on_attribute_error(self, mock_logger):
        """Returns False when manager lacks register_alert_callback."""
        mock_manager = MagicMock()
        mock_manager.register_alert_callback.side_effect = AttributeError("no method")

        with (
            patch(
                "aragora.billing.budget_manager.get_budget_manager",
                return_value=mock_manager,
            ),
        ):
            result = init_budget_notifications()

        assert result is False


class TestBudgetNotificationCallbackFiring:
    """Test that alert callbacks fire correctly after wiring."""

    def test_callback_registered_on_manager(self):
        """Verifies the notifier callback is registered on the manager."""
        mock_manager = MagicMock()

        with (
            patch(
                "aragora.billing.budget_manager.get_budget_manager",
                return_value=mock_manager,
            ),
            patch(
                "aragora.billing.budget_alert_notifier._notifier",
                None,
            ),
        ):
            init_budget_notifications()

        mock_manager.register_alert_callback.assert_called_once()
        # The callback should be the notifier's on_alert method
        callback = mock_manager.register_alert_callback.call_args[0][0]
        assert callable(callback)


def _make_import_blocker(blocked_module: str):
    """Create an __import__ side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {blocked_module} not available")
        return real_import(name, *args, **kwargs)

    return _blocker
