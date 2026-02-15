"""Tests for per-debate budget check wired into BudgetCoordinator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.budget_coordinator import BudgetCoordinator


class TestPerDebateBudgetCheck:
    """Test per-debate budget check integration in BudgetCoordinator."""

    def _make_coord(self, org_id="org1", extensions=None):
        return BudgetCoordinator(org_id=org_id, extensions=extensions)

    def test_org_check_runs_first(self):
        """Org-level budget check should run before per-debate check."""
        extensions = MagicMock()
        extensions.check_debate_budget.return_value = {"allowed": True, "message": "ok"}

        coord = self._make_coord(extensions=extensions)

        # Patch the module-level import target so the lazy import inside
        # check_budget_mid_debate picks up the mock.
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=ImportError("no manager"),
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=1)

        assert allowed is True
        extensions.check_debate_budget.assert_called_once_with("d1")

    def test_per_debate_check_runs_after_org_passes(self):
        """Per-debate check runs after org-level check passes."""
        manager = MagicMock()
        manager.check_budget.return_value = (True, "", None)

        extensions = MagicMock()
        extensions.check_debate_budget.return_value = {"allowed": True, "message": "ok"}

        coord = self._make_coord(extensions=extensions)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=manager,
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=2)

        assert allowed is True
        manager.check_budget.assert_called_once()
        extensions.check_debate_budget.assert_called_once_with("d1")

    def test_per_debate_exceeded_returns_false(self):
        """Per-debate budget exceeded returns (False, reason)."""
        extensions = MagicMock()
        extensions.check_debate_budget.return_value = {
            "allowed": False,
            "message": "Debate budget exceeded: $0.50 of $0.50 limit",
        }

        coord = self._make_coord(extensions=extensions)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=ImportError("no manager"),
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=3)

        assert allowed is False
        assert "Debate budget exceeded" in reason

    def test_org_exceeded_short_circuits_per_debate(self):
        """If org budget is exceeded, we return False before per-debate check."""
        manager = MagicMock()
        manager.check_budget.return_value = (False, "Org budget exhausted", None)

        extensions = MagicMock()

        coord = self._make_coord(extensions=extensions)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=manager,
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=1)

        assert allowed is False
        assert "Org budget exhausted" in reason
        # Per-debate check should NOT have been called since org check failed
        extensions.check_debate_budget.assert_not_called()

    def test_no_extensions_skips_per_debate_check(self):
        """When extensions is None, per-debate check is skipped."""
        coord = self._make_coord(extensions=None)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=ImportError("no manager"),
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=1)

        assert allowed is True
        assert reason == ""

    def test_both_checks_pass(self):
        """When both org and per-debate pass, result is (True, '')."""
        manager = MagicMock()
        manager.check_budget.return_value = (True, "", None)

        extensions = MagicMock()
        extensions.check_debate_budget.return_value = {"allowed": True, "message": "ok"}

        coord = self._make_coord(extensions=extensions)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=manager,
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=5)

        assert allowed is True
        assert reason == ""

    def test_extensions_without_check_method_skipped(self):
        """Extensions object without check_debate_budget is safely skipped."""
        extensions = object()  # No check_debate_budget attribute

        coord = self._make_coord(extensions=extensions)

        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=ImportError("no manager"),
        ):
            allowed, reason = coord.check_budget_mid_debate("d1", round_num=1)

        assert allowed is True
