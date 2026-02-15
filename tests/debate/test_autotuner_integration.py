"""Tests for Autotuner integration with BudgetCoordinator.

Verifies that the Autotuner is created when ArenaConfig specifies autotune_config,
that record_round is called during mid-debate budget checks, and that
should_continue affects the continuation decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.budget_coordinator import BudgetCoordinator
from aragora.runtime.autotune import AutotuneConfig, AutotuneDecision, Autotuner, StopReason


class TestAutotunerInBudgetCoordinator:
    """Test Autotuner integration in BudgetCoordinator."""

    def test_autotuner_record_round_called_during_check(self):
        """record_round should be called with round data during mid-debate check."""
        mock_autotuner = MagicMock()
        mock_autotuner.should_continue.return_value = AutotuneDecision(should_continue=True)

        coordinator = BudgetCoordinator(autotuner=mock_autotuner)

        allowed, reason = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=2,
            round_tokens=500,
            round_messages=4,
            support_scores=[0.7, 0.8],
        )

        assert allowed is True
        mock_autotuner.record_round.assert_called_once_with(
            round_num=2,
            tokens=500,
            messages=4,
            support_scores=[0.7, 0.8],
        )

    def test_should_continue_false_stops_debate(self):
        """When autotuner says stop, check_budget_mid_debate should return False."""
        mock_autotuner = MagicMock()
        mock_autotuner.should_continue.return_value = AutotuneDecision(
            should_continue=False,
            stop_reason=StopReason.MAX_COST,
        )

        coordinator = BudgetCoordinator(autotuner=mock_autotuner)

        allowed, reason = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=3,
        )

        assert allowed is False
        assert "max_cost" in reason.lower()

    def test_should_continue_true_allows_debate(self):
        """When autotuner says continue, check should allow."""
        mock_autotuner = MagicMock()
        mock_autotuner.should_continue.return_value = AutotuneDecision(should_continue=True)

        coordinator = BudgetCoordinator(autotuner=mock_autotuner)

        allowed, reason = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=1,
        )

        assert allowed is True
        assert reason == ""

    def test_none_autotuner_no_change(self):
        """When autotuner is None, behavior is unchanged."""
        coordinator = BudgetCoordinator(autotuner=None)

        allowed, reason = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=1,
        )

        assert allowed is True
        assert reason == ""

    def test_autotuner_error_continues_debate(self):
        """When autotuner raises, debate should continue (fail open)."""
        mock_autotuner = MagicMock()
        mock_autotuner.record_round.side_effect = TypeError("bad args")

        coordinator = BudgetCoordinator(autotuner=mock_autotuner)

        allowed, reason = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=2,
        )

        assert allowed is True

    def test_real_autotuner_max_rounds_stop(self):
        """Integration test with real Autotuner - stops at max_rounds."""
        config = AutotuneConfig(max_rounds=2)
        autotuner = Autotuner(config=config)
        autotuner.start()

        coordinator = BudgetCoordinator(autotuner=autotuner)

        # First round - should continue
        allowed, _ = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=0,
            round_tokens=100,
            round_messages=2,
            support_scores=[0.5],
        )
        assert allowed is True

        # Second round - should continue (1 completed so far)
        allowed, _ = coordinator.check_budget_mid_debate(
            debate_id="test-1",
            round_num=1,
            round_tokens=100,
            round_messages=2,
            support_scores=[0.6],
        )
        # Now 2 rounds completed, at max_rounds
        assert allowed is False


class TestAutotunerCreation:
    """Test Autotuner creation from ArenaConfig."""

    def test_create_from_autotune_config_object(self):
        """AutotuneConfig object should create Autotuner."""
        from aragora.debate.orchestrator_init import _create_autotuner

        config = AutotuneConfig(max_rounds=5, max_cost_dollars=2.0)
        autotuner = _create_autotuner(config)

        assert autotuner is not None
        assert autotuner.config.max_rounds == 5
        assert autotuner.config.max_cost_dollars == 2.0
        assert autotuner._start_time is not None  # start() was called

    def test_create_from_dict(self):
        """Dict should be unpacked into AutotuneConfig."""
        from aragora.debate.orchestrator_init import _create_autotuner

        autotuner = _create_autotuner({"max_rounds": 3})

        assert autotuner is not None
        assert autotuner.config.max_rounds == 3

    def test_create_from_truthy(self):
        """Truthy value should create Autotuner with defaults."""
        from aragora.debate.orchestrator_init import _create_autotuner

        autotuner = _create_autotuner(True)

        assert autotuner is not None
        assert autotuner.config.max_rounds == 5  # default

    def test_none_returns_none(self):
        """None should return None (no autotuner)."""
        from aragora.debate.orchestrator_init import _create_autotuner

        assert _create_autotuner(None) is None
