"""Tests for aragora.nomic.stopping_rules — StoppingRuleEngine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aragora.nomic.stopping_rules import StoppingConfig, StoppingRuleEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return StoppingRuleEngine()


@pytest.fixture
def default_config():
    return StoppingConfig()


def _mock_telemetry(
    total_cost: float = 0.0,
    cycle_count: int = 0,
    recent_cycles: list | None = None,
):
    """Create a mock telemetry object."""
    mock = MagicMock()
    mock.get_total_cost.return_value = total_cost
    mock.get_cycle_count.return_value = cycle_count
    mock.get_recent_cycles.return_value = recent_cycles or []
    return mock


def _mock_record(quality_delta: float = 0.1):
    r = MagicMock()
    r.quality_delta = quality_delta
    return r


# ---------------------------------------------------------------------------
# BudgetExhausted
# ---------------------------------------------------------------------------


class TestBudgetExhausted:
    def test_stops_when_over_budget(self, engine):
        telemetry = _mock_telemetry(total_cost=15.0)
        config = StoppingConfig(budget_limit=10.0)
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True
        assert "BudgetExhausted" in reason

    def test_does_not_stop_under_budget(self, engine):
        telemetry = _mock_telemetry(total_cost=5.0)
        config = StoppingConfig(budget_limit=10.0)
        stop, _ = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False

    def test_exactly_at_budget(self, engine):
        telemetry = _mock_telemetry(total_cost=10.0)
        config = StoppingConfig(budget_limit=10.0)
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True

    def test_budget_override(self, engine):
        # budget=0 means remaining budget is 0 -> total = limit
        config = StoppingConfig(budget_limit=10.0)
        stop, reason = engine.should_stop(budget=0.0, config=config)
        assert stop is True

    def test_no_telemetry_no_budget(self, engine):
        config = StoppingConfig(budget_limit=10.0)
        stop, _ = engine.should_stop(telemetry=None, config=config)
        assert stop is False

    def test_zero_budget_limit_never_triggers(self, engine):
        telemetry = _mock_telemetry(total_cost=100.0)
        config = StoppingConfig(budget_limit=0.0)
        stop, _ = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False


# ---------------------------------------------------------------------------
# DiminishingReturns
# ---------------------------------------------------------------------------


class TestDiminishingReturns:
    def test_stops_on_consecutive_low_delta(self, engine):
        records = [_mock_record(quality_delta=0.0001) for _ in range(3)]
        telemetry = _mock_telemetry(recent_cycles=records)
        config = StoppingConfig(
            min_quality_delta=0.001,
            consecutive_low_delta=3,
            budget_limit=0,
        )
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True
        assert "DiminishingReturns" in reason

    def test_does_not_stop_if_one_high_delta(self, engine):
        records = [
            _mock_record(quality_delta=0.0001),
            _mock_record(quality_delta=0.5),  # big improvement
            _mock_record(quality_delta=0.0001),
        ]
        telemetry = _mock_telemetry(recent_cycles=records)
        config = StoppingConfig(
            min_quality_delta=0.001,
            consecutive_low_delta=3,
            budget_limit=0,
        )
        stop, _ = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False

    def test_not_enough_records(self, engine):
        records = [_mock_record(quality_delta=0.0)]
        telemetry = _mock_telemetry(recent_cycles=records)
        config = StoppingConfig(
            consecutive_low_delta=3,
            budget_limit=0,
        )
        stop, _ = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False

    def test_negative_delta_counts_as_low(self, engine):
        records = [_mock_record(quality_delta=-0.0001) for _ in range(3)]
        telemetry = _mock_telemetry(recent_cycles=records)
        config = StoppingConfig(
            min_quality_delta=0.001,
            consecutive_low_delta=3,
            budget_limit=0,
        )
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True


# ---------------------------------------------------------------------------
# CycleLimit
# ---------------------------------------------------------------------------


class TestCycleLimit:
    def test_stops_at_max(self, engine):
        telemetry = _mock_telemetry(cycle_count=50)
        config = StoppingConfig(max_cycles=50, budget_limit=0)
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True
        assert "CycleLimit" in reason

    def test_does_not_stop_under_max(self, engine):
        telemetry = _mock_telemetry(cycle_count=10)
        config = StoppingConfig(max_cycles=50, budget_limit=0)
        stop, _ = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False


# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def test_stops_when_exceeded(self, engine):
        start = time.time() - 10 * 3600  # 10 hours ago
        config = StoppingConfig(max_duration_hours=8.0, budget_limit=0)
        stop, reason = engine.should_stop(start_time=start, config=config)
        assert stop is True
        assert "TimeLimit" in reason

    def test_does_not_stop_within_limit(self, engine):
        start = time.time() - 1 * 3600  # 1 hour ago
        config = StoppingConfig(max_duration_hours=8.0, budget_limit=0)
        stop, _ = engine.should_stop(start_time=start, config=config)
        assert stop is False

    def test_no_start_time_skips(self, engine):
        config = StoppingConfig(max_duration_hours=0.001, budget_limit=0)
        stop, _ = engine.should_stop(start_time=None, config=config)
        assert stop is False

    def test_zero_duration_never_triggers(self, engine):
        start = time.time() - 100 * 3600
        config = StoppingConfig(max_duration_hours=0.0, budget_limit=0)
        stop, _ = engine.should_stop(start_time=start, config=config)
        assert stop is False


# ---------------------------------------------------------------------------
# NoViableGoals
# ---------------------------------------------------------------------------


class TestNoViableGoals:
    def test_stops_when_no_goals(self, engine):
        mock_proposer = MagicMock()
        mock_proposer.propose_goals.return_value = []
        config = StoppingConfig(min_goal_confidence=0.7, budget_limit=0)
        stop, reason = engine.should_stop(goal_proposer=mock_proposer, config=config)
        assert stop is True
        assert "NoViableGoals" in reason

    def test_does_not_stop_with_goals(self, engine):
        mock_proposer = MagicMock()
        mock_proposer.propose_goals.return_value = [MagicMock()]
        config = StoppingConfig(budget_limit=0)
        stop, _ = engine.should_stop(goal_proposer=mock_proposer, config=config)
        assert stop is False

    def test_no_proposer_skips(self, engine):
        config = StoppingConfig(budget_limit=0)
        stop, _ = engine.should_stop(goal_proposer=None, config=config)
        assert stop is False

    def test_proposer_error_skips(self, engine):
        mock_proposer = MagicMock()
        mock_proposer.propose_goals.side_effect = RuntimeError("boom")
        config = StoppingConfig(budget_limit=0)
        stop, _ = engine.should_stop(goal_proposer=mock_proposer, config=config)
        assert stop is False


# ---------------------------------------------------------------------------
# Combination / priority
# ---------------------------------------------------------------------------


class TestCombinations:
    def test_first_triggered_rule_wins(self, engine):
        """When multiple rules trigger, the first one in evaluation order wins."""
        telemetry = _mock_telemetry(total_cost=20.0, cycle_count=100)
        config = StoppingConfig(budget_limit=10.0, max_cycles=50)
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is True
        # Budget is checked first
        assert "BudgetExhausted" in reason

    def test_no_triggers_returns_false(self, engine):
        telemetry = _mock_telemetry(total_cost=1.0, cycle_count=5)
        config = StoppingConfig(budget_limit=100.0, max_cycles=1000)
        stop, reason = engine.should_stop(telemetry=telemetry, config=config)
        assert stop is False
        assert reason == ""

    def test_default_config_works(self, engine):
        # With no telemetry or proposer, nothing triggers
        stop, reason = engine.should_stop()
        assert stop is False
