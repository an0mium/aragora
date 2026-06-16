"""Tests for the fail-closed monthly budget guard (anti-runaway-bill)."""

from __future__ import annotations

import json

import pytest

from aragora.billing import budget_guard
from aragora.billing.budget_guard import BudgetExceededError


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the guard at a temp store and return its path."""
    path = tmp_path / "budget_guard.json"
    monkeypatch.setenv("ARAGORA_BUDGET_GUARD_STORE", str(path))
    return path


def test_disabled_by_default_is_noop(store, monkeypatch):
    monkeypatch.delenv("ARAGORA_MONTHLY_BUDGET_USD", raising=False)
    assert budget_guard.is_enabled() is False
    # No cap -> never raises, records nothing, remaining is infinite.
    budget_guard.assert_within_budget(10_000.0)
    budget_guard.record_spend(10_000.0)
    assert budget_guard.current_spend_usd() == 0.0
    assert budget_guard.remaining_usd() == float("inf")
    assert not store.exists()  # nothing persisted when disabled


def test_enabled_allows_under_cap(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "100")
    assert budget_guard.is_enabled() is True
    budget_guard.record_spend(40.0)
    assert budget_guard.current_spend_usd() == pytest.approx(40.0)
    assert budget_guard.remaining_usd() == pytest.approx(60.0)
    # 40 spent + 50 estimate = 90 <= 100 -> allowed
    budget_guard.assert_within_budget(50.0)


def test_fails_closed_over_cap(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "100")
    budget_guard.record_spend(95.0)
    # 95 + 10 = 105 > 100 -> must raise (fail closed)
    with pytest.raises(BudgetExceededError):
        budget_guard.assert_within_budget(10.0)
    # And gating on already-recorded spend alone, once at/over the cap:
    budget_guard.record_spend(10.0)  # now 105
    with pytest.raises(BudgetExceededError):
        budget_guard.assert_within_budget(0.0)


def test_record_spend_accumulates(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "50")
    for _ in range(5):
        budget_guard.record_spend(3.0)
    assert budget_guard.current_spend_usd() == pytest.approx(15.0)


def test_month_rollover_resets(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "100")
    # Seed a stale prior-month total directly on disk.
    store.write_text(json.dumps({"month": "2000-01", "spent_usd": 999.0}), encoding="utf-8")
    # Reading in the current month rolls the counter back to zero.
    assert budget_guard.current_spend_usd() == 0.0
    budget_guard.assert_within_budget(10.0)  # not blocked by the stale total


def test_status_shape(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "200")
    budget_guard.record_spend(25.0)
    s = budget_guard.status()
    assert s["enabled"] is True
    assert s["cap_usd"] == pytest.approx(200.0)
    assert s["spent_usd"] == pytest.approx(25.0)
    assert s["remaining_usd"] == pytest.approx(175.0)
    assert isinstance(s["month"], str)


def test_invalid_cap_disables(store, monkeypatch):
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "not-a-number")
    assert budget_guard.is_enabled() is False
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "-5")
    assert budget_guard.is_enabled() is False
