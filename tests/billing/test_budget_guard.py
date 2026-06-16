"""Tests for the fail-closed monthly budget guard (anti-runaway-bill)."""

from __future__ import annotations

import json

import pytest

from aragora.billing import budget_guard
from aragora.billing.budget_guard import BudgetExceededError


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the guard at a temp store and reset the process-local fallback total."""
    path = tmp_path / "budget_guard.json"
    monkeypatch.setenv("ARAGORA_BUDGET_GUARD_STORE", str(path))
    budget_guard._mem_state.clear()  # the in-memory fallback is a shared global
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


def test_blocks_exactly_at_cap(store, monkeypatch):
    # `>=` means the cap stops AT the limit, not one call past it.
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "100")
    budget_guard.record_spend(100.0)
    with pytest.raises(BudgetExceededError):
        budget_guard.assert_within_budget(0.0)


def test_fails_closed_when_disk_store_unwritable(tmp_path, monkeypatch):
    # Point the store under a *file* so mkdir/open fail -> disk persistence is
    # impossible. The in-memory fallback must still enforce the cap (NOT fail open).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("ARAGORA_BUDGET_GUARD_STORE", str(blocker / "budget.json"))
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "10")
    budget_guard._mem_state.clear()
    budget_guard.record_spend(25.0)  # disk write fails; in-memory keeps it
    assert budget_guard.current_spend_usd() >= 25.0
    with pytest.raises(BudgetExceededError):
        budget_guard.assert_within_budget(0.0)


def test_no_metered_api_agent_bypasses_the_cap():
    """Every metered API agent's generate() must invoke the budget gate.

    Source-level guard: if a new metered agent forgets the precall, this fails.
    (Local agents like ollama/lm_studio are free and intentionally excluded.)
    """
    import pathlib

    base = pathlib.Path(budget_guard.__file__).resolve().parents[1] / "agents" / "api_agents"
    needles = ("_enforce_budget_precall", "assert_within_budget")
    for fname in ("anthropic.py", "gemini.py", "openrouter.py", "openai_compatible.py"):
        text = (base / fname).read_text(encoding="utf-8")
        assert any(n in text for n in needles), f"{fname} does not invoke the budget gate"


@pytest.mark.parametrize("corrupt", ["null", "[1, 2, 3]", "42", '"a string"'])
def test_corrupt_disk_store_falls_back_not_crash(store, monkeypatch, corrupt):
    """A non-object store (null/list/scalar) must degrade to mem-only, never raise.

    Regression: `_disk_read`/`_disk_add` called `.get()` on `json.load` output
    without an isinstance guard, so a corrupted store raised AttributeError on the
    next metered call (a crash, not the intended fail-closed mem fallback).
    """
    monkeypatch.setenv("ARAGORA_MONTHLY_BUDGET_USD", "100")
    store.write_text(corrupt, encoding="utf-8")

    # Read side: corrupt store reads as "unusable" (None), not an exception.
    assert budget_guard._disk_read() is None

    # Full surface stays callable: spend recording and the gate must not raise.
    budget_guard.record_spend(5.0)
    assert budget_guard.current_spend_usd() >= 5.0  # mem fallback carried the spend
    budget_guard.assert_within_budget(1.0, label="cheap-tier")  # under cap → no raise
