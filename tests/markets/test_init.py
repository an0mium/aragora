"""Tests for the AGT-04 markets module entry point and feature flag."""

from __future__ import annotations

import importlib

import pytest

from aragora import markets


def test_synthetic_markets_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", raising=False)
    assert markets.synthetic_markets_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_truthy_env_values_enable_markets(monkeypatch, value: str) -> None:
    monkeypatch.setenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", value)
    assert markets.synthetic_markets_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_falsy_env_values_keep_markets_disabled(monkeypatch, value: str) -> None:
    monkeypatch.setenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", value)
    assert markets.synthetic_markets_enabled() is False


def test_enable_synthetic_markets_flips_flag(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", raising=False)
    assert markets.synthetic_markets_enabled() is False
    markets.enable_synthetic_markets()
    try:
        assert markets.synthetic_markets_enabled() is True
    finally:
        monkeypatch.delenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", raising=False)


def test_public_exports_present() -> None:
    expected = {
        "Market",
        "MarketPosition",
        "MarketStore",
        "ResolutionEvent",
        "ResolutionOutcome",
        "QuestionKind",
        "GitHubMarketResolver",
        "ResolutionError",
        "resolve_market",
        "brier_score",
        "aggregate_brier",
        "binary_outcome_value",
        "BrierBreakdown",
        "synthetic_markets_enabled",
        "enable_synthetic_markets",
    }
    missing = expected - set(markets.__all__)
    assert not missing, f"missing public exports: {missing}"


def test_module_can_be_reimported_cleanly(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_SYNTHETIC_MARKETS_ENABLED", raising=False)
    importlib.reload(markets)
    assert markets.synthetic_markets_enabled() is False
