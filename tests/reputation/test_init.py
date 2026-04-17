"""Tests for the aragora.reputation package entry point and feature flag."""

from __future__ import annotations

import importlib

import pytest

from aragora import reputation


def test_feature_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
    assert reputation.reputation_flow_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_truthy_values_enable_flow(monkeypatch, value: str) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", value)
    assert reputation.reputation_flow_enabled() is True


def test_enable_helper_flips_flag(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
    assert reputation.reputation_flow_enabled() is False
    reputation.enable_reputation_flow()
    try:
        assert reputation.reputation_flow_enabled() is True
    finally:
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)


def test_public_exports_present() -> None:
    expected = {
        "StakeableClaim",
        "ResolvedClaim",
        "ReputationDelta",
        "settle_claim",
        "bridge_from_market_position",
        "DOMAIN_PREDICTION_MARKET",
        "reputation_flow_enabled",
        "enable_reputation_flow",
    }
    missing = expected - set(reputation.__all__)
    assert not missing, f"missing public exports: {missing}"


def test_module_reimports_cleanly() -> None:
    importlib.reload(reputation)
