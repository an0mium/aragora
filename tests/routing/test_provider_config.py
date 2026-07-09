"""Frontier model pin parity tests for routing price estimates."""

from __future__ import annotations

import logging

import pytest

from aragora.config import model_pins
from aragora.routing.provider_config import PROVIDER_PRICING, get_estimated_cost


def _canonical_frontier_pins() -> set[str]:
    return {
        getattr(model_pins, name)
        for name in model_pins.__all__
        if name.endswith("_DIRECT") or name.endswith("_VIA_OPENROUTER")
    }


def test_canonical_frontier_pins_have_nonzero_pricing() -> None:
    missing = {
        model
        for model in _canonical_frontier_pins()
        if get_estimated_cost(model, input_tokens=1_000, output_tokens=1_000) <= 0
    }

    assert not missing, f"Canonical frontier pins missing routing prices: {sorted(missing)}"


def test_current_frontier_entries_have_positive_rates() -> None:
    expected = {
        "claude-opus-4-8",
        "anthropic/claude-opus-4.8",
        "gpt-5.5",
        "openai/gpt-5.5",
        "gemini-3.1-pro-preview",
        "google/gemini-3.1-pro-preview",
        "grok-4.5",
        "x-ai/grok-4.5",
        "mistral-large-2512",
        "mistralai/mistral-large-2512",
    }

    for model in expected:
        pricing = PROVIDER_PRICING[model]
        assert pricing.input_cost_per_1k > 0
        assert pricing.output_cost_per_1k > 0


def test_unknown_model_warns_before_returning_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        cost = get_estimated_cost("missing-frontier-model", 1_000, 1_000)

    assert cost == 0.0
    assert "missing-frontier-model" in caplog.text
