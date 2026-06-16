"""PR-1 wiring tests for the OpenRouter Fusion agent.

Covers the four foundational seams: agent registration, cost pricing, feature
flags (default-OFF), and the Pareto routing profile. No network calls.
"""

from __future__ import annotations

from decimal import Decimal

import aragora.agents.api_agents.openrouter  # noqa: F401 - triggers registration
from aragora.agents.api_agents.openrouter import FUSION_MODEL, FusionAgent
from aragora.agents.registry import AgentRegistry
from aragora.billing.usage import calculate_token_cost
from aragora.config.feature_flags import FeatureFlagRegistry
from aragora.routing.selection import (
    DEFAULT_AGENT_EXPERTISE,
    FUSION_EXPERTISE,
    AgentSelector,
)


def test_fusion_agent_is_registered_with_openrouter_model() -> None:
    spec = AgentRegistry.get_spec("fusion")
    assert spec is not None
    assert spec.default_model == FUSION_MODEL == "openrouter/fusion"
    assert spec.env_vars == "OPENROUTER_API_KEY"


def test_fusion_agent_instantiates_with_fusion_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = FusionAgent()
    assert agent.model == "openrouter/fusion"
    assert agent.agent_type == "fusion"


def test_fusion_cost_is_billed_as_premium_not_default() -> None:
    # 1M in + 1M out. Fusion must cost ~4x the openrouter default, never the
    # cheap default fallback.
    fusion = calculate_token_cost("openrouter", "openrouter/fusion", 1_000_000, 1_000_000)
    default = calculate_token_cost("openrouter", "some-unknown-model", 1_000_000, 1_000_000)
    assert fusion == Decimal("8.00") + Decimal("32.00")
    assert fusion > default


def test_fusion_flags_exist_and_default_off(monkeypatch) -> None:
    # A developer/CI shell may export ARAGORA_ENABLE_FUSION; clear it so the
    # default-OFF assertion reflects the registered default, not the ambient env.
    monkeypatch.delenv("ARAGORA_ENABLE_FUSION", raising=False)
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("enable_fusion") is False
    assert reg.get_value("fusion_cost_budget_per_debate") == 50.0
    assert reg.get_value("fusion_cost_monthly_cap") == 5000.0


def test_fusion_routing_profile_is_high_cost(monkeypatch) -> None:
    # Enforcement: with enable_fusion ON, fusion is registered with a high
    # cost/latency profile so the Pareto optimizer skips it on low budgets.
    monkeypatch.setenv("ARAGORA_ENABLE_FUSION", "true")
    # Fusion is an opt-in agent, kept OUT of the always-on defaults; its
    # expertise lives in the separate FUSION_EXPERTISE (a domain->score map).
    assert "fusion" not in DEFAULT_AGENT_EXPERTISE
    assert FUSION_EXPERTISE.get("reasoning", 0) > 0
    selector = AgentSelector.create_with_defaults()
    profile = selector.agent_pool["fusion"]
    assert profile.cost_factor == 4.5
    assert profile.latency_ms == 4500.0
    assert selector.agent_pool["claude"].cost_factor == 1.0


def test_fusion_not_in_pool_when_flag_off(monkeypatch) -> None:
    # The opt-in contract: with enable_fusion OFF (default), fusion must NOT be
    # exposed to the optimizer, even though it's in DEFAULT_AGENT_EXPERTISE.
    monkeypatch.delenv("ARAGORA_ENABLE_FUSION", raising=False)
    selector = AgentSelector.create_with_defaults()
    assert "fusion" not in selector.agent_pool
    assert "claude" in selector.agent_pool  # other agents unaffected
