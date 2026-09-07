"""Tests for the opt-in OpenRouter Fusion agent registration.

Fusion is a multi-model council+judge endpoint -- itself a *blend*, so it is
registered as a normal selectable agent BUT deliberately kept out of the
default allowlist (opt-in) and out of the quorum family map (it must never
count as an independent consensus family). These tests lock in that contract.
"""

from __future__ import annotations

# Importing the module guarantees the @AgentRegistry.register decorator has run
# in this process, exactly as production does via api_agents/__init__.py.
from aragora.agents.api_agents import openrouter as _openrouter  # noqa: F401
from aragora.agents.registry import AgentRegistry


def test_fusion_model_constant() -> None:
    assert _openrouter.FUSION_MODEL == "openrouter/fusion"


def test_fusion_agent_is_registered() -> None:
    assert AgentRegistry.is_registered("fusion")
    spec = AgentRegistry.get_spec("fusion")
    assert spec.default_model == "openrouter/fusion"
    assert spec.env_vars == "OPENROUTER_API_KEY"
    # Categorized as an OpenRouter API agent (reuses that transport/resilience).
    assert "OpenRouter" in (spec.agent_type or "")


def test_fusion_agent_create_uses_fusion_model(monkeypatch) -> None:
    # A dummy key lets the OpenRouter transport construct; no network call is made.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")
    agent = AgentRegistry.create("fusion", name="fusion-judge", role="critic")
    # Reuses the OpenRouterAgent transport (circuit-breaker, rate-limit, tokens).
    assert isinstance(agent, _openrouter.OpenRouterAgent)
    assert agent.model == "openrouter/fusion"
    assert agent.agent_type == "fusion"
    assert agent.role == "critic"


def test_fusion_is_opt_in_not_allowlisted() -> None:
    """Default-OFF posture: registered, but NOT in the server allowlist.

    A deployment must explicitly opt fusion in; it never becomes a debate
    participant just by existing in the registry.
    """
    from aragora.config import ALLOWED_AGENT_TYPES

    assert "fusion" not in ALLOWED_AGENT_TYPES


def test_fusion_is_not_a_quorum_family() -> None:
    """Hard constraint: a blend must never count as an independent family."""
    from aragora.swarm.quorum_evidence import FAMILY_PROVIDERS

    assert "fusion" not in FAMILY_PROVIDERS
