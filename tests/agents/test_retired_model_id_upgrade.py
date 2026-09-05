"""Constructor-time upgrade of retired explicit model ids on native agents.

2026-09-05 merge-gate fix wave, finding O-P2a on #9989: the native API
agents send ``model`` straight to their provider endpoint, so an explicitly
configured id the provider has since retired (``gpt-5.5``, ``grok-4-latest``)
failed the call instead of upgrading. ``gemini.py`` already resolved its id
at construction; ``anthropic``/``openai``/``grok``/``mistral`` did not.

The contract these tests pin, per agent:

* a RETIRED (or explicitly upgrade-mapped) id is replaced by its current id;
* an ACTIVE id -- including an active alias -- is left EXACTLY as passed;
* an UNKNOWN id is left exactly as passed, so a model newer than the catalog
  is still callable.
"""

from __future__ import annotations

import logging

import pytest

from aragora.agents.api_agents import common
from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
from aragora.agents.api_agents.grok import GrokAgent
from aragora.agents.api_agents.mistral import CodestralAgent, MistralAPIAgent
from aragora.agents.api_agents.openai import OpenAIAPIAgent
from aragora.config.model_pins import (
    FABLE_51_DIRECT,
    GPT6_ASTRA_DIRECT,
    GROK_46_DIRECT,
    MISTRAL_LARGE_DIRECT,
)

# (agent class, retired id, its upgrade target, active id, unknown id)
_CASES = [
    pytest.param(
        AnthropicAPIAgent,
        "claude-fable-5",
        FABLE_51_DIRECT,
        "claude-fable-5-1",
        "claude-zeta-9-20991231",
        id="anthropic",
    ),
    pytest.param(
        OpenAIAPIAgent,
        "gpt-5.5",
        GPT6_ASTRA_DIRECT,
        "gpt-6-astra",
        "gpt-7-nova",
        id="openai",
    ),
    pytest.param(
        GrokAgent,
        "grok-4-latest",
        GROK_46_DIRECT,
        "grok-4.6",
        "grok-9-hyper",
        id="grok",
    ),
    pytest.param(
        MistralAPIAgent,
        "mistral-large-2411",
        MISTRAL_LARGE_DIRECT,
        "mistral-medium-2604",
        "mistral-nova-9",
        id="mistral",
    ),
]


@pytest.mark.parametrize(("agent_cls", "retired", "upgraded", "active", "unknown"), _CASES)
def test_retired_explicit_id_is_upgraded(
    agent_cls, retired: str, upgraded: str, active: str, unknown: str
) -> None:
    assert agent_cls(model=retired, api_key="test-key").model == upgraded
    assert agent_cls(model=retired, api_key="test-key").model != retired


@pytest.mark.parametrize(("agent_cls", "retired", "upgraded", "active", "unknown"), _CASES)
def test_active_explicit_id_passes_through_unchanged(
    agent_cls, retired: str, upgraded: str, active: str, unknown: str
) -> None:
    assert agent_cls(model=active, api_key="test-key").model == active


@pytest.mark.parametrize(("agent_cls", "retired", "upgraded", "active", "unknown"), _CASES)
def test_unknown_explicit_id_passes_through_unchanged(
    agent_cls, retired: str, upgraded: str, active: str, unknown: str
) -> None:
    """A model newer than the catalog must still be callable verbatim."""
    assert agent_cls(model=unknown, api_key="test-key").model == unknown


def test_active_alias_is_not_canonicalized() -> None:
    """An active ALIAS is a working native id; the agent must not rewrite it
    to the canonical spelling (that is a resolve_model_id() behaviour this
    deliberately does not use)."""
    assert AnthropicAPIAgent(model="claude-fable-5.1", api_key="k").model == "claude-fable-5.1"
    assert MistralAPIAgent(model="mistral-medium-latest", api_key="k").model == (
        "mistral-medium-latest"
    )


def test_codestral_keeps_its_own_live_sku() -> None:
    """``codestral-latest`` is an UPGRADES key only because the catalog has
    no Codestral row (which makes mistral-medium the right OpenRouter
    fallback target). It is a live native SKU, so CodestralAgent opts out of
    the constructor-time rewrite and keeps calling the code model."""
    assert CodestralAgent(api_key="test-key").model == "codestral-latest"
    assert CodestralAgent.UPGRADE_RETIRED_MODEL_ID is False
    assert MistralAPIAgent.UPGRADE_RETIRED_MODEL_ID is True


def test_upgrade_logs_once_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    common._LOGGED_MODEL_UPGRADES.discard(("gpt-5.5", GPT6_ASTRA_DIRECT))
    with caplog.at_level(logging.WARNING, logger=common.__name__):
        OpenAIAPIAgent(model="gpt-5.5", api_key="test-key")
        OpenAIAPIAgent(model="gpt-5.5", api_key="test-key")
    upgrade_records = [r for r in caplog.records if "upgraded to" in r.getMessage()]
    assert len(upgrade_records) == 1
    assert upgrade_records[0].levelno == logging.WARNING
    assert "gpt-5.5" in upgrade_records[0].getMessage()
    assert GPT6_ASTRA_DIRECT in upgrade_records[0].getMessage()


class TestUpgradeHelper:
    """Direct unit coverage of the shared helper the four agents call."""

    def test_retired_catalog_row_upgrades(self) -> None:
        assert common.upgrade_retired_model_id("grok-4.5") == GROK_46_DIRECT

    def test_upgrades_key_absent_from_catalog_upgrades(self) -> None:
        assert common.upgrade_retired_model_id("gpt-4o") == GPT6_ASTRA_DIRECT

    def test_active_row_unchanged(self) -> None:
        assert common.upgrade_retired_model_id("gpt-6-astra") == "gpt-6-astra"

    def test_unknown_unchanged(self) -> None:
        assert common.upgrade_retired_model_id("totally-new-model-2099") == (
            "totally-new-model-2099"
        )

    def test_empty_unchanged(self) -> None:
        assert common.upgrade_retired_model_id("") == ""
