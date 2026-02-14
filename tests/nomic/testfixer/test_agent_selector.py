"""Tests for ELO-based agent selection."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.nomic.testfixer.agent_selector import (
    AgentSelector,
    CATEGORY_TO_DOMAIN,
    DEFAULT_FALLBACK_AGENTS,
)
from aragora.nomic.testfixer.analyzer import FailureCategory


@dataclass
class FakeRating:
    agent_name: str
    elo: float = 1600.0


class TestCategoryToDomainMapping:
    def test_all_categories_mapped(self):
        for cat in FailureCategory:
            assert cat in CATEGORY_TO_DOMAIN, f"{cat} not mapped"

    def test_unknown_maps_to_general(self):
        assert CATEGORY_TO_DOMAIN[FailureCategory.UNKNOWN] == "general"


class TestAgentSelectorWithELO:
    def test_uses_elo_rankings_when_available(self):
        elo = MagicMock()
        elo.get_top_agents_for_domain.return_value = [
            FakeRating("gemini-api", 1800.0),
            FakeRating("anthropic-api", 1750.0),
        ]

        selector = AgentSelector(elo_system=elo, fallback_agents=["openai-api"])
        generators = selector.select_agents_for_category(FailureCategory.IMPL_BUG, limit=2)

        elo.get_top_agents_for_domain.assert_called_once_with("impl_bug", limit=2)
        assert len(generators) <= 2

    def test_falls_back_on_empty_elo(self):
        elo = MagicMock()
        elo.get_top_agents_for_domain.return_value = []

        selector = AgentSelector(elo_system=elo, fallback_agents=["openai-api"])
        agent_types = selector._get_agent_types("test_assertion", 3)

        assert agent_types == ["openai-api"]

    def test_falls_back_on_elo_error(self):
        elo = MagicMock()
        elo.get_top_agents_for_domain.side_effect = RuntimeError("db error")

        selector = AgentSelector(elo_system=elo, fallback_agents=["openai-api"])
        agent_types = selector._get_agent_types("impl_bug", 3)

        assert agent_types == ["openai-api"]


class TestAgentSelectorWithoutELO:
    def test_uses_default_fallback(self):
        selector = AgentSelector()
        agent_types = selector._get_agent_types("general", 3)
        assert agent_types == list(DEFAULT_FALLBACK_AGENTS)

    def test_custom_fallback_agents(self):
        selector = AgentSelector(fallback_agents=["gemini-api", "mistral-api"])
        agent_types = selector._get_agent_types("general", 1)
        assert agent_types == ["gemini-api"]
