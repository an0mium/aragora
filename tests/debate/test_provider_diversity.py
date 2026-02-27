"""Tests for the Provider Diversity Filter."""

from __future__ import annotations

import pytest
from aragora.debate.provider_diversity import (
    AgentInfo,
    DiversityReport,
    ProviderDiversityFilter,
    detect_provider,
)


class TestDetectProvider:
    def test_anthropic(self):
        assert detect_provider("claude-3-opus") == "anthropic"
        assert detect_provider("claude-sonnet-4") == "anthropic"

    def test_openai(self):
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("o1-preview") == "openai"

    def test_google(self):
        assert detect_provider("gemini-3.1-pro") == "google"

    def test_mistral(self):
        assert detect_provider("mistral-large") == "mistral"
        assert detect_provider("codestral") == "mistral"

    def test_xai(self):
        assert detect_provider("grok-2") == "xai"

    def test_meta(self):
        assert detect_provider("llama-3.1-70b") == "meta"

    def test_deepseek(self):
        assert detect_provider("deepseek-r1") == "deepseek"

    def test_unknown(self):
        assert detect_provider("custom-model") == "unknown"


class TestDiversityCheck:
    def test_diverse_team(self):
        f = ProviderDiversityFilter(min_providers=2)
        agents = [
            AgentInfo(name="a1", model="claude-3-opus"),
            AgentInfo(name="a2", model="gpt-4o"),
        ]
        report = f.check(agents)
        assert report.meets_minimum
        assert report.provider_count == 2

    def test_homogeneous_team(self):
        f = ProviderDiversityFilter(min_providers=2)
        agents = [
            AgentInfo(name="a1", model="claude-3-opus"),
            AgentInfo(name="a2", model="claude-sonnet-4"),
        ]
        report = f.check(agents)
        assert not report.meets_minimum
        assert report.provider_count == 1

    def test_three_provider_minimum(self):
        f = ProviderDiversityFilter(min_providers=3)
        agents = [
            AgentInfo(name="a1", model="claude-3-opus"),
            AgentInfo(name="a2", model="gpt-4o"),
        ]
        report = f.check(agents)
        assert not report.meets_minimum


class TestDiversityEnforce:
    def test_already_diverse(self):
        f = ProviderDiversityFilter()
        agents = [
            AgentInfo(name="a1", model="claude-3-opus", score=0.9),
            AgentInfo(name="a2", model="gpt-4o", score=0.8),
        ]
        result, report = f.enforce(agents)
        assert report.meets_minimum
        assert len(report.swaps_made) == 0

    def test_swap_to_diversify(self):
        f = ProviderDiversityFilter(min_providers=2)
        agents = [
            AgentInfo(name="claude1", model="claude-3-opus", score=0.9),
            AgentInfo(name="claude2", model="claude-sonnet-4", score=0.7),
            AgentInfo(name="claude3", model="claude-3-haiku", score=0.5),
        ]
        alternatives = [
            AgentInfo(name="gpt1", model="gpt-4o", score=0.8),
        ]
        result, report = f.enforce(agents, alternatives=alternatives)
        assert report.meets_minimum
        assert len(report.swaps_made) == 1
        # Lowest-scoring claude (claude3) should be swapped
        assert report.swaps_made[0] == ("claude3", "gpt1")
        result_names = {a.name for a in result}
        assert "gpt1" in result_names
        assert "claude3" not in result_names

    def test_no_alternatives_available(self):
        f = ProviderDiversityFilter(min_providers=2)
        agents = [
            AgentInfo(name="claude1", model="claude-3-opus", score=0.9),
            AgentInfo(name="claude2", model="claude-sonnet-4", score=0.7),
        ]
        result, report = f.enforce(agents, alternatives=[])
        assert not report.meets_minimum
        assert len(report.swaps_made) == 0

    def test_agent_info_auto_detect_provider(self):
        a = AgentInfo(name="x", model="gpt-4o")
        assert a.provider == "openai"

    def test_explicit_provider(self):
        a = AgentInfo(name="x", model="custom", provider="custom_co")
        assert a.provider == "custom_co"
