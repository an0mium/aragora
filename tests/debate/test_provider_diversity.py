"""Tests for the Provider Diversity Filter."""

from __future__ import annotations

import pytest
from aragora.core_types import DebateResult
from aragora.debate.provider_diversity import (
    AgentInfo,
    DiversityReport,
    ProviderDiversityFilter,
    detect_provider,
)
from aragora.gauntlet.receipt_models import DecisionReceipt
from aragora.pipeline.unified_orchestrator import UnifiedOrchestrator


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


class TestLargeRosterObservability:
    def test_large_roster_receipt_payload_is_bounded(self):
        f = ProviderDiversityFilter(min_providers=3)
        agents = [
            AgentInfo(name=f"claude{i}", model="claude-3-opus", score=1.0 - (i * 0.01))
            for i in range(10)
        ] + [
            AgentInfo(name="gpt-primary", model="gpt-4o", score=0.85),
            AgentInfo(name="gpt-secondary", model="gpt-4o-mini", score=0.75),
        ]
        alternatives = [
            AgentInfo(name="gemini-primary", model="gemini-3.1-pro", score=0.82),
            AgentInfo(name="mistral-primary", model="mistral-large", score=0.81),
        ]

        result, report = f.enforce(agents, alternatives=alternatives)

        assert len(result) == 12
        assert report.roster_size == 12
        assert report.alternative_pool_size == 2
        assert report.runtime_ms >= 0.0
        assert report.receipt_payload_bytes > 0

        payload = report.to_receipt_payload(max_agents_per_provider=2, max_swaps=1)

        assert payload["roster_size"] == 12
        assert payload["provider_count"] == 3
        assert payload["runtime_ms"] == round(report.runtime_ms, 4)
        assert payload["providers"]["anthropic"]["count"] == 9
        assert payload["providers"]["anthropic"]["sample_agents"] == ["claude0", "claude1"]
        assert payload["providers"]["anthropic"]["truncated_agents"] == 7
        assert len(payload["swaps_made"]) == 1
        assert payload["swaps_truncated"] == 0

    def test_large_roster_benchmark_records_runtime_envelope(self):
        f = ProviderDiversityFilter(min_providers=3)
        agents = [
            AgentInfo(name=f"claude{i}", model="claude-3-opus", score=1.0 - (i * 0.01))
            for i in range(11)
        ] + [AgentInfo(name="gpt-primary", model="gpt-4o", score=0.8)]
        alternatives = [
            AgentInfo(name="gemini-primary", model="gemini-3.1-pro", score=0.79),
            AgentInfo(name="mistral-primary", model="mistral-large", score=0.78),
        ]

        benchmark = f.benchmark(agents, alternatives=alternatives, iterations=4)

        assert benchmark.path == "provider_diversity_filter"
        assert benchmark.roster_size == 12
        assert benchmark.iterations == 4
        assert benchmark.average_runtime_ms >= 0.0
        assert benchmark.max_runtime_ms >= benchmark.average_runtime_ms
        assert benchmark.swap_budget == 1
        assert benchmark.swaps_made == 1
        assert benchmark.receipt_payload_bytes > 0

    def test_unified_orchestrator_and_receipt_preserve_large_roster_metadata(self):
        f = ProviderDiversityFilter(min_providers=3)
        agents = [
            AgentInfo(name=f"claude{i}", model="claude-3-opus", score=1.0 - (i * 0.01))
            for i in range(10)
        ] + [
            AgentInfo(name="gpt-primary", model="gpt-4o", score=0.85),
            AgentInfo(name="gpt-secondary", model="gpt-4o-mini", score=0.75),
        ]
        alternatives = [
            AgentInfo(name="gemini-primary", model="gemini-3.1-pro", score=0.82),
            AgentInfo(name="mistral-primary", model="mistral-large", score=0.81),
        ]

        _, report = f.enforce(agents, alternatives=alternatives)
        report.benchmark = f.benchmark(agents, alternatives=alternatives, iterations=2)
        report.receipt_payload_bytes = report.estimate_receipt_payload_bytes()

        result = DebateResult(
            debate_id="debate-large-roster",
            task="Plan a 12-agent heterogeneous debate",
            final_answer="Use a bounded large-roster path",
            confidence=0.82,
            consensus_reached=True,
            rounds_used=2,
            participants=[agent.name for agent in agents],
            metadata={},
        )

        UnifiedOrchestrator._annotate_provider_metadata(
            result,
            provider_hints=["claude-sonnet-4", "gpt-4o", "gemini-3.1-pro"],
            diversity_report=report,
        )

        receipt = DecisionReceipt.from_debate_result(result)

        assert result.metadata["provider_diversity_report"]["roster_size"] == 12
        assert result.metadata["large_roster_runtime"]["path"] == "provider_diversity_filter"
        assert receipt.config_used["provider_diversity"]["provider_count"] == 3
        assert receipt.config_used["large_roster_runtime"]["roster_size"] == 12
        assert receipt.config_used["large_roster_runtime"]["benchmark"]["iterations"] == 2
