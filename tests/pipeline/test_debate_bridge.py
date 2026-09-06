"""Tests for DebateOutcomeBridge (Task 8, #811).

Covers:
- Full debate result populates all fields
- Minimal/empty result returns defaults
- Dissent extraction (dict and non-dict entries)
- Agent scoring and recommended_agents ordering
- Complexity estimation from round counts
- Consensus claims → acceptance criteria
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.pipeline.debate_bridge import DebateOutcomeBridge


def _make_full_debate_result() -> dict[str, Any]:
    """Return a debate result dict with all expected fields populated."""
    return {
        "agent_scores": {
            "claude-sonnet": 0.92,
            "gpt-6-astra": 0.88,
            "gemini-3.1-pro-preview": 0.75,
            "deepseek-r1": 0.60,
        },
        "dissent": [
            {"concern": "Token bucket may not handle bursts well"},
            {"concern": "Needs benchmarking under load"},
        ],
        "consensus_claims": [
            {"claim": "Token bucket is the recommended approach"},
            {"claim": "Rate limit should be configurable per-tenant"},
        ],
        "rounds_completed": 3,
    }


class TestDebateOutcomeBridge:
    """Test DebateOutcomeBridge.extract_workflow_hints()."""

    def test_full_debate_result_populates_all_fields(self) -> None:
        """With a full result, all hint fields should be populated."""
        bridge = DebateOutcomeBridge()
        hints = bridge.extract_workflow_hints(_make_full_debate_result())

        assert "recommended_agents" in hints
        assert "risk_factors" in hints
        assert "dissent_summary" in hints
        assert "acceptance_criteria" in hints
        assert "estimated_complexity" in hints

        assert len(hints["recommended_agents"]) == 3
        assert len(hints["risk_factors"]) == 2
        assert hints["dissent_summary"] == "2 agent(s) dissented"
        assert len(hints["acceptance_criteria"]) == 2

    def test_minimal_result_returns_defaults(self) -> None:
        """An empty dict should return all default values."""
        bridge = DebateOutcomeBridge()
        hints = bridge.extract_workflow_hints({})

        assert hints["recommended_agents"] == []
        assert hints["risk_factors"] == []
        assert hints["dissent_summary"] == ""
        assert hints["acceptance_criteria"] == []
        assert hints["estimated_complexity"] == "medium"

    def test_agent_scoring_top_three_ordered(self) -> None:
        """recommended_agents should be the top 3 agents by score, descending."""
        bridge = DebateOutcomeBridge()
        result = {
            "agent_scores": {
                "alpha": 0.5,
                "beta": 0.9,
                "gamma": 0.7,
                "delta": 0.8,
            }
        }
        hints = bridge.extract_workflow_hints(result)

        assert hints["recommended_agents"] == ["beta", "delta", "gamma"]

    def test_agent_scoring_fewer_than_three(self) -> None:
        """When fewer than 3 agents, return all of them."""
        bridge = DebateOutcomeBridge()
        result = {"agent_scores": {"only-one": 1.0}}
        hints = bridge.extract_workflow_hints(result)

        assert hints["recommended_agents"] == ["only-one"]

    def test_dissent_extraction_with_dict_entries(self) -> None:
        """Dissent dicts with 'concern' key should be extracted."""
        bridge = DebateOutcomeBridge()
        result = {
            "dissent": [
                {"concern": "Latency risk", "agent": "agent-2"},
                {"concern": "Cost overrun"},
            ]
        }
        hints = bridge.extract_workflow_hints(result)

        assert hints["risk_factors"] == ["Latency risk", "Cost overrun"]
        assert hints["dissent_summary"] == "2 agent(s) dissented"

    def test_dissent_extraction_with_non_dict_entries(self) -> None:
        """Non-dict dissent entries should be stringified."""
        bridge = DebateOutcomeBridge()
        result = {"dissent": ["plain string dissent", 42]}
        hints = bridge.extract_workflow_hints(result)

        assert hints["risk_factors"] == ["plain string dissent", "42"]

    def test_dissent_extraction_mixed_entries(self) -> None:
        """Mixed dict and non-dict entries should all be handled."""
        bridge = DebateOutcomeBridge()
        result = {
            "dissent": [
                {"concern": "structured concern"},
                "unstructured concern",
            ]
        }
        hints = bridge.extract_workflow_hints(result)

        assert hints["risk_factors"] == [
            "structured concern",
            "unstructured concern",
        ]

    def test_consensus_claims_extraction(self) -> None:
        """consensus_claims dicts should populate acceptance_criteria."""
        bridge = DebateOutcomeBridge()
        result = {
            "consensus_claims": [
                {"claim": "Use TLS 1.3"},
                {"claim": "Rotate keys monthly"},
            ]
        }
        hints = bridge.extract_workflow_hints(result)

        assert hints["acceptance_criteria"] == ["Use TLS 1.3", "Rotate keys monthly"]

    def test_consensus_claims_non_dict(self) -> None:
        """Non-dict consensus claims should be stringified."""
        bridge = DebateOutcomeBridge()
        result = {"consensus_claims": ["plain claim"]}
        hints = bridge.extract_workflow_hints(result)

        assert hints["acceptance_criteria"] == ["plain claim"]

    def test_complexity_low_from_few_rounds(self) -> None:
        """1-2 rounds → low complexity."""
        bridge = DebateOutcomeBridge()
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 1})["estimated_complexity"] == "low"
        )
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 2})["estimated_complexity"] == "low"
        )

    def test_complexity_medium_from_moderate_rounds(self) -> None:
        """3-4 rounds → medium complexity."""
        bridge = DebateOutcomeBridge()
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 3})["estimated_complexity"]
            == "medium"
        )
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 4})["estimated_complexity"]
            == "medium"
        )

    def test_complexity_high_from_many_rounds(self) -> None:
        """5+ rounds → high complexity."""
        bridge = DebateOutcomeBridge()
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 5})["estimated_complexity"] == "high"
        )
        assert (
            bridge.extract_workflow_hints({"rounds_completed": 10})["estimated_complexity"]
            == "high"
        )

    def test_complexity_default_when_rounds_missing(self) -> None:
        """When rounds_completed is missing, default to medium."""
        bridge = DebateOutcomeBridge()
        hints = bridge.extract_workflow_hints({})
        assert hints["estimated_complexity"] == "medium"

    def test_empty_agent_scores_dict(self) -> None:
        """Empty agent_scores dict should produce empty recommended_agents."""
        bridge = DebateOutcomeBridge()
        hints = bridge.extract_workflow_hints({"agent_scores": {}})
        assert hints["recommended_agents"] == []

    def test_empty_dissent_list(self) -> None:
        """Empty dissent list should produce empty risk_factors and no summary."""
        bridge = DebateOutcomeBridge()
        hints = bridge.extract_workflow_hints({"dissent": []})
        assert hints["risk_factors"] == []
        assert hints["dissent_summary"] == ""
