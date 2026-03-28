"""Tests for provider routing integration in Decision Receipts.

Verifies that provider routing decisions are:
- Stored in the DecisionReceipt.provider_routing field
- Populated from DebateResult metadata via from_debate_result()
- Serialized/deserialized correctly through to_dict()/from_dict()
- Rendered in Markdown and HTML export formats
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.gauntlet.receipt_models import DecisionReceipt


# ---------------------------------------------------------------------------
# Helpers – lightweight stub for DebateResult metadata
# ---------------------------------------------------------------------------


@dataclass
class _StubMessage:
    agent: str = "agent-1"
    content: str = "test response"
    role: str = "proposer"
    round: int = 1
    timestamp: str = "2026-03-28T00:00:00"


@dataclass
class _StubVote:
    agent: str = "agent-1"
    choice: str = "accept"
    confidence: float = 0.9


@dataclass
class _StubDebateResult:
    """Minimal DebateResult-like stub with metadata."""

    debate_id: str = "test-debate-001"
    task: str = "Evaluate rate limiter design"
    consensus_reached: bool = True
    confidence: float = 0.85
    final_answer: str = "Use token bucket algorithm"
    winner: str = "agent-1"
    rounds_used: int = 3
    duration_seconds: float = 12.5
    participants: list[str] = field(default_factory=lambda: ["agent-1", "agent-2"])
    messages: list[_StubMessage] = field(default_factory=lambda: [_StubMessage()])
    votes: list[_StubVote] = field(default_factory=lambda: [_StubVote()])
    dissenting_views: list[str] = field(default_factory=list)
    critiques: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    convergence_similarity: float = 0.9
    consensus_strength: str = "strong"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROUTING_METADATA: dict[str, Any] = {
    "provider_routing": {
        "routing_applied": True,
        "routing_strategy": "pareto_balanced",
        "routed_agent_names": ["agent-1", "agent-2"],
        "provider_matches": {
            "agent-1": "claude-sonnet-4",
            "agent-2": "gpt-4o",
        },
        "provider_hint_scores": {
            "claude-sonnet-4": 0.92,
            "gpt-4o": 0.88,
        },
    },
    "provider_hints": ["claude-sonnet-4", "gpt-4o"],
    "provider_names": ["anthropic", "openai"],
}


def _make_receipt_with_routing() -> DecisionReceipt:
    stub = _StubDebateResult(metadata=dict(ROUTING_METADATA))
    return DecisionReceipt.from_debate_result(stub)


@pytest.fixture()
def receipt_with_routing() -> DecisionReceipt:
    return _make_receipt_with_routing()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProviderRoutingField:
    """provider_routing field presence and defaults."""

    def test_default_is_none(self):
        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-03-28T00:00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
        )
        assert receipt.provider_routing is None

    def test_field_accepts_dict(self):
        receipt = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            timestamp="2026-03-28T00:00:00",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            provider_routing={"routing_strategy": "balanced"},
        )
        assert receipt.provider_routing == {"routing_strategy": "balanced"}


class TestFromDebateResultRouting:
    """from_debate_result populates provider_routing from metadata."""

    def test_routing_populated(self, receipt_with_routing: DecisionReceipt):
        pr = receipt_with_routing.provider_routing
        assert pr is not None
        assert pr["routing_applied"] is True
        assert pr["routing_strategy"] == "pareto_balanced"
        assert pr["routed_agent_names"] == ["agent-1", "agent-2"]

    def test_provider_matches_preserved(self, receipt_with_routing: DecisionReceipt):
        pr = receipt_with_routing.provider_routing
        assert pr is not None
        assert pr["provider_matches"]["agent-1"] == "claude-sonnet-4"
        assert pr["provider_matches"]["agent-2"] == "gpt-4o"

    def test_hint_scores_preserved(self, receipt_with_routing: DecisionReceipt):
        pr = receipt_with_routing.provider_routing
        assert pr is not None
        assert pr["provider_hint_scores"]["claude-sonnet-4"] == 0.92

    def test_provider_hints_included(self, receipt_with_routing: DecisionReceipt):
        pr = receipt_with_routing.provider_routing
        assert pr is not None
        assert pr["provider_hints"] == ["claude-sonnet-4", "gpt-4o"]

    def test_provider_names_included(self, receipt_with_routing: DecisionReceipt):
        pr = receipt_with_routing.provider_routing
        assert pr is not None
        assert pr["provider_names"] == ["anthropic", "openai"]

    def test_no_metadata_yields_none(self):
        stub = _StubDebateResult(metadata={})
        receipt = DecisionReceipt.from_debate_result(stub)
        assert receipt.provider_routing is None

    def test_partial_metadata_provider_hints_only(self):
        stub = _StubDebateResult(
            metadata={
                "provider_hints": ["claude-sonnet-4"],
            }
        )
        receipt = DecisionReceipt.from_debate_result(stub)
        assert receipt.provider_routing is not None
        assert receipt.provider_routing["provider_hints"] == ["claude-sonnet-4"]

    def test_partial_metadata_provider_names_only(self):
        stub = _StubDebateResult(
            metadata={
                "provider_names": ["anthropic"],
            }
        )
        receipt = DecisionReceipt.from_debate_result(stub)
        assert receipt.provider_routing is not None
        assert receipt.provider_routing["provider_names"] == ["anthropic"]


class TestRoundtripSerialization:
    """to_dict / from_dict preserve provider_routing."""

    def test_roundtrip_with_routing(self, receipt_with_routing: DecisionReceipt):
        data = receipt_with_routing.to_dict()
        assert "provider_routing" in data
        assert data["provider_routing"]["routing_applied"] is True

        restored = DecisionReceipt.from_dict(data)
        assert restored.provider_routing == receipt_with_routing.provider_routing

    def test_roundtrip_none(self):
        stub = _StubDebateResult(metadata={})
        receipt = DecisionReceipt.from_debate_result(stub)
        data = receipt.to_dict()
        assert data["provider_routing"] is None

        restored = DecisionReceipt.from_dict(data)
        assert restored.provider_routing is None

    def test_json_roundtrip(self, receipt_with_routing: DecisionReceipt):
        json_str = receipt_with_routing.to_json()
        parsed = json.loads(json_str)
        assert parsed["provider_routing"]["routing_strategy"] == "pareto_balanced"


class TestMarkdownExport:
    """Markdown exporter renders provider routing section."""

    def test_routing_section_present(self, receipt_with_routing: DecisionReceipt):
        md = receipt_with_routing.to_markdown()
        assert "## Provider Routing" in md
        assert "pareto_balanced" in md
        assert "Routing Applied:** Yes" in md

    def test_provider_matches_in_markdown(self, receipt_with_routing: DecisionReceipt):
        md = receipt_with_routing.to_markdown()
        assert "claude-sonnet-4" in md
        assert "gpt-4o" in md

    def test_no_routing_no_section(self):
        stub = _StubDebateResult(metadata={})
        receipt = DecisionReceipt.from_debate_result(stub)
        md = receipt.to_markdown()
        assert "## Provider Routing" not in md


class TestHtmlExport:
    """HTML exporter renders provider routing section."""

    def test_routing_section_in_html(self, receipt_with_routing: DecisionReceipt):
        html = receipt_with_routing.to_html()
        assert "Provider Routing" in html
        assert "pareto_balanced" in html

    def test_no_routing_no_html_section(self):
        stub = _StubDebateResult(metadata={})
        receipt = DecisionReceipt.from_debate_result(stub)
        html = receipt.to_html()
        assert "Provider Routing" not in html
