"""Tests for receipt line-item cost breakdown structure.

Validates that cost_summary data on DecisionReceipt serializes into the
structure consumed by the ReceiptCostBreakdown React component:

  AgentLineItem { agent, provider, model, tokens_in, tokens_out,
                  cost_usd, latency_ms, api_calls, role }

These tests verify the Python→JSON→TypeScript contract so that changes
to the receipt model don't silently break the frontend component.
"""

from __future__ import annotations

import json

import pytest

from aragora.gauntlet.receipt_models import (
    AgentResponseRecord,
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cost_summary_with_line_items() -> dict:
    """Create a realistic cost_summary dict with per-agent and model_usage data."""
    return {
        "debate_id": "debate-line-item-001",
        "total_cost_usd": "0.0924",
        "total_tokens_in": 37300,
        "total_tokens_out": 7100,
        "total_calls": 7,
        "started_at": "2026-03-27T10:00:00+00:00",
        "completed_at": "2026-03-27T10:00:08+00:00",
        "per_agent": {
            "claude-advocate": {
                "agent_name": "claude-advocate",
                "total_cost_usd": "0.0423",
                "total_tokens_in": 12500,
                "total_tokens_out": 3200,
                "call_count": 3,
                "models_used": {"claude-sonnet-4-5-20250514": 3},
                "latency_ms": 4200,
                "role": "proposer",
            },
            "gpt4-critic": {
                "agent_name": "gpt4-critic",
                "total_cost_usd": "0.0312",
                "total_tokens_in": 9800,
                "total_tokens_out": 2100,
                "call_count": 3,
                "models_used": {"gpt-4o": 3},
                "latency_ms": 3100,
                "role": "critic",
            },
            "gemini-judge": {
                "agent_name": "gemini-judge",
                "total_cost_usd": "0.0189",
                "total_tokens_in": 15000,
                "total_tokens_out": 1800,
                "call_count": 1,
                "models_used": {"gemini-2.5-pro": 1},
                "latency_ms": 2800,
                "role": "judge",
            },
        },
        "per_round": {
            "1": {
                "round_number": 1,
                "total_cost_usd": "0.0500",
                "total_tokens_in": 20000,
                "total_tokens_out": 4000,
                "call_count": 4,
            },
        },
        "model_usage": {
            "anthropic/claude-sonnet-4-5-20250514": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250514",
                "total_cost_usd": "0.0423",
                "total_tokens_in": 12500,
                "total_tokens_out": 3200,
                "call_count": 3,
            },
            "openai/gpt-4o": {
                "provider": "openai",
                "model": "gpt-4o",
                "total_cost_usd": "0.0312",
                "total_tokens_in": 9800,
                "total_tokens_out": 2100,
                "call_count": 3,
            },
            "google/gemini-2.5-pro": {
                "provider": "google",
                "model": "gemini-2.5-pro",
                "total_cost_usd": "0.0189",
                "total_tokens_in": 15000,
                "total_tokens_out": 1800,
                "call_count": 1,
            },
        },
    }


def _make_receipt(cost_summary: dict | None = None) -> DecisionReceipt:
    """Build a minimal receipt with optional cost_summary."""
    return DecisionReceipt(
        receipt_id="receipt-line-item-001",
        gauntlet_id="gauntlet-001",
        timestamp="2026-03-27T10:00:08+00:00",
        input_summary="Test decision",
        input_hash="abc123",
        risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
        attacks_attempted=0,
        attacks_successful=0,
        probes_run=0,
        vulnerabilities_found=0,
        verdict="PASS",
        confidence=0.92,
        robustness_score=0.88,
        cost_summary=cost_summary,
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.92,
            supporting_agents=["claude-advocate", "gpt4-critic", "gemini-judge"],
        ),
    )


# =============================================================================
# Tests
# =============================================================================


class TestCostSummaryLineItemContract:
    """Ensure cost_summary structure has the fields the React component needs."""

    def test_per_agent_has_required_fields(self):
        """Each per_agent entry must have tokens, cost, call_count for the UI."""
        cost = _cost_summary_with_line_items()
        required_keys = {
            "agent_name",
            "total_cost_usd",
            "total_tokens_in",
            "total_tokens_out",
            "call_count",
            "models_used",
        }
        for agent_name, agent_data in cost["per_agent"].items():
            missing = required_keys - set(agent_data.keys())
            assert not missing, f"Agent {agent_name} missing keys: {missing}"

    def test_model_usage_has_required_fields(self):
        """Each model_usage entry must have provider, model, tokens, cost."""
        cost = _cost_summary_with_line_items()
        required_keys = {
            "provider",
            "model",
            "total_cost_usd",
            "total_tokens_in",
            "total_tokens_out",
            "call_count",
        }
        for model_key, model_data in cost["model_usage"].items():
            missing = required_keys - set(model_data.keys())
            assert not missing, f"Model {model_key} missing keys: {missing}"

    def test_per_agent_latency_ms_field(self):
        """per_agent entries may include latency_ms for the latency column."""
        cost = _cost_summary_with_line_items()
        for agent_name, agent_data in cost["per_agent"].items():
            assert "latency_ms" in agent_data, f"Agent {agent_name} missing latency_ms"
            assert isinstance(agent_data["latency_ms"], (int, float))

    def test_per_agent_role_field(self):
        """per_agent entries may include role for role badge display."""
        cost = _cost_summary_with_line_items()
        for agent_name, agent_data in cost["per_agent"].items():
            assert "role" in agent_data, f"Agent {agent_name} missing role"
            assert agent_data["role"] in {
                "proposer",
                "critic",
                "judge",
                "mediator",
                "advocate",
                "challenger",
            }


class TestReceiptCostSummarySerialization:
    """Verify receipt serialization preserves cost_summary for the frontend."""

    def test_to_dict_includes_cost_summary(self):
        """Receipt to_dict() should include cost_summary unchanged."""
        cost = _cost_summary_with_line_items()
        receipt = _make_receipt(cost_summary=cost)
        d = receipt.to_dict()
        assert d["cost_summary"] is not None
        assert d["cost_summary"]["total_cost_usd"] == "0.0924"
        assert len(d["cost_summary"]["per_agent"]) == 3

    def test_to_dict_cost_summary_none_when_absent(self):
        """Receipt to_dict() should have cost_summary=None when not set."""
        receipt = _make_receipt(cost_summary=None)
        d = receipt.to_dict()
        assert d["cost_summary"] is None

    def test_json_round_trip_preserves_cost_summary(self):
        """cost_summary survives JSON serialization round-trip."""
        cost = _cost_summary_with_line_items()
        receipt = _make_receipt(cost_summary=cost)
        json_str = json.dumps(receipt.to_dict())
        restored = json.loads(json_str)
        assert restored["cost_summary"]["per_agent"]["claude-advocate"]["total_tokens_in"] == 12500
        assert restored["cost_summary"]["model_usage"]["openai/gpt-4o"]["model"] == "gpt-4o"

    def test_from_dict_round_trip(self):
        """Receipt can be reconstructed from its dict representation."""
        cost = _cost_summary_with_line_items()
        receipt = _make_receipt(cost_summary=cost)
        d = receipt.to_dict()
        restored = DecisionReceipt.from_dict(d)
        assert restored.cost_summary is not None
        assert len(restored.cost_summary["per_agent"]) == 3
        assert restored.cost_summary["per_agent"]["gpt4-critic"]["call_count"] == 3


class TestLineItemTransformation:
    """Test transforming cost_summary into the flat AgentLineItem[] the UI expects."""

    @staticmethod
    def _transform_to_line_items(cost_summary: dict) -> list[dict]:
        """Transform cost_summary to AgentLineItem-shaped dicts.

        This mirrors the transformation the frontend API handler would perform
        to flatten cost_summary into the array of line items the React
        ReceiptCostBreakdown component consumes.
        """
        items = []
        per_agent = cost_summary.get("per_agent", {})
        model_usage = cost_summary.get("model_usage", {})

        # Build provider lookup from model_usage
        provider_by_model: dict[str, str] = {}
        for _key, mdata in model_usage.items():
            model_name = mdata.get("model", "")
            provider = mdata.get("provider", "")
            if model_name:
                provider_by_model[model_name] = provider

        for agent_name, adata in per_agent.items():
            models_used = adata.get("models_used", {})
            primary_model = ""
            if models_used:
                primary_model = max(models_used, key=lambda m: models_used[m])

            items.append(
                {
                    "agent": adata.get("agent_name", agent_name),
                    "provider": provider_by_model.get(primary_model, "unknown"),
                    "model": primary_model,
                    "tokens_in": adata.get("total_tokens_in", 0),
                    "tokens_out": adata.get("total_tokens_out", 0),
                    "cost_usd": float(adata.get("total_cost_usd", 0)),
                    "latency_ms": adata.get("latency_ms", 0),
                    "api_calls": adata.get("call_count", 0),
                    "role": adata.get("role"),
                }
            )

        return items

    def test_transforms_three_agents(self):
        cost = _cost_summary_with_line_items()
        items = self._transform_to_line_items(cost)
        assert len(items) == 3

    def test_agent_names_preserved(self):
        cost = _cost_summary_with_line_items()
        items = self._transform_to_line_items(cost)
        names = {i["agent"] for i in items}
        assert names == {"claude-advocate", "gpt4-critic", "gemini-judge"}

    def test_provider_resolved_from_model_usage(self):
        cost = _cost_summary_with_line_items()
        items = self._transform_to_line_items(cost)
        by_name = {i["agent"]: i for i in items}
        assert by_name["claude-advocate"]["provider"] == "anthropic"
        assert by_name["gpt4-critic"]["provider"] == "openai"
        assert by_name["gemini-judge"]["provider"] == "google"

    def test_tokens_and_cost_are_numeric(self):
        cost = _cost_summary_with_line_items()
        items = self._transform_to_line_items(cost)
        for item in items:
            assert isinstance(item["tokens_in"], int)
            assert isinstance(item["tokens_out"], int)
            assert isinstance(item["cost_usd"], float)
            assert isinstance(item["latency_ms"], (int, float))
            assert isinstance(item["api_calls"], int)

    def test_roles_carried_through(self):
        cost = _cost_summary_with_line_items()
        items = self._transform_to_line_items(cost)
        by_name = {i["agent"]: i for i in items}
        assert by_name["claude-advocate"]["role"] == "proposer"
        assert by_name["gpt4-critic"]["role"] == "critic"
        assert by_name["gemini-judge"]["role"] == "judge"

    def test_empty_cost_summary(self):
        items = self._transform_to_line_items({"per_agent": {}, "model_usage": {}})
        assert items == []

    def test_agent_with_no_models_used(self):
        cost = {
            "per_agent": {
                "fallback-agent": {
                    "agent_name": "fallback-agent",
                    "total_cost_usd": "0.001",
                    "total_tokens_in": 100,
                    "total_tokens_out": 50,
                    "call_count": 1,
                    "models_used": {},
                },
            },
            "model_usage": {},
        }
        items = self._transform_to_line_items(cost)
        assert len(items) == 1
        assert items[0]["model"] == ""
        assert items[0]["provider"] == "unknown"
