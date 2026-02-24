"""Tests for capability-specific channel routing and formatters."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aragora.server.debate_origin.formatting import (
    format_consensus_event,
    format_compliance_event,
    format_knowledge_event,
    format_graph_debate_event,
    format_workflow_event,
    format_agent_team_event,
    format_continuum_memory_event,
    format_marketplace_event,
    format_matrix_debate_event,
    format_nomic_loop_event,
    format_rbac_event,
    format_vertical_specialist_event,
    _format_result_message,
)
from aragora.server.debate_origin.models import DebateOrigin


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatConsensusEvent:
    def test_consensus_reached(self):
        result = format_consensus_event({
            "consensus_reached": True,
            "method": "majority_vote",
            "confidence": 0.85,
            "topic": "Should we migrate to Rust?",
            "participants": ["claude", "gpt-4o", "gemini"],
        })
        assert "Consensus Reached" in result
        assert "85%" in result
        assert "majority_vote" in result
        assert "claude" in result

    def test_no_consensus(self):
        result = format_consensus_event({
            "consensus_reached": False,
            "method": "supermajority",
            "confidence": 0.4,
        })
        assert "No Consensus" in result
        assert "40%" in result

    def test_with_proof_hash(self):
        result = format_consensus_event({
            "consensus_reached": True,
            "proof": {"hash": "abc123def456789"},
        })
        assert "abc123def456" in result

    def test_empty_payload(self):
        result = format_consensus_event({})
        assert "No Consensus" in result


class TestFormatComplianceEvent:
    def test_compliant(self):
        result = format_compliance_event({
            "compliant": True,
            "score": 0.95,
            "frameworks_checked": ["hipaa", "gdpr"],
        })
        assert "passed" in result
        assert "95%" in result
        assert "hipaa" in result

    def test_non_compliant_with_issues(self):
        result = format_compliance_event({
            "compliant": False,
            "score": 0.6,
            "issues": [
                {"severity": "critical", "description": "Missing encryption"},
                {"severity": "high", "description": "No audit trail"},
                {"severity": "low", "description": "Documentation gap"},
            ],
        })
        assert "FAILED" in result
        assert "3 total" in result
        assert "1 critical" in result
        assert "Missing encryption" in result

    def test_many_issues_truncated(self):
        issues = [{"severity": "low", "description": f"Issue {i}"} for i in range(10)]
        result = format_compliance_event({"compliant": False, "issues": issues})
        assert "7 more" in result


class TestFormatKnowledgeEvent:
    def test_ingestion_complete(self):
        result = format_knowledge_event({
            "km_event": "ingestion_complete",
            "topic": "Q4 sales data",
            "item_count": 42,
            "source": "salesforce",
        })
        assert "Knowledge Ingested" in result
        assert "42" in result
        assert "salesforce" in result

    def test_staleness_detected(self):
        result = format_knowledge_event({
            "km_event": "staleness_detected",
            "item_count": 5,
        })
        assert "Stale Knowledge" in result

    def test_with_search_items(self):
        result = format_knowledge_event({
            "km_event": "search_complete",
            "items": [
                {"title": "Revenue Report", "relevance_score": 0.92},
                {"title": "Hiring Plan", "relevance_score": 0.78},
            ],
        })
        assert "Revenue Report" in result
        assert "92%" in result


class TestFormatGraphDebateEvent:
    def test_complete_graph(self):
        result = format_graph_debate_event({
            "status": "complete",
            "topic": "Architecture decision",
            "node_count": 12,
            "edge_count": 8,
            "conclusion": "Microservices recommended",
        })
        assert "Graph Debate Complete" in result
        assert "12 claims" in result
        assert "Microservices recommended" in result

    def test_minimal(self):
        result = format_graph_debate_event({})
        assert "Graph Debate" in result


class TestFormatWorkflowEvent:
    def test_workflow_started(self):
        result = format_workflow_event({
            "wf_event": "workflow_started",
            "workflow_name": "Deploy Pipeline",
            "total_steps": 5,
        })
        assert "Workflow Started" in result
        assert "Deploy Pipeline" in result

    def test_step_completed(self):
        result = format_workflow_event({
            "wf_event": "step_completed",
            "workflow_name": "Data Ingestion",
            "current_step": {"name": "Transform"},
            "completed_steps": 2,
            "total_steps": 4,
        })
        assert "Step Complete" in result
        assert "Transform" in result
        assert "2/4" in result

    def test_approval_required(self):
        result = format_workflow_event({
            "wf_event": "approval_required",
            "workflow_name": "Production Deploy",
        })
        assert "Approval Required" in result
        assert "approve or reject" in result


class TestCapabilityEventInResultFormatter:
    def test_capability_event_passes_through(self):
        origin = DebateOrigin(
            debate_id="test-1", platform="slack",
            channel_id="C123", user_id="U456",
        )
        result = _format_result_message(
            {"_capability_event": True, "final_answer": "**Consensus Reached**\nAll good."},
            origin,
        )
        assert result == "**Consensus Reached**\nAll good."

    def test_non_capability_event_formats_normally(self):
        origin = DebateOrigin(
            debate_id="test-2", platform="slack",
            channel_id="C123", user_id="U456",
        )
        result = _format_result_message(
            {"consensus_reached": True, "final_answer": "Yes", "confidence": 0.9},
            origin,
        )
        assert "Debate Complete" in result


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------


class TestRouteCapabilityEvent:
    @pytest.mark.asyncio
    async def test_routes_consensus_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-1", platform="slack",
                channel_id="C123", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-1", "consensus_proof", {
                "consensus_reached": True,
                "confidence": 0.9,
            })
            assert ok is True
            mock_route.assert_awaited_once()
            call_args = mock_route.call_args
            assert call_args[0][1]["_capability_event"] is True

    @pytest.mark.asyncio
    async def test_no_origin_returns_false(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=None,
        ):
            ok = await route_capability_event("d-missing", "consensus_proof", {})
            assert ok is False

    @pytest.mark.asyncio
    async def test_unknown_event_type_returns_false(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-2", platform="slack",
                channel_id="C123", user_id="U456",
            ),
        ):
            ok = await route_capability_event("d-2", "unknown_capability", {})
            assert ok is False

    @pytest.mark.asyncio
    async def test_routes_compliance_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-3", platform="teams",
                channel_id="T789", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-3", "compliance_check", {
                "compliant": False,
                "score": 0.6,
                "frameworks_checked": ["hipaa"],
            })
            assert ok is True
            result = mock_route.call_args[0][1]
            assert "FAILED" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_routes_workflow_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-4", platform="discord",
                channel_id="D123", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-4", "workflow_event", {
                "wf_event": "approval_required",
                "workflow_name": "Deploy",
            })
            assert ok is True
            result = mock_route.call_args[0][1]
            assert "Approval Required" in result["final_answer"]
