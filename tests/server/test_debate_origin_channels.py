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


# ---------------------------------------------------------------------------
# New capability formatter tests (7 remaining capabilities)
# ---------------------------------------------------------------------------


class TestFormatAgentTeamEvent:
    def test_selection_complete(self):
        result = format_agent_team_event({
            "event": "selection_complete",
            "topic": "Rate limiter design",
            "team_size": 5,
            "strategy": "elo_weighted",
            "agents": [{"name": "claude"}, {"name": "gpt-4o"}, {"name": "gemini"}],
        })
        assert "Agent Team Selected" in result
        assert "elo_weighted" in result
        assert "claude" in result

    def test_empty(self):
        result = format_agent_team_event({})
        assert "Agent Team Update" in result

    def test_many_agents_truncated(self):
        agents = [{"name": f"agent-{i}"} for i in range(10)]
        result = format_agent_team_event({"agents": agents, "team_size": 10})
        assert "4 more" in result


class TestFormatContinuumMemoryEvent:
    def test_consolidation(self):
        result = format_continuum_memory_event({
            "cm_event": "consolidation",
            "tier": "slow",
            "item_count": 42,
            "summary": "Merged cross-session debate patterns",
        })
        assert "Memory Consolidated" in result
        assert "slow" in result
        assert "42" in result

    def test_promotion(self):
        result = format_continuum_memory_event({
            "cm_event": "promotion",
            "tier": "fast",
        })
        assert "Memory Promoted" in result

    def test_empty(self):
        result = format_continuum_memory_event({})
        assert "Memory Update" in result


class TestFormatMarketplaceEvent:
    def test_published(self):
        result = format_marketplace_event({
            "mp_event": "published",
            "name": "Security Audit Template",
            "category": "compliance",
            "rating": 4.5,
        })
        assert "Template Published" in result
        assert "Security Audit Template" in result
        assert "4.5" in result

    def test_empty(self):
        result = format_marketplace_event({})
        assert "Marketplace Update" in result


class TestFormatMatrixDebateEvent:
    def test_complete(self):
        result = format_matrix_debate_event({
            "status": "complete",
            "topic": "Cloud provider selection",
            "dimensions": ["cost", "reliability", "features"],
            "cell_count": 9,
            "conclusion": "AWS recommended for enterprise workloads",
        })
        assert "Matrix Debate Complete" in result
        assert "cost" in result
        assert "9" in result
        assert "AWS recommended" in result

    def test_empty(self):
        result = format_matrix_debate_event({})
        assert "Matrix Debate" in result


class TestFormatNomicLoopEvent:
    def test_cycle_complete(self):
        result = format_nomic_loop_event({
            "nl_event": "cycle_complete",
            "cycle": 3,
            "phase": "verify",
            "goal": "Improve test coverage for billing module",
            "files_changed": ["billing/cost_tracker.py", "tests/billing/test_cost.py"],
        })
        assert "Nomic Cycle Complete" in result
        assert "3" in result
        assert "verify" in result
        assert "Files Changed" in result

    def test_empty(self):
        result = format_nomic_loop_event({})
        assert "Nomic Loop Update" in result


class TestFormatRbacEvent:
    def test_role_assigned(self):
        result = format_rbac_event({
            "rbac_event": "role_assigned",
            "user": "alice@example.com",
            "role": "admin",
            "reason": "Promoted to workspace administrator",
        })
        assert "Role Assigned" in result
        assert "alice@example.com" in result
        assert "admin" in result

    def test_permission_denied(self):
        result = format_rbac_event({
            "rbac_event": "permission_denied",
            "user": "bob",
            "permission": "backups:delete",
        })
        assert "Permission Denied" in result
        assert "backups:delete" in result

    def test_empty(self):
        result = format_rbac_event({})
        assert "RBAC Update" in result


class TestFormatVerticalSpecialistEvent:
    def test_analysis_complete(self):
        result = format_vertical_specialist_event({
            "vs_event": "analysis_complete",
            "vertical": "healthcare",
            "specialist": "HIPAA Compliance Agent",
            "confidence": 0.92,
            "summary": "All PHI data handling meets HIPAA standards",
        })
        assert "Specialist Analysis Complete" in result
        assert "healthcare" in result
        assert "92%" in result

    def test_empty(self):
        result = format_vertical_specialist_event({})
        assert "Specialist Update" in result


class TestRouteNewCapabilityEvents:
    @pytest.mark.asyncio
    async def test_routes_agent_team_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-10", platform="slack",
                channel_id="C123", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-10", "agent_team_selection", {
                "event": "selection_complete",
                "team_size": 5,
            })
            assert ok is True
            result = mock_route.call_args[0][1]
            assert "Agent Team Selected" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_routes_nomic_loop_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-11", platform="teams",
                channel_id="T789", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-11", "nomic_loop", {
                "nl_event": "cycle_complete",
                "cycle": 5,
            })
            assert ok is True
            result = mock_route.call_args[0][1]
            assert "Nomic Cycle Complete" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_routes_matrix_debate_event(self):
        from aragora.server.debate_origin.router import route_capability_event

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=DebateOrigin(
                debate_id="d-12", platform="discord",
                channel_id="D123", user_id="U456",
            ),
        ), patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            ok = await route_capability_event("d-12", "matrix_debate", {
                "status": "complete",
                "topic": "Architecture",
            })
            assert ok is True
            result = mock_route.call_args[0][1]
            assert "Matrix Debate Complete" in result["final_answer"]
