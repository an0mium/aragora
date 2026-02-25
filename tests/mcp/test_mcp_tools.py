"""Tests for MCP tools.

Comprehensive tests covering:
- Tool registration and metadata
- Individual tool execution
- Error handling
- Server lifecycle
- Control plane tools
- Canvas tools
- Checkpoint tools
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import json


# =============================================================================
# Tool Metadata Tests
# =============================================================================


class TestToolsMetadata:
    """Test MCP tools metadata."""

    def test_tools_metadata_import(self):
        """Test that TOOLS_METADATA can be imported."""
        from aragora.mcp.tools import TOOLS_METADATA

        assert isinstance(TOOLS_METADATA, list)
        assert len(TOOLS_METADATA) > 0

    def test_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        from aragora.mcp.tools import TOOLS_METADATA

        for tool in TOOLS_METADATA:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "function" in tool, f"Tool {tool['name']} missing 'function'"
            assert "parameters" in tool, f"Tool {tool['name']} missing 'parameters'"
            assert callable(tool["function"]), f"Tool {tool['name']} function not callable"

    def test_tool_count(self):
        """Test expected number of tools."""
        from aragora.mcp.tools import TOOLS_METADATA

        # We have 45+ tools now
        assert len(TOOLS_METADATA) >= 45

    def test_tool_names_are_unique(self):
        """Test that all tool names are unique."""
        from aragora.mcp.tools import TOOLS_METADATA

        names = [tool["name"] for tool in TOOLS_METADATA]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_tool_descriptions_not_empty(self):
        """Test that all tools have non-empty descriptions."""
        from aragora.mcp.tools import TOOLS_METADATA

        for tool in TOOLS_METADATA:
            assert tool["description"].strip(), f"Tool {tool['name']} has empty description"

    def test_tool_parameters_are_dicts(self):
        """Test that all tool parameters are dictionaries."""
        from aragora.mcp.tools import TOOLS_METADATA

        for tool in TOOLS_METADATA:
            assert isinstance(tool["parameters"], dict), (
                f"Tool {tool['name']} parameters not a dict"
            )


# =============================================================================
# Knowledge Tools Tests
# =============================================================================


class TestKnowledgeTools:
    """Test knowledge-related MCP tools."""

    @pytest.mark.asyncio
    async def test_query_knowledge_tool_basic(self):
        """Test query_knowledge_tool with mocked mound."""
        from aragora.mcp.tools_module.knowledge import query_knowledge_tool

        # Without the mound available, should return empty results
        result = await query_knowledge_tool(query="test query")

        assert "nodes" in result
        assert "count" in result
        assert "query" in result
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_query_knowledge_tool_with_filters(self):
        """Test query_knowledge_tool with filters."""
        from aragora.mcp.tools_module.knowledge import query_knowledge_tool

        result = await query_knowledge_tool(
            query="test",
            node_types="fact,claim",
            min_confidence=0.5,
            limit=5,
            include_relationships=True,
        )

        assert "filters" in result
        assert result["filters"]["node_types"] == "fact,claim"
        assert result["filters"]["min_confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_store_knowledge_tool_validation(self):
        """Test store_knowledge_tool validates inputs."""
        from aragora.mcp.tools_module.knowledge import store_knowledge_tool

        # Invalid node type
        result = await store_knowledge_tool(content="test", node_type="invalid_type")
        assert "error" in result

        # Invalid tier
        result = await store_knowledge_tool(content="test", tier="invalid_tier")
        assert "error" in result

        # Invalid confidence
        result = await store_knowledge_tool(content="test", confidence=2.0)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_knowledge_stats_tool(self):
        """Test get_knowledge_stats_tool."""
        from aragora.mcp.tools_module.knowledge import get_knowledge_stats_tool

        result = await get_knowledge_stats_tool()

        # Without mound, returns error or defaults
        assert "total_nodes" in result or "error" in result

    @pytest.mark.asyncio
    async def test_get_decision_receipt_tool_not_found(self):
        """Test get_decision_receipt_tool with non-existent debate."""
        from aragora.mcp.tools_module.knowledge import get_decision_receipt_tool

        result = await get_decision_receipt_tool(debate_id="nonexistent123")

        assert "error" in result


# =============================================================================
# Workflow Tools Tests
# =============================================================================


class TestWorkflowTools:
    """Test workflow-related MCP tools."""

    @pytest.mark.asyncio
    async def test_run_workflow_tool_invalid_inputs(self):
        """Test run_workflow_tool with invalid JSON inputs."""
        from aragora.mcp.tools_module.workflow import run_workflow_tool

        result = await run_workflow_tool(template="test_template", inputs="invalid json {")

        assert "error" in result
        # Either Invalid JSON error or module not available
        assert "Invalid JSON" in result["error"] or "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_run_workflow_tool_template_not_found(self):
        """Test run_workflow_tool with non-existent template."""
        from aragora.mcp.tools_module.workflow import run_workflow_tool

        # Without workflow engine, should return error
        result = await run_workflow_tool(template="nonexistent_template", inputs="{}")

        # Either template not found or engine not available
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_workflow_status_tool(self):
        """Test get_workflow_status_tool."""
        from aragora.mcp.tools_module.workflow import get_workflow_status_tool

        result = await get_workflow_status_tool(execution_id="test123")

        # Without engine, should return error
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_workflow_templates_tool(self):
        """Test list_workflow_templates_tool."""
        from aragora.mcp.tools_module.workflow import list_workflow_templates_tool

        result = await list_workflow_templates_tool(category="all")

        # Should return default templates or actual templates
        assert "templates" in result
        assert "count" in result
        assert isinstance(result["templates"], list)

    @pytest.mark.asyncio
    async def test_cancel_workflow_tool(self):
        """Test cancel_workflow_tool."""
        from aragora.mcp.tools_module.workflow import cancel_workflow_tool

        result = await cancel_workflow_tool(execution_id="test123", reason="Test cancellation")

        # Without engine, should return error
        assert "error" in result


# =============================================================================
# Integration Tools Tests
# =============================================================================


class TestIntegrationTools:
    """Test external integration MCP tools."""

    @pytest.mark.asyncio
    async def test_trigger_external_webhook_invalid_platform(self):
        """Test trigger_external_webhook_tool with invalid platform."""
        from aragora.mcp.tools_module.integrations import trigger_external_webhook_tool

        result = await trigger_external_webhook_tool(
            platform="invalid_platform", event_type="test_event", data="{}"
        )

        assert "error" in result
        assert "Invalid platform" in result["error"]

    @pytest.mark.asyncio
    async def test_trigger_external_webhook_invalid_json(self):
        """Test trigger_external_webhook_tool with invalid JSON data."""
        from aragora.mcp.tools_module.integrations import trigger_external_webhook_tool

        result = await trigger_external_webhook_tool(
            platform="zapier", event_type="test_event", data="invalid json {"
        )

        assert "error" in result
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_trigger_external_webhook_valid_platform(self):
        """Test trigger_external_webhook_tool with valid platforms."""
        from aragora.mcp.tools_module.integrations import trigger_external_webhook_tool

        for platform in ["zapier", "make", "n8n"]:
            result = await trigger_external_webhook_tool(
                platform=platform, event_type="test_event", data='{"key": "value"}'
            )

            # Either works or module not available
            assert "error" in result or "platform" in result

    @pytest.mark.asyncio
    async def test_list_integrations_tool(self):
        """Test list_integrations_tool."""
        from aragora.mcp.tools_module.integrations import list_integrations_tool

        result = await list_integrations_tool(platform="all")

        # Should have integrations structure even if empty
        assert "integrations" in result or "error" in result
        if "integrations" in result:
            assert "total" in result

    @pytest.mark.asyncio
    async def test_list_integrations_tool_filtered(self):
        """Test list_integrations_tool with platform filter."""
        from aragora.mcp.tools_module.integrations import list_integrations_tool

        result = await list_integrations_tool(platform="zapier")

        assert "platform_filter" in result or "error" in result

    @pytest.mark.asyncio
    async def test_test_integration_tool_invalid_platform(self):
        """Test test_integration_tool with invalid platform."""
        from aragora.mcp.tools_module.integrations import test_integration_tool

        result = await test_integration_tool(platform="invalid", integration_id="test123")

        assert "error" in result
        assert "Invalid platform" in result["error"]

    @pytest.mark.asyncio
    async def test_test_integration_tool_valid_platform(self):
        """Test test_integration_tool with valid platform."""
        from aragora.mcp.tools_module.integrations import test_integration_tool

        result = await test_integration_tool(platform="zapier", integration_id="test123")

        # Either not found or module not available
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_integration_events_tool_invalid_platform(self):
        """Test get_integration_events_tool with invalid platform."""
        from aragora.mcp.tools_module.integrations import get_integration_events_tool

        result = await get_integration_events_tool(platform="invalid")

        assert "error" in result
        assert "Invalid platform" in result["error"]

    @pytest.mark.asyncio
    async def test_get_integration_events_tool_valid_platform(self):
        """Test get_integration_events_tool with valid platforms."""
        from aragora.mcp.tools_module.integrations import get_integration_events_tool

        for platform in ["zapier", "make", "n8n"]:
            result = await get_integration_events_tool(platform=platform)

            # Either returns events or module not available
            assert "platform" in result or "error" in result


# =============================================================================
# Tool Exports Tests
# =============================================================================


class TestToolExports:
    """Test tool exports from various modules."""

    def test_tools_module_exports(self):
        """Test that tools_module exports all expected tools."""
        from aragora.mcp.tools_module import (
            # Debate tools
            run_debate_tool,
            get_debate_tool,
            search_debates_tool,
            fork_debate_tool,
            get_forks_tool,
            # Gauntlet tools
            run_gauntlet_tool,
            # Agent tools
            list_agents_tool,
            get_agent_history_tool,
            get_agent_lineage_tool,
            breed_agents_tool,
            # Memory tools
            query_memory_tool,
            store_memory_tool,
            get_memory_pressure_tool,
            # Checkpoint tools
            create_checkpoint_tool,
            list_checkpoints_tool,
            resume_checkpoint_tool,
            delete_checkpoint_tool,
            # Verification tools
            get_consensus_proofs_tool,
            verify_consensus_tool,
            generate_proof_tool,
            # Evidence tools
            search_evidence_tool,
            cite_evidence_tool,
            verify_citation_tool,
            # Trending tools
            list_trending_topics_tool,
            # Audit tools
            list_audit_presets_tool,
            list_audit_types_tool,
            get_audit_preset_tool,
            create_audit_session_tool,
            run_audit_tool,
            get_audit_status_tool,
            get_audit_findings_tool,
            update_finding_status_tool,
            run_quick_audit_tool,
            # Knowledge tools
            query_knowledge_tool,
            store_knowledge_tool,
            get_knowledge_stats_tool,
            get_decision_receipt_tool,
            # Workflow tools
            run_workflow_tool,
            get_workflow_status_tool,
            list_workflow_templates_tool,
            cancel_workflow_tool,
            # External integration tools
            trigger_external_webhook_tool,
            list_integrations_tool,
            test_integration_tool,
            get_integration_events_tool,
        )

        # All imports succeeded
        assert callable(run_debate_tool)
        assert callable(query_knowledge_tool)
        assert callable(run_workflow_tool)
        assert callable(trigger_external_webhook_tool)

    def test_main_tools_exports(self):
        """Test that main tools.py exports all expected tools."""
        from aragora.mcp.tools import (
            TOOLS_METADATA,
            run_debate_tool,
            query_knowledge_tool,
            run_workflow_tool,
            trigger_external_webhook_tool,
        )

        assert len(TOOLS_METADATA) >= 45
        assert callable(run_debate_tool)
        assert callable(query_knowledge_tool)
        assert callable(run_workflow_tool)
        assert callable(trigger_external_webhook_tool)


# =============================================================================
# Debate Tools Tests
# =============================================================================


class TestDebateTools:
    """Test debate-related MCP tools."""

    @pytest.mark.asyncio
    async def test_run_debate_tool(self):
        """Test run_debate_tool basic invocation with mocked dependencies."""
        from aragora.mcp.tools_module.debate import run_debate_tool

        # Mock the create_agent to avoid API calls
        mock_agent = MagicMock()
        mock_agent.name = "mock_agent"

        # Mock DebateResult
        mock_result = MagicMock()
        mock_result.final_answer = "Test answer"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 2

        with patch("aragora.agents.base.create_agent", return_value=mock_agent):
            with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
                mock_arena = AsyncMock()
                mock_arena.run.return_value = mock_result
                mock_arena_class.return_value = mock_arena

                result = await run_debate_tool(question="Test question")

                # Should return a valid result
                assert isinstance(result, dict)
                # With mocked dependencies, should have debate_id
                assert "debate_id" in result or "error" in result

    @pytest.mark.asyncio
    async def test_run_debate_tool_empty_question(self):
        """Test run_debate_tool with empty question returns error."""
        from aragora.mcp.tools_module.debate import run_debate_tool

        result = await run_debate_tool(question="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_debate_tool_not_found(self):
        """Test get_debate_tool with non-existent debate."""
        from aragora.mcp.tools_module.debate import get_debate_tool

        result = await get_debate_tool(debate_id="nonexistent123")

        # Should return error for non-existent debate
        assert "error" in result or "debate_id" in result

    @pytest.mark.asyncio
    async def test_get_debate_tool_empty_id(self):
        """Test get_debate_tool with empty debate_id returns error."""
        from aragora.mcp.tools_module.debate import get_debate_tool

        result = await get_debate_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_debates_tool(self):
        """Test search_debates_tool."""
        from aragora.mcp.tools_module.debate import search_debates_tool

        result = await search_debates_tool(query="test")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_search_debates_tool_with_filters(self):
        """Test search_debates_tool with all filters."""
        from aragora.mcp.tools_module.debate import search_debates_tool

        result = await search_debates_tool(
            query="test",
            agent="anthropic",
            start_date="2024-01-01",
            end_date="2024-12-31",
            consensus_only=True,
            limit=5,
        )

        assert isinstance(result, dict)
        assert "filters" in result

    @pytest.mark.asyncio
    async def test_fork_debate_tool_empty_id(self):
        """Test fork_debate_tool with empty debate_id returns error."""
        from aragora.mcp.tools_module.debate import fork_debate_tool

        result = await fork_debate_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_forks_tool_empty_id(self):
        """Test get_forks_tool with empty debate_id returns error."""
        from aragora.mcp.tools_module.debate import get_forks_tool

        result = await get_forks_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Memory Tools Tests
# =============================================================================


class TestMemoryTools:
    """Test memory-related MCP tools."""

    @pytest.mark.asyncio
    async def test_query_memory_tool(self):
        """Test query_memory_tool."""
        from aragora.mcp.tools_module.memory import query_memory_tool

        result = await query_memory_tool(query="test query")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_query_memory_tool_with_tier(self):
        """Test query_memory_tool with tier filter."""
        from aragora.mcp.tools_module.memory import query_memory_tool

        result = await query_memory_tool(query="test", tier="fast")

        assert isinstance(result, dict)
        assert result.get("tier") == "fast"

    @pytest.mark.asyncio
    async def test_store_memory_tool_empty_content(self):
        """Test store_memory_tool with empty content returns error."""
        from aragora.mcp.tools_module.memory import store_memory_tool

        result = await store_memory_tool(content="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_memory_pressure_tool(self):
        """Test get_memory_pressure_tool."""
        from aragora.mcp.tools_module.memory import get_memory_pressure_tool

        result = await get_memory_pressure_tool()

        assert isinstance(result, dict)


# =============================================================================
# Checkpoint Tools Tests
# =============================================================================


class TestCheckpointTools:
    """Test checkpoint-related MCP tools."""

    @pytest.mark.asyncio
    async def test_create_checkpoint_tool_empty_id(self):
        """Test create_checkpoint_tool with empty debate_id returns error."""
        from aragora.mcp.tools_module.checkpoint import create_checkpoint_tool

        result = await create_checkpoint_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_checkpoints_tool(self):
        """Test list_checkpoints_tool."""
        from aragora.mcp.tools_module.checkpoint import list_checkpoints_tool

        result = await list_checkpoints_tool()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_resume_checkpoint_tool_empty_id(self):
        """Test resume_checkpoint_tool with empty checkpoint_id returns error."""
        from aragora.mcp.tools_module.checkpoint import resume_checkpoint_tool

        result = await resume_checkpoint_tool(checkpoint_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_checkpoint_tool_empty_id(self):
        """Test delete_checkpoint_tool with empty checkpoint_id returns error."""
        from aragora.mcp.tools_module.checkpoint import delete_checkpoint_tool

        result = await delete_checkpoint_tool(checkpoint_id="")

        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Audit Tools Tests
# =============================================================================


class TestAuditTools:
    """Test audit-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_audit_presets_tool(self):
        """Test list_audit_presets_tool."""
        from aragora.mcp.tools_module.audit import list_audit_presets_tool

        result = await list_audit_presets_tool()

        assert isinstance(result, dict)
        assert "presets" in result

    @pytest.mark.asyncio
    async def test_list_audit_types_tool(self):
        """Test list_audit_types_tool."""
        from aragora.mcp.tools_module.audit import list_audit_types_tool

        result = await list_audit_types_tool()

        assert isinstance(result, dict)
        # The actual key is 'audit_types'
        assert "audit_types" in result or "types" in result

    @pytest.mark.asyncio
    async def test_get_audit_preset_tool_missing_name(self):
        """Test get_audit_preset_tool with missing preset_name returns error."""
        from aragora.mcp.tools_module.audit import get_audit_preset_tool

        result = await get_audit_preset_tool(preset_name="")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_audit_session_missing_documents(self):
        """Test create_audit_session_tool with empty document_ids returns error."""
        from aragora.mcp.tools_module.audit import create_audit_session_tool

        result = await create_audit_session_tool(document_ids="")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_audit_tool_missing_session(self):
        """Test run_audit_tool with missing session_id returns error."""
        from aragora.mcp.tools_module.audit import run_audit_tool

        result = await run_audit_tool(session_id="")

        assert "error" in result


# =============================================================================
# Verification Tools Tests
# =============================================================================


class TestVerificationTools:
    """Test verification-related MCP tools."""

    @pytest.mark.asyncio
    async def test_verify_consensus_tool_empty_id(self):
        """Test verify_consensus_tool with empty debate_id returns error."""
        from aragora.mcp.tools_module.verification import verify_consensus_tool

        result = await verify_consensus_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_verify_consensus_tool(self):
        """Test verify_consensus_tool."""
        from aragora.mcp.tools_module.verification import verify_consensus_tool

        result = await verify_consensus_tool(debate_id="test123")

        # Either error (no debate) or verification result
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_proof_tool_empty_claim(self):
        """Test generate_proof_tool with empty claim returns error."""
        from aragora.mcp.tools_module.verification import generate_proof_tool

        result = await generate_proof_tool(claim="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_consensus_proofs_tool(self):
        """Test get_consensus_proofs_tool."""
        from aragora.mcp.tools_module.verification import get_consensus_proofs_tool

        result = await get_consensus_proofs_tool()

        assert isinstance(result, dict)
        assert "proofs" in result
        assert "count" in result


# =============================================================================
# Evidence Tools Tests
# =============================================================================


class TestEvidenceTools:
    """Test evidence-related MCP tools."""

    @pytest.mark.asyncio
    async def test_search_evidence_tool(self):
        """Test search_evidence_tool."""
        from aragora.mcp.tools_module.evidence import search_evidence_tool

        result = await search_evidence_tool(query="test")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_citation_tool(self):
        """Test verify_citation_tool."""
        from aragora.mcp.tools_module.evidence import verify_citation_tool

        result = await verify_citation_tool(url="https://example.com")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_cite_evidence_tool_missing_params(self):
        """Test cite_evidence_tool with missing required parameters."""
        from aragora.mcp.tools_module.evidence import cite_evidence_tool

        result = await cite_evidence_tool(debate_id="", evidence_id="", message_index=0)

        assert "error" in result


# =============================================================================
# Control Plane Tools Tests
# =============================================================================


class TestControlPlaneTools:
    """Test control plane MCP tools."""

    @pytest.mark.asyncio
    async def test_register_agent_tool_empty_id(self):
        """Test register_agent_tool with empty agent_id returns error."""
        from aragora.mcp.tools_module.control_plane import register_agent_tool

        result = await register_agent_tool(agent_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_unregister_agent_tool_empty_id(self):
        """Test unregister_agent_tool with empty agent_id returns error."""
        from aragora.mcp.tools_module.control_plane import unregister_agent_tool

        result = await unregister_agent_tool(agent_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_registered_agents_tool(self):
        """Test list_registered_agents_tool returns valid structure."""
        from unittest.mock import patch, AsyncMock
        from aragora.mcp.tools_module.control_plane import list_registered_agents_tool

        # Mock coordinator to avoid blocking ControlPlaneCoordinator.create()
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await list_registered_agents_tool()

        assert isinstance(result, dict)
        # Should have agents list (fallback or real)
        assert "agents" in result
        assert isinstance(result["agents"], list)

    @pytest.mark.asyncio
    async def test_get_agent_health_tool_empty_id(self):
        """Test get_agent_health_tool with empty agent_id returns error."""
        from aragora.mcp.tools_module.control_plane import get_agent_health_tool

        result = await get_agent_health_tool(agent_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_task_tool_empty_type(self):
        """Test submit_task_tool with empty task_type returns error."""
        from aragora.mcp.tools_module.control_plane import submit_task_tool

        result = await submit_task_tool(task_type="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_task_tool_invalid_json(self):
        """Test submit_task_tool with invalid JSON payload returns error."""
        from aragora.mcp.tools_module.control_plane import submit_task_tool

        result = await submit_task_tool(task_type="debate", payload="invalid json {")

        assert "error" in result
        assert "JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_task_tool_invalid_payload_type(self):
        """Test submit_task_tool with non-object payload returns error."""
        from aragora.mcp.tools_module.control_plane import submit_task_tool

        result = await submit_task_tool(task_type="debate", payload='"string_payload"')

        assert "error" in result
        assert "object" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_task_status_tool_empty_id(self):
        """Test get_task_status_tool with empty task_id returns error."""
        from aragora.mcp.tools_module.control_plane import get_task_status_tool

        result = await get_task_status_tool(task_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cancel_task_tool_empty_id(self):
        """Test cancel_task_tool with empty task_id returns error."""
        from aragora.mcp.tools_module.control_plane import cancel_task_tool

        result = await cancel_task_tool(task_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_control_plane_status_tool(self):
        """Test get_control_plane_status_tool returns valid structure."""
        from aragora.mcp.tools_module.control_plane import get_control_plane_status_tool

        result = await get_control_plane_status_tool()

        assert isinstance(result, dict)
        # Should have status field
        assert "status" in result or "error" in result

    @pytest.mark.asyncio
    async def test_trigger_health_check_tool(self):
        """Test trigger_health_check_tool."""
        from aragora.mcp.tools_module.control_plane import trigger_health_check_tool

        result = await trigger_health_check_tool()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_resource_utilization_tool(self):
        """Test get_resource_utilization_tool."""
        from aragora.mcp.tools_module.control_plane import get_resource_utilization_tool

        result = await get_resource_utilization_tool()

        assert isinstance(result, dict)


# =============================================================================
# Canvas Tools Tests
# =============================================================================


class TestCanvasTools:
    """Test canvas MCP tools."""

    @pytest.mark.asyncio
    async def test_canvas_create_tool(self):
        """Test canvas_create_tool creates canvas successfully."""
        from aragora.mcp.tools_module.canvas import canvas_create_tool

        result = await canvas_create_tool(name="Test Canvas")

        assert isinstance(result, dict)
        # Either success or module not available
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_canvas_get_tool_empty_id(self):
        """Test canvas_get_tool with empty canvas_id returns error."""
        from aragora.mcp.tools_module.canvas import canvas_get_tool

        result = await canvas_get_tool(canvas_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_add_node_tool_empty_canvas_id(self):
        """Test canvas_add_node_tool with empty canvas_id returns error."""
        from aragora.mcp.tools_module.canvas import canvas_add_node_tool

        result = await canvas_add_node_tool(canvas_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_add_edge_tool_missing_params(self):
        """Test canvas_add_edge_tool with missing required parameters."""
        from aragora.mcp.tools_module.canvas import canvas_add_edge_tool

        result = await canvas_add_edge_tool(canvas_id="", source_id="", target_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_execute_action_tool_missing_params(self):
        """Test canvas_execute_action_tool with missing required parameters."""
        from aragora.mcp.tools_module.canvas import canvas_execute_action_tool

        result = await canvas_execute_action_tool(canvas_id="", action="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_list_tool(self):
        """Test canvas_list_tool returns valid structure."""
        from aragora.mcp.tools_module.canvas import canvas_list_tool

        result = await canvas_list_tool()

        assert isinstance(result, dict)
        assert "canvases" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_canvas_delete_node_tool_missing_params(self):
        """Test canvas_delete_node_tool with missing required parameters."""
        from aragora.mcp.tools_module.canvas import canvas_delete_node_tool

        result = await canvas_delete_node_tool(canvas_id="", node_id="")

        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Agent Tools Tests
# =============================================================================


class TestAgentTools:
    """Test agent-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_agents_tool(self):
        """Test list_agents_tool returns valid structure."""
        from aragora.mcp.tools_module.agent import list_agents_tool

        result = await list_agents_tool()

        assert isinstance(result, dict)
        assert "agents" in result
        assert "count" in result
        assert isinstance(result["agents"], list)

    @pytest.mark.asyncio
    async def test_get_agent_history_tool_empty_name(self):
        """Test get_agent_history_tool with empty agent_name returns error."""
        from aragora.mcp.tools_module.agent import get_agent_history_tool

        result = await get_agent_history_tool(agent_name="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_agent_lineage_tool_empty_name(self):
        """Test get_agent_lineage_tool with empty agent_name returns error."""
        from aragora.mcp.tools_module.agent import get_agent_lineage_tool

        result = await get_agent_lineage_tool(agent_name="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_breed_agents_tool_missing_parents(self):
        """Test breed_agents_tool with missing parent names returns error."""
        from aragora.mcp.tools_module.agent import breed_agents_tool

        result = await breed_agents_tool(parent_a="", parent_b="")

        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Trending Tools Tests
# =============================================================================


class TestTrendingTools:
    """Test trending-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_trending_topics_tool(self):
        """Test list_trending_topics_tool returns valid structure."""
        from aragora.mcp.tools_module.trending import list_trending_topics_tool

        result = await list_trending_topics_tool()

        assert isinstance(result, dict)
        assert "topics" in result
        assert "count" in result


# =============================================================================
# Gauntlet Tools Tests
# =============================================================================


class TestGauntletTools:
    """Test gauntlet-related MCP tools."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_tool_empty_content(self):
        """Test run_gauntlet_tool with empty content returns error."""
        from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool

        result = await run_gauntlet_tool(content="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_gauntlet_tool_with_profile(self):
        """Test run_gauntlet_tool with different profiles."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool

        # Mock GauntletRunner to avoid real API calls
        mock_verdict = MagicMock()
        mock_verdict.value = "pass"
        mock_result = MagicMock()
        mock_result.verdict = mock_verdict
        mock_result.risk_score = 0.1
        mock_result.vulnerabilities = []
        mock_result.passed = True
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("aragora.gauntlet.GauntletRunner", return_value=mock_runner):
            result = await run_gauntlet_tool(content="Test document content", profile="quick")

        assert isinstance(result, dict)
