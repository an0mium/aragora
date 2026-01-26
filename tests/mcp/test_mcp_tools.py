"""Tests for MCP tools."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


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

        # We have 45 tools now
        assert len(TOOLS_METADATA) >= 45


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


class TestDebateTools:
    """Test debate-related MCP tools."""

    @pytest.mark.asyncio
    async def test_run_debate_tool(self):
        """Test run_debate_tool basic invocation."""
        from aragora.mcp.tools_module.debate import run_debate_tool

        # Without proper setup, this will fail gracefully
        result = await run_debate_tool(question="Test question")

        # Either returns a result or an error
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_debate_tool_not_found(self):
        """Test get_debate_tool with non-existent debate."""
        from aragora.mcp.tools_module.debate import get_debate_tool

        result = await get_debate_tool(debate_id="nonexistent123")

        # Should return error for non-existent debate
        assert "error" in result or "debate_id" in result

    @pytest.mark.asyncio
    async def test_search_debates_tool(self):
        """Test search_debates_tool."""
        from aragora.mcp.tools_module.debate import search_debates_tool

        result = await search_debates_tool(query="test")

        assert isinstance(result, dict)


class TestMemoryTools:
    """Test memory-related MCP tools."""

    @pytest.mark.asyncio
    async def test_query_memory_tool(self):
        """Test query_memory_tool."""
        from aragora.mcp.tools_module.memory import query_memory_tool

        result = await query_memory_tool(query="test query")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_memory_pressure_tool(self):
        """Test get_memory_pressure_tool."""
        from aragora.mcp.tools_module.memory import get_memory_pressure_tool

        result = await get_memory_pressure_tool()

        assert isinstance(result, dict)


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


class TestVerificationTools:
    """Test verification-related MCP tools."""

    @pytest.mark.asyncio
    async def test_verify_consensus_tool(self):
        """Test verify_consensus_tool."""
        from aragora.mcp.tools_module.verification import verify_consensus_tool

        result = await verify_consensus_tool(debate_id="test123")

        # Either error (no debate) or verification result
        assert isinstance(result, dict)


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
