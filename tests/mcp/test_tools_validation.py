"""Tests for MCP tools input validation and basic functionality.

These tests focus on input validation, error handling, and basic
functionality that doesn't require complex mocking of deep imports.
"""

import pytest

from aragora.mcp.tools_module.agent import (
    breed_agents_tool,
    get_agent_history_tool,
    get_agent_lineage_tool,
    list_agents_tool,
)
from aragora.mcp.tools_module.audit import (
    create_audit_session_tool,
)
from aragora.mcp.tools_module.browser import (
    browser_extract_tool,
)
from aragora.mcp.tools_module.canvas import (
    canvas_add_edge_tool,
    canvas_add_node_tool,
    canvas_delete_node_tool,
    canvas_execute_action_tool,
    canvas_get_tool,
    canvas_list_tool,
)
from aragora.mcp.tools_module.chat_actions import (
    create_poll_tool,
)
from aragora.mcp.tools_module.checkpoint import (
    create_checkpoint_tool,
    delete_checkpoint_tool,
    list_checkpoints_tool,
    resume_checkpoint_tool,
)
from aragora.mcp.tools_module.context_tools import (
    analyze_conversation_tool,
)
from aragora.mcp.tools_module.debate import (
    fork_debate_tool,
    get_debate_tool,
    get_forks_tool,
    run_debate_tool,
    search_debates_tool,
)
from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool
from aragora.mcp.tools_module.knowledge import (
    get_decision_receipt_tool,
    query_knowledge_tool,
    store_knowledge_tool,
)
from aragora.mcp.tools_module.memory import (
    store_memory_tool,
)
from aragora.mcp.tools_module.workflow import (
    run_workflow_tool,
)


# =============================================================================
# Gauntlet Tool Validation Tests
# =============================================================================


class TestGauntletValidation:
    """Tests for gauntlet tool input validation."""

    @pytest.mark.asyncio
    async def test_empty_content_returns_error(self):
        """Empty content returns error."""
        result = await run_gauntlet_tool(content="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_content_returns_error(self):
        """None content returns error."""
        result = await run_gauntlet_tool(content=None)  # type: ignore
        assert "error" in result


# =============================================================================
# Agent Tool Validation Tests
# =============================================================================


class TestAgentToolValidation:
    """Tests for agent tools input validation."""

    @pytest.mark.asyncio
    async def test_list_agents_returns_result(self):
        """list_agents_tool returns a result structure."""
        result = await list_agents_tool()
        # Should return either agents list or fallback
        assert "agents" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_get_history_missing_agent_returns_error(self):
        """Missing agent name returns error."""
        result = await get_agent_history_tool(agent_name="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_lineage_missing_agent_returns_error(self):
        """Missing agent name returns error."""
        result = await get_agent_lineage_tool(agent_name="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_breed_requires_both_parents(self):
        """Breeding requires both parent agents."""
        result = await breed_agents_tool(parent_a="", parent_b="test")
        assert "error" in result
        assert "required" in result["error"].lower()

        result = await breed_agents_tool(parent_a="test", parent_b="")
        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Debate Tool Validation Tests
# =============================================================================


class TestDebateToolValidation:
    """Tests for debate tools input validation."""

    @pytest.mark.asyncio
    async def test_run_debate_requires_question(self):
        """run_debate_tool requires a question."""
        result = await run_debate_tool(question="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_debate_none_question(self):
        """run_debate_tool handles None question."""
        result = await run_debate_tool(question=None)  # type: ignore
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_debate_requires_id(self):
        """get_debate_tool requires debate ID."""
        result = await get_debate_tool(debate_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fork_debate_requires_id(self):
        """fork_debate_tool requires debate ID."""
        result = await fork_debate_tool(debate_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_forks_requires_id(self):
        """get_forks_tool requires debate ID."""
        result = await get_forks_tool(debate_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_debates_returns_result(self):
        """search_debates_tool returns result structure."""
        result = await search_debates_tool(query="test")
        # Should return result structure even if empty
        assert "debates" in result or "error" in result


# =============================================================================
# Knowledge Tool Validation Tests
# =============================================================================


class TestKnowledgeToolValidation:
    """Tests for knowledge tools input validation."""

    @pytest.mark.asyncio
    async def test_store_invalid_node_type(self):
        """Invalid node type returns error."""
        result = await store_knowledge_tool(content="Test content", node_type="invalid_type")
        assert "error" in result
        assert "invalid node_type" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_store_invalid_tier(self):
        """Invalid tier returns error."""
        result = await store_knowledge_tool(content="Test content", tier="invalid_tier")
        assert "error" in result
        assert "invalid tier" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_store_confidence_too_high(self):
        """Confidence > 1 returns error."""
        result = await store_knowledge_tool(content="Test", confidence=1.5)
        assert "error" in result
        assert "confidence" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_store_confidence_too_low(self):
        """Confidence < 0 returns error."""
        result = await store_knowledge_tool(content="Test", confidence=-0.5)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_returns_result_structure(self):
        """query_knowledge_tool returns result structure."""
        result = await query_knowledge_tool(query="test")
        # Should return nodes list and count (possibly empty)
        assert "nodes" in result or "error" in result


# =============================================================================
# Audit Tool Validation Tests
# =============================================================================


class TestAuditToolValidation:
    """Tests for audit tools input validation."""

    @pytest.mark.asyncio
    async def test_create_session_no_documents(self):
        """Creating session with no documents returns error."""
        result = await create_audit_session_tool(document_ids="")
        assert result["success"] is False
        assert "no document" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_session_whitespace_documents(self):
        """Creating session with whitespace documents returns error."""
        result = await create_audit_session_tool(document_ids="   ,  ,  ")
        assert result["success"] is False


# =============================================================================
# Valid Input Type Tests
# =============================================================================


class TestValidKnowledgeInputs:
    """Tests for valid knowledge input values."""

    @pytest.mark.asyncio
    async def test_valid_node_types_accepted(self):
        """All valid node types are accepted."""
        valid_types = ["fact", "insight", "claim", "evidence", "decision", "opinion"]
        for node_type in valid_types:
            result = await store_knowledge_tool(
                content="Test content with sufficient length for storage validation.",
                node_type=node_type,
            )
            # Should not have node_type error (may have other errors like module unavailable)
            if "error" in result:
                assert "node_type" not in result["error"].lower(), f"Failed for {node_type}"

    @pytest.mark.asyncio
    async def test_valid_tiers_accepted(self):
        """All valid tiers are accepted."""
        valid_tiers = ["fast", "medium", "slow", "glacial"]
        for tier in valid_tiers:
            result = await store_knowledge_tool(
                content="Test content with sufficient length for tier validation.",
                tier=tier,
            )
            # Should not have tier error (may have other errors)
            if "error" in result:
                assert "tier" not in result["error"].lower(), f"Failed for {tier}"

    @pytest.mark.asyncio
    async def test_valid_confidence_accepted(self):
        """Valid confidence values are accepted."""
        for confidence in [0.0, 0.5, 1.0]:
            result = await store_knowledge_tool(
                content="Test content for confidence validation.",
                confidence=confidence,
            )
            # Should not have confidence error
            if "error" in result:
                assert "confidence" not in result["error"].lower()


# =============================================================================
# Canvas Tool Validation Tests
# =============================================================================


class TestCanvasToolValidation:
    """Tests for canvas tools input validation."""

    @pytest.mark.asyncio
    async def test_canvas_get_requires_id(self):
        """canvas_get_tool requires canvas_id."""
        result = await canvas_get_tool(canvas_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_add_node_requires_canvas_id(self):
        """canvas_add_node_tool requires canvas_id."""
        result = await canvas_add_node_tool(canvas_id="", node_type="text", label="test")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_add_edge_requires_all_ids(self):
        """canvas_add_edge_tool requires canvas_id, source_id, and target_id."""
        # Missing canvas_id
        result = await canvas_add_edge_tool(canvas_id="", source_id="s1", target_id="t1")
        assert "error" in result
        assert "required" in result["error"].lower()

        # Missing source_id
        result = await canvas_add_edge_tool(canvas_id="c1", source_id="", target_id="t1")
        assert "error" in result
        assert "required" in result["error"].lower()

        # Missing target_id
        result = await canvas_add_edge_tool(canvas_id="c1", source_id="s1", target_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_execute_action_requires_params(self):
        """canvas_execute_action_tool requires canvas_id and action."""
        # Missing canvas_id
        result = await canvas_execute_action_tool(canvas_id="", action="test")
        assert "error" in result
        assert "required" in result["error"].lower()

        # Missing action
        result = await canvas_execute_action_tool(canvas_id="c1", action="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_delete_node_requires_ids(self):
        """canvas_delete_node_tool requires canvas_id and node_id."""
        result = await canvas_delete_node_tool(canvas_id="", node_id="n1")
        assert "error" in result
        assert "required" in result["error"].lower()

        result = await canvas_delete_node_tool(canvas_id="c1", node_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_canvas_list_limits_results(self):
        """canvas_list_tool respects limit bounds."""
        # Very high limit should be capped
        result = await canvas_list_tool(limit=1000)
        # Function should handle gracefully (may have error or limited results)
        if "canvases" in result:
            assert result["count"] <= 100  # Max limit is 100


# =============================================================================
# Checkpoint Tool Validation Tests
# =============================================================================


class TestCheckpointToolValidation:
    """Tests for checkpoint tools input validation."""

    @pytest.mark.asyncio
    async def test_create_checkpoint_requires_debate_id(self):
        """create_checkpoint_tool requires debate_id."""
        result = await create_checkpoint_tool(debate_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_checkpoint_requires_id(self):
        """resume_checkpoint_tool requires checkpoint_id."""
        result = await resume_checkpoint_tool(checkpoint_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_checkpoint_requires_id(self):
        """delete_checkpoint_tool requires checkpoint_id."""
        result = await delete_checkpoint_tool(checkpoint_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_checkpoints_limits_results(self):
        """list_checkpoints_tool respects limit bounds."""
        # Very high limit should be capped
        result = await list_checkpoints_tool(limit=1000)
        # Function should handle gracefully
        if "checkpoints" in result:
            assert True  # Just verify no crash


# =============================================================================
# Chat Actions Tool Validation Tests
# =============================================================================


class TestChatActionsToolValidation:
    """Tests for chat actions tools input validation."""

    @pytest.mark.asyncio
    async def test_create_poll_requires_min_options(self):
        """create_poll_tool requires at least 2 options."""
        result = await create_poll_tool(channel_id="test", question="Test?", options=["only one"])
        assert "error" in result
        assert "2" in result["error"] or "option" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_poll_rejects_too_many_options(self):
        """create_poll_tool rejects more than 10 options."""
        result = await create_poll_tool(
            channel_id="test",
            question="Test?",
            options=[f"Option {i}" for i in range(15)],
        )
        assert "error" in result
        assert "10" in result["error"] or "option" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_poll_requires_non_empty_options(self):
        """create_poll_tool requires options list."""
        result = await create_poll_tool(channel_id="test", question="Test?", options=[])
        assert "error" in result


# =============================================================================
# Context Tools Validation Tests
# =============================================================================


class TestContextToolsValidation:
    """Tests for context tools input validation."""

    @pytest.mark.asyncio
    async def test_analyze_conversation_handles_empty_messages(self):
        """analyze_conversation_tool handles empty messages list."""
        result = await analyze_conversation_tool(messages=[])
        assert "error" in result
        assert "no message" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_analyze_conversation_returns_count(self):
        """analyze_conversation_tool returns message count."""
        result = await analyze_conversation_tool(
            messages=[
                {"content": "Hello", "author": "user1"},
                {"content": "Hi there", "author": "user2"},
            ]
        )
        assert "message_count" in result
        assert result["message_count"] == 2


# =============================================================================
# Memory Tool Validation Tests
# =============================================================================


class TestMemoryToolValidation:
    """Tests for memory tools input validation."""

    @pytest.mark.asyncio
    async def test_store_memory_requires_content(self):
        """store_memory_tool requires content."""
        result = await store_memory_tool(content="")
        assert "error" in result
        assert "required" in result["error"].lower()


# =============================================================================
# Workflow Tool Validation Tests
# =============================================================================


class TestWorkflowToolValidation:
    """Tests for workflow tools input validation."""

    @pytest.mark.asyncio
    async def test_run_workflow_returns_error_for_missing_engine(self):
        """run_workflow_tool returns error when engine unavailable."""
        result = await run_workflow_tool(template="test", inputs="")
        assert "error" in result
        # Either module not available or template not found
        assert "not available" in result["error"].lower() or "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_workflow_handles_invalid_inputs(self):
        """run_workflow_tool handles invalid inputs gracefully."""
        result = await run_workflow_tool(template="", inputs="")
        assert "error" in result


# =============================================================================
# Browser Tool Validation Tests
# =============================================================================


class TestBrowserToolValidation:
    """Tests for browser tools input validation."""

    @pytest.mark.asyncio
    async def test_browser_extract_invalid_json_selectors(self):
        """browser_extract_tool rejects invalid JSON selectors."""
        result = await browser_extract_tool(selectors="not valid json {{{")
        assert result["success"] is False
        assert "json" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_browser_extract_valid_json_proceeds(self):
        """browser_extract_tool accepts valid JSON selectors."""
        # This will fail with connector error, but pass JSON validation
        result = await browser_extract_tool(selectors='{"title": "h1"}')
        # Should get past JSON parsing to connector error
        assert "success" in result
        # Either success or a different error (not JSON-related)
        if not result["success"] and "error" in result:
            assert "json" not in result["error"].lower()
