"""Tests for KnowledgeMoundAdapter integration with RLM.

Tests the adapter that bridges Knowledge Mound with RLM context.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rlm.types import RLMContext, AbstractionLevel


class TestKnowledgeMoundAdapterInit:
    """Test KnowledgeMoundAdapter initialization."""

    def test_adapter_init(self):
        """Test adapter initialization with mound."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        mock_mound = MagicMock()
        adapter = KnowledgeMoundAdapter(mock_mound)

        assert adapter.mound is mock_mound


class TestKnowledgeMoundAdapterToRLMContext:
    """Test to_rlm_context method."""

    @pytest.mark.asyncio
    async def test_to_rlm_context_with_query(self):
        """Test building RLM context from query results."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        # Create mock mound with query_semantic
        mock_mound = MagicMock()
        mock_node = MagicMock()
        mock_node.id = "node-1"
        mock_node.content = "Contract requirements include 90-day notice period"
        mock_node.node_type = "fact"

        mock_mound.query_semantic = AsyncMock(return_value=[mock_node])

        adapter = KnowledgeMoundAdapter(mock_mound)

        context = await adapter.to_rlm_context(
            workspace_id="ws_test",
            query="contract notice requirements",
            max_nodes=10,
        )

        # Verify query_semantic was called correctly
        mock_mound.query_semantic.assert_called_once_with(
            text="contract notice requirements",
            limit=10,
            workspace_id="ws_test",
        )

        # Verify context structure
        assert isinstance(context, RLMContext)
        assert "node-1" in context.original_content
        assert context.source_type == "knowledge"

    @pytest.mark.asyncio
    async def test_to_rlm_context_without_query(self):
        """Test building RLM context from recent nodes."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        # Create mock mound with get_recent_nodes
        mock_mound = MagicMock()
        mock_node = MagicMock()
        mock_node.id = "recent-1"
        mock_node.content = "Recent knowledge item"
        mock_node.node_type = "memory"

        mock_mound.get_recent_nodes = AsyncMock(return_value=[mock_node])

        adapter = KnowledgeMoundAdapter(mock_mound)

        context = await adapter.to_rlm_context(
            workspace_id="ws_test",
            query=None,  # No query - get recent nodes
            max_nodes=50,
        )

        # Verify get_recent_nodes was called
        mock_mound.get_recent_nodes.assert_called_once_with(
            workspace_id="ws_test",
            limit=50,
        )

        assert "recent-1" in context.original_content

    @pytest.mark.asyncio
    async def test_to_rlm_context_groups_by_type(self):
        """Test that nodes are grouped by type in context."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        mock_mound = MagicMock()

        # Create nodes of different types
        fact_node = MagicMock()
        fact_node.id = "fact-1"
        fact_node.content = "A fact"
        fact_node.node_type = "fact"

        consensus_node = MagicMock()
        consensus_node.id = "cons-1"
        consensus_node.content = "A consensus"
        consensus_node.node_type = "consensus"

        mock_mound.query_semantic = AsyncMock(
            return_value=[fact_node, consensus_node]
        )

        adapter = KnowledgeMoundAdapter(mock_mound)

        context = await adapter.to_rlm_context(
            workspace_id="ws_test",
            query="test",
            max_nodes=10,
        )

        # Context should have levels with grouped nodes
        assert len(context.levels) > 0
        assert AbstractionLevel.SUMMARY in context.levels

    @pytest.mark.asyncio
    async def test_to_rlm_context_empty_results(self):
        """Test handling empty query results."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(return_value=[])

        adapter = KnowledgeMoundAdapter(mock_mound)

        context = await adapter.to_rlm_context(
            workspace_id="ws_test",
            query="nothing matches",
            max_nodes=10,
        )

        # Should still return a valid context
        assert isinstance(context, RLMContext)
        assert context.original_content == ""
        assert context.original_tokens == 0


class TestKnowledgeMoundFacadeRLMIntegration:
    """Test RLM integration in Knowledge Mound facade."""

    @pytest.mark.asyncio
    async def test_is_rlm_available(self):
        """Test that RLM availability check works."""
        from aragora.knowledge.mound.facade import KnowledgeMound

        mound = KnowledgeMound()

        # Should return True if RLM module is available
        result = mound.is_rlm_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_recent_nodes_method_exists(self):
        """Test that get_recent_nodes method exists on facade."""
        from aragora.knowledge.mound.facade import KnowledgeMound

        mound = KnowledgeMound()

        assert hasattr(mound, "get_recent_nodes")
        assert callable(mound.get_recent_nodes)

    @pytest.mark.asyncio
    async def test_query_with_rlm_method_exists(self):
        """Test that query_with_rlm method exists on facade."""
        from aragora.knowledge.mound.facade import KnowledgeMound

        mound = KnowledgeMound()

        assert hasattr(mound, "query_with_rlm")
        assert callable(mound.query_with_rlm)


class TestKnowledgeMoundAdapterAbstractionLevels:
    """Test abstraction level handling in adapter."""

    @pytest.mark.asyncio
    async def test_summary_level_includes_all_types(self):
        """Test that SUMMARY level includes all node types."""
        from aragora.rlm.bridge import KnowledgeMoundAdapter

        mock_mound = MagicMock()

        # Create multiple node types
        nodes = []
        for i, node_type in enumerate(["fact", "consensus", "memory", "evidence"]):
            node = MagicMock()
            node.id = f"node-{i}"
            node.content = f"Content for {node_type}"
            node.node_type = node_type
            nodes.append(node)

        mock_mound.query_semantic = AsyncMock(return_value=nodes)

        adapter = KnowledgeMoundAdapter(mock_mound)

        context = await adapter.to_rlm_context(
            workspace_id="ws_test",
            query="test",
            max_nodes=100,
        )

        # SUMMARY level should have abstraction nodes for each type
        summary_nodes = context.levels.get(AbstractionLevel.SUMMARY, [])
        node_ids = [n.id for n in summary_nodes]

        assert "type_fact" in node_ids
        assert "type_consensus" in node_ids
        assert "type_memory" in node_ids
        assert "type_evidence" in node_ids
