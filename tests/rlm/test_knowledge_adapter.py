"""Unit tests for the RLM Knowledge Mound adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.rlm.knowledge_adapter import KnowledgeMoundAdapter
from aragora.rlm.types import AbstractionLevel, RLMContext


@dataclass
class FakeKnowledgeNode:
    """Small test double for Knowledge Mound nodes."""

    id: str
    content: str
    node_type: str = "fact"
    confidence: float = 1.0
    relationships: dict[str, list[str]] = field(default_factory=dict)


def test_adapter_keeps_mound_reference() -> None:
    """Adapter initialization should retain the supplied Knowledge Mound object."""
    mound = MagicMock()

    adapter = KnowledgeMoundAdapter(mound)

    assert adapter.mound is mound


@pytest.mark.asyncio
async def test_to_rlm_context_with_query_uses_semantic_search() -> None:
    """Queries should be translated into semantic mound searches."""
    nodes = [
        FakeKnowledgeNode("node-1", "Contract requires a 90 day notice period.", "fact"),
    ]
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=nodes)

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query="contract notice",
        max_nodes=5,
    )

    mound.query_semantic.assert_awaited_once_with(
        text="contract notice",
        limit=5,
        workspace_id="workspace-1",
    )
    assert isinstance(context, RLMContext)
    assert context.source_type == "knowledge"
    assert "[node-1] Contract requires" in context.original_content


@pytest.mark.asyncio
async def test_to_rlm_context_without_query_uses_recent_nodes() -> None:
    """Missing query text should fall back to recent Knowledge Mound nodes."""
    nodes = [FakeKnowledgeNode("recent-1", "Recently captured decision.", "memory")]
    mound = MagicMock()
    mound.get_recent_nodes = AsyncMock(return_value=nodes)

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query=None,
        max_nodes=25,
    )

    mound.get_recent_nodes.assert_awaited_once_with(
        workspace_id="workspace-1",
        limit=25,
    )
    assert "[recent-1] Recently captured decision." in context.original_content


@pytest.mark.asyncio
async def test_to_rlm_context_handles_empty_results() -> None:
    """Empty mound results should still return a valid empty RLM context."""
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=[])

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query="no matches",
        max_nodes=10,
    )

    assert context.original_content == ""
    assert context.original_tokens == 0
    assert context.levels[AbstractionLevel.SUMMARY] == []
    assert context.nodes_by_id == {}


@pytest.mark.asyncio
async def test_to_rlm_context_groups_summary_nodes_by_type() -> None:
    """Summary abstraction nodes should group mound nodes by node type."""
    nodes = [
        FakeKnowledgeNode("fact-1", "A fact", "fact"),
        FakeKnowledgeNode("fact-2", "Another fact", "fact"),
        FakeKnowledgeNode("evidence-1", "Evidence", "evidence"),
    ]
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=nodes)

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query="mixed",
        max_nodes=10,
    )

    summary_by_id = {node.id: node for node in context.levels[AbstractionLevel.SUMMARY]}
    assert set(summary_by_id) == {"type_fact", "type_evidence"}
    assert summary_by_id["type_fact"].child_ids == ["fact-1", "fact-2"]
    assert summary_by_id["type_evidence"].child_ids == ["evidence-1"]
    assert context.nodes_by_id["type_fact"] is summary_by_id["type_fact"]


@pytest.mark.asyncio
async def test_to_rlm_context_uses_unknown_type_for_missing_node_type() -> None:
    """Nodes without node_type metadata should be grouped under unknown."""
    node = SimpleNamespace(id="unknown-1", content="Metadata-free node")
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=[node])

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query="metadata",
        max_nodes=10,
    )

    summary_node = context.nodes_by_id["type_unknown"]
    assert summary_node.child_ids == ["unknown-1"]
    assert "UNKNOWN" in summary_node.content


@pytest.mark.asyncio
async def test_to_rlm_context_truncates_long_summary_items() -> None:
    """Summary content should truncate long node bodies for compact RLM context."""
    long_content = "x" * 150
    mound = MagicMock()
    mound.query_semantic = AsyncMock(
        return_value=[FakeKnowledgeNode("long-1", long_content, "claim")]
    )

    context = await KnowledgeMoundAdapter(mound).to_rlm_context(
        workspace_id="workspace-1",
        query="long",
        max_nodes=10,
    )

    summary = context.nodes_by_id["type_claim"].content
    assert f"- {'x' * 100}..." in summary
    assert long_content not in summary


def test_get_repl_helpers_exposes_expected_callables() -> None:
    """REPL helper map should expose stable search and lookup helpers."""
    helpers = KnowledgeMoundAdapter(MagicMock()).get_repl_helpers()

    assert set(helpers) == {"search_mound", "get_mound_node"}
    assert callable(helpers["search_mound"])
    assert callable(helpers["get_mound_node"])


@pytest.mark.asyncio
async def test_search_mound_helper_formats_and_truncates_results() -> None:
    """search_mound should return compact dictionaries for REPL callers."""
    content = "A" * 250
    node = FakeKnowledgeNode("search-1", content, "decision", confidence=0.74)
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=[node])
    helpers = KnowledgeMoundAdapter(mound).get_repl_helpers()

    results = await helpers["search_mound"]("release risk", limit=3)

    mound.query_semantic.assert_awaited_once_with("release risk", limit=3)
    assert results == [
        {
            "id": "search-1",
            "type": "decision",
            "content": "A" * 200,
            "confidence": 0.74,
        }
    ]


@pytest.mark.asyncio
async def test_search_mound_helper_defaults_missing_metadata() -> None:
    """search_mound should provide conservative defaults for sparse nodes."""
    node = SimpleNamespace(id="search-2", content="Sparse node")
    mound = MagicMock()
    mound.query_semantic = AsyncMock(return_value=[node])
    helpers = KnowledgeMoundAdapter(mound).get_repl_helpers()

    results = await helpers["search_mound"]("sparse")

    assert results == [
        {
            "id": "search-2",
            "type": "unknown",
            "content": "Sparse node",
            "confidence": 1.0,
        }
    ]


@pytest.mark.asyncio
async def test_get_mound_node_helper_formats_node_details() -> None:
    """get_mound_node should return full node details including relationships."""
    node = FakeKnowledgeNode(
        "node-2",
        "Full content",
        "memory",
        confidence=0.91,
        relationships={"supports": ["node-1"]},
    )
    mound = MagicMock()
    mound.get_node = AsyncMock(return_value=node)
    helpers = KnowledgeMoundAdapter(mound).get_repl_helpers()

    result = await helpers["get_mound_node"]("node-2")

    mound.get_node.assert_awaited_once_with("node-2")
    assert result == {
        "id": "node-2",
        "type": "memory",
        "content": "Full content",
        "confidence": 0.91,
        "relationships": {"supports": ["node-1"]},
    }


@pytest.mark.asyncio
async def test_get_mound_node_helper_returns_none_for_missing_node() -> None:
    """get_mound_node should preserve a None lookup result."""
    mound = MagicMock()
    mound.get_node = AsyncMock(return_value=None)
    helpers = KnowledgeMoundAdapter(mound).get_repl_helpers()

    result = await helpers["get_mound_node"]("missing")

    mound.get_node.assert_awaited_once_with("missing")
    assert result is None
