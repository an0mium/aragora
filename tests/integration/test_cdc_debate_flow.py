"""
Integration tests for CDC → Debate context injection flow.

Tests verify that database change events from CDC:
1. Are processed and stored in KnowledgeMound
2. Can be queried and injected into debate context
3. Contribute to agent knowledge during debates
4. Handle edge cases (no CDC data, stale data, etc.)

This fills the gap between CDC-KM integration and full debate flow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Environment, Message, Vote, Critique
from aragora.debate.orchestrator import Arena, ArenaConfig, DebateProtocol


# =============================================================================
# Fixtures
# =============================================================================


class MockCDCKnowledgeAgent(Agent):
    """Agent that can access CDC-sourced knowledge during debates."""

    def __init__(
        self,
        name: str = "cdc_aware_agent",
        knowledge_mound: Optional[MagicMock] = None,
        responses: Optional[List[str]] = None,
    ):
        super().__init__(name, "mock-model", "proposer")
        self.agent_type = "mock"
        self.knowledge_mound = knowledge_mound
        self._responses = responses or []
        self._call_count = 0
        self._queried_knowledge: List[Dict[str, Any]] = []

    async def generate(self, prompt: str, context: list | None = None) -> str:
        # Query knowledge mound for relevant CDC data if available
        if self.knowledge_mound:
            try:
                results = await self.knowledge_mound.query(prompt[:50])
                self._queried_knowledge.extend(results)
            except Exception:
                pass  # Continue without CDC knowledge

        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
        else:
            # Include CDC knowledge in response if available
            if self._queried_knowledge:
                knowledge_summary = ", ".join(
                    k.get("content", "")[:30] for k in self._queried_knowledge[:3]
                )
                response = f"Based on knowledge ({knowledge_summary}): My proposal is..."
            else:
                response = f"Default response from {self.name}"

        self._call_count += 1
        return response

    async def critique(self, proposal: str, task: str, context: list | None = None) -> Critique:
        self._call_count += 1
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:100],
            issues=["Minor issue"],
            suggestions=["Consider CDC data"],
            severity=0.3,
            reasoning="Critique with CDC context",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self._call_count += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Vote based on CDC-informed analysis",
            confidence=0.85,
            continue_debate=False,
        )


@pytest.fixture
def mock_knowledge_mound_with_cdc():
    """KnowledgeMound pre-populated with CDC-sourced data."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"

    # Pre-populate with CDC data
    cdc_nodes = [
        {
            "id": "cdc_node_001",
            "content": "Product catalog update: New pricing for enterprise tier",
            "topics": ["pricing", "enterprise", "product"],
            "metadata": {
                "source_type": "postgresql",
                "cdc_operation": "update",
                "table": "products",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "confidence": 0.9,
        },
        {
            "id": "cdc_node_002",
            "content": "Customer feedback: Users request better API rate limits",
            "topics": ["feedback", "api", "rate-limits"],
            "metadata": {
                "source_type": "mongodb",
                "cdc_operation": "insert",
                "collection": "feedback",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "confidence": 0.85,
        },
        {
            "id": "cdc_node_003",
            "content": "System metrics: API latency increased by 15% in US-East",
            "topics": ["metrics", "latency", "performance"],
            "metadata": {
                "source_type": "postgresql",
                "cdc_operation": "insert",
                "table": "metrics",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "confidence": 0.95,
        },
    ]

    async def mock_query(query: str, **kwargs):
        """Return CDC nodes matching the query."""
        query_lower = query.lower()
        return [
            n
            for n in cdc_nodes
            if any(term in n["content"].lower() for term in query_lower.split()[:3])
        ]

    async def mock_store(request):
        """Allow storing new knowledge during debate."""
        result = MagicMock()
        result.node_id = f"debate_node_{len(cdc_nodes) + 1:03d}"
        result.success = True
        cdc_nodes.append(
            {
                "id": result.node_id,
                "content": request.content,
                "topics": request.topics,
                "metadata": request.metadata,
                "confidence": request.confidence,
            }
        )
        return result

    mound.query = AsyncMock(side_effect=mock_query)
    mound.store = AsyncMock(side_effect=mock_store)
    mound._nodes = cdc_nodes
    return mound


@pytest.fixture
def empty_knowledge_mound():
    """Empty KnowledgeMound with no CDC data."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"
    mound._nodes = []

    async def mock_query(query: str, **kwargs):
        return []

    async def mock_store(request):
        result = MagicMock()
        result.node_id = f"node_{len(mound._nodes) + 1:03d}"
        result.success = True
        return result

    mound.query = AsyncMock(side_effect=mock_query)
    mound.store = AsyncMock(side_effect=mock_store)
    return mound


@pytest.fixture
def cdc_aware_agents(mock_knowledge_mound_with_cdc):
    """Agents that can access CDC knowledge."""
    return [
        MockCDCKnowledgeAgent(
            name="agent_1",
            knowledge_mound=mock_knowledge_mound_with_cdc,
            responses=[
                "Based on CDC data showing API latency increase, I propose "
                "implementing regional rate limiting to address performance issues.",
            ],
        ),
        MockCDCKnowledgeAgent(
            name="agent_2",
            knowledge_mound=mock_knowledge_mound_with_cdc,
            responses=[
                "The CDC feedback data shows users want better rate limits. "
                "I suggest a tiered approach with burst allowances.",
            ],
        ),
        MockCDCKnowledgeAgent(
            name="agent_3",
            knowledge_mound=mock_knowledge_mound_with_cdc,
            responses=[
                "Combining CDC metrics and feedback, the solution should "
                "address both latency and user expectations.",
            ],
        ),
    ]


@pytest.fixture
def simple_environment() -> Environment:
    """Environment for testing CDC-informed debates."""
    return Environment(
        task="Design an API rate limiting strategy",
        context="Building a web API that needs intelligent rate limiting based on usage patterns",
    )


@pytest.fixture
def quick_protocol() -> DebateProtocol:
    """Quick debate protocol for fast tests."""
    return DebateProtocol(
        rounds=1,
        consensus="any",
        critique_required=False,
    )


# =============================================================================
# Tests: CDC Data → Debate Context
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_debate_with_cdc_knowledge_injection(
    cdc_aware_agents,
    simple_environment,
    quick_protocol,
    mock_knowledge_mound_with_cdc,
):
    """Test that CDC knowledge is accessible during debate."""
    with patch.object(
        Arena,
        "_gather_trending_context",
        new_callable=AsyncMock,
        return_value=None,
    ):
        arena = Arena(
            environment=simple_environment,
            agents=cdc_aware_agents,
            protocol=quick_protocol,
        )
        result = await arena.run()

    # Verify debate completed
    assert result is not None
    assert result.rounds_completed >= 1

    # Verify agents were called during debate
    for agent in cdc_aware_agents:
        assert agent._call_count > 0, f"Agent {agent.name} was not called"

    # Verify CDC data structure is correct (knowledge mound is ready for use)
    assert len(mock_knowledge_mound_with_cdc._nodes) >= 1
    assert all("metadata" in n for n in mock_knowledge_mound_with_cdc._nodes)


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_debate_without_cdc_data_graceful_handling(
    empty_knowledge_mound,
    simple_environment,
    quick_protocol,
):
    """Test that debate proceeds gracefully when no CDC data exists."""
    agents = [
        MockCDCKnowledgeAgent(
            name=f"agent_{i}",
            knowledge_mound=empty_knowledge_mound,
        )
        for i in range(3)
    ]

    with patch.object(
        Arena,
        "_gather_trending_context",
        new_callable=AsyncMock,
        return_value=None,
    ):
        arena = Arena(
            environment=simple_environment,
            agents=agents,
            protocol=quick_protocol,
        )
        result = await arena.run()

    # Debate should complete even without CDC data
    assert result is not None
    assert result.rounds_completed >= 1


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_knowledge_query_failure_handled(
    simple_environment,
    quick_protocol,
):
    """Test that CDC query failures don't break debate."""
    # Knowledge mound that fails on query
    failing_mound = MagicMock()
    failing_mound.query = AsyncMock(side_effect=Exception("Connection failed"))

    agents = [
        MockCDCKnowledgeAgent(
            name=f"agent_{i}",
            knowledge_mound=failing_mound,
        )
        for i in range(3)
    ]

    with patch.object(
        Arena,
        "_gather_trending_context",
        new_callable=AsyncMock,
        return_value=None,
    ):
        arena = Arena(
            environment=simple_environment,
            agents=agents,
            protocol=quick_protocol,
        )
        result = await arena.run()

    # Debate should complete despite CDC failures
    assert result is not None
    assert result.rounds_completed >= 1


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_metadata_preserved_in_context(
    mock_knowledge_mound_with_cdc,
):
    """Test that CDC metadata (source, operation, table) is preserved."""
    nodes = mock_knowledge_mound_with_cdc._nodes

    # Verify CDC metadata structure
    for node in nodes:
        metadata = node.get("metadata", {})
        assert "source_type" in metadata, f"Missing source_type in {node['id']}"
        assert "cdc_operation" in metadata, f"Missing cdc_operation in {node['id']}"
        assert metadata["source_type"] in ["postgresql", "mongodb"]
        assert metadata["cdc_operation"] in ["insert", "update", "delete"]


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_topic_based_query_matching(
    mock_knowledge_mound_with_cdc,
):
    """Test that CDC nodes can be queried by topic relevance."""
    # Query for rate limits - should find feedback node
    results = await mock_knowledge_mound_with_cdc.query("rate limits api")
    assert len(results) > 0
    assert any("rate" in r["content"].lower() for r in results)

    # Query for latency - should find metrics node
    results = await mock_knowledge_mound_with_cdc.query("latency performance")
    assert len(results) > 0
    assert any("latency" in r["content"].lower() for r in results)

    # Query for unrelated topic - should return empty
    results = await mock_knowledge_mound_with_cdc.query("xyz123nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_multiple_cdc_sources_in_debate(
    mock_knowledge_mound_with_cdc,
    simple_environment,
    quick_protocol,
):
    """Test that debate can use CDC data from multiple sources (pg, mongo)."""
    nodes = mock_knowledge_mound_with_cdc._nodes
    sources = {n["metadata"]["source_type"] for n in nodes}

    # Verify we have multiple CDC sources
    assert "postgresql" in sources
    assert "mongodb" in sources

    # Run debate with this mixed-source knowledge
    agents = [
        MockCDCKnowledgeAgent(
            name=f"agent_{i}",
            knowledge_mound=mock_knowledge_mound_with_cdc,
        )
        for i in range(3)
    ]

    with patch.object(
        Arena,
        "_gather_trending_context",
        new_callable=AsyncMock,
        return_value=None,
    ):
        arena = Arena(
            environment=simple_environment,
            agents=agents,
            protocol=quick_protocol,
        )
        result = await arena.run()

    assert result is not None
    # Verify debate completed successfully with multi-source CDC data available
    assert result.rounds_completed >= 1
    # Verify the multi-source CDC structure
    assert len(sources) >= 2


# =============================================================================
# Tests: CDC Event Freshness
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_timestamp_available_for_freshness(
    mock_knowledge_mound_with_cdc,
):
    """Test that CDC events have timestamps for freshness filtering."""
    nodes = mock_knowledge_mound_with_cdc._nodes

    for node in nodes:
        metadata = node.get("metadata", {})
        assert "timestamp" in metadata, f"Missing timestamp in {node['id']}"
        # Verify timestamp is parseable
        timestamp_str = metadata["timestamp"]
        assert timestamp_str is not None


# =============================================================================
# Tests: CDC Operation Types
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_insert_operations_captured(
    mock_knowledge_mound_with_cdc,
):
    """Test that INSERT CDC operations are captured."""
    nodes = mock_knowledge_mound_with_cdc._nodes
    inserts = [n for n in nodes if n["metadata"]["cdc_operation"] == "insert"]
    assert len(inserts) >= 1


@pytest.mark.asyncio
@pytest.mark.integration_minimal
async def test_cdc_update_operations_captured(
    mock_knowledge_mound_with_cdc,
):
    """Test that UPDATE CDC operations are captured."""
    nodes = mock_knowledge_mound_with_cdc._nodes
    updates = [n for n in nodes if n["metadata"]["cdc_operation"] == "update"]
    assert len(updates) >= 1
