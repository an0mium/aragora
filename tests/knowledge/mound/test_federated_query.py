"""
Tests for FederatedQueryAggregator - cross-adapter query aggregation.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass

from aragora.knowledge.mound.federated_query import (
    FederatedQueryAggregator,
    FederatedQueryResult,
    FederatedResult,
    QuerySource,
)


@dataclass
class MockSearchResult:
    """Mock search result item."""

    id: str
    content: str
    score: float = 0.5


class TestFederatedResult:
    """Tests for FederatedResult."""

    def test_create_result(self):
        """Test creating a federated result."""
        item = MockSearchResult(id="1", content="Test content")
        result = FederatedResult(
            source=QuerySource.EVIDENCE,
            item=item,
            relevance_score=0.8,
        )

        assert result.source == QuerySource.EVIDENCE
        assert result.content == "Test content"
        assert result.id == "1"
        assert result.relevance_score == 0.8

    def test_result_from_dict(self):
        """Test extracting content from dict item."""
        item = {"id": "2", "content": "Dict content"}
        result = FederatedResult(
            source=QuerySource.BELIEF,
            item=item,
        )

        assert result.content == "Dict content"
        assert result.id == "2"

    def test_to_dict(self):
        """Test dictionary conversion."""
        item = MockSearchResult(id="1", content="Test")
        result = FederatedResult(
            source=QuerySource.INSIGHTS,
            item=item,
            relevance_score=0.9,
        )

        d = result.to_dict()
        assert d["source"] == "insights"
        assert d["relevance_score"] == 0.9


class TestFederatedQueryResult:
    """Tests for FederatedQueryResult."""

    def test_create_query_result(self):
        """Test creating query result."""
        result = FederatedQueryResult(
            query="test query",
            sources_queried=[QuerySource.EVIDENCE, QuerySource.BELIEF],
        )

        assert result.query == "test query"
        assert len(result.sources_queried) == 2
        assert result.total_count == 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = FederatedQueryResult(
            query="test",
            results=[
                FederatedResult(
                    source=QuerySource.EVIDENCE,
                    item={"content": "test"},
                    relevance_score=0.5,
                )
            ],
            sources_queried=[QuerySource.EVIDENCE],
            sources_succeeded=[QuerySource.EVIDENCE],
        )

        d = result.to_dict()
        assert d["query"] == "test"
        assert len(d["results"]) == 1
        assert d["sources_succeeded"] == ["evidence"]


class TestFederatedQueryAggregator:
    """Tests for FederatedQueryAggregator."""

    def test_create_aggregator(self):
        """Test creating aggregator."""
        agg = FederatedQueryAggregator()

        assert agg._parallel is True
        assert agg._timeout_seconds == 10.0
        assert agg._default_limit == 20

    def test_register_adapter(self):
        """Test registering an adapter."""
        agg = FederatedQueryAggregator()

        mock_adapter = MagicMock()
        mock_adapter.search_by_topic = MagicMock(return_value=[])

        agg.register_adapter(
            source=QuerySource.EVIDENCE,
            adapter=mock_adapter,
            search_method="search_by_topic",
        )

        assert QuerySource.EVIDENCE in agg._adapters
        assert agg._adapters[QuerySource.EVIDENCE].enabled is True

    def test_register_adapter_with_string(self):
        """Test registering adapter with string source."""
        agg = FederatedQueryAggregator()

        mock_adapter = MagicMock()
        mock_adapter.search = MagicMock(return_value=[])

        agg.register_adapter(
            source="belief",
            adapter=mock_adapter,
            search_method="search",
        )

        assert QuerySource.BELIEF in agg._adapters

    def test_unregister_adapter(self):
        """Test unregistering adapter."""
        agg = FederatedQueryAggregator()

        mock_adapter = MagicMock()
        mock_adapter.search = MagicMock()

        agg.register_adapter(QuerySource.EVIDENCE, mock_adapter, "search")
        result = agg.unregister_adapter(QuerySource.EVIDENCE)

        assert result is True
        assert QuerySource.EVIDENCE not in agg._adapters

    def test_enable_disable_adapter(self):
        """Test enabling/disabling adapters."""
        agg = FederatedQueryAggregator()

        mock_adapter = MagicMock()
        mock_adapter.search = MagicMock()

        agg.register_adapter(QuerySource.INSIGHTS, mock_adapter, "search")

        agg.disable_adapter(QuerySource.INSIGHTS)
        assert agg._adapters[QuerySource.INSIGHTS].enabled is False

        agg.enable_adapter(QuerySource.INSIGHTS)
        assert agg._adapters[QuerySource.INSIGHTS].enabled is True

    @pytest.mark.asyncio
    async def test_query_single_adapter(self):
        """Test querying a single adapter."""
        agg = FederatedQueryAggregator()

        # Create mock adapter with async search
        mock_adapter = MagicMock()
        mock_adapter.search_by_topic = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="climate change evidence"),
                MockSearchResult(id="2", content="global warming data"),
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, mock_adapter, "search_by_topic")

        result = await agg.query("climate change", sources=[QuerySource.EVIDENCE])

        assert result.total_count == 2
        assert QuerySource.EVIDENCE in result.sources_succeeded
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_query_multiple_adapters(self):
        """Test querying multiple adapters."""
        agg = FederatedQueryAggregator()

        # Evidence adapter
        evidence_adapter = MagicMock()
        evidence_adapter.search = AsyncMock(
            return_value=[
                MockSearchResult(id="e1", content="evidence content"),
            ]
        )

        # Belief adapter
        belief_adapter = MagicMock()
        belief_adapter.search = AsyncMock(
            return_value=[
                MockSearchResult(id="b1", content="belief content"),
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, evidence_adapter, "search")
        agg.register_adapter(QuerySource.BELIEF, belief_adapter, "search")

        result = await agg.query("topic")

        assert result.total_count == 2
        assert QuerySource.EVIDENCE in result.sources_succeeded
        assert QuerySource.BELIEF in result.sources_succeeded

    @pytest.mark.asyncio
    async def test_query_all_enabled(self):
        """Test querying all enabled adapters."""
        agg = FederatedQueryAggregator()

        # Register 3 adapters
        for source in [QuerySource.EVIDENCE, QuerySource.BELIEF, QuerySource.INSIGHTS]:
            adapter = MagicMock()
            adapter.search = AsyncMock(
                return_value=[
                    MockSearchResult(id=f"{source.value}_1", content=f"{source.value} result"),
                ]
            )
            agg.register_adapter(source, adapter, "search")

        # Disable one
        agg.disable_adapter(QuerySource.INSIGHTS)

        # Query all enabled (should be 2)
        result = await agg.query("topic")

        assert len(result.sources_queried) == 2
        assert QuerySource.INSIGHTS not in result.sources_queried

    @pytest.mark.asyncio
    async def test_query_with_timeout(self):
        """Test query timeout handling."""
        agg = FederatedQueryAggregator(timeout_seconds=0.1)

        # Slow adapter
        slow_adapter = MagicMock()

        async def slow_search(**kwargs):
            await asyncio.sleep(1.0)  # Sleep longer than timeout
            return []

        slow_adapter.search = slow_search

        agg.register_adapter(QuerySource.EVIDENCE, slow_adapter, "search")

        result = await agg.query("topic")

        assert QuerySource.EVIDENCE in result.sources_failed
        assert "Timeout" in result.errors[QuerySource.EVIDENCE.value]

    @pytest.mark.asyncio
    async def test_query_with_error(self):
        """Test query error handling."""
        agg = FederatedQueryAggregator()

        error_adapter = MagicMock()
        error_adapter.search = AsyncMock(side_effect=ValueError("Test error"))

        agg.register_adapter(QuerySource.BELIEF, error_adapter, "search")

        result = await agg.query("topic")

        assert QuerySource.BELIEF in result.sources_failed
        assert "Test error" in result.errors[QuerySource.BELIEF.value]

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test result deduplication."""
        agg = FederatedQueryAggregator(deduplicate=True)

        # Two adapters returning same content
        adapter1 = MagicMock()
        adapter1.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="duplicate content"),
            ]
        )

        adapter2 = MagicMock()
        adapter2.search = AsyncMock(
            return_value=[
                MockSearchResult(id="2", content="duplicate content"),  # Same content
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, adapter1, "search")
        agg.register_adapter(QuerySource.BELIEF, adapter2, "search")

        result = await agg.query("topic")

        # Should deduplicate to 1 result
        assert result.total_count == 1

    @pytest.mark.asyncio
    async def test_relevance_scoring(self):
        """Test relevance-based scoring."""
        agg = FederatedQueryAggregator()

        adapter = MagicMock()
        adapter.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="climate change global warming"),
                MockSearchResult(id="2", content="unrelated topic"),
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, adapter, "search")

        result = await agg.query("climate change")

        # First result should have higher relevance
        assert result.results[0].relevance_score > result.results[1].relevance_score

    @pytest.mark.asyncio
    async def test_min_relevance_filter(self):
        """Test minimum relevance filtering."""
        agg = FederatedQueryAggregator()

        adapter = MagicMock()
        adapter.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="climate change evidence"),
                MockSearchResult(id="2", content="unrelated content xyz"),
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, adapter, "search")

        # High min relevance should filter out unrelated
        result = await agg.query("climate change", min_relevance=0.5)

        # Should only include highly relevant results
        assert all(r.relevance_score >= 0.5 for r in result.results)

    @pytest.mark.asyncio
    async def test_adapter_weights(self):
        """Test adapter weight affects relevance."""
        agg = FederatedQueryAggregator()

        # High weight adapter
        adapter1 = MagicMock()
        adapter1.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="topic result"),
            ]
        )

        # Low weight adapter
        adapter2 = MagicMock()
        adapter2.search = AsyncMock(
            return_value=[
                MockSearchResult(id="2", content="topic result"),  # Same content
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, adapter1, "search", weight=2.0)
        agg.register_adapter(QuerySource.BELIEF, adapter2, "search", weight=0.5)

        result = await agg.query("topic", deduplicate=False)

        # Evidence result should have higher relevance due to weight
        evidence_results = [r for r in result.results if r.source == QuerySource.EVIDENCE]
        belief_results = [r for r in result.results if r.source == QuerySource.BELIEF]

        if evidence_results and belief_results:
            assert evidence_results[0].relevance_score > belief_results[0].relevance_score

    def test_get_stats(self):
        """Test getting statistics."""
        agg = FederatedQueryAggregator()

        adapter = MagicMock()
        adapter.search = MagicMock()

        agg.register_adapter(QuerySource.EVIDENCE, adapter, "search")
        agg.register_adapter(QuerySource.BELIEF, adapter, "search", enabled=False)

        stats = agg.get_stats()

        assert stats["registered_adapters"] == 2
        assert stats["enabled_adapters"] == 1
        assert "evidence" in stats["adapters"]

    def test_get_registered_sources(self):
        """Test getting registered sources."""
        agg = FederatedQueryAggregator()

        adapter = MagicMock()
        adapter.search = MagicMock()

        agg.register_adapter(QuerySource.EVIDENCE, adapter, "search")
        agg.register_adapter(QuerySource.INSIGHTS, adapter, "search")

        sources = agg.get_registered_sources()

        assert "evidence" in sources
        assert "insights" in sources


class TestSequentialQuery:
    """Tests for sequential (non-parallel) querying."""

    @pytest.mark.asyncio
    async def test_sequential_query(self):
        """Test sequential adapter querying."""
        agg = FederatedQueryAggregator(parallel=False)

        adapter1 = MagicMock()
        adapter1.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="result 1"),
            ]
        )

        adapter2 = MagicMock()
        adapter2.search = AsyncMock(
            return_value=[
                MockSearchResult(id="2", content="result 2"),
            ]
        )

        agg.register_adapter(QuerySource.EVIDENCE, adapter1, "search")
        agg.register_adapter(QuerySource.BELIEF, adapter2, "search")

        result = await agg.query("topic")

        assert result.total_count == 2
        # Both should succeed
        assert len(result.sources_succeeded) == 2
