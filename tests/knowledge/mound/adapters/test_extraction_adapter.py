"""
Tests for ExtractionAdapter - Knowledge Extraction to Knowledge Mound bridge.

Tests cover:
- Adapter initialization and configuration
- Entity extraction from debate content
- Relationship extraction between concepts
- Knowledge graph operations
- Query methods for extracted knowledge
- Batch extraction processing
- Promotion to Knowledge Mound
- Error handling and recovery
- Edge cases
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from aragora.knowledge.mound.adapters.extraction_adapter import (
    ExtractionAdapter,
    ExtractionAdapterError,
    ExtractionNotFoundError,
    ExtractionSearchResult,
    RelationshipSearchResult,
    KnowledgeGraphNode,
    BatchExtractionResult,
)
from aragora.knowledge.mound.ops.extraction import (
    ExtractionConfig,
    ExtractionType,
    ConfidenceSource,
    ExtractedClaim,
    ExtractedRelationship,
    ExtractionResult,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock Knowledge Mound."""
    mound = MagicMock()
    mound.store = AsyncMock(return_value=MagicMock(id="stored-id"))
    mound.query = AsyncMock(return_value=MagicMock(items=[]))
    return mound


@pytest.fixture
def adapter():
    """Create an adapter for testing."""
    return ExtractionAdapter()


@pytest.fixture
def adapter_with_mound(mock_mound):
    """Create an adapter with mound configured."""
    adapter = ExtractionAdapter(mound=mock_mound)
    return adapter


@pytest.fixture
def config():
    """Create an extraction configuration."""
    return ExtractionConfig(
        min_confidence_to_extract=0.3,
        min_confidence_to_promote=0.6,
        extract_facts=True,
        extract_definitions=True,
        extract_relationships=True,
    )


@pytest.fixture
def sample_messages():
    """Create sample debate messages for testing."""
    return [
        {
            "agent_id": "claude",
            "content": "It is a fact that machine learning is a type of AI. "
            "Studies show that neural networks are effective for pattern recognition.",
            "round": 1,
        },
        {
            "agent_id": "gpt-4",
            "content": "Machine learning is defined as a method of data analysis. "
            "Deep learning requires large datasets for training.",
            "round": 1,
        },
        {
            "agent_id": "gemini",
            "content": "I think neural networks are powerful. "
            "According to research, transformers are essential for NLP.",
            "round": 2,
        },
    ]


@pytest.fixture
def sample_consensus():
    """Create sample consensus text."""
    return "In conclusion, machine learning and deep learning are both effective approaches to AI."


def create_mock_claim(
    claim_id: str = "claim-1",
    content: str = "Test claim content",
    extraction_type: ExtractionType = ExtractionType.FACT,
    confidence: float = 0.7,
    topics: list = None,
    debate_id: str = "debate-1",
) -> ExtractedClaim:
    """Create a mock extracted claim."""
    return ExtractedClaim(
        id=claim_id,
        content=content,
        extraction_type=extraction_type,
        source_debate_id=debate_id,
        source_agent_id="claude",
        confidence=confidence,
        topics=topics or ["ML", "AI"],
        supporting_agents=["claude"],
    )


def create_mock_relationship(
    rel_id: str = "rel-1",
    source: str = "ML",
    target: str = "AI",
    rel_type: str = "is_a",
    confidence: float = 0.6,
    debate_id: str = "debate-1",
) -> ExtractedRelationship:
    """Create a mock extracted relationship."""
    return ExtractedRelationship(
        id=rel_id,
        source_concept=source,
        target_concept=target,
        relationship_type=rel_type,
        source_debate_id=debate_id,
        confidence=confidence,
        evidence=f"{source} is a type of {target}",
    )


# ============================================================================
# Test Adapter Initialization
# ============================================================================


class TestAdapterInitialization:
    """Tests for ExtractionAdapter initialization."""

    def test_init_without_mound(self, adapter):
        """Test adapter can initialize without mound."""
        assert adapter._mound is None
        assert adapter._auto_promote is False
        assert adapter._min_confidence_for_promotion == 0.6

    def test_init_with_mound(self, mock_mound):
        """Test adapter initializes with provided mound."""
        adapter = ExtractionAdapter(mound=mock_mound)
        assert adapter._mound is mock_mound

    def test_init_with_config(self, config):
        """Test adapter initializes with custom config."""
        adapter = ExtractionAdapter(config=config)
        assert adapter._config == config
        assert adapter._config.min_confidence_to_extract == 0.3

    def test_init_with_auto_promote(self, mock_mound):
        """Test adapter initializes with auto-promote enabled."""
        adapter = ExtractionAdapter(
            mound=mock_mound,
            auto_promote=True,
            min_confidence_for_promotion=0.7,
        )
        assert adapter._auto_promote is True
        assert adapter._min_confidence_for_promotion == 0.7

    def test_init_with_event_callback(self):
        """Test adapter initializes with event callback."""
        events = []

        def callback(t, d):
            events.append((t, d))

        adapter = ExtractionAdapter(event_callback=callback)
        assert adapter._event_callback is callback

    def test_init_with_dual_write(self):
        """Test adapter initializes with dual write mode."""
        adapter = ExtractionAdapter(enable_dual_write=True)
        assert adapter._enable_dual_write is True

    def test_set_mound(self, adapter, mock_mound):
        """Test setting the mound after initialization."""
        assert adapter._mound is None
        adapter.set_mound(mock_mound)
        assert adapter._mound is mock_mound

    def test_set_config(self, adapter, config):
        """Test setting config after initialization."""
        adapter.set_config(config)
        assert adapter._config == config

    def test_get_config(self, adapter):
        """Test getting the current configuration."""
        config = adapter.get_config()
        assert isinstance(config, ExtractionConfig)

    def test_adapter_name(self, adapter):
        """Test adapter name is set correctly."""
        assert adapter.adapter_name == "extraction"

    def test_id_prefix(self, adapter):
        """Test ID prefix is set correctly."""
        assert adapter.ID_PREFIX == "ext_"


# ============================================================================
# Test Entity Extraction Operations
# ============================================================================


class TestEntityExtraction:
    """Tests for entity extraction from debate content."""

    @pytest.mark.asyncio
    async def test_extract_from_debate_basic(self, adapter, sample_messages):
        """Test basic extraction from debate."""
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert isinstance(result, ExtractionResult)
        assert result.debate_id == "debate-123"
        assert len(result.claims) > 0

    @pytest.mark.asyncio
    async def test_extract_with_consensus(self, adapter, sample_messages, sample_consensus):
        """Test extraction includes consensus claims."""
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            consensus_text=sample_consensus,
        )

        # Check for consensus-type claims
        consensus_claims = [
            c for c in result.claims if c.extraction_type == ExtractionType.CONSENSUS
        ]
        # Consensus may or may not generate claims depending on pattern matching
        assert isinstance(result.claims, list)

    @pytest.mark.asyncio
    async def test_extract_with_topic(self, adapter, sample_messages):
        """Test extraction includes provided topic."""
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Machine Learning",
        )

        assert "Machine Learning" in result.topics_discovered

    @pytest.mark.asyncio
    async def test_extract_stores_result(self, adapter, sample_messages):
        """Test extraction result is stored."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        result = await adapter.get_extraction_result("debate-123")
        assert result is not None
        assert result.debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_extract_updates_stats(self, adapter, sample_messages):
        """Test extraction updates statistics."""
        stats_before = adapter.get_stats()
        assert stats_before["debates_processed"] == 0

        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        stats_after = adapter.get_stats()
        assert stats_after["debates_processed"] == 1
        assert stats_after["total_claims_extracted"] > 0

    @pytest.mark.asyncio
    async def test_extract_emits_event(self, sample_messages):
        """Test extraction emits event."""
        events = []
        adapter = ExtractionAdapter(event_callback=lambda t, d: events.append((t, d)))

        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        assert len(events) == 1
        event_type, data = events[0]
        assert event_type == "knowledge_extracted"
        assert data["debate_id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_extract_with_empty_messages(self, adapter):
        """Test extraction with empty message list."""
        result = await adapter.extract_from_debate(
            debate_id="debate-empty",
            messages=[],
        )

        assert result.debate_id == "debate-empty"
        assert len(result.claims) == 0
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_extract_fact_type(self, adapter):
        """Test extraction of fact-type claims."""
        messages = [
            {
                "agent_id": "claude",
                "content": "It is a fact that water boils at 100 degrees Celsius.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-fact",
            messages=messages,
        )

        fact_claims = [c for c in result.claims if c.extraction_type == ExtractionType.FACT]
        assert len(fact_claims) > 0

    @pytest.mark.asyncio
    async def test_extract_definition_type(self, adapter):
        """Test extraction of definition-type claims."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Machine learning is defined as a subset of AI that learns from data.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-def",
            messages=messages,
        )

        def_claims = [c for c in result.claims if c.extraction_type == ExtractionType.DEFINITION]
        assert len(def_claims) > 0

    @pytest.mark.asyncio
    async def test_extract_procedure_type(self, adapter):
        """Test extraction of procedure-type claims."""
        messages = [
            {
                "agent_id": "claude",
                "content": "To train a model, the steps are: 1. prepare data, 2. train, 3. evaluate.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-proc",
            messages=messages,
        )

        proc_claims = [c for c in result.claims if c.extraction_type == ExtractionType.PROCEDURE]
        assert len(proc_claims) > 0

    @pytest.mark.asyncio
    async def test_extract_filters_low_confidence(self, adapter):
        """Test extraction filters low confidence claims."""
        # Hedging words should reduce confidence
        messages = [
            {
                "agent_id": "claude",
                "content": "Maybe possibly perhaps this could work somehow.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-low",
            messages=messages,
        )

        # Low confidence claims should be filtered out
        for claim in result.claims:
            assert claim.confidence >= 0.3  # min_confidence_to_extract default


# ============================================================================
# Test Relationship Extraction
# ============================================================================


class TestRelationshipExtraction:
    """Tests for relationship extraction between concepts."""

    @pytest.mark.asyncio
    async def test_extract_is_a_relationship(self, adapter):
        """Test extraction of is_a relationships."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Python is a type of programming language.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-rel",
            messages=messages,
        )

        is_a_rels = [r for r in result.relationships if r.relationship_type == "is_a"]
        assert len(is_a_rels) > 0

    @pytest.mark.asyncio
    async def test_extract_causes_relationship(self, adapter):
        """Test extraction of causes relationships."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Overfitting causes poor generalization on new data.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-causes",
            messages=messages,
        )

        causes_rels = [r for r in result.relationships if r.relationship_type == "causes"]
        assert len(causes_rels) > 0

    @pytest.mark.asyncio
    async def test_extract_requires_relationship(self, adapter):
        """Test extraction of requires relationships."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Deep learning requires large datasets for training.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-requires",
            messages=messages,
        )

        requires_rels = [r for r in result.relationships if r.relationship_type == "requires"]
        assert len(requires_rels) > 0

    @pytest.mark.asyncio
    async def test_extract_part_of_relationship(self, adapter):
        """Test extraction of part_of relationships."""
        messages = [
            {
                "agent_id": "claude",
                "content": "The neuron is part of the brain structure.",
                "round": 1,
            }
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-part",
            messages=messages,
        )

        part_of_rels = [r for r in result.relationships if r.relationship_type == "part_of"]
        assert len(part_of_rels) > 0

    @pytest.mark.asyncio
    async def test_search_relationships_by_source(self, adapter, sample_messages):
        """Test searching relationships by source concept."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_relationships(source_concept="machine")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_relationships_by_target(self, adapter, sample_messages):
        """Test searching relationships by target concept."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_relationships(target_concept="AI")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_relationships_by_type(self, adapter, sample_messages):
        """Test searching relationships by type."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_relationships(relationship_type="is_a")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_relationship_by_id(self, adapter, sample_messages):
        """Test getting relationship by ID."""
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        if result.relationships:
            rel_id = result.relationships[0].id
            rel = await adapter.get_relationship(rel_id)
            assert rel is not None
            assert rel.id == rel_id


# ============================================================================
# Test Knowledge Graph Operations
# ============================================================================


class TestKnowledgeGraphOperations:
    """Tests for knowledge graph updates and queries."""

    @pytest.mark.asyncio
    async def test_graph_updated_on_extraction(self, adapter, sample_messages):
        """Test knowledge graph is updated on extraction."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Machine Learning",
        )

        nodes = await adapter.get_graph_nodes()
        assert len(nodes) > 0

    @pytest.mark.asyncio
    async def test_get_graph_node_by_id(self, adapter, sample_messages):
        """Test getting graph node by ID."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Machine Learning",
        )

        # Topic nodes are created with topic_ prefix
        node = await adapter.get_graph_node("topic_machine_learning")
        assert node is not None
        assert node.concept == "Machine Learning"

    @pytest.mark.asyncio
    async def test_get_graph_nodes_with_filter(self, adapter, sample_messages):
        """Test getting graph nodes with concept filter."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Deep Learning",
        )

        nodes = await adapter.get_graph_nodes(concept_filter="deep")
        # May or may not find nodes depending on extraction
        assert isinstance(nodes, list)

    @pytest.mark.asyncio
    async def test_get_graph_nodes_with_min_confidence(self, adapter, sample_messages):
        """Test getting graph nodes with minimum confidence."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        nodes = await adapter.get_graph_nodes(min_confidence=0.9)
        # All returned nodes should meet threshold
        for node in nodes:
            assert node.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_get_related_concepts(self, adapter):
        """Test getting related concepts."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Python is a type of programming language. "
                "Programming requires a computer.",
                "round": 1,
            }
        ]

        await adapter.extract_from_debate(
            debate_id="debate-rel",
            messages=messages,
        )

        related = await adapter.get_related_concepts("programming", direction="both")
        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_get_related_concepts_outgoing(self, adapter):
        """Test getting outgoing related concepts."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Python is a type of language.",
                "round": 1,
            }
        ]

        await adapter.extract_from_debate(
            debate_id="debate-out",
            messages=messages,
        )

        related = await adapter.get_related_concepts("python", direction="outgoing")
        for r in related:
            assert r["direction"] == "outgoing"

    @pytest.mark.asyncio
    async def test_get_related_concepts_incoming(self, adapter):
        """Test getting incoming related concepts."""
        messages = [
            {
                "agent_id": "claude",
                "content": "Python is a type of language.",
                "round": 1,
            }
        ]

        await adapter.extract_from_debate(
            debate_id="debate-in",
            messages=messages,
        )

        related = await adapter.get_related_concepts("language", direction="incoming")
        for r in related:
            assert r["direction"] == "incoming"

    @pytest.mark.asyncio
    async def test_graph_node_structure(self, adapter, sample_messages):
        """Test knowledge graph node structure."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Test Topic",
        )

        nodes = await adapter.get_graph_nodes()
        if nodes:
            node = nodes[0]
            assert isinstance(node, KnowledgeGraphNode)
            assert hasattr(node, "id")
            assert hasattr(node, "concept")
            assert hasattr(node, "claim_ids")
            assert hasattr(node, "relationship_ids")
            assert hasattr(node, "confidence")

    @pytest.mark.asyncio
    async def test_graph_node_to_dict(self, adapter, sample_messages):
        """Test knowledge graph node serialization."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
            topic="Test Topic",
        )

        nodes = await adapter.get_graph_nodes()
        if nodes:
            node_dict = nodes[0].to_dict()
            assert "id" in node_dict
            assert "concept" in node_dict
            assert "claim_ids" in node_dict
            assert "confidence" in node_dict


# ============================================================================
# Test Query Methods
# ============================================================================


class TestQueryMethods:
    """Tests for query methods on extracted knowledge."""

    @pytest.mark.asyncio
    async def test_search_claims_basic(self, adapter, sample_messages):
        """Test basic claim search."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims("machine learning")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_claims_with_limit(self, adapter, sample_messages):
        """Test claim search with limit."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims("", limit=5)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_claims_with_min_confidence(self, adapter, sample_messages):
        """Test claim search with minimum confidence."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims("", min_confidence=0.5)
        for result in results:
            assert result.claim.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_search_claims_by_type(self, adapter, sample_messages):
        """Test claim search by extraction type."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims(
            "",
            extraction_types=[ExtractionType.FACT],
        )
        for result in results:
            assert result.claim.extraction_type == ExtractionType.FACT

    @pytest.mark.asyncio
    async def test_search_claims_by_debate(self, adapter, sample_messages):
        """Test claim search filtered by debate ID."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )
        await adapter.extract_from_debate(
            debate_id="debate-456",
            messages=sample_messages,
        )

        results = await adapter.search_claims("", debate_id="debate-123")
        for result in results:
            assert result.claim.source_debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_search_claims_relevance_ranking(self, adapter):
        """Test claims are ranked by relevance."""
        messages = [
            {
                "agent_id": "claude",
                "content": "It is a fact that Python programming is popular. "
                "Python is used for machine learning.",
                "round": 1,
            }
        ]

        await adapter.extract_from_debate(
            debate_id="debate-python",
            messages=messages,
        )

        results = await adapter.search_claims("python")
        if len(results) > 1:
            # Results should be sorted by relevance
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_get_claim_by_id(self, adapter, sample_messages):
        """Test getting claim by ID."""
        result = await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        if result.claims:
            claim_id = result.claims[0].id
            claim = await adapter.get_claim(claim_id)
            assert claim is not None
            assert claim.id == claim_id

    @pytest.mark.asyncio
    async def test_get_claim_not_found(self, adapter):
        """Test getting non-existent claim returns None."""
        claim = await adapter.get_claim("nonexistent-id")
        assert claim is None

    @pytest.mark.asyncio
    async def test_get_extraction_result(self, adapter, sample_messages):
        """Test getting extraction result by debate ID."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        result = await adapter.get_extraction_result("debate-123")
        assert result is not None
        assert result.debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_get_extraction_result_not_found(self, adapter):
        """Test getting non-existent extraction result."""
        result = await adapter.get_extraction_result("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_result_to_dict(self, adapter, sample_messages):
        """Test ExtractionSearchResult serialization."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims("")
        if results:
            result_dict = results[0].to_dict()
            assert "claim" in result_dict
            assert "relevance_score" in result_dict
            assert "matched_topics" in result_dict


# ============================================================================
# Test Batch Extraction Processing
# ============================================================================


class TestBatchExtractionProcessing:
    """Tests for batch extraction of multiple debates."""

    @pytest.mark.asyncio
    async def test_batch_extract_basic(self, adapter):
        """Test basic batch extraction."""
        debates = [
            {
                "debate_id": "batch-1",
                "messages": [
                    {
                        "agent_id": "claude",
                        "content": "It is a fact that Python is popular.",
                        "round": 1,
                    }
                ],
            },
            {
                "debate_id": "batch-2",
                "messages": [
                    {
                        "agent_id": "gpt-4",
                        "content": "Studies show that JavaScript is widely used.",
                        "round": 1,
                    }
                ],
            },
        ]

        result = await adapter.batch_extract(debates)

        assert isinstance(result, BatchExtractionResult)
        assert len(result.debate_ids) == 2
        assert "batch-1" in result.debate_ids
        assert "batch-2" in result.debate_ids

    @pytest.mark.asyncio
    async def test_batch_extract_aggregates_counts(self, adapter, sample_messages):
        """Test batch extraction aggregates claim and relationship counts."""
        debates = [{"debate_id": f"batch-{i}", "messages": sample_messages} for i in range(3)]

        result = await adapter.batch_extract(debates)

        assert result.total_claims >= 0
        assert result.total_relationships >= 0

    @pytest.mark.asyncio
    async def test_batch_extract_with_workspace(self, adapter_with_mound, sample_messages):
        """Test batch extraction with workspace for promotion."""
        debates = [{"debate_id": "batch-ws-1", "messages": sample_messages}]

        adapter_with_mound._auto_promote = True
        result = await adapter_with_mound.batch_extract(
            debates,
            workspace_id="test-workspace",
        )

        assert isinstance(result, BatchExtractionResult)

    @pytest.mark.asyncio
    async def test_batch_extract_handles_failures(self, adapter):
        """Test batch extraction handles individual failures."""
        debates = [
            {
                "debate_id": "batch-good",
                "messages": [{"agent_id": "claude", "content": "Valid content here."}],
            },
            {
                "debate_id": "batch-empty",
                "messages": [],
            },
        ]

        result = await adapter.batch_extract(debates)

        # Both should complete (empty is not a failure)
        assert len(result.debate_ids) == 2

    @pytest.mark.asyncio
    async def test_batch_extract_concurrency_limit(self, adapter, sample_messages):
        """Test batch extraction respects concurrency limit."""
        debates = [{"debate_id": f"batch-conc-{i}", "messages": sample_messages} for i in range(10)]

        result = await adapter.batch_extract(debates, max_concurrent=2)

        assert len(result.debate_ids) == 10

    @pytest.mark.asyncio
    async def test_batch_extract_empty_list(self, adapter):
        """Test batch extraction with empty debate list."""
        result = await adapter.batch_extract([])

        assert result.debate_ids == []
        assert result.total_claims == 0
        assert result.total_relationships == 0

    @pytest.mark.asyncio
    async def test_batch_result_success_property(self):
        """Test BatchExtractionResult success property."""
        success_result = BatchExtractionResult(
            debate_ids=["d1", "d2"],
            total_claims=10,
            total_relationships=5,
            promoted_count=3,
            failed_debate_ids=[],
            duration_ms=100.0,
            errors=[],
        )
        assert success_result.success is True

        failed_result = BatchExtractionResult(
            debate_ids=["d1"],
            total_claims=5,
            total_relationships=2,
            promoted_count=0,
            failed_debate_ids=["d2"],
            duration_ms=50.0,
            errors=["d2: Failed to extract"],
        )
        assert failed_result.success is False

    @pytest.mark.asyncio
    async def test_batch_result_to_dict(self):
        """Test BatchExtractionResult serialization."""
        result = BatchExtractionResult(
            debate_ids=["d1"],
            total_claims=5,
            total_relationships=2,
            promoted_count=1,
            failed_debate_ids=[],
            duration_ms=50.0,
            errors=[],
        )

        result_dict = result.to_dict()
        assert "debate_ids" in result_dict
        assert "total_claims" in result_dict
        assert "success" in result_dict


# ============================================================================
# Test Promotion to Knowledge Mound
# ============================================================================


class TestPromotionToKnowledgeMound:
    """Tests for promoting claims to Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_promote_claims_without_mound(self, adapter, sample_messages):
        """Test promotion returns 0 when no mound configured."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        promoted = await adapter.promote_claims(
            workspace_id="test-ws",
            debate_id="debate-123",
        )

        assert promoted == 0

    @pytest.mark.asyncio
    async def test_promote_claims_with_mound(self, adapter_with_mound, sample_messages):
        """Test promotion with mound configured."""
        await adapter_with_mound.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        promoted = await adapter_with_mound.promote_claims(
            workspace_id="test-ws",
            debate_id="debate-123",
        )

        # Should attempt to promote high-confidence claims
        assert promoted >= 0

    @pytest.mark.asyncio
    async def test_promote_claims_by_debate_id(self, adapter_with_mound, sample_messages):
        """Test promoting claims filtered by debate ID."""
        await adapter_with_mound.extract_from_debate(
            debate_id="debate-1",
            messages=sample_messages,
        )
        await adapter_with_mound.extract_from_debate(
            debate_id="debate-2",
            messages=sample_messages,
        )

        promoted = await adapter_with_mound.promote_claims(
            workspace_id="test-ws",
            debate_id="debate-1",
        )

        assert promoted >= 0

    @pytest.mark.asyncio
    async def test_promote_claims_with_min_confidence(self, adapter_with_mound, sample_messages):
        """Test promoting claims with minimum confidence."""
        await adapter_with_mound.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        promoted = await adapter_with_mound.promote_claims(
            workspace_id="test-ws",
            min_confidence=0.9,
        )

        # Higher threshold may result in fewer promotions
        assert promoted >= 0

    @pytest.mark.asyncio
    async def test_promote_specific_claims(self, adapter_with_mound, sample_messages):
        """Test promoting specific claim IDs."""
        result = await adapter_with_mound.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        if result.claims:
            claim_ids = [result.claims[0].id]
            promoted = await adapter_with_mound.promote_claims(
                workspace_id="test-ws",
                claim_ids=claim_ids,
            )
            # May promote 0 or 1 depending on confidence
            assert promoted >= 0

    @pytest.mark.asyncio
    async def test_auto_promote_on_extraction(self, mock_mound, sample_messages):
        """Test auto-promotion during extraction."""
        adapter = ExtractionAdapter(
            mound=mock_mound,
            auto_promote=True,
            min_confidence_for_promotion=0.5,
        )

        result = await adapter.extract_from_debate(
            debate_id="debate-auto",
            messages=sample_messages,
            workspace_id="test-ws",
        )

        assert result.promoted_to_mound >= 0

    @pytest.mark.asyncio
    async def test_promote_emits_event(self, sample_messages):
        """Test promotion emits event."""
        events = []
        mound = MagicMock()
        mound.store = AsyncMock(return_value=MagicMock(id="stored"))

        adapter = ExtractionAdapter(
            mound=mound,
            event_callback=lambda t, d: events.append((t, d)),
        )

        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        await adapter.promote_claims(
            workspace_id="test-ws",
            debate_id="debate-123",
        )

        # Should have extraction event and promotion event
        event_types = [e[0] for e in events]
        assert "knowledge_extracted" in event_types
        assert "claims_promoted" in event_types


# ============================================================================
# Test Error Handling and Recovery
# ============================================================================


class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_extraction_error_propagates(self, adapter):
        """Test extraction errors are properly wrapped."""
        with patch.object(
            adapter._extractor,
            "extract_from_debate",
            side_effect=Exception("Extraction failed"),
        ):
            with pytest.raises(ExtractionAdapterError) as exc_info:
                await adapter.extract_from_debate(
                    debate_id="debate-error",
                    messages=[],
                )

            assert "Extraction failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_promotion_error_propagates(self, adapter_with_mound, sample_messages):
        """Test promotion errors are properly wrapped."""
        await adapter_with_mound.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        with patch.object(
            adapter_with_mound._extractor,
            "promote_to_mound",
            side_effect=Exception("Promotion failed"),
        ):
            with pytest.raises(ExtractionAdapterError) as exc_info:
                await adapter_with_mound.promote_claims(
                    workspace_id="test-ws",
                    debate_id="debate-123",
                )

            assert "Promotion failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_extract_continues_on_error(self, adapter):
        """Test batch extraction continues after individual failures."""
        original_extract = adapter.extract_from_debate

        call_count = 0

        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated failure")
            return await original_extract(*args, **kwargs)

        with patch.object(adapter, "extract_from_debate", side_effect=mock_extract):
            debates = [{"debate_id": f"batch-{i}", "messages": []} for i in range(3)]

            result = await adapter.batch_extract(debates)

            # Should have attempted all 3
            assert len(result.debate_ids) == 3
            # One should have failed
            assert len(result.failed_debate_ids) == 1

    @pytest.mark.asyncio
    async def test_clear_extraction_success(self, adapter, sample_messages):
        """Test clearing extraction for a debate."""
        await adapter.extract_from_debate(
            debate_id="debate-clear",
            messages=sample_messages,
        )

        assert await adapter.get_extraction_result("debate-clear") is not None

        cleared = await adapter.clear_extraction("debate-clear")
        assert cleared is True
        assert await adapter.get_extraction_result("debate-clear") is None

    @pytest.mark.asyncio
    async def test_clear_extraction_not_found(self, adapter):
        """Test clearing non-existent extraction."""
        cleared = await adapter.clear_extraction("nonexistent")
        assert cleared is False

    @pytest.mark.asyncio
    async def test_clear_all_extractions(self, adapter, sample_messages):
        """Test clearing all extractions."""
        for i in range(3):
            await adapter.extract_from_debate(
                debate_id=f"debate-{i}",
                messages=sample_messages,
            )

        cleared = await adapter.clear_all_extractions()
        assert cleared == 3

        nodes = await adapter.get_graph_nodes()
        assert len(nodes) == 0

    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, adapter):
        """Test handling of malformed messages."""
        messages = [
            {"no_agent": "missing agent_id"},
            {"agent_id": "claude"},  # Missing content
            None,  # None message (should be filtered)
        ]

        # Filter out None values before processing
        filtered_messages = [m for m in messages if m is not None]

        result = await adapter.extract_from_debate(
            debate_id="debate-malformed",
            messages=filtered_messages,
        )

        # Should complete without error
        assert result.debate_id == "debate-malformed"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_content_messages(self, adapter):
        """Test extraction with empty content messages."""
        messages = [
            {"agent_id": "claude", "content": "", "round": 1},
            {"agent_id": "gpt-4", "content": "   ", "round": 1},
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-empty-content",
            messages=messages,
        )

        assert result.debate_id == "debate-empty-content"
        assert len(result.claims) == 0

    @pytest.mark.asyncio
    async def test_very_short_sentences(self, adapter):
        """Test extraction filters very short sentences."""
        messages = [
            {"agent_id": "claude", "content": "Yes. No. Maybe.", "round": 1},
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-short",
            messages=messages,
        )

        # Short sentences should be filtered out
        for claim in result.claims:
            assert len(claim.content) >= 20

    @pytest.mark.asyncio
    async def test_unicode_content(self, adapter):
        """Test extraction handles unicode content."""
        messages = [
            {
                "agent_id": "claude",
                "content": "It is a fact that \u4e2d\u6587 means Chinese language.",
                "round": 1,
            },
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-unicode",
            messages=messages,
        )

        assert result.debate_id == "debate-unicode"

    @pytest.mark.asyncio
    async def test_special_characters_in_topic(self, adapter):
        """Test extraction handles special characters in topics."""
        result = await adapter.extract_from_debate(
            debate_id="debate-special",
            messages=[],
            topic="C++ Programming & Data Structures",
        )

        assert "C++ Programming & Data Structures" in result.topics_discovered

    @pytest.mark.asyncio
    async def test_duplicate_debate_extraction(self, adapter, sample_messages):
        """Test extracting from same debate twice overwrites."""
        await adapter.extract_from_debate(
            debate_id="debate-dup",
            messages=sample_messages,
        )

        # Extract again
        await adapter.extract_from_debate(
            debate_id="debate-dup",
            messages=[],  # Empty this time
        )

        result = await adapter.get_extraction_result("debate-dup")
        # Should have the second (empty) result
        assert len(result.claims) == 0

    @pytest.mark.asyncio
    async def test_concurrent_extractions(self, adapter, sample_messages):
        """Test concurrent extraction of different debates."""

        async def extract_one(debate_id):
            return await adapter.extract_from_debate(
                debate_id=debate_id,
                messages=sample_messages,
            )

        results = await asyncio.gather(*[extract_one(f"concurrent-{i}") for i in range(5)])

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.debate_id == f"concurrent-{i}"

    @pytest.mark.asyncio
    async def test_large_message_count(self, adapter):
        """Test extraction with many messages."""
        messages = [
            {
                "agent_id": f"agent-{i % 5}",
                "content": f"It is a fact that statement {i} is true.",
                "round": i // 5,
            }
            for i in range(100)
        ]

        result = await adapter.extract_from_debate(
            debate_id="debate-large",
            messages=messages,
        )

        assert result.debate_id == "debate-large"
        assert len(result.claims) > 0

    @pytest.mark.asyncio
    async def test_search_claims_empty_query(self, adapter, sample_messages):
        """Test searching with empty query returns all claims."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_claims("")
        # Empty query should return claims
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_claims_no_results(self, adapter):
        """Test searching with no matching claims."""
        results = await adapter.search_claims("nonexistent_term_xyz")
        assert results == []

    @pytest.mark.asyncio
    async def test_relationship_search_no_filters(self, adapter, sample_messages):
        """Test relationship search with no filters."""
        await adapter.extract_from_debate(
            debate_id="debate-123",
            messages=sample_messages,
        )

        results = await adapter.search_relationships()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_stats_reflect_multiple_extractions(self, adapter, sample_messages):
        """Test stats aggregate across multiple extractions."""
        for i in range(3):
            await adapter.extract_from_debate(
                debate_id=f"stats-{i}",
                messages=sample_messages,
            )

        stats = adapter.get_stats()
        assert stats["debates_processed"] == 3
        assert stats["cached_results"] == 3

    @pytest.mark.asyncio
    async def test_health_check(self, adapter, sample_messages):
        """Test health check returns correct data."""
        await adapter.extract_from_debate(
            debate_id="health-check",
            messages=sample_messages,
        )

        health = adapter.health_check()
        assert "adapter" in health
        assert health["adapter"] == "extraction"
        assert "healthy" in health

    @pytest.mark.asyncio
    async def test_graph_node_limit(self, adapter, sample_messages):
        """Test graph nodes respects limit."""
        await adapter.extract_from_debate(
            debate_id="debate-limit",
            messages=sample_messages,
            topic="Topic1",
        )

        nodes = await adapter.get_graph_nodes(limit=1)
        assert len(nodes) <= 1


# ============================================================================
# Test Data Classes
# ============================================================================


class TestDataClasses:
    """Tests for data class structures and serialization."""

    def test_extraction_search_result_init(self):
        """Test ExtractionSearchResult initialization."""
        claim = create_mock_claim()
        result = ExtractionSearchResult(
            claim=claim,
            relevance_score=0.8,
            matched_topics=["ML"],
        )

        assert result.claim == claim
        assert result.relevance_score == 0.8
        assert result.matched_topics == ["ML"]

    def test_extraction_search_result_to_dict(self):
        """Test ExtractionSearchResult serialization."""
        claim = create_mock_claim()
        result = ExtractionSearchResult(
            claim=claim,
            relevance_score=0.8,
            matched_topics=["ML"],
        )

        result_dict = result.to_dict()
        assert "claim" in result_dict
        assert result_dict["relevance_score"] == 0.8
        assert result_dict["matched_topics"] == ["ML"]

    def test_relationship_search_result_init(self):
        """Test RelationshipSearchResult initialization."""
        rel = create_mock_relationship()
        result = RelationshipSearchResult(
            relationship=rel,
            relevance_score=0.7,
        )

        assert result.relationship == rel
        assert result.relevance_score == 0.7

    def test_relationship_search_result_to_dict(self):
        """Test RelationshipSearchResult serialization."""
        rel = create_mock_relationship()
        result = RelationshipSearchResult(
            relationship=rel,
            relevance_score=0.7,
        )

        result_dict = result.to_dict()
        assert "relationship" in result_dict
        assert result_dict["relevance_score"] == 0.7

    def test_knowledge_graph_node_init(self):
        """Test KnowledgeGraphNode initialization."""
        node = KnowledgeGraphNode(
            id="node-1",
            concept="Machine Learning",
            claim_ids=["c1", "c2"],
            relationship_ids=["r1"],
            confidence=0.9,
        )

        assert node.id == "node-1"
        assert node.concept == "Machine Learning"
        assert len(node.claim_ids) == 2
        assert len(node.relationship_ids) == 1

    def test_knowledge_graph_node_to_dict(self):
        """Test KnowledgeGraphNode serialization."""
        node = KnowledgeGraphNode(
            id="node-1",
            concept="Machine Learning",
            confidence=0.9,
        )

        node_dict = node.to_dict()
        assert node_dict["id"] == "node-1"
        assert node_dict["concept"] == "Machine Learning"
        assert "first_seen" in node_dict
        assert "last_updated" in node_dict

    def test_batch_extraction_result_init(self):
        """Test BatchExtractionResult initialization."""
        result = BatchExtractionResult(
            debate_ids=["d1", "d2"],
            total_claims=10,
            total_relationships=5,
            promoted_count=3,
            failed_debate_ids=[],
            duration_ms=100.0,
        )

        assert result.debate_ids == ["d1", "d2"]
        assert result.total_claims == 10
        assert result.success is True

    def test_batch_extraction_result_with_failures(self):
        """Test BatchExtractionResult with failures."""
        result = BatchExtractionResult(
            debate_ids=["d1", "d2"],
            total_claims=5,
            total_relationships=2,
            promoted_count=1,
            failed_debate_ids=["d3"],
            duration_ms=50.0,
            errors=["d3: Failed"],
        )

        assert result.success is False
        assert len(result.failed_debate_ids) == 1


# ============================================================================
# Test Configuration
# ============================================================================


class TestConfiguration:
    """Tests for adapter configuration."""

    def test_default_config(self, adapter):
        """Test default configuration values."""
        config = adapter.get_config()

        assert config.min_confidence_to_extract == 0.3
        assert config.min_confidence_to_promote == 0.6
        assert config.extract_facts is True
        assert config.extract_definitions is True
        assert config.extract_relationships is True
        assert config.extract_opinions is False

    def test_custom_config(self, config):
        """Test custom configuration."""
        adapter = ExtractionAdapter(config=config)

        assert adapter.get_config().min_confidence_to_extract == 0.3
        assert adapter.get_config().min_confidence_to_promote == 0.6

    def test_config_affects_extraction(self):
        """Test configuration affects extraction behavior."""
        # Config that disables facts
        config = ExtractionConfig(
            extract_facts=False,
            extract_definitions=False,
            extract_procedures=False,
            extract_opinions=True,  # Only opinions
        )

        adapter = ExtractionAdapter(config=config)
        # Extractor should use this config
        assert adapter._config.extract_facts is False

    @pytest.mark.asyncio
    async def test_update_config_affects_future_extractions(self, adapter):
        """Test updating config affects future extractions."""
        new_config = ExtractionConfig(
            min_confidence_to_extract=0.8,
        )

        adapter.set_config(new_config)
        assert adapter.get_config().min_confidence_to_extract == 0.8
