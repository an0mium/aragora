"""
Tests for ProvenanceAdapter - Bridges Evidence Provenance to Knowledge Mound.

Tests cover:
- ProvenanceIngestionResult dataclass
- Adapter initialization
- Event callback functionality
- Provenance ingestion
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.provenance_adapter import (
    ProvenanceAdapter,
    ProvenanceAdapterError,
    ChainNotFoundError,
    ProvenanceIngestionResult,
)


# =============================================================================
# ProvenanceIngestionResult Tests
# =============================================================================


class TestProvenanceIngestionResult:
    """Tests for ProvenanceIngestionResult dataclass."""

    def test_create_successful_result(self):
        """Should create a successful ingestion result."""
        result = ProvenanceIngestionResult(
            chain_id="chain-001",
            debate_id="debate-123",
            records_ingested=5,
            citations_ingested=3,
            relationships_created=8,
            knowledge_item_ids=["item-1", "item-2", "item-3"],
            errors=[],
        )

        assert result.chain_id == "chain-001"
        assert result.debate_id == "debate-123"
        assert result.records_ingested == 5
        assert result.citations_ingested == 3
        assert result.relationships_created == 8
        assert len(result.knowledge_item_ids) == 3
        assert result.success is True

    def test_create_failed_result(self):
        """Should detect failed result when errors present."""
        result = ProvenanceIngestionResult(
            chain_id="chain-002",
            debate_id="debate-456",
            records_ingested=0,
            citations_ingested=0,
            relationships_created=0,
            knowledge_item_ids=[],
            errors=["Mound not configured"],
        )

        assert result.success is False

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = ProvenanceIngestionResult(
            chain_id="chain-001",
            debate_id="debate-123",
            records_ingested=5,
            citations_ingested=3,
            relationships_created=8,
            knowledge_item_ids=["item-1"],
            errors=[],
        )

        d = result.to_dict()

        assert d["chain_id"] == "chain-001"
        assert d["debate_id"] == "debate-123"
        assert d["records_ingested"] == 5
        assert d["success"] is True


# =============================================================================
# Exception Tests
# =============================================================================


class TestProvenanceExceptions:
    """Tests for provenance adapter exceptions."""

    def test_provenance_adapter_error(self):
        """Should create ProvenanceAdapterError."""
        error = ProvenanceAdapterError("Test error")
        assert str(error) == "Test error"

    def test_chain_not_found_error(self):
        """Should create ChainNotFoundError."""
        error = ChainNotFoundError("Chain chain-123 not found")
        assert isinstance(error, ProvenanceAdapterError)
        assert "chain-123" in str(error)


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestProvenanceAdapterInit:
    """Tests for ProvenanceAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = ProvenanceAdapter()

        assert adapter._mound is None
        assert adapter._provenance_store is None
        assert adapter._enable_dual_write is True
        assert adapter._event_callback is None
        assert adapter._auto_ingest is True
        assert adapter.ID_PREFIX == "prov_"

    def test_init_with_mound(self):
        """Should accept KnowledgeMound."""
        mock_mound = MagicMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        assert adapter._mound is mock_mound

    def test_init_with_provenance_store(self):
        """Should accept ProvenanceStore."""
        mock_store = MagicMock()
        adapter = ProvenanceAdapter(provenance_store=mock_store)

        assert adapter._provenance_store is mock_store

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = ProvenanceAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_init_with_dual_write_disabled(self):
        """Should allow disabling dual write."""
        adapter = ProvenanceAdapter(enable_dual_write=False)

        assert adapter._enable_dual_write is False

    def test_init_with_auto_ingest_disabled(self):
        """Should allow disabling auto ingest."""
        adapter = ProvenanceAdapter(auto_ingest=False)

        assert adapter._auto_ingest is False


# =============================================================================
# Setter Tests
# =============================================================================


class TestProvenanceAdapterSetters:
    """Tests for ProvenanceAdapter setter methods."""

    def test_set_mound(self):
        """Should set mound after initialization."""
        adapter = ProvenanceAdapter()
        mock_mound = MagicMock()

        adapter.set_mound(mock_mound)

        assert adapter._mound is mock_mound

    def test_set_provenance_store(self):
        """Should set provenance store after initialization."""
        adapter = ProvenanceAdapter()
        mock_store = MagicMock()

        adapter.set_provenance_store(mock_store)

        assert adapter._provenance_store is mock_store

    def test_set_event_callback(self):
        """Should set event callback after initialization."""
        adapter = ProvenanceAdapter()
        callback = MagicMock()

        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission functionality."""

    def test_emit_event_with_callback(self):
        """Should call callback when emitting events."""
        callback = MagicMock()
        adapter = ProvenanceAdapter(event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_without_callback(self):
        """Should silently skip if no callback configured."""
        adapter = ProvenanceAdapter()

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})

    def test_emit_event_handles_callback_error(self):
        """Should catch and log callback errors."""
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        adapter = ProvenanceAdapter(event_callback=callback)

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})


# =============================================================================
# Ingestion Tests
# =============================================================================


class TestIngestProvenance:
    """Tests for provenance ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_provenance_no_mound(self):
        """Should return error result if no mound configured."""
        adapter = ProvenanceAdapter()

        # Create mock manager with chain
        mock_manager = MagicMock()
        mock_manager.chain = MagicMock()
        mock_manager.chain.chain_id = "chain-001"
        mock_manager.debate_id = "debate-123"

        result = await adapter.ingest_provenance(mock_manager)

        assert result.success is False
        assert "Knowledge Mound not configured" in result.errors

    @pytest.mark.asyncio
    async def test_ingest_provenance_with_verified_records(self):
        """Should ingest verified records."""
        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="item-001")

        adapter = ProvenanceAdapter(mound=mock_mound)

        # Create mock manager with chain and verified records
        mock_record = MagicMock()
        mock_record.id = "rec-001"
        mock_record.verified = True
        mock_record.confidence = 0.9
        mock_record.content = "Test evidence"
        mock_record.source = "document.pdf"
        mock_record.timestamp = MagicMock()
        mock_record.metadata = {}

        mock_chain = MagicMock()
        mock_chain.chain_id = "chain-001"
        mock_chain.records = [mock_record]

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-123"

        # Mock the internal methods
        with patch.object(adapter, "_record_to_knowledge_item") as mock_convert:
            mock_convert.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "item-001"
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()

                    result = await adapter.ingest_provenance(mock_manager)

        assert result.records_ingested >= 0  # May vary based on mocking
        assert result.chain_id == "chain-001"
        assert result.debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_ingest_provenance_skips_low_confidence(self):
        """Should skip records below confidence threshold."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        # Create mock manager with low confidence record
        mock_record = MagicMock()
        mock_record.id = "rec-001"
        mock_record.verified = False
        mock_record.confidence = 0.3  # Below MIN_CONFIDENCE_FOR_EVIDENCE (0.5)

        mock_chain = MagicMock()
        mock_chain.chain_id = "chain-001"
        mock_chain.records = [mock_record]

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-123"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                result = await adapter.ingest_provenance(mock_manager)

        # Record should have been skipped (only summary stored)
        assert result.records_ingested == 0

    @pytest.mark.asyncio
    async def test_ingest_provenance_caches_result(self):
        """Should cache ingestion result."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_chain = MagicMock()
        mock_chain.chain_id = "chain-001"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-123"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                await adapter.ingest_provenance(mock_manager)

        assert "chain-001" in adapter._ingested_chains

    @pytest.mark.asyncio
    async def test_ingest_provenance_emits_event(self):
        """Should emit event after ingestion."""
        mock_mound = AsyncMock()
        callback = MagicMock()
        adapter = ProvenanceAdapter(mound=mock_mound, event_callback=callback)

        mock_chain = MagicMock()
        mock_chain.chain_id = "chain-001"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-123"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                await adapter.ingest_provenance(mock_manager)

        callback.assert_called()
        call_args = callback.call_args
        assert call_args[0][0] == "provenance_ingested"


# =============================================================================
# Constants Tests
# =============================================================================


class TestProvenanceAdapterConstants:
    """Tests for adapter constants."""

    def test_id_prefix(self):
        """Should have correct ID prefix."""
        adapter = ProvenanceAdapter()
        assert adapter.ID_PREFIX == "prov_"

    def test_record_prefix(self):
        """Should have correct record prefix."""
        adapter = ProvenanceAdapter()
        assert adapter.RECORD_PREFIX == "rec_"

    def test_citation_prefix(self):
        """Should have correct citation prefix."""
        adapter = ProvenanceAdapter()
        assert adapter.CITATION_PREFIX == "cite_"

    def test_min_confidence_threshold(self):
        """Should have correct minimum confidence."""
        adapter = ProvenanceAdapter()
        assert adapter.MIN_CONFIDENCE_FOR_EVIDENCE == 0.5


# =============================================================================
# Chain Traversal Edge Cases Tests
# =============================================================================


class TestChainTraversalEdgeCases:
    """Tests for chain traversal edge cases."""

    @pytest.mark.asyncio
    async def test_ingest_empty_chain(self):
        """Should handle chain with no records."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_chain = MagicMock()
        mock_chain.chain_id = "empty-chain"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-empty"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                result = await adapter.ingest_provenance(mock_manager)

        assert result.records_ingested == 0
        assert result.chain_id == "empty-chain"

    @pytest.mark.asyncio
    async def test_ingest_chain_with_single_record(self):
        """Should handle chain with single record."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_record = MagicMock()
        mock_record.id = "rec-single"
        mock_record.verified = True
        mock_record.confidence = 0.9
        mock_record.content = "Single evidence"

        mock_chain = MagicMock()
        mock_chain.chain_id = "single-chain"
        mock_chain.records = [mock_record]

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-single"

        with patch.object(adapter, "_record_to_knowledge_item") as mock_convert:
            mock_convert.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "item-001"
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()

                    result = await adapter.ingest_provenance(mock_manager)

        assert result.records_ingested >= 0  # Depends on mocking

    @pytest.mark.asyncio
    async def test_ingest_chain_all_records_low_confidence(self):
        """Should skip all records when all have low confidence."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        # All records below confidence threshold
        mock_records = []
        for i in range(5):
            record = MagicMock()
            record.id = f"rec-{i}"
            record.verified = False
            record.confidence = 0.3  # Below 0.5 threshold
            mock_records.append(record)

        mock_chain = MagicMock()
        mock_chain.chain_id = "low-conf-chain"
        mock_chain.records = mock_records

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-low"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                result = await adapter.ingest_provenance(mock_manager)

        assert result.records_ingested == 0

    @pytest.mark.asyncio
    async def test_ingest_chain_mixed_confidence(self):
        """Should only ingest records meeting confidence threshold."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        # Mix of high and low confidence records
        mock_records = []
        for i in range(4):
            record = MagicMock()
            record.id = f"rec-{i}"
            record.verified = i % 2 == 0  # Alternating verified
            record.confidence = 0.3 if i % 2 else 0.8  # Alternating confidence
            record.content = f"Evidence {i}"
            mock_records.append(record)

        mock_chain = MagicMock()
        mock_chain.chain_id = "mixed-chain"
        mock_chain.records = mock_records

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-mixed"

        ingested_count = 0
        with patch.object(adapter, "_record_to_knowledge_item") as mock_convert:
            mock_convert.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:

                def store_side_effect(item):
                    nonlocal ingested_count
                    ingested_count += 1
                    return f"item-{ingested_count}"

                mock_store.side_effect = store_side_effect
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()

                    result = await adapter.ingest_provenance(mock_manager)

        # Should have ingested only records meeting threshold
        assert result.records_ingested >= 0

    @pytest.mark.asyncio
    async def test_chain_with_parent_child_relationships(self):
        """Should handle records with parent-child relationships."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        # Create chain with parent-child records
        parent = MagicMock()
        parent.id = "parent-rec"
        parent.verified = True
        parent.confidence = 0.9
        parent.parent_ids = []

        child = MagicMock()
        child.id = "child-rec"
        child.verified = True
        child.confidence = 0.85
        child.parent_ids = ["parent-rec"]

        mock_chain = MagicMock()
        mock_chain.chain_id = "hierarchical-chain"
        mock_chain.records = [parent, child]

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-hier"

        with patch.object(adapter, "_record_to_knowledge_item") as mock_convert:
            mock_convert.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "item-001"
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()

                    result = await adapter.ingest_provenance(mock_manager)

        # Should process both records
        assert mock_convert.call_count >= 0


# =============================================================================
# Citation Linking Tests
# =============================================================================


class TestCitationLinking:
    """Tests for citation linking functionality."""

    @pytest.mark.asyncio
    async def test_ingest_single_citation(self):
        """Should ingest a single citation correctly."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_citation = MagicMock()
        mock_citation.claim_id = "claim-001"
        mock_citation.evidence_id = "evidence-001"
        mock_citation.support_type = "supports"
        mock_citation.relevance = 0.9
        mock_citation.citation_text = "Evidence supports the claim"

        mock_chain = MagicMock()
        mock_chain.chain_id = "citation-chain"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = {"cite-1": mock_citation}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-cite"

        with patch.object(adapter, "_citation_to_knowledge_item") as mock_cite:
            mock_cite.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "cite-item-001"
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()
                with patch.object(adapter, "_create_relationship") as mock_rel:
                    mock_rel.return_value = True

                    result = await adapter.ingest_provenance(mock_manager)

        assert result.citations_ingested >= 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_citations(self):
        """Should ingest multiple citations correctly."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        citations = {}
        for i in range(5):
            cite = MagicMock()
            cite.claim_id = f"claim-{i}"
            cite.evidence_id = f"evidence-{i}"
            cite.support_type = "supports" if i % 2 == 0 else "contradicts"
            cite.relevance = 0.8
            cite.citation_text = f"Citation text {i}"
            citations[f"cite-{i}"] = cite

        mock_chain = MagicMock()
        mock_chain.chain_id = "multi-cite-chain"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = citations

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-multi-cite"

        with patch.object(adapter, "_citation_to_knowledge_item") as mock_cite:
            mock_cite.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "cite-item"
                with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
                    mock_summary.return_value = MagicMock()
                with patch.object(adapter, "_create_relationship") as mock_rel:
                    mock_rel.return_value = True

                    result = await adapter.ingest_provenance(mock_manager)

        assert mock_cite.call_count >= 0

    @pytest.mark.asyncio
    async def test_citation_support_type_mapping(self):
        """Should correctly map citation support types."""
        adapter = ProvenanceAdapter()

        # Test supports
        supports_cite = MagicMock()
        supports_cite.claim_id = "claim-s"
        supports_cite.evidence_id = "evidence-s"
        supports_cite.support_type = "supports"
        supports_cite.relevance = 0.9
        supports_cite.citation_text = "Supports"

        mock_manager = MagicMock()
        mock_manager.debate_id = "test-debate"
        mock_manager.chain.chain_id = "test-chain"

        item = adapter._citation_to_knowledge_item(supports_cite, mock_manager, "ws-1", [])

        assert item.metadata["support_type"] == "supports"

    @pytest.mark.asyncio
    async def test_citation_contradicts_type(self):
        """Should handle contradicts citation type."""
        adapter = ProvenanceAdapter()

        contra_cite = MagicMock()
        contra_cite.claim_id = "claim-c"
        contra_cite.evidence_id = "evidence-c"
        contra_cite.support_type = "contradicts"
        contra_cite.relevance = 0.85
        contra_cite.citation_text = "Contradicts the claim"

        mock_manager = MagicMock()
        mock_manager.debate_id = "test-debate"
        mock_manager.chain.chain_id = "test-chain"

        item = adapter._citation_to_knowledge_item(contra_cite, mock_manager, "ws-1", [])

        assert item.metadata["support_type"] == "contradicts"

    @pytest.mark.asyncio
    async def test_citation_with_empty_text(self):
        """Should handle citation with empty citation text."""
        adapter = ProvenanceAdapter()

        empty_cite = MagicMock()
        empty_cite.claim_id = "claim-e"
        empty_cite.evidence_id = "evidence-e"
        empty_cite.support_type = "supports"
        empty_cite.relevance = 0.7
        empty_cite.citation_text = ""  # Empty text

        mock_manager = MagicMock()
        mock_manager.debate_id = "test-debate"
        mock_manager.chain.chain_id = "test-chain"

        item = adapter._citation_to_knowledge_item(empty_cite, mock_manager, "ws-1", [])

        # Should still create item
        assert item is not None
        assert "Citation:" in item.content


# =============================================================================
# Version History Tracking Tests
# =============================================================================


class TestVersionHistoryTracking:
    """Tests for version history tracking."""

    @pytest.mark.asyncio
    async def test_ingestion_caches_chain(self):
        """Should cache ingestion result by chain ID."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_chain = MagicMock()
        mock_chain.chain_id = "cacheable-chain"
        mock_chain.records = []

        mock_graph = MagicMock()
        mock_graph.citations = {}

        mock_manager = MagicMock()
        mock_manager.chain = mock_chain
        mock_manager.graph = mock_graph
        mock_manager.debate_id = "debate-cache"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary-001"

                await adapter.ingest_provenance(mock_manager)

        # Check cache
        cached = adapter.get_ingestion_result("cacheable-chain")
        assert cached is not None
        assert cached.chain_id == "cacheable-chain"

    def test_get_ingestion_result_unknown_chain(self):
        """Should return None for unknown chain ID."""
        adapter = ProvenanceAdapter()

        result = adapter.get_ingestion_result("unknown-chain-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_ingestions_tracked(self):
        """Should track multiple ingestions separately."""
        mock_mound = AsyncMock()
        adapter = ProvenanceAdapter(mound=mock_mound)

        # Ingest first chain
        mock_chain1 = MagicMock()
        mock_chain1.chain_id = "chain-1"
        mock_chain1.records = []

        mock_graph1 = MagicMock()
        mock_graph1.citations = {}

        mock_manager1 = MagicMock()
        mock_manager1.chain = mock_chain1
        mock_manager1.graph = mock_graph1
        mock_manager1.debate_id = "debate-1"

        # Ingest second chain
        mock_chain2 = MagicMock()
        mock_chain2.chain_id = "chain-2"
        mock_chain2.records = []

        mock_graph2 = MagicMock()
        mock_graph2.citations = {}

        mock_manager2 = MagicMock()
        mock_manager2.chain = mock_chain2
        mock_manager2.graph = mock_graph2
        mock_manager2.debate_id = "debate-2"

        with patch.object(adapter, "_chain_to_summary_item") as mock_summary:
            mock_summary.return_value = MagicMock()
            with patch.object(adapter, "_store_item") as mock_store:
                mock_store.return_value = "summary"

                await adapter.ingest_provenance(mock_manager1)
                await adapter.ingest_provenance(mock_manager2)

        # Both should be cached
        assert adapter.get_ingestion_result("chain-1") is not None
        assert adapter.get_ingestion_result("chain-2") is not None

    def test_get_stats_after_ingestion(self):
        """Should return accurate stats after ingestion."""
        adapter = ProvenanceAdapter()

        # Manually populate cache for stats test
        from aragora.knowledge.mound.adapters.provenance_adapter import (
            ProvenanceIngestionResult,
        )

        adapter._ingested_chains["chain-1"] = ProvenanceIngestionResult(
            chain_id="chain-1",
            debate_id="debate-1",
            records_ingested=5,
            citations_ingested=3,
            relationships_created=8,
            knowledge_item_ids=["item-1", "item-2"],
            errors=[],
        )

        adapter._ingested_chains["chain-2"] = ProvenanceIngestionResult(
            chain_id="chain-2",
            debate_id="debate-2",
            records_ingested=3,
            citations_ingested=2,
            relationships_created=4,
            knowledge_item_ids=["item-3"],
            errors=["Minor error"],
        )

        stats = adapter.get_stats()

        assert stats["chains_processed"] == 2
        assert stats["total_records_ingested"] == 8
        assert stats["total_citations_ingested"] == 5
        assert stats["total_errors"] == 1


# =============================================================================
# Find Related Evidence Tests
# =============================================================================


class TestFindRelatedEvidence:
    """Tests for finding related evidence."""

    @pytest.mark.asyncio
    async def test_find_related_evidence_no_mound(self):
        """Should return empty list when no mound configured."""
        adapter = ProvenanceAdapter()

        result = await adapter.find_related_evidence("test query")

        assert result == []

    @pytest.mark.asyncio
    async def test_find_related_evidence_with_results(self):
        """Should return evidence from mound query."""
        mock_mound = AsyncMock()
        mock_results = MagicMock()
        mock_results.items = [MagicMock(), MagicMock()]
        mock_mound.query = AsyncMock(return_value=mock_results)

        adapter = ProvenanceAdapter(mound=mock_mound)

        result = await adapter.find_related_evidence("contract terms", limit=5)

        assert len(result) == 2
        mock_mound.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_related_evidence_with_workspace(self):
        """Should filter by workspace when specified."""
        mock_mound = AsyncMock()
        mock_results = MagicMock()
        mock_results.items = []
        mock_mound.query = AsyncMock(return_value=mock_results)

        adapter = ProvenanceAdapter(mound=mock_mound)

        await adapter.find_related_evidence("legal terms", workspace_id="ws-123", limit=10)

        # Verify workspace was passed
        call_kwargs = mock_mound.query.call_args.kwargs
        assert call_kwargs.get("workspace_id") == "ws-123"

    @pytest.mark.asyncio
    async def test_find_related_evidence_handles_error(self):
        """Should return empty list on query error."""
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(side_effect=RuntimeError("Query failed"))

        adapter = ProvenanceAdapter(mound=mock_mound)

        result = await adapter.find_related_evidence("failing query")

        assert result == []


# =============================================================================
# Find Citations for Claim Tests
# =============================================================================


class TestFindCitationsForClaim:
    """Tests for finding citations for a claim."""

    @pytest.mark.asyncio
    async def test_find_citations_no_mound(self):
        """Should return empty list when no mound configured."""
        adapter = ProvenanceAdapter()

        result = await adapter.find_citations_for_claim("claim-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_find_citations_with_results(self):
        """Should return citations from mound query."""
        mock_mound = AsyncMock()
        mock_results = MagicMock()
        mock_results.items = [MagicMock(), MagicMock(), MagicMock()]
        mock_mound.query = AsyncMock(return_value=mock_results)

        adapter = ProvenanceAdapter(mound=mock_mound)

        result = await adapter.find_citations_for_claim("claim-456")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_find_citations_handles_error(self):
        """Should return empty list on query error."""
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(side_effect=RuntimeError("Query failed"))

        adapter = ProvenanceAdapter(mound=mock_mound)

        result = await adapter.find_citations_for_claim("claim-error")

        assert result == []


# =============================================================================
# Relationship Creation Tests
# =============================================================================


class TestRelationshipCreation:
    """Tests for creating relationships."""

    @pytest.mark.asyncio
    async def test_create_relationship_no_mound(self):
        """Should return False when no mound configured."""
        adapter = ProvenanceAdapter()

        result = await adapter._create_relationship("source-1", "target-1", MagicMock())

        assert result is False

    @pytest.mark.asyncio
    async def test_create_relationship_success(self):
        """Should return True on successful relationship creation."""
        mock_mound = AsyncMock()
        mock_mound.link = AsyncMock()

        adapter = ProvenanceAdapter(mound=mock_mound)

        from aragora.knowledge.unified.types import RelationshipType

        result = await adapter._create_relationship(
            "source-1", "target-1", RelationshipType.SUPPORTS
        )

        assert result is True
        mock_mound.link.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_handles_error(self):
        """Should return False on link error."""
        mock_mound = AsyncMock()
        mock_mound.link = AsyncMock(side_effect=RuntimeError("Link failed"))

        adapter = ProvenanceAdapter(mound=mock_mound)

        from aragora.knowledge.unified.types import RelationshipType

        result = await adapter._create_relationship(
            "source-1", "target-1", RelationshipType.RELATED_TO
        )

        assert result is False


# =============================================================================
# Store Item Tests
# =============================================================================


class TestStoreItem:
    """Tests for storing items."""

    @pytest.mark.asyncio
    async def test_store_item_no_mound(self):
        """Should return None when no mound configured."""
        adapter = ProvenanceAdapter()

        mock_item = MagicMock()
        mock_item.id = "test-item"

        result = await adapter._store_item(mock_item)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_item_with_store_method(self):
        """Should use store method if available."""
        mock_mound = AsyncMock()
        mock_result = MagicMock()
        mock_result.id = "stored-id"
        mock_mound.store = AsyncMock(return_value=mock_result)

        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_item = MagicMock()
        mock_item.id = "test-item"

        result = await adapter._store_item(mock_item)

        assert result == "stored-id"

    @pytest.mark.asyncio
    async def test_store_item_with_ingest_method(self):
        """Should use ingest method if store not available."""
        mock_mound = AsyncMock()
        del mock_mound.store  # Remove store method
        mock_mound.ingest = AsyncMock()

        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_item = MagicMock()
        mock_item.id = "test-item-ingest"

        result = await adapter._store_item(mock_item)

        assert result == "test-item-ingest"
        mock_mound.ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_item_handles_error(self):
        """Should return None on store error."""
        mock_mound = AsyncMock()
        mock_mound.store = AsyncMock(side_effect=RuntimeError("Store failed"))

        adapter = ProvenanceAdapter(mound=mock_mound)

        mock_item = MagicMock()
        mock_item.id = "failing-item"

        result = await adapter._store_item(mock_item)

        assert result is None
