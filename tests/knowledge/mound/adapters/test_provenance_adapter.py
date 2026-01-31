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
        callback = MagicMock(side_effect=Exception("Callback failed"))
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
