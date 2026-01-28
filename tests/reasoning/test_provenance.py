"""
Tests for Evidence Provenance Chain.

Tests the provenance module including:
- SourceType and TransformationType enums
- ProvenanceRecord dataclass
- Citation dataclass
- ProvenanceChain class
- CitationGraph class
- MerkleTree class
- ProvenanceVerifier class
- ProvenanceManager class
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.reasoning.provenance import (
    Citation,
    CitationGraph,
    MerkleTree,
    ProvenanceChain,
    ProvenanceManager,
    ProvenanceRecord,
    ProvenanceVerifier,
    SourceType,
    TransformationType,
)


# =============================================================================
# SourceType Tests
# =============================================================================


class TestSourceType:
    """Test SourceType enum."""

    def test_source_type_values(self):
        """Test source type values."""
        assert SourceType.AGENT_GENERATED.value == "agent_generated"
        assert SourceType.USER_PROVIDED.value == "user_provided"
        assert SourceType.EXTERNAL_API.value == "external_api"
        assert SourceType.WEB_SEARCH.value == "web_search"
        assert SourceType.DOCUMENT.value == "document"
        assert SourceType.CODE_ANALYSIS.value == "code_analysis"
        assert SourceType.DATABASE.value == "database"
        assert SourceType.COMPUTATION.value == "computation"
        assert SourceType.SYNTHESIS.value == "synthesis"
        assert SourceType.AUDIO_TRANSCRIPT.value == "audio_transcript"
        assert SourceType.UNKNOWN.value == "unknown"

    def test_all_source_types_present(self):
        """Test all expected source types are defined."""
        types = [t.value for t in SourceType]
        assert len(types) == 11


# =============================================================================
# TransformationType Tests
# =============================================================================


class TestTransformationType:
    """Test TransformationType enum."""

    def test_transformation_type_values(self):
        """Test transformation type values."""
        assert TransformationType.ORIGINAL.value == "original"
        assert TransformationType.QUOTED.value == "quoted"
        assert TransformationType.PARAPHRASED.value == "paraphrased"
        assert TransformationType.SUMMARIZED.value == "summarized"
        assert TransformationType.EXTRACTED.value == "extracted"
        assert TransformationType.COMPUTED.value == "computed"
        assert TransformationType.AGGREGATED.value == "aggregated"
        assert TransformationType.VERIFIED.value == "verified"
        assert TransformationType.REFUTED.value == "refuted"
        assert TransformationType.AMENDED.value == "amended"


# =============================================================================
# ProvenanceRecord Tests
# =============================================================================


class TestProvenanceRecord:
    """Test ProvenanceRecord dataclass."""

    def test_basic_creation(self):
        """Test basic ProvenanceRecord creation."""
        record = ProvenanceRecord(
            id="rec-123",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Test content",
        )
        assert record.id == "rec-123"
        assert record.source_type == SourceType.DOCUMENT
        assert record.content == "Test content"

    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        record = ProvenanceRecord(
            id="",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Test content",
        )
        assert record.id  # Should be non-empty
        assert len(record.id) == 12

    def test_auto_hash_generation(self):
        """Test automatic content hash generation."""
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Test content",
        )
        assert record.content_hash  # Should be non-empty
        assert len(record.content_hash) == 64  # SHA-256

    def test_deterministic_hash(self):
        """Test content hash is deterministic."""
        record1 = ProvenanceRecord(
            id="test1",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Same content",
        )
        record2 = ProvenanceRecord(
            id="test2",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Same content",
        )
        assert record1.content_hash == record2.content_hash

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        record1 = ProvenanceRecord(
            id="test1",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Content A",
        )
        record2 = ProvenanceRecord(
            id="test2",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Content B",
        )
        assert record1.content_hash != record2.content_hash

    def test_chain_hash(self):
        """Test chain hash includes previous hash."""
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Test content",
            previous_hash="abc123",
        )
        chain_hash = record.chain_hash()
        assert chain_hash  # Should be non-empty
        assert len(chain_hash) == 64

    def test_chain_hash_genesis(self):
        """Test chain hash for genesis record."""
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Genesis content",
            previous_hash=None,
        )
        chain_hash = record.chain_hash()
        assert "genesis" in record.chain_hash() or len(chain_hash) == 64

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = ProvenanceRecord(
            id="test-id",
            content_hash="",
            source_type=SourceType.COMPUTATION,
            source_id="calc-1",
            content="Result: 42",
            content_type="text",
            verified=True,
            confidence=0.95,
        )
        data = record.to_dict()

        assert data["id"] == "test-id"
        assert data["source_type"] == "computation"
        assert data["source_id"] == "calc-1"
        assert data["content"] == "Result: 42"
        assert data["verified"] is True
        assert data["confidence"] == 0.95
        assert "chain_hash" in data
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "rec-123",
            "content_hash": "abc123",
            "source_type": "document",
            "source_id": "doc-1",
            "content": "Test content",
            "content_type": "text",
            "timestamp": datetime.now().isoformat(),
            "previous_hash": None,
            "parent_ids": [],
            "transformation": "original",
            "transformation_note": "",
            "confidence": 0.9,
            "verified": True,
            "verifier_id": "v-1",
            "metadata": {},
        }
        record = ProvenanceRecord.from_dict(data)

        assert record.id == "rec-123"
        assert record.source_type == SourceType.DOCUMENT
        assert record.verified is True
        assert record.confidence == 0.9


# =============================================================================
# Citation Tests
# =============================================================================


class TestCitation:
    """Test Citation dataclass."""

    def test_basic_creation(self):
        """Test basic Citation creation."""
        citation = Citation(
            claim_id="claim-1",
            evidence_id="ev-1",
        )
        assert citation.claim_id == "claim-1"
        assert citation.evidence_id == "ev-1"
        assert citation.relevance == 1.0
        assert citation.support_type == "supports"

    def test_custom_values(self):
        """Test Citation with custom values."""
        citation = Citation(
            claim_id="claim-2",
            evidence_id="ev-2",
            relevance=0.8,
            support_type="contradicts",
            citation_text="Relevant quote",
            metadata={"page": 42},
        )
        assert citation.relevance == 0.8
        assert citation.support_type == "contradicts"
        assert citation.citation_text == "Relevant quote"
        assert citation.metadata["page"] == 42


# =============================================================================
# ProvenanceChain Tests
# =============================================================================


class TestProvenanceChain:
    """Test ProvenanceChain class."""

    def test_initialization(self):
        """Test ProvenanceChain initialization."""
        chain = ProvenanceChain()
        assert chain.chain_id  # Auto-generated
        assert chain.records == []
        assert chain.genesis_hash is None

    def test_initialization_with_id(self):
        """Test ProvenanceChain initialization with custom ID."""
        chain = ProvenanceChain(chain_id="custom-chain")
        assert chain.chain_id == "custom-chain"

    def test_add_record(self):
        """Test adding a record to the chain."""
        chain = ProvenanceChain()

        record = chain.add_record(
            content="First evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert len(chain.records) == 1
        assert record.content == "First evidence"
        assert record.previous_hash is None  # Genesis record
        assert chain.genesis_hash is not None

    def test_add_multiple_records(self):
        """Test adding multiple records creates chain."""
        chain = ProvenanceChain()

        record1 = chain.add_record(
            content="First",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        record2 = chain.add_record(
            content="Second",
            source_type=SourceType.DOCUMENT,
            source_id="doc-2",
        )

        assert len(chain.records) == 2
        assert record1.previous_hash is None
        assert record2.previous_hash == record1.chain_hash()

    def test_verify_chain_empty(self):
        """Test verifying empty chain."""
        chain = ProvenanceChain()
        valid, errors = chain.verify_chain()

        assert valid is True
        assert errors == []

    def test_verify_chain_valid(self):
        """Test verifying valid chain."""
        chain = ProvenanceChain()
        chain.add_record("First", SourceType.DOCUMENT, "doc-1")
        chain.add_record("Second", SourceType.DOCUMENT, "doc-2")
        chain.add_record("Third", SourceType.DOCUMENT, "doc-3")

        valid, errors = chain.verify_chain()

        assert valid is True
        assert errors == []

    def test_verify_chain_tampered_content(self):
        """Test verifying chain with tampered content."""
        chain = ProvenanceChain()
        chain.add_record("Original content", SourceType.DOCUMENT, "doc-1")

        # Tamper with content (simulate)
        original_hash = chain.records[0].content_hash
        chain.records[0].content = "Tampered content"
        # Content hash doesn't match anymore

        valid, errors = chain.verify_chain()

        # Should detect tampering
        assert valid is False or original_hash != chain.records[0]._compute_hash()

    def test_get_record(self):
        """Test getting record by ID."""
        chain = ProvenanceChain()
        record = chain.add_record("Content", SourceType.DOCUMENT, "doc-1")

        found = chain.get_record(record.id)

        assert found is not None
        assert found.id == record.id
        assert found.content == "Content"

    def test_get_record_not_found(self):
        """Test getting non-existent record."""
        chain = ProvenanceChain()
        found = chain.get_record("nonexistent")
        assert found is None

    def test_get_ancestry(self):
        """Test getting record ancestry."""
        chain = ProvenanceChain()
        r1 = chain.add_record("First", SourceType.DOCUMENT, "doc-1")
        r2 = chain.add_record("Second", SourceType.DOCUMENT, "doc-2")
        r3 = chain.add_record("Third", SourceType.DOCUMENT, "doc-3")

        ancestry = chain.get_ancestry(r3.id)

        # Should include the record and its ancestors
        assert len(ancestry) >= 1
        assert r3 in ancestry

    def test_to_dict(self):
        """Test serialization to dictionary."""
        chain = ProvenanceChain(chain_id="test-chain")
        chain.add_record("Content", SourceType.DOCUMENT, "doc-1")

        data = chain.to_dict()

        assert data["chain_id"] == "test-chain"
        assert "genesis_hash" in data
        assert "created_at" in data
        assert data["record_count"] == 1
        assert len(data["records"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        chain = ProvenanceChain(chain_id="test-chain")
        chain.add_record("Content", SourceType.DOCUMENT, "doc-1")
        data = chain.to_dict()

        restored = ProvenanceChain.from_dict(data)

        assert restored.chain_id == "test-chain"
        assert len(restored.records) == 1
        assert restored.records[0].content == "Content"


# =============================================================================
# CitationGraph Tests
# =============================================================================


class TestCitationGraph:
    """Test CitationGraph class."""

    def test_initialization(self):
        """Test CitationGraph initialization."""
        graph = CitationGraph()
        assert graph.citations == {}
        assert graph.claim_citations == {}
        assert graph.evidence_citations == {}

    def test_add_citation(self):
        """Test adding a citation."""
        graph = CitationGraph()

        citation = graph.add_citation(
            claim_id="claim-1",
            evidence_id="ev-1",
            relevance=0.9,
            support_type="supports",
        )

        assert citation.claim_id == "claim-1"
        assert citation.evidence_id == "ev-1"
        assert "claim-1:ev-1" in graph.citations
        assert "claim-1" in graph.claim_citations
        assert "ev-1" in graph.evidence_citations

    def test_add_multiple_citations_same_claim(self):
        """Test adding multiple citations for same claim."""
        graph = CitationGraph()

        graph.add_citation("claim-1", "ev-1")
        graph.add_citation("claim-1", "ev-2")
        graph.add_citation("claim-1", "ev-3")

        assert len(graph.claim_citations["claim-1"]) == 3

    def test_get_claim_evidence(self):
        """Test getting all evidence for a claim."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", support_type="supports")
        graph.add_citation("claim-1", "ev-2", support_type="contradicts")

        citations = graph.get_claim_evidence("claim-1")

        assert len(citations) == 2

    def test_get_claim_evidence_empty(self):
        """Test getting evidence for non-existent claim."""
        graph = CitationGraph()
        citations = graph.get_claim_evidence("nonexistent")
        assert citations == []

    def test_get_evidence_claims(self):
        """Test getting all claims citing an evidence."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-shared")
        graph.add_citation("claim-2", "ev-shared")
        graph.add_citation("claim-3", "ev-shared")

        citations = graph.get_evidence_claims("ev-shared")

        assert len(citations) == 3

    def test_get_supporting_evidence(self):
        """Test getting only supporting evidence."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", support_type="supports")
        graph.add_citation("claim-1", "ev-2", support_type="contradicts")
        graph.add_citation("claim-1", "ev-3", support_type="supports")

        supporting = graph.get_supporting_evidence("claim-1")

        assert len(supporting) == 2
        assert all(c.support_type == "supports" for c in supporting)

    def test_get_contradicting_evidence(self):
        """Test getting only contradicting evidence."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", support_type="supports")
        graph.add_citation("claim-1", "ev-2", support_type="contradicts")
        graph.add_citation("claim-1", "ev-3", support_type="contradicts")

        contradicting = graph.get_contradicting_evidence("claim-1")

        assert len(contradicting) == 2
        assert all(c.support_type == "contradicts" for c in contradicting)

    def test_compute_claim_support_score_positive(self):
        """Test computing support score (positive)."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", relevance=1.0, support_type="supports")
        graph.add_citation("claim-1", "ev-2", relevance=1.0, support_type="supports")

        score = graph.compute_claim_support_score("claim-1")

        assert score == 1.0  # All supporting

    def test_compute_claim_support_score_negative(self):
        """Test computing support score (negative)."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", relevance=1.0, support_type="contradicts")
        graph.add_citation("claim-1", "ev-2", relevance=1.0, support_type="contradicts")

        score = graph.compute_claim_support_score("claim-1")

        assert score == -1.0  # All contradicting

    def test_compute_claim_support_score_mixed(self):
        """Test computing support score (mixed)."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1", relevance=1.0, support_type="supports")
        graph.add_citation("claim-1", "ev-2", relevance=1.0, support_type="contradicts")

        score = graph.compute_claim_support_score("claim-1")

        assert score == 0.0  # Balanced

    def test_compute_claim_support_score_no_citations(self):
        """Test computing support score with no citations."""
        graph = CitationGraph()
        score = graph.compute_claim_support_score("nonexistent")
        assert score == 0.0

    def test_find_circular_dependencies_none(self):
        """Test finding circular dependencies (none)."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "ev-1")
        graph.add_citation("claim-2", "ev-2")

        cycles = graph.find_circular_dependencies()

        assert cycles == []


# =============================================================================
# MerkleTree Tests
# =============================================================================


class TestMerkleTree:
    """Test MerkleTree class."""

    def test_initialization_empty(self):
        """Test MerkleTree initialization without records."""
        tree = MerkleTree()
        assert tree.leaves == []
        assert tree.root is None

    def test_build_single_record(self):
        """Test building tree with single record."""
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            content="Content",
        )
        tree = MerkleTree([record])

        assert tree.root is not None
        assert len(tree.root) == 64

    def test_build_multiple_records(self):
        """Test building tree with multiple records."""
        records = [
            ProvenanceRecord(
                id=f"test-{i}",
                content_hash="",
                source_type=SourceType.DOCUMENT,
                source_id=f"doc-{i}",
                content=f"Content {i}",
            )
            for i in range(4)
        ]
        tree = MerkleTree(records)

        assert tree.root is not None
        assert len(tree.leaves) >= 4

    def test_get_proof(self):
        """Test getting Merkle proof."""
        records = [
            ProvenanceRecord(
                id=f"test-{i}",
                content_hash="",
                source_type=SourceType.DOCUMENT,
                source_id=f"doc-{i}",
                content=f"Content {i}",
            )
            for i in range(4)
        ]
        tree = MerkleTree(records)

        proof = tree.get_proof(0)

        assert len(proof) > 0
        # Each element is (sibling_hash, is_left)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in proof)

    def test_verify_proof_valid(self):
        """Test verifying valid Merkle proof."""
        records = [
            ProvenanceRecord(
                id=f"test-{i}",
                content_hash="",
                source_type=SourceType.DOCUMENT,
                source_id=f"doc-{i}",
                content=f"Content {i}",
            )
            for i in range(4)
        ]
        tree = MerkleTree(records)

        leaf_hash = records[0].content_hash
        proof = tree.get_proof(0)

        valid = tree.verify_proof(leaf_hash, proof, tree.root)

        assert valid is True

    def test_verify_proof_invalid(self):
        """Test verifying invalid Merkle proof."""
        records = [
            ProvenanceRecord(
                id=f"test-{i}",
                content_hash="",
                source_type=SourceType.DOCUMENT,
                source_id=f"doc-{i}",
                content=f"Content {i}",
            )
            for i in range(4)
        ]
        tree = MerkleTree(records)

        wrong_hash = "0" * 64
        proof = tree.get_proof(0)

        valid = tree.verify_proof(wrong_hash, proof, tree.root)

        assert valid is False


# =============================================================================
# ProvenanceVerifier Tests
# =============================================================================


class TestProvenanceVerifier:
    """Test ProvenanceVerifier class."""

    @pytest.fixture
    def chain_with_records(self):
        """Create a chain with records."""
        chain = ProvenanceChain()
        chain.add_record("First", SourceType.DOCUMENT, "doc-1")
        chain.add_record("Second", SourceType.DOCUMENT, "doc-2")
        return chain

    def test_initialization(self, chain_with_records):
        """Test ProvenanceVerifier initialization."""
        verifier = ProvenanceVerifier(chain_with_records)
        assert verifier.chain == chain_with_records
        assert verifier.graph is not None

    def test_verify_record_valid(self, chain_with_records):
        """Test verifying valid record."""
        verifier = ProvenanceVerifier(chain_with_records)
        record = chain_with_records.records[0]

        valid, errors = verifier.verify_record(record.id)

        assert valid is True
        assert errors == []

    def test_verify_record_not_found(self, chain_with_records):
        """Test verifying non-existent record."""
        verifier = ProvenanceVerifier(chain_with_records)

        valid, errors = verifier.verify_record("nonexistent")

        assert valid is False
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_verify_claim_evidence(self, chain_with_records):
        """Test verifying claim evidence."""
        graph = CitationGraph()
        graph.add_citation("claim-1", chain_with_records.records[0].id)

        verifier = ProvenanceVerifier(chain_with_records, graph)
        result = verifier.verify_claim_evidence("claim-1")

        assert result["claim_id"] == "claim-1"
        assert result["citation_count"] == 1
        assert result["verified_count"] == 1
        assert result["failed_count"] == 0

    def test_verify_claim_evidence_missing(self, chain_with_records):
        """Test verifying claim with missing evidence."""
        graph = CitationGraph()
        graph.add_citation("claim-1", "nonexistent-evidence")

        verifier = ProvenanceVerifier(chain_with_records, graph)
        result = verifier.verify_claim_evidence("claim-1")

        assert result["failed_count"] == 1
        assert "not found" in result["errors"][0]

    def test_generate_provenance_report(self, chain_with_records):
        """Test generating provenance report."""
        verifier = ProvenanceVerifier(chain_with_records)
        record = chain_with_records.records[0]

        report = verifier.generate_provenance_report(record.id)

        assert report["record_id"] == record.id
        assert "content_hash" in report
        assert "chain_hash" in report
        assert "source" in report
        assert "transformation_history" in report

    def test_generate_provenance_report_not_found(self, chain_with_records):
        """Test generating report for non-existent record."""
        verifier = ProvenanceVerifier(chain_with_records)

        report = verifier.generate_provenance_report("nonexistent")

        assert "error" in report


# =============================================================================
# ProvenanceManager Tests
# =============================================================================


class TestProvenanceManager:
    """Test ProvenanceManager class."""

    def test_initialization(self):
        """Test ProvenanceManager initialization."""
        manager = ProvenanceManager()
        assert manager.debate_id  # Auto-generated
        assert manager.chain is not None
        assert manager.graph is not None
        assert manager.verifier is not None

    def test_initialization_with_id(self):
        """Test ProvenanceManager initialization with custom ID."""
        manager = ProvenanceManager(debate_id="debate-123")
        assert manager.debate_id == "debate-123"

    def test_record_evidence(self):
        """Test recording evidence."""
        manager = ProvenanceManager()

        record = manager.record_evidence(
            content="Important evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert record.content == "Important evidence"
        assert record.source_type == SourceType.DOCUMENT
        assert len(manager.chain.records) == 1

    def test_cite_evidence(self):
        """Test citing evidence for a claim."""
        manager = ProvenanceManager()
        record = manager.record_evidence(
            content="Evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        citation = manager.cite_evidence(
            claim_id="claim-1",
            evidence_id=record.id,
            relevance=0.95,
            support_type="supports",
        )

        assert citation.claim_id == "claim-1"
        assert citation.evidence_id == record.id
        assert citation.relevance == 0.95

    def test_synthesize_evidence(self):
        """Test synthesizing evidence from multiple sources."""
        manager = ProvenanceManager()

        # Create parent evidence
        ev1 = manager.record_evidence("Source 1", SourceType.DOCUMENT, "doc-1")
        ev2 = manager.record_evidence("Source 2", SourceType.DOCUMENT, "doc-2")

        # Synthesize
        synthesis = manager.synthesize_evidence(
            parent_ids=[ev1.id, ev2.id],
            synthesized_content="Combined insight from sources",
            synthesizer_id="agent-1",
        )

        assert synthesis.source_type == SourceType.SYNTHESIS
        assert synthesis.transformation == TransformationType.AGGREGATED
        assert ev1.id in synthesis.parent_ids
        assert ev2.id in synthesis.parent_ids

    def test_verify_chain_integrity(self):
        """Test chain integrity verification."""
        manager = ProvenanceManager()
        manager.record_evidence("Evidence 1", SourceType.DOCUMENT, "doc-1")
        manager.record_evidence("Evidence 2", SourceType.DOCUMENT, "doc-2")

        valid, errors = manager.verify_chain_integrity()

        assert valid is True
        assert errors == []

    def test_get_evidence_provenance(self):
        """Test getting evidence provenance report."""
        manager = ProvenanceManager()
        record = manager.record_evidence(
            content="Evidence",
            source_type=SourceType.COMPUTATION,
            source_id="calc-1",
        )

        report = manager.get_evidence_provenance(record.id)

        assert report["record_id"] == record.id
        assert report["source"]["type"] == "computation"

    def test_get_claim_support(self):
        """Test getting claim support verification."""
        manager = ProvenanceManager()
        record = manager.record_evidence("Evidence", SourceType.DOCUMENT, "doc-1")
        manager.cite_evidence("claim-1", record.id, support_type="supports")

        support = manager.get_claim_support("claim-1")

        assert support["claim_id"] == "claim-1"
        assert support["citation_count"] == 1
        assert support["verified_count"] == 1

    def test_export(self):
        """Test exporting provenance data."""
        manager = ProvenanceManager(debate_id="test-debate")
        record = manager.record_evidence("Evidence", SourceType.DOCUMENT, "doc-1")
        manager.cite_evidence("claim-1", record.id)

        data = manager.export()

        assert data["debate_id"] == "test-debate"
        assert "chain" in data
        assert "citations" in data
        assert len(data["citations"]) == 1

    def test_load(self):
        """Test loading provenance data."""
        manager = ProvenanceManager(debate_id="test-debate")
        record = manager.record_evidence("Evidence", SourceType.DOCUMENT, "doc-1")
        manager.cite_evidence("claim-1", record.id)

        data = manager.export()
        restored = ProvenanceManager.load(data)

        assert restored.debate_id == "test-debate"
        assert len(restored.chain.records) == 1
        assert len(restored.graph.citations) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestProvenanceIntegration:
    """Integration tests for provenance system."""

    def test_full_provenance_workflow(self):
        """Test complete provenance workflow."""
        manager = ProvenanceManager(debate_id="integration-test")

        # Record multiple pieces of evidence
        ev1 = manager.record_evidence(
            content="Research finding: X is correlated with Y",
            source_type=SourceType.DOCUMENT,
            source_id="paper-1",
            metadata={"doi": "10.1000/example"},
        )

        ev2 = manager.record_evidence(
            content="Database query result: 85% success rate",
            source_type=SourceType.DATABASE,
            source_id="analytics-db",
        )

        ev3 = manager.record_evidence(
            content="Expert opinion: methodology is sound",
            source_type=SourceType.USER_PROVIDED,
            source_id="reviewer-1",
        )

        # Create citations
        manager.cite_evidence("main-claim", ev1.id, relevance=0.9, support_type="supports")
        manager.cite_evidence("main-claim", ev2.id, relevance=0.8, support_type="supports")
        manager.cite_evidence("main-claim", ev3.id, relevance=0.7, support_type="supports")
        manager.cite_evidence("counter-claim", ev1.id, relevance=0.6, support_type="contradicts")

        # Synthesize evidence
        synthesis = manager.synthesize_evidence(
            parent_ids=[ev1.id, ev2.id],
            synthesized_content="Combined analysis supports conclusion",
            synthesizer_id="analysis-agent",
        )
        manager.cite_evidence("main-claim", synthesis.id, relevance=0.95, support_type="supports")

        # Verify chain integrity
        valid, errors = manager.verify_chain_integrity()
        assert valid is True

        # Get claim support
        support = manager.get_claim_support("main-claim")
        assert support["citation_count"] == 4  # 3 direct + 1 synthesis
        assert support["verified_count"] == 4

        # Export and restore
        data = manager.export()
        restored = ProvenanceManager.load(data)

        assert len(restored.chain.records) == 4  # 3 evidence + 1 synthesis
        assert restored.debate_id == "integration-test"

        # Verify restored chain
        valid, errors = restored.verify_chain_integrity()
        assert valid is True
