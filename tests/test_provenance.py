"""Tests for provenance module - evidence tracking and integrity verification."""

import hashlib
import json
import pytest
from datetime import datetime

from aragora.reasoning.provenance import (
    SourceType,
    TransformationType,
    ProvenanceRecord,
    Citation,
    ProvenanceChain,
    CitationGraph,
    MerkleTree,
    ProvenanceVerifier,
    ProvenanceManager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_record():
    """Create sample ProvenanceRecord."""
    return ProvenanceRecord(
        id="rec-123",
        content_hash="",  # Will be computed
        source_type=SourceType.AGENT_GENERATED,
        source_id="claude",
        content="The evidence shows X.",
    )


@pytest.fixture
def sample_chain():
    """Create empty ProvenanceChain."""
    return ProvenanceChain()


@pytest.fixture
def chain_with_records():
    """Create chain with sample records."""
    chain = ProvenanceChain()
    chain.add_record(
        content="Original evidence from web search",
        source_type=SourceType.WEB_SEARCH,
        source_id="google.com",
    )
    chain.add_record(
        content="Summarized version of the evidence",
        source_type=SourceType.AGENT_GENERATED,
        source_id="claude",
        transformation=TransformationType.SUMMARIZED,
    )
    return chain


@pytest.fixture
def sample_graph():
    """Create empty CitationGraph."""
    return CitationGraph()


@pytest.fixture
def graph_with_citations():
    """Create CitationGraph with sample citations."""
    graph = CitationGraph()
    graph.add_citation("claim-1", "ev-1", relevance=0.9, support_type="supports")
    graph.add_citation("claim-1", "ev-2", relevance=0.7, support_type="supports")
    graph.add_citation("claim-1", "ev-3", relevance=0.5, support_type="contradicts")
    graph.add_citation("claim-2", "ev-1", relevance=0.8, support_type="supports")
    return graph


@pytest.fixture
def sample_records():
    """Create list of sample records for MerkleTree."""
    records = []
    for i in range(4):
        record = ProvenanceRecord(
            id=f"rec-{i}",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id=f"doc-{i}",
            content=f"Content {i}",
        )
        records.append(record)
    return records


@pytest.fixture
def sample_manager():
    """Create ProvenanceManager."""
    return ProvenanceManager(debate_id="debate-123")


# =============================================================================
# SourceType Enum Tests
# =============================================================================


class TestSourceType:
    """Tests for SourceType enum."""

    def test_all_source_types_have_correct_values(self):
        """All 10 source types should have correct string values."""
        assert SourceType.AGENT_GENERATED.value == "agent_generated"
        assert SourceType.USER_PROVIDED.value == "user_provided"
        assert SourceType.EXTERNAL_API.value == "external_api"
        assert SourceType.WEB_SEARCH.value == "web_search"
        assert SourceType.DOCUMENT.value == "document"
        assert SourceType.CODE_ANALYSIS.value == "code_analysis"
        assert SourceType.DATABASE.value == "database"
        assert SourceType.COMPUTATION.value == "computation"
        assert SourceType.SYNTHESIS.value == "synthesis"
        assert SourceType.UNKNOWN.value == "unknown"

    def test_enum_from_string_value(self):
        """Enum should be creatable from string value."""
        assert SourceType("agent_generated") == SourceType.AGENT_GENERATED
        assert SourceType("web_search") == SourceType.WEB_SEARCH


# =============================================================================
# TransformationType Enum Tests
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_transformation_types_have_correct_values(self):
        """All 10 transformation types should have correct values."""
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

    def test_enum_from_string_value(self):
        """Enum should be creatable from string value."""
        assert TransformationType("summarized") == TransformationType.SUMMARIZED
        assert TransformationType("verified") == TransformationType.VERIFIED


# =============================================================================
# ProvenanceRecord Dataclass Tests
# =============================================================================


class TestProvenanceRecord:
    """Tests for ProvenanceRecord dataclass."""

    def test_all_fields_initialized(self, sample_record):
        """ProvenanceRecord should initialize all fields correctly."""
        assert sample_record.id == "rec-123"
        assert sample_record.source_type == SourceType.AGENT_GENERATED
        assert sample_record.source_id == "claude"
        assert sample_record.content == "The evidence shows X."
        assert sample_record.content_type == "text"
        assert sample_record.transformation == TransformationType.ORIGINAL
        assert sample_record.confidence == 1.0
        assert sample_record.verified is False

    def test_id_auto_generated_if_empty(self):
        """id should be auto-generated if empty."""
        record = ProvenanceRecord(
            id="",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="test",
            content="test content",
        )

        assert record.id != ""
        assert len(record.id) > 0

    def test_content_hash_computed_on_init(self):
        """content_hash should be computed on initialization."""
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="source",
            content="test content",
        )

        assert record.content_hash != ""
        assert len(record.content_hash) == 64  # SHA-256 hex

    def test_compute_hash_produces_consistent_hash(self):
        """_compute_hash should produce consistent hash for same content."""
        record1 = ProvenanceRecord(
            id="r1",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="source",
            content="same content",
        )
        record2 = ProvenanceRecord(
            id="r2",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="source",
            content="same content",
        )

        assert record1.content_hash == record2.content_hash

    def test_chain_hash_includes_previous_hash(self, sample_record):
        """chain_hash should include previous_hash."""
        sample_record.previous_hash = "prev_hash_123"
        chain_hash = sample_record.chain_hash()

        assert chain_hash != sample_record.content_hash
        assert len(chain_hash) == 64

    def test_to_dict_serializes_all_fields(self, sample_record):
        """to_dict should serialize all fields."""
        data = sample_record.to_dict()

        assert data["id"] == "rec-123"
        assert data["source_type"] == "agent_generated"
        assert data["source_id"] == "claude"
        assert data["content"] == "The evidence shows X."
        assert "content_hash" in data
        assert "chain_hash" in data
        assert "timestamp" in data

    def test_from_dict_deserializes_correctly(self, sample_record):
        """from_dict should deserialize correctly."""
        data = sample_record.to_dict()
        restored = ProvenanceRecord.from_dict(data)

        assert restored.id == sample_record.id
        assert restored.source_type == sample_record.source_type
        assert restored.content == sample_record.content

    def test_to_dict_from_dict_roundtrip(self, sample_record):
        """to_dict/from_dict should roundtrip correctly."""
        data = sample_record.to_dict()
        restored = ProvenanceRecord.from_dict(data)
        data2 = restored.to_dict()

        # Compare key fields
        assert data["id"] == data2["id"]
        assert data["content_hash"] == data2["content_hash"]
        assert data["source_type"] == data2["source_type"]

    def test_timestamp_defaults_to_now(self):
        """timestamp should default to now."""
        before = datetime.now()
        record = ProvenanceRecord(
            id="test",
            content_hash="",
            source_type=SourceType.DOCUMENT,
            source_id="test",
            content="test",
        )
        after = datetime.now()

        assert before <= record.timestamp <= after

    def test_parent_ids_defaults_to_empty_list(self, sample_record):
        """parent_ids should default to empty list."""
        assert sample_record.parent_ids == []


# =============================================================================
# Citation Dataclass Tests
# =============================================================================


class TestCitation:
    """Tests for Citation dataclass."""

    def test_all_fields_initialized(self):
        """Citation should initialize all fields correctly."""
        citation = Citation(
            claim_id="claim-1",
            evidence_id="ev-1",
            relevance=0.8,
            support_type="supports",
            citation_text="The quoted text",
        )

        assert citation.claim_id == "claim-1"
        assert citation.evidence_id == "ev-1"
        assert citation.relevance == 0.8
        assert citation.support_type == "supports"
        assert citation.citation_text == "The quoted text"

    def test_defaults(self):
        """Citation should have correct defaults."""
        citation = Citation(claim_id="c1", evidence_id="e1")

        assert citation.relevance == 1.0
        assert citation.support_type == "supports"
        assert citation.citation_text == ""

    def test_metadata_defaults_to_empty_dict(self):
        """metadata should default to empty dict."""
        citation = Citation(claim_id="c1", evidence_id="e1")

        assert citation.metadata == {}


# =============================================================================
# ProvenanceChain Initialization Tests
# =============================================================================


class TestProvenanceChainInit:
    """Tests for ProvenanceChain initialization."""

    def test_chain_id_auto_generated(self):
        """chain_id should be auto-generated."""
        chain = ProvenanceChain()

        assert chain.chain_id is not None
        assert len(chain.chain_id) > 0

    def test_records_starts_empty(self, sample_chain):
        """records should start empty."""
        assert sample_chain.records == []

    def test_genesis_hash_starts_none(self, sample_chain):
        """genesis_hash should start as None."""
        assert sample_chain.genesis_hash is None


# =============================================================================
# ProvenanceChain.add_record Tests
# =============================================================================


class TestProvenanceChainAddRecord:
    """Tests for ProvenanceChain.add_record method."""

    def test_creates_record_with_unique_id(self, sample_chain):
        """add_record should create record with unique id."""
        record = sample_chain.add_record(
            content="Test content",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert record.id is not None
        assert len(record.id) > 0

    def test_first_record_has_no_previous_hash(self, sample_chain):
        """First record should have no previous_hash."""
        record = sample_chain.add_record(
            content="First record",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert record.previous_hash is None

    def test_subsequent_records_link_to_previous(self, sample_chain):
        """Subsequent records should link to previous."""
        first = sample_chain.add_record(
            content="First",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        second = sample_chain.add_record(
            content="Second",
            source_type=SourceType.DOCUMENT,
            source_id="doc-2",
        )

        assert second.previous_hash == first.chain_hash()

    def test_sets_genesis_hash_on_first_record(self, sample_chain):
        """Should set genesis_hash on first record."""
        record = sample_chain.add_record(
            content="Genesis",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert sample_chain.genesis_hash == record.chain_hash()

    def test_parent_ids_passed_correctly(self, sample_chain):
        """parent_ids should be passed correctly."""
        record = sample_chain.add_record(
            content="Synthesized",
            source_type=SourceType.SYNTHESIS,
            source_id="claude",
            parent_ids=["parent-1", "parent-2"],
        )

        assert record.parent_ids == ["parent-1", "parent-2"]

    def test_transformation_passed_correctly(self, sample_chain):
        """transformation should be passed correctly."""
        record = sample_chain.add_record(
            content="Summary",
            source_type=SourceType.AGENT_GENERATED,
            source_id="claude",
            transformation=TransformationType.SUMMARIZED,
        )

        assert record.transformation == TransformationType.SUMMARIZED


# =============================================================================
# ProvenanceChain.verify_chain Tests
# =============================================================================


class TestProvenanceChainVerifyChain:
    """Tests for ProvenanceChain.verify_chain method."""

    def test_empty_chain_returns_true(self, sample_chain):
        """Empty chain should return (True, [])."""
        valid, errors = sample_chain.verify_chain()

        assert valid is True
        assert errors == []

    def test_valid_chain_returns_true(self, chain_with_records):
        """Valid chain should return (True, [])."""
        valid, errors = chain_with_records.verify_chain()

        assert valid is True
        assert errors == []

    def test_genesis_with_previous_hash_fails(self, chain_with_records):
        """Genesis record with previous_hash should fail."""
        chain_with_records.records[0].previous_hash = "invalid"

        valid, errors = chain_with_records.verify_chain()

        assert valid is False
        assert any("Genesis" in e or "genesis" in e.lower() for e in errors)

    def test_broken_chain_link_detected(self, chain_with_records):
        """Broken chain link should be detected."""
        chain_with_records.records[1].previous_hash = "tampered_hash"

        valid, errors = chain_with_records.verify_chain()

        assert valid is False
        assert any("Chain break" in e or "break" in e.lower() for e in errors)

    def test_content_hash_mismatch_detected(self, chain_with_records):
        """Content hash mismatch should be detected."""
        chain_with_records.records[0].content_hash = "wrong_hash"

        valid, errors = chain_with_records.verify_chain()

        assert valid is False
        assert any("hash" in e.lower() for e in errors)


# =============================================================================
# ProvenanceChain Methods Tests
# =============================================================================


class TestProvenanceChainMethods:
    """Tests for ProvenanceChain helper methods."""

    def test_get_record_finds_by_id(self, chain_with_records):
        """get_record should find record by id."""
        first_id = chain_with_records.records[0].id
        record = chain_with_records.get_record(first_id)

        assert record is not None
        assert record.id == first_id

    def test_get_record_returns_none_for_missing(self, chain_with_records):
        """get_record should return None for missing id."""
        record = chain_with_records.get_record("nonexistent-id")

        assert record is None

    def test_get_ancestry_returns_full_lineage(self, chain_with_records):
        """get_ancestry should return full lineage."""
        second_id = chain_with_records.records[1].id
        ancestry = chain_with_records.get_ancestry(second_id)

        # Should include the record and its ancestors
        assert len(ancestry) >= 1
        ids = [r.id for r in ancestry]
        assert second_id in ids

    def test_to_dict_serializes_chain(self, chain_with_records):
        """to_dict should serialize chain."""
        data = chain_with_records.to_dict()

        assert "chain_id" in data
        assert "genesis_hash" in data
        assert "created_at" in data
        assert "records" in data
        assert len(data["records"]) == 2

    def test_from_dict_deserializes_chain(self, chain_with_records):
        """from_dict should deserialize chain."""
        data = chain_with_records.to_dict()
        restored = ProvenanceChain.from_dict(data)

        assert restored.chain_id == chain_with_records.chain_id
        assert len(restored.records) == len(chain_with_records.records)


# =============================================================================
# CitationGraph Initialization Tests
# =============================================================================


class TestCitationGraphInit:
    """Tests for CitationGraph initialization."""

    def test_empty_citations_dict(self, sample_graph):
        """citations should start empty."""
        assert sample_graph.citations == {}

    def test_empty_index_dicts(self, sample_graph):
        """Index dicts should start empty."""
        assert sample_graph.claim_citations == {}
        assert sample_graph.evidence_citations == {}


# =============================================================================
# CitationGraph.add_citation Tests
# =============================================================================


class TestCitationGraphAddCitation:
    """Tests for CitationGraph.add_citation method."""

    def test_creates_citation_object(self, sample_graph):
        """add_citation should create Citation object."""
        citation = sample_graph.add_citation("claim-1", "ev-1", relevance=0.9)

        assert isinstance(citation, Citation)
        assert citation.claim_id == "claim-1"
        assert citation.evidence_id == "ev-1"

    def test_indexes_by_claim_id(self, sample_graph):
        """add_citation should index by claim_id."""
        sample_graph.add_citation("claim-1", "ev-1")

        assert "claim-1" in sample_graph.claim_citations
        assert len(sample_graph.claim_citations["claim-1"]) == 1

    def test_indexes_by_evidence_id(self, sample_graph):
        """add_citation should index by evidence_id."""
        sample_graph.add_citation("claim-1", "ev-1")

        assert "ev-1" in sample_graph.evidence_citations
        assert len(sample_graph.evidence_citations["ev-1"]) == 1

    def test_returns_created_citation(self, sample_graph):
        """add_citation should return created citation."""
        citation = sample_graph.add_citation("c1", "e1", relevance=0.5)

        assert citation.relevance == 0.5


# =============================================================================
# CitationGraph Query Methods Tests
# =============================================================================


class TestCitationGraphQueries:
    """Tests for CitationGraph query methods."""

    def test_get_claim_evidence_returns_citations(self, graph_with_citations):
        """get_claim_evidence should return citations for claim."""
        citations = graph_with_citations.get_claim_evidence("claim-1")

        assert len(citations) == 3

    def test_get_evidence_claims_returns_citations(self, graph_with_citations):
        """get_evidence_claims should return claims citing evidence."""
        citations = graph_with_citations.get_evidence_claims("ev-1")

        assert len(citations) == 2

    def test_get_supporting_evidence_filters(self, graph_with_citations):
        """get_supporting_evidence should filter by support_type."""
        supporting = graph_with_citations.get_supporting_evidence("claim-1")

        assert len(supporting) == 2
        for c in supporting:
            assert c.support_type == "supports"

    def test_get_contradicting_evidence_filters(self, graph_with_citations):
        """get_contradicting_evidence should filter by support_type."""
        contradicting = graph_with_citations.get_contradicting_evidence("claim-1")

        assert len(contradicting) == 1
        assert contradicting[0].support_type == "contradicts"

    def test_returns_empty_for_missing_ids(self, sample_graph):
        """Should return empty for missing ids."""
        assert sample_graph.get_claim_evidence("nonexistent") == []
        assert sample_graph.get_evidence_claims("nonexistent") == []

    def test_compute_claim_support_score_calculates_correctly(self, graph_with_citations):
        """compute_claim_support_score should calculate correctly."""
        # claim-1 has: 0.9 supports + 0.7 supports - 0.5 contradicts = 1.1
        # Average: 1.1 / 3 = 0.367
        score = graph_with_citations.compute_claim_support_score("claim-1")

        assert isinstance(score, float)
        # Rough check: more support than contradict = positive
        assert score > 0


# =============================================================================
# CitationGraph.find_circular_dependencies Tests
# =============================================================================


class TestCitationGraphCircularDependencies:
    """Tests for CitationGraph.find_circular_dependencies method."""

    def test_returns_empty_for_acyclic_graph(self, graph_with_citations):
        """Should return empty for acyclic graph."""
        cycles = graph_with_citations.find_circular_dependencies()

        # Simple claim -> evidence graph has no cycles
        assert cycles == []

    def test_detects_simple_cycle(self, sample_graph):
        """Should detect simple A -> B -> A cycle."""
        # Create claim-1 citing claim-2, and claim-2 citing claim-1
        sample_graph.add_citation("claim-1", "claim-2")
        sample_graph.add_citation("claim-2", "claim-1")

        cycles = sample_graph.find_circular_dependencies()

        # Should detect the cycle
        assert len(cycles) >= 1

    def test_detects_multi_node_cycle(self, sample_graph):
        """Should detect multi-node cycle."""
        sample_graph.add_citation("A", "B")
        sample_graph.add_citation("B", "C")
        sample_graph.add_citation("C", "A")

        cycles = sample_graph.find_circular_dependencies()

        assert len(cycles) >= 1


# =============================================================================
# MerkleTree Initialization Tests
# =============================================================================


class TestMerkleTreeInit:
    """Tests for MerkleTree initialization."""

    def test_empty_tree_has_none_root(self):
        """Empty tree should have None root."""
        tree = MerkleTree()

        assert tree.root is None

    def test_can_build_from_records(self, sample_records):
        """Should be able to build from records."""
        tree = MerkleTree(sample_records)

        assert tree.root is not None


# =============================================================================
# MerkleTree.build Tests
# =============================================================================


class TestMerkleTreeBuild:
    """Tests for MerkleTree.build method."""

    def test_empty_records_gives_empty_hash(self):
        """Empty records should give hash of empty string."""
        tree = MerkleTree()
        root = tree.build([])

        expected = hashlib.sha256("".encode()).hexdigest()
        assert root == expected

    def test_single_record_gives_leaf_hash(self, sample_records):
        """Single record should use leaf hash."""
        tree = MerkleTree()
        root = tree.build([sample_records[0]])

        assert root is not None
        assert len(root) == 64

    def test_multiple_records_builds_tree(self, sample_records):
        """Multiple records should build tree."""
        tree = MerkleTree()
        root = tree.build(sample_records)

        assert root is not None
        assert len(tree.tree) > 1  # Multiple levels

    def test_pads_to_power_of_2(self):
        """Should pad leaves to power of 2."""
        records = []
        for i in range(3):  # 3 records, not power of 2
            record = ProvenanceRecord(
                id=f"rec-{i}",
                content_hash="",
                source_type=SourceType.DOCUMENT,
                source_id=f"doc-{i}",
                content=f"Content {i}",
            )
            records.append(record)

        tree = MerkleTree()
        tree.build(records)

        # Should be padded to 4 leaves
        assert len(tree.leaves) == 4


# =============================================================================
# MerkleTree Proofs Tests
# =============================================================================


class TestMerkleTreeProofs:
    """Tests for MerkleTree proof methods."""

    def test_get_proof_returns_sibling_hashes(self, sample_records):
        """get_proof should return sibling hashes."""
        tree = MerkleTree(sample_records)
        proof = tree.get_proof(0)

        assert len(proof) > 0
        for sibling_hash, is_left in proof:
            assert len(sibling_hash) == 64
            assert isinstance(is_left, bool)

    def test_get_proof_invalid_index_returns_empty(self, sample_records):
        """get_proof for invalid index should return empty."""
        tree = MerkleTree(sample_records)
        proof = tree.get_proof(100)  # Invalid index

        assert proof == []

    def test_verify_proof_validates_correct_proof(self, sample_records):
        """verify_proof should validate correct proof."""
        tree = MerkleTree(sample_records)
        proof = tree.get_proof(0)

        is_valid = tree.verify_proof(sample_records[0].content_hash, proof, tree.root)

        assert is_valid is True

    def test_verify_proof_rejects_invalid_proof(self, sample_records):
        """verify_proof should reject invalid proof."""
        tree = MerkleTree(sample_records)
        proof = tree.get_proof(0)

        # Use wrong leaf hash
        is_valid = tree.verify_proof("wrong_hash", proof, tree.root)

        assert is_valid is False


# =============================================================================
# ProvenanceVerifier Initialization Tests
# =============================================================================


class TestProvenanceVerifierInit:
    """Tests for ProvenanceVerifier initialization."""

    def test_stores_chain_reference(self, chain_with_records):
        """Should store chain reference."""
        verifier = ProvenanceVerifier(chain_with_records)

        assert verifier.chain is chain_with_records

    def test_creates_graph_if_not_provided(self, chain_with_records):
        """Should create graph if not provided."""
        verifier = ProvenanceVerifier(chain_with_records)

        assert verifier.graph is not None
        assert isinstance(verifier.graph, CitationGraph)


# =============================================================================
# ProvenanceVerifier.verify_record Tests
# =============================================================================


class TestProvenanceVerifierVerifyRecord:
    """Tests for ProvenanceVerifier.verify_record method."""

    def test_returns_false_for_missing_record(self, chain_with_records):
        """Should return (False, [error]) for missing record."""
        verifier = ProvenanceVerifier(chain_with_records)

        valid, errors = verifier.verify_record("nonexistent")

        assert valid is False
        assert len(errors) >= 1
        assert "not found" in errors[0].lower()

    def test_detects_content_hash_mismatch(self, chain_with_records):
        """Should detect content hash mismatch."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[0].id
        chain_with_records.records[0].content_hash = "tampered"

        valid, errors = verifier.verify_record(record_id)

        assert valid is False
        assert any("hash" in e.lower() for e in errors)

    def test_detects_broken_chain_link(self, chain_with_records):
        """Should detect broken chain link."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[1].id
        chain_with_records.records[1].previous_hash = "broken"

        valid, errors = verifier.verify_record(record_id)

        assert valid is False
        assert any("not found" in e.lower() or "hash" in e.lower() for e in errors)

    def test_detects_missing_parent_records(self, chain_with_records):
        """Should detect missing parent records."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[0].id
        chain_with_records.records[0].parent_ids = ["missing-parent"]

        valid, errors = verifier.verify_record(record_id)

        assert valid is False
        assert any("parent" in e.lower() for e in errors)

    def test_valid_record_returns_true(self, chain_with_records):
        """Valid record should return (True, [])."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[0].id

        valid, errors = verifier.verify_record(record_id)

        assert valid is True
        assert errors == []


# =============================================================================
# ProvenanceVerifier.verify_claim_evidence Tests
# =============================================================================


class TestProvenanceVerifierVerifyClaimEvidence:
    """Tests for ProvenanceVerifier.verify_claim_evidence method."""

    def test_returns_verification_status(self, chain_with_records, graph_with_citations):
        """Should return verification status for all citations."""
        verifier = ProvenanceVerifier(chain_with_records, graph_with_citations)

        result = verifier.verify_claim_evidence("claim-1")

        assert "claim_id" in result
        assert "citation_count" in result
        assert "verified_count" in result
        assert "failed_count" in result

    def test_counts_verified_and_failed(self, chain_with_records, graph_with_citations):
        """Should count verified and failed."""
        verifier = ProvenanceVerifier(chain_with_records, graph_with_citations)

        result = verifier.verify_claim_evidence("claim-1")

        assert result["verified_count"] + result["failed_count"] == result["citation_count"]

    def test_includes_support_score(self, chain_with_records, graph_with_citations):
        """Should include support score."""
        verifier = ProvenanceVerifier(chain_with_records, graph_with_citations)

        result = verifier.verify_claim_evidence("claim-1")

        assert "support_score" in result

    def test_reports_missing_evidence(self, chain_with_records, graph_with_citations):
        """Should report missing evidence."""
        verifier = ProvenanceVerifier(chain_with_records, graph_with_citations)

        result = verifier.verify_claim_evidence("claim-1")

        # All evidence is missing (not in chain)
        assert result["failed_count"] >= 1
        assert "evidence_status" in result


# =============================================================================
# ProvenanceVerifier.generate_provenance_report Tests
# =============================================================================


class TestProvenanceVerifierGenerateReport:
    """Tests for ProvenanceVerifier.generate_provenance_report method."""

    def test_returns_error_for_missing_record(self, chain_with_records):
        """Should return error for missing record."""
        verifier = ProvenanceVerifier(chain_with_records)

        report = verifier.generate_provenance_report("nonexistent")

        assert "error" in report

    def test_includes_ancestry_chain(self, chain_with_records):
        """Should include ancestry chain."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[1].id

        report = verifier.generate_provenance_report(record_id)

        assert "ancestry_depth" in report
        assert "transformation_history" in report

    def test_includes_transformation_history(self, chain_with_records):
        """Should include transformation history."""
        verifier = ProvenanceVerifier(chain_with_records)
        record_id = chain_with_records.records[1].id

        report = verifier.generate_provenance_report(record_id)

        assert "transformation_history" in report
        assert len(report["transformation_history"]) >= 1


# =============================================================================
# ProvenanceManager Initialization Tests
# =============================================================================


class TestProvenanceManagerInit:
    """Tests for ProvenanceManager initialization."""

    def test_creates_chain_and_graph(self, sample_manager):
        """Should create chain and graph."""
        assert sample_manager.chain is not None
        assert sample_manager.graph is not None

    def test_debate_id_auto_generated(self):
        """debate_id should be auto-generated if not provided."""
        manager = ProvenanceManager()

        assert manager.debate_id is not None
        assert len(manager.debate_id) > 0


# =============================================================================
# ProvenanceManager Methods Tests
# =============================================================================


class TestProvenanceManagerMethods:
    """Tests for ProvenanceManager methods."""

    def test_record_evidence_adds_to_chain(self, sample_manager):
        """record_evidence should add to chain."""
        record = sample_manager.record_evidence(
            content="New evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        assert record is not None
        assert len(sample_manager.chain.records) == 1

    def test_cite_evidence_adds_to_graph(self, sample_manager):
        """cite_evidence should add to graph."""
        citation = sample_manager.cite_evidence(
            claim_id="claim-1",
            evidence_id="ev-1",
            relevance=0.9,
        )

        assert citation is not None
        assert len(sample_manager.graph.citations) == 1

    def test_synthesize_evidence_with_parents(self, sample_manager):
        """synthesize_evidence should create record with parents."""
        record = sample_manager.synthesize_evidence(
            parent_ids=["parent-1", "parent-2"],
            synthesized_content="Combined insight",
            synthesizer_id="claude",
        )

        assert record.parent_ids == ["parent-1", "parent-2"]
        assert record.source_type == SourceType.SYNTHESIS

    def test_verify_chain_integrity_delegates(self, sample_manager):
        """verify_chain_integrity should delegate to chain."""
        sample_manager.record_evidence(
            content="Test",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )

        valid, errors = sample_manager.verify_chain_integrity()

        assert valid is True
        assert errors == []

    def test_get_evidence_provenance_returns_report(self, sample_manager):
        """get_evidence_provenance should return report."""
        record = sample_manager.record_evidence(
            content="Test",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )

        report = sample_manager.get_evidence_provenance(record.id)

        assert "record_id" in report
        assert report["record_id"] == record.id

    def test_get_claim_support_returns_verification(self, sample_manager):
        """get_claim_support should return verification."""
        result = sample_manager.get_claim_support("claim-1")

        assert "claim_id" in result
        assert result["claim_id"] == "claim-1"


# =============================================================================
# ProvenanceManager Persistence Tests
# =============================================================================


class TestProvenanceManagerPersistence:
    """Tests for ProvenanceManager persistence methods."""

    def test_export_returns_serializable_dict(self, sample_manager):
        """export should return serializable dict."""
        sample_manager.record_evidence(
            content="Test",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )

        data = sample_manager.export()

        assert "debate_id" in data
        assert "chain" in data
        assert "citations" in data

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_load_restores_manager_state(self, sample_manager):
        """load should restore manager state."""
        sample_manager.record_evidence(
            content="Evidence 1",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        sample_manager.cite_evidence("claim-1", "ev-1")

        data = sample_manager.export()
        restored = ProvenanceManager.load(data)

        assert restored.debate_id == sample_manager.debate_id
        assert len(restored.chain.records) == len(sample_manager.chain.records)

    def test_export_load_roundtrip_preserves_data(self, sample_manager):
        """export/load roundtrip should preserve data."""
        record = sample_manager.record_evidence(
            content="Important evidence",
            source_type=SourceType.WEB_SEARCH,
            source_id="google.com",
        )

        data = sample_manager.export()
        restored = ProvenanceManager.load(data)

        # Verify chain integrity
        valid, errors = restored.verify_chain_integrity()
        assert valid is True

        # Verify record preserved
        restored_record = restored.chain.get_record(record.id)
        assert restored_record is not None
        assert restored_record.content == record.content
