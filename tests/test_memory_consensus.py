"""
Tests for aragora.memory.consensus module.

Tests ConsensusMemory, DissentRetriever, and related dataclasses.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from aragora.memory.consensus import (
    ConsensusMemory,
    ConsensusRecord,
    ConsensusStrength,
    DissentRecord,
    DissentRetriever,
    DissentType,
    SimilarDebate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def consensus_memory(temp_db):
    """ConsensusMemory with temporary database."""
    return ConsensusMemory(db_path=temp_db)


@pytest.fixture
def dissent_retriever(consensus_memory):
    """DissentRetriever with consensus memory."""
    return DissentRetriever(memory=consensus_memory)


@pytest.fixture
def sample_consensus_record():
    """Sample ConsensusRecord for testing."""
    return ConsensusRecord(
        id="cons-001",
        topic="How to implement caching",
        topic_hash="abc123def456",
        conclusion="Use Redis for distributed caching",
        strength=ConsensusStrength.STRONG,
        confidence=0.85,
        participating_agents=["agent1", "agent2", "agent3"],
        agreeing_agents=["agent1", "agent2"],
        dissenting_agents=["agent3"],
        key_claims=["Redis is fast", "Redis supports clustering"],
        supporting_evidence=["Benchmark results"],
        dissent_ids=[],
        domain="backend",
        tags=["caching", "redis"],
        timestamp=datetime.now(),
        debate_duration_seconds=120.0,
        rounds=3,
    )


@pytest.fixture
def sample_dissent_record():
    """Sample DissentRecord for testing."""
    return DissentRecord(
        id="diss-001",
        debate_id="cons-001",
        agent_id="agent3",
        dissent_type=DissentType.ALTERNATIVE_APPROACH,
        content="Consider in-memory caching first",
        reasoning="Simpler to implement and may be sufficient",
        confidence=0.7,
        acknowledged=False,
        rebuttal="",
        timestamp=datetime.now(),
        metadata={"priority": "medium"},
    )


# =============================================================================
# Test Enums
# =============================================================================


class TestConsensusStrength:
    """Tests for ConsensusStrength enum."""

    def test_has_unanimous(self):
        """Should have UNANIMOUS value."""
        assert ConsensusStrength.UNANIMOUS.value == "unanimous"

    def test_has_strong(self):
        """Should have STRONG value."""
        assert ConsensusStrength.STRONG.value == "strong"

    def test_has_moderate(self):
        """Should have MODERATE value."""
        assert ConsensusStrength.MODERATE.value == "moderate"

    def test_has_weak(self):
        """Should have WEAK value."""
        assert ConsensusStrength.WEAK.value == "weak"

    def test_has_split(self):
        """Should have SPLIT value."""
        assert ConsensusStrength.SPLIT.value == "split"

    def test_has_contested(self):
        """Should have CONTESTED value."""
        assert ConsensusStrength.CONTESTED.value == "contested"

    def test_total_values(self):
        """Should have exactly 6 values."""
        assert len(ConsensusStrength) == 6


class TestDissentType:
    """Tests for DissentType enum."""

    def test_has_minor_quibble(self):
        """Should have MINOR_QUIBBLE value."""
        assert DissentType.MINOR_QUIBBLE.value == "minor_quibble"

    def test_has_alternative_approach(self):
        """Should have ALTERNATIVE_APPROACH value."""
        assert DissentType.ALTERNATIVE_APPROACH.value == "alternative_approach"

    def test_has_fundamental_disagreement(self):
        """Should have FUNDAMENTAL_DISAGREEMENT value."""
        assert DissentType.FUNDAMENTAL_DISAGREEMENT.value == "fundamental_disagreement"

    def test_has_edge_case_concern(self):
        """Should have EDGE_CASE_CONCERN value."""
        assert DissentType.EDGE_CASE_CONCERN.value == "edge_case_concern"

    def test_has_risk_warning(self):
        """Should have RISK_WARNING value."""
        assert DissentType.RISK_WARNING.value == "risk_warning"

    def test_has_abstention(self):
        """Should have ABSTENTION value."""
        assert DissentType.ABSTENTION.value == "abstention"

    def test_total_values(self):
        """Should have exactly 6 values."""
        assert len(DissentType) == 6


# =============================================================================
# Test DissentRecord
# =============================================================================


class TestDissentRecord:
    """Tests for DissentRecord dataclass."""

    def test_creation_with_all_fields(self, sample_dissent_record):
        """Should create record with all fields."""
        record = sample_dissent_record
        assert record.id == "diss-001"
        assert record.debate_id == "cons-001"
        assert record.agent_id == "agent3"
        assert record.dissent_type == DissentType.ALTERNATIVE_APPROACH
        assert record.content == "Consider in-memory caching first"
        assert record.reasoning == "Simpler to implement and may be sufficient"
        assert record.confidence == 0.7
        assert record.acknowledged is False
        assert record.rebuttal == ""
        assert record.metadata == {"priority": "medium"}

    def test_to_dict_converts_enum(self, sample_dissent_record):
        """Should convert enum to string value in to_dict."""
        data = sample_dissent_record.to_dict()
        assert data["dissent_type"] == "alternative_approach"

    def test_to_dict_converts_timestamp(self, sample_dissent_record):
        """Should convert timestamp to ISO format."""
        data = sample_dissent_record.to_dict()
        assert isinstance(data["timestamp"], str)
        # Should be parseable
        datetime.fromisoformat(data["timestamp"])

    def test_from_dict_parses_correctly(self, sample_dissent_record):
        """Should deserialize from dict."""
        data = sample_dissent_record.to_dict()
        restored = DissentRecord.from_dict(data)

        assert restored.id == sample_dissent_record.id
        assert restored.dissent_type == sample_dissent_record.dissent_type
        assert restored.confidence == sample_dissent_record.confidence
        assert restored.acknowledged == sample_dissent_record.acknowledged

    def test_from_dict_parses_timestamp(self, sample_dissent_record):
        """Should parse timestamp from ISO format."""
        data = sample_dissent_record.to_dict()
        restored = DissentRecord.from_dict(data)
        assert isinstance(restored.timestamp, datetime)

    def test_metadata_preserved_in_round_trip(self, sample_dissent_record):
        """Should preserve metadata through serialization."""
        data = sample_dissent_record.to_dict()
        restored = DissentRecord.from_dict(data)
        assert restored.metadata == {"priority": "medium"}


# =============================================================================
# Test ConsensusRecord
# =============================================================================


class TestConsensusRecord:
    """Tests for ConsensusRecord dataclass."""

    def test_creation_with_all_fields(self, sample_consensus_record):
        """Should create record with all fields."""
        record = sample_consensus_record
        assert record.id == "cons-001"
        assert record.topic == "How to implement caching"
        assert record.strength == ConsensusStrength.STRONG
        assert record.confidence == 0.85
        assert len(record.participating_agents) == 3
        assert len(record.agreeing_agents) == 2
        assert len(record.dissenting_agents) == 1
        assert record.domain == "backend"

    def test_compute_agreement_ratio_all_agree(self):
        """Should return 1.0 when all agents agree."""
        record = ConsensusRecord(
            id="test",
            topic="test",
            topic_hash="test",
            conclusion="test",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=1.0,
            participating_agents=["a", "b", "c"],
            agreeing_agents=["a", "b", "c"],
            dissenting_agents=[],
        )
        assert record.compute_agreement_ratio() == 1.0

    def test_compute_agreement_ratio_mixed(self):
        """Should return correct ratio for mixed votes."""
        record = ConsensusRecord(
            id="test",
            topic="test",
            topic_hash="test",
            conclusion="test",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b", "c", "d", "e"],
            agreeing_agents=["a", "b", "c"],
            dissenting_agents=["d", "e"],
        )
        assert record.compute_agreement_ratio() == 0.6

    def test_compute_agreement_ratio_empty_agents(self):
        """Should return 0.0 when no agents participated."""
        record = ConsensusRecord(
            id="test",
            topic="test",
            topic_hash="test",
            conclusion="test",
            strength=ConsensusStrength.SPLIT,
            confidence=0.0,
            participating_agents=[],
            agreeing_agents=[],
        )
        assert record.compute_agreement_ratio() == 0.0

    def test_to_dict_serialization(self, sample_consensus_record):
        """Should serialize to dict correctly."""
        data = sample_consensus_record.to_dict()

        assert data["id"] == "cons-001"
        assert data["strength"] == "strong"
        assert data["confidence"] == 0.85
        assert len(data["participating_agents"]) == 3
        assert isinstance(data["timestamp"], str)

    def test_from_dict_deserialization(self, sample_consensus_record):
        """Should deserialize from dict correctly."""
        data = sample_consensus_record.to_dict()
        restored = ConsensusRecord.from_dict(data)

        assert restored.id == sample_consensus_record.id
        assert restored.strength == sample_consensus_record.strength
        assert restored.confidence == sample_consensus_record.confidence
        assert restored.participating_agents == sample_consensus_record.participating_agents

    def test_supersession_metadata(self):
        """Should track supersession chain."""
        record = ConsensusRecord(
            id="new",
            topic="Updated topic",
            topic_hash="hash",
            conclusion="New conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            supersedes="old",
        )
        assert record.supersedes == "old"
        assert record.superseded_by is None


# =============================================================================
# Test ConsensusMemory Database
# =============================================================================


class TestConsensusMemoryDatabase:
    """Tests for ConsensusMemory database operations."""

    def test_init_creates_tables(self, temp_db):
        """Should create consensus and dissent tables."""
        import sqlite3

        memory = ConsensusMemory(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

        assert "consensus" in tables
        assert "dissent" in tables

    def test_init_creates_indices(self, temp_db):
        """Should create indices for efficient queries."""
        import sqlite3

        ConsensusMemory(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = [row[0] for row in cursor.fetchall()]

        assert "idx_consensus_topic_hash" in indices
        assert "idx_consensus_domain" in indices
        assert "idx_dissent_debate" in indices
        assert "idx_dissent_type" in indices

    def test_hash_topic_is_deterministic(self, consensus_memory):
        """Same topic should produce same hash."""
        hash1 = consensus_memory._hash_topic("How to implement caching")
        hash2 = consensus_memory._hash_topic("How to implement caching")
        assert hash1 == hash2

    def test_hash_topic_normalizes_case(self, consensus_memory):
        """Topic hash should be case-insensitive."""
        hash1 = consensus_memory._hash_topic("How to implement CACHING")
        hash2 = consensus_memory._hash_topic("how to implement caching")
        assert hash1 == hash2

    def test_hash_topic_normalizes_word_order(self, consensus_memory):
        """Topic hash should be word-order independent."""
        hash1 = consensus_memory._hash_topic("implement caching how to")
        hash2 = consensus_memory._hash_topic("how to implement caching")
        assert hash1 == hash2

    def test_hash_topic_length(self, consensus_memory):
        """Topic hash should be 16 characters."""
        hash_val = consensus_memory._hash_topic("Any topic here")
        assert len(hash_val) == 16


# =============================================================================
# Test ConsensusMemory CRUD
# =============================================================================


class TestConsensusMemoryCRUD:
    """Tests for ConsensusMemory create/read/update/delete operations."""

    def test_store_consensus_returns_record(self, consensus_memory):
        """Should return ConsensusRecord on store."""
        record = consensus_memory.store_consensus(
            topic="Test topic",
            conclusion="Test conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a", "b"],
            agreeing_agents=["a", "b"],
        )

        assert isinstance(record, ConsensusRecord)
        assert record.topic == "Test topic"
        assert record.strength == ConsensusStrength.STRONG
        assert record.id is not None

    def test_store_dissent_returns_record(self, consensus_memory):
        """Should return DissentRecord on store."""
        # First create a consensus
        consensus = consensus_memory.store_consensus(
            topic="Test",
            conclusion="Test",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        dissent = consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Be careful",
            reasoning="Potential issues",
        )

        assert isinstance(dissent, DissentRecord)
        assert dissent.debate_id == consensus.id
        assert dissent.dissent_type == DissentType.RISK_WARNING

    def test_get_consensus_by_id(self, consensus_memory):
        """Should retrieve consensus by ID."""
        stored = consensus_memory.store_consensus(
            topic="Retrieve test",
            conclusion="Should work",
            strength=ConsensusStrength.STRONG,
            confidence=0.95,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        retrieved = consensus_memory.get_consensus(stored.id)

        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.topic == "Retrieve test"

    def test_get_consensus_not_found(self, consensus_memory):
        """Should return None for non-existent ID."""
        result = consensus_memory.get_consensus("non-existent-id")
        assert result is None

    def test_get_dissents_by_debate_id(self, consensus_memory):
        """Should retrieve all dissents for a debate."""
        consensus = consensus_memory.store_consensus(
            topic="Test",
            conclusion="Test",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b", "c"],
            agreeing_agents=["a"],
            dissenting_agents=["b", "c"],
        )

        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="Different way",
            reasoning="Reasons",
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.RISK_WARNING,
            content="Watch out",
            reasoning="Caution",
        )

        dissents = consensus_memory.get_dissents(consensus.id)

        assert len(dissents) == 2
        assert all(isinstance(d, DissentRecord) for d in dissents)

    def test_get_dissents_empty(self, consensus_memory):
        """Should return empty list when no dissents."""
        consensus = consensus_memory.store_consensus(
            topic="No dissent",
            conclusion="Everyone agreed",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=1.0,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        dissents = consensus_memory.get_dissents(consensus.id)
        assert dissents == []


# =============================================================================
# Test ConsensusMemory Search
# =============================================================================


class TestConsensusMemorySearch:
    """Tests for ConsensusMemory search operations."""

    def test_find_similar_debates_exact_match(self, consensus_memory):
        """Should find exact topic matches with score 1.0."""
        consensus_memory.store_consensus(
            topic="Implement user authentication",
            conclusion="Use OAuth2",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        results = consensus_memory.find_similar_debates("Implement user authentication")

        assert len(results) >= 1
        assert results[0].similarity_score == 1.0

    def test_find_similar_debates_word_overlap(self, consensus_memory):
        """Should find debates with word overlap."""
        consensus_memory.store_consensus(
            topic="Python web framework comparison",
            conclusion="Use FastAPI",
            strength=ConsensusStrength.STRONG,
            confidence=0.85,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        results = consensus_memory.find_similar_debates("Python framework for web API")

        assert len(results) >= 1
        assert results[0].similarity_score > 0.1

    def test_find_similar_debates_respects_min_confidence(self, consensus_memory):
        """Should filter by minimum confidence."""
        consensus_memory.store_consensus(
            topic="Low confidence topic",
            conclusion="Not sure",
            strength=ConsensusStrength.WEAK,
            confidence=0.3,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_consensus(
            topic="High confidence topic",
            conclusion="Very sure",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        results = consensus_memory.find_similar_debates(
            "confidence topic",
            min_confidence=0.5,
        )

        # Should only find the high confidence one
        for r in results:
            assert r.consensus.confidence >= 0.5

    def test_find_similar_debates_includes_dissents(self, consensus_memory):
        """Should include dissents with similar debates."""
        consensus = consensus_memory.store_consensus(
            topic="Testing strategy",
            conclusion="Use TDD",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b"],
            agreeing_agents=["a"],
            dissenting_agents=["b"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="BDD is better",
            reasoning="More readable",
        )

        results = consensus_memory.find_similar_debates("Testing strategy")

        assert len(results) >= 1
        assert len(results[0].dissents) == 1

    def test_find_relevant_dissent_filters_by_type(self, consensus_memory):
        """Should filter dissents by type."""
        consensus = consensus_memory.store_consensus(
            topic="Database choice",
            conclusion="Use PostgreSQL",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Scaling concerns",
            reasoning="May need sharding",
            confidence=0.8,
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.MINOR_QUIBBLE,
            content="Minor syntax issue",
            reasoning="Preference",
            confidence=0.3,
        )

        results = consensus_memory.find_relevant_dissent(
            topic="Database choice",
            dissent_types=[DissentType.RISK_WARNING],
        )

        for r in results:
            assert r.dissent_type == DissentType.RISK_WARNING

    def test_find_relevant_dissent_respects_min_confidence(self, consensus_memory):
        """Should filter dissents by minimum confidence."""
        consensus = consensus_memory.store_consensus(
            topic="API design",
            conclusion="REST",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="GraphQL",
            reasoning="More flexible",
            confidence=0.9,
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.MINOR_QUIBBLE,
            content="Naming",
            reasoning="Style",
            confidence=0.2,
        )

        results = consensus_memory.find_relevant_dissent(
            topic="API design",
            min_confidence=0.5,
        )

        for r in results:
            assert r.confidence >= 0.5

    def test_get_domain_consensus_history_filters(self, consensus_memory):
        """Should filter by domain."""
        consensus_memory.store_consensus(
            topic="Backend topic",
            conclusion="Backend conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="backend",
        )
        consensus_memory.store_consensus(
            topic="Frontend topic",
            conclusion="Frontend conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="frontend",
        )

        backend_results = consensus_memory.get_domain_consensus_history("backend")
        frontend_results = consensus_memory.get_domain_consensus_history("frontend")

        assert all(r.domain == "backend" for r in backend_results)
        assert all(r.domain == "frontend" for r in frontend_results)

    def test_get_domain_consensus_history_orders_by_timestamp(self, consensus_memory):
        """Should order results by timestamp descending."""
        # Store in specific order
        consensus_memory.store_consensus(
            topic="First",
            conclusion="First",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="test",
        )
        consensus_memory.store_consensus(
            topic="Second",
            conclusion="Second",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="test",
        )

        results = consensus_memory.get_domain_consensus_history("test")

        # Most recent (Second) should be first
        assert results[0].topic == "Second"

    def test_supersede_consensus_links_records(self, consensus_memory):
        """Should link old and new consensus records."""
        old = consensus_memory.store_consensus(
            topic="Original topic",
            conclusion="Original conclusion",
            strength=ConsensusStrength.MODERATE,
            confidence=0.6,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        new = consensus_memory.supersede_consensus(
            old_consensus_id=old.id,
            new_topic="Updated topic",
            new_conclusion="Updated conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a", "b"],
            agreeing_agents=["a", "b"],
        )

        # Check new record
        assert new.metadata.get("supersedes") == old.id

        # Check old record was updated
        updated_old = consensus_memory.get_consensus(old.id)
        assert updated_old.superseded_by == new.id


# =============================================================================
# Test ConsensusMemory Statistics
# =============================================================================


class TestConsensusMemoryStatistics:
    """Tests for ConsensusMemory statistics."""

    def test_get_statistics_counts(self, consensus_memory):
        """Should count consensus and dissents."""
        consensus = consensus_memory.store_consensus(
            topic="Test",
            conclusion="Test",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.MINOR_QUIBBLE,
            content="Test",
            reasoning="Test",
        )

        stats = consensus_memory.get_statistics()

        assert stats["total_consensus"] == 1
        assert stats["total_dissents"] == 1

    def test_get_statistics_by_domain(self, consensus_memory):
        """Should count by domain."""
        consensus_memory.store_consensus(
            topic="Backend 1",
            conclusion="Test",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="backend",
        )
        consensus_memory.store_consensus(
            topic="Backend 2",
            conclusion="Test",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="backend",
        )
        consensus_memory.store_consensus(
            topic="Frontend",
            conclusion="Test",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="frontend",
        )

        stats = consensus_memory.get_statistics()

        assert stats["by_domain"]["backend"] == 2
        assert stats["by_domain"]["frontend"] == 1

    def test_get_statistics_by_strength(self, consensus_memory):
        """Should count by strength."""
        consensus_memory.store_consensus(
            topic="Strong",
            conclusion="Test",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_consensus(
            topic="Weak",
            conclusion="Test",
            strength=ConsensusStrength.WEAK,
            confidence=0.5,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        stats = consensus_memory.get_statistics()

        assert stats["by_strength"]["strong"] == 1
        assert stats["by_strength"]["weak"] == 1

    def test_get_statistics_by_dissent_type(self, consensus_memory):
        """Should count by dissent type."""
        consensus = consensus_memory.store_consensus(
            topic="Test",
            conclusion="Test",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Risk",
            reasoning="Risk",
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.RISK_WARNING,
            content="Risk 2",
            reasoning="Risk 2",
        )

        stats = consensus_memory.get_statistics()

        assert stats["by_dissent_type"]["risk_warning"] == 2


# =============================================================================
# Test DissentRetriever
# =============================================================================


class TestDissentRetriever:
    """Tests for DissentRetriever class."""

    def test_retrieve_for_new_debate_structure(self, consensus_memory, dissent_retriever):
        """Should return expected structure."""
        result = dissent_retriever.retrieve_for_new_debate("Any topic")

        assert "similar_debates" in result
        assert "relevant_dissents" in result
        assert "dissent_by_type" in result
        assert "unacknowledged_dissents" in result
        assert "total_similar" in result
        assert "total_dissents" in result

    def test_retrieve_for_new_debate_finds_unacknowledged(
        self, consensus_memory, dissent_retriever
    ):
        """Should identify unacknowledged dissents."""
        consensus = consensus_memory.store_consensus(
            topic="Unacknowledged test topic",
            conclusion="Test",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Not addressed",
            reasoning="Important",
            acknowledged=False,
        )

        result = dissent_retriever.retrieve_for_new_debate("Unacknowledged test topic")

        assert len(result["unacknowledged_dissents"]) >= 1

    def test_find_contrarian_views_filters_types(self, consensus_memory, dissent_retriever):
        """Should find FUNDAMENTAL_DISAGREEMENT and ALTERNATIVE_APPROACH."""
        consensus = consensus_memory.store_consensus(
            topic="Contrarian topic",
            conclusion="Mainstream view",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.FUNDAMENTAL_DISAGREEMENT,
            content="Disagree completely",
            reasoning="Different view",
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.MINOR_QUIBBLE,
            content="Small issue",
            reasoning="Minor",
        )

        results = dissent_retriever.find_contrarian_views("Contrarian topic")

        for r in results:
            assert r.dissent_type in [
                DissentType.FUNDAMENTAL_DISAGREEMENT,
                DissentType.ALTERNATIVE_APPROACH,
            ]

    def test_find_risk_warnings_filters_types(self, consensus_memory, dissent_retriever):
        """Should find RISK_WARNING and EDGE_CASE_CONCERN."""
        consensus = consensus_memory.store_consensus(
            topic="Risk topic",
            conclusion="Proceed",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Danger here",
            reasoning="Be careful",
        )
        consensus_memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.EDGE_CASE_CONCERN,
            content="Edge case",
            reasoning="Special scenario",
        )

        results = dissent_retriever.find_risk_warnings("Risk topic")

        for r in results:
            assert r.dissent_type in [
                DissentType.RISK_WARNING,
                DissentType.EDGE_CASE_CONCERN,
            ]

    def test_get_debate_preparation_context_formats_markdown(
        self, consensus_memory, dissent_retriever
    ):
        """Should return markdown-formatted context."""
        consensus_memory.store_consensus(
            topic="Context topic",
            conclusion="Test conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        context = dissent_retriever.get_debate_preparation_context("Context topic")

        assert "# Historical Context for:" in context
        assert isinstance(context, str)

    def test_empty_results_handled(self, dissent_retriever):
        """Should handle empty results gracefully."""
        result = dissent_retriever.retrieve_for_new_debate("Nonexistent topic")

        assert result["total_similar"] == 0
        assert result["total_dissents"] == 0
        assert result["similar_debates"] == []


# =============================================================================
# Test SimilarDebate
# =============================================================================


class TestSimilarDebate:
    """Tests for SimilarDebate dataclass."""

    def test_creation(self, sample_consensus_record, sample_dissent_record):
        """Should create SimilarDebate with all fields."""
        similar = SimilarDebate(
            consensus=sample_consensus_record,
            similarity_score=0.85,
            dissents=[sample_dissent_record],
            relevance_notes="Very relevant",
        )

        assert similar.consensus == sample_consensus_record
        assert similar.similarity_score == 0.85
        assert len(similar.dissents) == 1
        assert similar.relevance_notes == "Very relevant"

    def test_default_relevance_notes(self, sample_consensus_record):
        """Should default relevance_notes to empty string."""
        similar = SimilarDebate(
            consensus=sample_consensus_record,
            similarity_score=0.5,
            dissents=[],
        )

        assert similar.relevance_notes == ""


# =============================================================================
# Test _get_dissents_batch optimization
# =============================================================================


class TestGetDissentsBatch:
    """Tests for the _get_dissents_batch optimization method."""

    @pytest.fixture
    def consensus_with_dissents(self, consensus_memory):
        """Create multiple consensus records with dissents for batch testing."""
        consensus_ids = []
        for i in range(3):
            record = consensus_memory.store_consensus(
                topic=f"Batch test topic {i}",
                conclusion=f"Conclusion {i}",
                strength=ConsensusStrength.STRONG,
                confidence=0.8,
                participating_agents=["agent1", "agent2"],
                agreeing_agents=["agent1"],
            )
            consensus_ids.append(record.id)
            # Add dissents for each consensus (debate_id links to consensus id)
            for j in range(2):
                consensus_memory.store_dissent(
                    debate_id=record.id,
                    agent_id=f"dissenter_{j}",
                    dissent_type=DissentType.ALTERNATIVE_APPROACH,
                    content=f"Dissent {j} for consensus {i}",
                    reasoning=f"Reasoning {j}",
                    confidence=0.7,
                )
        return consensus_ids

    def test_batch_returns_all_dissents(self, consensus_memory, consensus_with_dissents):
        """Should return dissents for all requested consensus IDs."""
        result = consensus_memory._get_dissents_batch(consensus_with_dissents)

        assert len(result) == 3
        for cid in consensus_with_dissents:
            assert cid in result
            assert len(result[cid]) == 2

    def test_batch_empty_list(self, consensus_memory):
        """Should handle empty consensus ID list."""
        result = consensus_memory._get_dissents_batch([])
        assert result == {}

    def test_batch_nonexistent_ids(self, consensus_memory):
        """Should return empty lists for nonexistent IDs."""
        result = consensus_memory._get_dissents_batch(["fake-id-1", "fake-id-2"])
        assert result == {"fake-id-1": [], "fake-id-2": []}

    def test_batch_mixed_existing_nonexistent(self, consensus_memory, consensus_with_dissents):
        """Should handle mix of existing and nonexistent IDs."""
        mixed_ids = [consensus_with_dissents[0], "fake-id", consensus_with_dissents[1]]
        result = consensus_memory._get_dissents_batch(mixed_ids)

        assert len(result) == 3
        assert len(result[consensus_with_dissents[0]]) == 2
        assert len(result["fake-id"]) == 0
        assert len(result[consensus_with_dissents[1]]) == 2

    def test_batch_preserves_dissent_data(self, consensus_memory, consensus_with_dissents):
        """Should correctly deserialize DissentRecord data."""
        result = consensus_memory._get_dissents_batch([consensus_with_dissents[0]])

        dissents = result[consensus_with_dissents[0]]
        assert len(dissents) == 2
        for d in dissents:
            assert isinstance(d, DissentRecord)
            assert d.dissent_type == DissentType.ALTERNATIVE_APPROACH
            assert d.confidence == 0.7

    def test_find_similar_debates_uses_batch(self, consensus_memory):
        """Should use batch fetching in find_similar_debates."""
        # Store multiple similar debates
        for i in range(5):
            record = consensus_memory.store_consensus(
                topic=f"Python async programming topic {i}",
                conclusion=f"Use asyncio {i}",
                strength=ConsensusStrength.STRONG,
                confidence=0.8,
                participating_agents=["a", "b"],
                agreeing_agents=["a"],
            )
            consensus_memory.store_dissent(
                debate_id=record.id,
                agent_id="critic",
                dissent_type=DissentType.EDGE_CASE_CONCERN,
                content=f"Edge case {i}",
                reasoning="Needs attention",
            )

        # Find similar debates - should use batch fetching internally
        results = consensus_memory.find_similar_debates("Python async programming", limit=3)

        assert len(results) <= 3
        for result in results:
            assert isinstance(result, SimilarDebate)
            # Each should have dissents fetched via batch
            assert len(result.dissents) >= 1

    def test_batch_handles_many_ids(self, consensus_memory):
        """Should handle large number of consensus IDs efficiently."""
        # Create many consensus records
        consensus_ids = []
        for i in range(50):
            record = consensus_memory.store_consensus(
                topic=f"Large batch topic {i}",
                conclusion=f"Conclusion {i}",
                strength=ConsensusStrength.MODERATE,
                confidence=0.6,
                participating_agents=["a"],
                agreeing_agents=["a"],
            )
            consensus_ids.append(record.id)
            if i % 5 == 0:  # Add dissents to every 5th consensus
                consensus_memory.store_dissent(
                    debate_id=record.id,
                    agent_id="critic",
                    dissent_type=DissentType.MINOR_QUIBBLE,
                    content=f"Minor issue {i}",
                    reasoning="Small concern",
                )

        # Batch fetch all 50
        result = consensus_memory._get_dissents_batch(consensus_ids)

        assert len(result) == 50
        # 10 should have dissents (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
        with_dissents = sum(1 for v in result.values() if len(v) > 0)
        assert with_dissents == 10
