"""Tests for consensus memory - debate outcome storage and retrieval."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from aragora.memory.consensus import (
    ConsensusStrength,
    DissentType,
    DissentRecord,
    ConsensusRecord,
    SimilarDebate,
    ConsensusMemory,
    DissentRetriever,
)


class TestConsensusStrength:
    """Test ConsensusStrength enum."""

    def test_all_strengths_defined(self):
        """Test all expected strengths exist."""
        expected = ["UNANIMOUS", "STRONG", "MODERATE", "WEAK", "SPLIT", "CONTESTED"]
        for s in expected:
            assert hasattr(ConsensusStrength, s)

    def test_strength_values(self):
        """Test strength values."""
        assert ConsensusStrength.UNANIMOUS.value == "unanimous"
        assert ConsensusStrength.STRONG.value == "strong"
        assert ConsensusStrength.MODERATE.value == "moderate"


class TestDissentType:
    """Test DissentType enum."""

    def test_all_types_defined(self):
        """Test all expected dissent types exist."""
        expected = [
            "MINOR_QUIBBLE",
            "ALTERNATIVE_APPROACH",
            "FUNDAMENTAL_DISAGREEMENT",
            "EDGE_CASE_CONCERN",
            "RISK_WARNING",
            "ABSTENTION",
        ]
        for t in expected:
            assert hasattr(DissentType, t)

    def test_type_values(self):
        """Test type values."""
        assert DissentType.MINOR_QUIBBLE.value == "minor_quibble"
        assert DissentType.RISK_WARNING.value == "risk_warning"


class TestDissentRecord:
    """Test DissentRecord dataclass."""

    def test_create_record(self):
        """Test creating a dissent record."""
        record = DissentRecord(
            id="d1",
            debate_id="debate1",
            agent_id="claude",
            dissent_type=DissentType.RISK_WARNING,
            content="Potential security risk",
            reasoning="The approach lacks input validation",
            confidence=0.8,
        )

        assert record.id == "d1"
        assert record.debate_id == "debate1"
        assert record.agent_id == "claude"
        assert record.dissent_type == DissentType.RISK_WARNING
        assert record.confidence == 0.8
        assert record.acknowledged is False

    def test_default_values(self):
        """Test default values."""
        record = DissentRecord(
            id="d1",
            debate_id="debate1",
            agent_id="claude",
            dissent_type=DissentType.MINOR_QUIBBLE,
            content="Small concern",
            reasoning="Minor issue",
        )

        assert record.confidence == 0.0
        assert record.acknowledged is False
        assert record.rebuttal == ""
        assert record.metadata == {}

    def test_to_dict(self):
        """Test serialization to dict."""
        record = DissentRecord(
            id="d1",
            debate_id="debate1",
            agent_id="claude",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="Try a different method",
            reasoning="More efficient approach",
            confidence=0.7,
            acknowledged=True,
            rebuttal="Original approach is simpler",
        )

        data = record.to_dict()

        assert data["id"] == "d1"
        assert data["dissent_type"] == "alternative_approach"
        assert data["confidence"] == 0.7
        assert data["acknowledged"] is True
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "d1",
            "debate_id": "debate1",
            "agent_id": "claude",
            "dissent_type": "risk_warning",
            "content": "Security risk",
            "reasoning": "Needs validation",
            "confidence": 0.8,
            "acknowledged": False,
            "rebuttal": "",
            "timestamp": "2025-01-01T12:00:00",
            "metadata": {"severity": "high"},
        }

        record = DissentRecord.from_dict(data)

        assert record.id == "d1"
        assert record.dissent_type == DissentType.RISK_WARNING
        assert record.confidence == 0.8
        assert record.metadata == {"severity": "high"}


class TestConsensusRecord:
    """Test ConsensusRecord dataclass."""

    def test_create_record(self):
        """Test creating a consensus record."""
        record = ConsensusRecord(
            id="c1",
            topic="Rate limiting implementation",
            topic_hash="abc123",
            conclusion="Use token bucket algorithm",
            strength=ConsensusStrength.STRONG,
            confidence=0.85,
            participating_agents=["claude", "gpt", "gemini"],
            agreeing_agents=["claude", "gpt"],
            dissenting_agents=["gemini"],
        )

        assert record.id == "c1"
        assert record.strength == ConsensusStrength.STRONG
        assert record.confidence == 0.85
        assert len(record.participating_agents) == 3
        assert len(record.agreeing_agents) == 2

    def test_compute_agreement_ratio(self):
        """Test agreement ratio computation."""
        record = ConsensusRecord(
            id="c1",
            topic="Test",
            topic_hash="hash",
            conclusion="Result",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b", "c", "d"],
            agreeing_agents=["a", "b", "c"],
        )

        assert record.compute_agreement_ratio() == 0.75

    def test_compute_agreement_ratio_empty(self):
        """Test agreement ratio with no participants."""
        record = ConsensusRecord(
            id="c1",
            topic="Test",
            topic_hash="hash",
            conclusion="Result",
            strength=ConsensusStrength.SPLIT,
            confidence=0.5,
            participating_agents=[],
            agreeing_agents=[],
        )

        assert record.compute_agreement_ratio() == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        record = ConsensusRecord(
            id="c1",
            topic="Test topic",
            topic_hash="hash123",
            conclusion="Conclusion text",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=0.95,
            participating_agents=["a", "b"],
            agreeing_agents=["a", "b"],
            domain="security",
            tags=["important", "reviewed"],
            rounds=5,
        )

        data = record.to_dict()

        assert data["id"] == "c1"
        assert data["strength"] == "unanimous"
        assert data["confidence"] == 0.95
        assert data["domain"] == "security"
        assert data["tags"] == ["important", "reviewed"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "c1",
            "topic": "Test",
            "topic_hash": "hash",
            "conclusion": "Result",
            "strength": "strong",
            "confidence": 0.8,
            "participating_agents": ["a", "b"],
            "agreeing_agents": ["a"],
            "dissenting_agents": ["b"],
            "key_claims": ["claim1"],
            "domain": "performance",
            "tags": ["urgent"],
            "timestamp": "2025-01-01T00:00:00",
            "rounds": 3,
        }

        record = ConsensusRecord.from_dict(data)

        assert record.id == "c1"
        assert record.strength == ConsensusStrength.STRONG
        assert record.domain == "performance"
        assert record.rounds == 3


class TestSimilarDebate:
    """Test SimilarDebate dataclass."""

    def test_create_similar_debate(self):
        """Test creating a similar debate record."""
        consensus = ConsensusRecord(
            id="c1",
            topic="Test",
            topic_hash="hash",
            conclusion="Result",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
        )

        dissents = [
            DissentRecord(
                id="d1",
                debate_id="c1",
                agent_id="agent1",
                dissent_type=DissentType.MINOR_QUIBBLE,
                content="Small issue",
                reasoning="Reason",
            )
        ]

        similar = SimilarDebate(
            consensus=consensus,
            similarity_score=0.85,
            dissents=dissents,
            relevance_notes="Highly relevant",
        )

        assert similar.consensus.id == "c1"
        assert similar.similarity_score == 0.85
        assert len(similar.dissents) == 1


class TestConsensusMemory:
    """Test ConsensusMemory class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_consensus.db"

    def test_init(self, temp_db):
        """Test initialization creates database."""
        memory = ConsensusMemory(str(temp_db))
        assert temp_db.exists()

    def test_hash_topic(self, temp_db):
        """Test topic hashing for similarity matching."""
        memory = ConsensusMemory(str(temp_db))

        # Same words in different order should hash the same
        hash1 = memory._hash_topic("rate limiting implementation")
        hash2 = memory._hash_topic("implementation rate limiting")
        assert hash1 == hash2

        # Different topics should hash differently
        hash3 = memory._hash_topic("authentication security")
        assert hash1 != hash3

    def test_store_consensus(self, temp_db):
        """Test storing a consensus record."""
        memory = ConsensusMemory(str(temp_db))

        record = memory.store_consensus(
            topic="Rate limiting design",
            conclusion="Use token bucket algorithm",
            strength=ConsensusStrength.STRONG,
            confidence=0.85,
            participating_agents=["claude", "gpt"],
            agreeing_agents=["claude", "gpt"],
            domain="architecture",
            tags=["performance", "scalability"],
        )

        assert record.id is not None
        assert record.topic == "Rate limiting design"
        assert record.strength == ConsensusStrength.STRONG

    def test_get_consensus(self, temp_db):
        """Test retrieving a consensus record."""
        memory = ConsensusMemory(str(temp_db))

        # Store first
        stored = memory.store_consensus(
            topic="Test topic",
            conclusion="Test conclusion",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        # Retrieve
        retrieved = memory.get_consensus(stored.id)

        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.topic == "Test topic"

    def test_get_consensus_not_found(self, temp_db):
        """Test retrieving non-existent consensus."""
        memory = ConsensusMemory(str(temp_db))
        result = memory.get_consensus("nonexistent")
        assert result is None

    def test_store_dissent(self, temp_db):
        """Test storing a dissent record."""
        memory = ConsensusMemory(str(temp_db))

        # Store consensus first
        consensus = memory.store_consensus(
            topic="Test",
            conclusion="Result",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a", "b"],
            agreeing_agents=["a"],
            dissenting_agents=["b"],
        )

        # Store dissent
        dissent = memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="Different method",
            reasoning="More efficient",
            confidence=0.6,
        )

        assert dissent.id is not None
        assert dissent.debate_id == consensus.id

    def test_get_dissents(self, temp_db):
        """Test retrieving dissents for a debate."""
        memory = ConsensusMemory(str(temp_db))

        # Store consensus
        consensus = memory.store_consensus(
            topic="Test",
            conclusion="Result",
            strength=ConsensusStrength.WEAK,
            confidence=0.6,
            participating_agents=["a", "b", "c"],
            agreeing_agents=["a"],
        )

        # Store multiple dissents
        memory.store_dissent(
            debate_id=consensus.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Risk 1",
            reasoning="Reason 1",
        )
        memory.store_dissent(
            debate_id=consensus.id,
            agent_id="c",
            dissent_type=DissentType.EDGE_CASE_CONCERN,
            content="Edge case",
            reasoning="Reason 2",
        )

        # Retrieve
        dissents = memory.get_dissents(consensus.id)

        assert len(dissents) == 2

    def test_store_verified_proof(self, temp_db):
        """Test storing a verified proof."""
        memory = ConsensusMemory(str(temp_db))

        # Store consensus first
        consensus = memory.store_consensus(
            topic="Mathematical claim",
            conclusion="Proven true",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=0.99,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        # Store proof
        proof_result = {
            "status": "proof_found",
            "language": "lean4",
            "formal_statement": "theorem test : True := trivial",
            "is_verified": True,
            "proof_hash": "abc123",
        }

        proof_id = memory.store_verified_proof(consensus.id, proof_result)
        assert proof_id is not None

    def test_get_verified_proof(self, temp_db):
        """Test retrieving a verified proof."""
        memory = ConsensusMemory(str(temp_db))

        # Store consensus and proof
        consensus = memory.store_consensus(
            topic="Test",
            conclusion="Result",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["a"],
            agreeing_agents=["a"],
        )

        memory.store_verified_proof(
            consensus.id,
            {"status": "verified", "is_verified": True},
        )

        # Retrieve
        proof = memory.get_verified_proof(consensus.id)
        assert proof is not None
        assert proof["is_verified"] is True

    def test_list_verified_debates(self, temp_db):
        """Test listing verified debates."""
        memory = ConsensusMemory(str(temp_db))

        # Store multiple with different verification status
        c1 = memory.store_consensus(
            topic="T1", conclusion="C1", strength=ConsensusStrength.STRONG,
            confidence=0.9, participating_agents=["a"], agreeing_agents=["a"],
        )
        c2 = memory.store_consensus(
            topic="T2", conclusion="C2", strength=ConsensusStrength.STRONG,
            confidence=0.9, participating_agents=["a"], agreeing_agents=["a"],
        )

        memory.store_verified_proof(c1.id, {"status": "verified", "is_verified": True})
        memory.store_verified_proof(c2.id, {"status": "failed", "is_verified": False})

        # List verified only
        verified = memory.list_verified_debates(verified_only=True)
        assert len(verified) == 1

        # List all
        all_proofs = memory.list_verified_debates(verified_only=False)
        assert len(all_proofs) == 2

    def test_find_similar_debates(self, temp_db):
        """Test finding similar debates."""
        memory = ConsensusMemory(str(temp_db))

        # Store some debates
        memory.store_consensus(
            topic="Rate limiting implementation design",
            conclusion="Use token bucket",
            strength=ConsensusStrength.STRONG,
            confidence=0.85,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="architecture",
        )

        memory.store_consensus(
            topic="Authentication flow design",
            conclusion="Use OAuth2",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["a"],
            agreeing_agents=["a"],
            domain="security",
        )

        # Find similar to "rate limiting"
        similar = memory.find_similar_debates("rate limiting algorithm")

        assert len(similar) >= 1
        # Should find the rate limiting debate
        topics = [s.consensus.topic for s in similar]
        assert any("rate" in t.lower() for t in topics)

    def test_get_domain_consensus_history(self, temp_db):
        """Test getting domain-specific history."""
        memory = ConsensusMemory(str(temp_db))

        # Store debates in different domains
        memory.store_consensus(
            topic="Security T1", conclusion="C1", strength=ConsensusStrength.STRONG,
            confidence=0.9, participating_agents=["a"], agreeing_agents=["a"],
            domain="security",
        )
        memory.store_consensus(
            topic="Security T2", conclusion="C2", strength=ConsensusStrength.MODERATE,
            confidence=0.7, participating_agents=["a"], agreeing_agents=["a"],
            domain="security",
        )
        memory.store_consensus(
            topic="Perf T1", conclusion="C3", strength=ConsensusStrength.STRONG,
            confidence=0.8, participating_agents=["a"], agreeing_agents=["a"],
            domain="performance",
        )

        # Get security history
        history = memory.get_domain_consensus_history("security")

        assert len(history) == 2
        assert all(r.domain == "security" for r in history)

    def test_get_statistics(self, temp_db):
        """Test getting statistics."""
        memory = ConsensusMemory(str(temp_db))

        # Store some data
        c1 = memory.store_consensus(
            topic="T1", conclusion="C1", strength=ConsensusStrength.STRONG,
            confidence=0.9, participating_agents=["a", "b"], agreeing_agents=["a"],
            domain="security",
        )
        memory.store_dissent(
            debate_id=c1.id, agent_id="b", dissent_type=DissentType.RISK_WARNING,
            content="Risk", reasoning="Reason",
        )

        stats = memory.get_statistics()

        assert stats["total_consensus"] == 1
        assert stats["total_dissents"] == 1
        assert "security" in stats["by_domain"]

    def test_update_cruxes(self, temp_db):
        """Test updating belief cruxes."""
        memory = ConsensusMemory(str(temp_db))

        consensus = memory.store_consensus(
            topic="Test", conclusion="Result", strength=ConsensusStrength.MODERATE,
            confidence=0.7, participating_agents=["a"], agreeing_agents=["a"],
        )

        cruxes = [
            {"claim": "Claim 1", "resolution": "Resolved"},
            {"claim": "Claim 2", "resolution": "Agreed"},
        ]

        result = memory.update_cruxes(consensus.id, cruxes)
        assert result is True

        # Non-existent consensus
        result = memory.update_cruxes("nonexistent", cruxes)
        assert result is False


class TestDissentRetriever:
    """Test DissentRetriever class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_retriever.db"

    @pytest.fixture
    def memory_with_data(self, temp_db):
        """Create memory with test data."""
        memory = ConsensusMemory(str(temp_db))

        # Store some debates with dissents
        c1 = memory.store_consensus(
            topic="Security best practices",
            conclusion="Use encryption",
            strength=ConsensusStrength.STRONG,
            confidence=0.85,
            participating_agents=["a", "b"],
            agreeing_agents=["a"],
            domain="security",
        )
        memory.store_dissent(
            debate_id=c1.id,
            agent_id="b",
            dissent_type=DissentType.RISK_WARNING,
            content="Performance overhead from encryption",
            reasoning="May impact latency",
            confidence=0.7,
        )

        return memory

    def test_retrieve_for_new_debate(self, memory_with_data):
        """Test retrieving context for new debate."""
        retriever = DissentRetriever(memory_with_data)

        context = retriever.retrieve_for_new_debate(
            topic="security implementation",
            domain="security",
        )

        assert "similar_debates" in context
        assert "relevant_dissents" in context
        assert "total_similar" in context

    def test_find_contrarian_views(self, memory_with_data):
        """Test finding contrarian views."""
        retriever = DissentRetriever(memory_with_data)

        contrarian = retriever.find_contrarian_views(
            consensus_position="encryption is always good"
        )

        # Should return list of dissents
        assert isinstance(contrarian, list)

    def test_find_risk_warnings(self, memory_with_data):
        """Test finding risk warnings."""
        retriever = DissentRetriever(memory_with_data)

        warnings = retriever.find_risk_warnings(
            topic="security implementation",
            domain="security",
        )

        assert isinstance(warnings, list)

    def test_get_debate_preparation_context(self, memory_with_data):
        """Test generating preparation context."""
        retriever = DissentRetriever(memory_with_data)

        context = retriever.get_debate_preparation_context(
            topic="security best practices",
            domain="security",
        )

        assert isinstance(context, str)
        assert "Historical Context" in context


class TestConsensusMemoryCleanup:
    """Test cleanup and archival functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cleanup.db"

    def test_cleanup_old_records(self, temp_db):
        """Test cleaning up old records."""
        memory = ConsensusMemory(str(temp_db))

        # Store a record
        memory.store_consensus(
            topic="Test", conclusion="Result", strength=ConsensusStrength.MODERATE,
            confidence=0.7, participating_agents=["a"], agreeing_agents=["a"],
        )

        # Run cleanup (won't delete new records)
        result = memory.cleanup_old_records(max_age_days=90, archive=True)

        assert "archived" in result
        assert "deleted" in result
