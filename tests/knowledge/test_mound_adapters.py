"""
Tests for Knowledge Mound adapters.

These tests verify that the adapters correctly bridge the existing memory
systems (ContinuumMemory, ConsensusMemory, CritiqueStore) to the Knowledge Mound.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# Mock the memory tier enum for testing
@dataclass
class MockMemoryTier:
    value: str


# Mock ContinuumMemoryEntry for testing
@dataclass
class MockContinuumEntry:
    id: str
    tier: MockMemoryTier
    content: str
    importance: float
    surprise_score: float
    consolidation_score: float
    update_count: int
    success_count: int
    failure_count: int
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = None
    red_line: bool = False
    red_line_reason: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def knowledge_mound_id(self) -> str:
        return f"cm_{self.id}"

    @property
    def tags(self) -> List[str]:
        return self.metadata.get("tags", [])

    @property
    def cross_references(self) -> List[str]:
        return self.metadata.get("cross_references", [])


# Mock ConsensusStrength enum
class MockConsensusStrength:
    def __init__(self, value: str):
        self.value = value


# Mock ConsensusRecord for testing
@dataclass
class MockConsensusRecord:
    id: str
    topic: str
    topic_hash: str
    conclusion: str
    strength: MockConsensusStrength
    confidence: float
    participating_agents: List[str] = None
    agreeing_agents: List[str] = None
    dissenting_agents: List[str] = None
    key_claims: List[str] = None
    supporting_evidence: List[str] = None
    dissent_ids: List[str] = None
    domain: str = "general"
    tags: List[str] = None
    timestamp: datetime = None
    debate_duration_seconds: float = 0.0
    rounds: int = 0
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.participating_agents is None:
            self.participating_agents = []
        if self.agreeing_agents is None:
            self.agreeing_agents = []
        if self.dissenting_agents is None:
            self.dissenting_agents = []
        if self.key_claims is None:
            self.key_claims = []
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.dissent_ids is None:
            self.dissent_ids = []
        if self.tags is None:
            self.tags = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def compute_agreement_ratio(self) -> float:
        total = len(self.participating_agents)
        if total == 0:
            return 0.0
        return len(self.agreeing_agents) / total


# Mock SimilarDebate for testing
@dataclass
class MockSimilarDebate:
    consensus: MockConsensusRecord
    dissents: List[Any]
    similarity: float


# Mock Pattern for testing
@dataclass
class MockPattern:
    id: str
    issue_type: str
    issue_text: str
    suggestion_text: str
    success_count: int
    failure_count: int
    avg_severity: float
    example_task: str
    created_at: str
    updated_at: str

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


class TestContinuumAdapter:
    """Tests for ContinuumAdapter."""

    def test_search_by_keyword(self):
        """Test keyword search wraps retrieve correctly."""
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        # Create mock entry
        mock_entry = MockContinuumEntry(
            id="entry_1",
            tier=MockMemoryTier("slow"),
            content="TypeError in agent response handling",
            importance=0.8,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=5,
            success_count=4,
            failure_count=1,
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        # Create mock ContinuumMemory
        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = [mock_entry]

        adapter = ContinuumAdapter(mock_continuum)
        results = adapter.search_by_keyword("TypeError", limit=10)

        assert len(results) == 1
        assert results[0].id == "entry_1"
        mock_continuum.retrieve.assert_called_once()

    def test_get_with_prefix(self):
        """Test get strips cm_ prefix."""
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        mock_entry = MockContinuumEntry(
            id="entry_1",
            tier=MockMemoryTier("slow"),
            content="Test content",
            importance=0.5,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=1,
            success_count=1,
            failure_count=0,
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        mock_continuum = MagicMock()
        mock_continuum.get.return_value = mock_entry

        adapter = ContinuumAdapter(mock_continuum)

        # Call with prefix
        result = adapter.get("cm_entry_1")

        # Should strip prefix
        mock_continuum.get.assert_called_with("entry_1")
        assert result.id == "entry_1"

    def test_to_knowledge_item(self):
        """Test conversion to KnowledgeItem."""
        from aragora.knowledge.mound.adapters import ContinuumAdapter
        from aragora.knowledge.mound.types import KnowledgeSource

        mock_entry = MockContinuumEntry(
            id="entry_1",
            tier=MockMemoryTier("slow"),
            content="Test content for conversion",
            importance=0.7,
            surprise_score=0.2,
            consolidation_score=0.6,
            update_count=10,
            success_count=8,
            failure_count=2,
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
            metadata={"tags": ["test", "example"]},
        )

        mock_continuum = MagicMock()
        adapter = ContinuumAdapter(mock_continuum)

        item = adapter.to_knowledge_item(mock_entry)

        assert item.id == "cm_entry_1"
        assert item.content == "Test content for conversion"
        assert item.source == KnowledgeSource.CONTINUUM
        assert item.source_id == "entry_1"
        assert item.importance == 0.7
        assert item.metadata["tier"] == "slow"
        assert item.metadata["surprise_score"] == 0.2
        assert item.metadata["success_rate"] == 0.8

    def test_store(self):
        """Test storing content."""
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        mock_continuum = MagicMock()
        adapter = ContinuumAdapter(mock_continuum)

        entry_id = adapter.store(
            content="New learning pattern",
            importance=0.6,
            tier="medium",
            entry_id="custom_id",
        )

        assert entry_id == "custom_id"
        mock_continuum.add.assert_called_once()
        call_kwargs = mock_continuum.add.call_args
        assert call_kwargs.kwargs["content"] == "New learning pattern"
        assert call_kwargs.kwargs["importance"] == 0.6

    def test_get_stats(self):
        """Test stats retrieval."""
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        mock_continuum = MagicMock()
        mock_continuum.get_stats.return_value = {"total_entries": 100}

        adapter = ContinuumAdapter(mock_continuum)
        stats = adapter.get_stats()

        assert stats["total_entries"] == 100


class TestConsensusAdapter:
    """Tests for ConsensusAdapter."""

    @pytest.mark.asyncio
    async def test_search_by_topic(self):
        """Test topic search wraps find_similar_debates correctly."""
        from aragora.knowledge.mound.adapters import ConsensusAdapter

        mock_record = MockConsensusRecord(
            id="consensus_1",
            topic="rate limiting implementation",
            topic_hash="abc123",
            conclusion="Use token bucket algorithm",
            strength=MockConsensusStrength("strong"),
            confidence=0.9,
            participating_agents=["claude", "gpt4"],
            agreeing_agents=["claude", "gpt4"],
        )

        mock_debate = MockSimilarDebate(
            consensus=mock_record,
            dissents=[],
            similarity=0.85,
        )

        mock_consensus = MagicMock()
        mock_consensus.find_similar_debates.return_value = [mock_debate]

        adapter = ConsensusAdapter(mock_consensus)
        results = await adapter.search_by_topic("rate limiting", limit=10)

        assert len(results) == 1
        assert results[0].record.id == "consensus_1"
        assert results[0].similarity == 0.85

    def test_get_with_prefix(self):
        """Test get strips cs_ prefix."""
        from aragora.knowledge.mound.adapters import ConsensusAdapter

        mock_record = MockConsensusRecord(
            id="consensus_1",
            topic="Test topic",
            topic_hash="abc",
            conclusion="Test conclusion",
            strength=MockConsensusStrength("moderate"),
            confidence=0.7,
        )

        mock_consensus = MagicMock()
        mock_consensus.get_consensus.return_value = mock_record

        adapter = ConsensusAdapter(mock_consensus)
        result = adapter.get("cs_consensus_1")

        mock_consensus.get_consensus.assert_called_with("consensus_1")
        assert result.id == "consensus_1"

    def test_to_knowledge_item(self):
        """Test conversion to KnowledgeItem."""
        from aragora.knowledge.mound.adapters import ConsensusAdapter
        from aragora.knowledge.mound.types import KnowledgeSource, ConfidenceLevel

        mock_record = MockConsensusRecord(
            id="consensus_1",
            topic="API design patterns",
            topic_hash="hash123",
            conclusion="Use RESTful conventions for public APIs",
            strength=MockConsensusStrength("strong"),
            confidence=0.85,
            participating_agents=["claude", "gpt4", "gemini"],
            agreeing_agents=["claude", "gpt4", "gemini"],
            domain="architecture",
            tags=["api", "design"],
        )

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        item = adapter.to_knowledge_item(mock_record)

        assert item.id == "cs_consensus_1"
        assert item.content == "Use RESTful conventions for public APIs"
        assert item.source == KnowledgeSource.CONSENSUS
        assert item.confidence == ConfidenceLevel.HIGH
        assert item.metadata["strength"] == "strong"
        assert item.metadata["domain"] == "architecture"
        assert item.metadata["agreement_ratio"] == 1.0

    def test_to_knowledge_item_with_search_result(self):
        """Test conversion handles ConsensusSearchResult wrapper."""
        from aragora.knowledge.mound.adapters import ConsensusAdapter, ConsensusSearchResult

        mock_record = MockConsensusRecord(
            id="consensus_2",
            topic="Testing strategies",
            topic_hash="hash456",
            conclusion="Prefer integration tests for APIs",
            strength=MockConsensusStrength("moderate"),
            confidence=0.7,
        )

        search_result = ConsensusSearchResult(
            record=mock_record,
            similarity=0.75,
            dissents=[],
        )

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        item = adapter.to_knowledge_item(search_result)

        assert item.id == "cs_consensus_2"
        assert item.metadata["similarity"] == 0.75


class TestCritiqueAdapter:
    """Tests for CritiqueAdapter."""

    def test_search_patterns_with_type(self):
        """Test pattern search with explicit type."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        mock_pattern = MockPattern(
            id="pattern_1",
            issue_type="performance",
            issue_text="Slow database query in hot path",
            suggestion_text="Add index on frequently queried columns",
            success_count=10,
            failure_count=2,
            avg_severity=0.7,
            example_task="Optimize user lookup",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = [mock_pattern]

        adapter = CritiqueAdapter(mock_store)
        results = adapter.search_patterns("performance", limit=10)

        assert len(results) == 1
        assert results[0].id == "pattern_1"
        mock_store.retrieve_patterns.assert_called_once()

    def test_search_patterns_keyword_filter(self):
        """Test pattern search with keyword filtering."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        pattern1 = MockPattern(
            id="pattern_1",
            issue_type="general",
            issue_text="Database connection timeout",
            suggestion_text="Increase timeout value",
            success_count=5,
            failure_count=1,
            avg_severity=0.5,
            example_task="Fix timeout issue",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )
        pattern2 = MockPattern(
            id="pattern_2",
            issue_type="general",
            issue_text="Memory leak in worker",
            suggestion_text="Add cleanup handler",
            success_count=8,
            failure_count=2,
            avg_severity=0.6,
            example_task="Fix memory issue",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = [pattern1, pattern2]

        adapter = CritiqueAdapter(mock_store)
        results = adapter.search_patterns("database timeout", limit=10)

        # Should filter to only pattern1 which contains "database" and "timeout"
        assert len(results) == 1
        assert results[0].id == "pattern_1"

    def test_to_knowledge_item(self):
        """Test conversion to KnowledgeItem."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter
        from aragora.knowledge.mound.types import KnowledgeSource, ConfidenceLevel

        mock_pattern = MockPattern(
            id="pattern_1",
            issue_type="security",
            issue_text="SQL injection vulnerability",
            suggestion_text="Use parameterized queries",
            success_count=15,
            failure_count=1,
            avg_severity=0.9,
            example_task="Fix login endpoint",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        mock_store = MagicMock()
        adapter = CritiqueAdapter(mock_store)

        item = adapter.to_knowledge_item(mock_pattern)

        assert item.id == "cr_pattern_1"
        assert "SQL injection vulnerability" in item.content
        assert "parameterized queries" in item.content
        assert item.source == KnowledgeSource.CRITIQUE
        assert item.confidence == ConfidenceLevel.HIGH  # success_rate = 0.9375
        assert item.metadata["issue_type"] == "security"
        assert item.metadata["success_count"] == 15

    def test_get_agent_reputation(self):
        """Test agent reputation retrieval."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        mock_reputation = MagicMock()
        mock_reputation.agent_name = "claude"
        mock_reputation.reputation_score = 0.85

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = mock_reputation

        adapter = CritiqueAdapter(mock_store)
        rep = adapter.get_agent_reputation("claude")

        assert rep.agent_name == "claude"
        assert rep.reputation_score == 0.85

    def test_get_vote_weights(self):
        """Test batch vote weight retrieval."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        mock_store = MagicMock()
        mock_store.get_vote_weights_batch.return_value = {
            "claude": 1.4,
            "gpt4": 1.2,
            "gemini": 1.0,
        }

        adapter = CritiqueAdapter(mock_store)
        weights = adapter.get_agent_vote_weights(["claude", "gpt4", "gemini"])

        assert weights["claude"] == 1.4
        assert weights["gpt4"] == 1.2
        assert weights["gemini"] == 1.0

    def test_get_stats(self):
        """Test stats retrieval."""
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_patterns": 150,
            "total_debates": 500,
        }

        adapter = CritiqueAdapter(mock_store)
        stats = adapter.get_stats()

        assert stats["total_patterns"] == 150
        assert stats["total_debates"] == 500


class TestAdapterIntegration:
    """Integration tests for adapter interactions."""

    def test_adapter_imports(self):
        """Test that all adapters can be imported."""
        from aragora.knowledge.mound.adapters import (
            ContinuumAdapter,
            ConsensusAdapter,
            CritiqueAdapter,
        )

        assert ContinuumAdapter is not None
        assert ConsensusAdapter is not None
        assert CritiqueAdapter is not None

    def test_knowledge_item_ids_unique(self):
        """Test that adapters generate unique prefixed IDs."""
        from aragora.knowledge.mound.adapters import (
            ContinuumAdapter,
            ConsensusAdapter,
            CritiqueAdapter,
        )

        # Create mock entries
        continuum_entry = MockContinuumEntry(
            id="shared_id",
            tier=MockMemoryTier("slow"),
            content="Content",
            importance=0.5,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=1,
            success_count=1,
            failure_count=0,
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        consensus_record = MockConsensusRecord(
            id="shared_id",
            topic="Topic",
            topic_hash="hash",
            conclusion="Conclusion",
            strength=MockConsensusStrength("strong"),
            confidence=0.8,
        )

        critique_pattern = MockPattern(
            id="shared_id",
            issue_type="general",
            issue_text="Issue",
            suggestion_text="Suggestion",
            success_count=5,
            failure_count=1,
            avg_severity=0.5,
            example_task="Task",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-16T10:00:00",
        )

        # Create adapters with mocks
        continuum_adapter = ContinuumAdapter(MagicMock())
        consensus_adapter = ConsensusAdapter(MagicMock())
        critique_adapter = CritiqueAdapter(MagicMock())

        # Convert to knowledge items
        item1 = continuum_adapter.to_knowledge_item(continuum_entry)
        item2 = consensus_adapter.to_knowledge_item(consensus_record)
        item3 = critique_adapter.to_knowledge_item(critique_pattern)

        # All should have different prefixed IDs even with same base ID
        assert item1.id == "cm_shared_id"
        assert item2.id == "cs_shared_id"
        assert item3.id == "cr_shared_id"
        assert len({item1.id, item2.id, item3.id}) == 3
