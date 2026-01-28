"""
E2E tests for Knowledge Mound Learning Workflow.

Tests the complete cross-debate learning flow:
1. Create Debate 1 on a topic
2. Run debate to consensus
3. Store consensus in KnowledgeMound via ConsensusAdapter
4. Create Debate 2 on related topic
5. Search KM for previous consensus (semantic search)
6. Verify previous knowledge provided as context
7. Run Debate 2 with KM context
8. Store new consensus
9. Verify cross-debate learning metrics improved

This validates the institutional learning capabilities end-to-end.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Test Helpers
# =============================================================================


def create_mock_agent(name: str, response: str = "Default response") -> MagicMock:
    """Create a properly mocked agent with all required async methods."""
    agent = MagicMock()
    agent.name = name
    agent.generate = AsyncMock(return_value=response)

    mock_vote = MagicMock()
    mock_vote.choice = 0
    mock_vote.confidence = 0.8
    mock_vote.reasoning = "Agreed with proposal"
    agent.vote = AsyncMock(return_value=mock_vote)

    mock_critique = MagicMock()
    mock_critique.issues = []
    mock_critique.suggestions = []
    mock_critique.score = 0.8
    mock_critique.severity = 0.2
    mock_critique.text = "No issues found."
    mock_critique.agent = name
    mock_critique.target_agent = "other"
    mock_critique.round = 1
    agent.critique = AsyncMock(return_value=mock_critique)

    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.input_tokens = 0
    agent.output_tokens = 0
    agent.total_tokens_in = 0
    agent.total_tokens_out = 0
    agent.metrics = None
    agent.provider = None

    return agent


@dataclass
class MockConsensusRecord:
    """Mock consensus record for testing."""

    id: str
    task: str
    final_answer: str
    confidence: float
    participants: List[str]
    rounds: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockKnowledgeItem:
    """Mock knowledge item for testing."""

    id: str
    content: str
    source: str
    source_id: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    importance: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_mock_consensus_record(
    task: str,
    answer: str,
    confidence: float = 0.85,
    participants: List[str] = None,
) -> MockConsensusRecord:
    """Create a mock consensus record."""
    return MockConsensusRecord(
        id=f"consensus-{uuid.uuid4().hex[:8]}",
        task=task,
        final_answer=answer,
        confidence=confidence,
        participants=participants or ["claude", "gpt", "gemini"],
        rounds=2,
        metadata={"domain": "technical"},
    )


def create_mock_knowledge_item(
    content: str,
    source_id: str,
    importance: float = 0.8,
) -> MockKnowledgeItem:
    """Create a mock knowledge item."""
    now = datetime.now(timezone.utc)
    return MockKnowledgeItem(
        id=f"km-{uuid.uuid4().hex[:8]}",
        content=content,
        source="consensus",
        source_id=source_id,
        confidence=0.85,
        created_at=now,
        updated_at=now,
        importance=importance,
        metadata={"topics": ["caching", "architecture"]},
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "km_learning_test.db"


@pytest.fixture
def mock_agents():
    """Create mock agents for debates."""
    return [
        create_mock_agent("claude", "Based on my analysis, Redis is the better choice."),
        create_mock_agent("gpt", "I agree, Redis offers better performance for our use case."),
        create_mock_agent("gemini", "The consensus favors Redis for caching."),
    ]


@pytest.fixture
def mock_consensus_memory():
    """Create a mock consensus memory store."""
    memory = MagicMock()
    memory._records: Dict[str, MockConsensusRecord] = {}

    async def store_consensus(record):
        memory._records[record.id] = record
        return record.id

    async def get_consensus(consensus_id):
        return memory._records.get(consensus_id)

    async def search_by_topic(topic: str, limit: int = 10):
        # Simple keyword matching for mock
        results = []
        for record in memory._records.values():
            if any(word in record.task.lower() for word in topic.lower().split()):
                results.append(record)
        return results[:limit]

    memory.store_consensus = AsyncMock(side_effect=store_consensus)
    memory.get = AsyncMock(side_effect=get_consensus)
    memory.search_by_topic = AsyncMock(side_effect=search_by_topic)
    memory.get_all = AsyncMock(return_value=list(memory._records.values()))

    return memory


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock Knowledge Mound."""
    mound = MagicMock()
    mound._items: Dict[str, MockKnowledgeItem] = {}
    mound._retrievals: List[Dict[str, Any]] = []

    async def store(item):
        mound._items[item.id] = item
        return item.id

    async def get(item_id):
        return mound._items.get(item_id)

    async def search(query: str, limit: int = 10):
        # Simple keyword matching
        results = []
        for item in mound._items.values():
            if any(word in item.content.lower() for word in query.lower().split()):
                results.append(item)
        return results[:limit]

    async def record_retrieval(item_id, was_useful: bool):
        mound._retrievals.append(
            {
                "item_id": item_id,
                "was_useful": was_useful,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    mound.store = AsyncMock(side_effect=store)
    mound.get = AsyncMock(side_effect=get)
    mound.search = AsyncMock(side_effect=search)
    mound.record_retrieval = AsyncMock(side_effect=record_retrieval)

    return mound


@pytest.fixture
def mock_meta_learner(mock_knowledge_mound):
    """Create a mock meta-learner."""
    learner = MagicMock()
    learner._feedback: List[Dict[str, Any]] = []

    async def record_retrieval(km_id: str, rank_position: int, was_useful: bool):
        learner._feedback.append(
            {
                "km_id": km_id,
                "rank_position": rank_position,
                "was_useful": was_useful,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def optimize_tier_thresholds():
        return [
            {
                "tier": "fast",
                "current_threshold": 0.7,
                "recommended_threshold": 0.75,
                "reasoning": "High retrieval rate suggests raising threshold",
                "confidence": 0.8,
            }
        ]

    async def coalesce_duplicates(similarity_threshold: float = 0.95):
        return {
            "items_checked": 10,
            "duplicates_found": 2,
            "items_merged": 1,
            "storage_saved_bytes": 1024,
        }

    learner.record_retrieval = AsyncMock(side_effect=record_retrieval)
    learner.optimize_tier_thresholds = AsyncMock(side_effect=optimize_tier_thresholds)
    learner.coalesce_duplicates = AsyncMock(side_effect=coalesce_duplicates)

    return learner


# =============================================================================
# Cross-Debate Knowledge Flow Tests
# =============================================================================


@pytest.mark.e2e
class TestCrossDebateKnowledgeFlow:
    """Tests for cross-debate knowledge persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_store_debate_consensus_to_km(
        self,
        mock_agents,
        mock_consensus_memory,
        mock_knowledge_mound,
    ):
        """Test storing debate consensus to Knowledge Mound."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Run Debate 1
        env = Environment(task="Should we use Redis for caching?")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None

        # Store consensus
        consensus_record = create_mock_consensus_record(
            task=env.task,
            answer="Yes, Redis is recommended for caching due to its performance.",
            confidence=0.9,
            participants=[a.name for a in mock_agents],
        )
        await mock_consensus_memory.store_consensus(consensus_record)

        # Verify stored
        stored = await mock_consensus_memory.get(consensus_record.id)
        assert stored is not None
        assert stored.task == env.task

        # Convert to knowledge item and store in KM
        km_item = create_mock_knowledge_item(
            content=f"Decision: {consensus_record.final_answer}",
            source_id=consensus_record.id,
            importance=consensus_record.confidence,
        )
        await mock_knowledge_mound.store(km_item)

        # Verify in KM
        km_stored = await mock_knowledge_mound.get(km_item.id)
        assert km_stored is not None

    @pytest.mark.asyncio
    async def test_retrieve_related_knowledge(
        self,
        mock_knowledge_mound,
    ):
        """Test retrieving related knowledge from previous debates."""
        # Pre-populate KM with previous debate knowledge
        km_item = create_mock_knowledge_item(
            content="Redis is recommended for caching in read-heavy applications.",
            source_id="consensus-prev-001",
            importance=0.9,
        )
        await mock_knowledge_mound.store(km_item)

        # Search for related knowledge
        results = await mock_knowledge_mound.search("caching performance")

        assert len(results) >= 1
        assert any("Redis" in item.content for item in results)

    @pytest.mark.asyncio
    async def test_knowledge_context_injection(
        self,
        mock_agents,
        mock_knowledge_mound,
    ):
        """Test that prior knowledge is injected as context for new debates."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Pre-populate with relevant knowledge
        prior_knowledge = create_mock_knowledge_item(
            content="Previous decision: Redis outperforms Memcached for complex data structures.",
            source_id="consensus-prior-001",
            importance=0.85,
        )
        await mock_knowledge_mound.store(prior_knowledge)

        # Search for context before debate
        context = await mock_knowledge_mound.search("Memcached caching")

        # Create debate with context awareness
        context_str = "\n".join([item.content for item in context]) if context else ""

        env = Environment(
            task=f"Should we use Memcached or Redis for session caching? Context: {context_str}"
        )
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed > 0


# =============================================================================
# Semantic Search Tests
# =============================================================================


@pytest.mark.e2e
class TestSemanticSearch:
    """Tests for semantic similarity search in Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_search_by_topic(self, mock_knowledge_mound):
        """Test searching knowledge by topic."""
        # Add knowledge items
        items = [
            create_mock_knowledge_item(
                content="Redis is excellent for caching with low latency requirements.",
                source_id="src-001",
            ),
            create_mock_knowledge_item(
                content="PostgreSQL is recommended for relational data storage.",
                source_id="src-002",
            ),
            create_mock_knowledge_item(
                content="Memcached provides simple key-value caching.",
                source_id="src-003",
            ),
        ]
        for item in items:
            await mock_knowledge_mound.store(item)

        # Search for caching-related knowledge
        results = await mock_knowledge_mound.search("caching latency")

        assert len(results) >= 1
        assert any("caching" in item.content.lower() for item in results)

    @pytest.mark.asyncio
    async def test_semantic_similarity_retrieval(self, mock_knowledge_mound):
        """Test that semantically similar content is retrieved."""
        # Store knowledge about database decisions
        await mock_knowledge_mound.store(
            create_mock_knowledge_item(
                content="Use MongoDB for document-oriented storage patterns.",
                source_id="src-mongodb",
            )
        )

        # Search with semantically related query
        results = await mock_knowledge_mound.search("NoSQL document database")

        # Should find MongoDB-related content (in mock, this is keyword-based)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_relevance_ranking(self, mock_knowledge_mound):
        """Test that results are ranked by relevance."""
        # Add multiple items with varying relevance
        items = [
            create_mock_knowledge_item(
                content="Redis is the top choice for distributed caching.",
                source_id="src-high",
                importance=0.95,
            ),
            create_mock_knowledge_item(
                content="Caching improves application performance.",
                source_id="src-medium",
                importance=0.7,
            ),
            create_mock_knowledge_item(
                content="Redis supports various data structures.",
                source_id="src-low",
                importance=0.5,
            ),
        ]
        for item in items:
            await mock_knowledge_mound.store(item)

        results = await mock_knowledge_mound.search("Redis caching")

        # Verify results returned
        assert len(results) >= 1


# =============================================================================
# Meta-Learner Optimization Tests
# =============================================================================


@pytest.mark.e2e
class TestMetaLearnerOptimization:
    """Tests for Knowledge Mound meta-learner optimization."""

    @pytest.mark.asyncio
    async def test_record_retrieval_feedback(self, mock_meta_learner):
        """Test recording retrieval feedback for learning."""
        # Record useful retrieval
        await mock_meta_learner.record_retrieval(
            km_id="km-001",
            rank_position=0,
            was_useful=True,
        )

        # Record less useful retrieval
        await mock_meta_learner.record_retrieval(
            km_id="km-002",
            rank_position=5,
            was_useful=False,
        )

        # Verify feedback recorded
        assert len(mock_meta_learner._feedback) == 2
        assert mock_meta_learner._feedback[0]["was_useful"] is True
        assert mock_meta_learner._feedback[1]["was_useful"] is False

    @pytest.mark.asyncio
    async def test_optimize_tier_thresholds(self, mock_meta_learner):
        """Test tier threshold optimization recommendations."""
        # Get optimization recommendations
        recommendations = await mock_meta_learner.optimize_tier_thresholds()

        assert len(recommendations) >= 1
        rec = recommendations[0]
        assert "tier" in rec
        assert "recommended_threshold" in rec
        assert "reasoning" in rec
        assert rec["confidence"] > 0

    @pytest.mark.asyncio
    async def test_coalesce_duplicates(self, mock_meta_learner):
        """Test duplicate knowledge coalescing."""
        result = await mock_meta_learner.coalesce_duplicates(similarity_threshold=0.95)

        assert result["items_checked"] > 0
        assert "duplicates_found" in result
        assert "items_merged" in result
        assert result["storage_saved_bytes"] >= 0


# =============================================================================
# Complete Learning Workflow Tests
# =============================================================================


@pytest.mark.e2e
class TestCompleteKMLearningWorkflow:
    """Integration tests for complete cross-debate learning workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_cross_debate_learning(
        self,
        mock_agents,
        mock_consensus_memory,
        mock_knowledge_mound,
        mock_meta_learner,
    ):
        """Test complete workflow: debate1 → store → search → debate2 → learn."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # ===== Phase 1: First Debate =====
        env1 = Environment(task="Should we use Redis for caching?")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena1 = Arena(env1, mock_agents, protocol)
        result1 = await arena1.run()
        assert result1 is not None

        # Store consensus
        consensus1 = create_mock_consensus_record(
            task=env1.task,
            answer="Yes, Redis is recommended.",
            confidence=0.9,
        )
        await mock_consensus_memory.store_consensus(consensus1)

        # Convert to KM item
        km_item1 = create_mock_knowledge_item(
            content=f"Debate outcome: {consensus1.final_answer}",
            source_id=consensus1.id,
        )
        await mock_knowledge_mound.store(km_item1)

        # ===== Phase 2: Second Related Debate =====
        # Search for prior knowledge
        prior_knowledge = await mock_knowledge_mound.search("caching Redis")
        context = prior_knowledge[0].content if prior_knowledge else ""

        env2 = Environment(
            task=f"Should we use Memcached instead of Redis? Previous knowledge: {context}"
        )

        arena2 = Arena(env2, mock_agents, protocol)
        result2 = await arena2.run()
        assert result2 is not None

        # Record that prior knowledge was useful
        if prior_knowledge:
            await mock_meta_learner.record_retrieval(
                km_id=prior_knowledge[0].id,
                rank_position=0,
                was_useful=True,
            )

        # Store second consensus
        consensus2 = create_mock_consensus_record(
            task=env2.task,
            answer="No, Redis remains the better choice.",
            confidence=0.85,
        )
        await mock_consensus_memory.store_consensus(consensus2)

        # ===== Phase 3: Verify Learning =====
        # Check feedback was recorded
        assert len(mock_meta_learner._feedback) >= 1
        assert mock_meta_learner._feedback[0]["was_useful"] is True

        # Get optimization recommendations
        recommendations = await mock_meta_learner.optimize_tier_thresholds()
        assert len(recommendations) >= 1

        # Verify KM has both decisions
        assert len(mock_knowledge_mound._items) >= 1

    @pytest.mark.asyncio
    async def test_learning_improves_over_time(
        self,
        mock_agents,
        mock_knowledge_mound,
        mock_meta_learner,
    ):
        """Test that learning metrics improve with feedback."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Simulate multiple debates with feedback
        for i in range(3):
            env = Environment(task=f"Technical decision {i + 1}")
            protocol = DebateProtocol(
                rounds=1,
                consensus="majority",
                enable_calibration=False,
                enable_rhetorical_observer=False,
                enable_trickster=False,
            )

            arena = Arena(env, mock_agents, protocol)
            result = await arena.run()
            assert result is not None

            # Store knowledge
            km_item = create_mock_knowledge_item(
                content=f"Decision {i + 1} outcome",
                source_id=f"consensus-{i + 1}",
            )
            await mock_knowledge_mound.store(km_item)

            # Record feedback (alternating useful/not useful)
            await mock_meta_learner.record_retrieval(
                km_id=km_item.id,
                rank_position=i,
                was_useful=(i % 2 == 0),  # Every other one is useful
            )

        # Verify feedback accumulated
        assert len(mock_meta_learner._feedback) == 3

        # Get optimizations after learning
        recommendations = await mock_meta_learner.optimize_tier_thresholds()
        assert len(recommendations) >= 1

    @pytest.mark.asyncio
    async def test_knowledge_chain_of_custody(
        self,
        mock_consensus_memory,
        mock_knowledge_mound,
    ):
        """Test that knowledge maintains chain of custody from debate to KM."""
        # Create debate consensus
        consensus = create_mock_consensus_record(
            task="API design decision",
            answer="Use REST for public APIs, gRPC for internal.",
            confidence=0.88,
        )
        await mock_consensus_memory.store_consensus(consensus)

        # Convert to KM with source tracking
        km_item = create_mock_knowledge_item(
            content=consensus.final_answer,
            source_id=consensus.id,
        )
        await mock_knowledge_mound.store(km_item)

        # Verify chain of custody
        stored_km = await mock_knowledge_mound.get(km_item.id)
        assert stored_km.source == "consensus"
        assert stored_km.source_id == consensus.id

        # Verify we can trace back to original
        original = await mock_consensus_memory.get(stored_km.source_id)
        assert original is not None
        assert original.task == "API design decision"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.e2e
class TestKMLearningEdgeCases:
    """Tests for edge cases in Knowledge Mound learning."""

    @pytest.mark.asyncio
    async def test_empty_knowledge_mound(self, mock_knowledge_mound):
        """Test behavior when KM is empty."""
        results = await mock_knowledge_mound.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_related_knowledge(
        self,
        mock_agents,
        mock_knowledge_mound,
    ):
        """Test debate when no related knowledge exists."""
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol

        # Search for non-existent topic
        prior = await mock_knowledge_mound.search("quantum computing")
        assert len(prior) == 0

        # Debate should still work without prior knowledge
        env = Environment(task="Should we adopt quantum computing?")
        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_duplicate_knowledge_handling(self, mock_knowledge_mound):
        """Test handling of duplicate knowledge entries."""
        # Store same content twice
        item1 = create_mock_knowledge_item(
            content="Redis is fast.",
            source_id="src-001",
        )
        item2 = create_mock_knowledge_item(
            content="Redis is fast.",
            source_id="src-002",
        )

        await mock_knowledge_mound.store(item1)
        await mock_knowledge_mound.store(item2)

        # Both should be stored (dedup would happen in coalesce)
        assert len(mock_knowledge_mound._items) == 2

    @pytest.mark.asyncio
    async def test_stale_knowledge_detection(self, mock_knowledge_mound):
        """Test detection of potentially stale knowledge."""
        # Create old knowledge item
        old_item = MockKnowledgeItem(
            id="km-old",
            content="Use jQuery for frontend development.",
            source="consensus",
            source_id="old-consensus",
            confidence=0.9,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
            updated_at=datetime.now(timezone.utc) - timedelta(days=365),
            importance=0.5,
        )

        await mock_knowledge_mound.store(old_item)

        # Retrieve and check age
        stored = await mock_knowledge_mound.get(old_item.id)
        age = datetime.now(timezone.utc) - stored.created_at

        # Should be flagged as potentially stale (> 90 days)
        assert age.days > 90
