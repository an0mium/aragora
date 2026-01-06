"""
Tests for aragora.memory.streams module.

Tests MemoryStream, Memory, and RetrievedMemory classes.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from aragora.memory.streams import Memory, MemoryStream, RetrievedMemory


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
def memory_stream(temp_db):
    """MemoryStream without embedding provider."""
    return MemoryStream(db_path=temp_db, embedding_provider=None)


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    provider = AsyncMock()
    # Return consistent embeddings for testing
    provider.embed = AsyncMock(return_value=[0.1] * 256)
    return provider


@pytest.fixture
def sample_memory():
    """Sample Memory for testing."""
    return Memory(
        id="mem-001",
        agent_name="test-agent",
        memory_type="observation",
        content="The debate went well",
        importance=0.7,
        created_at=datetime.now().isoformat(),
        debate_id="debate-001",
        metadata={"round": 1},
    )


# =============================================================================
# Test Memory Dataclass
# =============================================================================


class TestMemory:
    """Tests for Memory dataclass."""

    def test_creation_with_all_fields(self, sample_memory):
        """Should create memory with all fields."""
        assert sample_memory.id == "mem-001"
        assert sample_memory.agent_name == "test-agent"
        assert sample_memory.memory_type == "observation"
        assert sample_memory.content == "The debate went well"
        assert sample_memory.importance == 0.7
        assert sample_memory.debate_id == "debate-001"
        assert sample_memory.metadata == {"round": 1}

    def test_age_hours_recent(self):
        """Should return small age for recent memory."""
        memory = Memory(
            id="test",
            agent_name="agent",
            memory_type="observation",
            content="test",
            importance=0.5,
            created_at=datetime.now().isoformat(),
        )
        # Should be less than 1 minute old
        assert memory.age_hours < 0.02

    def test_age_hours_old(self):
        """Should return correct age for old memory."""
        old_time = datetime.now() - timedelta(hours=48)
        memory = Memory(
            id="test",
            agent_name="agent",
            memory_type="observation",
            content="test",
            importance=0.5,
            created_at=old_time.isoformat(),
        )
        # Should be approximately 48 hours
        assert 47 < memory.age_hours < 49

    def test_default_metadata(self):
        """Should default metadata to empty dict."""
        memory = Memory(
            id="test",
            agent_name="agent",
            memory_type="observation",
            content="test",
            importance=0.5,
            created_at=datetime.now().isoformat(),
        )
        assert memory.metadata == {}

    def test_default_debate_id(self):
        """Should default debate_id to None."""
        memory = Memory(
            id="test",
            agent_name="agent",
            memory_type="observation",
            content="test",
            importance=0.5,
            created_at=datetime.now().isoformat(),
        )
        assert memory.debate_id is None


# =============================================================================
# Test RetrievedMemory Dataclass
# =============================================================================


class TestRetrievedMemory:
    """Tests for RetrievedMemory dataclass."""

    def test_creation(self, sample_memory):
        """Should create RetrievedMemory with scores."""
        retrieved = RetrievedMemory(
            memory=sample_memory,
            recency_score=0.9,
            importance_score=0.7,
            relevance_score=0.8,
        )
        assert retrieved.memory == sample_memory
        assert retrieved.recency_score == 0.9
        assert retrieved.importance_score == 0.7
        assert retrieved.relevance_score == 0.8

    def test_total_score_weighting(self, sample_memory):
        """Should apply 30/30/40 weighting."""
        retrieved = RetrievedMemory(
            memory=sample_memory,
            recency_score=1.0,
            importance_score=1.0,
            relevance_score=1.0,
        )
        # All 1.0 should give total of 1.0
        assert retrieved.total_score == pytest.approx(1.0, abs=0.01)

    def test_high_recency_dominates(self, sample_memory):
        """Recency has 30% weight."""
        retrieved = RetrievedMemory(
            memory=sample_memory,
            recency_score=1.0,
            importance_score=0.0,
            relevance_score=0.0,
        )
        assert retrieved.total_score == pytest.approx(0.3, abs=0.01)

    def test_high_importance_dominates(self, sample_memory):
        """Importance has 30% weight."""
        retrieved = RetrievedMemory(
            memory=sample_memory,
            recency_score=0.0,
            importance_score=1.0,
            relevance_score=0.0,
        )
        assert retrieved.total_score == pytest.approx(0.3, abs=0.01)

    def test_high_relevance_dominates(self, sample_memory):
        """Relevance has 40% weight (highest)."""
        retrieved = RetrievedMemory(
            memory=sample_memory,
            recency_score=0.0,
            importance_score=0.0,
            relevance_score=1.0,
        )
        assert retrieved.total_score == pytest.approx(0.4, abs=0.01)


# =============================================================================
# Test MemoryStream Database
# =============================================================================


class TestMemoryStreamDatabase:
    """Tests for MemoryStream database operations."""

    def test_init_creates_tables(self, temp_db):
        """Should create memories and reflection_schedule tables."""
        MemoryStream(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert "memories" in tables
        assert "reflection_schedule" in tables

    def test_init_creates_indices(self, temp_db):
        """Should create indices for efficient queries."""
        MemoryStream(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indices = [row[0] for row in cursor.fetchall()]

        assert "idx_memories_agent" in indices
        assert "idx_memories_type" in indices

    def test_generate_id_is_deterministic_with_same_timestamp(self, memory_stream):
        """Same inputs at same time should produce same ID (when timestamp mocked)."""
        with patch("aragora.memory.streams.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            id1 = memory_stream._generate_id("agent", "content")
            id2 = memory_stream._generate_id("agent", "content")
            assert id1 == id2

    def test_generate_id_length(self, memory_stream):
        """Generated ID should be 16 characters."""
        mem_id = memory_stream._generate_id("agent", "content")
        assert len(mem_id) == 16


# =============================================================================
# Test MemoryStream Add Operations
# =============================================================================


class TestMemoryStreamAdd:
    """Tests for MemoryStream add operations."""

    def test_add_returns_memory(self, memory_stream):
        """Should return Memory object."""
        memory = memory_stream.add(
            agent_name="test-agent",
            content="Test observation",
            memory_type="observation",
            importance=0.5,
        )

        assert isinstance(memory, Memory)
        assert memory.agent_name == "test-agent"
        assert memory.content == "Test observation"
        assert memory.memory_type == "observation"

    def test_add_persists_to_database(self, memory_stream, temp_db):
        """Should persist memory to database."""
        memory_stream.add(
            agent_name="test-agent",
            content="Persisted content",
        )

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]

        assert count == 1

    def test_add_increments_reflection_counter(self, memory_stream, temp_db):
        """Should increment memories_since_reflection."""
        memory_stream.add(agent_name="test-agent", content="First")
        memory_stream.add(agent_name="test-agent", content="Second")
        memory_stream.add(agent_name="test-agent", content="Third")

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memories_since_reflection FROM reflection_schedule WHERE agent_name = ?",
                ("test-agent",),
            )
            count = cursor.fetchone()[0]

        assert count == 3

    def test_observe_sets_observation_type(self, memory_stream):
        """observe() should set memory_type to 'observation'."""
        memory = memory_stream.observe("agent", "Content")
        assert memory.memory_type == "observation"

    def test_observe_default_importance(self, memory_stream):
        """observe() should default importance to 0.5."""
        memory = memory_stream.observe("agent", "Content")
        assert memory.importance == 0.5

    def test_reflect_sets_reflection_type(self, memory_stream):
        """reflect() should set memory_type to 'reflection'."""
        memory = memory_stream.reflect("agent", "Content")
        assert memory.memory_type == "reflection"

    def test_reflect_default_importance(self, memory_stream):
        """reflect() should default importance to 0.7."""
        memory = memory_stream.reflect("agent", "Content")
        assert memory.importance == 0.7

    def test_insight_sets_insight_type(self, memory_stream):
        """insight() should set memory_type to 'insight'."""
        memory = memory_stream.insight("agent", "Content")
        assert memory.memory_type == "insight"

    def test_insight_default_importance(self, memory_stream):
        """insight() should default importance to 0.9."""
        memory = memory_stream.insight("agent", "Content")
        assert memory.importance == 0.9


# =============================================================================
# Test MemoryStream Retrieval
# =============================================================================


class TestMemoryStreamRetrieval:
    """Tests for MemoryStream retrieval operations."""

    def test_retrieve_returns_list(self, memory_stream):
        """Should return list of RetrievedMemory."""
        memory_stream.add("agent", "Test content")
        results = memory_stream.retrieve("agent")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], RetrievedMemory)

    def test_retrieve_filters_by_agent(self, memory_stream):
        """Should only return memories for specified agent."""
        memory_stream.add("agent1", "Content 1")
        memory_stream.add("agent2", "Content 2")

        results = memory_stream.retrieve("agent1")
        for r in results:
            assert r.memory.agent_name == "agent1"

    def test_retrieve_filters_by_type(self, memory_stream):
        """Should filter by memory_type."""
        memory_stream.observe("agent", "Observation")
        memory_stream.insight("agent", "Insight")

        results = memory_stream.retrieve("agent", memory_type="insight")
        for r in results:
            assert r.memory.memory_type == "insight"

    def test_retrieve_filters_by_min_importance(self, memory_stream):
        """Should filter by minimum importance."""
        memory_stream.add("agent", "Low importance", importance=0.2)
        memory_stream.add("agent", "High importance", importance=0.8)

        results = memory_stream.retrieve("agent", min_importance=0.5)
        for r in results:
            assert r.memory.importance >= 0.5

    def test_retrieve_respects_limit(self, memory_stream):
        """Should respect limit parameter."""
        for i in range(20):
            memory_stream.add("agent", f"Memory {i}")

        results = memory_stream.retrieve("agent", limit=5)
        assert len(results) <= 5

    def test_recency_score_new_memory(self, memory_stream):
        """New memory should have recency score close to 1.0."""
        memory_stream.add("agent", "New content")
        results = memory_stream.retrieve("agent")

        assert len(results) >= 1
        assert results[0].recency_score > 0.99

    def test_recency_score_decays(self, memory_stream, temp_db):
        """Older memories should have lower recency scores."""
        # Add memory with old timestamp
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (id, agent_name, memory_type, content, importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("old-mem", "agent", "observation", "Old content", 0.5, old_time),
            )
            conn.commit()

        results = memory_stream.retrieve("agent")

        # Find the old memory
        old_result = [r for r in results if r.memory.id == "old-mem"]
        assert len(old_result) == 1
        # After 48 hours, recency should be about 0.25 (0.5^2)
        assert old_result[0].recency_score < 0.5

    def test_relevance_score_keyword_fallback(self, memory_stream):
        """Should use keyword matching when no embedding provider."""
        memory_stream.add("agent", "Python programming is great")

        results = memory_stream.retrieve("agent", query="Python")

        assert len(results) >= 1
        # Should have some relevance due to keyword match
        assert results[0].relevance_score > 0

    def test_relevance_score_no_query(self, memory_stream):
        """Should return 0.5 relevance when no query."""
        memory_stream.add("agent", "Some content")

        results = memory_stream.retrieve("agent")

        assert len(results) >= 1
        assert results[0].relevance_score == 0.5

    def test_get_recent_orders_by_created_at(self, memory_stream):
        """Should order by created_at descending."""
        memory_stream.add("agent", "First")
        memory_stream.add("agent", "Second")
        memory_stream.add("agent", "Third")

        results = memory_stream.get_recent("agent")

        # Most recent should be first
        assert results[0].content == "Third"


# =============================================================================
# Test MemoryStream Reflection
# =============================================================================


class TestMemoryStreamReflection:
    """Tests for MemoryStream reflection cycle."""

    def test_should_reflect_false_initially(self, memory_stream):
        """Should return False when no memories recorded."""
        assert memory_stream.should_reflect("agent") is False

    def test_should_reflect_false_below_threshold(self, memory_stream):
        """Should return False when below threshold."""
        for i in range(5):
            memory_stream.add("agent", f"Memory {i}")

        assert memory_stream.should_reflect("agent", threshold=10) is False

    def test_should_reflect_true_at_threshold(self, memory_stream):
        """Should return True when at or above threshold."""
        for i in range(10):
            memory_stream.add("agent", f"Memory {i}")

        assert memory_stream.should_reflect("agent", threshold=10) is True

    def test_mark_reflected_resets_counter(self, memory_stream, temp_db):
        """Should reset memories_since_reflection to 0."""
        for i in range(10):
            memory_stream.add("agent", f"Memory {i}")

        memory_stream.mark_reflected("agent")

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memories_since_reflection FROM reflection_schedule WHERE agent_name = ?",
                ("agent",),
            )
            count = cursor.fetchone()[0]

        assert count == 0

    def test_generate_reflection_prompt_includes_memories(self, memory_stream):
        """Should include recent memories in prompt."""
        memory_stream.observe("agent", "Made a good critique")
        memory_stream.observe("agent", "Found an edge case")

        prompt = memory_stream.generate_reflection_prompt("agent")

        assert "Made a good critique" in prompt
        assert "Found an edge case" in prompt

    def test_generate_reflection_prompt_empty(self, memory_stream):
        """Should return empty string when no memories."""
        prompt = memory_stream.generate_reflection_prompt("agent")
        assert prompt == ""

    def test_parse_reflections_extracts_insights(self, memory_stream):
        """Should extract lines starting with INSIGHT:."""
        response = """
        Some preamble text.
        INSIGHT: I notice I'm good at finding edge cases.
        More text here.
        INSIGHT: I should focus more on clarity.
        """

        insights = memory_stream.parse_reflections("agent", response)

        assert len(insights) == 2
        assert insights[0].memory_type == "insight"
        assert "edge cases" in insights[0].content

    def test_parse_reflections_creates_memories(self, memory_stream):
        """Should create insight memories."""
        response = "INSIGHT: Testing is important."

        insights = memory_stream.parse_reflections("agent", response)

        assert len(insights) == 1
        assert insights[0].memory_type == "insight"
        assert insights[0].importance == 0.9

    def test_parse_reflections_marks_reflected(self, memory_stream):
        """Should call mark_reflected when insights found."""
        for i in range(10):
            memory_stream.add("agent", f"Memory {i}")

        response = "INSIGHT: Something learned."
        memory_stream.parse_reflections("agent", response)

        # Counter should be reset
        assert memory_stream.should_reflect("agent", threshold=10) is False

    def test_parse_reflections_empty_response(self, memory_stream):
        """Should return empty list for response without insights."""
        response = "Just some text without insights."
        insights = memory_stream.parse_reflections("agent", response)
        assert insights == []


# =============================================================================
# Test MemoryStream Context and Stats
# =============================================================================


class TestMemoryStreamContextAndStats:
    """Tests for context generation and statistics."""

    def test_get_context_for_debate_formats_correctly(self, memory_stream):
        """Should format context with labels."""
        memory_stream.insight("agent", "Testing is crucial")
        memory_stream.reflect("agent", "Focus on edge cases")
        memory_stream.observe("agent", "Found a bug")

        context = memory_stream.get_context_for_debate("agent", "code review")

        assert "Relevant past experience:" in context

    def test_get_context_labels_by_type(self, memory_stream):
        """Should label memories by type."""
        memory_stream.insight("agent", "Insight content")
        memory_stream.reflect("agent", "Reflection content")
        memory_stream.observe("agent", "Observation content")

        context = memory_stream.get_context_for_debate("agent", "task")

        assert "[Insight]" in context
        assert "[Learning]" in context
        assert "[Experience]" in context

    def test_get_context_empty(self, memory_stream):
        """Should return empty string when no memories."""
        context = memory_stream.get_context_for_debate("agent", "task")
        assert context == ""

    def test_get_stats_counts_by_type(self, memory_stream):
        """Should count memories by type."""
        memory_stream.observe("agent", "Obs 1")
        memory_stream.observe("agent", "Obs 2")
        memory_stream.reflect("agent", "Ref 1")
        memory_stream.insight("agent", "Ins 1")

        stats = memory_stream.get_stats("agent")

        assert stats["total_memories"] == 4
        assert stats["observations"] == 2
        assert stats["reflections"] == 1
        assert stats["insights"] == 1

    def test_get_stats_calculates_avg_importance(self, memory_stream):
        """Should calculate average importance."""
        memory_stream.add("agent", "Low", importance=0.2)
        memory_stream.add("agent", "High", importance=0.8)

        stats = memory_stream.get_stats("agent")

        assert stats["avg_importance"] == pytest.approx(0.5, abs=0.01)

    def test_get_stats_empty_agent(self, memory_stream):
        """Should handle agent with no memories."""
        stats = memory_stream.get_stats("nonexistent")

        assert stats["total_memories"] == 0
        assert stats["observations"] == 0
        assert stats["reflections"] == 0
        assert stats["insights"] == 0
        assert stats["avg_importance"] == 0.0


# =============================================================================
# Test MemoryStream with Embedding Provider
# =============================================================================


class TestMemoryStreamWithEmbeddings:
    """Tests for MemoryStream with embedding provider."""

    def test_relevance_uses_embeddings_when_available(
        self, temp_db, mock_embedding_provider
    ):
        """Should use embedding provider for relevance scoring."""
        stream = MemoryStream(
            db_path=temp_db,
            embedding_provider=mock_embedding_provider,
        )

        with patch.object(stream, "_embedding_similarity", return_value=0.85):
            stream.add("agent", "Python code")
            results = stream.retrieve("agent", query="Python")

            # Should have used embeddings (mocked to return 0.85)
            assert len(results) >= 1

    def test_embedding_fallback_on_error(self, temp_db, mock_embedding_provider):
        """Should fall back to keywords if embedding fails."""
        mock_embedding_provider.embed.side_effect = Exception("API error")

        stream = MemoryStream(
            db_path=temp_db,
            embedding_provider=mock_embedding_provider,
        )

        stream.add("agent", "Python programming language")
        results = stream.retrieve("agent", query="Python")

        # Should still work via keyword fallback
        assert len(results) >= 1
        assert results[0].relevance_score > 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestMemoryStreamEdgeCases:
    """Tests for edge cases and error handling."""

    def test_special_characters_in_content(self, memory_stream):
        """Should handle special characters."""
        content = "Test with 'quotes', \"double\", and emoji "
        memory = memory_stream.add("agent", content)
        assert memory.content == content

    def test_unicode_content(self, memory_stream):
        """Should handle unicode content."""
        content = "Unicode test: 日本語, العربية, 中文"
        memory = memory_stream.add("agent", content)
        assert memory.content == content

    def test_very_long_content(self, memory_stream):
        """Should handle long content."""
        content = "A" * 10000
        memory = memory_stream.add("agent", content)
        assert len(memory.content) == 10000

    def test_multiple_agents_isolated(self, memory_stream):
        """Different agents should have isolated memories."""
        memory_stream.add("agent1", "Agent 1 memory")
        memory_stream.add("agent2", "Agent 2 memory")

        results1 = memory_stream.retrieve("agent1")
        results2 = memory_stream.retrieve("agent2")

        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].memory.content == "Agent 1 memory"
        assert results2[0].memory.content == "Agent 2 memory"

    def test_metadata_json_serialization(self, memory_stream, temp_db):
        """Should properly serialize and deserialize metadata."""
        metadata = {"key": "value", "nested": {"inner": 123}}
        memory_stream.add("agent", "Content", metadata=metadata)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM memories")
            row = cursor.fetchone()

        import json

        stored = json.loads(row[0])
        assert stored == metadata
