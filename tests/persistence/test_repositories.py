"""
Tests for repository pattern implementations.

Tests MemoryRepository, DebateRepository, and EloRepository
using in-memory SQLite databases for isolation.
"""

import tempfile
from pathlib import Path

import pytest


class TestMemoryRepository:
    """Tests for MemoryRepository."""

    @pytest.fixture
    def repo(self):
        """Create a fresh MemoryRepository with temp database."""
        from aragora.persistence.repositories import MemoryRepository

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        repo = MemoryRepository(db_path)
        yield repo

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_add_memory_creates_entity(self, repo):
        """add_memory creates a new memory entity."""
        memory = repo.add_memory(
            agent_name="claude",
            content="Test observation",
            memory_type="observation",
            importance=0.7,
        )

        assert memory.id is not None
        assert memory.agent_name == "claude"
        assert memory.content == "Test observation"
        assert memory.memory_type == "observation"
        assert memory.importance == 0.7

    def test_add_memory_with_debate_id(self, repo):
        """add_memory can associate with a debate."""
        memory = repo.add_memory(
            agent_name="claude",
            content="Learned from debate",
            debate_id="debate-123",
        )

        assert memory.debate_id == "debate-123"

    def test_add_memory_with_metadata(self, repo):
        """add_memory stores metadata."""
        memory = repo.add_memory(
            agent_name="claude",
            content="Test with metadata",
            metadata={"source": "test", "round": 3},
        )

        assert memory.metadata["source"] == "test"
        assert memory.metadata["round"] == 3

    def test_observe_convenience_method(self, repo):
        """observe() creates observation type."""
        memory = repo.observe("claude", "Something happened")

        assert memory.memory_type == "observation"
        assert memory.importance == 0.5  # default

    def test_reflect_convenience_method(self, repo):
        """reflect() creates reflection type."""
        memory = repo.reflect("claude", "I learned something")

        assert memory.memory_type == "reflection"
        assert memory.importance == 0.7  # default for reflections

    def test_insight_convenience_method(self, repo):
        """insight() creates insight type."""
        memory = repo.insight("claude", "Key insight")

        assert memory.memory_type == "insight"
        assert memory.importance == 0.9  # default for insights

    def test_get_by_agent_returns_memories(self, repo):
        """get_by_agent retrieves all memories for agent."""
        repo.observe("claude", "Observation 1")
        repo.observe("claude", "Observation 2")
        repo.observe("other", "Other agent")

        memories = repo.get_by_agent("claude")

        assert len(memories) == 2
        assert all(m.agent_name == "claude" for m in memories)

    def test_get_by_agent_filters_by_type(self, repo):
        """get_by_agent can filter by memory type."""
        repo.observe("claude", "Observation")
        repo.reflect("claude", "Reflection")
        repo.insight("claude", "Insight")

        reflections = repo.get_by_agent("claude", memory_type="reflection")

        assert len(reflections) == 1
        assert reflections[0].memory_type == "reflection"

    def test_get_by_agent_filters_by_importance(self, repo):
        """get_by_agent respects min_importance threshold."""
        repo.add_memory("claude", "Low importance", importance=0.3)
        repo.add_memory("claude", "High importance", importance=0.8)

        memories = repo.get_by_agent("claude", min_importance=0.5)

        assert len(memories) == 1
        assert memories[0].importance == 0.8

    def test_get_by_agent_respects_limit(self, repo):
        """get_by_agent respects limit parameter."""
        for i in range(10):
            repo.observe("claude", f"Memory {i}")

        memories = repo.get_by_agent("claude", limit=5)

        assert len(memories) == 5

    def test_retrieve_ranks_by_score(self, repo):
        """retrieve returns memories ranked by combined score."""
        repo.add_memory("claude", "Low importance old", importance=0.3)
        repo.add_memory("claude", "High importance new", importance=0.9)

        retrieved = repo.retrieve("claude", limit=2)

        # Higher importance should rank first (recency similar)
        assert retrieved[0].memory.importance > retrieved[1].memory.importance

    def test_retrieve_with_query_uses_relevance(self, repo):
        """retrieve with query factors in relevance score."""
        repo.observe("claude", "Discussion about Python programming")
        repo.observe("claude", "Weather was nice today")

        retrieved = repo.retrieve("claude", query="Python programming")

        # Python-related should be first
        assert "Python" in retrieved[0].memory.content

    def test_should_reflect_false_initially(self, repo):
        """should_reflect returns False for new agents."""
        assert repo.should_reflect("new-agent") is False

    def test_should_reflect_after_threshold(self, repo):
        """should_reflect returns True after threshold memories."""
        for i in range(10):
            repo.observe("claude", f"Observation {i}")

        assert repo.should_reflect("claude", threshold=10) is True
        assert repo.should_reflect("claude", threshold=15) is False

    def test_mark_reflected_resets_counter(self, repo):
        """mark_reflected resets memories_since_reflection."""
        for i in range(15):
            repo.observe("claude", f"Observation {i}")

        assert repo.should_reflect("claude") is True

        repo.mark_reflected("claude")

        assert repo.should_reflect("claude") is False

    def test_get_reflection_schedule(self, repo):
        """get_reflection_schedule returns schedule state."""
        repo.observe("claude", "First observation")

        schedule = repo.get_reflection_schedule("claude")

        assert schedule is not None
        assert schedule.agent_name == "claude"
        assert schedule.memories_since_reflection == 1

    def test_get_stats_returns_counts(self, repo):
        """get_stats returns memory statistics."""
        repo.observe("claude", "Obs 1")
        repo.observe("claude", "Obs 2")
        repo.reflect("claude", "Reflect 1")
        repo.insight("claude", "Insight 1")

        stats = repo.get_stats("claude")

        assert stats["total_memories"] == 4
        assert stats["observations"] == 2
        assert stats["reflections"] == 1
        assert stats["insights"] == 1
        assert 0.5 <= stats["avg_importance"] <= 0.9

    def test_get_context_for_debate(self, repo):
        """get_context_for_debate returns formatted context."""
        repo.insight("claude", "Key insight about testing")
        repo.observe("claude", "Testing observation")

        context = repo.get_context_for_debate("claude", "testing strategies")

        assert "Relevant past experience" in context
        assert "[Insight]" in context or "[Experience]" in context

    def test_delete_by_agent_removes_memories(self, repo):
        """delete_by_agent removes all agent memories."""
        repo.observe("claude", "Memory 1")
        repo.observe("claude", "Memory 2")
        repo.observe("other", "Other memory")

        deleted = repo.delete_by_agent("claude")

        assert deleted == 2
        assert len(repo.get_by_agent("claude")) == 0
        assert len(repo.get_by_agent("other")) == 1

    def test_memory_entity_age_hours(self, repo):
        """MemoryEntity.age_hours calculates correctly."""
        memory = repo.observe("claude", "Recent memory")

        # Just created, should be near zero
        assert memory.age_hours < 0.01


class TestDebateRepository:
    """Tests for DebateRepository."""

    @pytest.fixture
    def repo(self):
        """Create a fresh DebateRepository with temp database."""
        from aragora.persistence.repositories import DebateRepository

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        repo = DebateRepository(db_path)
        yield repo

        Path(db_path).unlink(missing_ok=True)

    def test_save_and_get_debate(self, repo):
        """Can save and retrieve a debate."""
        from aragora.persistence.repositories import DebateEntity

        debate = DebateEntity(
            id="test-123",
            slug="test-debate",
            task="Test Task",
            agents=["claude", "gemini"],
            artifact_json="{}",
        )

        repo.save(debate)
        retrieved = repo.get("test-123")

        assert retrieved is not None
        assert retrieved.slug == "test-debate"
        assert retrieved.task == "Test Task"

    def test_get_by_slug(self, repo):
        """Can retrieve debate by slug."""
        from aragora.persistence.repositories import DebateRepository, DebateEntity

        debate = DebateEntity(
            id="test-456",
            slug="unique-slug",
            task="Unique Task",
            agents=["claude"],
            artifact_json="{}",
        )
        repo.save(debate)

        retrieved = repo.get_by_slug("unique-slug")

        assert retrieved is not None
        assert retrieved.id == "test-456"

    def test_generate_slug_basic(self, repo):
        """generate_slug creates basic slug from task."""
        slug = repo.generate_slug("Test Rate Limiter")

        # Should contain key words (excluding stop words) and date
        assert "rate" in slug
        assert "limiter" in slug
        assert "-" in slug  # Date separator

    def test_generate_slug_handles_collisions(self, repo):
        """generate_slug handles collisions by appending counter."""
        from aragora.persistence.repositories import DebateEntity
        from datetime import datetime

        # Create a task that will generate a specific slug
        task = "Test Slug Collision"
        base_slug = repo.generate_slug(task)

        # Save a debate with that slug
        debate = DebateEntity(
            id="collision-1",
            slug=base_slug,
            task=task,
            agents=["claude"],
            artifact_json="{}",
        )
        repo.save(debate)

        # Generate another slug for same task - should get numbered variant
        second_slug = repo.generate_slug(task)

        assert second_slug != base_slug
        assert second_slug == f"{base_slug}-2"

    def test_generate_slug_handles_multiple_collisions(self, repo):
        """generate_slug finds next available number with multiple collisions."""
        from aragora.persistence.repositories import DebateEntity

        task = "Multiple Collision Test"
        base_slug = repo.generate_slug(task)

        # Create debates for base slug and -2 through -5
        for i, suffix in enumerate(["", "-2", "-3", "-4", "-5"]):
            debate = DebateEntity(
                id=f"multi-collision-{i}",
                slug=f"{base_slug}{suffix}",
                task=task,
                agents=["claude"],
                artifact_json="{}",
            )
            repo.save(debate)

        # Next slug should be -6
        next_slug = repo.generate_slug(task)
        assert next_slug == f"{base_slug}-6"

    def test_generate_slug_batch_query_efficiency(self, repo):
        """generate_slug uses batch query rather than N+1 queries."""
        from aragora.persistence.repositories import DebateEntity
        from unittest.mock import patch

        task = "Batch Query Test"
        base_slug = repo.generate_slug(task)

        # Create several collisions
        for i in range(5):
            suffix = "" if i == 0 else f"-{i + 1}"
            debate = DebateEntity(
                id=f"batch-test-{i}",
                slug=f"{base_slug}{suffix}",
                task=task,
                agents=["claude"],
                artifact_json="{}",
            )
            repo.save(debate)

        # Track query count
        query_count = 0
        original_fetch_all = repo._fetch_all

        def counting_fetch_all(*args, **kwargs):
            nonlocal query_count
            query_count += 1
            return original_fetch_all(*args, **kwargs)

        with patch.object(repo, "_fetch_all", counting_fetch_all):
            repo.generate_slug(task)

        # Should use exactly 1 batch query, not N queries
        assert query_count == 1

    def test_increment_view_count_basic(self, repo):
        """increment_view_count increments and returns new count."""
        from aragora.persistence.repositories import DebateEntity

        debate = DebateEntity(
            id="view-test-1",
            slug="view-count-test",
            task="View Count Test",
            agents=["claude"],
            artifact_json="{}",
            view_count=0,
        )
        repo.save(debate)

        new_count = repo.increment_view_count("view-count-test")
        assert new_count == 1

        new_count = repo.increment_view_count("view-count-test")
        assert new_count == 2

    def test_increment_view_count_nonexistent(self, repo):
        """increment_view_count returns 0 for nonexistent slug."""
        count = repo.increment_view_count("nonexistent-slug")
        assert count == 0

    def test_increment_view_count_uses_returning_clause(self, repo):
        """increment_view_count uses UPDATE with RETURNING clause."""
        from aragora.persistence.repositories import DebateEntity

        debate = DebateEntity(
            id="returning-test",
            slug="returning-clause-slug",
            task="Returning Clause Test",
            agents=["claude"],
            artifact_json="{}",
            view_count=10,
        )
        repo.save(debate)

        # Verify the method works correctly with initial value
        result = repo.increment_view_count("returning-clause-slug")
        assert result == 11

        # Verify the actual value was persisted correctly
        retrieved = repo.get_by_slug("returning-clause-slug")
        assert retrieved is not None
        assert retrieved.view_count == 11

        # Increment again to ensure consistency
        result2 = repo.increment_view_count("returning-clause-slug")
        assert result2 == 12

        retrieved2 = repo.get_by_slug("returning-clause-slug")
        assert retrieved2.view_count == 12


class TestEloRepository:
    """Tests for EloRepository."""

    @pytest.fixture
    def repo(self):
        """Create a fresh EloRepository with temp database."""
        from aragora.persistence.repositories import EloRepository

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        repo = EloRepository(db_path)
        yield repo

        Path(db_path).unlink(missing_ok=True)

    def test_get_rating_creates_default_for_new_agent(self, repo):
        """get_rating creates default rating for unknown agent."""
        rating = repo.get_rating("new-agent")

        assert rating is not None
        assert rating.agent_name == "new-agent"
        assert rating.elo == 1500  # Default ELO

    def test_save_and_get_rating(self, repo):
        """Can save and retrieve a rating."""
        from aragora.persistence.repositories import RatingEntity

        rating = RatingEntity(
            agent_name="test-agent",
            elo=1200,
        )
        repo.save(rating)

        retrieved = repo.get_rating("test-agent")

        assert retrieved is not None
        assert retrieved.agent_name == "test-agent"
        assert retrieved.elo == 1200

    def test_record_match(self, repo):
        """Can record a match between agents."""
        from aragora.persistence.repositories import RatingEntity

        # Create ratings for both agents
        repo.save(RatingEntity(agent_name="agent1", elo=1000))
        repo.save(RatingEntity(agent_name="agent2", elo=1000))

        match_id = repo.record_match(
            debate_id="debate-1",
            winner="agent1",
            participants=["agent1", "agent2"],
            scores={"agent1": 1.0, "agent2": 0.0},
            elo_changes={"agent1": 16.0, "agent2": -16.0},
        )

        assert match_id is not None
        assert isinstance(match_id, int)

    def test_get_leaderboard(self, repo):
        """get_leaderboard returns ranked agents."""
        from aragora.persistence.repositories import RatingEntity

        repo.save(RatingEntity(agent_name="agent1", elo=1200))
        repo.save(RatingEntity(agent_name="agent2", elo=1100))

        leaderboard = repo.get_leaderboard(limit=10)

        assert len(leaderboard) >= 2
        # First should have higher rating
        assert leaderboard[0].elo >= leaderboard[1].elo
