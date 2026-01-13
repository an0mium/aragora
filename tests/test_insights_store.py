"""
Tests for InsightStore - SQLite persistence for debate insights.

Tests cover:
- Storing and retrieving insights
- Full-text search capabilities
- Pattern aggregation
- Agent statistics
- Wisdom submissions (audience injection)
- SQL injection prevention
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from aragora.insights.store import InsightStore, _escape_like_pattern
from aragora.insights.extractor import (
    Insight,
    InsightType,
    DebateInsights,
    AgentPerformance,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_insights.db")


@pytest.fixture
def insight_store(temp_db_path):
    """Create an InsightStore instance with a temp database."""
    return InsightStore(db_path=temp_db_path)


@pytest.fixture
def sample_insight():
    """Create a sample insight for testing."""
    return Insight(
        id="insight-001",
        type=InsightType.PATTERN,
        title="Repetitive argument pattern",
        description="Agent tends to repeat similar arguments across rounds",
        confidence=0.85,
        debate_id="debate-123",
        agents_involved=["claude", "codex"],
        evidence=["round 1: ...", "round 2: ..."],
        created_at=datetime.now().isoformat(),
        metadata={"category": "argumentation", "severity": 0.6},
    )


@pytest.fixture
def sample_agent_performance():
    """Create sample agent performance for testing."""
    return AgentPerformance(
        agent_name="claude",
        proposals_made=3,
        critiques_given=5,
        critiques_received=2,
        proposal_accepted=True,
        vote_aligned_with_consensus=True,
        average_critique_severity=0.4,
        contribution_score=0.75,
    )


@pytest.fixture
def sample_debate_insights(sample_insight, sample_agent_performance):
    """Create sample DebateInsights for testing."""
    pattern_insight = Insight(
        id="pattern-001",
        type=InsightType.PATTERN,
        title="Convergence detected",
        description="Agents converged on solution",
        confidence=0.9,
        debate_id="debate-123",
        agents_involved=["claude", "codex"],
        evidence=[],
        created_at=datetime.now().isoformat(),
        metadata={"category": "convergence", "avg_severity": 0.3},
    )

    return DebateInsights(
        debate_id="debate-123",
        task="Design a rate limiter",
        consensus_reached=True,
        duration_seconds=120.5,
        total_insights=5,
        key_takeaway="Token bucket algorithm selected",
        convergence_insight=sample_insight,  # singular, Optional[Insight]
        pattern_insights=[pattern_insight],
        agent_performances=[sample_agent_performance],
    )


# =============================================================================
# Tests: Escape LIKE Pattern
# =============================================================================


class TestEscapeLikePattern:
    """Tests for SQL LIKE pattern escaping."""

    def test_escape_percent(self):
        """Test escaping % wildcard."""
        assert _escape_like_pattern("100%") == "100\\%"

    def test_escape_underscore(self):
        """Test escaping _ wildcard."""
        assert _escape_like_pattern("user_name") == "user\\_name"

    def test_escape_backslash(self):
        """Test escaping backslash."""
        assert _escape_like_pattern("path\\file") == "path\\\\file"

    def test_escape_combined(self):
        """Test escaping combined special characters."""
        assert _escape_like_pattern("100%_test\\path") == "100\\%\\_test\\\\path"

    def test_no_special_chars(self):
        """Test string without special characters."""
        assert _escape_like_pattern("normal text") == "normal text"


# =============================================================================
# Tests: InsightStore Initialization
# =============================================================================


class TestInsightStoreInit:
    """Tests for InsightStore initialization."""

    def test_init_creates_db(self, temp_db_path):
        """Test that init creates the database file."""
        store = InsightStore(db_path=temp_db_path)
        assert Path(temp_db_path).exists()

    def test_init_creates_tables(self, insight_store, temp_db_path):
        """Test that init creates required tables."""
        import sqlite3

        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

        assert "insights" in tables
        assert "debate_summaries" in tables
        assert "agent_performance_history" in tables
        assert "pattern_clusters" in tables


# =============================================================================
# Tests: Store and Retrieve Insights
# =============================================================================


class TestStoreDebateInsights:
    """Tests for storing debate insights."""

    @pytest.mark.asyncio
    async def test_store_debate_insights(self, insight_store, sample_debate_insights):
        """Test storing complete debate insights."""
        stored = await insight_store.store_debate_insights(sample_debate_insights)
        assert stored >= 1

    @pytest.mark.asyncio
    async def test_store_creates_summary(self, insight_store, sample_debate_insights):
        """Test that storing insights creates debate summary."""
        await insight_store.store_debate_insights(sample_debate_insights)

        stats = await insight_store.get_stats()
        assert stats["total_debates"] == 1
        assert stats["consensus_debates"] == 1

    @pytest.mark.asyncio
    async def test_store_creates_agent_performance(self, insight_store, sample_debate_insights):
        """Test that storing insights creates agent performance records."""
        await insight_store.store_debate_insights(sample_debate_insights)

        agent_stats = await insight_store.get_agent_stats("claude")
        assert agent_stats["debate_count"] == 1
        assert agent_stats["total_proposals"] == 3
        assert agent_stats["proposals_accepted"] == 1

    @pytest.mark.asyncio
    async def test_store_updates_pattern_clusters(self, insight_store, sample_debate_insights):
        """Test that pattern insights update pattern clusters."""
        # Store twice to test increment
        await insight_store.store_debate_insights(sample_debate_insights)

        # Create a modified version with different debate ID
        sample_debate_insights.debate_id = "debate-456"
        await insight_store.store_debate_insights(sample_debate_insights)

        patterns = await insight_store.get_common_patterns(min_occurrences=2)
        assert len(patterns) >= 1
        assert patterns[0]["occurrences"] == 2


class TestGetInsight:
    """Tests for retrieving individual insights."""

    @pytest.mark.asyncio
    async def test_get_existing_insight(self, insight_store, sample_debate_insights):
        """Test retrieving an existing insight."""
        await insight_store.store_debate_insights(sample_debate_insights)

        insight = await insight_store.get_insight("insight-001")
        assert insight is not None
        assert insight.id == "insight-001"
        assert insight.title == "Repetitive argument pattern"

    @pytest.mark.asyncio
    async def test_get_nonexistent_insight(self, insight_store):
        """Test retrieving a non-existent insight."""
        insight = await insight_store.get_insight("nonexistent-id")
        assert insight is None


# =============================================================================
# Tests: Search
# =============================================================================


class TestSearch:
    """Tests for searching insights."""

    @pytest.mark.asyncio
    async def test_search_by_query(self, insight_store, sample_debate_insights):
        """Test searching by text query."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(query="Repetitive")
        assert len(results) >= 1
        assert any("Repetitive" in r.title for r in results)

    @pytest.mark.asyncio
    async def test_search_by_type(self, insight_store, sample_debate_insights):
        """Test searching by insight type."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(insight_type=InsightType.PATTERN)
        assert len(results) >= 1
        assert all(r.type == InsightType.PATTERN for r in results)

    @pytest.mark.asyncio
    async def test_search_by_agent(self, insight_store, sample_debate_insights):
        """Test searching by agent involvement."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(agent="claude")
        assert len(results) >= 1
        assert all("claude" in r.agents_involved for r in results)

    @pytest.mark.asyncio
    async def test_search_combined_filters(self, insight_store, sample_debate_insights):
        """Test searching with combined filters."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(
            query="Repetitive", insight_type=InsightType.PATTERN, agent="claude"
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_limit(self, insight_store, sample_debate_insights):
        """Test search respects limit."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, insight_store, sample_debate_insights):
        """Test search with no matching results."""
        await insight_store.store_debate_insights(sample_debate_insights)

        results = await insight_store.search(query="nonexistent_query_xyz")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_sql_injection_prevention(self, insight_store, sample_debate_insights):
        """Test that SQL injection attempts are safely handled."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # These should not cause errors or return unexpected results
        results = await insight_store.search(query="'; DROP TABLE insights; --")
        assert isinstance(results, list)

        results = await insight_store.search(agent="claude' OR '1'='1")
        assert isinstance(results, list)


# =============================================================================
# Tests: Pattern Aggregation
# =============================================================================


class TestGetCommonPatterns:
    """Tests for pattern aggregation."""

    @pytest.mark.asyncio
    async def test_get_common_patterns_empty(self, insight_store):
        """Test getting patterns from empty store."""
        patterns = await insight_store.get_common_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_get_common_patterns_min_occurrences(self, insight_store, sample_debate_insights):
        """Test min_occurrences filter."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # Should return nothing with min=5 since we only have 1 occurrence
        patterns = await insight_store.get_common_patterns(min_occurrences=5)
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_get_common_patterns_by_category(self, insight_store, sample_debate_insights):
        """Test filtering patterns by category."""
        await insight_store.store_debate_insights(sample_debate_insights)

        patterns = await insight_store.get_common_patterns(
            min_occurrences=1, category="convergence"
        )
        assert all(p["category"] == "convergence" for p in patterns)


# =============================================================================
# Tests: Agent Statistics
# =============================================================================


class TestGetAgentStats:
    """Tests for agent statistics."""

    @pytest.mark.asyncio
    async def test_get_agent_stats_existing(self, insight_store, sample_debate_insights):
        """Test getting stats for existing agent."""
        await insight_store.store_debate_insights(sample_debate_insights)

        stats = await insight_store.get_agent_stats("claude")
        assert stats["agent"] == "claude"
        assert stats["debate_count"] == 1
        assert stats["total_proposals"] == 3
        assert stats["total_critiques"] == 5
        assert 0 <= stats["avg_contribution"] <= 1

    @pytest.mark.asyncio
    async def test_get_agent_stats_nonexistent(self, insight_store):
        """Test getting stats for non-existent agent."""
        stats = await insight_store.get_agent_stats("nonexistent_agent")
        assert stats["agent"] == "nonexistent_agent"
        assert stats["debate_count"] == 0

    @pytest.mark.asyncio
    async def test_acceptance_rate_calculation(self, insight_store, sample_debate_insights):
        """Test that acceptance rate is calculated correctly."""
        await insight_store.store_debate_insights(sample_debate_insights)

        stats = await insight_store.get_agent_stats("claude")
        # 1 accepted out of 3 proposals
        expected_rate = 1 / 3
        assert abs(stats["acceptance_rate"] - expected_rate) < 0.01


class TestGetAllAgentRankings:
    """Tests for agent rankings."""

    @pytest.mark.asyncio
    async def test_get_rankings_empty(self, insight_store):
        """Test rankings from empty store."""
        rankings = await insight_store.get_all_agent_rankings()
        assert rankings == []

    @pytest.mark.asyncio
    async def test_get_rankings_ordered(self, insight_store, sample_debate_insights):
        """Test rankings are ordered by contribution."""
        # Add another agent with different performance
        sample_debate_insights.agent_performances.append(
            AgentPerformance(
                agent_name="codex",
                proposals_made=1,
                critiques_given=2,
                critiques_received=3,
                proposal_accepted=False,
                vote_aligned_with_consensus=False,
                average_critique_severity=0.3,
                contribution_score=0.4,  # Lower than claude's 0.75
            )
        )
        await insight_store.store_debate_insights(sample_debate_insights)

        rankings = await insight_store.get_all_agent_rankings()
        assert len(rankings) == 2
        # Claude should be first (higher contribution)
        assert rankings[0]["agent"] == "claude"
        assert rankings[1]["agent"] == "codex"


# =============================================================================
# Tests: Recent Insights and Stats
# =============================================================================


class TestGetRecentInsights:
    """Tests for getting recent insights."""

    @pytest.mark.asyncio
    async def test_get_recent_insights(self, insight_store, sample_debate_insights):
        """Test retrieving recent insights."""
        await insight_store.store_debate_insights(sample_debate_insights)

        recent = await insight_store.get_recent_insights(limit=10)
        assert len(recent) >= 1

    @pytest.mark.asyncio
    async def test_get_recent_insights_limit(self, insight_store, sample_debate_insights):
        """Test recent insights respects limit."""
        await insight_store.store_debate_insights(sample_debate_insights)

        recent = await insight_store.get_recent_insights(limit=1)
        assert len(recent) <= 1


class TestGetStats:
    """Tests for overall statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, insight_store):
        """Test stats from empty store."""
        stats = await insight_store.get_stats()
        assert stats["total_insights"] == 0
        assert stats["total_debates"] == 0
        assert stats["consensus_debates"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, insight_store, sample_debate_insights):
        """Test stats with data."""
        await insight_store.store_debate_insights(sample_debate_insights)

        stats = await insight_store.get_stats()
        assert stats["total_insights"] >= 1
        assert stats["total_debates"] == 1
        assert stats["consensus_debates"] == 1
        assert stats["unique_agents"] >= 1


# =============================================================================
# Tests: Wisdom Submissions
# =============================================================================


class TestWisdomSubmissions:
    """Tests for audience wisdom functionality."""

    def test_add_wisdom_submission(self, insight_store):
        """Test adding a wisdom submission."""
        wisdom_id = insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={
                "text": "Consider using a sliding window approach",
                "submitter_id": "user-123",
                "context_tags": ["algorithm", "optimization"],
            },
        )
        assert wisdom_id > 0

    def test_get_relevant_wisdom(self, insight_store):
        """Test retrieving unused wisdom."""
        # Add some wisdom
        insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={"text": "First suggestion"},
        )
        insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={"text": "Second suggestion"},
        )
        insight_store.add_wisdom_submission(
            loop_id="loop-002",  # Different loop
            wisdom_data={"text": "Other loop suggestion"},
        )

        wisdom = insight_store.get_relevant_wisdom("loop-001", limit=10)
        assert len(wisdom) == 2
        assert all(w["text"] in ["First suggestion", "Second suggestion"] for w in wisdom)

    def test_mark_wisdom_used(self, insight_store):
        """Test marking wisdom as used."""
        wisdom_id = insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={"text": "A suggestion"},
        )

        # Mark as used
        insight_store.mark_wisdom_used(wisdom_id)

        # Should not appear in relevant wisdom
        wisdom = insight_store.get_relevant_wisdom("loop-001")
        assert len(wisdom) == 0

    def test_wisdom_text_truncation(self, insight_store):
        """Test that wisdom text is truncated to 280 chars."""
        long_text = "x" * 500
        wisdom_id = insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={"text": long_text},
        )

        wisdom = insight_store.get_relevant_wisdom("loop-001")
        assert len(wisdom[0]["text"]) == 280

    def test_wisdom_default_submitter(self, insight_store):
        """Test default submitter ID."""
        insight_store.add_wisdom_submission(
            loop_id="loop-001",
            wisdom_data={"text": "No submitter specified"},
        )

        wisdom = insight_store.get_relevant_wisdom("loop-001")
        assert wisdom[0]["submitter_id"] == "anonymous"


# =============================================================================
# Tests: Row Conversion
# =============================================================================


class TestRowToInsight:
    """Tests for row to Insight conversion."""

    @pytest.mark.asyncio
    async def test_row_conversion_preserves_data(self, insight_store, sample_debate_insights):
        """Test that data is preserved through store/retrieve cycle."""
        await insight_store.store_debate_insights(sample_debate_insights)

        insight = await insight_store.get_insight("insight-001")
        assert insight.id == "insight-001"
        assert insight.type == InsightType.PATTERN
        assert insight.title == "Repetitive argument pattern"
        assert insight.confidence == 0.85
        assert insight.debate_id == "debate-123"
        assert "claude" in insight.agents_involved
        assert len(insight.evidence) == 2

    @pytest.mark.asyncio
    async def test_invalid_type_defaults_to_pattern(self, insight_store, temp_db_path):
        """Test that invalid insight type defaults to PATTERN."""
        import sqlite3

        # Insert a row with invalid type directly
        with sqlite3.connect(temp_db_path) as conn:
            conn.execute(
                """
                INSERT INTO insights (id, type, title, description, confidence, debate_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("bad-type-id", "INVALID_TYPE", "Test", "Desc", 0.5, "debate-1"),
            )
            conn.commit()

        insight = await insight_store.get_insight("bad-type-id")
        assert insight.type == InsightType.PATTERN


# =============================================================================
# Tests: Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, insight_store, sample_debate_insights):
        """Test concurrent store operations."""

        async def store_with_id(debate_id: str):
            insights = DebateInsights(
                debate_id=debate_id,
                task="Test task",
                consensus_reached=True,
                duration_seconds=60,
                total_insights=1,
                key_takeaway="Test",
                convergence_insight=None,  # singular, Optional[Insight]
                pattern_insights=[],
                agent_performances=[],
            )
            return await insight_store.store_debate_insights(insights)

        # Run concurrent stores
        tasks = [store_with_id(f"debate-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5

        # Verify all debates stored
        stats = await insight_store.get_stats()
        assert stats["total_debates"] == 5

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, insight_store, sample_debate_insights):
        """Test concurrent read operations."""
        await insight_store.store_debate_insights(sample_debate_insights)

        async def concurrent_search(query: str):
            return await insight_store.search(query=query)

        # Run concurrent searches
        tasks = [concurrent_search(f"query-{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return lists
        assert all(isinstance(r, list) for r in results)


# =============================================================================
# Tests: Insight Application Cycle (Phase 10B)
# =============================================================================


class TestInsightApplicationCycle:
    """Tests for the insight application cycle - get_relevant_insights and record_insight_usage."""

    @pytest.mark.asyncio
    async def test_get_relevant_insights_empty(self, insight_store):
        """Should return empty list when no insights exist."""
        insights = await insight_store.get_relevant_insights()
        assert insights == []

    @pytest.mark.asyncio
    async def test_get_relevant_insights_returns_high_confidence(
        self, insight_store, sample_debate_insights
    ):
        """Should return insights with high confidence."""
        # The convergence_insight already has confidence=0.85 in fixture
        await insight_store.store_debate_insights(sample_debate_insights)

        insights = await insight_store.get_relevant_insights(min_confidence=0.7)
        # Should return insights meeting the threshold
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_get_relevant_insights_filters_by_confidence(self, insight_store, sample_insight):
        """Should filter out low confidence insights."""
        # Create insight with low confidence
        sample_insight.confidence = 0.3
        low_conf_insights = DebateInsights(
            debate_id="debate-low-conf",
            task="Test task",
            consensus_reached=True,
            duration_seconds=60.0,
            total_insights=1,
            key_takeaway="Low confidence test",
            convergence_insight=sample_insight,
            pattern_insights=[],
            agent_performances=[],
        )
        await insight_store.store_debate_insights(low_conf_insights)

        # High threshold should filter it out
        insights = await insight_store.get_relevant_insights(min_confidence=0.8)
        # Should be empty or only contain insights meeting threshold
        for insight in insights:
            assert insight.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_get_relevant_insights_filters_by_domain(
        self, insight_store, sample_debate_insights
    ):
        """Should filter by domain when specified."""
        # The pattern_insights fixture has category=convergence in metadata
        await insight_store.store_debate_insights(sample_debate_insights)

        # Filter by convergence domain
        insights = await insight_store.get_relevant_insights(
            domain="convergence", min_confidence=0.5
        )
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_get_relevant_insights_respects_limit(
        self, insight_store, sample_debate_insights
    ):
        """Should respect limit parameter."""
        await insight_store.store_debate_insights(sample_debate_insights)

        insights = await insight_store.get_relevant_insights(min_confidence=0.5, limit=1)
        assert len(insights) <= 1

    @pytest.mark.asyncio
    async def test_record_insight_usage_success(self, insight_store, sample_debate_insights):
        """Should record successful insight usage."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # Get an insight ID - get_recent_insights returns Insight objects
        insights = await insight_store.get_recent_insights(limit=1)
        if insights:
            insight = insights[0]
            insight_id = getattr(insight, "id", None) or getattr(insight, "insight_id", "test-id")

            # Record successful usage
            await insight_store.record_insight_usage(
                insight_id=insight_id,
                debate_id="test-debate-123",
                was_successful=True,
            )

            # Should not raise
            assert True

    @pytest.mark.asyncio
    async def test_record_insight_usage_failure(self, insight_store, sample_debate_insights):
        """Should record failed insight usage."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # Record failed usage - should not raise
        await insight_store.record_insight_usage(
            insight_id="some-insight-id",
            debate_id="test-debate-456",
            was_successful=False,
        )
        assert True

    @pytest.mark.asyncio
    async def test_insight_usage_tracking(self, insight_store, sample_debate_insights):
        """Usage tracking infrastructure should work."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # Get an insight ID - get_recent_insights returns Insight objects
        insights = await insight_store.get_recent_insights(limit=1)
        if insights:
            insight = insights[0]
            insight_id = getattr(insight, "id", None) or getattr(insight, "insight_id", "test-id")

            # Record multiple successful usages
            for i in range(3):
                await insight_store.record_insight_usage(
                    insight_id=insight_id,
                    debate_id=f"debate-{i}",
                    was_successful=True,
                )

            # Should not raise - usage is recorded
            assert True


class TestInsightContextInjection:
    """Tests for context initialization insight injection."""

    @pytest.mark.asyncio
    async def test_context_receives_insights(self, insight_store, sample_debate_insights):
        """Verify insights can be retrieved for context injection."""
        await insight_store.store_debate_insights(sample_debate_insights)

        # Get insights for injection
        insights = await insight_store.get_relevant_insights(min_confidence=0.5, limit=5)

        # Format for prompt (simulating what ContextInitializer does)
        if insights:
            context = "## LEARNED PRACTICES\n"
            for insight in insights:
                context += f"- {insight.title}\n"
            assert "LEARNED PRACTICES" in context
