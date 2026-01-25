"""
Tests for PostgreSQL Critique Store implementation.

These tests verify the PostgresCritiqueStore class provides the same
functionality as the SQLite-based CritiqueStore.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager


# Check if asyncpg is available
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class TestPostgresCritiqueSchema:
    """Tests for schema definitions (no asyncpg required)."""

    def test_schema_module_imports(self):
        """Module should import without errors."""
        from aragora.memory.postgres_critique import (
            POSTGRES_CRITIQUE_SCHEMA,
            POSTGRES_CRITIQUE_SCHEMA_VERSION,
        )

        assert POSTGRES_CRITIQUE_SCHEMA_VERSION >= 1
        assert len(POSTGRES_CRITIQUE_SCHEMA) > 100

    def test_schema_has_required_tables(self):
        """Schema should define all required tables."""
        from aragora.memory.postgres_critique import POSTGRES_CRITIQUE_SCHEMA

        required_tables = [
            "debates",
            "critiques",
            "patterns",
            "pattern_embeddings",
            "agent_reputation",
            "patterns_archive",
        ]

        for table in required_tables:
            assert table in POSTGRES_CRITIQUE_SCHEMA, f"Missing table: {table}"

    def test_schema_has_required_indexes(self):
        """Schema should define performance indexes."""
        from aragora.memory.postgres_critique import POSTGRES_CRITIQUE_SCHEMA

        required_indexes = [
            "idx_critiques_debate",
            "idx_critiques_agent",
            "idx_patterns_type",
            "idx_patterns_success",
            "idx_patterns_type_success",
            "idx_reputation_score",
            "idx_reputation_agent",
            "idx_debates_consensus",
        ]

        for index in required_indexes:
            assert index in POSTGRES_CRITIQUE_SCHEMA, f"Missing index: {index}"

    def test_schema_uses_jsonb_for_data(self):
        """Schema should use JSONB for JSON fields."""
        from aragora.memory.postgres_critique import POSTGRES_CRITIQUE_SCHEMA

        assert "JSONB" in POSTGRES_CRITIQUE_SCHEMA

    def test_schema_uses_timestamptz(self):
        """Schema should use TIMESTAMPTZ for timestamps."""
        from aragora.memory.postgres_critique import POSTGRES_CRITIQUE_SCHEMA

        assert "TIMESTAMPTZ" in POSTGRES_CRITIQUE_SCHEMA

    def test_schema_has_foreign_keys(self):
        """Schema should have foreign keys for referential integrity."""
        from aragora.memory.postgres_critique import POSTGRES_CRITIQUE_SCHEMA

        assert "REFERENCES debates" in POSTGRES_CRITIQUE_SCHEMA
        assert "REFERENCES patterns" in POSTGRES_CRITIQUE_SCHEMA


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresCritiqueStore:
    """Tests for PostgresCritiqueStore class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        """Create a mocked PostgresCritiqueStore instance."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        db = PostgresCritiqueStore(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        @asynccontextmanager
        async def mock_transaction_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        db.transaction = mock_transaction_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_store_debate(self, mock_db):
        """store_debate should create a new debate record."""
        db, mock_conn = mock_db

        result = await db.store_debate(
            debate_id="debate-123",
            task="Design a rate limiter",
            final_answer="Use token bucket algorithm",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
            duration_seconds=45.5,
        )

        mock_conn.execute.assert_called()
        assert result["id"] == "debate-123"
        assert result["task"] == "Design a rate limiter"
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_debate_returns_none_for_missing(self, mock_db):
        """get_debate should return None for non-existent record."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        result = await db.get_debate("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_debate_returns_data(self, mock_db):
        """get_debate should return record data."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        mock_conn.fetchrow.return_value = {
            "id": "debate-123",
            "task": "Test task",
            "final_answer": "Test answer",
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 3,
            "duration_seconds": 30.0,
            "grounded_verdict": None,
            "created_at": datetime.now(timezone.utc),
        }

        result = await db.get_debate("debate-123")
        assert result is not None
        assert result["task"] == "Test task"
        assert result["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_store_critique(self, mock_db):
        """store_critique should create a new critique record."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"id": 1}

        result = await db.store_critique(
            debate_id="debate-123",
            agent="claude",
            target_agent="gpt4",
            issues=["Logic flaw", "Missing edge case"],
            suggestions=["Add validation", "Handle null"],
            severity=0.7,
            reasoning="The approach needs refinement",
        )

        mock_conn.fetchrow.assert_called()
        assert result == 1

    @pytest.mark.asyncio
    async def test_get_recent_critiques(self, mock_db):
        """get_recent_critiques should return recent critique records."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        mock_rows = [
            {
                "id": 1,
                "debate_id": "d1",
                "agent": "claude",
                "target_agent": "gpt4",
                "issues": ["Issue 1"],
                "suggestions": ["Fix 1"],
                "severity": 0.5,
                "reasoning": "Reasoning 1",
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": 2,
                "debate_id": "d2",
                "agent": "gpt4",
                "target_agent": "claude",
                "issues": ["Issue 2"],
                "suggestions": ["Fix 2"],
                "severity": 0.6,
                "reasoning": "Reasoning 2",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_recent_critiques(limit=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_db):
        """get_stats should return comprehensive statistics."""
        db, mock_conn = mock_db

        mock_conn.fetchrow.return_value = {
            "total_debates": 42,
            "consensus_debates": 35,
            "total_critiques": 150,
            "total_patterns": 25,
            "avg_confidence": 0.85,
        }
        mock_conn.fetch.return_value = [
            {"issue_type": "security", "cnt": 10},
            {"issue_type": "performance", "cnt": 8},
        ]

        result = await db.get_stats()

        assert result["total_debates"] == 42
        assert result["consensus_debates"] == 35
        assert result["total_critiques"] == 150
        assert result["total_patterns"] == 25
        assert result["patterns_by_type"]["security"] == 10


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresPatternOperations:
    """Tests for pattern operations."""

    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        db = PostgresCritiqueStore(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_store_pattern(self, mock_db):
        """store_pattern should create a new pattern record."""
        db, mock_conn = mock_db
        # Mock fetchrow to return different values for each call:
        # 1. _update_surprise_score: SELECT issue_type
        # 2. _calculate_surprise: SELECT base_rate
        # 3. _update_surprise_score: SELECT base_rate
        mock_conn.fetchrow.side_effect = [
            {"issue_type": "security"},
            {"base_rate": 0.5},
            {"base_rate": 0.5},
        ]
        mock_conn.fetch.return_value = []

        result = await db.store_pattern(
            issue_text="SQL injection vulnerability",
            suggestion_text="Use parameterized queries",
            severity=0.9,
            example_task="Fix user login",
        )

        assert mock_conn.execute.call_count >= 1
        assert result["issue_type"] == "security"
        assert "injection" in result["issue_text"].lower()

    @pytest.mark.asyncio
    async def test_retrieve_patterns(self, mock_db):
        """retrieve_patterns should return patterns with ranking."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        mock_rows = [
            {
                "id": "pattern-1",
                "issue_type": "security",
                "issue_text": "SQL injection",
                "suggestion_text": "Use prepared statements",
                "success_count": 10,
                "failure_count": 2,
                "avg_severity": 0.8,
                "example_task": "Login form",
                "created_at": now,
                "updated_at": now,
                "decay_score": 9.5,
            },
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.retrieve_patterns(issue_type="security", min_success=2, limit=10)

        assert len(result) == 1
        assert result[0].issue_type == "security"
        assert result[0].success_count == 10

    @pytest.mark.asyncio
    async def test_fail_pattern(self, mock_db):
        """fail_pattern should increment failure count."""
        db, mock_conn = mock_db
        mock_conn.execute.return_value = "UPDATE 1"
        # Mock fetchrow to return different values for each call:
        # 1. _update_surprise_score: SELECT issue_type
        # 2. _calculate_surprise: SELECT base_rate
        # 3. _update_surprise_score: SELECT base_rate
        mock_conn.fetchrow.side_effect = [
            {"issue_type": "general"},
            {"base_rate": 0.5},
            {"base_rate": 0.5},
        ]

        await db.fail_pattern("Some issue text")

        mock_conn.execute.assert_called()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresReputationOperations:
    """Tests for reputation operations."""

    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        db = PostgresCritiqueStore(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_get_reputation(self, mock_db):
        """get_reputation should return agent reputation data."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        mock_conn.fetchrow.return_value = {
            "agent_name": "claude",
            "proposals_made": 50,
            "proposals_accepted": 40,
            "critiques_given": 30,
            "critiques_valuable": 25,
            "updated_at": datetime.now(timezone.utc),
            "total_predictions": 20,
            "total_prediction_error": 2.0,
            "calibration_score": 0.9,
        }

        result = await db.get_reputation("claude")

        assert result is not None
        assert result.agent_name == "claude"
        assert result.proposals_made == 50
        assert result.proposals_accepted == 40
        assert result.calibration_score == 0.9

    @pytest.mark.asyncio
    async def test_get_vote_weight(self, mock_db):
        """get_vote_weight should return calculated weight."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        mock_conn.fetchrow.return_value = {
            "agent_name": "claude",
            "proposals_made": 100,
            "proposals_accepted": 80,
            "critiques_given": 50,
            "critiques_valuable": 40,
            "updated_at": datetime.now(timezone.utc),
            "total_predictions": 30,
            "total_prediction_error": 3.0,
            "calibration_score": 0.9,
        }

        weight = await db.get_vote_weight("claude")

        assert 0.4 <= weight <= 1.6

    @pytest.mark.asyncio
    async def test_get_vote_weights_batch(self, mock_db):
        """get_vote_weights_batch should return weights for multiple agents."""
        db, mock_conn = mock_db

        mock_rows = [
            {
                "agent_name": "claude",
                "proposals_made": 100,
                "proposals_accepted": 80,
                "critiques_given": 50,
                "critiques_valuable": 40,
                "calibration_score": 0.9,
            },
            {
                "agent_name": "gpt4",
                "proposals_made": 50,
                "proposals_accepted": 35,
                "critiques_given": 30,
                "critiques_valuable": 20,
                "calibration_score": 0.7,
            },
        ]
        mock_conn.fetch.return_value = mock_rows

        weights = await db.get_vote_weights_batch(["claude", "gpt4", "unknown"])

        assert "claude" in weights
        assert "gpt4" in weights
        assert "unknown" in weights
        assert weights["unknown"] == 1.0  # Default for unknown

    @pytest.mark.asyncio
    async def test_update_reputation(self, mock_db):
        """update_reputation should update metrics."""
        db, mock_conn = mock_db

        await db.update_reputation(
            "claude",
            proposal_made=True,
            proposal_accepted=True,
            critique_given=True,
        )

        assert mock_conn.execute.call_count >= 2  # INSERT + UPDATE

    @pytest.mark.asyncio
    async def test_get_all_reputations(self, mock_db):
        """get_all_reputations should return all agents."""
        db, mock_conn = mock_db
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        mock_rows = [
            {
                "agent_name": "claude",
                "proposals_made": 100,
                "proposals_accepted": 80,
                "critiques_given": 50,
                "critiques_valuable": 40,
                "updated_at": now,
                "total_predictions": 30,
                "total_prediction_error": 3.0,
                "calibration_score": 0.9,
            },
            {
                "agent_name": "gpt4",
                "proposals_made": 50,
                "proposals_accepted": 35,
                "critiques_given": 30,
                "critiques_valuable": 20,
                "updated_at": now,
                "total_predictions": 15,
                "total_prediction_error": 2.0,
                "calibration_score": 0.85,
            },
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_all_reputations(limit=10)

        assert len(result) == 2
        assert result[0].agent_name == "claude"


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresAdaptiveForgetting:
    """Tests for adaptive forgetting (Titans/MIRAS)."""

    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 5")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        db = PostgresCritiqueStore(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        @asynccontextmanager
        async def mock_transaction_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        db.transaction = mock_transaction_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_prune_stale_patterns(self, mock_db):
        """prune_stale_patterns should remove old unsuccessful patterns."""
        db, mock_conn = mock_db
        mock_conn.execute.return_value = "DELETE 5"

        result = await db.prune_stale_patterns(
            max_age_days=90,
            min_success_rate=0.3,
            archive=True,
        )

        assert result == 5
        assert mock_conn.execute.call_count >= 2  # Archive + Delete

    @pytest.mark.asyncio
    async def test_get_archive_stats(self, mock_db):
        """get_archive_stats should return archive statistics."""
        db, mock_conn = mock_db

        mock_conn.fetchrow.return_value = {"total": 100}
        mock_conn.fetch.return_value = [
            {"issue_type": "security", "cnt": 40},
            {"issue_type": "performance", "cnt": 30},
        ]

        result = await db.get_archive_stats()

        assert result["total_archived"] == 100
        assert result["archived_by_type"]["security"] == 40


class TestPostgresCritiqueFactory:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_factory_raises_without_asyncpg(self):
        """get_postgres_critique_store should raise if asyncpg unavailable."""
        from aragora.memory import postgres_critique

        original = postgres_critique.ASYNCPG_AVAILABLE
        postgres_critique.ASYNCPG_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="asyncpg"):
                await postgres_critique.get_postgres_critique_store()
        finally:
            postgres_critique.ASYNCPG_AVAILABLE = original


class TestIssueCategorization:
    """Tests for issue categorization logic."""

    def test_categorize_security_issues(self):
        """Security-related issues should be categorized as 'security'."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        # Create minimal instance for testing
        store = object.__new__(PostgresCritiqueStore)

        assert store._categorize_issue("SQL injection vulnerability") == "security"
        assert store._categorize_issue("XSS attack possible") == "security"
        assert store._categorize_issue("Missing authentication") == "security"

    def test_categorize_performance_issues(self):
        """Performance-related issues should be categorized as 'performance'."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        store = object.__new__(PostgresCritiqueStore)

        assert store._categorize_issue("Slow database query") == "performance"
        assert store._categorize_issue("High latency") == "performance"
        assert store._categorize_issue("Need to optimize") == "performance"

    def test_categorize_correctness_issues(self):
        """Correctness issues should be categorized as 'correctness'."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        store = object.__new__(PostgresCritiqueStore)

        assert store._categorize_issue("Bug in calculation") == "correctness"
        assert store._categorize_issue("Error handling missing") == "correctness"
        assert store._categorize_issue("Incorrect result") == "correctness"

    def test_categorize_general_issues(self):
        """Unknown issues should be categorized as 'general'."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        store = object.__new__(PostgresCritiqueStore)

        assert store._categorize_issue("Something else entirely") == "general"
        assert store._categorize_issue("Random text here") == "general"


class TestPredictionOutcome:
    """Tests for prediction outcome tracking."""

    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        db = PostgresCritiqueStore(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        @asynccontextmanager
        async def mock_transaction_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        db.transaction = mock_transaction_ctx
        return db, mock_connection

    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
    @pytest.mark.asyncio
    async def test_update_prediction_outcome(self, mock_db):
        """update_prediction_outcome should calculate and store error."""
        db, mock_conn = mock_db

        mock_conn.fetchrow.return_value = {
            "expected_usefulness": 0.8,
            "agent": "claude",
        }

        error = await db.update_prediction_outcome(
            critique_id=1,
            actual_usefulness=0.5,
        )

        # |0.8 - 0.5| = 0.3
        assert error == pytest.approx(0.3)

    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
    @pytest.mark.asyncio
    async def test_update_prediction_outcome_missing_critique(self, mock_db):
        """update_prediction_outcome should return 0 for missing critique."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        error = await db.update_prediction_outcome(
            critique_id=999,
            actual_usefulness=0.5,
        )

        assert error == 0.0
