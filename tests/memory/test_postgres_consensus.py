"""
Tests for PostgreSQL Consensus Memory implementation.

These tests verify the PostgresConsensusMemory class provides the same
functionality as the SQLite-based ConsensusMemory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager
import json

import asyncpg


class TestPostgresConsensusSchema:
    """Tests for schema definitions (no asyncpg required)."""

    def test_schema_module_imports(self):
        """Module should import without errors."""
        from aragora.memory.postgres_consensus import (
            POSTGRES_CONSENSUS_SCHEMA,
            POSTGRES_CONSENSUS_SCHEMA_VERSION,
        )

        assert POSTGRES_CONSENSUS_SCHEMA_VERSION >= 1
        assert len(POSTGRES_CONSENSUS_SCHEMA) > 100

    def test_schema_has_required_tables(self):
        """Schema should define all required tables."""
        from aragora.memory.postgres_consensus import POSTGRES_CONSENSUS_SCHEMA

        required_tables = [
            "consensus",
            "dissent",
            "verified_proofs",
        ]

        for table in required_tables:
            assert table in POSTGRES_CONSENSUS_SCHEMA, f"Missing table: {table}"

    def test_schema_has_required_indexes(self):
        """Schema should define performance indexes."""
        from aragora.memory.postgres_consensus import POSTGRES_CONSENSUS_SCHEMA

        required_indexes = [
            "idx_consensus_topic_hash",
            "idx_consensus_domain",
            "idx_dissent_debate",
            "idx_dissent_type",
            "idx_verified_proofs_debate",
        ]

        for index in required_indexes:
            assert index in POSTGRES_CONSENSUS_SCHEMA, f"Missing index: {index}"

    def test_schema_uses_jsonb_for_data(self):
        """Schema should use JSONB for data fields."""
        from aragora.memory.postgres_consensus import POSTGRES_CONSENSUS_SCHEMA

        assert "JSONB" in POSTGRES_CONSENSUS_SCHEMA

    def test_schema_uses_timestamptz(self):
        """Schema should use TIMESTAMPTZ for timestamps."""
        from aragora.memory.postgres_consensus import POSTGRES_CONSENSUS_SCHEMA

        assert "TIMESTAMPTZ" in POSTGRES_CONSENSUS_SCHEMA

    def test_schema_has_foreign_keys(self):
        """Schema should have foreign keys for referential integrity."""
        from aragora.memory.postgres_consensus import POSTGRES_CONSENSUS_SCHEMA

        assert "REFERENCES consensus" in POSTGRES_CONSENSUS_SCHEMA


class TestPostgresConsensusMemory:
    """Tests for PostgresConsensusMemory class."""

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
        """Create a mocked PostgresConsensusMemory instance."""
        from aragora.memory.postgres_consensus import PostgresConsensusMemory

        db = PostgresConsensusMemory(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_store_consensus(self, mock_db):
        """store_consensus should create a new consensus record."""
        db, mock_conn = mock_db

        result = await db.store_consensus(
            topic="Rate limiting approach",
            conclusion="Use token bucket algorithm",
            strength="strong",
            confidence=0.85,
            participating_agents=["claude", "gpt4"],
            agreeing_agents=["claude", "gpt4"],
            domain="engineering",
        )

        mock_conn.execute.assert_called()
        assert result["topic"] == "Rate limiting approach"
        assert result["conclusion"] == "Use token bucket algorithm"
        assert result["strength"] == "strong"
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_consensus_returns_none_for_missing(self, mock_db):
        """get_consensus should return None for non-existent record."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        result = await db.get_consensus("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_consensus_returns_data(self, mock_db):
        """get_consensus should return record data."""
        db, mock_conn = mock_db
        mock_data = {
            "id": "consensus-123",
            "topic": "Test topic",
            "conclusion": "Test conclusion",
            "strength": "strong",
            "confidence": 0.9,
        }
        mock_conn.fetchrow.return_value = {"data": mock_data}

        result = await db.get_consensus("consensus-123")
        assert result is not None
        assert result["topic"] == "Test topic"

    @pytest.mark.asyncio
    async def test_find_similar_with_hash_match(self, mock_db):
        """find_similar should find exact hash matches first."""
        db, mock_conn = mock_db
        mock_data = {
            "id": "consensus-123",
            "topic": "Rate limiting",
            "conclusion": "Use token bucket",
            "confidence": 0.9,
        }
        mock_conn.fetchrow.return_value = {"data": mock_data}
        mock_conn.fetch.return_value = []

        result = await db.find_similar("Rate limiting", limit=5)
        assert len(result) == 1
        assert result[0]["topic"] == "Rate limiting"

    @pytest.mark.asyncio
    async def test_get_recent(self, mock_db):
        """get_recent should return recent consensus records."""
        db, mock_conn = mock_db
        mock_rows = [
            {"data": {"id": "1", "topic": "Topic 1", "confidence": 0.9}},
            {"data": {"id": "2", "topic": "Topic 2", "confidence": 0.8}},
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_recent(limit=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_count(self, mock_db):
        """count should return total consensus count."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"count": 42}

        result = await db.count()
        assert result == 42


class TestPostgresDissentOperations:
    """Tests for dissent operations."""

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
        from aragora.memory.postgres_consensus import PostgresConsensusMemory

        db = PostgresConsensusMemory(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_store_dissent(self, mock_db):
        """store_dissent should create a new dissent record."""
        db, mock_conn = mock_db

        result = await db.store_dissent(
            debate_id="consensus-123",
            agent_id="claude",
            dissent_type="alternative_approach",
            content="Consider rate limiting by IP instead",
            reasoning="Better for distributed systems",
            confidence=0.7,
        )

        assert mock_conn.execute.call_count >= 1
        assert result["debate_id"] == "consensus-123"
        assert result["agent_id"] == "claude"
        assert result["dissent_type"] == "alternative_approach"

    @pytest.mark.asyncio
    async def test_get_dissents_for_debate(self, mock_db):
        """get_dissents_for_debate should return all dissents for a debate."""
        db, mock_conn = mock_db
        mock_rows = [
            {"data": {"id": "d1", "agent_id": "claude", "content": "Dissent 1"}},
            {"data": {"id": "d2", "agent_id": "gpt4", "content": "Dissent 2"}},
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_dissents_for_debate("consensus-123")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_acknowledge_dissent(self, mock_db):
        """acknowledge_dissent should mark dissent as acknowledged."""
        db, mock_conn = mock_db
        mock_conn.execute.return_value = "UPDATE 1"

        result = await db.acknowledge_dissent(
            dissent_id="dissent-123",
            rebuttal="Good point, but token bucket handles this case",
        )

        assert result is True
        mock_conn.execute.assert_called()


class TestPostgresProofOperations:
    """Tests for verified proof operations."""

    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 1")
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        from aragora.memory.postgres_consensus import PostgresConsensusMemory

        db = PostgresConsensusMemory(mock_pool)

        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_store_proof(self, mock_db):
        """store_proof should create a new proof record."""
        db, mock_conn = mock_db

        result = await db.store_proof(
            debate_id="consensus-123",
            proof_status="verified",
            language="lean4",
            formal_statement="theorem rate_limit_sound",
            is_verified=True,
            prover_version="lean4-4.0.0",
        )

        mock_conn.execute.assert_called()
        assert result["debate_id"] == "consensus-123"
        assert result["proof_status"] == "verified"
        assert result["is_verified"] is True

    @pytest.mark.asyncio
    async def test_get_verified_proofs(self, mock_db):
        """get_verified_proofs should return verified proofs only."""
        db, mock_conn = mock_db
        mock_rows = [
            {"data": {"id": "p1", "is_verified": True, "language": "lean4"}},
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_verified_proofs(limit=10)
        assert len(result) == 1
        assert result[0]["is_verified"] is True


class TestPostgresConsensusFactory:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_factory_raises_without_asyncpg(self):
        """get_postgres_consensus_memory should raise if asyncpg unavailable."""
        from aragora.memory import postgres_consensus

        original = postgres_consensus.ASYNCPG_AVAILABLE
        postgres_consensus.ASYNCPG_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="asyncpg"):
                await postgres_consensus.get_postgres_consensus_memory()
        finally:
            postgres_consensus.ASYNCPG_AVAILABLE = original


class TestTopicHashing:
    """Tests for topic hashing functionality."""

    def test_hash_topic_normalizes(self):
        """Topic hash should normalize input."""
        import hashlib

        def hash_topic(topic: str) -> str:
            """Replicate the hashing logic."""
            words = sorted(set(topic.lower().split()))
            normalized = " ".join(words)
            return hashlib.sha256(normalized.encode()).hexdigest()[:16]

        hash1 = hash_topic("Rate Limiting Approach")
        hash2 = hash_topic("approach rate limiting")
        hash3 = hash_topic("RATE LIMITING APPROACH")

        # All should produce same hash (normalized)
        assert hash1 == hash2 == hash3

    def test_hash_different_topics(self):
        """Different topics should have different hashes."""
        import hashlib

        def hash_topic(topic: str) -> str:
            """Replicate the hashing logic."""
            words = sorted(set(topic.lower().split()))
            normalized = " ".join(words)
            return hashlib.sha256(normalized.encode()).hexdigest()[:16]

        hash1 = hash_topic("Rate limiting")
        hash2 = hash_topic("Authentication methods")

        assert hash1 != hash2
