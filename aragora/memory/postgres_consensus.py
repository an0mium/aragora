"""
PostgreSQL implementation of Consensus Memory.

Provides async PostgreSQL-backed storage for debate consensus and dissent
with connection pooling for production deployments requiring horizontal
scaling and concurrent writes.

Usage:
    from aragora.memory.postgres_consensus import get_postgres_consensus_memory

    # Get the PostgreSQL consensus memory instance
    cms = await get_postgres_consensus_memory()

    # Store a consensus
    record = await cms.store_consensus(
        topic="Rate limiting approach",
        conclusion="Use token bucket algorithm",
        strength=ConsensusStrength.STRONG,
        confidence=0.85,
        participating_agents=["claude", "gpt4"],
        agreeing_agents=["claude", "gpt4"],
    )

    # Find similar past debates
    similar = await cms.find_similar(topic="API rate limits", limit=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from aragora.storage.postgres_store import PostgresStore, ASYNCPG_AVAILABLE

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


# PostgreSQL schema version for consensus memory
POSTGRES_CONSENSUS_SCHEMA_VERSION = 1

# PostgreSQL-optimized schema
POSTGRES_CONSENSUS_SCHEMA = """
    -- Main consensus table
    CREATE TABLE IF NOT EXISTS consensus (
        id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        topic_hash TEXT NOT NULL,
        conclusion TEXT NOT NULL,
        strength TEXT NOT NULL,
        confidence REAL,
        domain TEXT DEFAULT 'general',
        tags JSONB DEFAULT '[]',
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        data JSONB NOT NULL
    );

    -- Dissent records table
    CREATE TABLE IF NOT EXISTS dissent (
        id TEXT PRIMARY KEY,
        debate_id TEXT NOT NULL REFERENCES consensus(id) ON DELETE CASCADE,
        agent_id TEXT NOT NULL,
        dissent_type TEXT NOT NULL,
        content TEXT NOT NULL,
        reasoning TEXT DEFAULT '',
        confidence REAL DEFAULT 0.0,
        acknowledged BOOLEAN DEFAULT FALSE,
        rebuttal TEXT DEFAULT '',
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        data JSONB NOT NULL
    );

    -- Verified proofs table for formal verification results
    CREATE TABLE IF NOT EXISTS verified_proofs (
        id TEXT PRIMARY KEY,
        debate_id TEXT NOT NULL REFERENCES consensus(id) ON DELETE CASCADE,
        proof_status TEXT NOT NULL,
        language TEXT,
        formal_statement TEXT,
        is_verified BOOLEAN DEFAULT FALSE,
        proof_hash TEXT,
        translation_time_ms REAL,
        proof_search_time_ms REAL,
        prover_version TEXT,
        error_message TEXT,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        data JSONB NOT NULL
    );

    -- Performance indexes
    CREATE INDEX IF NOT EXISTS idx_consensus_topic_hash ON consensus(topic_hash);
    CREATE INDEX IF NOT EXISTS idx_consensus_domain ON consensus(domain);
    CREATE INDEX IF NOT EXISTS idx_consensus_confidence_ts ON consensus(confidence DESC, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_consensus_strength ON consensus(strength);
    CREATE INDEX IF NOT EXISTS idx_consensus_timestamp ON consensus(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_dissent_debate ON dissent(debate_id);
    CREATE INDEX IF NOT EXISTS idx_dissent_type ON dissent(dissent_type);
    CREATE INDEX IF NOT EXISTS idx_dissent_agent ON dissent(agent_id);
    CREATE INDEX IF NOT EXISTS idx_dissent_timestamp ON dissent(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_verified_proofs_debate ON verified_proofs(debate_id);
    CREATE INDEX IF NOT EXISTS idx_verified_proofs_status ON verified_proofs(proof_status);
    CREATE INDEX IF NOT EXISTS idx_verified_proofs_verified ON verified_proofs(is_verified);
"""


class PostgresConsensusMemory(PostgresStore):
    """
    PostgreSQL implementation of Consensus Memory.

    Provides async operations for storing and retrieving debate consensus
    and dissent records with:
    - Connection pooling for horizontal scaling
    - JSONB for efficient metadata queries
    - TIMESTAMPTZ for proper timestamp handling
    - Topic hashing for similarity search

    Usage:
        pool = await get_postgres_pool()
        cms = PostgresConsensusMemory(pool)
        await cms.initialize()

        # Store consensus
        record = await cms.store_consensus(...)

        # Find similar debates
        similar = await cms.find_similar(topic="...", limit=5)
    """

    SCHEMA_NAME = "consensus_memory"
    SCHEMA_VERSION = POSTGRES_CONSENSUS_SCHEMA_VERSION
    INITIAL_SCHEMA = POSTGRES_CONSENSUS_SCHEMA

    def __init__(self, pool: "Pool"):
        """
        Initialize PostgreSQL consensus memory.

        Args:
            pool: asyncpg connection pool
        """
        super().__init__(pool)

    # =========================================================================
    # Topic Hashing
    # =========================================================================

    def _hash_topic(self, topic: str) -> str:
        """Create a hash for topic similarity matching."""
        words = sorted(set(topic.lower().split()))
        normalized = " ".join(words)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    # =========================================================================
    # Consensus Operations
    # =========================================================================

    async def store_consensus(
        self,
        topic: str,
        conclusion: str,
        strength: str,
        confidence: float,
        participating_agents: list[str],
        agreeing_agents: list[str],
        dissenting_agents: Optional[list[str]] = None,
        key_claims: Optional[list[str]] = None,
        supporting_evidence: Optional[list[str]] = None,
        domain: str = "general",
        tags: Optional[list[str]] = None,
        debate_duration: float = 0.0,
        rounds: int = 0,
        supersedes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Store a new consensus record.

        Args:
            topic: Debate topic
            conclusion: Final consensus conclusion
            strength: Consensus strength (unanimous, strong, moderate, weak, split, contested)
            confidence: Confidence level (0-1)
            participating_agents: List of agent IDs that participated
            agreeing_agents: List of agent IDs that agreed
            dissenting_agents: List of agent IDs that dissented
            key_claims: Key claims that led to consensus
            supporting_evidence: Evidence supporting the conclusion
            domain: Domain/category of the debate
            tags: Tags for categorization
            debate_duration: Duration in seconds
            rounds: Number of debate rounds
            supersedes: ID of consensus this replaces
            metadata: Additional metadata

        Returns:
            Created consensus record as dict
        """
        record_id = str(uuid.uuid4())
        topic_hash = self._hash_topic(topic)
        now = datetime.now()

        data = {
            "id": record_id,
            "topic": topic,
            "topic_hash": topic_hash,
            "conclusion": conclusion,
            "strength": strength,
            "confidence": confidence,
            "participating_agents": participating_agents,
            "agreeing_agents": agreeing_agents,
            "dissenting_agents": dissenting_agents or [],
            "key_claims": key_claims or [],
            "supporting_evidence": supporting_evidence or [],
            "dissent_ids": [],
            "domain": domain,
            "tags": tags or [],
            "timestamp": now.isoformat(),
            "debate_duration_seconds": debate_duration,
            "rounds": rounds,
            "supersedes": supersedes,
            "superseded_by": None,
            "metadata": metadata or {},
        }

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO consensus
                (id, topic, topic_hash, conclusion, strength, confidence, domain, tags, timestamp, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                record_id,
                topic,
                topic_hash,
                conclusion,
                strength,
                confidence,
                domain,
                json.dumps(tags or []),
                now,
                json.dumps(data),
            )

            # Mark old consensus as superseded if provided
            if supersedes:
                await conn.execute(
                    """
                    UPDATE consensus
                    SET data = jsonb_set(data, '{superseded_by}', $1::jsonb)
                    WHERE id = $2
                    """,
                    json.dumps(record_id),
                    supersedes,
                )

        return data

    async def get_consensus(self, consensus_id: str) -> Optional[dict[str, Any]]:
        """
        Get a consensus record by ID.

        Args:
            consensus_id: The consensus ID

        Returns:
            Consensus record dict or None if not found
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM consensus WHERE id = $1",
                consensus_id,
            )

        if not row:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return data

    async def get_by_topic_hash(self, topic_hash: str) -> Optional[dict[str, Any]]:
        """
        Get the most recent consensus for a topic hash.

        Args:
            topic_hash: Hash of the topic

        Returns:
            Most recent consensus record or None
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT data FROM consensus
                WHERE topic_hash = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                topic_hash,
            )

        if not row:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return data

    async def find_similar(
        self,
        topic: str,
        limit: int = 5,
        min_confidence: float = 0.0,
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Find similar past debates by topic.

        Uses topic hash for exact matches and keyword search for partial matches.

        Args:
            topic: Topic to search for
            limit: Maximum results
            min_confidence: Minimum confidence threshold
            domain: Optional domain filter

        Returns:
            List of similar consensus records
        """
        topic_hash = self._hash_topic(topic)
        keywords = [kw.strip().lower() for kw in topic.split()[:20] if kw.strip()]

        async with self.connection() as conn:
            # First try exact hash match
            exact_match = await conn.fetchrow(
                """
                SELECT data FROM consensus
                WHERE topic_hash = $1 AND confidence >= $2
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                topic_hash,
                min_confidence,
            )

            results = []
            if exact_match:
                data = exact_match["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                results.append(data)

            # Then keyword search
            if keywords and len(results) < limit:
                keyword_conditions = [f"LOWER(topic) LIKE ${i + 3}" for i in range(len(keywords))]

                query = f"""
                    SELECT data FROM consensus
                    WHERE ({" OR ".join(keyword_conditions)})
                      AND confidence >= $1
                      AND topic_hash != $2
                """

                if domain:
                    query += f" AND domain = ${len(keywords) + 3}"
                    params = (
                        [min_confidence, topic_hash] + [f"%{kw}%" for kw in keywords] + [domain]
                    )
                else:
                    params = [min_confidence, topic_hash] + [f"%{kw}%" for kw in keywords]

                query += f" ORDER BY confidence DESC, timestamp DESC LIMIT ${len(params) + 1}"
                params.append(limit - len(results))

                rows = await conn.fetch(query, *params)

                for row in rows:
                    data = row["data"]
                    if isinstance(data, str):
                        data = json.loads(data)
                    results.append(data)

        return results

    async def get_recent(
        self,
        limit: int = 10,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Get recent consensus records.

        Args:
            limit: Maximum results
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of recent consensus records
        """
        async with self.connection() as conn:
            if domain:
                rows = await conn.fetch(
                    """
                    SELECT data FROM consensus
                    WHERE domain = $1 AND confidence >= $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                    """,
                    domain,
                    min_confidence,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT data FROM consensus
                    WHERE confidence >= $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    min_confidence,
                    limit,
                )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    async def get_by_domain(
        self,
        domain: str,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Get all consensus records for a domain."""
        return await self.get_recent(limit=limit, domain=domain, min_confidence=min_confidence)

    async def count(self, domain: Optional[str] = None) -> int:  # type: ignore[override]
        """Count consensus records, optionally filtered by domain."""
        async with self.connection() as conn:
            if domain:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM consensus WHERE domain = $1",
                    domain,
                )
            else:
                row = await conn.fetchrow("SELECT COUNT(*) as count FROM consensus")

        return row["count"] if row else 0

    # =========================================================================
    # Dissent Operations
    # =========================================================================

    async def store_dissent(
        self,
        debate_id: str,
        agent_id: str,
        dissent_type: str,
        content: str,
        reasoning: str = "",
        confidence: float = 0.0,
        acknowledged: bool = False,
        rebuttal: str = "",
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Store a dissent record.

        Args:
            debate_id: ID of the consensus this dissents from
            agent_id: ID of the dissenting agent
            dissent_type: Type of dissent
            content: Dissent content
            reasoning: Reasoning for the dissent
            confidence: Confidence in the dissent
            acknowledged: Whether the dissent was acknowledged
            rebuttal: Majority's rebuttal if any
            metadata: Additional metadata

        Returns:
            Created dissent record as dict
        """
        dissent_id = str(uuid.uuid4())
        now = datetime.now()

        data = {
            "id": dissent_id,
            "debate_id": debate_id,
            "agent_id": agent_id,
            "dissent_type": dissent_type,
            "content": content,
            "reasoning": reasoning,
            "confidence": confidence,
            "acknowledged": acknowledged,
            "rebuttal": rebuttal,
            "timestamp": now.isoformat(),
            "metadata": metadata or {},
        }

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO dissent
                (id, debate_id, agent_id, dissent_type, content, reasoning,
                 confidence, acknowledged, rebuttal, timestamp, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                dissent_id,
                debate_id,
                agent_id,
                dissent_type,
                content,
                reasoning,
                confidence,
                acknowledged,
                rebuttal,
                now,
                json.dumps(data),
            )

            # Update consensus with dissent ID
            await conn.execute(
                """
                UPDATE consensus
                SET data = jsonb_set(
                    data,
                    '{dissent_ids}',
                    (COALESCE(data->'dissent_ids', '[]'::jsonb) || $1::jsonb)
                )
                WHERE id = $2
                """,
                json.dumps([dissent_id]),
                debate_id,
            )

        return data

    async def get_dissent(self, dissent_id: str) -> Optional[dict[str, Any]]:
        """Get a dissent record by ID."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM dissent WHERE id = $1",
                dissent_id,
            )

        if not row:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return data

    async def get_dissents_for_debate(self, debate_id: str) -> list[dict[str, Any]]:
        """Get all dissent records for a debate."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM dissent
                WHERE debate_id = $1
                ORDER BY timestamp ASC
                """,
                debate_id,
            )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    async def get_dissents_by_agent(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get all dissent records by a specific agent."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM dissent
                WHERE agent_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    async def get_dissents_by_type(
        self,
        dissent_type: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get dissent records by type."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM dissent
                WHERE dissent_type = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                dissent_type,
                limit,
            )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    async def acknowledge_dissent(
        self,
        dissent_id: str,
        rebuttal: str = "",
    ) -> bool:
        """Mark a dissent as acknowledged."""
        async with self.connection() as conn:
            result = await conn.execute(
                """
                UPDATE dissent
                SET acknowledged = TRUE,
                    rebuttal = $1,
                    data = jsonb_set(
                        jsonb_set(data, '{acknowledged}', 'true'),
                        '{rebuttal}', $2::jsonb
                    )
                WHERE id = $3
                """,
                rebuttal,
                json.dumps(rebuttal),
                dissent_id,
            )
            return result == "UPDATE 1"

    # =========================================================================
    # Verified Proofs Operations
    # =========================================================================

    async def store_proof(
        self,
        debate_id: str,
        proof_status: str,
        language: Optional[str] = None,
        formal_statement: Optional[str] = None,
        is_verified: bool = False,
        proof_hash: Optional[str] = None,
        translation_time_ms: Optional[float] = None,
        proof_search_time_ms: Optional[float] = None,
        prover_version: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Store a verification proof record."""
        proof_id = str(uuid.uuid4())
        now = datetime.now()

        data = {
            "id": proof_id,
            "debate_id": debate_id,
            "proof_status": proof_status,
            "language": language,
            "formal_statement": formal_statement,
            "is_verified": is_verified,
            "proof_hash": proof_hash,
            "translation_time_ms": translation_time_ms,
            "proof_search_time_ms": proof_search_time_ms,
            "prover_version": prover_version,
            "error_message": error_message,
            "timestamp": now.isoformat(),
            "metadata": metadata or {},
        }

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO verified_proofs
                (id, debate_id, proof_status, language, formal_statement, is_verified,
                 proof_hash, translation_time_ms, proof_search_time_ms, prover_version,
                 error_message, timestamp, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                proof_id,
                debate_id,
                proof_status,
                language,
                formal_statement,
                is_verified,
                proof_hash,
                translation_time_ms,
                proof_search_time_ms,
                prover_version,
                error_message,
                now,
                json.dumps(data),
            )

        return data

    async def get_proofs_for_debate(self, debate_id: str) -> list[dict[str, Any]]:
        """Get all proof records for a debate."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM verified_proofs
                WHERE debate_id = $1
                ORDER BY timestamp ASC
                """,
                debate_id,
            )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    async def get_verified_proofs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get successfully verified proofs."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM verified_proofs
                WHERE is_verified = TRUE
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit,
            )

        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append(data)
        return results

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the consensus memory."""
        async with self.connection() as conn:
            # Count by strength
            strength_counts = await conn.fetch("""
                SELECT strength, COUNT(*) as count
                FROM consensus
                GROUP BY strength
                """)

            # Count by domain
            domain_counts = await conn.fetch("""
                SELECT domain, COUNT(*) as count
                FROM consensus
                GROUP BY domain
                ORDER BY count DESC
                LIMIT 10
                """)

            # Total counts
            totals = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM consensus) as total_consensus,
                    (SELECT COUNT(*) FROM dissent) as total_dissent,
                    (SELECT COUNT(*) FROM verified_proofs) as total_proofs,
                    (SELECT COUNT(*) FROM verified_proofs WHERE is_verified = TRUE) as verified_proofs,
                    (SELECT AVG(confidence) FROM consensus) as avg_confidence
                """)

        return {
            "total_consensus": totals["total_consensus"] if totals else 0,
            "total_dissent": totals["total_dissent"] if totals else 0,
            "total_proofs": totals["total_proofs"] if totals else 0,
            "verified_proofs": totals["verified_proofs"] if totals else 0,
            "avg_confidence": (
                float(totals["avg_confidence"]) if totals and totals["avg_confidence"] else 0.0
            ),
            "by_strength": {row["strength"]: row["count"] for row in strength_counts},
            "by_domain": {row["domain"]: row["count"] for row in domain_counts},
        }


# =========================================================================
# Factory Function
# =========================================================================

_postgres_consensus_memory: Optional[PostgresConsensusMemory] = None


async def get_postgres_consensus_memory(
    pool: Optional["Pool"] = None,
) -> PostgresConsensusMemory:
    """
    Get or create the PostgreSQL consensus memory instance.

    Args:
        pool: Optional asyncpg pool. If not provided, gets from settings.

    Returns:
        PostgresConsensusMemory instance

    Raises:
        RuntimeError: If asyncpg is not installed or PostgreSQL not configured
    """
    global _postgres_consensus_memory

    if not ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install aragora[postgres] or pip install asyncpg"
        )

    if _postgres_consensus_memory is not None:
        return _postgres_consensus_memory

    if pool is None:
        from aragora.storage.postgres_store import get_postgres_pool_from_settings

        pool = await get_postgres_pool_from_settings()

    _postgres_consensus_memory = PostgresConsensusMemory(pool)
    await _postgres_consensus_memory.initialize()

    return _postgres_consensus_memory
