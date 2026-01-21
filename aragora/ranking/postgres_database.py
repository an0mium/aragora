"""
PostgreSQL database implementation for the ELO ranking system.

Provides async PostgreSQL-backed storage for ELO ratings, matches, and calibration.
This is the production-ready alternative to the SQLite-based EloDatabase.

Usage:
    from aragora.ranking.postgres_database import PostgresEloDatabase

    # Initialize with existing pool
    pool = await get_postgres_pool()
    db = PostgresEloDatabase(pool)
    await db.initialize()

    # Or use factory function
    db = await get_postgres_elo_database()
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from aragora.storage.postgres_store import PostgresStore, get_postgres_pool, ASYNCPG_AVAILABLE

logger = logging.getLogger(__name__)

__all__ = [
    "PostgresEloDatabase",
    "get_postgres_elo_database",
    "POSTGRES_ELO_SCHEMA_VERSION",
]

POSTGRES_ELO_SCHEMA_VERSION = 1

# PostgreSQL schema (equivalent to SQLite schema in database.py)
POSTGRES_ELO_SCHEMA = """
    -- Agent ratings
    CREATE TABLE IF NOT EXISTS elo_ratings (
        agent_name TEXT PRIMARY KEY,
        elo REAL DEFAULT 1500,
        domain_elos JSONB DEFAULT '{}',
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        debates_count INTEGER DEFAULT 0,
        critiques_accepted INTEGER DEFAULT 0,
        critiques_total INTEGER DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Match history
    CREATE TABLE IF NOT EXISTS elo_matches (
        id SERIAL PRIMARY KEY,
        debate_id TEXT UNIQUE,
        winner TEXT,
        participants JSONB,
        domain TEXT,
        scores JSONB,
        elo_changes JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- ELO history for tracking progression
    CREATE TABLE IF NOT EXISTS elo_history (
        id SERIAL PRIMARY KEY,
        agent_name TEXT NOT NULL,
        elo REAL NOT NULL,
        debate_id TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Calibration predictions table
    CREATE TABLE IF NOT EXISTS elo_calibration_predictions (
        id SERIAL PRIMARY KEY,
        tournament_id TEXT NOT NULL,
        predictor_agent TEXT NOT NULL,
        predicted_winner TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(tournament_id, predictor_agent)
    );

    -- Domain-specific calibration tracking
    CREATE TABLE IF NOT EXISTS elo_domain_calibration (
        agent_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        total_predictions INTEGER DEFAULT 0,
        total_correct INTEGER DEFAULT 0,
        brier_sum REAL DEFAULT 0.0,
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (agent_name, domain)
    );

    -- Calibration by confidence bucket
    CREATE TABLE IF NOT EXISTS elo_calibration_buckets (
        agent_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        bucket_key TEXT NOT NULL,
        predictions INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        brier_sum REAL DEFAULT 0.0,
        PRIMARY KEY (agent_name, domain, bucket_key)
    );

    -- Agent relationships tracking
    CREATE TABLE IF NOT EXISTS elo_agent_relationships (
        agent_a TEXT NOT NULL,
        agent_b TEXT NOT NULL,
        debate_count INTEGER DEFAULT 0,
        agreement_count INTEGER DEFAULT 0,
        critique_count_a_to_b INTEGER DEFAULT 0,
        critique_count_b_to_a INTEGER DEFAULT 0,
        critique_accepted_a_to_b INTEGER DEFAULT 0,
        critique_accepted_b_to_a INTEGER DEFAULT 0,
        avg_critique_severity_a_to_b REAL DEFAULT 0.0,
        avg_critique_severity_b_to_a REAL DEFAULT 0.0,
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (agent_a, agent_b)
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_elo_matches_created_at ON elo_matches(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_elo_matches_winner ON elo_matches(winner);
    CREATE INDEX IF NOT EXISTS idx_elo_history_agent ON elo_history(agent_name, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_elo_ratings_elo ON elo_ratings(elo DESC);
"""


class PostgresEloDatabase(PostgresStore):
    """
    PostgreSQL implementation of ELO database.

    Provides the same interface as EloDatabase but uses PostgreSQL
    for production scalability and concurrent access.
    """

    SCHEMA_NAME = "elo"
    SCHEMA_VERSION = POSTGRES_ELO_SCHEMA_VERSION
    INITIAL_SCHEMA = POSTGRES_ELO_SCHEMA

    async def get_rating(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get rating for an agent."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM elo_ratings WHERE agent_name = $1",
                agent_name,
            )
            if row:
                return dict(row)
            return None

    async def set_rating(
        self,
        agent_name: str,
        elo: float,
        domain_elos: Optional[dict] = None,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
        debates_count: int = 0,
        critiques_accepted: int = 0,
        critiques_total: int = 0,
    ) -> None:
        """Set or update rating for an agent."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO elo_ratings (
                    agent_name, elo, domain_elos, wins, losses, draws,
                    debates_count, critiques_accepted, critiques_total, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (agent_name) DO UPDATE SET
                    elo = EXCLUDED.elo,
                    domain_elos = EXCLUDED.domain_elos,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    draws = EXCLUDED.draws,
                    debates_count = EXCLUDED.debates_count,
                    critiques_accepted = EXCLUDED.critiques_accepted,
                    critiques_total = EXCLUDED.critiques_total,
                    updated_at = NOW()
                """,
                agent_name,
                elo,
                json.dumps(domain_elos or {}),
                wins,
                losses,
                draws,
                debates_count,
                critiques_accepted,
                critiques_total,
            )

    async def update_elo(
        self,
        agent_name: str,
        new_elo: float,
        domain: Optional[str] = None,
        domain_elo: Optional[float] = None,
    ) -> None:
        """Update ELO rating for an agent."""
        async with self.connection() as conn:
            if domain and domain_elo is not None:
                # Update domain-specific ELO
                await conn.execute(
                    """
                    UPDATE elo_ratings
                    SET elo = $2,
                        domain_elos = jsonb_set(
                            COALESCE(domain_elos, '{}'),
                            $3::text[],
                            to_jsonb($4::float)
                        ),
                        updated_at = NOW()
                    WHERE agent_name = $1
                    """,
                    agent_name,
                    new_elo,
                    [domain],
                    domain_elo,
                )
            else:
                await conn.execute(
                    """
                    UPDATE elo_ratings
                    SET elo = $2, updated_at = NOW()
                    WHERE agent_name = $1
                    """,
                    agent_name,
                    new_elo,
                )

    async def increment_stats(
        self,
        agent_name: str,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
        debates: int = 0,
        critiques_accepted: int = 0,
        critiques_total: int = 0,
    ) -> None:
        """Increment statistics for an agent."""
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE elo_ratings
                SET wins = wins + $2,
                    losses = losses + $3,
                    draws = draws + $4,
                    debates_count = debates_count + $5,
                    critiques_accepted = critiques_accepted + $6,
                    critiques_total = critiques_total + $7,
                    updated_at = NOW()
                WHERE agent_name = $1
                """,
                agent_name,
                wins,
                losses,
                draws,
                debates,
                critiques_accepted,
                critiques_total,
            )

    async def get_leaderboard(
        self,
        limit: int = 10,
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get top agents by ELO rating."""
        async with self.connection() as conn:
            if domain:
                rows = await conn.fetch(
                    """
                    SELECT agent_name, elo,
                           (domain_elos->>$2)::float as domain_elo,
                           wins, losses, draws, debates_count
                    FROM elo_ratings
                    WHERE domain_elos ? $2
                    ORDER BY (domain_elos->>$2)::float DESC
                    LIMIT $1
                    """,
                    limit,
                    domain,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT agent_name, elo, domain_elos,
                           wins, losses, draws, debates_count
                    FROM elo_ratings
                    ORDER BY elo DESC
                    LIMIT $1
                    """,
                    limit,
                )
            return [dict(row) for row in rows]

    async def save_match(
        self,
        debate_id: str,
        winner: Optional[str],
        participants: list[str],
        domain: Optional[str],
        scores: dict[str, float],
        elo_changes: dict[str, float],
    ) -> int:
        """Save a match result."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO elo_matches (
                    debate_id, winner, participants, domain, scores, elo_changes
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (debate_id) DO UPDATE SET
                    winner = EXCLUDED.winner,
                    participants = EXCLUDED.participants,
                    domain = EXCLUDED.domain,
                    scores = EXCLUDED.scores,
                    elo_changes = EXCLUDED.elo_changes
                RETURNING id
                """,
                debate_id,
                winner,
                json.dumps(participants),
                domain,
                json.dumps(scores),
                json.dumps(elo_changes),
            )
            return row["id"] if row else 0

    async def get_match(self, debate_id: str) -> Optional[dict[str, Any]]:
        """Get a match by debate ID."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM elo_matches WHERE debate_id = $1",
                debate_id,
            )
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get("participants"):
                    result["participants"] = json.loads(result["participants"])
                if result.get("scores"):
                    result["scores"] = json.loads(result["scores"])
                if result.get("elo_changes"):
                    result["elo_changes"] = json.loads(result["elo_changes"])
                return result
            return None

    async def get_recent_matches(
        self,
        agent_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent matches, optionally filtered by agent."""
        async with self.connection() as conn:
            if agent_name:
                rows = await conn.fetch(
                    """
                    SELECT * FROM elo_matches
                    WHERE participants::text LIKE $2
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                    f'%"{agent_name}"%',
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM elo_matches
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

            results = []
            for row in rows:
                result = dict(row)
                if result.get("participants"):
                    result["participants"] = json.loads(result["participants"])
                if result.get("scores"):
                    result["scores"] = json.loads(result["scores"])
                if result.get("elo_changes"):
                    result["elo_changes"] = json.loads(result["elo_changes"])
                results.append(result)
            return results

    async def save_elo_history(
        self,
        agent_name: str,
        elo: float,
        debate_id: Optional[str] = None,
    ) -> None:
        """Save ELO history entry."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO elo_history (agent_name, elo, debate_id)
                VALUES ($1, $2, $3)
                """,
                agent_name,
                elo,
                debate_id,
            )

    async def get_elo_history(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get ELO history for an agent."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM elo_history
                WHERE agent_name = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                agent_name,
                limit,
            )
            return [dict(row) for row in rows]

    async def save_calibration_prediction(
        self,
        tournament_id: str,
        predictor_agent: str,
        predicted_winner: str,
        confidence: float,
    ) -> None:
        """Save a calibration prediction."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO elo_calibration_predictions (
                    tournament_id, predictor_agent, predicted_winner, confidence
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (tournament_id, predictor_agent) DO UPDATE SET
                    predicted_winner = EXCLUDED.predicted_winner,
                    confidence = EXCLUDED.confidence
                """,
                tournament_id,
                predictor_agent,
                predicted_winner,
                confidence,
            )

    async def update_domain_calibration(
        self,
        agent_name: str,
        domain: str,
        correct: bool,
        brier_score: float,
    ) -> None:
        """Update domain calibration stats."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO elo_domain_calibration (
                    agent_name, domain, total_predictions, total_correct, brier_sum
                ) VALUES ($1, $2, 1, $3, $4)
                ON CONFLICT (agent_name, domain) DO UPDATE SET
                    total_predictions = elo_domain_calibration.total_predictions + 1,
                    total_correct = elo_domain_calibration.total_correct + $3,
                    brier_sum = elo_domain_calibration.brier_sum + $4,
                    updated_at = NOW()
                """,
                agent_name,
                domain,
                1 if correct else 0,
                brier_score,
            )

    async def get_domain_calibration(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get domain calibration stats for an agent."""
        async with self.connection() as conn:
            if domain:
                rows = await conn.fetch(
                    """
                    SELECT * FROM elo_domain_calibration
                    WHERE agent_name = $1 AND domain = $2
                    """,
                    agent_name,
                    domain,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM elo_domain_calibration
                    WHERE agent_name = $1
                    """,
                    agent_name,
                )
            return [dict(row) for row in rows]

    async def update_relationship(
        self,
        agent_a: str,
        agent_b: str,
        debate_increment: int = 0,
        agreement_increment: int = 0,
        critique_a_to_b: int = 0,
        critique_b_to_a: int = 0,
        accepted_a_to_b: int = 0,
        accepted_b_to_a: int = 0,
    ) -> None:
        """Update relationship tracking between two agents."""
        # Normalize order (always store with alphabetically first agent as agent_a)
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
            critique_a_to_b, critique_b_to_a = critique_b_to_a, critique_a_to_b
            accepted_a_to_b, accepted_b_to_a = accepted_b_to_a, accepted_a_to_b

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO elo_agent_relationships (
                    agent_a, agent_b, debate_count, agreement_count,
                    critique_count_a_to_b, critique_count_b_to_a,
                    critique_accepted_a_to_b, critique_accepted_b_to_a
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (agent_a, agent_b) DO UPDATE SET
                    debate_count = elo_agent_relationships.debate_count + $3,
                    agreement_count = elo_agent_relationships.agreement_count + $4,
                    critique_count_a_to_b = elo_agent_relationships.critique_count_a_to_b + $5,
                    critique_count_b_to_a = elo_agent_relationships.critique_count_b_to_a + $6,
                    critique_accepted_a_to_b = elo_agent_relationships.critique_accepted_a_to_b + $7,
                    critique_accepted_b_to_a = elo_agent_relationships.critique_accepted_b_to_a + $8,
                    updated_at = NOW()
                """,
                agent_a,
                agent_b,
                debate_increment,
                agreement_increment,
                critique_a_to_b,
                critique_b_to_a,
                accepted_a_to_b,
                accepted_b_to_a,
            )

    async def get_relationship(
        self,
        agent_a: str,
        agent_b: str,
    ) -> Optional[dict[str, Any]]:
        """Get relationship stats between two agents."""
        # Normalize order
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a

        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM elo_agent_relationships
                WHERE agent_a = $1 AND agent_b = $2
                """,
                agent_a,
                agent_b,
            )
            if row:
                return dict(row)
            return None

    async def get_all_ratings(self) -> list[dict[str, Any]]:
        """Get all agent ratings."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM elo_ratings ORDER BY elo DESC"
            )
            return [dict(row) for row in rows]

    async def delete_rating(self, agent_name: str) -> bool:
        """Delete an agent's rating."""
        async with self.connection() as conn:
            result = await conn.execute(
                "DELETE FROM elo_ratings WHERE agent_name = $1",
                agent_name,
            )
            return "DELETE 1" in result

    async def count_ratings(self) -> int:
        """Count total number of rated agents."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM elo_ratings")
            return row["count"] if row else 0

    async def count_matches(self) -> int:
        """Count total number of matches."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM elo_matches")
            return row["count"] if row else 0


# Singleton instance
_postgres_elo_db: Optional[PostgresEloDatabase] = None


async def get_postgres_elo_database() -> PostgresEloDatabase:
    """
    Get or create the singleton PostgreSQL ELO database.

    Returns:
        PostgresEloDatabase instance

    Raises:
        RuntimeError: If PostgreSQL is not available
    """
    global _postgres_elo_db

    if not ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install aragora[postgres] or pip install asyncpg"
        )

    if _postgres_elo_db is None:
        pool = await get_postgres_pool()
        _postgres_elo_db = PostgresEloDatabase(pool)
        await _postgres_elo_db.initialize()

    return _postgres_elo_db
