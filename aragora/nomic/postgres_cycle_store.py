"""PostgreSQL Cycle Learning Store - persists Nomic cycle records for cross-cycle learning.

Provides PostgreSQL-backed storage for NomicCycleRecord with:
- Save/load cycle records
- Query recent cycles
- Topic-based similarity search
- Agent trajectory tracking
- Pattern aggregation
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from aragora.nomic.cycle_record import NomicCycleRecord
from aragora.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


class PostgresCycleLearningStore(PostgresStore):
    """PostgreSQL-backed storage for Nomic cycle records.

    Persists cycle data for cross-cycle learning, enabling:
    - Historical context injection
    - Agent performance trajectory
    - Pattern success tracking
    - Topic similarity queries
    """

    SCHEMA_NAME = "cycle_learning"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS cycles (
            cycle_id TEXT PRIMARY KEY,
            started_at DOUBLE PRECISION NOT NULL,
            completed_at DOUBLE PRECISION,
            duration_seconds DOUBLE PRECISION,
            success BOOLEAN DEFAULT FALSE,
            topics JSONB DEFAULT '[]',
            data JSONB NOT NULL,
            search_vector TSVECTOR
        );

        CREATE INDEX IF NOT EXISTS idx_cycles_started
        ON cycles(started_at DESC);

        CREATE INDEX IF NOT EXISTS idx_cycles_success
        ON cycles(success);

        CREATE INDEX IF NOT EXISTS idx_cycles_topics
        ON cycles USING GIN(topics);

        CREATE INDEX IF NOT EXISTS idx_cycles_search
        ON cycles USING GIN(search_vector);

        -- Function to update search vector
        CREATE OR REPLACE FUNCTION update_cycle_search_vector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                to_tsvector('english', COALESCE(
                    (SELECT string_agg(topic, ' ')
                     FROM jsonb_array_elements_text(NEW.topics) AS topic),
                    ''
                ));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Trigger to auto-update search vector
        DROP TRIGGER IF EXISTS cycle_search_vector_trigger ON cycles;
        CREATE TRIGGER cycle_search_vector_trigger
            BEFORE INSERT OR UPDATE ON cycles
            FOR EACH ROW
            EXECUTE FUNCTION update_cycle_search_vector();
    """

    # =========================================================================
    # Sync wrappers for compatibility
    # =========================================================================

    def save_cycle(self, record: NomicCycleRecord) -> None:
        """Save a cycle record (sync wrapper)."""
        asyncio.get_event_loop().run_until_complete(self.save_cycle_async(record))

    def load_cycle(self, cycle_id: str) -> Optional[NomicCycleRecord]:
        """Load a specific cycle by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.load_cycle_async(cycle_id))

    def get_recent_cycles(self, n: int = 10) -> list[NomicCycleRecord]:
        """Get the N most recent cycles (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_recent_cycles_async(n))

    def get_successful_cycles(self, n: int = 10) -> list[NomicCycleRecord]:
        """Get the N most recent successful cycles (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_successful_cycles_async(n))

    def query_by_topic(self, topic: str, limit: int = 10) -> list[NomicCycleRecord]:
        """Find cycles that addressed similar topics (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.query_by_topic_async(topic, limit))

    def get_agent_trajectory(self, agent_name: str, n: int = 20) -> list[dict[str, Any]]:
        """Get performance trajectory for an agent (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_agent_trajectory_async(agent_name, n)
        )

    def get_pattern_statistics(self) -> dict[str, dict[str, Any]]:
        """Aggregate pattern success statistics (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_pattern_statistics_async())

    def get_surprise_summary(self, n: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Get summary of surprise events (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_surprise_summary_async(n))

    def cleanup_old_cycles(self, keep_count: int = 100) -> int:
        """Remove old cycles (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.cleanup_old_cycles_async(keep_count)
        )

    def get_cycle_count(self) -> int:
        """Get total number of stored cycles (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_cycle_count_async())

    # =========================================================================
    # Async implementations
    # =========================================================================

    async def save_cycle_async(self, record: NomicCycleRecord) -> None:
        """Save a cycle record asynchronously."""
        async with self.connection() as conn:
            topics_json = json.dumps(record.topics_debated)
            data_json = json.dumps(record.to_dict())

            await conn.execute(
                """
                INSERT INTO cycles
                (cycle_id, started_at, completed_at, duration_seconds, success, topics, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (cycle_id) DO UPDATE SET
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    duration_seconds = EXCLUDED.duration_seconds,
                    success = EXCLUDED.success,
                    topics = EXCLUDED.topics,
                    data = EXCLUDED.data
                """,
                record.cycle_id,
                record.started_at,
                record.completed_at,
                record.duration_seconds,
                record.success,
                topics_json,
                data_json,
            )

        logger.debug(f"cycle_saved cycle_id={record.cycle_id} success={record.success}")

    async def load_cycle_async(self, cycle_id: str) -> Optional[NomicCycleRecord]:
        """Load a specific cycle by ID asynchronously."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM cycles WHERE cycle_id = $1",
                cycle_id,
            )
            if row:
                data = row["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                return NomicCycleRecord.from_dict(data)
            return None

    async def get_recent_cycles_async(self, n: int = 10) -> list[NomicCycleRecord]:
        """Get the N most recent cycles asynchronously."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                "SELECT data FROM cycles ORDER BY started_at DESC LIMIT $1",
                n,
            )
            results = []
            for row in rows:
                data = row["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                results.append(NomicCycleRecord.from_dict(data))
            return results

    async def get_successful_cycles_async(self, n: int = 10) -> list[NomicCycleRecord]:
        """Get the N most recent successful cycles asynchronously."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM cycles
                WHERE success = TRUE
                ORDER BY started_at DESC
                LIMIT $1
                """,
                n,
            )
            results = []
            for row in rows:
                data = row["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                results.append(NomicCycleRecord.from_dict(data))
            return results

    async def query_by_topic_async(self, topic: str, limit: int = 10) -> list[NomicCycleRecord]:
        """Find cycles that addressed similar topics asynchronously.

        Uses PostgreSQL full-text search for efficient topic matching.
        """
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM cycles
                WHERE search_vector @@ plainto_tsquery('english', $1)
                ORDER BY started_at DESC
                LIMIT $2
                """,
                topic,
                limit,
            )
            results = []
            for row in rows:
                data = row["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                results.append(NomicCycleRecord.from_dict(data))
            return results

    async def get_agent_trajectory_async(
        self, agent_name: str, n: int = 20
    ) -> list[dict[str, Any]]:
        """Get performance trajectory for an agent across cycles asynchronously."""
        cycles = await self.get_recent_cycles_async(n)
        trajectory: list[dict[str, Any]] = []

        for cycle in cycles:
            if agent_name in cycle.agent_contributions:
                contrib = cycle.agent_contributions[agent_name]
                trajectory.append(
                    {
                        "cycle_id": cycle.cycle_id,
                        "timestamp": cycle.started_at,
                        "proposals_made": contrib.proposals_made,
                        "proposals_accepted": contrib.proposals_accepted,
                        "acceptance_rate": (
                            contrib.proposals_accepted / contrib.proposals_made
                            if contrib.proposals_made > 0
                            else 0.0
                        ),
                        "critiques_given": contrib.critiques_given,
                        "critiques_valuable": contrib.critiques_valuable,
                        "quality_score": contrib.quality_score,
                        "cycle_success": cycle.success,
                    }
                )

        return trajectory

    async def get_pattern_statistics_async(self) -> dict[str, dict[str, Any]]:
        """Aggregate pattern success statistics across all cycles asynchronously."""
        cycles = await self.get_recent_cycles_async(100)
        stats: dict[str, dict[str, Any]] = {}

        for cycle in cycles:
            for reinforcement in cycle.pattern_reinforcements:
                pattern = reinforcement.pattern_type
                if pattern not in stats:
                    stats[pattern] = {
                        "success_count": 0,
                        "failure_count": 0,
                        "total_confidence": 0.0,
                        "examples": [],
                    }

                if reinforcement.success:
                    stats[pattern]["success_count"] += 1
                else:
                    stats[pattern]["failure_count"] += 1

                stats[pattern]["total_confidence"] += reinforcement.confidence

                # Keep a few examples
                if len(stats[pattern]["examples"]) < 3:
                    stats[pattern]["examples"].append(reinforcement.description)

        # Calculate success rates
        for pattern, data in stats.items():
            total = data["success_count"] + data["failure_count"]
            data["success_rate"] = data["success_count"] / total if total > 0 else 0.0
            data["avg_confidence"] = data["total_confidence"] / total if total > 0 else 0.0

        return stats

    async def get_surprise_summary_async(self, n: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Get summary of surprise events grouped by phase asynchronously."""
        cycles = await self.get_recent_cycles_async(n)
        summary: dict[str, list[dict[str, Any]]] = {}

        for cycle in cycles:
            for surprise in cycle.surprise_events:
                if surprise.phase not in summary:
                    summary[surprise.phase] = []

                summary[surprise.phase].append(
                    {
                        "cycle_id": cycle.cycle_id,
                        "description": surprise.description,
                        "expected": surprise.expected,
                        "actual": surprise.actual,
                        "impact": surprise.impact,
                    }
                )

        return summary

    async def cleanup_old_cycles_async(self, keep_count: int = 100) -> int:
        """Remove old cycles, keeping the most recent ones asynchronously."""
        async with self.connection() as conn:
            # Find the cutoff
            row = await conn.fetchrow(
                """
                SELECT started_at FROM cycles
                ORDER BY started_at DESC
                LIMIT 1 OFFSET $1
                """,
                keep_count - 1,
            )
            if not row:
                return 0

            cutoff = row["started_at"]

            # Delete older cycles
            result = await conn.execute(
                "DELETE FROM cycles WHERE started_at < $1",
                cutoff,
            )

            # Parse result like "DELETE 5" to get count
            try:
                deleted = int(result.split()[-1])
            except (IndexError, ValueError):
                deleted = 0

            if deleted > 0:
                logger.info(f"cycles_cleaned deleted={deleted} retained={keep_count}")

            return deleted

    async def get_cycle_count_async(self) -> int:
        """Get total number of stored cycles asynchronously."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM cycles")
            return row["count"] if row else 0

    def close(self) -> None:
        """No-op for pool-based store (pool managed externally)."""
        pass


__all__ = ["PostgresCycleLearningStore"]
