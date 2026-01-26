"""
PostgreSQL Insight Store - PostgreSQL persistence for debate insights.

Provides async storage for insights with:
- Full-text search capabilities using tsvector
- Aggregation queries
- Pattern clustering
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from aragora.insights.extractor import DebateInsights, Insight, InsightType
from aragora.storage.postgres_store import PostgresStore
from aragora.utils.json_helpers import safe_json_loads

if TYPE_CHECKING:
    from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

logger = logging.getLogger(__name__)


class PostgresInsightStore(PostgresStore):
    """PostgreSQL-backed storage for debate insights.

    Enables:
    - Storing insights from completed debates
    - Searching insights by type, agent, or content
    - Aggregating patterns across multiple debates
    - Tracking agent performance history
    """

    SCHEMA_NAME = "insight_store"
    SCHEMA_VERSION = 2

    INITIAL_SCHEMA = """
        -- Insights table with tsvector for FTS
        CREATE TABLE IF NOT EXISTS insights (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            confidence REAL DEFAULT 0.5,
            debate_id TEXT NOT NULL,
            agents_involved JSONB DEFAULT '[]',
            evidence JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}',
            search_vector TSVECTOR
        );

        -- Debate summaries table
        CREATE TABLE IF NOT EXISTS debate_summaries (
            debate_id TEXT PRIMARY KEY,
            task TEXT,
            consensus_reached BOOLEAN DEFAULT FALSE,
            duration_seconds REAL,
            total_insights INTEGER,
            key_takeaway TEXT,
            agent_performances JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Agent performance history
        CREATE TABLE IF NOT EXISTS agent_performance_history (
            id SERIAL PRIMARY KEY,
            agent_name TEXT NOT NULL,
            debate_id TEXT NOT NULL REFERENCES debate_summaries(debate_id) ON DELETE CASCADE,
            proposals_made INTEGER DEFAULT 0,
            critiques_given INTEGER DEFAULT 0,
            critiques_received INTEGER DEFAULT 0,
            proposal_accepted BOOLEAN DEFAULT FALSE,
            vote_aligned BOOLEAN DEFAULT FALSE,
            avg_critique_severity REAL DEFAULT 0.0,
            contribution_score REAL DEFAULT 0.5,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Pattern clusters (aggregated from pattern insights)
        CREATE TABLE IF NOT EXISTS pattern_clusters (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL,
            pattern_text TEXT NOT NULL,
            occurrence_count INTEGER DEFAULT 1,
            avg_severity REAL DEFAULT 0.5,
            debate_ids JSONB DEFAULT '[]',
            first_seen TIMESTAMPTZ,
            last_seen TIMESTAMPTZ,
            UNIQUE(category, pattern_text)
        );

        -- Wisdom submissions table
        CREATE TABLE IF NOT EXISTS wisdom_submissions (
            id SERIAL PRIMARY KEY,
            loop_id TEXT NOT NULL,
            text TEXT NOT NULL,
            submitter_id TEXT DEFAULT 'anonymous',
            context_tags JSONB DEFAULT '[]',
            used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Insight usage tracking table
        CREATE TABLE IF NOT EXISTS insight_usage (
            id SERIAL PRIMARY KEY,
            insight_id TEXT NOT NULL,
            debate_id TEXT NOT NULL,
            was_successful BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(insight_id, debate_id)
        );

        -- Indexes for insights
        CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type);
        CREATE INDEX IF NOT EXISTS idx_insights_debate ON insights(debate_id);
        CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_insights_type_confidence ON insights(type, confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_insights_search ON insights USING GIN(search_vector);

        -- Indexes for agent performance
        CREATE INDEX IF NOT EXISTS idx_agent_perf_name ON agent_performance_history(agent_name);
        CREATE INDEX IF NOT EXISTS idx_agent_perf_debate ON agent_performance_history(debate_id);
        CREATE INDEX IF NOT EXISTS idx_agent_perf_name_debate ON agent_performance_history(agent_name, debate_id);

        -- Indexes for pattern clusters
        CREATE INDEX IF NOT EXISTS idx_pattern_category ON pattern_clusters(category);
        CREATE INDEX IF NOT EXISTS idx_pattern_last_seen ON pattern_clusters(last_seen DESC);

        -- Indexes for debate summaries
        CREATE INDEX IF NOT EXISTS idx_debate_summaries_created ON debate_summaries(created_at DESC);

        -- Indexes for wisdom submissions
        CREATE INDEX IF NOT EXISTS idx_wisdom_loop_id ON wisdom_submissions(loop_id, used);

        -- Indexes for insight usage
        CREATE INDEX IF NOT EXISTS idx_insight_usage_insight ON insight_usage(insight_id);

        -- Function to update search vector for insights
        CREATE OR REPLACE FUNCTION update_insight_search_vector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Trigger to auto-update search vector
        DROP TRIGGER IF EXISTS insight_search_vector_trigger ON insights;
        CREATE TRIGGER insight_search_vector_trigger
            BEFORE INSERT OR UPDATE ON insights
            FOR EACH ROW
            EXECUTE FUNCTION update_insight_search_vector();
    """

    def __init__(
        self,
        pool: Any,
        km_adapter: Optional["InsightsAdapter"] = None,
        km_min_confidence: float = 0.7,
    ):
        """Initialize the insight store.

        Args:
            pool: asyncpg connection pool
            km_adapter: Optional Knowledge Mound adapter for sync
            km_min_confidence: Minimum confidence for KM sync
        """
        super().__init__(pool)
        self._km_adapter = km_adapter
        self._km_min_confidence = km_min_confidence

    def set_km_adapter(self, adapter: "InsightsAdapter") -> None:
        """Set the Knowledge Mound adapter for bidirectional sync."""
        self._km_adapter = adapter

    # =========================================================================
    # Sync wrappers for compatibility
    # =========================================================================

    def store_debate_insights_sync(self, insights: DebateInsights) -> int:
        """Store debate insights (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.store_debate_insights(insights))

    def get_insight_sync(self, insight_id: str) -> Optional[Insight]:
        """Get insight by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_insight(insight_id))

    def search_sync(
        self,
        query: str = "",
        insight_type: Optional[InsightType] = None,
        agent: Optional[str] = None,
        limit: int = 20,
    ) -> list[Insight]:
        """Search insights (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.search(query, insight_type, agent, limit)
        )

    # =========================================================================
    # Core async methods
    # =========================================================================

    async def store_debate_insights(self, insights: DebateInsights) -> int:
        """Store all insights from a debate.

        Args:
            insights: DebateInsights object to store

        Returns:
            Number of insights stored
        """
        async with self.transaction() as conn:
            # Store debate summary
            await conn.execute(
                """
                INSERT INTO debate_summaries
                (debate_id, task, consensus_reached, duration_seconds,
                 total_insights, key_takeaway, agent_performances, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (debate_id) DO UPDATE SET
                    task = EXCLUDED.task,
                    consensus_reached = EXCLUDED.consensus_reached,
                    duration_seconds = EXCLUDED.duration_seconds,
                    total_insights = EXCLUDED.total_insights,
                    key_takeaway = EXCLUDED.key_takeaway,
                    agent_performances = EXCLUDED.agent_performances
                """,
                insights.debate_id,
                insights.task,
                insights.consensus_reached,
                insights.duration_seconds,
                insights.total_insights,
                insights.key_takeaway,
                json.dumps(
                    [
                        {
                            "agent": p.agent_name,
                            "proposals": p.proposals_made,
                            "accepted": p.proposal_accepted,
                            "score": p.contribution_score,
                        }
                        for p in insights.agent_performances
                    ]
                ),
            )

            # Store all insights in batch
            all_insights = list(insights.all_insights())
            stored_count = 0

            if all_insights:
                try:
                    await conn.executemany(
                        """
                        INSERT INTO insights
                        (id, type, title, description, confidence, debate_id,
                         agents_involved, evidence, created_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO UPDATE SET
                            type = EXCLUDED.type,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            confidence = EXCLUDED.confidence,
                            agents_involved = EXCLUDED.agents_involved,
                            evidence = EXCLUDED.evidence,
                            metadata = EXCLUDED.metadata
                        """,
                        [
                            (
                                insight.id,
                                insight.type.value,
                                insight.title,
                                insight.description,
                                insight.confidence,
                                insight.debate_id,
                                json.dumps(insight.agents_involved),
                                json.dumps(insight.evidence),
                                insight.created_at,
                                json.dumps(insight.metadata),
                            )
                            for insight in all_insights
                        ],
                    )
                    stored_count = len(all_insights)
                except Exception as e:
                    logger.error(f"Error batch storing insights: {e}")

            # Store agent performances in batch
            if insights.agent_performances:
                await conn.executemany(
                    """
                    INSERT INTO agent_performance_history
                    (agent_name, debate_id, proposals_made, critiques_given,
                     critiques_received, proposal_accepted, vote_aligned,
                     avg_critique_severity, contribution_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    [
                        (
                            perf.agent_name,
                            insights.debate_id,
                            perf.proposals_made,
                            perf.critiques_given,
                            perf.critiques_received,
                            perf.proposal_accepted,
                            perf.vote_aligned_with_consensus,
                            perf.average_critique_severity,
                            perf.contribution_score,
                        )
                        for perf in insights.agent_performances
                    ],
                )

            # Update pattern clusters
            if insights.pattern_insights:
                for insight in insights.pattern_insights:
                    now = datetime.now(timezone.utc)
                    category = insight.metadata.get("category", "general")
                    avg_severity = insight.metadata.get("avg_severity", 0.5)

                    await conn.execute(
                        """
                        INSERT INTO pattern_clusters
                        (category, pattern_text, occurrence_count, avg_severity,
                         debate_ids, first_seen, last_seen)
                        VALUES ($1, $2, 1, $3, $4, $5, $5)
                        ON CONFLICT (category, pattern_text) DO UPDATE SET
                            occurrence_count = pattern_clusters.occurrence_count + 1,
                            avg_severity = (pattern_clusters.avg_severity *
                                pattern_clusters.occurrence_count + $3) /
                                (pattern_clusters.occurrence_count + 1),
                            debate_ids = pattern_clusters.debate_ids || $6::jsonb,
                            last_seen = $5
                        """,
                        category,
                        insight.title,
                        avg_severity,
                        json.dumps([insights.debate_id]),
                        now,
                        json.dumps([insights.debate_id]),
                    )

        # Sync high-confidence insights to Knowledge Mound
        if self._km_adapter:
            for insight in all_insights:
                if insight.confidence >= self._km_min_confidence:
                    try:
                        self._km_adapter.store_insight(insight)
                        logger.debug(f"Insight synced to Knowledge Mound: {insight.id}")
                    except Exception as e:
                        logger.warning(f"Failed to sync insight to KM: {e}")

        return stored_count

    async def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Retrieve a specific insight by ID."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, type, title, description, confidence,
                       debate_id, agents_involved, evidence, created_at, metadata
                FROM insights WHERE id = $1
                """,
                insight_id,
            )
            if not row:
                return None
            return self._row_to_insight(row)

    async def search(
        self,
        query: str = "",
        insight_type: Optional[InsightType] = None,
        agent: Optional[str] = None,
        limit: int = 20,
    ) -> list[Insight]:
        """Search insights by query, type, or agent.

        Args:
            query: Text to search in title/description
            insight_type: Filter by insight type
            agent: Filter by agent involvement
            limit: Maximum results

        Returns:
            List of matching insights
        """
        async with self.connection() as conn:
            sql = """
                SELECT id, type, title, description, confidence,
                       debate_id, agents_involved, evidence, created_at, metadata
                FROM insights WHERE 1=1
            """
            params: list[Any] = []
            param_num = 1

            if query:
                sql += f" AND search_vector @@ plainto_tsquery('english', ${param_num})"
                params.append(query)
                param_num += 1

            if insight_type:
                sql += f" AND type = ${param_num}"
                params.append(insight_type.value)
                param_num += 1

            if agent:
                # JSONB containment check for agent in agents_involved array
                sql += f" AND agents_involved @> ${param_num}::jsonb"
                params.append(json.dumps([agent]))
                param_num += 1

            sql += f" ORDER BY created_at DESC LIMIT ${param_num}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)
            return [self._row_to_insight(row) for row in rows]

    async def get_common_patterns(
        self,
        min_occurrences: int = 2,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get commonly occurring patterns across debates.

        Returns:
            List of pattern dictionaries with occurrence counts
        """
        async with self.connection() as conn:
            sql = """
                SELECT category, pattern_text, occurrence_count, avg_severity, last_seen
                FROM pattern_clusters
                WHERE occurrence_count >= $1
            """
            params: list[Any] = [min_occurrences]
            param_num = 2

            if category:
                sql += f" AND category = ${param_num}"
                params.append(category)
                param_num += 1

            sql += f" ORDER BY occurrence_count DESC LIMIT ${param_num}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)
            return [
                {
                    "category": row["category"],
                    "pattern": row["pattern_text"],
                    "occurrences": row["occurrence_count"],
                    "avg_severity": row["avg_severity"],
                    "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
                }
                for row in rows
            ]

    async def get_agent_stats(self, agent_name: str) -> dict:
        """Get aggregate statistics for an agent."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as debate_count,
                    COALESCE(SUM(proposals_made), 0) as total_proposals,
                    COALESCE(SUM(CASE WHEN proposal_accepted THEN 1 ELSE 0 END), 0) as proposals_accepted,
                    COALESCE(SUM(critiques_given), 0) as total_critiques,
                    COALESCE(AVG(contribution_score), 0.5) as avg_contribution,
                    COALESCE(AVG(avg_critique_severity), 0.0) as avg_severity_received
                FROM agent_performance_history
                WHERE agent_name = $1
                """,
                agent_name,
            )

            if not row or row["debate_count"] == 0:
                return {"agent": agent_name, "debate_count": 0}

            return {
                "agent": agent_name,
                "debate_count": row["debate_count"],
                "total_proposals": row["total_proposals"],
                "proposals_accepted": row["proposals_accepted"],
                "acceptance_rate": row["proposals_accepted"] / max(1, row["total_proposals"]),
                "total_critiques": row["total_critiques"],
                "avg_contribution": float(row["avg_contribution"]),
                "avg_severity_received": float(row["avg_severity_received"]),
            }

    async def get_all_agent_rankings(self) -> list[dict]:
        """Get ranked list of all agents by performance."""
        async with self.connection() as conn:
            rows = await conn.fetch("""
                SELECT
                    agent_name,
                    COUNT(*) as debate_count,
                    COALESCE(SUM(CASE WHEN proposal_accepted THEN 1 ELSE 0 END), 0) as proposals_accepted,
                    COALESCE(AVG(contribution_score), 0.5) as avg_contribution
                FROM agent_performance_history
                GROUP BY agent_name
                HAVING COUNT(*) >= 1
                ORDER BY avg_contribution DESC
                """)
            return [
                {
                    "agent": row["agent_name"],
                    "debate_count": row["debate_count"],
                    "proposals_accepted": row["proposals_accepted"],
                    "avg_contribution": float(row["avg_contribution"]),
                }
                for row in rows
            ]

    async def get_recent_insights(self, limit: int = 20) -> list[Insight]:
        """Get most recent insights across all debates."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, type, title, description, confidence,
                       debate_id, agents_involved, evidence, created_at, metadata
                FROM insights
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [self._row_to_insight(row) for row in rows]

    async def get_stats(self) -> dict:
        """Get overall statistics about stored insights."""
        async with self.connection() as conn:
            stats: dict[str, Any] = {}

            row = await conn.fetchrow("SELECT COUNT(*) as count FROM insights")
            stats["total_insights"] = row["count"] if row else 0

            row = await conn.fetchrow("SELECT COUNT(*) as count FROM debate_summaries")
            stats["total_debates"] = row["count"] if row else 0

            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM debate_summaries WHERE consensus_reached = TRUE"
            )
            stats["consensus_debates"] = row["count"] if row else 0

            rows = await conn.fetch("SELECT type, COUNT(*) as count FROM insights GROUP BY type")
            stats["insights_by_type"] = {row["type"]: row["count"] for row in rows}

            row = await conn.fetchrow(
                "SELECT COUNT(DISTINCT agent_name) as count FROM agent_performance_history"
            )
            stats["unique_agents"] = row["count"] if row else 0

            row = await conn.fetchrow("SELECT AVG(total_insights) as avg FROM debate_summaries")
            stats["avg_insights_per_debate"] = float(row["avg"]) if row and row["avg"] else 0

            return stats

    # =========================================================================
    # Audience Wisdom Injection
    # =========================================================================

    async def add_wisdom_submission(self, loop_id: str, wisdom_data: dict) -> int:
        """Add audience wisdom with proper loop association.

        Args:
            loop_id: The debate/loop ID this wisdom is for
            wisdom_data: Dict with 'text', optional 'submitter_id', 'context_tags'

        Returns:
            The ID of the inserted wisdom submission
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO wisdom_submissions (loop_id, text, submitter_id, context_tags)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                loop_id,
                wisdom_data.get("text", "")[:280],  # Character limit
                wisdom_data.get("submitter_id", "anonymous"),
                json.dumps(wisdom_data.get("context_tags", [])),
            )
            wisdom_id = row["id"]
            logger.info(f"[wisdom] Added submission {wisdom_id} for loop {loop_id}")
            return wisdom_id

    async def get_relevant_wisdom(self, loop_id: str, limit: int = 3) -> list[dict]:
        """Retrieve unused wisdom for current loop.

        Args:
            loop_id: The debate/loop ID to get wisdom for
            limit: Maximum number of wisdom entries to return

        Returns:
            List of wisdom dicts
        """
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, text, submitter_id, context_tags, created_at
                FROM wisdom_submissions
                WHERE loop_id = $1 AND used = FALSE
                ORDER BY created_at DESC
                LIMIT $2
                """,
                loop_id,
                limit,
            )
            return [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "submitter_id": row["submitter_id"],
                    "context_tags": (
                        row["context_tags"]
                        if isinstance(row["context_tags"], list)
                        else safe_json_loads(row["context_tags"], [])
                    ),
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]

    async def mark_wisdom_used(self, wisdom_id: int) -> None:
        """Mark a wisdom submission as used."""
        async with self.connection() as conn:
            await conn.execute(
                "UPDATE wisdom_submissions SET used = TRUE WHERE id = $1",
                wisdom_id,
            )

    # =========================================================================
    # Insight Application Cycle (B2)
    # =========================================================================

    async def get_relevant_insights(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.7,
        limit: int = 5,
    ) -> list[Insight]:
        """Get high-confidence insights relevant to a domain for injection into debates.

        Args:
            domain: Optional domain to filter by (e.g., "security", "design")
            min_confidence: Minimum confidence threshold (default 0.7)
            limit: Maximum number of insights to return

        Returns:
            List of high-confidence Insight objects
        """
        async with self.connection() as conn:
            sql = """
                SELECT id, type, title, description, confidence,
                       debate_id, agents_involved, evidence, created_at, metadata
                FROM insights
                WHERE confidence >= $1
            """
            params: list[Any] = [min_confidence]
            param_num = 2

            if domain:
                sql += f"""
                    AND (
                        metadata::text ILIKE ${param_num}
                        OR title ILIKE ${param_num}
                        OR description ILIKE ${param_num}
                    )
                """
                params.append(f"%{domain}%")
                param_num += 1

            sql += f" ORDER BY confidence DESC, created_at DESC LIMIT ${param_num}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)
            return [self._row_to_insight(row) for row in rows]

    async def record_insight_usage(
        self,
        insight_id: str,
        debate_id: str,
        was_successful: bool = True,
    ) -> None:
        """Record that an insight was applied to a debate.

        Args:
            insight_id: ID of the insight that was applied
            debate_id: ID of the debate where it was applied
            was_successful: Whether the debate outcome correlated with success
        """
        async with self.transaction() as conn:
            # Record the usage
            await conn.execute(
                """
                INSERT INTO insight_usage (insight_id, debate_id, was_successful, created_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (insight_id, debate_id) DO UPDATE SET
                    was_successful = EXCLUDED.was_successful,
                    created_at = NOW()
                """,
                insight_id,
                debate_id,
                was_successful,
            )

            # Update insight confidence based on usage success
            adjustment = 0.05 if was_successful else -0.03
            await conn.execute(
                """
                UPDATE insights
                SET confidence = LEAST(0.99, GREATEST(0.1, confidence + $1))
                WHERE id = $2
                """,
                adjustment,
                insight_id,
            )

        logger.debug(
            f"[insight] Recorded usage: insight={insight_id} "
            f"debate={debate_id} success={was_successful}"
        )

    async def get_insight_usage_stats(self, insight_id: str) -> dict:
        """Get usage statistics for a specific insight."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_uses,
                    COALESCE(SUM(CASE WHEN was_successful THEN 1 ELSE 0 END), 0) as successful_uses
                FROM insight_usage
                WHERE insight_id = $1
                """,
                insight_id,
            )

            total = row["total_uses"] if row else 0
            successful = row["successful_uses"] if row else 0

            return {
                "insight_id": insight_id,
                "total_uses": total,
                "successful_uses": successful,
                "success_rate": successful / total if total > 0 else 0.0,
            }

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _row_to_insight(self, row: Any) -> Insight:
        """Convert a database row to an Insight object."""
        try:
            insight_type = InsightType(row["type"])
        except ValueError:
            logger.warning(f"Invalid insight type '{row['type']}', defaulting to PATTERN")
            insight_type = InsightType.PATTERN

        # Handle JSONB columns
        agents_involved = row["agents_involved"]
        if isinstance(agents_involved, str):
            agents_involved = safe_json_loads(agents_involved, [])
        elif agents_involved is None:
            agents_involved = []

        evidence = row["evidence"]
        if isinstance(evidence, str):
            evidence = safe_json_loads(evidence, [])
        elif evidence is None:
            evidence = []

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = safe_json_loads(metadata, {})
        elif metadata is None:
            metadata = {}

        created_at = row["created_at"]
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        elif created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        return Insight(
            id=row["id"],
            type=insight_type,
            title=row["title"],
            description=row["description"] or "",
            confidence=row["confidence"] or 0.5,
            debate_id=row["debate_id"],
            agents_involved=agents_involved,
            evidence=evidence,
            created_at=created_at,
            metadata=metadata,
        )

    def close(self) -> None:
        """No-op for pool-based store (pool managed externally)."""
        pass
