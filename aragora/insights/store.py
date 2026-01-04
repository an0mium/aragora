"""
Insight Store - SQLite persistence for debate insights.

Stores and retrieves insights with:
- Full-text search capabilities
- Aggregation queries
- Pattern clustering
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from aragora.insights.extractor import Insight, InsightType, DebateInsights


def _escape_like_pattern(value: str) -> str:
    """Escape special characters in SQL LIKE patterns to prevent injection.

    SQLite LIKE uses % and _ as wildcards. This escapes them using backslash.
    """
    # Escape backslash first, then LIKE special characters
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


class InsightStore:
    """
    SQLite-based storage for debate insights.

    Enables:
    - Storing insights from completed debates
    - Searching insights by type, agent, or content
    - Aggregating patterns across multiple debates
    - Tracking agent performance history

    Usage:
        store = InsightStore("insights.db")
        await store.store_debate_insights(insights)

        # Search
        results = await store.search("convergence", limit=10)

        # Aggregate
        patterns = await store.get_common_patterns(min_occurrences=3)
    """

    def __init__(self, db_path: str = "aragora_insights.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            # Insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    confidence REAL DEFAULT 0.5,
                    debate_id TEXT NOT NULL,
                    agents_involved TEXT,  -- JSON array
                    evidence TEXT,  -- JSON array
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON object
                )
            """)

            # Debate summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debate_summaries (
                    debate_id TEXT PRIMARY KEY,
                    task TEXT,
                    consensus_reached INTEGER,
                    duration_seconds REAL,
                    total_insights INTEGER,
                    key_takeaway TEXT,
                    agent_performances TEXT,  -- JSON array
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Agent performance history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    debate_id TEXT NOT NULL,
                    proposals_made INTEGER DEFAULT 0,
                    critiques_given INTEGER DEFAULT 0,
                    critiques_received INTEGER DEFAULT 0,
                    proposal_accepted INTEGER DEFAULT 0,
                    vote_aligned INTEGER DEFAULT 0,
                    avg_critique_severity REAL DEFAULT 0.0,
                    contribution_score REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debate_summaries(debate_id)
                )
            """)

            # Pattern clusters (aggregated from pattern insights)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    pattern_text TEXT NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    avg_severity REAL DEFAULT 0.5,
                    debate_ids TEXT,  -- JSON array
                    first_seen TEXT,
                    last_seen TEXT,
                    UNIQUE(category, pattern_text)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_debate ON insights(debate_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_perf_name ON agent_performance_history(agent_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_category ON pattern_clusters(category)")

            conn.commit()

    async def store_debate_insights(self, insights: DebateInsights) -> int:
        """
        Store all insights from a debate.

        Returns:
            Number of insights stored
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            # Store debate summary
            cursor.execute(
                """
                INSERT OR REPLACE INTO debate_summaries
                (debate_id, task, consensus_reached, duration_seconds,
                 total_insights, key_takeaway, agent_performances)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insights.debate_id,
                    insights.task,
                    1 if insights.consensus_reached else 0,
                    insights.duration_seconds,
                    insights.total_insights,
                    insights.key_takeaway,
                    json.dumps([{
                        "agent": p.agent_name,
                        "proposals": p.proposals_made,
                        "accepted": p.proposal_accepted,
                        "score": p.contribution_score,
                    } for p in insights.agent_performances]),
                )
            )

            # Store each insight
            stored_count = 0
            for insight in insights.all_insights():
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO insights
                        (id, type, title, description, confidence, debate_id,
                         agents_involved, evidence, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
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
                    )
                    stored_count += 1
                except Exception as e:
                    print(f"Error storing insight {insight.id}: {e}")

            # Store agent performances
            for perf in insights.agent_performances:
                cursor.execute(
                    """
                    INSERT INTO agent_performance_history
                    (agent_name, debate_id, proposals_made, critiques_given,
                     critiques_received, proposal_accepted, vote_aligned,
                     avg_critique_severity, contribution_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        perf.agent_name,
                        insights.debate_id,
                        perf.proposals_made,
                        perf.critiques_given,
                        perf.critiques_received,
                        1 if perf.proposal_accepted else 0,
                        1 if perf.vote_aligned_with_consensus else 0,
                        perf.average_critique_severity,
                        perf.contribution_score,
                    )
                )

            # Update pattern clusters
            for insight in insights.pattern_insights:
                category = insight.metadata.get('category', 'general')
                pattern_text = insight.title

                cursor.execute(
                    """
                    INSERT INTO pattern_clusters (category, pattern_text, occurrence_count,
                        avg_severity, debate_ids, first_seen, last_seen)
                    VALUES (?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(category, pattern_text) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        avg_severity = (avg_severity * occurrence_count + ?) / (occurrence_count + 1),
                        debate_ids = json_insert(debate_ids, '$[#]', ?),
                        last_seen = ?
                    """,
                    (
                        category,
                        pattern_text,
                        insight.metadata.get('avg_severity', 0.5),
                        json.dumps([insights.debate_id]),
                        insight.created_at,
                        insight.created_at,
                        insight.metadata.get('avg_severity', 0.5),
                        insights.debate_id,
                        insight.created_at,
                    )
                )

            conn.commit()

        return stored_count

    async def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Retrieve a specific insight by ID."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM insights WHERE id = ?", (insight_id,))
            row = cursor.fetchone()

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
        """
        Search insights by query, type, or agent.

        Args:
            query: Text to search in title/description
            insight_type: Filter by insight type
            agent: Filter by agent involvement
            limit: Maximum results

        Returns:
            List of matching insights
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM insights WHERE 1=1"
            params = []

            if query:
                # Escape LIKE special characters to prevent SQL injection
                escaped_query = _escape_like_pattern(query)
                sql += " AND (title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')"
                params.extend([f"%{escaped_query}%", f"%{escaped_query}%"])

            if insight_type:
                sql += " AND type = ?"
                params.append(insight_type.value)

            if agent:
                # Escape LIKE special characters to prevent SQL injection
                escaped_agent = _escape_like_pattern(agent)
                sql += " AND agents_involved LIKE ? ESCAPE '\\'"
                params.append(f'%"{escaped_agent}"%')

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_insight(row) for row in rows]

    async def get_common_patterns(
        self,
        min_occurrences: int = 2,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get commonly occurring patterns across debates.

        Returns:
            List of pattern dictionaries with occurrence counts
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT category, pattern_text, occurrence_count, avg_severity, last_seen
                FROM pattern_clusters
                WHERE occurrence_count >= ?
            """
            params = [min_occurrences]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY occurrence_count DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [
            {
                "category": row[0],
                "pattern": row[1],
                "occurrences": row[2],
                "avg_severity": row[3],
                "last_seen": row[4],
            }
            for row in rows
        ]

    async def get_agent_stats(self, agent_name: str) -> dict:
        """
        Get aggregate statistics for an agent.

        Returns:
            Dictionary with performance metrics
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    COUNT(*) as debate_count,
                    SUM(proposals_made) as total_proposals,
                    SUM(proposal_accepted) as proposals_accepted,
                    SUM(critiques_given) as total_critiques,
                    AVG(contribution_score) as avg_contribution,
                    AVG(avg_critique_severity) as avg_severity_received
                FROM agent_performance_history
                WHERE agent_name = ?
                """,
                (agent_name,)
            )
            row = cursor.fetchone()

        if not row or row[0] == 0:
            return {"agent": agent_name, "debate_count": 0}

        return {
            "agent": agent_name,
            "debate_count": row[0],
            "total_proposals": row[1] or 0,
            "proposals_accepted": row[2] or 0,
            "acceptance_rate": (row[2] or 0) / max(1, row[1] or 1),
            "total_critiques": row[3] or 0,
            "avg_contribution": row[4] or 0.5,
            "avg_severity_received": row[5] or 0.0,
        }

    async def get_all_agent_rankings(self) -> list[dict]:
        """
        Get ranked list of all agents by performance.

        Returns:
            List of agent stats ordered by contribution score
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    agent_name,
                    COUNT(*) as debate_count,
                    SUM(proposal_accepted) as proposals_accepted,
                    AVG(contribution_score) as avg_contribution
                FROM agent_performance_history
                GROUP BY agent_name
                HAVING debate_count >= 1
                ORDER BY avg_contribution DESC
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "agent": row[0],
                "debate_count": row[1],
                "proposals_accepted": row[2] or 0,
                "avg_contribution": row[3] or 0.5,
            }
            for row in rows
        ]

    async def get_recent_insights(self, limit: int = 20) -> list[Insight]:
        """Get most recent insights across all debates."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()

        return [self._row_to_insight(row) for row in rows]

    async def get_stats(self) -> dict:
        """Get overall statistics about stored insights."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) FROM insights")
            stats["total_insights"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM debate_summaries")
            stats["total_debates"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM debate_summaries WHERE consensus_reached = 1")
            stats["consensus_debates"] = cursor.fetchone()[0]

            cursor.execute("SELECT type, COUNT(*) FROM insights GROUP BY type")
            stats["insights_by_type"] = dict(cursor.fetchall())

            cursor.execute("SELECT COUNT(DISTINCT agent_name) FROM agent_performance_history")
            stats["unique_agents"] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(total_insights) FROM debate_summaries")
            avg = cursor.fetchone()[0]
            stats["avg_insights_per_debate"] = avg if avg else 0

        return stats

    def _row_to_insight(self, row) -> Insight:
        """Convert a database row to an Insight object."""
        return Insight(
            id=row[0],
            type=InsightType(row[1]),
            title=row[2],
            description=row[3] or "",
            confidence=row[4] or 0.5,
            debate_id=row[5],
            agents_involved=json.loads(row[6]) if row[6] else [],
            evidence=json.loads(row[7]) if row[7] else [],
            created_at=row[8] or datetime.now().isoformat(),
            metadata=json.loads(row[9]) if row[9] else {},
        )
