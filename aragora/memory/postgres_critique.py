"""
PostgreSQL-based critique pattern store for self-improvement.

Provides async PostgreSQL-backed storage with connection pooling for production
deployments that need horizontal scaling and concurrent writes.

This is the PostgreSQL equivalent of CritiqueStore (SQLite-based).
"""

__all__ = [
    "PostgresCritiqueStore",
    "POSTGRES_CRITIQUE_SCHEMA",
    "POSTGRES_CRITIQUE_SCHEMA_VERSION",
    "get_postgres_critique_store",
]

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from aragora.memory.store import AgentReputation, Pattern

logger = logging.getLogger(__name__)

# Optional asyncpg import
try:
    import asyncpg
    from asyncpg import Connection, Pool

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    Pool = Any
    Connection = Any
    ASYNCPG_AVAILABLE = False

# Schema version for migrations
POSTGRES_CRITIQUE_SCHEMA_VERSION = 1

# PostgreSQL-native schema
POSTGRES_CRITIQUE_SCHEMA = """
    -- Debates table
    CREATE TABLE IF NOT EXISTS debates (
        id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        final_answer TEXT,
        consensus_reached BOOLEAN DEFAULT FALSE,
        confidence REAL,
        rounds_used INTEGER,
        duration_seconds REAL,
        grounded_verdict JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Critiques table (includes Titans/MIRAS prediction tracking)
    CREATE TABLE IF NOT EXISTS critiques (
        id SERIAL PRIMARY KEY,
        debate_id TEXT REFERENCES debates(id) ON DELETE CASCADE,
        agent TEXT NOT NULL,
        target_agent TEXT,
        issues JSONB DEFAULT '[]'::jsonb,
        suggestions JSONB DEFAULT '[]'::jsonb,
        severity REAL,
        reasoning TEXT,
        led_to_improvement BOOLEAN DEFAULT FALSE,
        expected_usefulness REAL DEFAULT 0.5,
        actual_usefulness REAL,
        prediction_error REAL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Patterns table (includes Titans/MIRAS surprise scoring)
    CREATE TABLE IF NOT EXISTS patterns (
        id TEXT PRIMARY KEY,
        issue_type TEXT NOT NULL,
        issue_text TEXT NOT NULL,
        suggestion_text TEXT,
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        avg_severity REAL DEFAULT 0.5,
        surprise_score REAL DEFAULT 0.0,
        base_rate REAL DEFAULT 0.5,
        avg_prediction_error REAL DEFAULT 0.0,
        prediction_count INTEGER DEFAULT 0,
        example_task TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Pattern embeddings for semantic search (optional, for future)
    CREATE TABLE IF NOT EXISTS pattern_embeddings (
        pattern_id TEXT PRIMARY KEY REFERENCES patterns(id) ON DELETE CASCADE,
        embedding BYTEA
    );

    -- Agent reputation tracking (includes Titans/MIRAS calibration)
    CREATE TABLE IF NOT EXISTS agent_reputation (
        agent_name TEXT PRIMARY KEY,
        proposals_made INTEGER DEFAULT 0,
        proposals_accepted INTEGER DEFAULT 0,
        critiques_given INTEGER DEFAULT 0,
        critiques_valuable INTEGER DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        total_predictions INTEGER DEFAULT 0,
        total_prediction_error REAL DEFAULT 0.0,
        calibration_score REAL DEFAULT 0.5
    );

    -- Patterns archive table for adaptive forgetting
    CREATE TABLE IF NOT EXISTS patterns_archive (
        id TEXT,
        issue_type TEXT,
        issue_text TEXT,
        suggestion_text TEXT,
        success_count INTEGER,
        failure_count INTEGER,
        avg_severity REAL,
        surprise_score REAL,
        example_task TEXT,
        created_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ,
        archived_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_critiques_debate ON critiques(debate_id);
    CREATE INDEX IF NOT EXISTS idx_critiques_agent ON critiques(agent);
    CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(issue_type);
    CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(success_count DESC);
    CREATE INDEX IF NOT EXISTS idx_patterns_type_success ON patterns(issue_type, success_count DESC);
    CREATE INDEX IF NOT EXISTS idx_patterns_success_updated ON patterns(success_count DESC, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_reputation_score ON agent_reputation(proposals_accepted DESC);
    CREATE INDEX IF NOT EXISTS idx_reputation_agent ON agent_reputation(agent_name);
    CREATE INDEX IF NOT EXISTS idx_debates_consensus ON debates(consensus_reached);
    CREATE INDEX IF NOT EXISTS idx_debates_created ON debates(created_at DESC);
"""


class PostgresCritiqueStore:
    """
    PostgreSQL-backed storage for critique patterns.

    Enables self-improvement by:
    1. Storing successful critique -> fix patterns
    2. Retrieving similar patterns for new critiques
    3. Tracking which patterns lead to consensus

    This is the async PostgreSQL equivalent of CritiqueStore.
    """

    SCHEMA_NAME = "critique_store"
    SCHEMA_VERSION = POSTGRES_CRITIQUE_SCHEMA_VERSION

    def __init__(self, pool: "Pool"):
        """
        Initialize the store with a connection pool.

        Args:
            pool: asyncpg connection pool
        """
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required for PostgresCritiqueStore")

        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self.connection() as conn:
            # Create schema version tracking table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """
            )

            # Check current version
            row = await conn.fetchrow(
                "SELECT version FROM _schema_versions WHERE module = $1", self.SCHEMA_NAME
            )
            current_version = row["version"] if row else 0

            if current_version == 0:
                logger.info(f"[{self.SCHEMA_NAME}] Creating initial schema v{self.SCHEMA_VERSION}")
                await conn.execute(POSTGRES_CRITIQUE_SCHEMA)
                await conn.execute(
                    """
                    INSERT INTO _schema_versions (module, version)
                    VALUES ($1, $2)
                    ON CONFLICT (module) DO UPDATE SET version = $2, updated_at = NOW()
                """,
                    self.SCHEMA_NAME,
                    self.SCHEMA_VERSION,
                )

            elif current_version < self.SCHEMA_VERSION:
                logger.info(
                    f"[{self.SCHEMA_NAME}] Migrating from v{current_version} to v{self.SCHEMA_VERSION}"
                )
                await conn.execute(POSTGRES_CRITIQUE_SCHEMA)
                await conn.execute(
                    """
                    UPDATE _schema_versions SET version = $1, updated_at = NOW()
                    WHERE module = $2
                """,
                    self.SCHEMA_VERSION,
                    self.SCHEMA_NAME,
                )

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized at version {self.SCHEMA_VERSION}")

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["Connection", None]:
        """Context manager for database operations."""
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator["Connection", None]:
        """Context manager for transactional operations."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    # =========================================================================
    # Debate Operations
    # =========================================================================

    async def store_debate(
        self,
        debate_id: str,
        task: str,
        final_answer: Optional[str] = None,
        consensus_reached: bool = False,
        confidence: Optional[float] = None,
        rounds_used: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        grounded_verdict: Optional[dict] = None,
        critiques: Optional[list[dict]] = None,
    ) -> dict:
        """
        Store a complete debate result.

        Args:
            debate_id: Unique debate identifier
            task: Debate task/question
            final_answer: Final consensus answer
            consensus_reached: Whether consensus was reached
            confidence: Confidence level (0-1)
            rounds_used: Number of debate rounds
            duration_seconds: Total debate duration
            grounded_verdict: Evidence/grounding data
            critiques: List of critique dicts to store

        Returns:
            Stored debate record as dict
        """
        async with self.transaction() as conn:
            # Store debate
            await conn.execute(
                """
                INSERT INTO debates
                    (id, task, final_answer, consensus_reached, confidence,
                     rounds_used, duration_seconds, grounded_verdict)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    task = $2,
                    final_answer = $3,
                    consensus_reached = $4,
                    confidence = $5,
                    rounds_used = $6,
                    duration_seconds = $7,
                    grounded_verdict = $8
            """,
                debate_id,
                task,
                final_answer,
                consensus_reached,
                confidence,
                rounds_used,
                duration_seconds,
                json.dumps(grounded_verdict) if grounded_verdict else None,
            )

            # Store critiques if provided
            if critiques:
                for critique in critiques:
                    await conn.execute(
                        """
                        INSERT INTO critiques
                            (debate_id, agent, target_agent, issues, suggestions,
                             severity, reasoning)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        debate_id,
                        critique.get("agent", ""),
                        critique.get("target_agent"),
                        json.dumps(critique.get("issues", [])),
                        json.dumps(critique.get("suggestions", [])),
                        critique.get("severity"),
                        critique.get("reasoning"),
                    )

        return {
            "id": debate_id,
            "task": task,
            "final_answer": final_answer,
            "consensus_reached": consensus_reached,
            "confidence": confidence,
        }

    async def get_debate(self, debate_id: str) -> Optional[dict]:
        """Get a debate record by ID."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT * FROM debates WHERE id = $1", debate_id)

            if not row:
                return None

            return {
                "id": row["id"],
                "task": row["task"],
                "final_answer": row["final_answer"],
                "consensus_reached": row["consensus_reached"],
                "confidence": row["confidence"],
                "rounds_used": row["rounds_used"],
                "duration_seconds": row["duration_seconds"],
                "grounded_verdict": json.loads(row["grounded_verdict"])
                if row["grounded_verdict"]
                else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }

    async def delete_debate(self, debate_id: str, cascade_critiques: bool = True) -> bool:
        """
        Delete a debate record and optionally its critiques.

        Args:
            debate_id: ID of the debate to delete
            cascade_critiques: If True, also delete associated critiques

        Returns:
            True if deleted, False if not found
        """
        async with self.transaction() as conn:
            # Check if exists
            row = await conn.fetchrow("SELECT 1 FROM debates WHERE id = $1", debate_id)
            if not row:
                return False

            # Delete critiques first if cascading (FK will handle if cascade enabled)
            if cascade_critiques:
                await conn.execute("DELETE FROM critiques WHERE debate_id = $1", debate_id)

            # Delete debate
            await conn.execute("DELETE FROM debates WHERE id = $1", debate_id)
            return True

    # =========================================================================
    # Critique Operations
    # =========================================================================

    async def store_critique(
        self,
        debate_id: Optional[str],
        agent: str,
        target_agent: Optional[str],
        issues: list[str],
        suggestions: list[str],
        severity: float,
        reasoning: str,
        expected_usefulness: float = 0.5,
    ) -> int:
        """
        Store a critique record.

        Args:
            debate_id: Associated debate ID
            agent: Agent that made the critique
            target_agent: Agent being critiqued
            issues: List of identified issues
            suggestions: List of suggested improvements
            severity: Severity rating (0-1)
            reasoning: Reasoning behind the critique
            expected_usefulness: Predicted usefulness (0-1)

        Returns:
            Inserted critique ID
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO critiques
                    (debate_id, agent, target_agent, issues, suggestions,
                     severity, reasoning, expected_usefulness)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                debate_id,
                agent,
                target_agent,
                json.dumps(issues),
                json.dumps(suggestions),
                severity,
                reasoning,
                expected_usefulness,
            )
            return row["id"]

    async def get_recent_critiques(self, limit: int = 20) -> list[dict]:
        """Return the most recent critiques."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, debate_id, agent, target_agent, issues, suggestions,
                       severity, reasoning, created_at
                FROM critiques
                ORDER BY created_at DESC
                LIMIT $1
            """,
                limit,
            )

            return [
                {
                    "id": row["id"],
                    "debate_id": row["debate_id"],
                    "agent": row["agent"],
                    "target_agent": row["target_agent"],
                    "issues": row["issues"] if isinstance(row["issues"], list) else [],
                    "suggestions": row["suggestions"]
                    if isinstance(row["suggestions"], list)
                    else [],
                    "severity": row["severity"] or 0.0,
                    "reasoning": row["reasoning"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]

    async def get_critiques_for_debate(self, debate_id: str) -> list[dict]:
        """Get all critiques for a specific debate."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, agent, target_agent, issues, suggestions,
                       severity, reasoning, created_at
                FROM critiques
                WHERE debate_id = $1
                ORDER BY created_at
            """,
                debate_id,
            )

            return [
                {
                    "id": row["id"],
                    "agent": row["agent"],
                    "target_agent": row["target_agent"],
                    "issues": row["issues"] if isinstance(row["issues"], list) else [],
                    "suggestions": row["suggestions"]
                    if isinstance(row["suggestions"], list)
                    else [],
                    "severity": row["severity"] or 0.0,
                    "reasoning": row["reasoning"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]

    async def update_prediction_outcome(
        self,
        critique_id: int,
        actual_usefulness: float,
        agent_name: Optional[str] = None,
    ) -> float:
        """
        Update critique with actual outcome, return prediction error.

        Args:
            critique_id: Database ID of the critique
            actual_usefulness: How useful the critique actually was (0-1)
            agent_name: Optional agent name to update calibration

        Returns:
            Prediction error (|expected - actual|)
        """
        async with self.transaction() as conn:
            # Get expected usefulness
            row = await conn.fetchrow(
                "SELECT expected_usefulness, agent FROM critiques WHERE id = $1",
                critique_id,
            )
            if not row:
                return 0.0

            expected = row["expected_usefulness"] if row["expected_usefulness"] else 0.5
            agent = agent_name or row["agent"]

            # Calculate prediction error
            prediction_error = abs(expected - actual_usefulness)

            # Update critique with outcome
            await conn.execute(
                """
                UPDATE critiques
                SET actual_usefulness = $1, prediction_error = $2
                WHERE id = $3
            """,
                actual_usefulness,
                prediction_error,
                critique_id,
            )

            # Update agent's calibration score if agent provided
            if agent:
                await self._update_agent_calibration(conn, agent, prediction_error)

            return prediction_error

    async def _update_agent_calibration(
        self,
        conn: "Connection",
        agent_name: str,
        prediction_error: float,
    ) -> None:
        """Update agent's calibration score based on prediction accuracy."""
        # Ensure agent exists
        await conn.execute(
            """
            INSERT INTO agent_reputation (agent_name)
            VALUES ($1)
            ON CONFLICT (agent_name) DO NOTHING
        """,
            agent_name,
        )

        # Update prediction tracking and calibration
        await conn.execute(
            """
            UPDATE agent_reputation
            SET total_predictions = total_predictions + 1,
                total_prediction_error = total_prediction_error + $1,
                calibration_score = 1.0 - (
                    (total_prediction_error + $1) / (total_predictions + 1)
                ),
                updated_at = NOW()
            WHERE agent_name = $2
        """,
            prediction_error,
            agent_name,
        )

    # =========================================================================
    # Pattern Operations
    # =========================================================================

    async def store_pattern(
        self,
        issue_text: str,
        suggestion_text: str,
        severity: float,
        example_task: str,
    ) -> dict:
        """
        Store a successful critique pattern.

        Args:
            issue_text: The issue text
            suggestion_text: The suggested fix
            severity: Severity rating
            example_task: Example task where this was used

        Returns:
            Pattern record as dict
        """
        # Create pattern ID from issue hash
        pattern_id = hashlib.sha256(issue_text.lower().encode()).hexdigest()[:12]
        issue_type = self._categorize_issue(issue_text)
        now = datetime.now(timezone.utc)

        async with self.connection() as conn:
            # Atomic upsert
            await conn.execute(
                """
                INSERT INTO patterns
                    (id, issue_type, issue_text, suggestion_text, success_count,
                     avg_severity, example_task, created_at, updated_at)
                VALUES ($1, $2, $3, $4, 1, $5, $6, $7, $7)
                ON CONFLICT (id) DO UPDATE SET
                    success_count = patterns.success_count + 1,
                    avg_severity = (patterns.avg_severity * patterns.success_count + $5) /
                                   (patterns.success_count + 1),
                    updated_at = $7
            """,
                pattern_id,
                issue_type,
                issue_text,
                suggestion_text,
                severity,
                example_task[:500],
                now,
            )

            # Update surprise score
            await self._update_surprise_score(conn, pattern_id, is_success=True)

        return {
            "id": pattern_id,
            "issue_type": issue_type,
            "issue_text": issue_text,
            "suggestion_text": suggestion_text,
        }

    def _categorize_issue(self, issue: str) -> str:
        """Simple issue categorization."""
        issue_lower = issue.lower()

        categories = {
            "performance": ["slow", "performance", "efficient", "optimize", "speed", "latency"],
            "security": ["security", "vulnerab", "injection", "auth", "permission", "xss", "csrf"],
            "correctness": ["bug", "error", "incorrect", "wrong", "fail", "break", "crash"],
            "clarity": ["unclear", "confusing", "readab", "document", "comment", "naming"],
            "architecture": ["design", "structure", "pattern", "modular", "coupling", "cohesion"],
            "completeness": ["missing", "incomplete", "todo", "edge case", "handle"],
            "testing": ["test", "coverage", "assert", "mock", "unit", "integration"],
        }

        for category, keywords in categories.items():
            if any(kw in issue_lower for kw in keywords):
                return category

        return "general"

    async def fail_pattern(self, issue_text: str) -> None:
        """
        Record a pattern failure.

        Called when a critique with matching issue text did NOT lead to improvement.
        """
        pattern_id = hashlib.sha256(issue_text.lower().encode()).hexdigest()[:12]

        async with self.connection() as conn:
            result = await conn.execute(
                """
                UPDATE patterns
                SET failure_count = failure_count + 1,
                    updated_at = NOW()
                WHERE id = $1
            """,
                pattern_id,
            )

            if result != "UPDATE 0":
                await self._update_surprise_score(conn, pattern_id, is_success=False)

    async def _calculate_surprise(
        self, conn: "Connection", issue_type: str, is_success: bool
    ) -> float:
        """Calculate surprise score based on deviation from base rate."""
        row = await conn.fetchrow(
            """
            SELECT AVG(
                CAST(success_count AS REAL) /
                NULLIF(success_count + failure_count, 0)
            ) as base_rate
            FROM patterns
            WHERE issue_type = $1 AND (success_count + failure_count) > 0
        """,
            issue_type,
        )
        base_rate = row["base_rate"] if row and row["base_rate"] else 0.5

        actual = 1.0 if is_success else 0.0
        surprise = abs(actual - base_rate)
        return min(1.0, surprise * 2)

    async def _update_surprise_score(
        self, conn: "Connection", pattern_id: str, is_success: bool
    ) -> None:
        """Update surprise score for a pattern after success/failure."""
        row = await conn.fetchrow("SELECT issue_type FROM patterns WHERE id = $1", pattern_id)
        if not row:
            return

        issue_type = row["issue_type"]
        surprise = await self._calculate_surprise(conn, issue_type, is_success)

        # Get base rate for this issue type
        base_row = await conn.fetchrow(
            """
            SELECT AVG(
                CAST(success_count AS REAL) /
                NULLIF(success_count + failure_count, 0)
            ) as base_rate
            FROM patterns WHERE issue_type = $1
        """,
            issue_type,
        )
        base_rate = base_row["base_rate"] if base_row and base_row["base_rate"] else 0.5

        await conn.execute(
            """
            UPDATE patterns
            SET surprise_score = surprise_score * 0.7 + $1 * 0.3,
                base_rate = $2
            WHERE id = $3
        """,
            surprise,
            base_rate,
            pattern_id,
        )

    async def retrieve_patterns(
        self,
        issue_type: Optional[str] = None,
        min_success: int = 2,
        limit: int = 10,
        decay_halflife_days: int = 30,
    ) -> list[Pattern]:
        """
        Retrieve successful patterns with Titans/MIRAS-inspired ranking.

        Ranking formula:
            score = (success_count * (1 + surprise_score)) /
                    (1 + age_days / decay_halflife_days)
        """
        async with self.connection() as conn:
            if issue_type:
                rows = await conn.fetch(
                    """
                    SELECT id, issue_type, issue_text, suggestion_text, success_count,
                           failure_count, avg_severity, example_task, created_at, updated_at,
                           (success_count * (1 + COALESCE(surprise_score, 0))) /
                           (1 + EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400 / $1) as decay_score
                    FROM patterns
                    WHERE success_count >= $2 AND issue_type = $3
                    ORDER BY decay_score DESC
                    LIMIT $4
                """,
                    decay_halflife_days,
                    min_success,
                    issue_type,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, issue_type, issue_text, suggestion_text, success_count,
                           failure_count, avg_severity, example_task, created_at, updated_at,
                           (success_count * (1 + COALESCE(surprise_score, 0))) /
                           (1 + EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400 / $1) as decay_score
                    FROM patterns
                    WHERE success_count >= $2
                    ORDER BY decay_score DESC
                    LIMIT $3
                """,
                    decay_halflife_days,
                    min_success,
                    limit,
                )

            return [
                Pattern(
                    id=row["id"],
                    issue_type=row["issue_type"],
                    issue_text=row["issue_text"],
                    suggestion_text=row["suggestion_text"] or "",
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                    avg_severity=row["avg_severity"] or 0.5,
                    example_task=row["example_task"] or "",
                    created_at=row["created_at"].isoformat() if row["created_at"] else "",
                    updated_at=row["updated_at"].isoformat() if row["updated_at"] else "",
                )
                for row in rows
            ]

    async def get_relevant(
        self, issue_type: Optional[str] = None, limit: int = 10
    ) -> list[Pattern]:
        """Backward-compatible wrapper for retrieve_patterns()."""
        return await self.retrieve_patterns(issue_type=issue_type, min_success=1, limit=limit)

    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern record."""
        async with self.connection() as conn:
            result = await conn.execute("DELETE FROM patterns WHERE id = $1", pattern_id)
            return result != "DELETE 0"

    # =========================================================================
    # Agent Reputation Operations
    # =========================================================================

    async def get_reputation(self, agent_name: str) -> Optional[AgentReputation]:
        """Get reputation for an agent."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT agent_name, proposals_made, proposals_accepted,
                       critiques_given, critiques_valuable, updated_at,
                       COALESCE(total_predictions, 0) as total_predictions,
                       COALESCE(total_prediction_error, 0.0) as total_prediction_error,
                       COALESCE(calibration_score, 0.5) as calibration_score
                FROM agent_reputation
                WHERE agent_name = $1
            """,
                agent_name,
            )

            if not row:
                return None

            return AgentReputation(
                agent_name=row["agent_name"],
                proposals_made=row["proposals_made"],
                proposals_accepted=row["proposals_accepted"],
                critiques_given=row["critiques_given"],
                critiques_valuable=row["critiques_valuable"],
                updated_at=row["updated_at"].isoformat() if row["updated_at"] else "",
                total_predictions=row["total_predictions"],
                total_prediction_error=row["total_prediction_error"],
                calibration_score=row["calibration_score"],
            )

    async def get_vote_weight(self, agent_name: str) -> float:
        """Get vote weight for an agent (0.4-1.6 range based on reputation)."""
        rep = await self.get_reputation(agent_name)
        if not rep:
            return 1.0
        return rep.vote_weight

    async def get_vote_weights_batch(self, agent_names: list[str]) -> dict[str, float]:
        """Get vote weights for multiple agents in a single query."""
        if not agent_names:
            return {}

        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT agent_name, proposals_made, proposals_accepted,
                       critiques_given, critiques_valuable,
                       COALESCE(calibration_score, 0.5) as calibration_score
                FROM agent_reputation
                WHERE agent_name = ANY($1)
            """,
                agent_names,
            )

            weights: dict[str, float] = {}
            for row in rows:
                agent_name = row["agent_name"]
                proposals_made = row["proposals_made"]
                proposals_accepted = row["proposals_accepted"]
                critiques_given = row["critiques_given"]
                critiques_valuable = row["critiques_valuable"]
                calibration_score = row["calibration_score"]

                if proposals_made == 0:
                    score = 0.5
                else:
                    acceptance = proposals_accepted / proposals_made
                    critique_quality = (
                        critiques_valuable / critiques_given if critiques_given > 0 else 0.5
                    )
                    score = 0.6 * acceptance + 0.4 * critique_quality

                base_weight = 0.5 + score
                calibration_bonus = (calibration_score - 0.5) * 0.2
                weights[agent_name] = max(0.4, min(1.6, base_weight + calibration_bonus))

            # Fill in missing agents with default weight
            for name in agent_names:
                if name not in weights:
                    weights[name] = 1.0

            return weights

    async def update_reputation(
        self,
        agent_name: str,
        proposal_made: bool = False,
        proposal_accepted: bool = False,
        critique_given: bool = False,
        critique_valuable: bool = False,
    ) -> None:
        """Update reputation metrics for an agent."""
        async with self.connection() as conn:
            # Ensure agent exists
            await conn.execute(
                """
                INSERT INTO agent_reputation (agent_name)
                VALUES ($1)
                ON CONFLICT (agent_name) DO NOTHING
            """,
                agent_name,
            )

            # Build update
            updates = []
            if proposal_made:
                updates.append("proposals_made = proposals_made + 1")
            if proposal_accepted:
                updates.append("proposals_accepted = proposals_accepted + 1")
            if critique_given:
                updates.append("critiques_given = critiques_given + 1")
            if critique_valuable:
                updates.append("critiques_valuable = critiques_valuable + 1")

            if updates:
                updates.append("updated_at = NOW()")
                sql = f"""
                    UPDATE agent_reputation
                    SET {", ".join(updates)}
                    WHERE agent_name = $1
                """  # nosec B608 - updates are hardcoded strings
                await conn.execute(sql, agent_name)

    async def get_all_reputations(self, limit: int = 500) -> list[AgentReputation]:
        """Get all agent reputations, ordered by score."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT agent_name, proposals_made, proposals_accepted,
                       critiques_given, critiques_valuable, updated_at,
                       COALESCE(total_predictions, 0) as total_predictions,
                       COALESCE(total_prediction_error, 0.0) as total_prediction_error,
                       COALESCE(calibration_score, 0.5) as calibration_score
                FROM agent_reputation
                ORDER BY proposals_accepted DESC
                LIMIT $1
            """,
                limit,
            )

            return [
                AgentReputation(
                    agent_name=row["agent_name"],
                    proposals_made=row["proposals_made"],
                    proposals_accepted=row["proposals_accepted"],
                    critiques_given=row["critiques_given"],
                    critiques_valuable=row["critiques_valuable"],
                    updated_at=row["updated_at"].isoformat() if row["updated_at"] else "",
                    total_predictions=row["total_predictions"],
                    total_prediction_error=row["total_prediction_error"],
                    calibration_score=row["calibration_score"],
                )
                for row in rows
            ]

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get statistics about stored patterns and debates."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    (SELECT COUNT(*) FROM debates) as total_debates,
                    (SELECT COUNT(*) FROM debates WHERE consensus_reached = TRUE) as consensus_debates,
                    (SELECT COUNT(*) FROM critiques) as total_critiques,
                    (SELECT COUNT(*) FROM patterns) as total_patterns,
                    (SELECT AVG(confidence) FROM debates WHERE consensus_reached = TRUE) as avg_confidence
            """
            )

            stats = {
                "total_debates": row["total_debates"] or 0,
                "consensus_debates": row["consensus_debates"] or 0,
                "total_critiques": row["total_critiques"] or 0,
                "total_patterns": row["total_patterns"] or 0,
                "avg_consensus_confidence": float(row["avg_confidence"] or 0.0),
            }

            # Patterns by type
            type_rows = await conn.fetch(
                "SELECT issue_type, COUNT(*) as cnt FROM patterns GROUP BY issue_type"
            )
            stats["patterns_by_type"] = {row["issue_type"]: row["cnt"] for row in type_rows}

            return stats

    async def export_for_training(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        """Export successful patterns for potential fine-tuning."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT d.task, c.issues, c.suggestions, d.final_answer, d.consensus_reached
                FROM critiques c
                JOIN debates d ON c.debate_id = d.id
                WHERE d.consensus_reached = TRUE
                LIMIT $1 OFFSET $2
            """,
                limit,
                offset,
            )

            return [
                {
                    "task": row["task"],
                    "issues": row["issues"] if isinstance(row["issues"], list) else [],
                    "suggestions": row["suggestions"]
                    if isinstance(row["suggestions"], list)
                    else [],
                    "successful_answer": row["final_answer"],
                }
                for row in rows
            ]

    # =========================================================================
    # Adaptive Forgetting (Titans/MIRAS)
    # =========================================================================

    async def prune_stale_patterns(
        self,
        max_age_days: int = 90,
        min_success_rate: float = 0.3,
        archive: bool = True,
    ) -> int:
        """
        Remove or archive patterns that are stale or unsuccessful.

        Args:
            max_age_days: Patterns older than this without updates get pruned
            min_success_rate: Patterns below this success rate get pruned
            archive: If True, move to archive table instead of deleting

        Returns:
            Number of patterns pruned
        """
        async with self.transaction() as conn:
            if archive:
                # Move stale/unsuccessful patterns to archive table
                await conn.execute(
                    """
                    INSERT INTO patterns_archive
                        (id, issue_type, issue_text, suggestion_text, success_count,
                         failure_count, avg_severity, surprise_score, example_task,
                         created_at, updated_at)
                    SELECT id, issue_type, issue_text, suggestion_text, success_count,
                           failure_count, avg_severity, surprise_score, example_task,
                           created_at, updated_at
                    FROM patterns
                    WHERE EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400 > $1
                      AND (
                        CAST(success_count AS REAL) /
                        NULLIF(success_count + failure_count, 0)
                      ) < $2
                """,
                    max_age_days,
                    min_success_rate,
                )

            # Delete stale/unsuccessful patterns
            result = await conn.execute(
                """
                DELETE FROM patterns
                WHERE EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400 > $1
                  AND (
                    CAST(success_count AS REAL) /
                    NULLIF(success_count + failure_count, 0)
                  ) < $2
            """,
                max_age_days,
                min_success_rate,
            )

            # Extract count from "DELETE X"
            try:
                pruned = int(result.split()[-1])
            except (ValueError, IndexError):
                pruned = 0

            return pruned

    async def get_archive_stats(self) -> dict:
        """Get statistics about archived patterns."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as total FROM patterns_archive")
            total = row["total"] if row else 0

            type_rows = await conn.fetch(
                """
                SELECT issue_type, COUNT(*) as cnt
                FROM patterns_archive
                GROUP BY issue_type
            """
            )
            by_type = {row["issue_type"]: row["cnt"] for row in type_rows}

            return {"total_archived": total, "archived_by_type": by_type}

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def count_debates(self) -> int:
        """Return total number of debates."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM debates")
            return row["cnt"] if row else 0

    async def count_critiques(self) -> int:
        """Return total number of critiques."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM critiques")
            return row["cnt"] if row else 0

    async def count_patterns(self) -> int:
        """Return total number of patterns."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM patterns")
            return row["cnt"] if row else 0


async def get_postgres_critique_store(
    dsn: Optional[str] = None,
) -> PostgresCritiqueStore:
    """
    Factory function to create an initialized PostgresCritiqueStore.

    Args:
        dsn: PostgreSQL connection string. If not provided, uses environment.

    Returns:
        Initialized PostgresCritiqueStore instance

    Raises:
        RuntimeError: If asyncpg is not installed
    """
    if not ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install aragora[postgres] or pip install asyncpg"
        )

    from aragora.storage.postgres_store import get_postgres_pool

    pool = await get_postgres_pool(dsn=dsn)
    store = PostgresCritiqueStore(pool)
    await store.initialize()
    return store
