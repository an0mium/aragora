"""
Audience Suggestion Feedback Tracking.

Records which suggestions were injected into debates and tracks their
effectiveness based on debate outcomes. Enables learning from community
participation to improve suggestion quality over time.

Features:
- Track suggestion injection into debates
- Record debate outcomes with suggestion context
- Calculate suggestion effectiveness scores
- Identify high-value contributors
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from aragora.config import resolve_db_path
from aragora.storage.base_store import SQLiteStore


@dataclass
class SuggestionRecord:
    """A recorded suggestion injection."""

    id: str
    debate_id: str
    suggestion_text: str
    cluster_count: int  # How many similar suggestions
    user_ids: list[str]  # Contributing users
    injected_at: str

    # Outcome tracking (filled after debate)
    debate_completed: bool = False
    consensus_reached: bool = False
    consensus_confidence: float = 0.0
    duration_seconds: float = 0.0

    # Effectiveness metrics
    effectiveness_score: float = 0.0  # 0-1, higher is better

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "suggestion_text": self.suggestion_text,
            "cluster_count": self.cluster_count,
            "user_ids": self.user_ids,
            "injected_at": self.injected_at,
            "debate_completed": self.debate_completed,
            "consensus_reached": self.consensus_reached,
            "consensus_confidence": self.consensus_confidence,
            "duration_seconds": self.duration_seconds,
            "effectiveness_score": self.effectiveness_score,
        }


@dataclass
class ContributorStats:
    """Statistics for a suggestion contributor."""

    user_id: str
    total_suggestions: int = 0
    suggestions_in_consensus: int = 0
    avg_effectiveness: float = 0.0
    reputation_score: float = 0.5  # 0-1, starts neutral

    @property
    def consensus_rate(self) -> float:
        if self.total_suggestions == 0:
            return 0.0
        return self.suggestions_in_consensus / self.total_suggestions


class SuggestionFeedbackTracker(SQLiteStore):
    """
    Tracks suggestion effectiveness and contributor reputation.

    Usage:
        tracker = SuggestionFeedbackTracker("suggestions.db")

        # When suggestions are injected
        tracker.record_injection(debate_id, clusters)

        # When debate completes
        tracker.record_outcome(debate_id, result)

        # Get contributor stats
        stats = tracker.get_contributor_stats(user_id)
    """

    SCHEMA_NAME = "suggestion_feedback"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS suggestion_injections (
            id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            suggestion_text TEXT NOT NULL,
            cluster_count INTEGER DEFAULT 1,
            user_ids TEXT,
            injected_at TEXT,
            debate_completed INTEGER DEFAULT 0,
            consensus_reached INTEGER DEFAULT 0,
            consensus_confidence REAL DEFAULT 0.0,
            duration_seconds REAL DEFAULT 0.0,
            effectiveness_score REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS contributor_stats (
            user_id TEXT PRIMARY KEY,
            total_suggestions INTEGER DEFAULT 0,
            suggestions_in_consensus INTEGER DEFAULT 0,
            avg_effectiveness REAL DEFAULT 0.0,
            reputation_score REAL DEFAULT 0.5,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_injection_debate
        ON suggestion_injections(debate_id);

        CREATE INDEX IF NOT EXISTS idx_injection_completed
        ON suggestion_injections(debate_completed);
    """

    def __init__(self, db_path: str = "suggestion_feedback.db"):
        super().__init__(resolve_db_path(db_path))

    def record_injection(
        self,
        debate_id: str,
        clusters: list,  # List of SuggestionCluster
    ) -> list[str]:
        """
        Record suggestion injections for a debate.

        Args:
            debate_id: Unique debate identifier
            clusters: List of SuggestionCluster objects

        Returns:
            List of injection IDs
        """
        import uuid

        injection_ids = []

        with self.connection() as conn:
            cursor = conn.cursor()

            for cluster in clusters:
                injection_id = str(uuid.uuid4())

                # Handle both SuggestionCluster objects and dicts
                if hasattr(cluster, "representative"):
                    text = cluster.representative
                    count = cluster.count
                    user_ids = cluster.user_ids
                else:
                    text = cluster.get("representative", "")
                    count = cluster.get("count", 1)
                    user_ids = cluster.get("user_ids", [])

                cursor.execute(
                    """
                    INSERT INTO suggestion_injections
                    (id, debate_id, suggestion_text, cluster_count, user_ids, injected_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        injection_id,
                        debate_id,
                        text,
                        count,
                        json.dumps(user_ids),
                        datetime.now().isoformat(),
                    ),
                )

                injection_ids.append(injection_id)

            conn.commit()

        return injection_ids

    def record_outcome(
        self,
        debate_id: str,
        consensus_reached: bool,
        consensus_confidence: float,
        duration_seconds: float = 0.0,
        baseline_consensus_rate: float = 0.6,  # Historical average
    ) -> int:
        """
        Record debate outcome and update suggestion effectiveness.

        Args:
            debate_id: Debate identifier
            consensus_reached: Whether consensus was achieved
            consensus_confidence: Confidence level (0-1)
            duration_seconds: Debate duration
            baseline_consensus_rate: Historical consensus rate for comparison

        Returns:
            Number of suggestions updated
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            # Get all injections for this debate
            cursor.execute(
                "SELECT id, user_ids, cluster_count FROM suggestion_injections WHERE debate_id = ?",
                (debate_id,),
            )
            injections = cursor.fetchall()

            if not injections:
                return 0

            # Calculate effectiveness score
            # Higher if consensus reached with high confidence
            # Bonus if faster than average
            effectiveness = 0.0
            if consensus_reached:
                effectiveness = 0.5 + (consensus_confidence * 0.4)
                # Bonus for strong consensus
                if consensus_confidence >= 0.8:
                    effectiveness += 0.1
            else:
                # Partial credit for high-confidence disagreement (at least it's clear)
                effectiveness = 0.2 + (consensus_confidence * 0.2)

            effectiveness = min(1.0, max(0.0, effectiveness))

            # Update all injections for this debate
            cursor.execute(
                """
                UPDATE suggestion_injections
                SET debate_completed = 1,
                    consensus_reached = ?,
                    consensus_confidence = ?,
                    duration_seconds = ?,
                    effectiveness_score = ?
                WHERE debate_id = ?
            """,
                (
                    1 if consensus_reached else 0,
                    consensus_confidence,
                    duration_seconds,
                    effectiveness,
                    debate_id,
                ),
            )

            # Update contributor stats
            for injection_id, user_ids_json, cluster_count in injections:
                user_ids = json.loads(user_ids_json) if user_ids_json else []
                for user_id in user_ids:
                    if user_id:
                        self._update_contributor_stats(
                            cursor, user_id, consensus_reached, effectiveness
                        )

            conn.commit()

        return len(injections)

    def _update_contributor_stats(
        self,
        cursor,
        user_id: str,
        consensus_reached: bool,
        effectiveness: float,
    ) -> None:
        """Update stats for a contributor."""
        # Get existing stats
        cursor.execute(
            "SELECT total_suggestions, suggestions_in_consensus, avg_effectiveness, reputation_score "
            "FROM contributor_stats WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()

        if row:
            total = row[0] + 1
            in_consensus = row[1] + (1 if consensus_reached else 0)
            # Exponential moving average for effectiveness
            alpha = 0.3
            new_avg_eff = alpha * effectiveness + (1 - alpha) * row[2]
            # Update reputation based on effectiveness trend
            old_rep = row[3]
            new_rep = old_rep + (effectiveness - 0.5) * 0.1  # Adjust by ±0.05 per suggestion
            new_rep = min(1.0, max(0.0, new_rep))
        else:
            total = 1
            in_consensus = 1 if consensus_reached else 0
            new_avg_eff = effectiveness
            new_rep = 0.5 + (effectiveness - 0.5) * 0.1

        cursor.execute(
            """
            INSERT OR REPLACE INTO contributor_stats
            (user_id, total_suggestions, suggestions_in_consensus, avg_effectiveness, reputation_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                total,
                in_consensus,
                new_avg_eff,
                new_rep,
                datetime.now().isoformat(),
            ),
        )

    def get_contributor_stats(self, user_id: str) -> Optional[ContributorStats]:
        """Get stats for a specific contributor."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, total_suggestions, suggestions_in_consensus, avg_effectiveness, reputation_score "
                "FROM contributor_stats WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return ContributorStats(
            user_id=row[0],
            total_suggestions=row[1],
            suggestions_in_consensus=row[2],
            avg_effectiveness=row[3],
            reputation_score=row[4],
        )

    def get_top_contributors(self, limit: int = 10) -> list[ContributorStats]:
        """Get top contributors by reputation."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, total_suggestions, suggestions_in_consensus, avg_effectiveness, reputation_score
                FROM contributor_stats
                WHERE total_suggestions >= 3
                ORDER BY reputation_score DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            ContributorStats(
                user_id=row[0],
                total_suggestions=row[1],
                suggestions_in_consensus=row[2],
                avg_effectiveness=row[3],
                reputation_score=row[4],
            )
            for row in rows
        ]

    def get_debate_suggestions(self, debate_id: str) -> list[SuggestionRecord]:
        """Get all suggestions for a debate."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM suggestion_injections WHERE debate_id = ?", (debate_id,))
            rows = cursor.fetchall()

        return [
            SuggestionRecord(
                id=row[0],
                debate_id=row[1],
                suggestion_text=row[2],
                cluster_count=row[3],
                user_ids=json.loads(row[4]) if row[4] else [],
                injected_at=row[5],
                debate_completed=bool(row[6]),
                consensus_reached=bool(row[7]),
                consensus_confidence=row[8],
                duration_seconds=row[9],
                effectiveness_score=row[10],
            )
            for row in rows
        ]

    def get_effectiveness_stats(self) -> dict[str, Any]:
        """Get overall suggestion effectiveness statistics."""
        with self.connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total suggestions
            cursor.execute("SELECT COUNT(*) FROM suggestion_injections")
            row = cursor.fetchone()
            stats["total_suggestions"] = row[0] if row else 0

            # Completed debates with suggestions
            cursor.execute(
                "SELECT COUNT(DISTINCT debate_id) FROM suggestion_injections WHERE debate_completed = 1"
            )
            row = cursor.fetchone()
            stats["debates_with_suggestions"] = row[0] if row else 0

            # Consensus rate for debates with suggestions
            cursor.execute(
                """
                SELECT AVG(consensus_reached) FROM suggestion_injections
                WHERE debate_completed = 1
            """
            )
            row = cursor.fetchone()
            avg = row[0] if row else None
            stats["consensus_rate"] = avg if avg else 0.0

            # Average effectiveness
            cursor.execute(
                """
                SELECT AVG(effectiveness_score) FROM suggestion_injections
                WHERE debate_completed = 1
            """
            )
            row = cursor.fetchone()
            avg = row[0] if row else None
            stats["avg_effectiveness"] = avg if avg else 0.0

            # Total contributors
            cursor.execute("SELECT COUNT(*) FROM contributor_stats")
            row = cursor.fetchone()
            stats["total_contributors"] = row[0] if row else 0

            # Top effectiveness suggestions
            cursor.execute(
                """
                SELECT suggestion_text, effectiveness_score, cluster_count
                FROM suggestion_injections
                WHERE debate_completed = 1 AND effectiveness_score >= 0.7
                ORDER BY effectiveness_score DESC
                LIMIT 5
            """
            )
            stats["top_suggestions"] = [
                {"text": row[0][:100], "score": row[1], "count": row[2]}
                for row in cursor.fetchall()
            ]

        return stats

    def filter_by_reputation(
        self,
        suggestions: list[dict],
        min_reputation: float = 0.3,
    ) -> list[dict]:
        """
        Filter suggestions to prioritize those from reputable contributors.

        Args:
            suggestions: Raw suggestions with user_id
            min_reputation: Minimum reputation to include

        Returns:
            Filtered and sorted suggestions (high rep first)
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            filtered = []
            for s in suggestions:
                user_id = s.get("user_id", "")[:8]
                if not user_id:
                    # Unknown user - include with neutral score
                    filtered.append((s, 0.5))
                    continue

                cursor.execute(
                    "SELECT reputation_score FROM contributor_stats WHERE user_id = ?", (user_id,)
                )
                row = cursor.fetchone()
                rep = row[0] if row else 0.5

                if rep >= min_reputation:
                    filtered.append((s, rep))

        # Sort by reputation (highest first)
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in filtered]
