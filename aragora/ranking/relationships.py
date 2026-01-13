"""
Agent Relationship Tracking for Grounded Personas.

Tracks alliances, rivalries, and interaction patterns between agents.
Extracted from EloSystem to separate social dynamics from competitive ranking.

Usage:
    tracker = RelationshipTracker(db_path)
    tracker.update_relationship("claude", "gemini", debate_increment=1)
    metrics = tracker.compute_metrics("claude", "gemini")
    rivals = tracker.get_rivals("claude")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aragora.ranking.database import EloDatabase

logger = logging.getLogger(__name__)

# Maximum relationship query limit to prevent resource exhaustion
MAX_RELATIONSHIP_LIMIT = 1000


@dataclass
class RelationshipStats:
    """Statistics for a relationship between two agents."""

    agent_a: str
    agent_b: str
    debate_count: int
    agreement_count: int
    critique_count_a_to_b: int
    critique_count_b_to_a: int
    critique_accepted_a_to_b: int
    critique_accepted_b_to_a: int
    position_changes_a_after_b: int
    position_changes_b_after_a: int
    a_wins_over_b: int
    b_wins_over_a: int


@dataclass
class RelationshipMetrics:
    """Computed metrics for a relationship."""

    agent_a: str
    agent_b: str
    rivalry_score: float
    alliance_score: float
    relationship: str  # "rival", "ally", "neutral", "acquaintance", "no_history", "unknown"
    debate_count: int
    agreement_rate: float = 0.0
    head_to_head: str = "0-0"


class RelationshipTracker:
    """
    Tracks and analyzes relationships between agents.

    Monitors debates, critiques, position changes, and wins to compute
    rivalry and alliance scores. Used by grounded personas to inform
    interaction strategies.

    Usage:
        tracker = RelationshipTracker(db_path)

        # Update after a debate
        tracker.update_relationship("claude", "gemini",
            debate_increment=1, a_win=1)

        # Batch update for efficiency
        tracker.update_batch([
            {"agent_a": "claude", "agent_b": "gemini", "debate_increment": 1},
            {"agent_a": "claude", "agent_b": "gpt4", "agreement_increment": 1},
        ])

        # Query relationships
        metrics = tracker.compute_metrics("claude", "gemini")
        rivals = tracker.get_rivals("claude", limit=5)
        allies = tracker.get_allies("claude", limit=5)
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize the relationship tracker.

        Args:
            db_path: Path to database file (same as EloSystem)
        """
        self.db_path = Path(db_path)
        self._db = EloDatabase(db_path)

    def update_relationship(
        self,
        agent_a: str,
        agent_b: str,
        debate_increment: int = 0,
        agreement_increment: int = 0,
        critique_a_to_b: int = 0,
        critique_b_to_a: int = 0,
        critique_accepted_a_to_b: int = 0,
        critique_accepted_b_to_a: int = 0,
        position_change_a_after_b: int = 0,
        position_change_b_after_a: int = 0,
        a_win: int = 0,
        b_win: int = 0,
    ) -> None:
        """
        Update relationship stats between two agents.

        Maintains canonical ordering (agent_a < agent_b) internally.

        Args:
            agent_a: First agent name
            agent_b: Second agent name
            debate_increment: Number of debates to add
            agreement_increment: Number of agreements to add
            critique_a_to_b: Critiques from a to b
            critique_b_to_a: Critiques from b to a
            critique_accepted_a_to_b: Accepted critiques from a to b
            critique_accepted_b_to_a: Accepted critiques from b to a
            position_change_a_after_b: Times a changed position after b's input
            position_change_b_after_a: Times b changed position after a's input
            a_win: Wins for agent a over b
            b_win: Wins for agent b over a
        """
        # Maintain canonical ordering
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
            critique_a_to_b, critique_b_to_a = critique_b_to_a, critique_a_to_b
            critique_accepted_a_to_b, critique_accepted_b_to_a = (
                critique_accepted_b_to_a,
                critique_accepted_a_to_b,
            )
            position_change_a_after_b, position_change_b_after_a = (
                position_change_b_after_a,
                position_change_a_after_b,
            )
            a_win, b_win = b_win, a_win

        now = datetime.now().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_relationships (agent_a, agent_b, debate_count, agreement_count,
                    critique_count_a_to_b, critique_count_b_to_a, critique_accepted_a_to_b, critique_accepted_b_to_a,
                    position_changes_a_after_b, position_changes_b_after_a, a_wins_over_b, b_wins_over_a, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_a, agent_b) DO UPDATE SET
                    debate_count = debate_count + ?, agreement_count = agreement_count + ?,
                    critique_count_a_to_b = critique_count_a_to_b + ?, critique_count_b_to_a = critique_count_b_to_a + ?,
                    critique_accepted_a_to_b = critique_accepted_a_to_b + ?, critique_accepted_b_to_a = critique_accepted_b_to_a + ?,
                    position_changes_a_after_b = position_changes_a_after_b + ?, position_changes_b_after_a = position_changes_b_after_a + ?,
                    a_wins_over_b = a_wins_over_b + ?, b_wins_over_a = b_wins_over_a + ?, updated_at = ?
                """,
                (
                    agent_a,
                    agent_b,
                    debate_increment,
                    agreement_increment,
                    critique_a_to_b,
                    critique_b_to_a,
                    critique_accepted_a_to_b,
                    critique_accepted_b_to_a,
                    position_change_a_after_b,
                    position_change_b_after_a,
                    a_win,
                    b_win,
                    now,
                    debate_increment,
                    agreement_increment,
                    critique_a_to_b,
                    critique_b_to_a,
                    critique_accepted_a_to_b,
                    critique_accepted_b_to_a,
                    position_change_a_after_b,
                    position_change_b_after_a,
                    a_win,
                    b_win,
                    now,
                ),
            )
            conn.commit()

    def update_batch(self, updates: list[dict]) -> None:
        """
        Batch update multiple agent relationships in a single transaction.

        More efficient than calling update_relationship() in a loop.

        Args:
            updates: List of dicts, each containing:
                - agent_a: str
                - agent_b: str
                - debate_increment: int (default 0)
                - agreement_increment: int (default 0)
                - a_win: int (default 0)
                - b_win: int (default 0)
        """
        if not updates:
            return

        now = datetime.now().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()
            for upd in updates:
                agent_a = upd.get("agent_a", "")
                agent_b = upd.get("agent_b", "")
                if not agent_a or not agent_b:
                    continue

                debate_increment = upd.get("debate_increment", 0)
                agreement_increment = upd.get("agreement_increment", 0)
                a_win = upd.get("a_win", 0)
                b_win = upd.get("b_win", 0)

                # Maintain canonical ordering (a < b)
                if agent_a > agent_b:
                    agent_a, agent_b = agent_b, agent_a
                    a_win, b_win = b_win, a_win

                cursor.execute(
                    """
                    INSERT INTO agent_relationships (agent_a, agent_b, debate_count, agreement_count,
                        critique_count_a_to_b, critique_count_b_to_a, critique_accepted_a_to_b, critique_accepted_b_to_a,
                        position_changes_a_after_b, position_changes_b_after_a, a_wins_over_b, b_wins_over_a, updated_at)
                    VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0, ?, ?, ?)
                    ON CONFLICT(agent_a, agent_b) DO UPDATE SET
                        debate_count = debate_count + ?, agreement_count = agreement_count + ?,
                        a_wins_over_b = a_wins_over_b + ?, b_wins_over_a = b_wins_over_a + ?, updated_at = ?
                    """,
                    (
                        agent_a,
                        agent_b,
                        debate_increment,
                        agreement_increment,
                        a_win,
                        b_win,
                        now,
                        debate_increment,
                        agreement_increment,
                        a_win,
                        b_win,
                        now,
                    ),
                )
            conn.commit()

    def get_raw(self, agent_a: str, agent_b: str) -> RelationshipStats | None:
        """
        Get raw relationship data between two agents.

        Args:
            agent_a: First agent name
            agent_b: Second agent name

        Returns:
            RelationshipStats or None if no relationship exists
        """
        # Maintain canonical ordering
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT debate_count, agreement_count, critique_count_a_to_b, critique_count_b_to_a,
                          critique_accepted_a_to_b, critique_accepted_b_to_a,
                          position_changes_a_after_b, position_changes_b_after_a,
                          a_wins_over_b, b_wins_over_a
                   FROM agent_relationships WHERE agent_a = ? AND agent_b = ?""",
                (agent_a, agent_b),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return RelationshipStats(
            agent_a=agent_a,
            agent_b=agent_b,
            debate_count=row[0],
            agreement_count=row[1],
            critique_count_a_to_b=row[2],
            critique_count_b_to_a=row[3],
            critique_accepted_a_to_b=row[4],
            critique_accepted_b_to_a=row[5],
            position_changes_a_after_b=row[6],
            position_changes_b_after_a=row[7],
            a_wins_over_b=row[8],
            b_wins_over_a=row[9],
        )

    def get_all_for_agent(self, agent_name: str, limit: int = 100) -> list[RelationshipStats]:
        """
        Get all relationships involving an agent.

        Args:
            agent_name: The agent to get relationships for
            limit: Maximum number of relationships to return (default 100, max 1000)

        Returns:
            List of RelationshipStats, ordered by debate_count descending
        """
        # Enforce maximum limit to prevent resource exhaustion
        limit = min(limit, MAX_RELATIONSHIP_LIMIT)

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT agent_a, agent_b, debate_count, agreement_count,
                          critique_count_a_to_b, critique_count_b_to_a,
                          critique_accepted_a_to_b, critique_accepted_b_to_a,
                          position_changes_a_after_b, position_changes_b_after_a,
                          a_wins_over_b, b_wins_over_a
                   FROM agent_relationships
                   WHERE agent_a = ? OR agent_b = ?
                   ORDER BY debate_count DESC
                   LIMIT ?""",
                (agent_name, agent_name, limit),
            )
            rows = cursor.fetchall()

        return [
            RelationshipStats(
                agent_a=r[0],
                agent_b=r[1],
                debate_count=r[2],
                agreement_count=r[3],
                critique_count_a_to_b=r[4],
                critique_count_b_to_a=r[5],
                critique_accepted_a_to_b=r[6],
                critique_accepted_b_to_a=r[7],
                position_changes_a_after_b=r[8],
                position_changes_b_after_a=r[9],
                a_wins_over_b=r[10],
                b_wins_over_a=r[11],
            )
            for r in rows
        ]

    def _compute_metrics_from_stats(
        self, agent_a: str, agent_b: str, stats: RelationshipStats
    ) -> RelationshipMetrics:
        """
        Compute relationship metrics from raw stats (no database call).

        Args:
            agent_a: First agent (for consistent output ordering)
            agent_b: Second agent
            stats: Raw relationship statistics

        Returns:
            Computed RelationshipMetrics
        """
        debate_count = stats.debate_count
        if debate_count == 0:
            return RelationshipMetrics(
                agent_a=agent_a,
                agent_b=agent_b,
                rivalry_score=0.0,
                alliance_score=0.0,
                relationship="no_history",
                debate_count=0,
            )

        # Agreement rate (0-1, higher = more allied)
        agreement_rate = stats.agreement_count / debate_count

        # Win competitiveness
        a_wins = stats.a_wins_over_b
        b_wins = stats.b_wins_over_a
        total_wins = a_wins + b_wins
        win_balance = (
            min(a_wins, b_wins) / max(a_wins, b_wins)
            if total_wins > 0 and max(a_wins, b_wins) > 0
            else 0.5
        )

        # Critique acceptance rate
        critiques_given = stats.critique_count_a_to_b + stats.critique_count_b_to_a
        critiques_accepted = stats.critique_accepted_a_to_b + stats.critique_accepted_b_to_a
        critique_acceptance = critiques_accepted / critiques_given if critiques_given > 0 else 0.5

        # Rivalry score: high debates + low agreement + competitive wins
        rivalry_score = (
            min(1.0, debate_count / 20) * 0.3  # Engagement factor (caps at 20 debates)
            + (1 - agreement_rate) * 0.4  # Disagreement factor
            + win_balance * 0.3  # Competitiveness factor
        )

        # Alliance score: high agreement + high critique acceptance
        if total_wins > 2:
            alliance_score = (
                agreement_rate * 0.5 + critique_acceptance * 0.3 + (1 - win_balance) * 0.2
            )
        else:
            alliance_score = agreement_rate * 0.5 + critique_acceptance * 0.5

        # Determine relationship type
        if rivalry_score > 0.6 and rivalry_score > alliance_score:
            relationship = "rival"
        elif alliance_score > 0.6 and alliance_score > rivalry_score:
            relationship = "ally"
        elif debate_count < 3:
            relationship = "acquaintance"
        else:
            relationship = "neutral"

        return RelationshipMetrics(
            agent_a=agent_a,
            agent_b=agent_b,
            rivalry_score=round(rivalry_score, 3),
            alliance_score=round(alliance_score, 3),
            relationship=relationship,
            debate_count=debate_count,
            agreement_rate=round(agreement_rate, 3),
            head_to_head=f"{a_wins}-{b_wins}",
        )

    def compute_metrics(self, agent_a: str, agent_b: str) -> RelationshipMetrics:
        """
        Compute rivalry and alliance scores between two agents.

        Rivalry is high when: many debates, low agreement, competitive wins.
        Alliance is high when: high agreement, mutual critique acceptance.

        Args:
            agent_a: First agent name
            agent_b: Second agent name

        Returns:
            RelationshipMetrics with rivalry_score, alliance_score, and relationship type
        """
        stats = self.get_raw(agent_a, agent_b)
        if not stats:
            return RelationshipMetrics(
                agent_a=agent_a,
                agent_b=agent_b,
                rivalry_score=0.0,
                alliance_score=0.0,
                relationship="unknown",
                debate_count=0,
            )

        return self._compute_metrics_from_stats(agent_a, agent_b, stats)

    def get_rivals(
        self, agent_name: str, limit: int = 5, min_score: float = 0.3
    ) -> list[RelationshipMetrics]:
        """
        Get agent's top rivals by rivalry score.

        Optimized to use single database query.

        Args:
            agent_name: Agent to get rivals for
            limit: Maximum rivals to return
            min_score: Minimum rivalry score to include

        Returns:
            List of RelationshipMetrics sorted by rivalry_score descending
        """
        relationships = self.get_all_for_agent(agent_name)
        scored = []

        for stats in relationships:
            other = stats.agent_b if stats.agent_a == agent_name else stats.agent_a
            metrics = self._compute_metrics_from_stats(agent_name, other, stats)
            if metrics.rivalry_score > min_score:
                scored.append(metrics)

        scored.sort(key=lambda x: x.rivalry_score, reverse=True)
        return scored[:limit]

    def get_allies(
        self, agent_name: str, limit: int = 5, min_score: float = 0.3
    ) -> list[RelationshipMetrics]:
        """
        Get agent's top allies by alliance score.

        Optimized to use single database query.

        Args:
            agent_name: Agent to get allies for
            limit: Maximum allies to return
            min_score: Minimum alliance score to include

        Returns:
            List of RelationshipMetrics sorted by alliance_score descending
        """
        relationships = self.get_all_for_agent(agent_name)
        scored = []

        for stats in relationships:
            other = stats.agent_b if stats.agent_a == agent_name else stats.agent_a
            metrics = self._compute_metrics_from_stats(agent_name, other, stats)
            if metrics.alliance_score > min_score:
                scored.append(metrics)

        scored.sort(key=lambda x: x.alliance_score, reverse=True)
        return scored[:limit]
