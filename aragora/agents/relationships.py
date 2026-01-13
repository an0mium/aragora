"""
Agent relationship tracking for grounded personas.

Computes relationship metrics from existing debate data including
rivalry scores, alliance scores, and influence networks.
"""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from aragora.config import DB_ELO_PATH
from aragora.ranking.database import EloDatabase

logger = logging.getLogger(__name__)


@dataclass
class AgentRelationship:
    """Relationship metrics between two agents."""

    agent_a: str
    agent_b: str
    debate_count: int = 0
    agreement_count: int = 0
    critique_count_a_to_b: int = 0
    critique_count_b_to_a: int = 0
    critique_accepted_a_to_b: int = 0
    critique_accepted_b_to_a: int = 0
    position_changes_a_after_b: int = 0
    position_changes_b_after_a: int = 0
    a_wins_over_b: int = 0
    b_wins_over_a: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def rivalry_score(self) -> float:
        """High debates + low agreement + competitive win rate."""
        if self.debate_count < 3:
            return 0.0
        disagreement_rate = 1 - (self.agreement_count / self.debate_count)
        total_wins = self.a_wins_over_b + self.b_wins_over_a
        competitiveness = 1 - abs(self.a_wins_over_b - self.b_wins_over_a) / max(total_wins, 1)
        frequency_factor = min(1.0, self.debate_count / 20)
        return disagreement_rate * competitiveness * frequency_factor

    @property
    def alliance_score(self) -> float:
        """High agreement + mutual critique acceptance."""
        if self.debate_count < 3:
            return 0.0
        agreement_rate = self.agreement_count / self.debate_count
        total_critiques = self.critique_count_a_to_b + self.critique_count_b_to_a
        total_accepted = self.critique_accepted_a_to_b + self.critique_accepted_b_to_a
        acceptance_rate = total_accepted / max(total_critiques, 1)
        return agreement_rate * 0.6 + acceptance_rate * 0.4

    @property
    def influence_a_on_b(self) -> float:
        """How much A influences B's positions."""
        if self.debate_count == 0:
            return 0.0
        return self.position_changes_b_after_a / self.debate_count

    @property
    def influence_b_on_a(self) -> float:
        """How much B influences A's positions."""
        if self.debate_count == 0:
            return 0.0
        return self.position_changes_a_after_b / self.debate_count

    def get_influence(self, from_agent: str) -> float:
        """Get influence score from one agent to the other."""
        if from_agent == self.agent_a:
            return self.influence_a_on_b
        elif from_agent == self.agent_b:
            return self.influence_b_on_a
        return 0.0


class RelationshipTracker:
    """
    Computes relationship metrics from existing debate data.

    Uses EloSystem's matches table and CritiqueStore's critiques table.
    """

    def __init__(
        self,
        elo_db_path: str = DB_ELO_PATH,
    ):
        self.elo_db_path = Path(elo_db_path)
        self.db = EloDatabase(elo_db_path)
        self._init_tables()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with guaranteed cleanup."""
        with self.db.connection() as conn:
            yield conn

    def _init_tables(self) -> None:
        """Add agent_relationships table if not exists."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_relationships (
                    agent_a TEXT NOT NULL,
                    agent_b TEXT NOT NULL,
                    debate_count INTEGER DEFAULT 0,
                    agreement_count INTEGER DEFAULT 0,
                    critique_count_a_to_b INTEGER DEFAULT 0,
                    critique_count_b_to_a INTEGER DEFAULT 0,
                    critique_accepted_a_to_b INTEGER DEFAULT 0,
                    critique_accepted_b_to_a INTEGER DEFAULT 0,
                    position_changes_a_after_b INTEGER DEFAULT 0,
                    position_changes_b_after_a INTEGER DEFAULT 0,
                    a_wins_over_b INTEGER DEFAULT 0,
                    b_wins_over_a INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_a, agent_b),
                    CHECK (agent_a < agent_b)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_a ON agent_relationships(agent_a)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_b ON agent_relationships(agent_b)"
            )
            conn.commit()

    def _canonical_pair(self, agent_a: str, agent_b: str) -> tuple[str, str]:
        """Return agents in canonical order (alphabetical)."""
        if agent_a < agent_b:
            return (agent_a, agent_b)
        return (agent_b, agent_a)

    def update_from_debate(
        self,
        debate_id: str,
        participants: list[str],
        winner: Optional[str],
        votes: dict[str, str],
        critiques: list[dict],
        position_changes: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Update all relationship metrics from a completed debate."""
        with self._get_connection() as conn:
            # Update relationships for each pair
            for i, agent_a in enumerate(participants):
                for agent_b in participants[i + 1 :]:
                    canonical_a, canonical_b = self._canonical_pair(agent_a, agent_b)
                    is_swapped = canonical_a != agent_a

                    # Get or create relationship
                    cursor = conn.execute(
                        "SELECT * FROM agent_relationships WHERE agent_a = ? AND agent_b = ?",
                        (canonical_a, canonical_b),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        conn.execute(
                            "INSERT INTO agent_relationships (agent_a, agent_b) VALUES (?, ?)",
                            (canonical_a, canonical_b),
                        )

                    # Increment debate count
                    conn.execute(
                        """
                        UPDATE agent_relationships
                        SET debate_count = debate_count + 1, updated_at = ?
                        WHERE agent_a = ? AND agent_b = ?
                        """,
                        (datetime.now().isoformat(), canonical_a, canonical_b),
                    )

                    # Check agreement (same vote)
                    if votes.get(agent_a) == votes.get(agent_b):
                        conn.execute(
                            """
                            UPDATE agent_relationships
                            SET agreement_count = agreement_count + 1
                            WHERE agent_a = ? AND agent_b = ?
                            """,
                            (canonical_a, canonical_b),
                        )

                    # Track winner
                    if winner == agent_a:
                        col = "a_wins_over_b" if not is_swapped else "b_wins_over_a"
                        conn.execute(
                            f"""
                            UPDATE agent_relationships
                            SET {col} = {col} + 1
                            WHERE agent_a = ? AND agent_b = ?
                            """,
                            (canonical_a, canonical_b),
                        )
                    elif winner == agent_b:
                        col = "b_wins_over_a" if not is_swapped else "a_wins_over_b"
                        conn.execute(
                            f"""
                            UPDATE agent_relationships
                            SET {col} = {col} + 1
                            WHERE agent_a = ? AND agent_b = ?
                            """,
                            (canonical_a, canonical_b),
                        )

            # Track critiques
            for critique in critiques:
                critic = critique.get("agent") or critique.get("critic")
                target = critique.get("target") or critique.get("target_agent")
                if not critic or not target or critic == target:
                    # Log why critique was dropped for debugging
                    if not critic:
                        logger.debug(
                            f"Critique dropped: missing critic field in debate {debate_id}"
                        )
                    elif not target:
                        logger.debug(
                            f"Critique dropped: missing target field in debate {debate_id}"
                        )
                    elif critic == target:
                        logger.debug(
                            f"Critique dropped: self-critique by {critic} in debate {debate_id}"
                        )
                    continue

                canonical_a, canonical_b = self._canonical_pair(critic, target)
                is_critic_a = canonical_a == critic

                col = "critique_count_a_to_b" if is_critic_a else "critique_count_b_to_a"
                conn.execute(
                    f"""
                    UPDATE agent_relationships
                    SET {col} = {col} + 1
                    WHERE agent_a = ? AND agent_b = ?
                    """,
                    (canonical_a, canonical_b),
                )

            conn.commit()

    def get_relationship(self, agent_a: str, agent_b: str) -> AgentRelationship:
        """Get relationship between two agents."""
        canonical_a, canonical_b = self._canonical_pair(agent_a, agent_b)

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM agent_relationships WHERE agent_a = ? AND agent_b = ?",
                (canonical_a, canonical_b),
            )
            row = cursor.fetchone()

        if row is None:
            return AgentRelationship(agent_a=canonical_a, agent_b=canonical_b)

        return AgentRelationship(
            agent_a=row["agent_a"],
            agent_b=row["agent_b"],
            debate_count=row["debate_count"],
            agreement_count=row["agreement_count"],
            critique_count_a_to_b=row["critique_count_a_to_b"],
            critique_count_b_to_a=row["critique_count_b_to_a"],
            critique_accepted_a_to_b=row["critique_accepted_a_to_b"],
            critique_accepted_b_to_a=row["critique_accepted_b_to_a"],
            position_changes_a_after_b=row["position_changes_a_after_b"],
            position_changes_b_after_a=row["position_changes_b_after_a"],
            a_wins_over_b=row["a_wins_over_b"],
            b_wins_over_a=row["b_wins_over_a"],
            updated_at=row["updated_at"],
        )

    def get_all_relationships(self, agent_name: str) -> list[AgentRelationship]:
        """Get all relationships for an agent."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM agent_relationships
                WHERE agent_a = ? OR agent_b = ?
                ORDER BY debate_count DESC
                """,
                (agent_name, agent_name),
            )
            rows = cursor.fetchall()

        return [
            AgentRelationship(
                agent_a=row["agent_a"],
                agent_b=row["agent_b"],
                debate_count=row["debate_count"],
                agreement_count=row["agreement_count"],
                critique_count_a_to_b=row["critique_count_a_to_b"],
                critique_count_b_to_a=row["critique_count_b_to_a"],
                critique_accepted_a_to_b=row["critique_accepted_a_to_b"],
                critique_accepted_b_to_a=row["critique_accepted_b_to_a"],
                position_changes_a_after_b=row["position_changes_a_after_b"],
                position_changes_b_after_a=row["position_changes_b_after_a"],
                a_wins_over_b=row["a_wins_over_b"],
                b_wins_over_a=row["b_wins_over_a"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_rivals(self, agent_name: str, limit: int = 5) -> list[tuple[str, float]]:
        """Get top rivals by rivalry score."""
        relationships = self.get_all_relationships(agent_name)
        rivals = []

        for rel in relationships:
            other = rel.agent_b if rel.agent_a == agent_name else rel.agent_a
            score = rel.rivalry_score
            if score > 0:
                rivals.append((other, score))

        rivals.sort(key=lambda x: x[1], reverse=True)
        return rivals[:limit]

    def get_allies(self, agent_name: str, limit: int = 5) -> list[tuple[str, float]]:
        """Get top allies by alliance score."""
        relationships = self.get_all_relationships(agent_name)
        allies = []

        for rel in relationships:
            other = rel.agent_b if rel.agent_a == agent_name else rel.agent_a
            score = rel.alliance_score
            if score > 0:
                allies.append((other, score))

        allies.sort(key=lambda x: x[1], reverse=True)
        return allies[:limit]

    def get_influence_network(self, agent_name: str) -> dict[str, list[tuple[str, float]]]:
        """Get who this agent influences and who influences them."""
        relationships = self.get_all_relationships(agent_name)

        influences = []  # Who this agent influences
        influenced_by = []  # Who influences this agent

        for rel in relationships:
            other = rel.agent_b if rel.agent_a == agent_name else rel.agent_a
            is_a = rel.agent_a == agent_name

            if is_a:
                if rel.influence_a_on_b > 0:
                    influences.append((other, rel.influence_a_on_b))
                if rel.influence_b_on_a > 0:
                    influenced_by.append((other, rel.influence_b_on_a))
            else:
                if rel.influence_b_on_a > 0:
                    influences.append((other, rel.influence_b_on_a))
                if rel.influence_a_on_b > 0:
                    influenced_by.append((other, rel.influence_a_on_b))

        influences.sort(key=lambda x: x[1], reverse=True)
        influenced_by.sort(key=lambda x: x[1], reverse=True)

        return {"influences": influences, "influenced_by": influenced_by}
