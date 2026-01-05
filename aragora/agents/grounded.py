"""
Grounded Personas - Evidence-based agent identity system.

Tracks agent positions, calibration accuracy, and inter-agent relationships
to generate rich, verifiable identity prompts. Agents earn their reputations
through actual performance, not assigned traits.

Components:
- PositionLedger: Track positions taken with outcomes
- RelationshipTracker: Compute rivalry/alliance from debate history
- PersonaSynthesizer: Generate identity prompts from all data sources
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

from .personas import Persona, PersonaManager, EXPERTISE_DOMAINS

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A position/claim taken by an agent during debate."""

    id: str
    agent_name: str
    claim: str
    confidence: float  # 0.0-1.0 expressed confidence
    debate_id: str
    round_num: int
    outcome: Literal["correct", "incorrect", "unresolved", "pending"] = "pending"
    reversed: bool = False
    reversal_debate_id: Optional[str] = None
    domain: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        agent_name: str,
        claim: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: Optional[str] = None,
    ) -> "Position":
        """Create a new position with generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            claim=claim,
            confidence=max(0.0, min(1.0, confidence)),
            debate_id=debate_id,
            round_num=round_num,
            domain=domain,
        )


@dataclass
class CalibrationBucket:
    """Calibration statistics for a confidence range."""

    bucket_start: float  # e.g., 0.8
    bucket_end: float  # e.g., 0.9
    predictions: int = 0
    correct: int = 0
    brier_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        """Actual accuracy in this bucket."""
        return self.correct / self.predictions if self.predictions > 0 else 0.0

    @property
    def expected_accuracy(self) -> float:
        """Expected accuracy (midpoint of bucket)."""
        return (self.bucket_start + self.bucket_end) / 2

    @property
    def calibration_error(self) -> float:
        """Expected calibration error for this bucket."""
        return abs(self.accuracy - self.expected_accuracy)

    @property
    def bucket_key(self) -> str:
        """Key for storage (e.g., '0.8-0.9')."""
        return f"{self.bucket_start:.1f}-{self.bucket_end:.1f}"


@dataclass
class DomainCalibration:
    """Per-domain calibration tracking."""

    domain: str
    total_predictions: int = 0
    total_correct: int = 0
    brier_sum: float = 0.0
    buckets: dict[str, CalibrationBucket] = field(default_factory=dict)

    @property
    def calibration_score(self) -> float:
        """1 - avg_brier_score (higher is better)."""
        if self.total_predictions == 0:
            return 0.5
        return 1.0 - (self.brier_sum / self.total_predictions)

    @property
    def accuracy(self) -> float:
        """Overall accuracy in this domain."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions


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
        competitiveness = (
            1 - abs(self.a_wins_over_b - self.b_wins_over_a) / max(total_wins, 1)
        )
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


@dataclass
class GroundedPersona:
    """Full grounded persona with all evidence-based attributes."""

    agent_name: str
    # From PersonaManager
    base_persona: Optional[Persona] = None
    # From EloSystem
    elo: float = 1500.0
    domain_elos: dict[str, float] = field(default_factory=dict)
    win_rate: float = 0.0
    games_played: int = 0
    # From PositionLedger
    positions_taken: int = 0
    positions_correct: int = 0
    positions_incorrect: int = 0
    reversals: int = 0
    # Calibration
    overall_calibration: float = 0.5
    domain_calibrations: dict[str, DomainCalibration] = field(default_factory=dict)
    # Relationships
    rivals: list[tuple[str, float]] = field(default_factory=list)
    allies: list[tuple[str, float]] = field(default_factory=list)
    influences: list[tuple[str, float]] = field(default_factory=list)
    influenced_by: list[tuple[str, float]] = field(default_factory=list)
    # Meta
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def reversal_rate(self) -> float:
        """Rate of position reversals."""
        if self.positions_taken == 0:
            return 0.0
        return self.reversals / self.positions_taken

    @property
    def position_accuracy(self) -> float:
        """Accuracy of resolved positions."""
        resolved = self.positions_correct + self.positions_incorrect
        if resolved == 0:
            return 0.0
        return self.positions_correct / resolved


class PositionLedger:
    """
    Tracks every position an agent takes across debates.

    Integrates with PersonaManager's database for unified agent data.
    """

    def __init__(self, db_path: str = "aragora_personas.db"):
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        """Add positions table if not exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                claim TEXT NOT NULL,
                confidence REAL NOT NULL,
                debate_id TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                outcome TEXT DEFAULT 'pending',
                reversed INTEGER DEFAULT 0,
                reversal_debate_id TEXT,
                domain TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_debate ON positions(debate_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_outcome ON positions(outcome)"
        )
        conn.commit()
        conn.close()

    def record_position(
        self,
        agent_name: str,
        claim: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: Optional[str] = None,
    ) -> str:
        """Record a new position. Returns position ID."""
        position = Position.create(
            agent_name=agent_name,
            claim=claim,
            confidence=confidence,
            debate_id=debate_id,
            round_num=round_num,
            domain=domain,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO positions
            (id, agent_name, claim, confidence, debate_id, round_num, domain, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.id,
                position.agent_name,
                position.claim,
                position.confidence,
                position.debate_id,
                position.round_num,
                position.domain,
                position.created_at,
            ),
        )
        conn.commit()
        conn.close()

        return position.id

    def resolve_position(
        self,
        position_id: str,
        outcome: Literal["correct", "incorrect", "unresolved"],
    ) -> None:
        """Mark a position's outcome after debate conclusion."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE positions
            SET outcome = ?, resolved_at = ?
            WHERE id = ?
            """,
            (outcome, datetime.now().isoformat(), position_id),
        )
        conn.commit()
        conn.close()

    def record_reversal(
        self,
        agent_name: str,
        original_position_id: str,
        new_debate_id: str,
    ) -> None:
        """Record when agent reverses a previous position."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE positions
            SET reversed = 1, reversal_debate_id = ?
            WHERE id = ? AND agent_name = ?
            """,
            (new_debate_id, original_position_id, agent_name),
        )
        conn.commit()
        conn.close()

    def get_agent_positions(
        self,
        agent_name: str,
        limit: int = 100,
        outcome_filter: Optional[str] = None,
    ) -> list[Position]:
        """Get positions for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM positions WHERE agent_name = ?"
        params: list = [agent_name]

        if outcome_filter:
            query += " AND outcome = ?"
            params.append(outcome_filter)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            Position(
                id=row["id"],
                agent_name=row["agent_name"],
                claim=row["claim"],
                confidence=row["confidence"],
                debate_id=row["debate_id"],
                round_num=row["round_num"],
                outcome=row["outcome"],
                reversed=bool(row["reversed"]),
                reversal_debate_id=row["reversal_debate_id"],
                domain=row["domain"],
                created_at=row["created_at"],
                resolved_at=row["resolved_at"],
            )
            for row in rows
        ]

    def get_position_stats(self, agent_name: str) -> dict:
        """Get aggregate position statistics."""
        conn = sqlite3.connect(self.db_path)

        # Total positions
        cursor = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE agent_name = ?", (agent_name,)
        )
        total = cursor.fetchone()[0]

        # By outcome
        cursor = conn.execute(
            """
            SELECT outcome, COUNT(*)
            FROM positions
            WHERE agent_name = ?
            GROUP BY outcome
            """,
            (agent_name,),
        )
        outcomes = dict(cursor.fetchall())

        # Reversals
        cursor = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE agent_name = ? AND reversed = 1",
            (agent_name,),
        )
        reversals = cursor.fetchone()[0]

        # Average confidence by outcome
        cursor = conn.execute(
            """
            SELECT outcome, AVG(confidence)
            FROM positions
            WHERE agent_name = ? AND outcome != 'pending'
            GROUP BY outcome
            """,
            (agent_name,),
        )
        avg_confidence = dict(cursor.fetchall())

        conn.close()

        return {
            "total": total,
            "correct": outcomes.get("correct", 0),
            "incorrect": outcomes.get("incorrect", 0),
            "unresolved": outcomes.get("unresolved", 0),
            "pending": outcomes.get("pending", 0),
            "reversals": reversals,
            "avg_confidence_when_correct": avg_confidence.get("correct", 0),
            "avg_confidence_when_incorrect": avg_confidence.get("incorrect", 0),
        }

    def get_positions_for_debate(self, debate_id: str) -> list[Position]:
        """Get all positions from a specific debate."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            "SELECT * FROM positions WHERE debate_id = ? ORDER BY round_num, agent_name",
            (debate_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            Position(
                id=row["id"],
                agent_name=row["agent_name"],
                claim=row["claim"],
                confidence=row["confidence"],
                debate_id=row["debate_id"],
                round_num=row["round_num"],
                outcome=row["outcome"],
                reversed=bool(row["reversed"]),
                reversal_debate_id=row["reversal_debate_id"],
                domain=row["domain"],
                created_at=row["created_at"],
                resolved_at=row["resolved_at"],
            )
            for row in rows
        ]

    def detect_domain(self, content: str) -> Optional[str]:
        """Detect expertise domain from content using keywords."""
        content_lower = content.lower()

        domain_keywords = {
            "security": ["security", "vulnerability", "auth", "xss", "injection", "csrf"],
            "performance": ["performance", "optimization", "latency", "throughput", "cache"],
            "architecture": ["architecture", "design pattern", "modularity", "coupling"],
            "testing": ["test", "coverage", "mock", "fixture", "assertion"],
            "error_handling": ["error", "exception", "fallback", "retry", "graceful"],
            "concurrency": ["async", "parallel", "thread", "race condition", "deadlock"],
            "api_design": ["api", "endpoint", "rest", "graphql", "interface"],
            "database": ["database", "query", "index", "transaction", "schema"],
            "frontend": ["ui", "component", "render", "css", "responsive"],
            "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return domain

        return None


class RelationshipTracker:
    """
    Computes relationship metrics from existing debate data.

    Uses EloSystem's matches table and CritiqueStore's critiques table.
    """

    def __init__(
        self,
        elo_db_path: str = "aragora_elo.db",
    ):
        self.elo_db_path = Path(elo_db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        """Add agent_relationships table if not exists."""
        conn = sqlite3.connect(self.elo_db_path)
        conn.execute("""
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
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_a ON agent_relationships(agent_a)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_b ON agent_relationships(agent_b)"
        )
        conn.commit()
        conn.close()

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
        conn = sqlite3.connect(self.elo_db_path)

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
        conn.close()

    def get_relationship(self, agent_a: str, agent_b: str) -> AgentRelationship:
        """Get relationship between two agents."""
        canonical_a, canonical_b = self._canonical_pair(agent_a, agent_b)

        conn = sqlite3.connect(self.elo_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM agent_relationships WHERE agent_a = ? AND agent_b = ?",
            (canonical_a, canonical_b),
        )
        row = cursor.fetchone()
        conn.close()

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
        conn = sqlite3.connect(self.elo_db_path)
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
        conn.close()

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

    def get_influence_network(
        self, agent_name: str
    ) -> dict[str, list[tuple[str, float]]]:
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


class PersonaSynthesizer:
    """
    Generates rich identity prompts from all data sources.

    Combines:
    - PersonaManager: base traits, expertise
    - EloSystem: performance stats, calibration
    - PositionLedger: position history, reversals
    - RelationshipTracker: rivalries, alliances, influence
    """

    def __init__(
        self,
        persona_manager: Optional[PersonaManager] = None,
        elo_system=None,  # Optional[EloSystem] - avoid circular import
        position_ledger: Optional[PositionLedger] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
    ):
        self.persona_manager = persona_manager
        self.elo_system = elo_system
        self.position_ledger = position_ledger
        self.relationship_tracker = relationship_tracker

    def get_grounded_persona(self, agent_name: str) -> GroundedPersona:
        """Build complete grounded persona from all sources."""
        persona = GroundedPersona(agent_name=agent_name)

        # Base persona
        if self.persona_manager:
            try:
                base = self.persona_manager.get_persona(agent_name)
                if base:
                    persona.base_persona = base
            except Exception as e:
                logger.debug(f"Failed to load base persona for {agent_name}: {e}")

        # ELO stats
        if self.elo_system:
            try:
                rating = self.elo_system.get_rating(agent_name)
                if rating:
                    persona.elo = rating.elo
                    persona.domain_elos = rating.domain_elos or {}
                    total_games = rating.wins + rating.losses + rating.draws
                    persona.games_played = total_games
                    if total_games > 0:
                        persona.win_rate = rating.wins / total_games

                    # Calibration
                    if hasattr(rating, "calibration_score"):
                        persona.overall_calibration = rating.calibration_score
            except Exception as e:
                logger.debug(f"Failed to load ELO stats for {agent_name}: {e}")

        # Position stats
        if self.position_ledger:
            try:
                stats = self.position_ledger.get_position_stats(agent_name)
                persona.positions_taken = stats.get("total", 0)
                persona.positions_correct = stats.get("correct", 0)
                persona.positions_incorrect = stats.get("incorrect", 0)
                persona.reversals = stats.get("reversals", 0)
            except Exception as e:
                logger.debug(f"Failed to load position stats for {agent_name}: {e}")

        # Relationships
        if self.relationship_tracker:
            try:
                persona.rivals = self.relationship_tracker.get_rivals(agent_name)
                persona.allies = self.relationship_tracker.get_allies(agent_name)
                influence = self.relationship_tracker.get_influence_network(agent_name)
                persona.influences = influence.get("influences", [])
                persona.influenced_by = influence.get("influenced_by", [])
            except Exception as e:
                logger.debug(f"Failed to load relationships for {agent_name}: {e}")

        return persona

    def synthesize_identity_prompt(
        self,
        agent_name: str,
        context: Optional[str] = None,
        opponent_names: Optional[list[str]] = None,
        include_sections: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a rich identity prompt for an agent.

        The prompt grounds the agent in their actual track record.
        """
        if include_sections is None:
            include_sections = ["performance", "calibration", "relationships", "positions"]

        persona = self.get_grounded_persona(agent_name)
        sections = []

        # Header
        sections.append(f"## Your Identity: {agent_name}")

        # Base persona context
        if persona.base_persona:
            base_context = persona.base_persona.to_prompt_context()
            if base_context:
                sections.append(base_context)

        # Performance section
        if "performance" in include_sections and persona.games_played > 0:
            sections.append(self._format_performance_section(persona))

        # Calibration section
        if "calibration" in include_sections:
            sections.append(self._format_calibration_section(persona))

        # Relationships section
        if "relationships" in include_sections:
            rel_section = self._format_relationship_section(persona, opponent_names)
            if rel_section:
                sections.append(rel_section)

        # Position history section
        if "positions" in include_sections and persona.positions_taken > 0:
            sections.append(self._format_position_history_section(persona))

        return "\n\n".join(sections)

    def _format_performance_section(self, persona: GroundedPersona) -> str:
        """Format performance stats for prompt."""
        lines = ["### Your Track Record"]
        lines.append(f"- ELO Rating: {persona.elo:.0f}")
        lines.append(f"- Win Rate: {persona.win_rate:.0%} ({persona.games_played} debates)")

        if persona.domain_elos:
            top_domains = sorted(
                persona.domain_elos.items(), key=lambda x: x[1], reverse=True
            )[:3]
            if top_domains:
                domain_str = ", ".join(
                    [f"{d} ({elo:.0f})" for d, elo in top_domains]
                )
                lines.append(f"- Strong domains: {domain_str}")

        return "\n".join(lines)

    def _format_calibration_section(self, persona: GroundedPersona) -> str:
        """Format calibration stats for prompt."""
        lines = ["### Your Calibration"]

        if persona.overall_calibration > 0.5:
            quality = "well-calibrated" if persona.overall_calibration > 0.7 else "reasonably calibrated"
            lines.append(f"- You are {quality} (score: {persona.overall_calibration:.2f})")
        else:
            lines.append("- Calibration data still accumulating")

        if persona.positions_taken > 0:
            accuracy = persona.position_accuracy
            lines.append(f"- Position accuracy: {accuracy:.0%} of resolved positions")

        return "\n".join(lines)

    def _format_relationship_section(
        self,
        persona: GroundedPersona,
        opponent_names: Optional[list[str]] = None,
    ) -> str:
        """Format relationship info, highlighting current opponents."""
        lines = ["### Your Relationships"]

        # Highlight current opponents
        if opponent_names and self.relationship_tracker:
            for opp in opponent_names:
                rel = self.relationship_tracker.get_relationship(persona.agent_name, opp)
                if rel.debate_count > 0:
                    desc = []
                    if rel.rivalry_score > 0.3:
                        desc.append("rival")
                    if rel.alliance_score > 0.5:
                        desc.append("frequent ally")

                    # Win rate against this opponent
                    is_a = rel.agent_a == persona.agent_name
                    my_wins = rel.a_wins_over_b if is_a else rel.b_wins_over_a
                    their_wins = rel.b_wins_over_a if is_a else rel.a_wins_over_b
                    if my_wins + their_wins > 0:
                        wr = my_wins / (my_wins + their_wins)
                        desc.append(f"{wr:.0%} win rate in {rel.debate_count} debates")

                    if desc:
                        lines.append(f"- vs {opp}: {', '.join(desc)}")

        # General rivals/allies
        if persona.rivals:
            rival_str = ", ".join([f"{r[0]} ({r[1]:.2f})" for r in persona.rivals[:3]])
            lines.append(f"- Rivals: {rival_str}")

        if persona.allies:
            ally_str = ", ".join([f"{a[0]} ({a[1]:.2f})" for a in persona.allies[:3]])
            lines.append(f"- Allies: {ally_str}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_position_history_section(self, persona: GroundedPersona) -> str:
        """Format position history for prompt."""
        lines = ["### Your Position History"]
        lines.append(f"- Positions taken: {persona.positions_taken}")
        lines.append(f"- Correct: {persona.positions_correct}, Incorrect: {persona.positions_incorrect}")

        if persona.reversals > 0:
            lines.append(f"- Reversals: {persona.reversals} ({persona.reversal_rate:.0%} reversal rate)")
            lines.append("  (Reversals indicate intellectual flexibility when evidence warrants)")

        return "\n".join(lines)

    def get_opponent_briefing(self, agent_name: str, opponent_name: str) -> str:
        """Get specific briefing about an opponent."""
        if not self.relationship_tracker:
            return ""

        rel = self.relationship_tracker.get_relationship(agent_name, opponent_name)
        if rel.debate_count == 0:
            return f"You have not debated {opponent_name} before."

        lines = [f"### Briefing: {opponent_name}"]
        lines.append(f"- Previous debates: {rel.debate_count}")
        lines.append(f"- Agreement rate: {rel.agreement_count / rel.debate_count:.0%}")

        is_a = rel.agent_a == agent_name
        my_wins = rel.a_wins_over_b if is_a else rel.b_wins_over_a
        their_wins = rel.b_wins_over_a if is_a else rel.a_wins_over_b

        if my_wins + their_wins > 0:
            lines.append(f"- Your record: {my_wins}-{their_wins}")

        if rel.rivalry_score > 0.3:
            lines.append(f"- This is a significant rivalry (score: {rel.rivalry_score:.2f})")

        return "\n".join(lines)


@dataclass
class SignificantMoment:
    """A significant narrative event in an agent's debate history."""

    id: str
    moment_type: Literal[
        "upset_victory",       # Low-rated agent beats high-rated
        "position_reversal",   # Agent publicly changes stance
        "calibration_vindication",  # Prediction proven correct
        "alliance_shift",      # Relationship dynamic changes
        "consensus_breakthrough",  # Agreement on contentious issue
        "streak_achievement",  # Win/loss streak milestone
        "domain_mastery",      # Agent becomes top in a domain
    ]
    agent_name: str
    description: str
    significance_score: float  # 0.0-1.0, higher = more significant
    debate_id: Optional[str] = None
    other_agents: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MomentDetector:
    """
    Detects genuinely significant narrative moments from debate history.

    Part of the Emergent Persona Laboratory - identifies moments that
    define an agent's identity through actual performance, not manufactured drama.

    Significant moments include:
    - Upset victories (underdog wins)
    - Position reversals (changing stance with evidence)
    - Calibration vindications (predictions proven right)
    - Alliance shifts (relationship changes)
    - Consensus breakthroughs (resolving disagreements)
    """

    def __init__(
        self,
        elo_system=None,
        position_ledger: Optional[PositionLedger] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
    ):
        self.elo_system = elo_system
        self.position_ledger = position_ledger
        self.relationship_tracker = relationship_tracker
        self._moment_cache: dict[str, list[SignificantMoment]] = {}

    def detect_upset_victory(
        self,
        winner: str,
        loser: str,
        debate_id: str,
    ) -> Optional[SignificantMoment]:
        """Detect if a match result is a significant upset."""
        if not self.elo_system:
            return None

        try:
            winner_rating = self.elo_system.get_rating(winner)
            loser_rating = self.elo_system.get_rating(loser)

            elo_diff = loser_rating.elo - winner_rating.elo

            # Significant upset: winner was 100+ ELO below loser
            if elo_diff >= 100:
                # Scale significance by ELO difference
                significance = min(1.0, elo_diff / 300)

                return SignificantMoment(
                    id=str(uuid.uuid4())[:8],
                    moment_type="upset_victory",
                    agent_name=winner,
                    description=f"{winner} defeated {loser} despite being {elo_diff:.0f} ELO lower",
                    significance_score=significance,
                    debate_id=debate_id,
                    other_agents=[loser],
                    metadata={
                        "winner_elo": winner_rating.elo,
                        "loser_elo": loser_rating.elo,
                        "elo_difference": elo_diff,
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to detect upset victory for {winner} vs {loser}: {e}")

        return None

    def detect_position_reversal(
        self,
        agent_name: str,
        original_position: Position,
        new_position: Position,
        debate_id: str,
    ) -> Optional[SignificantMoment]:
        """Detect when an agent reverses a significant position."""
        if not original_position.reversed:
            return None

        # Calculate significance based on original confidence
        # Higher confidence reversal = more significant
        significance = original_position.confidence * 0.8

        # Bonus if the original position was later proven wrong
        if original_position.outcome == "incorrect":
            significance = min(1.0, significance + 0.2)

        return SignificantMoment(
            id=str(uuid.uuid4())[:8],
            moment_type="position_reversal",
            agent_name=agent_name,
            description=f"{agent_name} reversed position on '{original_position.claim[:50]}...' (was {original_position.confidence:.0%} confident)",
            significance_score=significance,
            debate_id=debate_id,
            metadata={
                "original_position_id": original_position.id,
                "original_confidence": original_position.confidence,
                "original_outcome": original_position.outcome,
            },
        )

    def detect_calibration_vindication(
        self,
        agent_name: str,
        prediction_confidence: float,
        was_correct: bool,
        domain: str,
        debate_id: str,
    ) -> Optional[SignificantMoment]:
        """Detect when a high-confidence prediction is vindicated."""
        if not was_correct or prediction_confidence < 0.85:
            return None

        # High-confidence correct prediction is significant
        significance = (prediction_confidence - 0.5) * 2  # Scale 0.5-1.0 to 0.0-1.0

        return SignificantMoment(
            id=str(uuid.uuid4())[:8],
            moment_type="calibration_vindication",
            agent_name=agent_name,
            description=f"{agent_name}'s {prediction_confidence:.0%} confidence prediction in {domain} was correct",
            significance_score=significance,
            debate_id=debate_id,
            metadata={
                "prediction_confidence": prediction_confidence,
                "domain": domain,
            },
        )

    def detect_streak_achievement(
        self,
        agent_name: str,
        streak_type: Literal["win", "loss"],
        streak_length: int,
        debate_id: str,
    ) -> Optional[SignificantMoment]:
        """Detect significant win/loss streaks."""
        # Minimum 5 for significance
        if streak_length < 5:
            return None

        # Scale significance: 5 = 0.5, 10 = 1.0
        significance = min(1.0, streak_length / 10)

        if streak_type == "win":
            description = f"{agent_name} achieves {streak_length}-debate winning streak"
        else:
            description = f"{agent_name} faces {streak_length}-debate losing streak"

        return SignificantMoment(
            id=str(uuid.uuid4())[:8],
            moment_type="streak_achievement",
            agent_name=agent_name,
            description=description,
            significance_score=significance,
            debate_id=debate_id,
            metadata={
                "streak_type": streak_type,
                "streak_length": streak_length,
            },
        )

    def detect_domain_mastery(
        self,
        agent_name: str,
        domain: str,
        rank: int,
        elo: float,
    ) -> Optional[SignificantMoment]:
        """Detect when an agent becomes top-ranked in a domain."""
        if rank != 1:
            return None

        return SignificantMoment(
            id=str(uuid.uuid4())[:8],
            moment_type="domain_mastery",
            agent_name=agent_name,
            description=f"{agent_name} becomes #1 in {domain} domain with {elo:.0f} ELO",
            significance_score=0.9,  # Reaching #1 is always significant
            metadata={
                "domain": domain,
                "elo": elo,
            },
        )

    def detect_consensus_breakthrough(
        self,
        agents: list[str],
        topic: str,
        confidence: float,
        debate_id: str,
    ) -> Optional[SignificantMoment]:
        """Detect when opposing agents reach consensus."""
        if len(agents) < 2 or confidence < 0.7:
            return None

        # Check if these agents have been rivals
        rivalry_score = 0.0
        if self.relationship_tracker and len(agents) >= 2:
            try:
                rel = self.relationship_tracker.get_relationship(agents[0], agents[1])
                rivalry_score = rel.rivalry_score
            except Exception as e:
                logger.debug(f"Failed to get rivalry score for {agents[0]}, {agents[1]}: {e}")

        # Consensus between rivals is more significant
        base_significance = confidence * 0.6
        if rivalry_score > 0.3:
            base_significance += rivalry_score * 0.4

        significance = min(1.0, base_significance)

        return SignificantMoment(
            id=str(uuid.uuid4())[:8],
            moment_type="consensus_breakthrough",
            agent_name=agents[0],  # First agent as primary
            description=f"Consensus reached on '{topic[:50]}...' with {confidence:.0%} confidence",
            significance_score=significance,
            debate_id=debate_id,
            other_agents=agents[1:],
            metadata={
                "topic": topic,
                "confidence": confidence,
                "rivalry_score": rivalry_score,
                "participants": agents,
            },
        )

    def get_agent_moments(
        self,
        agent_name: str,
        limit: int = 10,
        moment_types: Optional[list[str]] = None,
    ) -> list[SignificantMoment]:
        """Get significant moments for an agent."""
        moments = self._moment_cache.get(agent_name, [])

        if moment_types:
            moments = [m for m in moments if m.moment_type in moment_types]

        # Sort by significance and recency
        moments.sort(key=lambda m: (m.significance_score, m.created_at), reverse=True)

        return moments[:limit]

    def record_moment(self, moment: SignificantMoment):
        """Record a detected moment."""
        if moment.agent_name not in self._moment_cache:
            self._moment_cache[moment.agent_name] = []

        self._moment_cache[moment.agent_name].append(moment)

        # Also record for other involved agents
        for other in moment.other_agents:
            if other not in self._moment_cache:
                self._moment_cache[other] = []
            self._moment_cache[other].append(moment)

    def format_moment_narrative(self, moment: SignificantMoment) -> str:
        """Format a moment as a narrative string for prompts."""
        significance_labels = {
            (0.0, 0.3): "notable",
            (0.3, 0.6): "significant",
            (0.6, 0.8): "major",
            (0.8, 1.0): "defining",
        }

        label = "notable"
        for (low, high), lbl in significance_labels.items():
            if low <= moment.significance_score < high:
                label = lbl
                break

        return f"**{label.title()} Moment**: {moment.description}"

    def get_narrative_summary(self, agent_name: str, limit: int = 5) -> str:
        """Get a narrative summary of an agent's significant moments."""
        moments = self.get_agent_moments(agent_name, limit=limit)

        if not moments:
            return f"{agent_name} has not yet established defining moments."

        lines = [f"### {agent_name}'s Defining Moments"]
        for moment in moments:
            lines.append(f"- {self.format_moment_narrative(moment)}")

        return "\n".join(lines)
