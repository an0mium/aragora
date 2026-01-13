"""
Truth-Grounded Persona System.

Tracks agent positions across debates and links them to outcomes:
- What did the agent claim?
- Did the agent's position win the debate?
- Was the winning position later verified correct?

This enables persona synthesis from verifiable data rather than self-reported traits.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from aragora.insights.database import InsightsDatabase


@dataclass
class Position:
    """A recorded position/claim by an agent."""

    debate_id: str
    agent_name: str
    position_type: str  # 'proposal', 'vote', 'critique'
    position_text: str
    round_num: int = 0
    confidence: float = 0.5
    was_winning: Optional[bool] = None
    verified_correct: Optional[bool] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_row(cls, row: tuple) -> "Position":
        """Create Position from database row.

        Expected row format:
            (debate_id, agent_name, position_type, position_text, round_num,
             confidence, was_winning, verified_correct, created_at)
        """
        return cls(
            debate_id=row[0],
            agent_name=row[1],
            position_type=row[2],
            position_text=row[3],
            round_num=row[4],
            confidence=row[5],
            was_winning=bool(row[6]) if row[6] is not None else None,
            verified_correct=bool(row[7]) if row[7] is not None else None,
            created_at=row[8],
        )


class PositionTracker:
    """
    Tracks agent positions and links them to outcomes.

    Integrates with existing systems:
    - Arena: Records positions as debates progress
    - EloSystem: Updates ELO based on position accuracy
    - PersonaManager: Feeds accurate data for trait inference
    """

    def __init__(self, db_path: str = "aragora_positions.db"):
        self.db_path = Path(db_path)
        self.db = InsightsDatabase(db_path)

    def record_position(
        self,
        debate_id: str,
        agent_name: str,
        position_type: str,
        position_text: str,
        round_num: int = 0,
        confidence: float = 0.5,
    ) -> Position:
        """Record an agent's position during a debate."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO position_history
                (debate_id, agent_name, position_type, position_text, round_num, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (debate_id, agent_name, position_type, position_text, round_num, confidence),
            )

            conn.commit()

        return Position(
            debate_id=debate_id,
            agent_name=agent_name,
            position_type=position_type,
            position_text=position_text,
            round_num=round_num,
            confidence=confidence,
        )

    def finalize_debate(
        self,
        debate_id: str,
        winning_agent: str,
        winning_position: str,
        consensus_confidence: float,
    ):
        """Mark debate complete and update was_winning_position for all agents."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Record the outcome
            cursor.execute(
                """
                INSERT OR REPLACE INTO debate_outcomes
                (debate_id, winning_agent, winning_position, consensus_confidence)
                VALUES (?, ?, ?, ?)
                """,
                (debate_id, winning_agent, winning_position, consensus_confidence),
            )

            # Mark winning positions
            cursor.execute(
                """
                UPDATE position_history
                SET was_winning_position = (agent_name = ?)
                WHERE debate_id = ?
                """,
                (winning_agent, debate_id),
            )

            conn.commit()

    def record_verification(
        self,
        debate_id: str,
        result: bool,
        source: str = "manual",
    ):
        """Record whether the winning position was actually correct."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Update outcome
            cursor.execute(
                """
                UPDATE debate_outcomes
                SET verified_at = ?, verification_result = ?, verification_source = ?
                WHERE debate_id = ?
                """,
                (datetime.now().isoformat(), 1 if result else 0, source, debate_id),
            )

            # Propagate to positions - winning positions get the verification result
            cursor.execute(
                """
                UPDATE position_history
                SET verified_correct = CASE
                    WHEN was_winning_position = 1 THEN ?
                    ELSE 0
                END
                WHERE debate_id = ?
                """,
                (1 if result else 0, debate_id),
            )

            conn.commit()

    def get_agent_position_accuracy(
        self,
        agent_name: str,
        min_verifications: int = 5,
    ) -> dict:
        """Get position accuracy stats for an agent.

        Returns:
            {
                'total_positions': int,
                'winning_positions': int,
                'verified_positions': int,
                'verified_correct': int,
                'win_rate': float,
                'accuracy_rate': float,
                'calibration': float,
            }
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_winning_position = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN verified_correct IS NOT NULL THEN 1 ELSE 0 END) as verified,
                    SUM(CASE WHEN verified_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_confidence
                FROM position_history
                WHERE agent_name = ? AND position_type = 'vote'
                """,
                (agent_name,),
            )
            row = cursor.fetchone()

        total = row[0] or 0
        wins = row[1] or 0
        verified = row[2] or 0
        correct = row[3] or 0
        avg_confidence = row[4] or 0.5

        win_rate = wins / total if total > 0 else 0.0
        accuracy_rate = correct / verified if verified >= min_verifications else 0.0

        # Calibration: how close is confidence to actual accuracy
        calibration = (
            1.0 - abs(avg_confidence - accuracy_rate) if verified >= min_verifications else 0.0
        )

        return {
            "total_positions": total,
            "winning_positions": wins,
            "verified_positions": verified,
            "verified_correct": correct,
            "win_rate": win_rate,
            "accuracy_rate": accuracy_rate,
            "calibration": calibration,
        }

    def get_position_history(
        self,
        agent_name: str,
        limit: int = 50,
        verified_only: bool = False,
    ) -> list[Position]:
        """Get an agent's position history."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT debate_id, agent_name, position_type, position_text,
                       round_num, confidence, was_winning_position, verified_correct, created_at
                FROM position_history
                WHERE agent_name = ?
            """
            if verified_only:
                query += " AND verified_correct IS NOT NULL"
            query += " ORDER BY created_at DESC LIMIT ?"

            cursor.execute(query, (agent_name, limit))
            rows = cursor.fetchall()

        return [Position.from_row(row) for row in rows]


@dataclass
class TruthGroundedPersona:
    """A persona synthesized from verifiable data."""

    agent_name: str

    # From ELO system
    elo_rating: float = 1500.0
    elo_win_rate: float = 0.0
    elo_calibration: float = 0.0

    # From position tracking
    position_accuracy: float = 0.0
    verified_positions: int = 0
    winning_positions: int = 0
    total_positions: int = 0

    # Synthesized metrics
    overall_reliability: float = 0.0

    # Inferred characteristics
    strength_domains: list[str] = field(default_factory=list)
    weakness_domains: list[str] = field(default_factory=list)

    # Behavioral patterns
    contrarian_score: float = 0.0  # How often they disagree with eventual winners
    early_adopter_score: float = 0.0  # How often first to correct answer


class TruthGroundedLaboratory:
    """
    Main interface for the Emergent Persona Laboratory v2.

    Combines:
    - PositionTracker for position history
    - EloSystem for competitive ranking
    - PersonaManager for trait storage
    """

    def __init__(
        self,
        position_tracker: Optional[PositionTracker] = None,
        elo_system=None,
        persona_manager=None,
        db_path: str = ".nomic/aragora_positions.db",
    ):
        self.position_tracker = position_tracker or PositionTracker(db_path)
        self.elo_system = elo_system
        self.persona_manager = persona_manager

    def synthesize_persona(self, agent_name: str) -> TruthGroundedPersona:
        """
        Synthesize a truth-grounded persona from all available data.

        Combines:
        - ELO rating and win rate
        - Calibration score from predictions
        - Position accuracy from verification
        """
        # Get position accuracy
        pos_stats = self.position_tracker.get_agent_position_accuracy(agent_name)

        # Get ELO data if available
        elo_rating = 1500.0
        elo_win_rate = 0.0
        elo_calibration = 0.0

        if self.elo_system:
            rating = self.elo_system.get_rating(agent_name)
            elo_rating = rating.elo
            elo_win_rate = rating.win_rate
            elo_calibration = rating.calibration_score

        # Calculate overall reliability (weighted combination)
        # 40% ELO performance, 40% position accuracy, 20% calibration
        reliability_components = []
        if pos_stats["total_positions"] > 0:
            reliability_components.append(pos_stats["win_rate"] * 0.4)
        if pos_stats["verified_positions"] >= 5:
            reliability_components.append(pos_stats["accuracy_rate"] * 0.4)
        if elo_calibration > 0:
            reliability_components.append(elo_calibration * 0.2)

        overall_reliability = sum(reliability_components) if reliability_components else 0.0

        return TruthGroundedPersona(
            agent_name=agent_name,
            elo_rating=elo_rating,
            elo_win_rate=elo_win_rate,
            elo_calibration=elo_calibration,
            position_accuracy=pos_stats["accuracy_rate"],
            verified_positions=pos_stats["verified_positions"],
            winning_positions=pos_stats["winning_positions"],
            total_positions=pos_stats["total_positions"],
            overall_reliability=overall_reliability,
        )

    def get_reliable_agents(
        self,
        min_verified: int = 10,
        min_accuracy: float = 0.6,
    ) -> list[TruthGroundedPersona]:
        """Get agents that have demonstrated reliable accuracy."""
        with self.position_tracker.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT agent_name,
                       COUNT(*) as total,
                       SUM(CASE WHEN verified_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM position_history
                WHERE verified_correct IS NOT NULL
                GROUP BY agent_name
                HAVING COUNT(*) >= ?
                """,
                (min_verified,),
            )
            rows = cursor.fetchall()

        reliable = []
        for agent_name, total, correct in rows:
            accuracy = correct / total if total > 0 else 0
            if accuracy >= min_accuracy:
                persona = self.synthesize_persona(agent_name)
                reliable.append(persona)

        return sorted(reliable, key=lambda p: p.overall_reliability, reverse=True)

    def get_all_personas(self, limit: int = 50) -> list[TruthGroundedPersona]:
        """Get all agents with position history."""
        with self.position_tracker.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT DISTINCT agent_name
                FROM position_history
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [self.synthesize_persona(row[0]) for row in rows]

    def get_debate_summary(self, debate_id: str) -> dict:
        """Get summary of positions and outcome for a debate."""
        with self.position_tracker.db.connection() as conn:
            cursor = conn.cursor()

            # Get outcome
            cursor.execute(
                """
                SELECT winning_agent, winning_position, consensus_confidence,
                       verification_result, verification_source
                FROM debate_outcomes
                WHERE debate_id = ?
                """,
                (debate_id,),
            )
            outcome = cursor.fetchone()

            # Get positions
            cursor.execute(
                """
                SELECT agent_name, position_type, position_text, confidence,
                       was_winning_position, verified_correct
                FROM position_history
                WHERE debate_id = ?
                ORDER BY round_num, position_type
                """,
                (debate_id,),
            )
            positions = cursor.fetchall()

        return {
            "debate_id": debate_id,
            "outcome": (
                {
                    "winning_agent": outcome[0] if outcome else None,
                    "winning_position": outcome[1] if outcome else None,
                    "confidence": outcome[2] if outcome else None,
                    "verified": outcome[3] if outcome else None,
                    "verification_source": outcome[4] if outcome else None,
                }
                if outcome
                else None
            ),
            "positions": [
                {
                    "agent": pos[0],
                    "type": pos[1],
                    "text": pos[2][:200],
                    "confidence": pos[3],
                    "was_winning": bool(pos[4]) if pos[4] is not None else None,
                    "verified_correct": bool(pos[5]) if pos[5] is not None else None,
                }
                for pos in positions
            ],
        }
