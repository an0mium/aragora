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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Literal, Optional, Union

from aragora.config import DB_ELO_PATH, DB_PERSONAS_PATH, DB_TIMEOUT_SECONDS
from aragora.insights.database import InsightsDatabase
from aragora.ranking.database import EloDatabase
from .personas import Persona, PersonaManager, EXPERTISE_DOMAINS

# Import from extracted modules for backward compatibility
from .positions import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    PositionLedger,
)
from .relationships import (
    AgentRelationship,
    RelationshipTracker,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "Position",
    "CalibrationBucket",
    "DomainCalibration",
    "AgentRelationship",
    "GroundedPersona",
    "PositionLedger",
    "RelationshipTracker",
    "PersonaSynthesizer",
    "SignificantMoment",
    "MomentDetector",
]


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
            except (KeyError, AttributeError, TypeError, ValueError) as e:
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
            except (KeyError, AttributeError, TypeError, ValueError, ZeroDivisionError) as e:
                logger.debug(f"Failed to load ELO stats for {agent_name}: {e}")

        # Position stats
        if self.position_ledger:
            try:
                stats = self.position_ledger.get_position_stats(agent_name)
                persona.positions_taken = stats.get("total", 0)
                persona.positions_correct = stats.get("correct", 0)
                persona.positions_incorrect = stats.get("incorrect", 0)
                persona.reversals = stats.get("reversals", 0)
            except (KeyError, AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Failed to load position stats for {agent_name}: {e}")

        # Relationships
        if self.relationship_tracker:
            try:
                persona.rivals = self.relationship_tracker.get_rivals(agent_name)
                persona.allies = self.relationship_tracker.get_allies(agent_name)
                influence = self.relationship_tracker.get_influence_network(agent_name)
                persona.influences = influence.get("influences", [])
                persona.influenced_by = influence.get("influenced_by", [])
            except (KeyError, AttributeError, TypeError, ValueError) as e:
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
            top_domains = sorted(persona.domain_elos.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_domains:
                domain_str = ", ".join([f"{d} ({elo:.0f})" for d, elo in top_domains])
                lines.append(f"- Strong domains: {domain_str}")

        return "\n".join(lines)

    def _format_calibration_section(self, persona: GroundedPersona) -> str:
        """Format calibration stats for prompt."""
        lines = ["### Your Calibration"]

        if persona.overall_calibration > 0.5:
            quality = (
                "well-calibrated" if persona.overall_calibration > 0.7 else "reasonably calibrated"
            )
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
        lines.append(
            f"- Correct: {persona.positions_correct}, Incorrect: {persona.positions_incorrect}"
        )

        if persona.reversals > 0:
            lines.append(
                f"- Reversals: {persona.reversals} ({persona.reversal_rate:.0%} reversal rate)"
            )
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
        "upset_victory",  # Low-rated agent beats high-rated
        "position_reversal",  # Agent publicly changes stance
        "calibration_vindication",  # Prediction proven correct
        "alliance_shift",  # Relationship dynamic changes
        "consensus_breakthrough",  # Agreement on contentious issue
        "streak_achievement",  # Win/loss streak milestone
        "domain_mastery",  # Agent becomes top in a domain
    ]
    agent_name: str
    description: str
    significance_score: float  # 0.0-1.0, higher = more significant
    debate_id: Optional[str] = None
    other_agents: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON transmission."""
        return {
            "id": self.id,
            "moment_type": self.moment_type,
            "agent_name": self.agent_name,
            "description": self.description,
            "significance_score": self.significance_score,
            "debate_id": self.debate_id,
            "other_agents": self.other_agents,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


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
        max_moments_per_agent: int = 100,
    ):
        self.elo_system = elo_system
        self.position_ledger = position_ledger
        self.relationship_tracker = relationship_tracker
        self._moment_cache: dict[str, list[SignificantMoment]] = {}
        self._max_moments_per_agent = max_moments_per_agent

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
        except (KeyError, AttributeError, TypeError, ValueError) as e:
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
            except (KeyError, AttributeError, TypeError, ValueError, IndexError) as e:
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

    def _trim_moments(self, agent_name: str) -> None:
        """Trim moments list to max size, keeping most significant."""
        moments = self._moment_cache.get(agent_name, [])
        if len(moments) > self._max_moments_per_agent:
            # Sort by significance (desc), then by recency (desc)
            moments.sort(key=lambda m: (m.significance_score, m.created_at), reverse=True)
            self._moment_cache[agent_name] = moments[: self._max_moments_per_agent]

    def record_moment(self, moment: SignificantMoment) -> None:
        """Record a detected moment."""
        if moment.agent_name not in self._moment_cache:
            self._moment_cache[moment.agent_name] = []

        self._moment_cache[moment.agent_name].append(moment)
        self._trim_moments(moment.agent_name)

        # Also record for other involved agents
        for other in moment.other_agents:
            if other not in self._moment_cache:
                self._moment_cache[other] = []
            self._moment_cache[other].append(moment)
            self._trim_moments(other)

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
