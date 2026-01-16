"""
Grounded operations for debate verdicts and agent relationships.

Extracted from Arena to improve code organization. Handles:
- Recording grounded positions to the PositionLedger
- Updating agent relationships after debate completion
- Creating grounded verdicts with evidence analysis
- Formally verifying claims using Z3 SMT solver
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.agents.positions import PositionLedger
    from aragora.core import DebateResult, Vote
    from aragora.ranking.elo import EloSystem
    from aragora.reasoning.evidence_grounding import EvidenceGrounder

logger = logging.getLogger(__name__)


class GroundedOperations:
    """Manages grounded verdicts, positions, and agent relationships.

    Extracted from Arena to centralize evidence-grounding and relationship
    tracking operations.

    Usage:
        ops = GroundedOperations(
            position_ledger=ledger,
            elo_system=elo,
            evidence_grounder=grounder,
        )
        ops.record_position(agent_name, content, debate_id, round_num)
        ops.update_relationships(debate_id, participants, winner, votes)
        verdict = ops.create_grounded_verdict(result)
    """

    def __init__(
        self,
        position_ledger: Optional["PositionLedger"] = None,
        elo_system: Optional["EloSystem"] = None,
        evidence_grounder: Optional["EvidenceGrounder"] = None,
    ) -> None:
        """Initialize grounded operations.

        Args:
            position_ledger: Optional ledger for recording agent positions
            elo_system: Optional ELO system for relationship tracking
            evidence_grounder: Optional grounder for verdict creation
        """
        self.position_ledger = position_ledger
        self.elo_system = elo_system
        self.evidence_grounder = evidence_grounder

    def record_position(
        self,
        agent_name: str,
        content: str,
        debate_id: str,
        round_num: int,
        confidence: float = 0.7,
        domain: Optional[str] = None,
    ) -> None:
        """Record a position to the grounded persona ledger.

        Args:
            agent_name: Name of the agent taking the position
            content: The position content (truncated to 1000 chars)
            debate_id: ID of the current debate
            round_num: Current round number
            confidence: Position confidence (default 0.7)
            domain: Optional domain category for the position
        """
        if not self.position_ledger:
            return
        try:
            self.position_ledger.record_position(
                agent_name=agent_name,
                claim=content[:1000],
                confidence=confidence,
                debate_id=debate_id,
                round_num=round_num,
                domain=domain,
            )
        except (AttributeError, TypeError, ValueError) as e:
            # Expected parameter or state errors
            logger.warning(f"Position ledger error: {e}")
        except (KeyError, RuntimeError, OSError) as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Position ledger error (type={type(e).__name__}): {e}")

    def update_relationships(
        self,
        debate_id: str,
        participants: list[str],
        winner: Optional[str],
        votes: list["Vote"],
    ) -> None:
        """Update agent relationships after debate completion.

        Uses batch update for O(1) database connections instead of O(n²)
        for n participants.

        Args:
            debate_id: ID of the completed debate
            participants: List of participant agent names
            winner: Name of the winning agent, if any
            votes: List of votes cast during the debate
        """
        if not self.elo_system:
            return
        try:
            vote_choices = {
                v.agent: v.choice for v in votes if hasattr(v, "agent") and hasattr(v, "choice")
            }
            # Build batch of relationship updates
            updates = []
            for i, agent_a in enumerate(participants):
                for agent_b in participants[i + 1 :]:
                    agreed = (
                        agent_a in vote_choices
                        and agent_b in vote_choices
                        and vote_choices[agent_a] == vote_choices[agent_b]
                    )
                    a_win = 1 if winner == agent_a else 0
                    b_win = 1 if winner == agent_b else 0
                    updates.append(
                        {
                            "agent_a": agent_a,
                            "agent_b": agent_b,
                            "debate_increment": 1,
                            "agreement_increment": 1 if agreed else 0,
                            "a_win": a_win,
                            "b_win": b_win,
                        }
                    )
            # Single transaction for all updates
            self.elo_system.update_relationships_batch(updates)
        except (AttributeError, TypeError, KeyError) as e:
            # Expected data access errors
            logger.warning(f"Relationship update error: {e}")
        except (ValueError, RuntimeError, OSError) as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Relationship update error (type={type(e).__name__}): {e}")

    def create_grounded_verdict(self, result: "DebateResult") -> Any:
        """Create a GroundedVerdict for the final answer.

        Heavy3-inspired: Wrap final answers with evidence grounding analysis.

        Args:
            result: The debate result with final_answer and confidence

        Returns:
            GroundedVerdict instance or None if no final answer
        """
        if not result.final_answer:
            return None

        if not self.evidence_grounder:
            return None

        return self.evidence_grounder.create_grounded_verdict(
            final_answer=result.final_answer,
            confidence=result.confidence,
        )

    async def verify_claims_formally(self, result: "DebateResult") -> None:
        """Verify decidable claims using Z3 SMT solver.

        For arithmetic, logic, and constraint claims, attempts formal
        verification to provide machine-verified evidence.

        Args:
            result: The debate result with grounded_verdict to verify
        """
        if not result.grounded_verdict:
            return

        if not self.evidence_grounder:
            return

        await self.evidence_grounder.verify_claims_formally(result.grounded_verdict)


__all__ = ["GroundedOperations"]
