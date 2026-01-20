"""
Debate lifecycle hooks for post-debate processing.

This module extracts post-debate callback logic from the Arena class,
following the Single Responsibility Principle. Hooks are triggered
after key debate events (round completion, debate completion).

Usage:
    # Create hooks with optional subsystems
    hooks = DebateHooks(
        position_ledger=ledger,
        elo_system=elo,
        memory_manager=memory_mgr,
    )

    # After round completion
    hooks.on_round_complete(ctx, round_num, proposals, votes)

    # After debate completion
    await hooks.on_debate_complete(ctx, result, agents)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.positions import PositionLedger
    from aragora.core import Agent, DebateResult, Vote
    from aragora.debate.context import DebateContext
    from aragora.debate.event_emission import EventEmitter
    from aragora.debate.memory_manager import MemoryManager
    from aragora.ranking.elo import EloSystem
    from aragora.reasoning.evidence_grounding import EvidenceGrounder

logger = logging.getLogger(__name__)


@dataclass
class DebateHooks:
    """Lifecycle hooks for debate events.

    Provides a centralized place for post-debate processing callbacks:

    **Round Hooks:**
    - Record agent positions to ledger
    - Detect significant moments
    - Track round-level analytics

    **Debate Completion Hooks:**
    - Update agent relationships in ELO system
    - Store debate outcome in memory
    - Update retrieved memory outcomes
    - Trigger formal verification of claims
    - Update calibration scores and emit events

    All hooks are optional and fail gracefully - errors don't break the debate.
    """

    # Subsystem references
    position_ledger: Optional["PositionLedger"] = None
    elo_system: Optional["EloSystem"] = None
    memory_manager: Optional["MemoryManager"] = None
    evidence_grounder: Optional["EvidenceGrounder"] = None
    calibration_tracker: Optional["CalibrationTracker"] = None
    event_emitter: Optional["EventEmitter"] = None

    # Tracking state
    _continuum_retrieved_ids: List[str] = field(default_factory=list)
    _continuum_retrieved_tiers: dict = field(default_factory=dict)

    # =========================================================================
    # Round Hooks
    # =========================================================================

    def on_round_complete(
        self,
        ctx: "DebateContext",
        round_num: int,
        proposals: dict[str, str],
        domain: Optional[str] = None,
    ) -> None:
        """Called when a debate round completes.

        Records agent positions to the position ledger for tracking
        stance changes and building grounded personas.

        Args:
            ctx: The debate context
            round_num: The round number that completed (1-indexed)
            proposals: Agent name -> proposal content mapping
            domain: Optional domain classification for the debate
        """
        if not self.position_ledger:
            return

        for agent_name, content in proposals.items():
            self._record_position(
                agent_name=agent_name,
                content=content,
                debate_id=ctx.debate_id,
                round_num=round_num,
                domain=domain,
            )

    def _record_position(
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
            agent_name: Name of the agent
            content: The position/proposal content
            debate_id: ID of the current debate
            round_num: Round number (1-indexed)
            confidence: Confidence score for the position
            domain: Optional domain classification
        """
        if not self.position_ledger:
            return

        try:
            self.position_ledger.record_position(
                agent_name=agent_name,
                claim=content[:1000],  # Truncate for storage
                confidence=confidence,
                debate_id=debate_id,
                round_num=round_num,
                domain=domain,
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Position ledger error: {e}")
        except (KeyError, RuntimeError, OSError) as e:
            logger.warning(f"Position ledger error (type={type(e).__name__}): {e}")

    # =========================================================================
    # Debate Completion Hooks
    # =========================================================================

    async def on_debate_complete(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
        agents: List["Agent"],
        task: str,
        belief_cruxes: Optional[List[str]] = None,
    ) -> None:
        """Called when a debate completes.

        Triggers all post-debate processing:
        1. Update agent relationships in ELO system
        2. Store debate outcome in memory
        3. Update retrieved memory outcomes
        4. Trigger formal verification (async)

        Args:
            ctx: The debate context
            result: The final debate result
            agents: List of participating agents
            task: The debate task/question
            belief_cruxes: Optional belief cruxes from analytics
        """
        # Get participant names
        participants = [a.name for a in agents]

        # Update agent relationships
        self._update_relationships(
            debate_id=ctx.debate_id,
            participants=participants,
            winner=getattr(result, "winner", None),
            votes=getattr(result, "votes", []),
        )

        # Store debate outcome in memory
        self._store_debate_outcome(result, task, belief_cruxes)

        # Update memory outcomes for retrieved memories
        self._update_memory_outcomes(result)

        # Update calibration scores and emit events
        self._update_calibration(
            ctx=ctx,
            result=result,
            participants=participants,
        )

        # Trigger formal verification (async)
        await self._verify_claims(result)

    def _update_relationships(
        self,
        debate_id: str,
        participants: List[str],
        winner: Optional[str],
        votes: List["Vote"],
    ) -> None:
        """Update agent relationships after debate completion.

        Uses batch update for O(1) database connections instead of O(n²).

        Args:
            debate_id: The debate ID
            participants: List of participant agent names
            winner: Name of the winning agent (if any)
            votes: List of vote objects
        """
        if not self.elo_system:
            return

        try:
            # Extract vote choices
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
            if updates:
                self.elo_system.update_relationships_batch(updates)

        except (AttributeError, TypeError, KeyError) as e:
            logger.warning(f"Relationship update error: {e}")
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning(f"Relationship update error (type={type(e).__name__}): {e}")

    def _store_debate_outcome(
        self,
        result: "DebateResult",
        task: str,
        belief_cruxes: Optional[List[str]] = None,
    ) -> None:
        """Store debate outcome in memory for future retrieval.

        Args:
            result: The debate result
            task: The debate task/question
            belief_cruxes: Optional belief cruxes from analytics
        """
        if not self.memory_manager:
            return

        try:
            # Normalize belief cruxes
            if belief_cruxes:
                belief_cruxes = [str(c) for c in belief_cruxes[:10]]

            self.memory_manager.store_debate_outcome(result, task, belief_cruxes=belief_cruxes)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Memory storage error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected memory storage error: {e}")

    def _update_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome.

        Updates the relevance/usefulness scores for memories that
        were retrieved at the start of this debate.

        Args:
            result: The debate result
        """
        if not self.memory_manager:
            return

        try:
            # Sync tracked IDs and tier info
            if self._continuum_retrieved_ids:
                self.memory_manager.track_retrieved_ids(
                    self._continuum_retrieved_ids,
                    tiers=self._continuum_retrieved_tiers,
                )
                self.memory_manager.update_memory_outcomes(result)

            # Clear tracking
            self._continuum_retrieved_ids = []
            self._continuum_retrieved_tiers = {}

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Memory outcome update error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected memory outcome update error: {e}")

    def _update_calibration(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
        participants: List[str],
    ) -> None:
        """Update calibration scores and emit events.

        Records prediction outcomes for each agent based on whether their
        position aligned with the final consensus/verdict.

        Args:
            ctx: The debate context
            result: The debate result
            participants: List of participant agent names
        """
        if not self.calibration_tracker:
            return

        try:
            winner = getattr(result, "winner", None)
            confidence = getattr(result, "confidence", 0.7)
            debate_id = ctx.debate_id

            # Record predictions for each participant
            for agent_name in participants:
                # Consider agent "correct" if they were the winner
                # or if they voted for the winning position
                correct = agent_name == winner

                # Get confidence from agent's last vote if available
                agent_confidence = confidence
                votes = getattr(result, "votes", [])
                for vote in votes:
                    if getattr(vote, "agent", None) == agent_name:
                        agent_confidence = getattr(vote, "confidence", confidence)
                        # Check if vote aligned with winner
                        if hasattr(vote, "choice") and winner:
                            correct = vote.choice == winner
                        break

                # Record the prediction
                self.calibration_tracker.record_prediction(
                    agent=agent_name,
                    confidence=agent_confidence,
                    correct=correct,
                    domain=ctx.domain or "general",
                    debate_id=debate_id,
                )

            # Emit calibration update events
            if self.event_emitter:
                for agent_name in participants:
                    try:
                        # Get updated calibration stats
                        curve = self.calibration_tracker.get_calibration_curve(agent_name)
                        if curve:
                            total_predictions = sum(b.count for b in curve)
                            total_correct = sum(b.count * b.actual for b in curve)
                            accuracy = total_correct / total_predictions if total_predictions > 0 else 0

                            # Compute Brier score (average squared error)
                            brier = 0.0
                            for bucket in curve:
                                if bucket.count > 0:
                                    expected = (bucket.range_start + bucket.range_end) / 2
                                    brier += bucket.count * (expected - bucket.actual) ** 2
                            brier = brier / total_predictions if total_predictions > 0 else 0

                            self.event_emitter.emit_calibration_update(
                                agent_name=agent_name,
                                brier_score=brier,
                                prediction_count=total_predictions,
                                accuracy=accuracy,
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug(f"Calibration event emission failed for {agent_name}: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected calibration event emission error for {agent_name}: {e}")

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Calibration update error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected calibration update error: {e}")

    async def _verify_claims(self, result: "DebateResult") -> None:
        """Verify decidable claims using formal methods.

        For arithmetic, logic, and constraint claims, attempts formal
        verification using Z3 SMT solver.

        Args:
            result: The debate result with grounded verdict
        """
        if not self.evidence_grounder:
            return

        if not getattr(result, "grounded_verdict", None):
            return

        try:
            await self.evidence_grounder.verify_claims_formally(result.grounded_verdict)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.debug(f"Formal verification timed out: {e}")
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Formal verification failed with data error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected formal verification error: {e}")

    # =========================================================================
    # Memory Tracking
    # =========================================================================

    def track_retrieved_memories(
        self,
        retrieved_ids: List[str],
        retrieved_tiers: dict,
    ) -> None:
        """Track which memories were retrieved for this debate.

        Called by context gatherer when memories are retrieved at
        debate start. These IDs are used later to update memory
        outcomes based on debate success.

        Args:
            retrieved_ids: List of retrieved memory IDs
            retrieved_tiers: Mapping of ID -> MemoryTier
        """
        self._continuum_retrieved_ids = retrieved_ids
        self._continuum_retrieved_tiers = retrieved_tiers

    # =========================================================================
    # Evidence Storage
    # =========================================================================

    def store_evidence(self, evidence_snippets: List[dict], task: str) -> None:
        """Store collected evidence snippets in memory.

        Args:
            evidence_snippets: List of evidence snippet dicts
            task: The debate task for context
        """
        if not self.memory_manager:
            return

        try:
            self.memory_manager.store_evidence(evidence_snippets, task)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Evidence storage error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected evidence storage error: {e}")

    # =========================================================================
    # Grounded Verdict Creation
    # =========================================================================

    def create_grounded_verdict(self, result: "DebateResult") -> Optional[Any]:
        """Create a GroundedVerdict for the final answer.

        Heavy3-inspired: Wrap final answers with evidence grounding analysis.

        Args:
            result: The debate result

        Returns:
            GroundedVerdict if created, None otherwise
        """
        if not self.evidence_grounder:
            return None

        if not result.final_answer:
            return None

        try:
            return self.evidence_grounder.create_grounded_verdict(
                final_answer=result.final_answer,
                confidence=result.confidence,
            )
        except Exception as e:
            logger.warning(f"Grounded verdict error: {e}")
            return None

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_status(self) -> dict:
        """Get status of all hook subsystems.

        Returns:
            Dictionary with subsystem availability
        """
        return {
            "subsystems": {
                "position_ledger": self.position_ledger is not None,
                "elo_system": self.elo_system is not None,
                "memory_manager": self.memory_manager is not None,
                "evidence_grounder": self.evidence_grounder is not None,
            },
            "tracking": {
                "retrieved_memory_count": len(self._continuum_retrieved_ids),
            },
        }


@dataclass
class HooksConfig:
    """Configuration for creating DebateHooks.

    Provides a clean way to configure hooks before creating the instance.
    """

    position_ledger: Optional[Any] = None
    elo_system: Optional[Any] = None
    memory_manager: Optional[Any] = None
    evidence_grounder: Optional[Any] = None

    def create_hooks(self) -> DebateHooks:
        """Create DebateHooks from this configuration.

        Returns:
            Configured DebateHooks instance
        """
        return DebateHooks(
            position_ledger=self.position_ledger,
            elo_system=self.elo_system,
            memory_manager=self.memory_manager,
            evidence_grounder=self.evidence_grounder,
        )


__all__ = ["DebateHooks", "HooksConfig"]
