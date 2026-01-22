"""
ELO feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles ELO match recording, voting accuracy, and learning bonuses.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.type_protocols import EloSystemProtocol, EventEmitterProtocol

logger = logging.getLogger(__name__)


class EloFeedback:
    """Handles ELO-related feedback operations."""

    def __init__(
        self,
        elo_system: Optional["EloSystemProtocol"] = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
        loop_id: Optional[str] = None,
    ):
        self.elo_system = elo_system
        self.event_emitter = event_emitter
        self.loop_id = loop_id

    def record_elo_match(self, ctx: "DebateContext") -> None:
        """Record ELO match results."""
        if not self.elo_system:
            return

        result = ctx.result
        if not result or not result.winner:
            return

        try:
            from aragora.agents.errors import _build_error_action

            participants = [agent.name for agent in ctx.agents]
            scores = {}

            for agent_name in participants:
                if agent_name == result.winner:
                    scores[agent_name] = 1.0
                elif result.consensus_reached:
                    scores[agent_name] = 0.5  # Draw for non-winners in consensus
                else:
                    scores[agent_name] = 0.0

            self.elo_system.record_match(ctx.debate_id, participants, scores, domain=ctx.domain)

            # Emit MATCH_RECORDED event
            self._emit_match_recorded_event(ctx, participants)

        except Exception as e:
            from aragora.agents.errors import _build_error_action

            _, msg, exc_info = _build_error_action(e, "elo")
            logger.warning(
                "ELO update failed for debate %s: %s", ctx.debate_id, msg, exc_info=exc_info
            )

    def _emit_match_recorded_event(self, ctx: "DebateContext", participants: list[str]) -> None:
        """Emit MATCH_RECORDED event for real-time leaderboard updates."""
        if not self.event_emitter or not self.elo_system:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            # Batch fetch all ratings
            ratings_batch = self.elo_system.get_ratings_batch(participants)
            elo_changes = {
                name: ratings_batch[name].elo if name in ratings_batch else 1500.0
                for name in participants
            }

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.MATCH_RECORDED,
                    loop_id=self.loop_id,
                    data={
                        "debate_id": ctx.debate_id,
                        "participants": participants,
                        "elo_changes": elo_changes,
                        "domain": ctx.domain,
                        "winner": ctx.result.winner,
                    },
                )
            )

            # Emit per-agent ELO updates for granular tracking
            for agent_name in participants:
                rating = ratings_batch.get(agent_name)
                if rating:
                    self.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.AGENT_ELO_UPDATED,
                            loop_id=self.loop_id,
                            agent=agent_name,
                            data={
                                "agent": agent_name,
                                "new_elo": rating.elo,
                                "debate_id": ctx.debate_id,
                                "domain": ctx.domain,
                                "is_winner": agent_name == ctx.result.winner,
                            },
                        )
                    )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.warning(f"ELO event emission error: {e}")

    def record_voting_accuracy(self, ctx: "DebateContext") -> None:
        """
        Record voting accuracy for agents based on consensus outcome.

        Cross-pollinates vote distribution with agent skill tracking.
        Agents who consistently vote for the consensus winner get small ELO bonus.
        """
        if not self.elo_system:
            return

        result = ctx.result
        if not result or not result.votes or not result.winner:
            return

        try:
            # Determine what the winning choice was
            winning_choice = result.winner.lower()

            for vote in result.votes:
                if not hasattr(vote, "agent") or not hasattr(vote, "choice"):
                    continue

                agent_name = vote.agent
                vote_choice = str(vote.choice).lower() if vote.choice else ""

                # Check if this agent voted for the consensus winner
                voted_for_consensus = (
                    winning_choice in vote_choice
                    or vote_choice in winning_choice
                    or winning_choice == vote_choice
                )

                # Update voting accuracy tracking
                self.elo_system.update_voting_accuracy(  # type: ignore[attr-defined]
                    agent_name=agent_name,
                    voted_for_consensus=voted_for_consensus,
                    domain=ctx.domain or "general",
                    debate_id=ctx.debate_id,
                    apply_elo_bonus=True,
                )

            logger.debug(
                f"[voting_accuracy] Recorded voting accuracy for {len(result.votes)} votes "
                f"in debate {ctx.debate_id}"
            )

        except Exception as e:
            logger.debug(f"[voting_accuracy] Recording failed: {e}")

    def apply_learning_bonuses(self, ctx: "DebateContext") -> None:
        """
        Apply learning efficiency bonuses to participating agents.

        Cross-pollinates debate outcomes with agent learning tracking.
        Agents who demonstrate consistent improvement over time get ELO bonuses.
        """
        if not self.elo_system:
            return

        result = ctx.result
        if not result or not result.winner:
            return

        # Only apply learning bonuses for successful debates
        if not result.consensus_reached:
            return

        try:
            domain = ctx.domain or "general"
            for agent in ctx.agents:
                try:
                    bonus = self.elo_system.apply_learning_bonus(  # type: ignore[attr-defined]
                        agent_name=agent.name,
                        domain=domain,
                        debate_id=ctx.debate_id,
                        bonus_factor=0.5,  # Moderate bonus factor
                    )
                    if bonus > 0:
                        logger.debug(
                            f"[learning] Applied learning bonus {bonus:.2f} "
                            f"to {agent.name} in domain {domain}"
                        )
                except Exception as e:
                    logger.debug(f"[learning] Bonus failed for {agent.name}: {e}")

        except Exception as e:
            logger.debug(f"[learning] Learning bonus application failed: {e}")


__all__ = ["EloFeedback"]
