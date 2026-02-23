"""
Calibration feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles calibration data recording, bidirectional calibration feedback,
consensus-based auto-calibration, and Brier score updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.type_protocols import CalibrationTrackerProtocol, EventEmitterProtocol

logger = logging.getLogger(__name__)


class CalibrationFeedback:
    """Handles calibration-related feedback operations."""

    def __init__(
        self,
        calibration_tracker: CalibrationTrackerProtocol | None = None,
        event_emitter: EventEmitterProtocol | None = None,
        knowledge_mound: Any | None = None,
        loop_id: str | None = None,
    ):
        self.calibration_tracker = calibration_tracker
        self.event_emitter = event_emitter
        self.knowledge_mound = knowledge_mound
        self.loop_id = loop_id

    def record_calibration(self, ctx: DebateContext) -> None:
        """Record calibration data from agent votes with confidence.

        Tracks (confidence, outcome) pairs for measuring prediction accuracy.
        Each vote with confidence is recorded as a calibration data point,
        comparing the voted choice against the actual debate winner.
        """
        if not self.calibration_tracker:
            return

        result = ctx.result
        if not result:
            return

        # Determine the actual outcome
        actual_winner = result.winner or "no_consensus"

        try:
            # Record each vote with confidence as a calibration data point
            recorded = 0
            for vote in result.votes:
                # Check if vote has confidence attribute
                confidence = getattr(vote, "confidence", None)
                if confidence is None:
                    continue

                # Determine if the prediction was correct
                # A vote is "correct" if the voted choice matches the actual winner
                correct = vote.choice == actual_winner

                # Record the prediction with correct parameter names
                self.calibration_tracker.record_prediction(
                    agent=vote.agent,
                    confidence=confidence,
                    correct=correct,
                    debate_id=getattr(result, "debate_id", ""),
                )
                recorded += 1

            if recorded > 0:
                logger.debug("[calibration] Recorded %s predictions", recorded)
                # Emit CALIBRATION_UPDATE event for real-time panel updates
                self._emit_calibration_update(ctx, recorded)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[calibration] Failed to record: %s", e)

    def _emit_calibration_update(self, ctx: DebateContext, recorded_count: int) -> None:
        """Emit CALIBRATION_UPDATE event to WebSocket."""
        if not self.event_emitter or not self.calibration_tracker:
            return

        try:
            from aragora.events.types import StreamEvent, StreamEventType

            # Get summary stats from calibration tracker
            summary = {}
            if hasattr(self.calibration_tracker, "get_summary"):
                summary = self.calibration_tracker.get_summary()

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.CALIBRATION_UPDATE,
                    loop_id=self.loop_id,
                    data={
                        "debate_id": ctx.debate_id,
                        "predictions_recorded": recorded_count,
                        "total_predictions": summary.get("total_predictions", 0),
                        "overall_accuracy": summary.get("overall_accuracy", 0.0),
                        "domain": ctx.domain,
                    },
                )
            )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug("Calibration event emission error: %s", e)

    def apply_calibration_feedback(self, ctx: DebateContext) -> None:
        """Apply bidirectional calibration adjustment based on consensus alignment.

        Agents who vote correctly with high confidence get a positive adjustment.
        Agents who vote incorrectly with high confidence get a negative adjustment.
        This creates a feedback loop that rewards well-calibrated agents.
        """
        if not self.calibration_tracker:
            return

        result = ctx.result
        if not result:
            return

        # Skip if no consensus was reached
        if not result.consensus_reached:
            return

        actual_winner = result.winner
        if not actual_winner:
            return

        try:
            for vote in result.votes:
                confidence = getattr(vote, "confidence", None)
                if confidence is None or confidence <= 0.7:
                    continue

                canonical = ctx.choice_mapping.get(vote.choice, vote.choice)
                voted_correctly = canonical == actual_winner

                if voted_correctly:
                    adjustment = 0.02 * confidence
                else:
                    adjustment = -0.03 * confidence

                # Use record_prediction with the adjustment encoded as a calibration signal
                self.calibration_tracker.record_prediction(
                    agent=vote.agent,
                    confidence=confidence,
                    correct=voted_correctly,
                    domain=ctx.domain,
                    debate_id=getattr(result, "debate_id", ctx.debate_id),
                    prediction_type="consensus_feedback",
                )

                logger.debug(
                    "[calibration_feedback] %s: %+.3f (confidence=%.2f, correct=%s)",
                    vote.agent,
                    adjustment,
                    confidence,
                    voted_correctly,
                )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.warning("[calibration_feedback] Failed: %s", e)

    def record_consensus_calibration(self, ctx: DebateContext) -> None:
        """Record auto-calibration from consensus alignment.

        Every debate is a calibration event: each agent's final position
        is implicitly a prediction about the outcome. Agents whose proposals
        aligned with the eventual consensus were "correct"; those whose
        proposals diverged were "incorrect".

        This records domain-specific calibration predictions for ALL
        participating agents, not just those who explicitly voted with
        confidence. Over time, the system learns which agents are
        reliable on which domains without requiring explicit tournaments.
        """
        if not self.calibration_tracker:
            return

        result = ctx.result
        if not result or not result.consensus_reached:
            return

        winner = result.winner
        if not winner:
            return

        try:
            recorded = 0
            for agent in ctx.agents:
                # Determine if this agent's position aligned with consensus
                # Check proposals for this agent
                agent_proposals = [
                    p for p in (result.proposals or []) if getattr(p, "agent", None) == agent.name
                ]
                if not agent_proposals:
                    continue

                # Agent aligned with consensus if they proposed the winning position
                # or if they voted for the winner
                aligned = agent.name == winner
                # Also check votes
                for vote in result.votes:
                    if vote.agent == agent.name:
                        canonical = ctx.choice_mapping.get(vote.choice, vote.choice)
                        if canonical == winner:
                            aligned = True
                        break

                # Estimate implicit confidence from proposal strength
                implicit_confidence = 0.6  # baseline

                self.calibration_tracker.record_prediction(
                    agent=agent.name,
                    confidence=implicit_confidence,
                    correct=aligned,
                    domain=ctx.domain,
                    debate_id=ctx.debate_id,
                    prediction_type="consensus_alignment",
                )
                recorded += 1

            if recorded > 0:
                logger.debug(
                    "[auto_calibration] Recorded %d consensus alignment predictions (domain=%s)",
                    recorded,
                    ctx.domain,
                )
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.debug("[auto_calibration] Failed: %s", e)

    def update_calibration_feedback(self, ctx: DebateContext) -> None:
        """Close the calibration feedback loop for agent selection.

        For each participating agent, computes prediction accuracy against the
        debate outcome, updates their Brier scores via the CalibrationTracker,
        and stores the calibration delta in KnowledgeMound via the
        CalibrationFusionAdapter so the agent selection system can incorporate
        calibration quality into future team composition.

        This bridges the gap between calibration tracking (which records data)
        and agent selection (which uses calibration for team building).
        """
        if not self.calibration_tracker:
            return

        result = ctx.result
        if not result or not result.consensus_reached:
            return

        actual_winner = result.winner
        if not actual_winner:
            return

        try:
            calibration_deltas: dict[str, dict[str, float]] = {}

            for agent in ctx.agents:
                agent_name = agent.name

                # Compute prediction accuracy for this agent:
                # Did they vote for the winning position?
                predicted_correctly = False
                agent_confidence = 0.5  # default if no vote found

                for vote in result.votes:
                    if vote.agent == agent_name:
                        confidence = getattr(vote, "confidence", None)
                        if confidence is not None:
                            agent_confidence = confidence
                        canonical = ctx.choice_mapping.get(vote.choice, vote.choice)
                        predicted_correctly = canonical == actual_winner
                        break

                # Compute Brier score component: (confidence - outcome)^2
                outcome = 1.0 if predicted_correctly else 0.0
                brier_component = (agent_confidence - outcome) ** 2

                # Get previous Brier score for delta computation
                previous_brier = 0.5  # default baseline
                if hasattr(self.calibration_tracker, "get_brier_score"):
                    try:
                        previous_brier = self.calibration_tracker.get_brier_score(
                            agent_name, domain=ctx.domain
                        )
                    except (KeyError, TypeError, ValueError):
                        pass

                # Record the prediction to update the tracker's running score
                self.calibration_tracker.record_prediction(
                    agent=agent_name,
                    confidence=agent_confidence,
                    correct=predicted_correctly,
                    domain=ctx.domain,
                    debate_id=ctx.debate_id,
                    prediction_type="selection_feedback",
                )

                # Compute new Brier score after update
                new_brier = previous_brier  # default if can't query
                if hasattr(self.calibration_tracker, "get_brier_score"):
                    try:
                        new_brier = self.calibration_tracker.get_brier_score(
                            agent_name, domain=ctx.domain
                        )
                    except (KeyError, TypeError, ValueError):
                        pass

                calibration_deltas[agent_name] = {
                    "brier_before": previous_brier,
                    "brier_after": new_brier,
                    "brier_delta": new_brier - previous_brier,
                    "brier_component": brier_component,
                    "predicted_correctly": float(predicted_correctly),
                    "confidence": agent_confidence,
                }

            if not calibration_deltas:
                return

            # Store calibration deltas in KnowledgeMound via CalibrationFusionAdapter
            if self.knowledge_mound:
                self._store_calibration_in_mound(ctx, calibration_deltas)

            logger.info(
                "[calibration_feedback] Updated Brier scores for %d agents (debate=%s, domain=%s)",
                len(calibration_deltas),
                ctx.debate_id,
                ctx.domain,
            )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[calibration_feedback] Update failed: %s", e)

    def _store_calibration_in_mound(
        self,
        ctx: DebateContext,
        calibration_deltas: dict[str, dict[str, float]],
    ) -> None:
        """Store calibration deltas in KnowledgeMound for cross-debate learning.

        Uses the knowledge_mound's store method (if available) to persist
        calibration feedback so future debates can factor in historical
        agent reliability when selecting teams.
        """
        if not self.knowledge_mound:
            return

        try:
            record = {
                "type": "calibration_feedback",
                "debate_id": ctx.debate_id,
                "domain": ctx.domain,
                "agent_deltas": calibration_deltas,
                "consensus_reached": True,
                "winner": ctx.result.winner if ctx.result else None,
            }

            if hasattr(self.knowledge_mound, "store_calibration_feedback"):
                self.knowledge_mound.store_calibration_feedback(record)
            elif hasattr(self.knowledge_mound, "store"):
                self.knowledge_mound.store(
                    key=f"calibration:{ctx.debate_id}",
                    value=record,
                    category="calibration_feedback",
                )

            logger.debug(
                "[calibration_feedback] Stored calibration deltas in KnowledgeMound for %d agents",
                len(calibration_deltas),
            )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.debug("[calibration_feedback] Mound storage failed: %s", e)


__all__ = ["CalibrationFeedback"]
