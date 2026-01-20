"""
Convergence tracking for debate rounds.

Extracted from debate_rounds.py to handle:
- Semantic convergence detection between rounds
- Novelty tracking to prevent stale debates
- RLM ready signal quorum for agent self-termination
- Trickster integration for hollow consensus detection
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from aragora.debate.phases.ready_signal import (
    CollectiveReadiness,
    parse_ready_signal,
)

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.debate.convergence import ConvergenceDetector
    from aragora.debate.trickster import Trickster
    from aragora.debate.novelty import NoveltyTracker

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Result of convergence check."""

    converged: bool = False
    status: str = ""
    similarity: float = 0.0
    blocked_by_trickster: bool = False


class DebateConvergenceTracker:
    """
    Tracks convergence, novelty, and readiness in debates.

    Handles:
    - Semantic similarity between rounds (convergence detection)
    - Novelty tracking to ensure proposals evolve
    - RLM ready signal quorum for early termination
    - Trickster integration for hollow consensus detection
    """

    def __init__(
        self,
        convergence_detector: Optional["ConvergenceDetector"] = None,
        novelty_tracker: Optional["NoveltyTracker"] = None,
        trickster: Optional["Trickster"] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        event_emitter: Any = None,
        notify_spectator: Optional[Callable] = None,
        inject_challenge: Optional[Callable] = None,
    ):
        """
        Initialize convergence tracker.

        Args:
            convergence_detector: Detector for semantic convergence
            novelty_tracker: Tracker for proposal novelty
            trickster: Hollow consensus detector
            hooks: Event hooks for callbacks
            event_emitter: WebSocket event emitter
            notify_spectator: Callback to notify spectators
            inject_challenge: Callback to inject trickster challenges
        """
        self.convergence_detector = convergence_detector
        self.novelty_tracker = novelty_tracker
        self.trickster = trickster
        self.hooks = hooks or {}
        self.event_emitter = event_emitter
        self._notify_spectator = notify_spectator
        self._inject_challenge = inject_challenge

        # State tracking
        self._previous_round_responses: Dict[str, str] = {}
        self._collective_readiness = CollectiveReadiness()

    def reset(self) -> None:
        """Reset tracker state for a new debate."""
        self._previous_round_responses = {}
        self._collective_readiness = CollectiveReadiness()

    def check_convergence(self, ctx: "DebateContext", round_num: int) -> ConvergenceResult:
        """
        Check for convergence and return result.

        Args:
            ctx: The debate context with current proposals
            round_num: Current round number

        Returns:
            ConvergenceResult with convergence status
        """
        result = ConvergenceResult()

        if not self.convergence_detector:
            return result

        current_responses = dict(ctx.proposals)

        if not self._previous_round_responses:
            self._previous_round_responses = current_responses
            return result

        convergence = self.convergence_detector.check_convergence(
            current_responses, self._previous_round_responses, round_num
        )

        self._previous_round_responses = current_responses

        if not convergence:
            return result

        # Update context result
        ctx_result = ctx.result
        ctx_result.convergence_status = convergence.status
        ctx_result.convergence_similarity = convergence.avg_similarity
        ctx_result.per_agent_similarity = convergence.per_agent_similarity

        result.status = convergence.status
        result.similarity = convergence.avg_similarity

        logger.info(
            f"convergence_check status={convergence.status} "
            f"similarity={convergence.avg_similarity:.0%}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "convergence",
                details=f"{convergence.status}",
                metric=convergence.avg_similarity,
            )

        # Emit convergence event
        if "on_convergence_check" in self.hooks:
            self.hooks["on_convergence_check"](
                status=convergence.status,
                similarity=convergence.avg_similarity,
                per_agent=convergence.per_agent_similarity,
                round_num=round_num,
            )

        # Check for hollow consensus using trickster
        if self.trickster and convergence.avg_similarity > 0.5:
            intervention = self.trickster.check_and_intervene(
                responses=current_responses,
                convergence_similarity=convergence.avg_similarity,
                round_num=round_num,
            )
            if intervention:
                logger.info(
                    f"trickster_intervention round={round_num} "
                    f"type={intervention.intervention_type.value} "
                    f"targets={intervention.target_agents}"
                )
                # Notify spectator about hollow consensus
                if self._notify_spectator:
                    self._notify_spectator(
                        "hollow_consensus",
                        details="Evidence quality challenge triggered",
                        metric=intervention.priority,
                        agent="trickster",
                    )

                # Emit trickster events via EventEmitter for WebSocket clients
                if self.event_emitter:
                    self.event_emitter.emit_sync(
                        event_type="hollow_consensus",
                        debate_id=ctx.debate_id if hasattr(ctx, "debate_id") else "",
                        confidence=intervention.priority,
                        indicators=list(intervention.evidence_gaps.keys())[:5],
                        recommendation=intervention.challenge_text[:200],
                    )
                    self.event_emitter.emit_sync(
                        event_type="trickster_intervention",
                        debate_id=ctx.debate_id if hasattr(ctx, "debate_id") else "",
                        intervention_type=intervention.intervention_type.value,
                        challenge=intervention.challenge_text,
                        target_claim=", ".join(intervention.target_agents),
                        round=round_num,
                    )

                # Call hooks for WebSocket broadcasting
                if "on_hollow_consensus" in self.hooks:
                    self.hooks["on_hollow_consensus"](
                        confidence=intervention.priority,
                        indicators=list(intervention.evidence_gaps.keys())[:5],
                        recommendation=intervention.challenge_text[:200],
                    )
                if "on_trickster_intervention" in self.hooks:
                    self.hooks["on_trickster_intervention"](
                        intervention_type=intervention.intervention_type.value,
                        targets=intervention.target_agents,
                        challenge=intervention.challenge_text,
                        round_num=round_num,
                    )

                # Inject challenge into context for next round
                if self._inject_challenge:
                    self._inject_challenge(intervention.challenge_text, ctx)

                # Block convergence if hollow
                if intervention.priority > 0.5:
                    logger.info(f"hollow_consensus_blocked round={round_num}")
                    result.blocked_by_trickster = True
                    return result

        if convergence.converged:
            logger.info(f"debate_converged round={round_num}")
            result.converged = True

        return result

    def track_novelty(self, ctx: "DebateContext", round_num: int) -> None:
        """
        Track novelty of current proposals compared to prior proposals.

        Updates context with novelty scores and triggers trickster intervention
        if proposals are too similar to previous rounds.

        Args:
            ctx: The debate context with current proposals
            round_num: Current round number
        """
        if not self.novelty_tracker:
            return

        current_proposals = dict(ctx.proposals)
        if not current_proposals:
            return

        # Compute novelty against prior proposals
        novelty_result = self.novelty_tracker.compute_novelty(current_proposals, round_num)

        # Update context with novelty scores
        for agent, novelty in novelty_result.per_agent_novelty.items():
            if agent not in ctx.per_agent_novelty:
                ctx.per_agent_novelty[agent] = []
            ctx.per_agent_novelty[agent].append(novelty)

        ctx.avg_novelty = novelty_result.avg_novelty
        ctx.low_novelty_agents = novelty_result.low_novelty_agents

        # Add to history for future comparisons
        self.novelty_tracker.add_to_history(current_proposals)

        logger.info(
            f"novelty_check round={round_num} avg={novelty_result.avg_novelty:.2f} "
            f"min={novelty_result.min_novelty:.2f} low_novelty={novelty_result.low_novelty_agents}"
        )

        # Notify spectator about novelty
        if self._notify_spectator:
            self._notify_spectator(
                "novelty",
                details=f"Avg novelty: {novelty_result.avg_novelty:.0%}",
                metric=novelty_result.avg_novelty,
            )

        # Emit novelty event
        if "on_novelty_check" in self.hooks:
            self.hooks["on_novelty_check"](
                avg_novelty=novelty_result.avg_novelty,
                per_agent=novelty_result.per_agent_novelty,
                low_novelty_agents=novelty_result.low_novelty_agents,
                round_num=round_num,
            )

        # Check for low novelty and trigger trickster intervention
        if novelty_result.has_low_novelty() and self.trickster:
            if hasattr(self.trickster, "create_novelty_challenge"):
                intervention = self.trickster.create_novelty_challenge(
                    low_novelty_agents=novelty_result.low_novelty_agents,
                    novelty_scores=novelty_result.per_agent_novelty,
                    round_num=round_num,
                )
                if intervention:
                    logger.info(
                        f"novelty_challenge round={round_num} targets={intervention.target_agents}"
                    )
                    if self._notify_spectator:
                        self._notify_spectator(
                            "low_novelty",
                            details="Proposals too similar to prior rounds",
                            metric=novelty_result.min_novelty,
                            agent="trickster",
                        )

                    if self.event_emitter:
                        self.event_emitter.emit_sync(
                            event_type="trickster_intervention",
                            debate_id=ctx.debate_id if hasattr(ctx, "debate_id") else "",
                            intervention_type=intervention.intervention_type.value,
                            challenge=intervention.challenge_text,
                            target_claim=", ".join(intervention.target_agents),
                            round=round_num,
                        )

                    if "on_trickster_intervention" in self.hooks:
                        self.hooks["on_trickster_intervention"](
                            intervention_type=intervention.intervention_type.value,
                            targets=intervention.target_agents,
                            challenge=intervention.challenge_text,
                            round_num=round_num,
                        )

                    if self._inject_challenge:
                        self._inject_challenge(intervention.challenge_text, ctx)

    def check_rlm_ready_quorum(self, ctx: "DebateContext", round_num: int) -> bool:
        """
        Check if RLM ready signal quorum has been reached.

        Implements the RLM "answer ready" pattern where agents can signal
        when they've reached their final position with high confidence.

        Args:
            ctx: The DebateContext with proposals
            round_num: Current round number

        Returns:
            True if quorum of agents are ready to terminate
        """
        # Parse ready signals from current proposals
        for agent_name, proposal in ctx.proposals.items():
            signal = parse_ready_signal(agent_name, proposal, round_num)
            self._collective_readiness.update(signal)

        # Check if quorum reached
        if self._collective_readiness.has_quorum():
            ready_agents = [
                s.agent for s in self._collective_readiness.signals.values() if s.should_terminate()
            ]
            logger.info(
                f"rlm_ready_quorum_reached round={round_num} "
                f"ready={len(ready_agents)}/{self._collective_readiness.total_count} "
                f"avg_confidence={self._collective_readiness.avg_confidence:.2f}"
            )

            if self._notify_spectator:
                self._notify_spectator(
                    "rlm_ready",
                    details=f"Agent quorum ready ({len(ready_agents)} agents)",
                    metric=self._collective_readiness.avg_confidence,
                    agent="system",
                )

            if "on_rlm_ready_quorum" in self.hooks:
                self.hooks["on_rlm_ready_quorum"](
                    round_num=round_num,
                    ready_agents=ready_agents,
                    total_agents=self._collective_readiness.total_count,
                    avg_confidence=self._collective_readiness.avg_confidence,
                )

            return True

        return False

    @property
    def collective_readiness(self) -> CollectiveReadiness:
        """Get the collective readiness state."""
        return self._collective_readiness

    @property
    def previous_responses(self) -> Dict[str, str]:
        """Get previous round responses for comparison."""
        return self._previous_round_responses
