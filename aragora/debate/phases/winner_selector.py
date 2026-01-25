"""Winner selection and result finalization for consensus phase.

This module extracts winner determination logic from ConsensusPhase:
- Majority winner determination
- Unanimous winner setting
- No-unanimity handling
- Belief network analysis
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.agents.errors import _build_error_action

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class WinnerSelector:
    """Handles winner selection and result finalization.

    Extracted from ConsensusPhase to improve modularity and testability.
    """

    def __init__(
        self,
        *,
        # Dependencies
        protocol: Any = None,
        position_tracker: Any = None,
        calibration_tracker: Any = None,
        recorder: Any = None,
        # Callbacks
        notify_spectator: Optional[Callable[..., Any]] = None,
        extract_debate_domain: Optional[Callable[..., Any]] = None,
        get_belief_analyzer: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the winner selector.

        Args:
            protocol: Debate protocol
            position_tracker: Position tracking for truth-grounded personas
            calibration_tracker: Calibration tracking
            recorder: Event recorder
            notify_spectator: Spectator notification callback
            extract_debate_domain: Domain extraction callback
            get_belief_analyzer: Belief analyzer getter callback
        """
        self.protocol = protocol
        self.position_tracker = position_tracker
        self.calibration_tracker = calibration_tracker
        self.recorder = recorder

        self._notify_spectator = notify_spectator
        self._extract_debate_domain = extract_debate_domain
        self._get_belief_analyzer = get_belief_analyzer

    def determine_majority_winner(
        self,
        ctx: "DebateContext",
        vote_counts: dict[str, float],
        total_votes: float,
        choice_mapping: dict[str, str],
        normalize_choice: Callable[[str, list["Agent"], dict[str, str]], str],
        threshold_override: Optional[float] = None,
    ) -> None:
        """Determine winner for majority consensus.

        Args:
            ctx: Debate context
            vote_counts: Vote counts by choice
            total_votes: Total weighted votes
            choice_mapping: Mapping from choices to canonical choices
            normalize_choice: Function to normalize choice to agent name
            threshold_override: Optional override for consensus threshold
        """
        result = ctx.result
        proposals = ctx.proposals

        if not vote_counts:
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            result.confidence = 0.5
            return

        # Find winner (highest vote count)
        winner_choice = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        count = vote_counts[winner_choice]
        if threshold_override is not None:
            threshold = threshold_override
        else:
            threshold = self.protocol.consensus_threshold if self.protocol else 0.5

        winner_agent = normalize_choice(winner_choice, ctx.agents, proposals)

        result.final_answer = proposals.get(
            winner_agent, list(proposals.values())[0] if proposals else ""
        )
        result.consensus_reached = (count / total_votes >= threshold) if total_votes > 0 else False
        result.confidence = count / total_votes if total_votes > 0 else 0.5
        ctx.winner_agent = winner_agent
        result.winner = winner_agent

        # Calculate consensus variance and strength
        if len(vote_counts) > 1:
            counts = list(vote_counts.values())
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            result.consensus_variance = variance

            if variance < 1:
                result.consensus_strength = "strong"
            elif variance < 2:
                result.consensus_strength = "medium"
            else:
                result.consensus_strength = "weak"

            logger.info(
                f"consensus_strength strength={result.consensus_strength} variance={variance:.2f}"
            )
        else:
            result.consensus_strength = "unanimous"
            result.consensus_variance = 0.0

        # Track dissenting views
        for agent, prop in proposals.items():
            if agent != winner_agent:
                result.dissenting_views.append(f"[{agent}]: {prop}")

        logger.info(f"consensus_winner winner={winner_agent} votes={count}/{len(ctx.agents)}")

        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"Majority vote: {winner_agent}",
                metric=result.confidence,
            )

        if self.recorder:
            try:
                self.recorder.record_phase_change(f"consensus_reached: {winner_agent}")
            except Exception as e:
                logger.debug(f"Recorder error for consensus: {e}")

        # Finalize for truth-grounded personas
        if self.position_tracker:
            try:
                debate_id = (
                    result.id if hasattr(result, "id") else (ctx.env.task[:50] if ctx.env else "")
                )
                self.position_tracker.finalize_debate(
                    debate_id=debate_id,
                    winning_agent=winner_agent,
                    winning_position=result.final_answer[:1000],
                    consensus_confidence=result.confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracker finalize error: {e}")

        # Record calibration predictions
        self._record_calibration_predictions(ctx, winner_agent, choice_mapping)

    def _record_calibration_predictions(
        self,
        ctx: "DebateContext",
        winner_agent: str,
        choice_mapping: dict[str, str],
    ) -> None:
        """Record calibration predictions for all votes."""
        if not self.calibration_tracker:
            return

        result = ctx.result
        try:
            debate_id = (
                result.id if hasattr(result, "id") else (ctx.env.task[:50] if ctx.env else "")
            )
            domain = self._extract_debate_domain() if self._extract_debate_domain else "general"
            for v in result.votes:
                if not isinstance(v, Exception):
                    canonical = choice_mapping.get(v.choice, v.choice)
                    correct = canonical == winner_agent
                    self.calibration_tracker.record_prediction(
                        agent=v.agent,
                        confidence=v.confidence,
                        correct=correct,
                        domain=domain,
                        debate_id=debate_id,
                    )
            logger.debug(f"calibration_recorded predictions={len(result.votes)}")
        except Exception as e:
            category, msg, exc_info = _build_error_action(e, "calibration")
            logger.warning(f"calibration_error category={category} error={msg}", exc_info=exc_info)

    def set_unanimous_winner(
        self,
        ctx: "DebateContext",
        winner: str,
        unanimity_ratio: float,
        total_voters: int,
        count: int,
    ) -> None:
        """Set result for unanimous consensus.

        Args:
            ctx: Debate context
            winner: Winning agent name
            unanimity_ratio: Ratio of votes for winner
            total_voters: Total number of voters
            count: Number of votes for winner
        """
        result = ctx.result
        proposals = ctx.proposals

        result.final_answer = proposals.get(
            winner, list(proposals.values())[0] if proposals else ""
        )
        result.consensus_reached = True
        result.confidence = unanimity_ratio
        result.consensus_strength = "unanimous"
        result.consensus_variance = 0.0
        ctx.winner_agent = winner
        result.winner = winner

        logger.info(
            f"consensus_unanimous winner={winner} votes={count}/{total_voters} "
            f"ratio={unanimity_ratio:.0%}"
        )

        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"Unanimous: {winner}",
                metric=result.confidence,
            )

        if self.recorder:
            try:
                self.recorder.record_phase_change(f"consensus_reached: {winner}")
            except Exception as e:
                logger.debug(f"Recorder error for unanimous consensus: {e}")

        # Record calibration predictions
        if self.calibration_tracker:
            try:
                debate_id = (
                    result.id if hasattr(result, "id") else (ctx.env.task[:50] if ctx.env else "")
                )
                domain = self._extract_debate_domain() if self._extract_debate_domain else "general"
                for v in result.votes:
                    if not isinstance(v, Exception):
                        correct = v.choice == winner
                        self.calibration_tracker.record_prediction(
                            agent=v.agent,
                            confidence=v.confidence,
                            correct=correct,
                            domain=domain,
                            debate_id=debate_id,
                        )
                logger.debug(f"calibration_recorded_unanimous predictions={len(result.votes)}")
            except Exception as e:
                category, msg, exc_info = _build_error_action(e, "calibration")
                logger.warning(
                    f"calibration_error_unanimous category={category} error={msg}",
                    exc_info=exc_info,
                )

    def set_no_unanimity(
        self,
        ctx: "DebateContext",
        winner: str,
        unanimity_ratio: float,
        total_voters: int,
        count: int,
        choice_mapping: dict[str, str],
    ) -> None:
        """Set result when unanimity not reached.

        Args:
            ctx: Debate context
            winner: Best choice (most votes)
            unanimity_ratio: Ratio of votes for winner
            total_voters: Total number of voters
            count: Number of votes for winner
            choice_mapping: Mapping from choices to canonical choices
        """
        result = ctx.result
        proposals = ctx.proposals
        vote_counts: Counter[str] = Counter()
        for v in result.votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                vote_counts[canonical] += 1

        result.final_answer = (
            "[No unanimous consensus reached]\n\nProposals:\n"
            + "\n\n---\n\n".join(
                f"[{agent}] ({vote_counts.get(choice_mapping.get(agent, agent), 0)} votes):\n{prop}"
                for agent, prop in proposals.items()
            )
        )
        result.consensus_reached = False
        result.confidence = unanimity_ratio
        result.consensus_strength = "none"

        # Track all views as dissenting
        for agent, prop in proposals.items():
            result.dissenting_views.append(f"[{agent}]: {prop}")

        logger.info(
            f"consensus_not_unanimous best={winner} ratio={unanimity_ratio:.0%} "
            f"votes={count}/{total_voters}"
        )

        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"No unanimity: {winner} got {unanimity_ratio:.0%}",
                metric=unanimity_ratio,
            )

    def analyze_belief_network(self, ctx: "DebateContext") -> None:
        """Analyze belief network to identify debate cruxes.

        Args:
            ctx: Debate context
        """
        if not self._get_belief_analyzer:
            return

        result = ctx.result
        if not result.messages:
            return

        BN, BPA = self._get_belief_analyzer()
        if not BN:
            return

        try:
            # Import CruxDetector for crux identification
            from aragora.reasoning.crux_detector import CruxDetector

            network = BN(max_iterations=3)
            for i, msg in enumerate(result.messages):
                if msg.role in ("proposer", "critic"):
                    # Use correct add_claim signature: claim_id, statement, author
                    network.add_claim(
                        claim_id=f"msg_{i}_{msg.agent}",
                        statement=msg.content[:500],
                        author=msg.agent,
                    )

            # Use CruxDetector to identify cruxes
            detector = CruxDetector(network)
            analysis = detector.detect_cruxes(top_k=5, min_score=0.1)

            if analysis.cruxes:
                setattr(
                    result,
                    "cruxes",
                    [
                        {
                            "claim": c.statement,
                            "score": c.crux_score,
                            "agents": c.contesting_agents,
                        }
                        for c in analysis.cruxes
                    ],
                )
                logger.info(f"belief_cruxes_identified count={len(analysis.cruxes)}")

        except Exception as e:
            logger.debug(f"Belief network analysis failed: {e}")


__all__ = ["WinnerSelector"]
