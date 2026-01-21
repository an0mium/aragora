"""Vote bonus calculations for consensus phase.

Extracted from consensus_phase.py to handle:
- Evidence citation bonuses based on evidence quality scoring
- Process evaluation bonuses using Agent-as-a-Judge bias mitigation
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from aragora.observability.metrics import (
    record_evidence_citation_bonus,
    record_process_evaluation_bonus,
)

if TYPE_CHECKING:
    from aragora.core import Vote
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class VoteBonusCalculator:
    """Calculates bonuses to apply to vote counts based on evidence and process quality.

    This class extracts the evidence citation bonus and process evaluation bonus
    logic from ConsensusPhase to improve modularity and testability.
    """

    def __init__(self, protocol: Any = None):
        """Initialize the bonus calculator.

        Args:
            protocol: Debate protocol with configuration for bonus calculations
        """
        self.protocol = protocol

    def apply_evidence_citation_bonuses(
        self,
        ctx: "DebateContext",
        votes: list["Vote"],
        vote_counts: dict[str, float],
        choice_mapping: dict[str, str],
    ) -> dict[str, float]:
        """Apply evidence citation bonuses to vote counts.

        Enhanced to use evidence quality scores when available:
        - semantic_relevance: How relevant the evidence is to the topic (0.4 weight)
        - authority: Source credibility (0.3 weight)
        - freshness: How recent the evidence is (0.2 weight)
        - completeness: Evidence completeness (0.1 weight)

        Quality-weighted bonus = base_bonus * quality_score^0.5

        Args:
            ctx: Debate context with evidence pack
            votes: List of agent votes
            vote_counts: Current vote count dictionary
            choice_mapping: Maps vote choices to canonical agent names

        Returns:
            Updated vote counts with evidence bonuses applied
        """
        if not self.protocol or not getattr(self.protocol, "enable_evidence_weighting", False):
            return vote_counts

        evidence_pack = getattr(ctx, "evidence_pack", None)
        if not evidence_pack or not hasattr(evidence_pack, "snippets"):
            return vote_counts

        # Build evidence lookup with quality scores
        evidence_lookup: dict[str, dict] = {}
        for snippet in evidence_pack.snippets:
            evidence_lookup[snippet.id] = {
                "snippet": snippet,
                "quality_scores": getattr(snippet, "quality_scores", {}),
            }

        if not evidence_lookup:
            return vote_counts

        evidence_bonus = getattr(self.protocol, "evidence_citation_bonus", 0.15)
        use_quality_scores = getattr(self.protocol, "enable_evidence_quality_weighting", True)
        evidence_citations: dict[str, int] = {}
        evidence_quality_totals: dict[str, float] = {}

        for vote in votes:
            if isinstance(vote, Exception):
                continue

            cited_ids = set(re.findall(r"EVID-([a-zA-Z0-9]+)", vote.reasoning))
            valid_citation_ids = cited_ids & set(evidence_lookup.keys())
            valid_citations = len(valid_citation_ids)

            if valid_citations > 0:
                canonical = choice_mapping.get(vote.choice, vote.choice)
                if canonical in vote_counts:
                    # Calculate quality-weighted bonus
                    total_quality = 0.0
                    for evid_id in valid_citation_ids:
                        evid_data = evidence_lookup.get(evid_id, {})
                        quality_scores = evid_data.get("quality_scores", {})

                        if use_quality_scores and quality_scores:
                            # Weighted quality score
                            quality = (
                                quality_scores.get("semantic_relevance", 0.5) * 0.4
                                + quality_scores.get("authority", 0.5) * 0.3
                                + quality_scores.get("freshness", 0.5) * 0.2
                                + quality_scores.get("completeness", 0.5) * 0.1
                            )
                        else:
                            quality = 0.5  # Default quality

                        total_quality += quality

                    # Quality-adjusted bonus (diminishing returns via sqrt)
                    if use_quality_scores and total_quality > 0:
                        quality_factor = (total_quality / valid_citations) ** 0.5
                        bonus = evidence_bonus * valid_citations * quality_factor
                    else:
                        bonus = evidence_bonus * valid_citations

                    current_count = vote_counts[canonical]
                    vote_counts[canonical] = current_count + bonus

                    evidence_citations[vote.agent] = valid_citations
                    evidence_quality_totals[vote.agent] = total_quality

                    # Record metrics
                    record_evidence_citation_bonus(agent=vote.agent)

                    logger.debug(
                        f"evidence_citation_bonus agent={vote.agent} "
                        f"citations={valid_citations} quality={total_quality:.2f} bonus={bonus:.3f}"
                    )

        result = ctx.result
        if result is not None and evidence_citations:
            # Store evidence citations and quality in verification_results
            if not result.verification_results:
                result.verification_results = {}
            for agent, count in evidence_citations.items():
                result.verification_results[f"evidence_{agent}"] = count
                if agent in evidence_quality_totals:
                    result.verification_results[f"evidence_quality_{agent}"] = round(
                        evidence_quality_totals[agent], 3
                    )

        if evidence_citations:
            total_quality = sum(evidence_quality_totals.values())
            logger.info(
                f"evidence_weighting applied: {len(evidence_citations)} agents cited evidence, "
                f"total citations={sum(evidence_citations.values())}, "
                f"total quality={total_quality:.2f}"
            )

        return vote_counts

    async def apply_process_evaluation_bonuses(
        self,
        ctx: "DebateContext",
        vote_counts: dict[str, float],
        choice_mapping: dict[str, str],
    ) -> dict[str, float]:
        """Apply process-based evaluation bonuses to vote counts.

        Uses ProcessEvaluator to score each proposal on reasoning quality,
        evidence usage, counterargument consideration, etc. Higher process
        scores result in bonuses to that proposal's vote count.

        This is an Agent-as-a-Judge bias mitigation technique that rewards
        proposals with strong reasoning process, not just persuasive content.

        Args:
            ctx: Debate context with proposals
            vote_counts: Current vote count dictionary
            choice_mapping: Maps vote choices to canonical agent names

        Returns:
            Updated vote counts with process bonuses applied
        """
        if not self.protocol or not getattr(self.protocol, "enable_process_evaluation", False):
            return vote_counts

        proposals = ctx.proposals
        if not proposals:
            return vote_counts

        from aragora.debate.bias_mitigation import ProcessEvaluator

        task = ctx.env.task if ctx.env else ""
        evidence_pack = getattr(ctx, "evidence_pack", None)

        # Create evaluator
        evaluator = ProcessEvaluator()

        # Evaluate each proposal
        process_scores: dict[str, float] = {}
        process_bonus = 0.2  # Max bonus for perfect process score

        for agent_name, proposal in proposals.items():
            try:
                result = await evaluator.evaluate_proposal(
                    agent_name=agent_name,
                    proposal=proposal,
                    task=task,
                    evidence_pack=evidence_pack,
                )

                process_scores[agent_name] = result.weighted_total
                canonical = choice_mapping.get(agent_name, agent_name)

                if canonical in vote_counts:
                    # Apply bonus proportional to process score (0-1)
                    bonus = process_bonus * result.weighted_total
                    vote_counts[canonical] = vote_counts.get(canonical, 0.0) + bonus

                    # Record metrics
                    record_process_evaluation_bonus(agent=agent_name)

                    logger.debug(
                        f"process_evaluation agent={agent_name} "
                        f"overall={result.weighted_total:.2f} bonus={bonus:.2f} "
                        f"criteria={list(result.criterion_scores.keys())}"
                    )

            except Exception as e:
                logger.warning(f"process_evaluation_error agent={agent_name}: {e}")
                # No bonus on error

        if process_scores:
            logger.info(
                f"process_evaluation applied: {len(process_scores)} proposals scored, "
                f"avg={sum(process_scores.values())/len(process_scores):.2f}"
            )

        return vote_counts
