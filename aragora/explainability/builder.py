"""
ExplanationBuilder for constructing Decision entities.

Builds comprehensive Decision objects from debate results by extracting
and analyzing evidence chains, vote influences, belief changes, and
confidence attribution.

Usage:
    from aragora.explainability import ExplanationBuilder

    builder = ExplanationBuilder()
    decision = await builder.build(debate_result)
    explanation = builder.generate_summary(decision)
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .decision import (
    BeliefChange,
    ConfidenceAttribution,
    Counterfactual,
    Decision,
    EvidenceLink,
    VotePivot,
)

logger = logging.getLogger(__name__)


class ExplanationBuilder:
    """
    Builds Decision entities from debate results.

    Extracts and analyzes:
    - Evidence chains from proposals and critiques
    - Vote influences and pivot analysis
    - Belief changes across rounds
    - Confidence attribution factors
    - Counterfactual sensitivity analysis
    """

    def __init__(
        self,
        *,
        evidence_tracker: Optional[Any] = None,
        belief_network: Optional[Any] = None,
        calibration_tracker: Optional[Any] = None,
        elo_system: Optional[Any] = None,
        provenance_tracker: Optional[Any] = None,
    ):
        """
        Initialize the builder with optional tracking systems.

        Args:
            evidence_tracker: EvidenceTracker for grounding scores
            belief_network: BeliefNetwork for belief state analysis
            calibration_tracker: CalibrationTracker for confidence adjustment data
            elo_system: EloSystem for agent skill ratings
            provenance_tracker: ProvenanceTracker for claim lineage
        """
        self.evidence_tracker = evidence_tracker
        self.belief_network = belief_network
        self.calibration_tracker = calibration_tracker
        self.elo_system = elo_system
        self.provenance_tracker = provenance_tracker

    async def build(
        self,
        result: Any,  # DebateResult
        context: Optional[Any] = None,  # DebateContext
        include_counterfactuals: bool = True,
    ) -> Decision:
        """
        Build a Decision entity from a debate result.

        Args:
            result: DebateResult from the debate
            context: Optional DebateContext for additional data
            include_counterfactuals: Whether to compute counterfactuals

        Returns:
            Fully populated Decision entity
        """
        debate_id = getattr(result, "id", "") or self._generate_id(result)

        decision = Decision(
            decision_id="",
            debate_id=debate_id,
            timestamp=datetime.utcnow().isoformat(),
            conclusion=getattr(result, "final_answer", "") or "",
            consensus_reached=getattr(result, "consensus_reached", False),
            confidence=getattr(result, "confidence", 0.0),
            consensus_type=getattr(result, "consensus_type", "majority"),
            task=self._extract_task(result, context),
            domain=self._extract_domain(result, context),
            rounds_used=getattr(result, "rounds_used", 0),
            agents_participated=self._extract_agents(result, context),
        )

        # Build components
        decision.evidence_chain = await self._build_evidence_chain(result, context)
        decision.vote_pivots = self._build_vote_pivots(result, context)
        decision.belief_changes = self._build_belief_changes(result, context)
        decision.confidence_attribution = self._build_confidence_attribution(
            result, context, decision
        )

        if include_counterfactuals:
            decision.counterfactuals = self._build_counterfactuals(result, decision)

        # Compute summary metrics
        decision.evidence_quality_score = self._compute_evidence_quality(decision)
        decision.agent_agreement_score = self._compute_agreement_score(result, decision)
        decision.belief_stability_score = self._compute_belief_stability(decision)

        return decision

    def _generate_id(self, result: Any) -> str:
        """Generate an ID for the debate."""
        task = getattr(result, "task", "")
        ts = datetime.utcnow().isoformat()
        return hashlib.sha256(f"{task}:{ts}".encode()).hexdigest()[:16]

    def _extract_task(self, result: Any, context: Optional[Any]) -> str:
        """Extract task/topic from result or context."""
        if hasattr(result, "task"):
            return result.task
        if context and hasattr(context, "env"):
            return getattr(context.env, "task", "")
        return ""

    def _extract_domain(self, result: Any, context: Optional[Any]) -> str:
        """Extract domain from result or context."""
        if hasattr(result, "domain"):
            return result.domain
        if context and hasattr(context, "domain"):
            return context.domain
        return "general"

    def _extract_agents(self, result: Any, context: Optional[Any]) -> List[str]:
        """Extract list of participating agents."""
        if hasattr(result, "participants"):
            return result.participants
        if hasattr(result, "agents"):
            return [a.name if hasattr(a, "name") else str(a) for a in result.agents]
        if context and hasattr(context, "agents"):
            return [a.name if hasattr(a, "name") else str(a) for a in context.agents]
        return []

    # ==========================================================================
    # Evidence Chain
    # ==========================================================================

    async def _build_evidence_chain(
        self, result: Any, context: Optional[Any]
    ) -> List[EvidenceLink]:
        """Build evidence chain from proposals and critiques."""
        evidence: List[EvidenceLink] = []

        # Extract from proposals
        proposals = getattr(result, "proposals", {})
        for agent, proposal in proposals.items():
            proposal_text = proposal if isinstance(proposal, str) else getattr(proposal, "text", "")

            # Create evidence link for the proposal
            link = EvidenceLink(
                id=f"prop-{agent[:8]}-{hashlib.sha256(proposal_text.encode()).hexdigest()[:8]}",
                content=proposal_text[:500],
                source=agent,
                relevance_score=0.8,  # Proposals are inherently relevant
                grounding_type="argument",
                metadata={"round": 0, "type": "proposal"},
            )

            # Try to get quality scores from evidence tracker
            if self.evidence_tracker:
                try:
                    scores = await self._get_evidence_scores(proposal_text)
                    link.quality_scores = scores
                    link.relevance_score = scores.get("semantic_relevance", 0.8)
                except Exception as e:
                    logger.debug(f"Evidence scoring failed: {e}")

            evidence.append(link)

        # Extract from critiques
        critiques = getattr(result, "critiques", {}) or getattr(result, "all_critiques", {})
        for round_data in critiques.values() if isinstance(critiques, dict) else []:
            for agent, critique in round_data.items() if isinstance(round_data, dict) else []:
                critique_text = (
                    critique if isinstance(critique, str) else getattr(critique, "text", "")
                )

                link = EvidenceLink(
                    id=f"crit-{agent[:8]}-{hashlib.sha256(critique_text.encode()).hexdigest()[:8]}",
                    content=critique_text[:500],
                    source=agent,
                    relevance_score=0.7,
                    grounding_type="critique",
                    metadata={"type": "critique"},
                )
                evidence.append(link)

        # Extract from provenance if available
        if self.provenance_tracker:
            try:
                provenance_evidence = self._extract_provenance_evidence()
                evidence.extend(provenance_evidence)
            except Exception as e:
                logger.debug(f"Provenance extraction failed: {e}")

        return evidence

    async def _get_evidence_scores(self, text: str) -> Dict[str, float]:
        """Get quality scores for evidence text."""
        if not self.evidence_tracker:
            return {}

        try:
            scores = await self.evidence_tracker.score_evidence(text)
            return {
                "semantic_relevance": scores.get("relevance", 0.5),
                "authority": scores.get("authority", 0.5),
                "freshness": scores.get("freshness", 0.5),
                "completeness": scores.get("completeness", 0.5),
            }
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Evidence scoring returned incomplete data: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error during evidence scoring: {e}")
            return {}

    def _extract_provenance_evidence(self) -> List[EvidenceLink]:
        """Extract evidence from provenance tracker."""
        if not self.provenance_tracker:
            return []

        evidence = []
        try:
            claims = self.provenance_tracker.get_all_claims()
            for claim in claims[:20]:  # Limit to top 20
                evidence.append(
                    EvidenceLink(
                        id=f"prov-{claim.id[:8]}",
                        content=claim.content[:500],
                        source=claim.source,
                        relevance_score=claim.confidence,
                        grounding_type="claim",
                        cited_by=claim.cited_by or [],
                        metadata={"provenance": True},
                    )
                )
        except Exception as e:
            logger.debug(f"Provenance claim extraction failed: {e}")

        return evidence

    # ==========================================================================
    # Vote Pivots
    # ==========================================================================

    def _build_vote_pivots(self, result: Any, context: Optional[Any]) -> List[VotePivot]:
        """Build vote pivot analysis."""
        pivots: List[VotePivot] = []

        votes = getattr(result, "votes", [])
        if not votes:
            return pivots

        # Count votes by choice to find consensus
        choice_counts: Counter[str] = Counter()
        total_weight = 0.0
        vote_data: List[Tuple[Any, float]] = []

        for vote in votes:
            choice = getattr(vote, "choice", "")
            agent = getattr(vote, "agent", "")
            confidence = getattr(vote, "confidence", 0.5)

            # Get weight
            weight = self._get_vote_weight(agent)
            choice_counts[choice] += weight
            total_weight += weight
            vote_data.append((vote, weight))

        # Calculate influence scores
        winner = choice_counts.most_common(1)[0][0] if choice_counts else ""

        for vote, weight in vote_data:
            choice = getattr(vote, "choice", "")
            agent = getattr(vote, "agent", "")
            confidence = getattr(vote, "confidence", 0.5)
            reasoning = getattr(vote, "reasoning", "")

            # Influence = how much this vote affected the margin
            choice_counts[winner] - sum(v for k, v in choice_counts.items() if k != winner)
            influence = (weight / total_weight) if total_weight > 0 else 0.0

            # Bonus for votes that match winner
            if choice == winner:
                influence *= 1.5

            # Get calibration adjustment
            calibration_adj = self._get_calibration_adjustment(agent)
            elo = self._get_elo_rating(agent)

            pivots.append(
                VotePivot(
                    agent=agent,
                    choice=choice,
                    confidence=confidence,
                    weight=weight,
                    reasoning_summary=reasoning[:200] if reasoning else "",
                    influence_score=min(1.0, influence),
                    calibration_adjustment=calibration_adj,
                    elo_rating=elo,
                    flip_detected=self._detect_flip(agent, result),
                )
            )

        # Sort by influence
        return sorted(pivots, key=lambda p: p.influence_score, reverse=True)

    def _get_vote_weight(self, agent: str) -> float:
        """Get computed weight for an agent's vote."""
        base = 1.0

        # ELO contribution
        if self.elo_system:
            try:
                elo = self.elo_system.get_rating(agent)
                elo_factor = (elo - 1000) / 500
                base *= max(0.5, min(2.0, 1.0 + elo_factor * 0.3))
            except (KeyError, AttributeError) as e:
                logger.debug(f"ELO rating not available for {agent}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error getting ELO rating for {agent}: {e}")

        # Calibration contribution
        if self.calibration_tracker:
            try:
                calibration = self.calibration_tracker.get_weight(agent)
                base *= calibration
            except (KeyError, AttributeError) as e:
                logger.debug(f"Calibration weight not available for {agent}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error getting calibration for {agent}: {e}")

        return base

    def _get_calibration_adjustment(self, agent: str) -> Optional[float]:
        """Get calibration adjustment for agent."""
        if not self.calibration_tracker:
            return None
        try:
            return self.calibration_tracker.get_adjustment(agent)
        except (KeyError, AttributeError) as e:
            logger.debug(f"Calibration adjustment not available for {agent}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting calibration adjustment for {agent}: {e}")
            return None

    def _get_elo_rating(self, agent: str) -> Optional[float]:
        """Get ELO rating for agent."""
        if not self.elo_system:
            return None
        try:
            return self.elo_system.get_rating(agent)
        except (KeyError, AttributeError) as e:
            logger.debug(f"ELO rating not available for {agent}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting ELO rating for {agent}: {e}")
            return None

    def _detect_flip(self, agent: str, result: Any) -> bool:
        """Detect if agent flipped position during debate."""
        # Check if there's flip detection data
        if hasattr(result, "flip_data"):
            return agent in result.flip_data.get("flipped_agents", [])
        return False

    # ==========================================================================
    # Belief Changes
    # ==========================================================================

    def _build_belief_changes(self, result: Any, context: Optional[Any]) -> List[BeliefChange]:
        """Build belief change analysis."""
        changes: List[BeliefChange] = []

        # Try to get from belief network
        if self.belief_network:
            try:
                network_changes = self.belief_network.get_changes()
                for change in network_changes:
                    changes.append(
                        BeliefChange(
                            agent=change.agent,
                            round=change.round,
                            topic=change.topic,
                            prior_belief=change.prior,
                            posterior_belief=change.posterior,
                            prior_confidence=change.prior_confidence,
                            posterior_confidence=change.posterior_confidence,
                            trigger=change.trigger,
                            trigger_source=change.trigger_source,
                        )
                    )
            except Exception as e:
                logger.debug(f"Belief network extraction failed: {e}")

        # Extract from result's position history if available
        if hasattr(result, "position_history"):
            for agent, history in result.position_history.items():
                for i in range(1, len(history)):
                    prior = history[i - 1]
                    current = history[i]

                    if prior.get("position") != current.get("position"):
                        changes.append(
                            BeliefChange(
                                agent=agent,
                                round=i,
                                topic=getattr(result, "task", ""),
                                prior_belief=prior.get("position", ""),
                                posterior_belief=current.get("position", ""),
                                prior_confidence=prior.get("confidence", 0.5),
                                posterior_confidence=current.get("confidence", 0.5),
                                trigger="critique",
                                trigger_source="debate",
                            )
                        )

        return changes

    # ==========================================================================
    # Confidence Attribution
    # ==========================================================================

    def _build_confidence_attribution(
        self,
        result: Any,
        context: Optional[Any],
        decision: Decision,
    ) -> List[ConfidenceAttribution]:
        """Build confidence attribution analysis."""
        attributions: List[ConfidenceAttribution] = []

        # Consensus strength factor
        if hasattr(result, "consensus_margin"):
            margin = result.consensus_margin
            contribution = margin * 0.4  # Consensus is major factor
            attributions.append(
                ConfidenceAttribution(
                    factor="consensus_strength",
                    contribution=contribution,
                    explanation=f"Agreement level among agents ({margin:.0%} margin)",
                    raw_value=margin,
                )
            )

        # Evidence quality factor
        if decision.evidence_chain:
            avg_quality = sum(e.relevance_score for e in decision.evidence_chain) / len(
                decision.evidence_chain
            )
            contribution = avg_quality * 0.3
            attributions.append(
                ConfidenceAttribution(
                    factor="evidence_quality",
                    contribution=contribution,
                    explanation=f"Quality of supporting evidence ({avg_quality:.0%} average)",
                    raw_value=avg_quality,
                )
            )

        # Agent calibration factor
        if self.calibration_tracker and decision.agents_participated:
            try:
                avg_calibration = sum(
                    self.calibration_tracker.get_weight(a) for a in decision.agents_participated
                ) / len(decision.agents_participated)
                contribution = (avg_calibration - 0.5) * 0.2
                attributions.append(
                    ConfidenceAttribution(
                        factor="agent_calibration",
                        contribution=abs(contribution),
                        explanation="Historical accuracy of participating agents",
                        raw_value=avg_calibration,
                    )
                )
            except (KeyError, AttributeError, ZeroDivisionError) as e:
                logger.debug(f"Calibration factor calculation skipped: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error calculating calibration factor: {e}")

        # Rounds to consensus factor
        rounds_used = decision.rounds_used
        if rounds_used > 0:
            # Faster consensus = higher confidence boost
            rounds_factor = 1.0 - (min(rounds_used, 5) / 5) * 0.5
            contribution = rounds_factor * 0.1
            attributions.append(
                ConfidenceAttribution(
                    factor="debate_efficiency",
                    contribution=contribution,
                    explanation=f"Reached consensus in {rounds_used} rounds",
                    raw_value=float(rounds_used),
                )
            )

        # Normalize contributions to sum to 1.0
        total = sum(a.contribution for a in attributions)
        if total > 0:
            for attr in attributions:
                attr.contribution /= total

        return sorted(attributions, key=lambda a: a.contribution, reverse=True)

    # ==========================================================================
    # Counterfactuals
    # ==========================================================================

    def _build_counterfactuals(self, result: Any, decision: Decision) -> List[Counterfactual]:
        """Build counterfactual analysis."""
        counterfactuals: List[Counterfactual] = []

        # Vote removal counterfactuals
        if decision.vote_pivots:
            top_pivots = decision.vote_pivots[:3]
            for pivot in top_pivots:
                if pivot.influence_score > 0.2:
                    counterfactuals.append(
                        Counterfactual(
                            condition=f"If {pivot.agent} had voted differently",
                            outcome_change="Possible change in consensus or confidence",
                            likelihood=0.3,
                            sensitivity=pivot.influence_score,
                            affected_agents=[pivot.agent],
                        )
                    )

        # Evidence removal counterfactuals
        if decision.evidence_chain:
            top_evidence = decision.get_top_evidence(3)
            for evidence in top_evidence:
                if evidence.relevance_score > 0.7:
                    counterfactuals.append(
                        Counterfactual(
                            condition=f"Without evidence from {evidence.source}",
                            outcome_change="Lower confidence or different conclusion",
                            likelihood=0.2,
                            sensitivity=evidence.relevance_score,
                            affected_agents=evidence.cited_by,
                        )
                    )

        # Agent removal counterfactual
        if len(decision.agents_participated) > 2:
            counterfactuals.append(
                Counterfactual(
                    condition="With fewer participating agents",
                    outcome_change="Potentially lower confidence",
                    likelihood=0.5,
                    sensitivity=0.3,
                    affected_agents=decision.agents_participated,
                )
            )

        return sorted(counterfactuals, key=lambda c: c.sensitivity, reverse=True)

    # ==========================================================================
    # Summary Metrics
    # ==========================================================================

    def _compute_evidence_quality(self, decision: Decision) -> float:
        """Compute overall evidence quality score."""
        if not decision.evidence_chain:
            return 0.0

        scores = [e.relevance_score for e in decision.evidence_chain]
        quality_scores = [
            sum(e.quality_scores.values()) / len(e.quality_scores)
            for e in decision.evidence_chain
            if e.quality_scores
        ]

        all_scores = scores + quality_scores
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def _compute_agreement_score(self, result: Any, decision: Decision) -> float:
        """Compute agent agreement score."""
        if not decision.vote_pivots:
            return 0.0

        # Group by choice
        choices: Counter[str] = Counter()
        for pivot in decision.vote_pivots:
            choices[pivot.choice] += 1

        if not choices:
            return 0.0

        # Agreement = proportion voting for winner
        winner_count = choices.most_common(1)[0][1]
        total = sum(choices.values())

        return winner_count / total if total > 0 else 0.0

    def _compute_belief_stability(self, decision: Decision) -> float:
        """Compute belief stability score (1 = no changes)."""
        if not decision.belief_changes:
            return 1.0

        # More changes = lower stability
        num_changes = len(decision.belief_changes)
        max_expected = len(decision.agents_participated) * decision.rounds_used

        if max_expected == 0:
            return 1.0

        change_ratio = num_changes / max_expected
        return max(0.0, 1.0 - change_ratio)

    # ==========================================================================
    # Summary Generation
    # ==========================================================================

    def generate_summary(self, decision: Decision) -> str:
        """Generate a human-readable summary of the decision."""
        lines = []

        # Header
        consensus_status = "reached" if decision.consensus_reached else "not reached"
        lines.append("## Decision Summary")
        lines.append("")
        lines.append(f"**Consensus:** {consensus_status.title()}")
        lines.append(f"**Confidence:** {decision.confidence:.0%}")
        lines.append(f"**Rounds:** {decision.rounds_used}")
        lines.append("")

        # Conclusion
        if decision.conclusion:
            lines.append("### Conclusion")
            lines.append(decision.conclusion[:500])
            lines.append("")

        # Key evidence
        top_evidence = decision.get_top_evidence(3)
        if top_evidence:
            lines.append("### Key Evidence")
            for e in top_evidence:
                lines.append(f"- **{e.source}**: {e.content[:100]}...")
            lines.append("")

        # Pivotal votes
        pivotal = decision.get_pivotal_votes(0.3)
        if pivotal:
            lines.append("### Most Influential Votes")
            for v in pivotal[:3]:
                lines.append(
                    f"- **{v.agent}** voted '{v.choice}' " f"(influence: {v.influence_score:.0%})"
                )
            lines.append("")

        # Confidence breakdown
        major_factors = decision.get_major_confidence_factors(0.15)
        if major_factors:
            lines.append("### Confidence Factors")
            for f in major_factors:
                lines.append(f"- {f.factor}: {f.explanation}")
            lines.append("")

        return "\n".join(lines)


__all__ = ["ExplanationBuilder"]
