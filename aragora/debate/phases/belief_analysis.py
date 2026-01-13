"""
Belief network analysis for debate cruxes.

Provides utilities for:
- Building belief networks from debate messages
- Identifying crux claims (key disagreement points)
- Suggesting evidence targets
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import Message
    from aragora.reasoning.belief import BeliefNetwork, BeliefPropagationAnalyzer

logger = logging.getLogger(__name__)

# Lazy-loaded belief classes
_BeliefNetwork = None
_BeliefPropagationAnalyzer = None


def _load_belief_classes():
    """Lazy-load belief classes to avoid circular imports."""
    global _BeliefNetwork, _BeliefPropagationAnalyzer
    if _BeliefPropagationAnalyzer is None:
        try:
            from aragora.reasoning.belief import (
                BeliefNetwork as BN,
                BeliefPropagationAnalyzer as BPA,
            )

            _BeliefNetwork = BN
            _BeliefPropagationAnalyzer = BPA
        except ImportError:
            logger.debug("Belief network module not available")
    return _BeliefNetwork, _BeliefPropagationAnalyzer


@dataclass
class BeliefAnalysisResult:
    """Result of belief network analysis."""

    cruxes: list[dict[str, Any]] = field(default_factory=list)
    evidence_suggestions: list[str] = field(default_factory=list)
    network_size: int = 0
    analysis_error: Optional[str] = None


class DebateBeliefAnalyzer:
    """Analyzes debate messages to identify crux claims and evidence needs.

    This class wraps the belief network functionality to provide
    debate-specific analysis utilities.
    """

    def __init__(
        self,
        propagation_iterations: int = 3,
        crux_threshold: float = 0.6,
        max_claims: int = 20,
    ):
        """Initialize analyzer.

        Args:
            propagation_iterations: Number of belief propagation iterations
            crux_threshold: Threshold for identifying crux claims
            max_claims: Maximum claims to analyze per debate
        """
        self.propagation_iterations = propagation_iterations
        self.crux_threshold = crux_threshold
        self.max_claims = max_claims
        self._network = None

    def analyze_messages(
        self,
        messages: list["Message"],
        top_k_cruxes: int = 3,
        top_k_evidence: int = 3,
    ) -> BeliefAnalysisResult:
        """Analyze debate messages to identify cruxes and evidence needs.

        Args:
            messages: List of debate Message objects
            top_k_cruxes: Number of top crux claims to return
            top_k_evidence: Number of top evidence suggestions to return

        Returns:
            BeliefAnalysisResult with cruxes and suggestions
        """
        result = BeliefAnalysisResult()

        BN, BPA = _load_belief_classes()
        if not BN or not BPA:
            result.analysis_error = "Belief module not available"
            return result

        if not messages:
            return result

        try:
            # Build belief network from messages
            network = BN(max_iterations=self.propagation_iterations)
            claim_count = 0

            for msg in messages:
                if msg.role in ("proposer", "critic") and claim_count < self.max_claims:
                    # Add claims from debate as belief nodes
                    claim_id = f"{msg.agent}_{hash(msg.content[:100])}"
                    network.add_claim(
                        claim_id=claim_id,
                        statement=msg.content[:500],
                        author=msg.agent,
                        initial_confidence=0.5,
                    )
                    claim_count += 1

            result.network_size = len(network.nodes) if hasattr(network, "nodes") else 0

            if result.network_size > 0:
                # Run belief propagation
                network.propagate()

                # Identify cruxes
                analyzer = BPA(network)
                result.cruxes = analyzer.identify_debate_cruxes(top_k=top_k_cruxes)

                # Get evidence suggestions
                suggestions = analyzer.suggest_evidence_targets()
                result.evidence_suggestions = suggestions[:top_k_evidence]

                if result.cruxes:
                    logger.debug(f"belief_cruxes count={len(result.cruxes)}")

        except Exception as e:
            result.analysis_error = str(e)
            logger.warning(f"belief_analysis_error error={e}")

        return result

    def analyze_claims(
        self,
        claims: list[Any],
        top_k_cruxes: int = 3,
    ) -> BeliefAnalysisResult:
        """Analyze grounded claims to identify cruxes.

        Args:
            claims: List of claim objects with statement and confidence
            top_k_cruxes: Number of top crux claims to return

        Returns:
            BeliefAnalysisResult with cruxes
        """
        result = BeliefAnalysisResult()

        BN, BPA = _load_belief_classes()
        if not BPA:
            result.analysis_error = "Belief module not available"
            return result

        if not claims:
            return result

        try:
            analyzer = BPA()

            # Add claims to analyzer
            for claim in claims[: self.max_claims]:
                claim_id = getattr(claim, "claim_id", str(hash(claim.statement[:50])))
                confidence = getattr(claim, "confidence", 0.5)
                analyzer.add_claim(
                    claim_id=claim_id,
                    statement=claim.statement,
                    prior=confidence,
                )
                result.network_size += 1

            # Identify cruxes
            cruxes = analyzer.identify_debate_cruxes(threshold=self.crux_threshold)
            result.cruxes = cruxes

            if cruxes:
                logger.debug(f"belief_cruxes_identified count={len(cruxes)}")
                for crux in cruxes[:3]:
                    claim_preview = crux.get("claim", "unknown")[:60]
                    uncertainty = crux.get("uncertainty", 0)
                    logger.debug(f"belief_crux claim={claim_preview} uncertainty={uncertainty:.2f}")

        except Exception as e:
            result.analysis_error = str(e)
            logger.debug(f"Belief analysis failed: {e}")

        return result
