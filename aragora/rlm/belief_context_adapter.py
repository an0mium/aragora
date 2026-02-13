"""
Belief Context Adapter for RLM Integration.

Bridges the Belief Network's probabilistic reasoning capabilities
with RLM's recursive examination of context. This enables:

1. Belief-guided reasoning: RLM can consult what the system knows
   about claim confidence, contested points, and evidence veracity
2. Uncertainty propagation: RLM results flow back to update beliefs
3. Crux detection: RLM can identify pivotal claims needing investigation

Usage in RLM REPL:
    # These functions are injected into the REPL namespace
    belief = get_belief("claim-123")  # Get confidence for a claim
    cruxes = find_cruxes("rate limiting")  # Find pivotal claims
    conf = evidence_confidence("source-456")  # Check evidence reliability
    uncertain = search_uncertain(min_entropy=0.5)  # Find contested claims
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.reasoning.belief import BeliefNetwork

logger = logging.getLogger(__name__)


@dataclass
class BeliefContextResult:
    """Result of a belief context lookup."""

    claim_id: str
    statement: str | None = None
    confidence: float = 0.5
    p_true: float = 0.5
    p_false: float = 0.25
    p_unknown: float = 0.25
    centrality: float = 0.0
    is_crux: bool = False
    crux_score: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for REPL use."""
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "p_true": self.p_true,
            "p_false": self.p_false,
            "p_unknown": self.p_unknown,
            "centrality": self.centrality,
            "is_crux": self.is_crux,
            "crux_score": self.crux_score,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_claims": self.contradicting_claims,
        }


@dataclass
class CruxResult:
    """Result of crux detection query."""

    claim_id: str
    statement: str
    crux_score: float
    sensitivity: float  # How much conclusion changes if this claim flips
    uncertainty: float  # Current uncertainty in this claim
    related_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "crux_score": self.crux_score,
            "sensitivity": self.sensitivity,
            "uncertainty": self.uncertainty,
            "related_claims": self.related_claims,
        }


class BeliefContextAdapter:
    """
    Injects belief network knowledge into RLM REPL environment.

    Provides functions that RLM can call during recursive examination:
    - get_belief_for_claim: Check confidence in a specific claim
    - find_related_cruxes: Find pivotal claims for a topic
    - get_evidence_confidence: Check reliability of evidence sources
    - search_uncertain_claims: Find contested claims needing clarification
    - get_load_bearing_claims: Find high-centrality claims to verify

    These functions are designed for use in the RLM REPL namespace,
    enabling belief-augmented reasoning during context examination.
    """

    def __init__(
        self,
        belief_network: BeliefNetwork | None = None,
        km_adapter: Any = None,
        min_confidence_threshold: float = 0.7,
        max_results: int = 20,
    ):
        """Initialize the adapter.

        Args:
            belief_network: Optional BeliefNetwork instance to query
            km_adapter: Optional KM adapter for historical belief queries
            min_confidence_threshold: Minimum confidence for relevant beliefs
            max_results: Maximum results to return from queries
        """
        self._network = belief_network
        self._km_adapter = km_adapter
        self._min_confidence = min_confidence_threshold
        self._max_results = max_results

        # Track which beliefs were consulted (for provenance)
        self._consulted_beliefs: list[str] = []
        self._consulted_cruxes: list[str] = []

    def set_belief_network(self, network: BeliefNetwork) -> None:
        """Set or update the belief network."""
        self._network = network

    def reset_tracking(self) -> None:
        """Reset belief consultation tracking."""
        self._consulted_beliefs = []
        self._consulted_cruxes = []

    def get_consulted_beliefs(self) -> list[str]:
        """Get list of belief IDs that were consulted."""
        return list(self._consulted_beliefs)

    def get_consulted_cruxes(self) -> list[str]:
        """Get list of crux IDs that were consulted."""
        return list(self._consulted_cruxes)

    # =========================================================================
    # REPL-Injectable Functions
    # =========================================================================

    def get_belief_for_claim(self, claim_id: str) -> dict[str, Any]:
        """
        Get belief distribution for a claim.

        Use this in RLM REPL to check what the system knows about a claim's
        probability of being true, including confidence and centrality.

        Args:
            claim_id: The claim identifier to look up

        Returns:
            Dict with p_true, confidence, centrality, and related info
        """
        if not self._network:
            return {"error": "No belief network available", "claim_id": claim_id}

        node = self._network.get_node_by_claim(claim_id)
        if not node:
            # Try to find by statement content
            for n in self._network.nodes.values():
                if claim_id in n.claim_id or claim_id.lower() in n.claim_statement.lower():
                    node = n
                    break

        if not node:
            return {
                "claim_id": claim_id,
                "found": False,
                "message": "Claim not in belief network",
            }

        # Track that we consulted this belief
        if node.claim_id not in self._consulted_beliefs:
            self._consulted_beliefs.append(node.claim_id)

        posterior = node.posterior
        result = BeliefContextResult(
            claim_id=node.claim_id,
            statement=node.claim_statement,
            confidence=posterior.confidence,
            p_true=posterior.p_true,
            p_false=posterior.p_false,
            p_unknown=posterior.p_unknown,
            centrality=getattr(node, "centrality", 0.0),
            is_crux=getattr(node, "is_crux", False),
            crux_score=getattr(node, "crux_score", 0.0),
            metadata=node.metadata,
        )

        return result.to_dict()

    def find_related_cruxes(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Find pivotal claims (cruxes) related to a topic.

        Cruxes are claims that, if their truth value changed, would
        significantly impact the conclusion. Use this to identify
        areas needing deeper investigation.

        Args:
            topic: Topic or keyword to search for related cruxes
            limit: Maximum number of cruxes to return

        Returns:
            List of crux dictionaries with scores and related info
        """
        if not self._network:
            return [{"error": "No belief network available"}]

        results = []

        # First, try KM historical cruxes
        if self._network._km_adapter:
            try:
                historical = self._network.query_km_historical_cruxes(topic, limit=limit)
                for h in historical:
                    results.append(
                        CruxResult(
                            claim_id=h.get("claim_id", ""),
                            statement=h.get("statement", ""),
                            crux_score=h.get("crux_score", 0.0),
                            sensitivity=h.get("sensitivity", 0.0),
                            uncertainty=h.get("uncertainty", 0.5),
                            related_claims=h.get("related_claims", []),
                        ).to_dict()
                    )
            except Exception as e:
                logger.debug(f"KM crux query failed: {e}")

        # Then check current network for topic-related high-centrality claims
        topic_lower = topic.lower()
        for node in self._network.nodes.values():
            if topic_lower in node.claim_statement.lower():
                centrality = getattr(node, "centrality", 0.0)
                entropy = node.posterior.entropy if hasattr(node.posterior, "entropy") else 0.5

                # High centrality + high uncertainty = likely crux
                crux_score = centrality * entropy

                if crux_score > 0.1:  # Threshold for relevance
                    results.append(
                        CruxResult(
                            claim_id=node.claim_id,
                            statement=node.claim_statement,
                            crux_score=crux_score,
                            sensitivity=centrality,
                            uncertainty=entropy,
                        ).to_dict()
                    )

                    # Track consultation
                    if node.claim_id not in self._consulted_cruxes:
                        self._consulted_cruxes.append(node.claim_id)

        # Sort by crux score and limit
        results.sort(key=lambda x: x.get("crux_score", 0), reverse=True)
        return results[:limit]

    def get_evidence_confidence(self, source_id: str) -> dict[str, Any]:
        """
        Get confidence in an evidence source from provenance tracking.

        Use this to check if evidence has been verified, its transformation
        history, and overall reliability score.

        Args:
            source_id: Evidence source identifier

        Returns:
            Dict with confidence score and provenance info
        """
        # Check if provenance tracking is available
        try:
            from aragora.reasoning.provenance import get_provenance_store  # type: ignore[attr-defined]

            store = get_provenance_store()
            if store:
                record = store.get_record(source_id)
                if record:
                    return {
                        "source_id": source_id,
                        "found": True,
                        "confidence": record.confidence,
                        "source_type": record.source_type.value
                        if record.source_type
                        else "unknown",
                        "transformations": [t.value for t in record.transformations]
                        if record.transformations
                        else [],
                        "verified": record.verified,
                        "verification_count": record.verification_count,
                    }
        except (ImportError, AttributeError) as e:
            logger.debug(f"Provenance lookup not available: {e}")

        return {
            "source_id": source_id,
            "found": False,
            "confidence": 0.5,
            "message": "Evidence not in provenance store",
        }

    def search_uncertain_claims(self, min_entropy: float = 0.5) -> list[dict[str, Any]]:
        """
        Find claims with high uncertainty (contested/undecided).

        These are claims where the belief distribution is spread out,
        indicating disagreement or lack of evidence.

        Args:
            min_entropy: Minimum entropy threshold (0-1, higher = more uncertain)

        Returns:
            List of uncertain claim dictionaries
        """
        if not self._network:
            return [{"error": "No belief network available"}]

        uncertain = self._network.get_most_uncertain_claims(limit=self._max_results)

        results = []
        for node, entropy in uncertain:
            if entropy >= min_entropy:
                results.append(
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "entropy": entropy,
                        "p_true": node.posterior.p_true,
                        "p_false": node.posterior.p_false,
                        "needs_clarification": True,
                    }
                )

        return results

    def get_load_bearing_claims(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get claims with high centrality (importance to conclusions).

        These are claims that many other claims depend on. Verifying
        these has high impact on overall confidence.

        Args:
            limit: Maximum number of claims to return

        Returns:
            List of high-centrality claim dictionaries
        """
        if not self._network:
            return [{"error": "No belief network available"}]

        # Get centrality scores
        centralities = self._network._compute_centralities()

        results = []
        for node_id, centrality in sorted(centralities.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]:
            node = self._network.nodes.get(node_id)
            if node:
                results.append(
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "centrality": centrality,
                        "confidence": node.posterior.confidence,
                        "dependents": len(
                            [
                                f
                                for f in self._network.factors.values()
                                if node_id in (f.source_node_id, f.target_node_id)
                            ]
                        ),
                    }
                )

        return results

    def query_related_beliefs(
        self, topic: str, limit: int = 10, min_confidence: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Query for beliefs related to a topic from both network and KM.

        Args:
            topic: Topic to search for
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of related belief dictionaries
        """
        if not self._network:
            return [{"error": "No belief network available"}]

        # Query KM if available
        try:
            km_beliefs = self._network.query_km_related_beliefs(
                topic=topic,
                limit=limit,
                min_confidence=min_confidence,
            )
            return km_beliefs
        except Exception as e:
            logger.debug(f"KM belief query failed: {e}")

        # Fallback to local network search
        topic_lower = topic.lower()
        results = []
        for node in self._network.nodes.values():
            if (
                topic_lower in node.claim_statement.lower()
                and node.posterior.confidence >= min_confidence
            ):
                results.append(
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "confidence": node.posterior.confidence,
                        "p_true": node.posterior.p_true,
                    }
                )
        return results[:limit]

    # =========================================================================
    # REPL Namespace Injection
    # =========================================================================

    def get_repl_namespace(self) -> dict[str, Any]:
        """
        Get dictionary of functions to inject into RLM REPL namespace.

        Returns:
            Dict mapping function names to callable functions
        """
        return {
            "get_belief": self.get_belief_for_claim,
            "find_cruxes": self.find_related_cruxes,
            "evidence_confidence": self.get_evidence_confidence,
            "search_uncertain": self.search_uncertain_claims,
            "load_bearing_claims": self.get_load_bearing_claims,
            "related_beliefs": self.query_related_beliefs,
        }

    # =========================================================================
    # Context Building
    # =========================================================================

    def build_belief_context_summary(self, topic: str) -> str:
        """
        Build a text summary of belief context for RLM prompt augmentation.

        Args:
            topic: The topic being investigated

        Returns:
            Formatted text summary of relevant beliefs and cruxes
        """
        lines = ["## Belief Network Context\n"]

        # Related beliefs
        beliefs = self.query_related_beliefs(topic, limit=5)
        if beliefs and not beliefs[0].get("error"):
            lines.append("### Prior Beliefs (from institutional memory):\n")
            for b in beliefs[:5]:
                conf = b.get("confidence", 0.5)
                lines.append(f"- [{conf:.0%} confident] {b.get('statement', 'Unknown')}")
            lines.append("")

        # Cruxes
        cruxes = self.find_related_cruxes(topic, limit=3)
        if cruxes and not cruxes[0].get("error"):
            lines.append("### Critical Cruxes (debate-pivotal claims):\n")
            for c in cruxes[:3]:
                score = c.get("crux_score", 0)
                lines.append(f"- [{score:.0%} pivotal] {c.get('statement', 'Unknown')}")
            lines.append("")

        # Uncertain claims
        uncertain = self.search_uncertain_claims(min_entropy=0.6)
        if uncertain and not uncertain[0].get("error"):
            lines.append("### Contested Claims (high uncertainty):\n")
            for u in uncertain[:3]:
                entropy = u.get("entropy", 0.5)
                lines.append(f"- [{entropy:.0%} uncertain] {u.get('statement', 'Unknown')}")
            lines.append("")

        return "\n".join(lines)


# Factory function
def get_belief_context_adapter(
    belief_network: BeliefNetwork | None = None,
) -> BeliefContextAdapter:
    """Get or create a belief context adapter instance."""
    return BeliefContextAdapter(belief_network=belief_network)
