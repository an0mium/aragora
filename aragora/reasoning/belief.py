"""
Bayesian Belief Propagation Network.

Extends the Claims Kernel with probabilistic graphical model capabilities:
- BeliefNode: Claims with probability distributions over truth values
- BeliefNetwork: Factor graph for message passing
- Loopy belief propagation for cyclic argument graphs
- Centrality analysis for identifying load-bearing claims
- Confidence intervals and uncertainty quantification

This moves aragora from binary accept/reject to nuanced probabilistic reasoning.
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from enum import Enum
import hashlib
import json

from aragora.reasoning.claims import ClaimsKernel, TypedClaim, ClaimType, RelationType


class BeliefStatus(Enum):
    """Status of a belief node."""
    PRIOR = "prior"  # Initial belief before evidence
    UPDATED = "updated"  # Updated via propagation
    CONVERGED = "converged"  # Stable after propagation
    CONTESTED = "contested"  # Multiple conflicting updates


@dataclass
class BeliefDistribution:
    """
    Probability distribution over claim truth values.

    Represents P(claim=true), P(claim=false), and optional
    probability mass for "unknown/undecidable".
    """
    p_true: float = 0.5
    p_false: float = 0.5
    p_unknown: float = 0.0

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        """Ensure probabilities sum to 1."""
        total = self.p_true + self.p_false + self.p_unknown
        if total > 0:
            self.p_true /= total
            self.p_false /= total
            self.p_unknown /= total

    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution."""
        h = 0.0
        for p in [self.p_true, self.p_false, self.p_unknown]:
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def confidence(self) -> float:
        """Confidence as max probability."""
        return max(self.p_true, self.p_false, self.p_unknown)

    @property
    def expected_truth(self) -> float:
        """Expected value treating true=1, false=0, unknown=0.5."""
        return self.p_true + 0.5 * self.p_unknown

    def kl_divergence(self, other: "BeliefDistribution") -> float:
        """KL divergence from self to other."""
        kl = 0.0
        for p_self, p_other in [
            (self.p_true, other.p_true),
            (self.p_false, other.p_false),
            (self.p_unknown, other.p_unknown),
        ]:
            if p_self > 0 and p_other > 0:
                kl += p_self * math.log2(p_self / p_other)
        return kl

    def to_dict(self) -> dict:
        return {
            "p_true": self.p_true,
            "p_false": self.p_false,
            "p_unknown": self.p_unknown,
            "entropy": self.entropy,
            "confidence": self.confidence,
        }

    @classmethod
    def from_confidence(cls, confidence: float, lean_true: bool = True) -> "BeliefDistribution":
        """Create distribution from a confidence score."""
        if lean_true:
            return cls(p_true=confidence, p_false=1-confidence)
        else:
            return cls(p_true=1-confidence, p_false=confidence)

    @classmethod
    def uniform(cls) -> "BeliefDistribution":
        """Create uniform (maximum uncertainty) distribution."""
        return cls(p_true=0.5, p_false=0.5, p_unknown=0.0)


@dataclass
class BeliefNode:
    """
    A node in the belief network representing a claim with uncertainty.

    Wraps a TypedClaim with probabilistic beliefs that can be updated
    via message passing.
    """
    node_id: str
    claim_id: str
    claim_statement: str
    author: str

    # Belief state
    prior: BeliefDistribution = field(default_factory=BeliefDistribution.uniform)
    posterior: BeliefDistribution = field(default_factory=BeliefDistribution.uniform)
    status: BeliefStatus = BeliefStatus.PRIOR

    # Message passing state
    incoming_messages: dict[str, BeliefDistribution] = field(default_factory=dict)
    outgoing_messages: dict[str, BeliefDistribution] = field(default_factory=dict)

    # Graph structure
    parent_ids: list[str] = field(default_factory=list)  # Claims this depends on
    child_ids: list[str] = field(default_factory=list)  # Claims that depend on this

    # Metrics
    centrality: float = 0.0  # How important is this node
    update_count: int = 0
    last_update: Optional[str] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_posterior(self):
        """Compute posterior from prior and incoming messages."""
        # Start with prior
        log_true = math.log(self.prior.p_true + 1e-10)
        log_false = math.log(self.prior.p_false + 1e-10)

        # Multiply in incoming messages (add in log space)
        for msg in self.incoming_messages.values():
            log_true += math.log(msg.p_true + 1e-10)
            log_false += math.log(msg.p_false + 1e-10)

        # Normalize
        max_log = max(log_true, log_false)
        p_true = math.exp(log_true - max_log)
        p_false = math.exp(log_false - max_log)

        self.posterior = BeliefDistribution(p_true=p_true, p_false=p_false)
        self.update_count += 1
        self.last_update = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "claim_id": self.claim_id,
            "claim_statement": self.claim_statement[:200],
            "author": self.author,
            "prior": self.prior.to_dict(),
            "posterior": self.posterior.to_dict(),
            "status": self.status.value,
            "centrality": self.centrality,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "update_count": self.update_count,
        }


@dataclass
class Factor:
    """
    A factor in the belief network representing a relationship.

    Encodes conditional probability P(child | parents) for:
    - SUPPORTS: P(child=T | parent=T) high
    - CONTRADICTS: P(child=T | parent=T) low
    - DEPENDS_ON: Child true requires parent true
    """
    factor_id: str
    relation_type: RelationType
    source_node_id: str
    target_node_id: str
    strength: float = 1.0  # Relationship strength

    def get_factor_potential(
        self,
        source_true: bool,
        target_true: bool,
    ) -> float:
        """
        Get factor potential for given assignments.

        Returns unnormalized probability mass.
        """
        if self.relation_type == RelationType.SUPPORTS:
            # Supporting evidence: if source true, target more likely true
            if source_true and target_true:
                return 0.7 + 0.3 * self.strength
            elif source_true and not target_true:
                return 0.3 - 0.2 * self.strength
            elif not source_true and target_true:
                return 0.4  # Source false doesn't say much about target
            else:
                return 0.5

        elif self.relation_type == RelationType.CONTRADICTS:
            # Contradicting: if source true, target more likely false
            if source_true and target_true:
                return 0.2 - 0.15 * self.strength
            elif source_true and not target_true:
                return 0.8 + 0.15 * self.strength
            elif not source_true and target_true:
                return 0.5
            else:
                return 0.5

        elif self.relation_type == RelationType.DEPENDS_ON:
            # Dependency: target can only be true if source is true
            if not source_true and target_true:
                return 0.1  # Very unlikely
            elif source_true and target_true:
                return 0.8
            else:
                return 0.5

        else:
            # Default: slight positive correlation
            if source_true == target_true:
                return 0.6
            else:
                return 0.4


@dataclass
class PropagationResult:
    """Result of belief propagation."""
    converged: bool
    iterations: int
    max_change: float
    node_posteriors: dict[str, BeliefDistribution]
    centralities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "max_change": self.max_change,
            "node_posteriors": {
                k: v.to_dict() for k, v in self.node_posteriors.items()
            },
            "centralities": self.centralities,
        }


class BeliefNetwork:
    """
    Bayesian belief network for probabilistic debate reasoning.

    Implements loopy belief propagation over a factor graph
    constructed from claims and their relationships.
    """

    def __init__(
        self,
        debate_id: Optional[str] = None,
        damping: float = 0.5,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
    ):
        self.debate_id = debate_id or str(uuid.uuid4())
        self.damping = damping  # For stable convergence
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Graph structure
        self.nodes: dict[str, BeliefNode] = {}
        self.factors: dict[str, Factor] = {}

        # Indices
        self.claim_to_node: dict[str, str] = {}  # claim_id -> node_id
        self.node_factors: dict[str, list[str]] = {}  # node_id -> [factor_ids]

        # Metadata
        self.created_at = datetime.now()
        self.propagation_count = 0

    def add_node_from_claim(
        self,
        claim: TypedClaim,
        prior_confidence: Optional[float] = None,
    ) -> BeliefNode:
        """Add a belief node from a typed claim."""
        node_id = f"bn-{len(self.nodes):04d}"

        # Set prior based on claim confidence
        if prior_confidence is not None:
            prior = BeliefDistribution.from_confidence(prior_confidence)
        else:
            prior = BeliefDistribution.from_confidence(claim.adjusted_confidence)

        node = BeliefNode(
            node_id=node_id,
            claim_id=claim.claim_id,
            claim_statement=claim.statement,
            author=claim.author,
            prior=prior,
            posterior=BeliefDistribution(
                p_true=prior.p_true,
                p_false=prior.p_false,
            ),
            parent_ids=claim.premises.copy(),
        )

        self.nodes[node_id] = node
        self.claim_to_node[claim.claim_id] = node_id
        self.node_factors[node_id] = []

        return node

    def add_factor(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
    ) -> Optional[Factor]:
        """Add a factor representing a relationship between claims."""
        source_node_id = self.claim_to_node.get(source_claim_id)
        target_node_id = self.claim_to_node.get(target_claim_id)

        if not source_node_id or not target_node_id:
            return None

        factor_id = f"f-{len(self.factors):04d}"

        factor = Factor(
            factor_id=factor_id,
            relation_type=relation_type,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            strength=strength,
        )

        self.factors[factor_id] = factor

        # Update indices
        self.node_factors[source_node_id].append(factor_id)
        self.node_factors[target_node_id].append(factor_id)

        # Update node graph structure
        self.nodes[target_node_id].parent_ids.append(source_node_id)
        self.nodes[source_node_id].child_ids.append(target_node_id)

        return factor

    def from_claims_kernel(self, kernel: ClaimsKernel) -> "BeliefNetwork":
        """Build belief network from a claims kernel."""
        # Add all claims as nodes
        for claim in kernel.claims.values():
            self.add_node_from_claim(claim)

        # Add all relations as factors
        for relation in kernel.relations.values():
            self.add_factor(
                source_claim_id=relation.source_claim_id,
                target_claim_id=relation.target_claim_id,
                relation_type=relation.relation_type,
                strength=relation.strength,
            )

        return self

    def propagate(self) -> PropagationResult:
        """
        Run loopy belief propagation to update all posteriors.

        Uses sum-product message passing with damping for
        numerical stability on cyclic graphs.
        """
        self.propagation_count += 1

        # Initialize messages
        for node in self.nodes.values():
            node.incoming_messages.clear()
            node.outgoing_messages.clear()

        converged = False
        iteration = 0
        max_change = float('inf')

        while iteration < self.max_iterations and not converged:
            old_posteriors = {
                nid: BeliefDistribution(
                    p_true=n.posterior.p_true,
                    p_false=n.posterior.p_false,
                )
                for nid, n in self.nodes.items()
            }

            # Send messages along all factors
            for factor in self.factors.values():
                self._send_messages(factor)

            # Update posteriors
            for node in self.nodes.values():
                node.update_posterior()

            # Check convergence
            max_change = 0.0
            for node_id, node in self.nodes.items():
                old = old_posteriors[node_id]
                change = abs(node.posterior.p_true - old.p_true)
                max_change = max(max_change, change)

            converged = max_change < self.convergence_threshold
            iteration += 1

        # Mark nodes as converged
        for node in self.nodes.values():
            node.status = BeliefStatus.CONVERGED if converged else BeliefStatus.UPDATED

        # Compute centralities
        centralities = self._compute_centralities()
        for node_id, centrality in centralities.items():
            self.nodes[node_id].centrality = centrality

        return PropagationResult(
            converged=converged,
            iterations=iteration,
            max_change=max_change,
            node_posteriors={
                nid: node.posterior for nid, node in self.nodes.items()
            },
            centralities=centralities,
        )

    def _send_messages(self, factor: Factor):
        """Send messages from factor to connected nodes."""
        source_node = self.nodes[factor.source_node_id]
        target_node = self.nodes[factor.target_node_id]

        # Message from source to target (through factor)
        # Marginalize out source variable
        msg_to_target = self._compute_message(
            factor, source_node, target_node, to_target=True
        )

        # Apply damping
        if factor.factor_id in target_node.incoming_messages:
            old_msg = target_node.incoming_messages[factor.factor_id]
            msg_to_target = BeliefDistribution(
                p_true=self.damping * old_msg.p_true + (1 - self.damping) * msg_to_target.p_true,
                p_false=self.damping * old_msg.p_false + (1 - self.damping) * msg_to_target.p_false,
            )

        target_node.incoming_messages[factor.factor_id] = msg_to_target

        # Message from target to source
        msg_to_source = self._compute_message(
            factor, source_node, target_node, to_target=False
        )

        if factor.factor_id in source_node.incoming_messages:
            old_msg = source_node.incoming_messages[factor.factor_id]
            msg_to_source = BeliefDistribution(
                p_true=self.damping * old_msg.p_true + (1 - self.damping) * msg_to_source.p_true,
                p_false=self.damping * old_msg.p_false + (1 - self.damping) * msg_to_source.p_false,
            )

        source_node.incoming_messages[factor.factor_id] = msg_to_source

    def _compute_message(
        self,
        factor: Factor,
        source_node: BeliefNode,
        target_node: BeliefNode,
        to_target: bool,
    ) -> BeliefDistribution:
        """Compute message through a factor."""
        if to_target:
            # Marginalize over source to get message to target
            msg_true = (
                source_node.posterior.p_true * factor.get_factor_potential(True, True) +
                source_node.posterior.p_false * factor.get_factor_potential(False, True)
            )
            msg_false = (
                source_node.posterior.p_true * factor.get_factor_potential(True, False) +
                source_node.posterior.p_false * factor.get_factor_potential(False, False)
            )
        else:
            # Marginalize over target to get message to source
            msg_true = (
                target_node.posterior.p_true * factor.get_factor_potential(True, True) +
                target_node.posterior.p_false * factor.get_factor_potential(True, False)
            )
            msg_false = (
                target_node.posterior.p_true * factor.get_factor_potential(False, True) +
                target_node.posterior.p_false * factor.get_factor_potential(False, False)
            )

        return BeliefDistribution(p_true=msg_true, p_false=msg_false)

    def _compute_centralities(self) -> dict[str, float]:
        """
        Compute centrality scores for all nodes.

        Uses a simplified PageRank-like algorithm where:
        - Nodes with many children have high centrality
        - Nodes that affect high-uncertainty outcomes are important
        """
        n = len(self.nodes)
        if n == 0:
            return {}

        # Initialize with uniform centrality
        centralities = {nid: 1.0 / n for nid in self.nodes}

        # Iterate PageRank-style
        damping = 0.85
        for _ in range(20):
            new_centralities = {}
            for node_id, node in self.nodes.items():
                # Base centrality
                rank = (1 - damping) / n

                # Add contribution from children
                for child_id in node.child_ids:
                    if child_id in self.nodes:
                        child = self.nodes[child_id]
                        # Weight by entropy (uncertain nodes are more important)
                        weight = child.posterior.entropy
                        n_parents = len(child.parent_ids) or 1
                        rank += damping * centralities.get(child_id, 0) * weight / n_parents

                new_centralities[node_id] = rank

            # Normalize
            total = sum(new_centralities.values()) or 1
            centralities = {k: v / total for k, v in new_centralities.items()}

        return centralities

    def get_node_by_claim(self, claim_id: str) -> Optional[BeliefNode]:
        """Get belief node for a claim."""
        node_id = self.claim_to_node.get(claim_id)
        return self.nodes.get(node_id) if node_id else None

    def get_most_uncertain_claims(self, limit: int = 5) -> list[tuple[BeliefNode, float]]:
        """Get claims with highest entropy (most uncertain)."""
        scored = [
            (node, node.posterior.entropy)
            for node in self.nodes.values()
        ]
        return sorted(scored, key=lambda x: -x[1])[:limit]

    def get_load_bearing_claims(self, limit: int = 5) -> list[tuple[BeliefNode, float]]:
        """Get claims with highest centrality (most load-bearing)."""
        scored = [
            (node, node.centrality)
            for node in self.nodes.values()
        ]
        return sorted(scored, key=lambda x: -x[1])[:limit]

    def get_contested_claims(self) -> list[BeliefNode]:
        """Get claims where evidence is contradictory."""
        contested = []
        for node in self.nodes.values():
            # Check if incoming messages disagree
            if len(node.incoming_messages) >= 2:
                probs = [m.p_true for m in node.incoming_messages.values()]
                if max(probs) - min(probs) > 0.3:
                    contested.append(node)
        return contested

    def conditional_probability(
        self,
        query_claim_id: str,
        evidence: dict[str, bool],
    ) -> BeliefDistribution:
        """
        Compute P(query | evidence) by conditioning.

        Args:
            query_claim_id: Claim to query
            evidence: Dict of claim_id -> truth value to condition on

        Returns:
            Posterior distribution for query given evidence
        """
        # Set evidence nodes to deterministic
        for claim_id, value in evidence.items():
            node = self.get_node_by_claim(claim_id)
            if node:
                if value:
                    node.posterior = BeliefDistribution(p_true=0.999, p_false=0.001)
                else:
                    node.posterior = BeliefDistribution(p_true=0.001, p_false=0.999)

        # Propagate
        self.propagate()

        # Return query posterior
        query_node = self.get_node_by_claim(query_claim_id)
        return query_node.posterior if query_node else BeliefDistribution.uniform()

    def sensitivity_analysis(
        self,
        target_claim_id: str,
    ) -> dict[str, float]:
        """
        Compute sensitivity of target claim to each other claim.

        Returns dict of claim_id -> sensitivity score (how much
        changing that claim affects the target).
        """
        target_node = self.get_node_by_claim(target_claim_id)
        if not target_node:
            return {}

        sensitivities = {}
        baseline = target_node.posterior.p_true

        for claim_id in self.claim_to_node:
            if claim_id == target_claim_id:
                continue

            # Test sensitivity by setting claim to true
            p_given_true = self.conditional_probability(
                target_claim_id,
                {claim_id: True}
            ).p_true

            # Reset and test with false
            self.propagate()  # Reset
            p_given_false = self.conditional_probability(
                target_claim_id,
                {claim_id: False}
            ).p_true

            # Sensitivity is the difference
            sensitivities[claim_id] = abs(p_given_true - p_given_false)

            # Reset
            self.propagate()

        return sensitivities

    def generate_summary(self) -> str:
        """Generate a text summary of the belief network."""
        lines = [
            f"# Belief Network Summary",
            f"",
            f"**Debate ID:** {self.debate_id}",
            f"**Nodes:** {len(self.nodes)}",
            f"**Factors:** {len(self.factors)}",
            f"**Propagations:** {self.propagation_count}",
            f"",
        ]

        # Most certain claims
        lines.append("## Most Certain Claims")
        lines.append("")
        certain = sorted(
            self.nodes.values(),
            key=lambda n: n.posterior.confidence,
            reverse=True
        )[:5]
        for node in certain:
            verdict = "TRUE" if node.posterior.p_true > 0.5 else "FALSE"
            lines.append(
                f"- [{node.posterior.confidence:.0%} {verdict}] "
                f"**{node.author}**: {node.claim_statement[:80]}..."
            )
        lines.append("")

        # Most uncertain
        lines.append("## Most Uncertain Claims")
        lines.append("")
        uncertain = sorted(
            self.nodes.values(),
            key=lambda n: n.posterior.entropy,
            reverse=True
        )[:5]
        for node in uncertain:
            lines.append(
                f"- [Entropy: {node.posterior.entropy:.2f}] "
                f"**{node.author}**: {node.claim_statement[:80]}..."
            )
        lines.append("")

        # Load-bearing claims
        lines.append("## Load-Bearing Claims (High Centrality)")
        lines.append("")
        for node, centrality in self.get_load_bearing_claims(5):
            lines.append(
                f"- [Centrality: {centrality:.3f}] "
                f"**{node.author}**: {node.claim_statement[:80]}..."
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize network to dictionary."""
        return {
            "debate_id": self.debate_id,
            "created_at": self.created_at.isoformat(),
            "propagation_count": self.propagation_count,
            "damping": self.damping,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "factors": [
                {
                    "factor_id": f.factor_id,
                    "relation_type": f.relation_type.value,
                    "source_node_id": f.source_node_id,
                    "target_node_id": f.target_node_id,
                    "strength": f.strength,
                }
                for f in self.factors.values()
            ],
            "claim_to_node": self.claim_to_node,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class BeliefPropagationAnalyzer:
    """
    High-level analyzer for belief propagation in debates.

    Provides insights into:
    - Which claims need more evidence
    - Where the debate is stuck
    - What would move the needle on key claims
    """

    def __init__(self, network: BeliefNetwork):
        self.network = network

    def identify_debate_cruxes(self, top_k: int = 3) -> list[dict]:
        """
        Identify the key claims that, if resolved, would most
        impact the debate outcome.

        A "crux" is a claim with high centrality and high uncertainty.
        """
        cruxes = []

        for node in self.network.nodes.values():
            # Crux score = centrality * entropy
            score = node.centrality * node.posterior.entropy

            cruxes.append({
                "claim_id": node.claim_id,
                "statement": node.claim_statement,
                "author": node.author,
                "crux_score": score,
                "centrality": node.centrality,
                "entropy": node.posterior.entropy,
                "current_belief": node.posterior.to_dict(),
            })

        return sorted(cruxes, key=lambda x: -x["crux_score"])[:top_k]

    def suggest_evidence_targets(self) -> list[dict]:
        """
        Suggest which claims need more evidence to reduce uncertainty.
        """
        suggestions = []

        for node in self.network.nodes.values():
            # High entropy + high centrality = needs evidence
            if node.posterior.entropy > 0.8 and node.centrality > 0.05:
                suggestions.append({
                    "claim_id": node.claim_id,
                    "statement": node.claim_statement,
                    "author": node.author,
                    "current_uncertainty": node.posterior.entropy,
                    "importance": node.centrality,
                    "suggestion": f"Provide evidence to resolve: '{node.claim_statement[:100]}...'"
                })

        return sorted(suggestions, key=lambda x: -x["importance"])

    def compute_consensus_probability(self) -> dict:
        """
        Estimate probability of reaching consensus.

        Based on average certainty and agreement between claims.
        """
        if not self.network.nodes:
            return {"probability": 0.0, "explanation": "No claims in network"}

        # Average confidence across all claims
        avg_confidence = sum(
            n.posterior.confidence for n in self.network.nodes.values()
        ) / len(self.network.nodes)

        # Count contested claims
        contested = len(self.network.get_contested_claims())
        contest_ratio = contested / len(self.network.nodes)

        # Consensus probability
        consensus_prob = avg_confidence * (1 - contest_ratio)

        return {
            "probability": consensus_prob,
            "average_confidence": avg_confidence,
            "contested_claims": contested,
            "contest_ratio": contest_ratio,
            "explanation": (
                f"Consensus probability {consensus_prob:.0%} based on "
                f"average confidence {avg_confidence:.0%} and "
                f"{contested} contested claims ({contest_ratio:.0%})"
            ),
        }

    def what_if_analysis(
        self,
        hypothetical: dict[str, bool],
    ) -> dict:
        """
        Analyze: "What if these claims were true/false?"

        Returns how the debate state would change.
        """
        # Save current state
        original_posteriors = {
            nid: BeliefDistribution(
                p_true=n.posterior.p_true,
                p_false=n.posterior.p_false,
            )
            for nid, n in self.network.nodes.items()
        }

        # Apply hypothetical and propagate
        for claim_id, value in hypothetical.items():
            node = self.network.get_node_by_claim(claim_id)
            if node:
                if value:
                    node.posterior = BeliefDistribution(p_true=0.99, p_false=0.01)
                else:
                    node.posterior = BeliefDistribution(p_true=0.01, p_false=0.99)

        result = self.network.propagate()

        # Compute changes
        changes = []
        for nid, node in self.network.nodes.items():
            original = original_posteriors[nid]
            delta = node.posterior.p_true - original.p_true
            if abs(delta) > 0.05:
                changes.append({
                    "claim_id": node.claim_id,
                    "statement": node.claim_statement[:100],
                    "original_p_true": original.p_true,
                    "new_p_true": node.posterior.p_true,
                    "delta": delta,
                })

        # Restore original state
        for nid, posterior in original_posteriors.items():
            self.network.nodes[nid].posterior = posterior

        return {
            "hypothetical": hypothetical,
            "affected_claims": len(changes),
            "changes": sorted(changes, key=lambda x: -abs(x["delta"])),
        }
