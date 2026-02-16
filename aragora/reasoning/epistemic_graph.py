"""Cross-Debate Epistemic Graph — belief inheritance across debates.

Bridges the gap between per-debate BeliefNetworks and organizational knowledge
by enabling belief inheritance: outcomes from past debates become priors for
future debates on related topics.

Architecture:
    ConsensusProof (debate N) → EpistemicGraph.absorb_consensus()
        → BeliefNode with posterior from consensus confidence
        → Stored in Knowledge Mound via BeliefAdapter

    New debate (N+1) → EpistemicGraph.inject_priors(topic)
        → Queries KM for related beliefs from past debates
        → Seeds BeliefNetwork with inherited priors
        → Agents start with institutional knowledge instead of flat priors

This creates organizational memory where beliefs compound across debates,
high-confidence conclusions become strong priors, and dissent from past
debates surfaces as uncertainty in future ones.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InheritedBelief:
    """A belief inherited from a prior debate for seeding new debates."""

    belief_id: str
    statement: str
    confidence: float  # 0.0-1.0 (from consensus confidence)
    source_debate_id: str
    source_type: str = "consensus"  # consensus, dissent, partial_consensus
    domain: str = ""
    supporting_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    decay_factor: float = 1.0  # Decays over time (0.0-1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_confidence(self) -> float:
        """Confidence after applying temporal decay."""
        return self.confidence * self.decay_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "source_debate_id": self.source_debate_id,
            "source_type": self.source_type,
            "domain": self.domain,
            "supporting_agents": self.supporting_agents,
            "dissenting_agents": self.dissenting_agents,
            "decay_factor": self.decay_factor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InheritedBelief:
        return cls(
            belief_id=data.get("belief_id", ""),
            statement=data.get("statement", ""),
            confidence=data.get("confidence", 0.5),
            source_debate_id=data.get("source_debate_id", ""),
            source_type=data.get("source_type", "consensus"),
            domain=data.get("domain", ""),
            supporting_agents=data.get("supporting_agents", []),
            dissenting_agents=data.get("dissenting_agents", []),
            decay_factor=data.get("decay_factor", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BeliefEdge:
    """A relationship between two inherited beliefs."""

    source_id: str
    target_id: str
    relation: str  # supports, contradicts, refines, supersedes
    strength: float = 1.0
    source_debate_id: str = ""


class EpistemicGraph:
    """Cross-debate belief graph with inheritance and decay.

    Maintains a growing graph of organizational beliefs that spans
    multiple debates. Each debate can:
    1. Absorb beliefs from its consensus (absorb_consensus)
    2. Inject prior beliefs from the graph into new debates (inject_priors)
    3. Record dissent as reduced confidence (absorb_dissent)
    4. Track belief supersession when new evidence overrides old beliefs
    """

    def __init__(
        self,
        belief_adapter: Any | None = None,
        decay_rate: float = 0.95,  # Per-debate confidence decay
        min_confidence: float = 0.1,  # Below this, beliefs are pruned
        max_priors: int = 10,  # Max beliefs injected per debate
    ):
        self.belief_adapter = belief_adapter
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.max_priors = max_priors

        # In-memory graph (also persisted to KM via belief_adapter)
        self._beliefs: dict[str, InheritedBelief] = {}
        self._edges: list[BeliefEdge] = []
        # Index: domain → [belief_ids]
        self._domain_index: dict[str, list[str]] = {}

    def absorb_consensus(
        self,
        debate_id: str,
        final_claim: str,
        confidence: float,
        domain: str = "",
        supporting_agents: list[str] | None = None,
        dissenting_agents: list[str] | None = None,
        claims: list[dict[str, Any]] | None = None,
    ) -> list[InheritedBelief]:
        """Absorb consensus outcomes from a completed debate into the graph.

        The final claim becomes a high-confidence belief. Individual claims
        become supporting beliefs linked to the main claim.
        """
        absorbed: list[InheritedBelief] = []

        # Main consensus belief
        main_belief = InheritedBelief(
            belief_id=f"belief_{uuid.uuid4().hex[:12]}",
            statement=final_claim,
            confidence=confidence,
            source_debate_id=debate_id,
            source_type="consensus",
            domain=domain,
            supporting_agents=supporting_agents or [],
            dissenting_agents=dissenting_agents or [],
        )
        self._add_belief(main_belief)
        absorbed.append(main_belief)

        # Absorb individual claims as supporting beliefs
        for claim_data in (claims or []):
            statement = claim_data.get("statement", "")
            if not statement:
                continue
            claim_confidence = claim_data.get("confidence", 0.5)
            claim_belief = InheritedBelief(
                belief_id=f"belief_{uuid.uuid4().hex[:12]}",
                statement=statement,
                confidence=claim_confidence * confidence,  # Modulated by consensus confidence
                source_debate_id=debate_id,
                source_type="claim",
                domain=domain,
                supporting_agents=[claim_data.get("author", "")],
            )
            self._add_belief(claim_belief)
            absorbed.append(claim_belief)

            # Link claim → consensus
            self._edges.append(BeliefEdge(
                source_id=claim_belief.belief_id,
                target_id=main_belief.belief_id,
                relation="supports",
                strength=claim_confidence,
                source_debate_id=debate_id,
            ))

        # Persist to KM if adapter available
        if self.belief_adapter:
            self._persist_beliefs(absorbed)

        logger.info(
            "epistemic_graph_absorb debate=%s beliefs=%d confidence=%.2f",
            debate_id, len(absorbed), confidence,
        )
        return absorbed

    def absorb_dissent(
        self,
        debate_id: str,
        dissent_statement: str,
        dissenting_agent: str,
        severity: float = 0.5,
        related_belief_id: str | None = None,
        domain: str = "",
    ) -> InheritedBelief:
        """Absorb a dissenting view, reducing confidence in related beliefs."""
        dissent_belief = InheritedBelief(
            belief_id=f"belief_{uuid.uuid4().hex[:12]}",
            statement=dissent_statement,
            confidence=severity,  # Higher severity = stronger dissent
            source_debate_id=debate_id,
            source_type="dissent",
            domain=domain,
            dissenting_agents=[dissenting_agent],
        )
        self._add_belief(dissent_belief)

        # Link dissent → contradicts related belief
        if related_belief_id and related_belief_id in self._beliefs:
            self._edges.append(BeliefEdge(
                source_id=dissent_belief.belief_id,
                target_id=related_belief_id,
                relation="contradicts",
                strength=severity,
                source_debate_id=debate_id,
            ))
            # Reduce confidence in the contradicted belief
            related = self._beliefs[related_belief_id]
            related.confidence *= (1.0 - severity * 0.3)

        return dissent_belief

    def inject_priors(
        self,
        topic: str,
        domain: str = "",
        limit: int | None = None,
    ) -> list[InheritedBelief]:
        """Get beliefs relevant to a topic for seeding a new debate.

        Returns beliefs from past debates that are relevant to the topic,
        sorted by effective confidence (confidence * decay).

        These become priors in the new debate's BeliefNetwork.
        """
        limit = limit or self.max_priors

        # Apply temporal decay to all beliefs
        self._apply_decay()

        candidates: list[InheritedBelief] = []

        # First: try KM-based semantic search
        if self.belief_adapter:
            try:
                km_results = self.belief_adapter.search_beliefs(
                    query=topic,
                    limit=limit * 2,  # Fetch extra for filtering
                    min_confidence=self.min_confidence,
                )
                for result in km_results:
                    belief = InheritedBelief.from_dict(result)
                    if belief.effective_confidence >= self.min_confidence:
                        candidates.append(belief)
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug("KM belief search failed: %s", e)

        # Fallback: in-memory domain filtering + keyword matching
        if not candidates:
            topic_lower = topic.lower()
            for belief in self._beliefs.values():
                if belief.effective_confidence < self.min_confidence:
                    continue
                # Domain match
                if domain and belief.domain and belief.domain != domain:
                    continue
                # Simple keyword relevance
                if any(word in belief.statement.lower() for word in topic_lower.split()[:5]):
                    candidates.append(belief)
                # Domain match without keyword (lower priority)
                elif domain and belief.domain == domain:
                    candidates.append(belief)

        # Sort by effective confidence (highest first)
        candidates.sort(key=lambda b: b.effective_confidence, reverse=True)

        priors = candidates[:limit]
        if priors:
            logger.info(
                "epistemic_graph_inject topic=%s priors=%d top_confidence=%.2f",
                topic[:50], len(priors),
                priors[0].effective_confidence if priors else 0.0,
            )
        return priors

    def get_belief(self, belief_id: str) -> InheritedBelief | None:
        """Get a specific belief by ID."""
        return self._beliefs.get(belief_id)

    def get_edges_for(self, belief_id: str) -> list[BeliefEdge]:
        """Get all edges involving a belief."""
        return [
            e for e in self._edges
            if e.source_id == belief_id or e.target_id == belief_id
        ]

    def get_contradictions(self, belief_id: str) -> list[InheritedBelief]:
        """Get beliefs that contradict a given belief."""
        contradicting_ids = [
            e.source_id for e in self._edges
            if e.target_id == belief_id and e.relation == "contradicts"
        ] + [
            e.target_id for e in self._edges
            if e.source_id == belief_id and e.relation == "contradicts"
        ]
        return [
            self._beliefs[bid] for bid in contradicting_ids
            if bid in self._beliefs
        ]

    def supersede(
        self, old_belief_id: str, new_belief_id: str, debate_id: str = ""
    ) -> None:
        """Mark an old belief as superseded by a new one."""
        if old_belief_id in self._beliefs:
            old = self._beliefs[old_belief_id]
            old.confidence *= 0.1  # Drastically reduce old belief
            old.metadata["superseded_by"] = new_belief_id

        self._edges.append(BeliefEdge(
            source_id=new_belief_id,
            target_id=old_belief_id,
            relation="supersedes",
            strength=1.0,
            source_debate_id=debate_id,
        ))

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_beliefs": len(self._beliefs),
            "total_edges": len(self._edges),
            "domains": list(self._domain_index.keys()),
            "avg_confidence": (
                sum(b.confidence for b in self._beliefs.values()) / len(self._beliefs)
                if self._beliefs else 0.0
            ),
            "by_type": {
                source_type: sum(
                    1 for b in self._beliefs.values()
                    if b.source_type == source_type
                )
                for source_type in {"consensus", "claim", "dissent"}
            },
        }

    # --- Internal methods ---

    def _add_belief(self, belief: InheritedBelief) -> None:
        """Add a belief to the graph and update indices."""
        self._beliefs[belief.belief_id] = belief
        if belief.domain:
            self._domain_index.setdefault(belief.domain, []).append(belief.belief_id)

    def _apply_decay(self) -> None:
        """Apply temporal decay to all beliefs and prune dead ones."""
        to_remove: list[str] = []
        for belief_id, belief in self._beliefs.items():
            belief.decay_factor *= self.decay_rate
            if belief.effective_confidence < self.min_confidence:
                to_remove.append(belief_id)

        for belief_id in to_remove:
            belief = self._beliefs.pop(belief_id, None)
            if belief and belief.domain and belief.domain in self._domain_index:
                domain_beliefs = self._domain_index[belief.domain]
                if belief_id in domain_beliefs:
                    domain_beliefs.remove(belief_id)

    def _persist_beliefs(self, beliefs: list[InheritedBelief]) -> None:
        """Persist beliefs to Knowledge Mound via adapter."""
        for belief in beliefs:
            try:
                self.belief_adapter.store_belief(
                    belief_id=belief.belief_id,
                    statement=belief.statement,
                    confidence=belief.confidence,
                    domain=belief.domain,
                    metadata={
                        "source_debate_id": belief.source_debate_id,
                        "source_type": belief.source_type,
                        "supporting_agents": belief.supporting_agents,
                        "dissenting_agents": belief.dissenting_agents,
                    },
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug("Failed to persist belief %s: %s", belief.belief_id, e)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph for persistence."""
        return {
            "beliefs": {k: v.to_dict() for k, v in self._beliefs.items()},
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "strength": e.strength,
                    "source_debate_id": e.source_debate_id,
                }
                for e in self._edges
            ],
            "decay_rate": self.decay_rate,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], belief_adapter: Any = None) -> EpistemicGraph:
        """Deserialize a graph from persisted data."""
        graph = cls(
            belief_adapter=belief_adapter,
            decay_rate=data.get("decay_rate", 0.95),
            min_confidence=data.get("min_confidence", 0.1),
        )
        for belief_data in data.get("beliefs", {}).values():
            belief = InheritedBelief.from_dict(belief_data)
            graph._add_belief(belief)
        for edge_data in data.get("edges", []):
            graph._edges.append(BeliefEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relation=edge_data.get("relation", "supports"),
                strength=edge_data.get("strength", 1.0),
                source_debate_id=edge_data.get("source_debate_id", ""),
            ))
        return graph


# Module-level singleton
_epistemic_graph: EpistemicGraph | None = None


def get_epistemic_graph() -> EpistemicGraph:
    """Get the global epistemic graph instance."""
    global _epistemic_graph
    if _epistemic_graph is None:
        _epistemic_graph = EpistemicGraph()
    return _epistemic_graph


def reset_epistemic_graph() -> None:
    """Reset the global epistemic graph (for testing)."""
    global _epistemic_graph
    _epistemic_graph = None
