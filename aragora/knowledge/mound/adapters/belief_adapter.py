"""
BeliefAdapter - Bridges the Belief Network to the Knowledge Mound.

This adapter enables bidirectional integration between the reasoning/belief
system and the Knowledge Mound:

- Data flow IN: Converged beliefs, cruxes, and provenance chains stored in KM
- Data flow OUT: Historical cruxes and related claims retrieved for context
- Reverse flow: KM cross-references inform belief network construction

The adapter provides:
- Belief storage after propagation converges
- Crux persistence for cross-debate learning
- Provenance chain verification tracking
- Historical crux retrieval for prediction

ID Prefixes:
- bl_: Belief nodes
- cx_: Crux claims
- pv_: Provenance chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.reasoning.belief import BeliefNetwork, BeliefNode, CruxClaim

logger = logging.getLogger(__name__)


@dataclass
class BeliefSearchResult:
    """Wrapper for belief search results with adapter metadata."""

    belief: Dict[str, Any]
    relevance_score: float = 0.0
    matched_topics: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_topics is None:
            self.matched_topics = []


@dataclass
class CruxSearchResult:
    """Wrapper for crux search results."""

    crux: Dict[str, Any]
    relevance_score: float = 0.0
    debate_ids: List[str] = None

    def __post_init__(self) -> None:
        if self.debate_ids is None:
            self.debate_ids = []


class BeliefAdapter:
    """
    Adapter that bridges BeliefNetwork to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_converged_belief: Store beliefs after propagation
    - store_crux: Store pivotal claims for cross-debate learning
    - store_provenance: Store verified provenance chains
    - search_similar_cruxes: Find historical cruxes on topic
    - search_related_claims: Find claims related by topic

    Usage:
        from aragora.reasoning.belief import BeliefNetwork, CruxClaim
        from aragora.knowledge.mound.adapters import BeliefAdapter

        network = BeliefNetwork(debate_id="debate-123")
        adapter = BeliefAdapter(network)

        # After propagation, store converged beliefs
        result = network.propagate()
        if result.converged:
            adapter.store_converged_beliefs(min_confidence=0.8)

        # Store detected cruxes
        cruxes = detector.detect_cruxes(min_crux_score=0.3)
        for crux in cruxes:
            adapter.store_crux(crux)
    """

    BELIEF_PREFIX = "bl_"
    CRUX_PREFIX = "cx_"
    PROVENANCE_PREFIX = "pv_"

    # Thresholds from plan
    MIN_BELIEF_CONFIDENCE = 0.8  # Only store high-confidence beliefs
    MIN_CRUX_SCORE = 0.3  # Store cruxes above this threshold

    def __init__(
        self,
        network: Optional["BeliefNetwork"] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            network: Optional BeliefNetwork instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._network = network
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._beliefs: Dict[str, Dict[str, Any]] = {}
        self._cruxes: Dict[str, Dict[str, Any]] = {}
        self._provenance: Dict[str, Dict[str, Any]] = {}

        # Indices for fast lookup
        self._debate_beliefs: Dict[str, List[str]] = {}  # debate_id -> [belief_ids]
        self._debate_cruxes: Dict[str, List[str]] = {}  # debate_id -> [crux_ids]
        self._topic_cruxes: Dict[str, List[str]] = {}  # topic -> [crux_ids]

    @property
    def network(self) -> Optional["BeliefNetwork"]:
        """Access the underlying BeliefNetwork."""
        return self._network

    def set_network(self, network: "BeliefNetwork") -> None:
        """Set the belief network to use."""
        self._network = network

    def store_converged_belief(
        self,
        node: "BeliefNode",
        debate_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a converged belief node in the Knowledge Mound.

        Args:
            node: The BeliefNode to store
            debate_id: Optional debate ID (uses network's if not provided)

        Returns:
            The belief ID if stored, None if below threshold
        """
        # Check confidence threshold
        confidence = node.posterior.p_true if hasattr(node.posterior, 'p_true') else 0.5
        if confidence < self.MIN_BELIEF_CONFIDENCE and (1 - confidence) < self.MIN_BELIEF_CONFIDENCE:
            logger.debug(f"Belief {node.node_id} below confidence threshold: {confidence:.2f}")
            return None

        debate_id = debate_id or (self._network.debate_id if self._network else None)
        belief_id = f"{self.BELIEF_PREFIX}{node.node_id}"

        belief_data = {
            "id": belief_id,
            "node_id": node.node_id,
            "claim_id": node.claim_id,
            "claim_statement": node.claim_statement,
            "author": node.author,
            "confidence": confidence,
            "prior_confidence": node.prior.p_true if hasattr(node.prior, 'p_true') else 0.5,
            "status": node.status.value if hasattr(node.status, 'value') else str(node.status),
            "centrality": node.centrality,
            "update_count": node.update_count,
            "debate_id": debate_id,
            "parent_ids": node.parent_ids,
            "child_ids": node.child_ids,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": node.metadata if hasattr(node, 'metadata') else {},
        }

        self._beliefs[belief_id] = belief_data

        # Update indices
        if debate_id:
            if debate_id not in self._debate_beliefs:
                self._debate_beliefs[debate_id] = []
            self._debate_beliefs[debate_id].append(belief_id)

        logger.info(f"Stored converged belief: {belief_id} (confidence={confidence:.2f})")
        return belief_id

    def store_converged_beliefs(
        self,
        min_confidence: float = None,
    ) -> List[str]:
        """
        Store all converged beliefs from the network above threshold.

        Args:
            min_confidence: Minimum confidence threshold (default: MIN_BELIEF_CONFIDENCE)

        Returns:
            List of stored belief IDs
        """
        if not self._network:
            logger.warning("No network set, cannot store beliefs")
            return []

        min_conf = min_confidence or self.MIN_BELIEF_CONFIDENCE
        stored_ids = []

        for node in self._network.nodes.values():
            confidence = node.posterior.p_true if hasattr(node.posterior, 'p_true') else 0.5
            # Store if confidence is high in either direction
            if confidence >= min_conf or (1 - confidence) >= min_conf:
                belief_id = self.store_converged_belief(node)
                if belief_id:
                    stored_ids.append(belief_id)

        logger.info(f"Stored {len(stored_ids)} converged beliefs from network")
        return stored_ids

    def store_crux(
        self,
        crux: "CruxClaim",
        debate_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Store a crux claim in the Knowledge Mound.

        Args:
            crux: The CruxClaim to store
            debate_id: Optional debate ID
            topics: Optional topic tags for indexing

        Returns:
            The crux ID if stored, None if below threshold
        """
        if crux.crux_score < self.MIN_CRUX_SCORE:
            logger.debug(f"Crux {crux.claim_id} below score threshold: {crux.crux_score:.2f}")
            return None

        debate_id = debate_id or (self._network.debate_id if self._network else None)
        crux_id = f"{self.CRUX_PREFIX}{crux.claim_id}"

        crux_data = {
            "id": crux_id,
            "claim_id": crux.claim_id,
            "statement": crux.statement,
            "author": crux.author,
            "crux_score": crux.crux_score,
            "influence_score": crux.influence_score,
            "disagreement_score": crux.disagreement_score,
            "uncertainty_score": crux.uncertainty_score,
            "centrality_score": crux.centrality_score,
            "affected_claims": crux.affected_claims,
            "contesting_agents": crux.contesting_agents,
            "resolution_impact": crux.resolution_impact,
            "debate_id": debate_id,
            "topics": topics or [],
            "created_at": datetime.utcnow().isoformat(),
        }

        self._cruxes[crux_id] = crux_data

        # Update indices
        if debate_id:
            if debate_id not in self._debate_cruxes:
                self._debate_cruxes[debate_id] = []
            self._debate_cruxes[debate_id].append(crux_id)

        for topic in (topics or []):
            topic_lower = topic.lower()
            if topic_lower not in self._topic_cruxes:
                self._topic_cruxes[topic_lower] = []
            self._topic_cruxes[topic_lower].append(crux_id)

        logger.info(f"Stored crux: {crux_id} (score={crux.crux_score:.2f})")
        return crux_id

    def store_provenance(
        self,
        chain_id: str,
        source_id: str,
        claim_ids: List[str],
        verified: bool,
        verification_method: str,
        debate_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a provenance chain in the Knowledge Mound.

        Args:
            chain_id: Unique identifier for the provenance chain
            source_id: ID of the original source
            claim_ids: List of claim IDs in the chain
            verified: Whether the chain is verified
            verification_method: How verification was done
            debate_id: Optional debate ID
            metadata: Optional additional metadata

        Returns:
            The provenance ID if stored, None if not verified
        """
        # Only store verified chains per plan
        if not verified:
            logger.debug(f"Provenance chain {chain_id} not verified, skipping")
            return None

        debate_id = debate_id or (self._network.debate_id if self._network else None)
        prov_id = f"{self.PROVENANCE_PREFIX}{chain_id}"

        prov_data = {
            "id": prov_id,
            "chain_id": chain_id,
            "source_id": source_id,
            "claim_ids": claim_ids,
            "verified": verified,
            "verification_method": verification_method,
            "debate_id": debate_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        self._provenance[prov_id] = prov_data

        logger.info(f"Stored provenance chain: {prov_id}")
        return prov_id

    def get_belief(self, belief_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific belief by ID.

        Args:
            belief_id: The belief ID (may be prefixed with "bl_")

        Returns:
            Belief dict or None
        """
        if not belief_id.startswith(self.BELIEF_PREFIX):
            belief_id = f"{self.BELIEF_PREFIX}{belief_id}"
        return self._beliefs.get(belief_id)

    def get_crux(self, crux_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific crux by ID.

        Args:
            crux_id: The crux ID (may be prefixed with "cx_")

        Returns:
            Crux dict or None
        """
        if not crux_id.startswith(self.CRUX_PREFIX):
            crux_id = f"{self.CRUX_PREFIX}{crux_id}"
        return self._cruxes.get(crux_id)

    def search_similar_cruxes(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find historical cruxes similar to the query.

        This enables cross-debate learning by finding cruxes
        that appeared in previous debates on similar topics.

        Args:
            query: Search query (keywords from claim statement)
            limit: Maximum results to return
            min_score: Minimum crux score threshold

        Returns:
            List of matching crux dicts
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for crux in self._cruxes.values():
            if crux["crux_score"] < min_score:
                continue

            statement_lower = crux["statement"].lower()
            statement_words = set(statement_lower.split())

            # Simple keyword overlap scoring
            overlap = len(query_words & statement_words)
            if overlap > 0:
                relevance = overlap / max(len(query_words), 1)
                results.append({
                    **crux,
                    "relevance_score": relevance,
                })

        # Sort by relevance * crux_score
        results.sort(
            key=lambda x: x["relevance_score"] * x["crux_score"],
            reverse=True,
        )

        return results[:limit]

    def search_cruxes_by_topic(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find cruxes tagged with a specific topic.

        Args:
            topic: Topic to search for
            limit: Maximum results

        Returns:
            List of matching crux dicts
        """
        topic_lower = topic.lower()
        crux_ids = self._topic_cruxes.get(topic_lower, [])

        results = []
        for crux_id in crux_ids[:limit]:
            crux = self._cruxes.get(crux_id)
            if crux:
                results.append(crux)

        return results

    def get_debate_beliefs(
        self,
        debate_id: str,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all beliefs from a specific debate.

        Args:
            debate_id: The debate ID
            min_confidence: Minimum confidence filter

        Returns:
            List of belief dicts
        """
        belief_ids = self._debate_beliefs.get(debate_id, [])
        results = []

        for belief_id in belief_ids:
            belief = self._beliefs.get(belief_id)
            if belief and belief.get("confidence", 0) >= min_confidence:
                results.append(belief)

        return results

    def get_debate_cruxes(
        self,
        debate_id: str,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all cruxes from a specific debate.

        Args:
            debate_id: The debate ID
            min_score: Minimum crux score filter

        Returns:
            List of crux dicts
        """
        crux_ids = self._debate_cruxes.get(debate_id, [])
        results = []

        for crux_id in crux_ids:
            crux = self._cruxes.get(crux_id)
            if crux and crux.get("crux_score", 0) >= min_score:
                results.append(crux)

        return results

    def to_knowledge_item(self, belief: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a belief dict to a KnowledgeItem.

        Args:
            belief: The belief dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map confidence to level
        confidence_val = belief.get("confidence", 0.5)
        if confidence_val >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif confidence_val >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif confidence_val >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_val >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = belief.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        return KnowledgeItem(
            id=belief["id"],
            content=belief.get("claim_statement", ""),
            source=KnowledgeSource.BELIEF,
            source_id=belief.get("node_id", belief["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "author": belief.get("author", ""),
                "debate_id": belief.get("debate_id", ""),
                "centrality": belief.get("centrality", 0.0),
                "prior_confidence": belief.get("prior_confidence", 0.5),
                "update_count": belief.get("update_count", 0),
            },
            importance=confidence_val,
        )

    def crux_to_knowledge_item(self, crux: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a crux dict to a KnowledgeItem.

        Args:
            crux: The crux dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Crux score maps to HIGH confidence (they are pivotal claims)
        crux_score = crux.get("crux_score", 0.5)
        if crux_score >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif crux_score >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        created_at = crux.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        return KnowledgeItem(
            id=crux["id"],
            content=crux.get("statement", ""),
            source=KnowledgeSource.BELIEF,  # Cruxes are a type of belief
            source_id=crux.get("claim_id", crux["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "author": crux.get("author", ""),
                "debate_id": crux.get("debate_id", ""),
                "crux_score": crux_score,
                "influence_score": crux.get("influence_score", 0.0),
                "disagreement_score": crux.get("disagreement_score", 0.0),
                "resolution_impact": crux.get("resolution_impact", 0.0),
                "topics": crux.get("topics", []),
                "is_crux": True,
            },
            importance=crux_score,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored beliefs and cruxes."""
        return {
            "total_beliefs": len(self._beliefs),
            "total_cruxes": len(self._cruxes),
            "total_provenance_chains": len(self._provenance),
            "debates_with_beliefs": len(self._debate_beliefs),
            "debates_with_cruxes": len(self._debate_cruxes),
            "topics_indexed": len(self._topic_cruxes),
        }


__all__ = [
    "BeliefAdapter",
    "BeliefSearchResult",
    "CruxSearchResult",
]
