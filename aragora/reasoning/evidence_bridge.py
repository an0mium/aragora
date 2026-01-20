"""
Evidence-Provenance Bridge - Connects evidence collection to reasoning systems.

This module bridges the gap between the evidence collection system
(`aragora.evidence`) and the reasoning/provenance system (`aragora.reasoning`),
enabling:

1. Evidence → Provenance: Register evidence snippets as provenance records
2. Evidence → Belief: Update belief distributions based on evidence strength
3. Evidence chains: Track how evidence supports or contradicts claims

Usage:
    from aragora.reasoning.evidence_bridge import EvidenceProvenanceBridge

    bridge = EvidenceProvenanceBridge()

    # Register evidence and get provenance record
    record = bridge.register_evidence(snippet)

    # Link evidence to a claim
    bridge.link_evidence_to_claim(snippet, claim_id)

    # Update beliefs based on evidence
    updated_belief = bridge.update_belief_from_evidence(belief, evidence_list)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from aragora.evidence.collector import EvidenceSnippet
from aragora.reasoning.belief import BeliefDistribution
from aragora.reasoning.provenance import (
    ProvenanceManager,
    ProvenanceRecord,
    SourceType,
    TransformationType,
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceLink:
    """Link between evidence and a claim/belief."""

    evidence_id: str
    provenance_id: str
    claim_id: str
    relevance: float = 1.0
    support_direction: float = 1.0  # +1 supports, -1 contradicts, 0 neutral
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvidenceImpact:
    """Impact of evidence on a belief distribution."""

    original_belief: BeliefDistribution
    updated_belief: BeliefDistribution
    evidence_count: int
    total_weight: float
    avg_relevance: float
    direction: str  # "supporting", "contradicting", "mixed", "neutral"


class EvidenceProvenanceBridge:
    """
    Bridge between evidence collection and provenance/belief systems.

    Provides bidirectional linkage:
    - Evidence → Provenance: Converts evidence snippets to provenance records
    - Evidence → Belief: Updates belief distributions based on evidence

    Example:
        bridge = EvidenceProvenanceBridge()

        # Register evidence as provenance
        record = bridge.register_evidence(snippet)

        # Link to claim
        bridge.link_evidence_to_claim(snippet, "claim_123")

        # Update belief based on evidence
        updated = bridge.update_belief_from_evidence(
            current_belief,
            [evidence1, evidence2]
        )
    """

    # Map evidence source names to provenance source types
    SOURCE_TYPE_MAP = {
        "local_docs": SourceType.DOCUMENT,
        "github": SourceType.CODE_ANALYSIS,
        "web": SourceType.WEB_SEARCH,
        "api": SourceType.EXTERNAL_API,
        "database": SourceType.DATABASE,
        "user": SourceType.USER_PROVIDED,
        "agent": SourceType.AGENT_GENERATED,
    }

    def __init__(
        self,
        provenance_manager: Optional[ProvenanceManager] = None,
    ):
        """
        Initialize the evidence-provenance bridge.

        Args:
            provenance_manager: Optional provenance manager for chain tracking
        """
        self._provenance_manager = provenance_manager or ProvenanceManager()
        self._evidence_to_provenance: dict[str, str] = {}  # evidence_id -> provenance_id
        self._provenance_records: dict[str, ProvenanceRecord] = {}  # provenance_id -> record
        self._claim_links: dict[str, list[EvidenceLink]] = {}  # claim_id -> links
        self._evidence_links: dict[str, list[EvidenceLink]] = {}  # evidence_id -> links
        self._chains: dict[str, list[ProvenanceRecord]] = {}  # chain_id -> records

    @property
    def provenance_manager(self) -> ProvenanceManager:
        """Get the underlying provenance manager."""
        return self._provenance_manager

    def register_evidence(
        self,
        snippet: EvidenceSnippet,
        chain_id: Optional[str] = None,
    ) -> ProvenanceRecord:
        """
        Register an evidence snippet as a provenance record.

        Converts the evidence snippet into a cryptographically tracked
        provenance record for chain of custody.

        Args:
            snippet: Evidence snippet to register
            chain_id: Optional chain ID for linking to existing chain

        Returns:
            ProvenanceRecord with cryptographic hash
        """
        # Determine source type from evidence source
        source_type = self._map_source_type(snippet.source)

        # Create provenance record
        record = ProvenanceRecord(
            id=f"ev_{snippet.id}",
            content_hash="",  # Will be computed in __post_init__
            source_type=source_type,
            source_id=snippet.url or snippet.source,
            content=snippet.snippet,
            content_type="text",
            timestamp=snippet.fetched_at,
            transformation=TransformationType.ORIGINAL,
            metadata={
                "title": snippet.title,
                "reliability_score": snippet.reliability_score,
                "freshness_score": snippet.freshness_score,
                "original_metadata": snippet.metadata,
            },
            confidence=snippet.reliability_score,
        )

        # Store record locally and add to provenance manager's chain
        self._provenance_records[record.id] = record

        # Add to chain tracking
        if chain_id and chain_id in self._chains:
            self._chains[chain_id].append(record)
        else:
            # Create new chain for this evidence
            self._chains[record.id] = [record]

        # Track mapping
        self._evidence_to_provenance[snippet.id] = record.id

        logger.debug(f"Registered evidence {snippet.id} as provenance {record.id}")
        return record

    def get_provenance_for_evidence(self, evidence_id: str) -> Optional[ProvenanceRecord]:
        """
        Get the provenance record for an evidence snippet.

        Args:
            evidence_id: ID of the evidence snippet

        Returns:
            ProvenanceRecord if registered, None otherwise
        """
        provenance_id = self._evidence_to_provenance.get(evidence_id)
        if not provenance_id:
            return None

        return self._provenance_records.get(provenance_id)

    def link_evidence_to_claim(
        self,
        snippet: EvidenceSnippet,
        claim_id: str,
        relevance: float = 1.0,
        support_direction: float = 1.0,
    ) -> EvidenceLink:
        """
        Link evidence to a claim for provenance tracking.

        Args:
            snippet: Evidence snippet
            claim_id: ID of the claim being supported/contradicted
            relevance: How relevant this evidence is (0-1)
            support_direction: +1 supports, -1 contradicts, 0 neutral

        Returns:
            EvidenceLink object
        """
        # Ensure evidence is registered
        provenance_id = self._evidence_to_provenance.get(snippet.id)
        if not provenance_id:
            record = self.register_evidence(snippet)
            provenance_id = record.id

        # Calculate weight based on reliability and freshness
        weight = (snippet.reliability_score * 0.7 + snippet.freshness_score * 0.3) * relevance

        link = EvidenceLink(
            evidence_id=snippet.id,
            provenance_id=provenance_id,
            claim_id=claim_id,
            relevance=relevance,
            support_direction=support_direction,
            weight=weight,
        )

        # Store link for both evidence and claim lookups
        if claim_id not in self._claim_links:
            self._claim_links[claim_id] = []
        self._claim_links[claim_id].append(link)

        if snippet.id not in self._evidence_links:
            self._evidence_links[snippet.id] = []
        self._evidence_links[snippet.id].append(link)

        # Create citation in provenance manager
        self._provenance_manager.cite_evidence(
            claim_id=claim_id,
            evidence_id=provenance_id,
            relevance=relevance,
            support_type="supports" if support_direction > 0 else "contradicts",
        )

        logger.debug(f"Linked evidence {snippet.id} to claim {claim_id}")
        return link

    def get_evidence_for_claim(self, claim_id: str) -> list[EvidenceLink]:
        """
        Get all evidence links for a claim.

        Args:
            claim_id: ID of the claim

        Returns:
            List of evidence links supporting/contradicting the claim
        """
        return self._claim_links.get(claim_id, [])

    def get_claims_for_evidence(self, evidence_id: str) -> list[EvidenceLink]:
        """
        Get all claims linked to an evidence snippet.

        Args:
            evidence_id: ID of the evidence

        Returns:
            List of evidence links from this evidence to claims
        """
        return self._evidence_links.get(evidence_id, [])

    def update_belief_from_evidence(
        self,
        belief: BeliefDistribution,
        evidence: list[EvidenceSnippet],
        base_update_strength: float = 0.1,
    ) -> EvidenceImpact:
        """
        Update a belief distribution based on evidence.

        Uses evidence reliability, freshness, and relevance to adjust
        the belief distribution. Supporting evidence increases p_true,
        contradicting evidence increases p_false.

        Args:
            belief: Current belief distribution
            evidence: List of evidence snippets
            base_update_strength: Base strength of each piece of evidence (0-1)

        Returns:
            EvidenceImpact with original and updated beliefs
        """
        if not evidence:
            return EvidenceImpact(
                original_belief=belief,
                updated_belief=belief,
                evidence_count=0,
                total_weight=0.0,
                avg_relevance=0.0,
                direction="neutral",
            )

        # Calculate weighted evidence impact
        total_positive = 0.0
        total_negative = 0.0
        total_weight = 0.0
        total_relevance = 0.0

        for snippet in evidence:
            # Weight based on reliability and freshness
            weight = snippet.reliability_score * 0.7 + snippet.freshness_score * 0.3
            total_weight += weight
            total_relevance += 1.0  # Default relevance when not linked

            # Determine direction from metadata if available
            direction = snippet.metadata.get("support_direction", 1.0)

            if direction > 0:
                total_positive += weight * direction
            else:
                total_negative += weight * abs(direction)

        # Calculate belief updates
        if total_weight > 0:
            avg_relevance = total_relevance / len(evidence)

            # Normalize and apply update
            net_direction = (total_positive - total_negative) / total_weight
            update_magnitude = base_update_strength * min(total_weight, 1.0)

            # Update probabilities
            if net_direction > 0:
                # Evidence supports claim
                new_p_true = min(0.99, belief.p_true + update_magnitude * net_direction)
                new_p_false = max(0.01, belief.p_false - update_magnitude * net_direction * 0.5)
            else:
                # Evidence contradicts claim
                new_p_true = max(0.01, belief.p_true + update_magnitude * net_direction * 0.5)
                new_p_false = min(0.99, belief.p_false - update_magnitude * net_direction)

            updated_belief = BeliefDistribution(
                p_true=new_p_true,
                p_false=new_p_false,
                p_unknown=belief.p_unknown,
            )

            # Determine direction label
            if abs(net_direction) < 0.1:
                direction = "neutral"
            elif net_direction > 0:
                direction = "supporting"
            else:
                direction = "contradicting"

            if total_positive > 0 and total_negative > 0:
                direction = "mixed"

        else:
            updated_belief = belief
            avg_relevance = 0.0
            direction = "neutral"

        return EvidenceImpact(
            original_belief=belief,
            updated_belief=updated_belief,
            evidence_count=len(evidence),
            total_weight=total_weight,
            avg_relevance=avg_relevance,
            direction=direction,
        )

    def create_evidence_chain(
        self,
        snippets: list[EvidenceSnippet],
        claim_id: Optional[str] = None,
    ) -> str:
        """
        Create a provenance chain from multiple evidence snippets.

        Links evidence together to show chain of reasoning/discovery.

        Args:
            snippets: List of evidence snippets in chronological order
            claim_id: Optional claim this chain supports

        Returns:
            Chain ID for the created provenance chain
        """
        if not snippets:
            raise ValueError("Cannot create chain from empty evidence list")

        chain_id = str(uuid.uuid4())[:12]
        self._chains[chain_id] = []
        previous_hash = None

        for i, snippet in enumerate(snippets):
            if i == 0:
                # First record starts the chain
                record = self.register_evidence(snippet, None)
                self._chains[chain_id] = [record]
            else:
                # Subsequent records link to previous - register_evidence will append
                record = self.register_evidence(snippet, chain_id)
                record.previous_hash = previous_hash

            previous_hash = record.chain_hash()

            # Link to claim if provided
            if claim_id:
                self.link_evidence_to_claim(snippet, claim_id)

        logger.info(f"Created evidence chain {chain_id} with {len(snippets)} snippets")
        return chain_id

    def get_chain_summary(self, chain_id: str) -> dict:
        """
        Get summary of an evidence chain.

        Args:
            chain_id: ID of the chain

        Returns:
            Summary dict with chain statistics
        """
        records = self._chains.get(chain_id, [])
        if not records:
            return {"error": "Chain not found", "chain_id": chain_id}

        return {
            "chain_id": chain_id,
            "length": len(records),
            "first_record": records[0].id if records else None,
            "last_record": records[-1].id if records else None,
            "sources": list(set(r.source_type.value for r in records)),
            "avg_confidence": sum(r.confidence for r in records) / len(records) if records else 0,
            "verified_count": sum(1 for r in records if r.verified),
        }

    def _map_source_type(self, source: str) -> SourceType:
        """Map evidence source string to SourceType enum."""
        source_lower = source.lower()

        for key, source_type in self.SOURCE_TYPE_MAP.items():
            if key in source_lower:
                return source_type

        return SourceType.UNKNOWN


# Global bridge instance
_global_bridge: Optional[EvidenceProvenanceBridge] = None


def get_evidence_bridge() -> EvidenceProvenanceBridge:
    """Get or create the global evidence-provenance bridge."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = EvidenceProvenanceBridge()
    return _global_bridge


def reset_evidence_bridge() -> None:
    """Reset the global bridge (for testing)."""
    global _global_bridge
    _global_bridge = None


__all__ = [
    "EvidenceProvenanceBridge",
    "EvidenceLink",
    "EvidenceImpact",
    "get_evidence_bridge",
    "reset_evidence_bridge",
]
