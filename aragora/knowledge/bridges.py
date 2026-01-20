"""
Knowledge Mound Integration Bridges.

Connects stranded features to the unified KnowledgeMound:
- MetaLearner → pattern nodes (hyperparameter adjustments, efficiency metrics)
- Evidence Collector → evidence nodes (external data, citations)
- Pattern Extractor → pattern nodes (detected patterns from debates)

These bridges enable the KnowledgeMound to serve as the central
knowledge repository that captures insights from all Aragora systems.

Usage:
    from aragora.knowledge.bridges import (
        MetaLearnerBridge,
        EvidenceBridge,
        PatternBridge,
    )

    # Connect MetaLearner to KnowledgeMound
    meta_bridge = MetaLearnerBridge(mound)
    meta_bridge.capture_adjustment(metrics, adjustments)

    # Connect Evidence Collector
    evidence_bridge = EvidenceBridge(mound)
    evidence_bridge.store_evidence(evidence)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.evidence.collector import Evidence as CollectorEvidence
    from aragora.knowledge.mound import KnowledgeMound
    from aragora.learning.meta import HyperparameterState, LearningMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "MetaLearnerBridge",
    "EvidenceBridge",
    "PatternBridge",
    "KnowledgeBridgeHub",
]


class MetaLearnerBridge:
    """
    Bridge between MetaLearner and KnowledgeMound.

    Captures meta-learning patterns as KnowledgeNodes, enabling:
    - Historical tracking of hyperparameter adjustments
    - Pattern detection in learning efficiency
    - Cross-workspace learning insights

    Each adjustment becomes a pattern node with:
    - node_type: "pattern"
    - content: Description of the adjustment and its rationale
    - confidence: Based on the prediction accuracy
    - provenance: Links to the MetaLearner and cycle
    """

    def __init__(self, mound: "KnowledgeMound"):
        """
        Initialize MetaLearner bridge.

        Args:
            mound: KnowledgeMound to store patterns in
        """
        self.mound = mound

    async def capture_adjustment(
        self,
        metrics: "LearningMetrics",
        adjustments: dict[str, Any],
        hyperparams: "HyperparameterState",
        cycle_number: int = 0,
    ) -> Optional[str]:
        """
        Capture a hyperparameter adjustment as a pattern node.

        Args:
            metrics: The metrics that triggered the adjustment
            adjustments: Dict of adjustments made
            hyperparams: Current hyperparameter state
            cycle_number: The nomic cycle number

        Returns:
            Node ID if created, None if no adjustments
        """
        if not adjustments:
            return None

        # Build content description
        content_parts = [
            f"Meta-learning adjustment at cycle {cycle_number}:",
            f"- Pattern retention: {metrics.pattern_retention_rate:.1%}",
            f"- Forgetting rate: {metrics.forgetting_rate:.1%}",
            f"- Prediction accuracy: {metrics.prediction_accuracy:.1%}",
            "",
            "Adjustments made:",
        ]
        for key, value in adjustments.items():
            content_parts.append(f"- {key}: {value}")

        content = "\n".join(content_parts)

        # Create provenance chain
        from aragora.knowledge.mound import ProvenanceChain, ProvenanceType

        provenance = ProvenanceChain(
            source_type=ProvenanceType.AGENT,  # Meta-learner is an agent-type source
            source_id=f"meta_learner_cycle_{cycle_number}",
            transformations=[
                {
                    "type": "evaluation",
                    "details": {
                        "retention": metrics.pattern_retention_rate,
                        "forgetting": metrics.forgetting_rate,
                        "accuracy": metrics.prediction_accuracy,
                    },
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "type": "adjustment",
                    "details": {"changes": adjustments},
                    "timestamp": datetime.now().isoformat(),
                },
            ],
        )

        # Confidence based on prediction accuracy and retention
        confidence = (metrics.prediction_accuracy + metrics.pattern_retention_rate) / 2

        # Create pattern node
        from aragora.knowledge.mound import KnowledgeNode
        from aragora.memory.tier_manager import MemoryTier

        node = KnowledgeNode(
            node_type="pattern",
            content=content,
            confidence=confidence,
            provenance=provenance,
            tier=MemoryTier.MEDIUM,  # Meta-learning patterns are medium-term
            workspace_id=self.mound._workspace_id,
            surprise_score=0.5,  # Neutral surprise for meta-learning
        )

        node_id = await self.mound.add_node(node)
        logger.info(f"Captured meta-learning adjustment as pattern node: {node_id}")
        return node_id

    async def capture_learning_summary(
        self,
        summary: dict[str, Any],
    ) -> Optional[str]:
        """
        Capture a meta-learning summary as a pattern node.

        Args:
            summary: Learning summary from MetaLearner.get_learning_summary()

        Returns:
            Node ID if created
        """
        if summary.get("status") == "no data":
            return None

        content_parts = [
            f"Meta-learning summary ({summary.get('evaluations', 0)} evaluations):",
            f"- Average retention: {summary.get('avg_retention', 0):.1%}",
            f"- Average forgetting: {summary.get('avg_forgetting', 0):.1%}",
            f"- Learning velocity: {summary.get('avg_learning_velocity', 0):.1f}/cycle",
            f"- Trend: {summary.get('trend', 'unknown')}",
        ]
        content = "\n".join(content_parts)

        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        provenance = ProvenanceChain(
            source_type=ProvenanceType.AGENT,
            source_id="meta_learner_summary",
        )

        # Confidence based on retention and trend
        base_confidence = summary.get("avg_retention", 0.5)
        if summary.get("trend") == "improving":
            base_confidence = min(1.0, base_confidence + 0.1)
        elif summary.get("trend") == "declining":
            base_confidence = max(0.0, base_confidence - 0.1)

        node = KnowledgeNode(
            node_type="pattern",
            content=content,
            confidence=base_confidence,
            provenance=provenance,
            tier=MemoryTier.SLOW,  # Summaries are long-term
            workspace_id=self.mound._workspace_id,
        )

        node_id = await self.mound.add_node(node)
        logger.info(f"Captured meta-learning summary as pattern node: {node_id}")
        return node_id


class EvidenceBridge:
    """
    Bridge between Evidence Collector and KnowledgeMound.

    Converts evidence from external sources into KnowledgeNodes:
    - node_type: "evidence"
    - content: The evidence content
    - confidence: Based on source reliability
    - provenance: Full source chain

    This connects the evidence system to the unified knowledge store.
    """

    def __init__(self, mound: "KnowledgeMound"):
        """
        Initialize Evidence bridge.

        Args:
            mound: KnowledgeMound to store evidence in
        """
        self.mound = mound

    async def store_evidence(
        self,
        content: str,
        source: str,
        evidence_type: str = "citation",
        supports_claim: bool = True,
        strength: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Store evidence as a KnowledgeNode.

        Args:
            content: The evidence content
            source: Source identifier (URL, document ID, etc.)
            evidence_type: Type of evidence (citation, data, tool_output, etc.)
            supports_claim: Whether this supports or refutes a claim
            strength: Evidence strength (0-1)
            metadata: Additional metadata

        Returns:
            Node ID
        """
        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        provenance = ProvenanceChain(
            source_type=ProvenanceType.DOCUMENT,  # Evidence comes from documents/external sources
            source_id=source,
            transformations=[
                {
                    "type": "collection",
                    "details": {
                        "source": source,
                        "evidence_type": evidence_type,
                        "supports": supports_claim,
                        "strength": strength,
                        "metadata": metadata or {},
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        )

        node = KnowledgeNode(
            node_type="evidence",
            content=content,
            confidence=strength,
            provenance=provenance,
            tier=MemoryTier.SLOW,  # Evidence is long-term
            workspace_id=self.mound._workspace_id,
        )

        node_id = await self.mound.add_node(node)
        logger.info(f"Stored evidence as node: {node_id} (source: {source})")
        return node_id

    async def store_from_collector_evidence(
        self,
        evidence: "CollectorEvidence",
        claim_node_id: Optional[str] = None,
    ) -> str:
        """
        Store evidence from the Evidence Collector module.

        Args:
            evidence: Evidence object from aragora.evidence.collector
            claim_node_id: Optional ID of the claim this evidence relates to

        Returns:
            Node ID
        """
        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        provenance = ProvenanceChain(
            source_type=ProvenanceType.DOCUMENT,
            source_id=evidence.evidence_id,
            transformations=[
                {
                    "type": "collection",
                    "details": {
                        "source": evidence.source,
                        "evidence_type": evidence.evidence_type,
                        "supports": evidence.supports_claim,
                        "metadata": evidence.metadata if hasattr(evidence, 'metadata') else {},
                    },
                    "timestamp": evidence.timestamp if hasattr(evidence, 'timestamp') else datetime.now().isoformat(),
                }
            ],
        )

        # Build relationship based on support/refute
        supports = []
        contradicts = []
        if claim_node_id:
            if evidence.supports_claim:
                supports.append(claim_node_id)
            else:
                contradicts.append(claim_node_id)

        node = KnowledgeNode(
            node_type="evidence",
            content=evidence.content,
            confidence=evidence.strength,
            provenance=provenance,
            tier=MemoryTier.SLOW,
            workspace_id=self.mound._workspace_id,
            supports=supports,
            contradicts=contradicts,
        )

        node_id = await self.mound.add_node(node)
        logger.info(f"Stored collector evidence as node: {node_id}")
        return node_id


class PatternBridge:
    """
    Bridge between Pattern Extractor and KnowledgeMound.

    Converts extracted patterns into KnowledgeNodes:
    - node_type: "pattern"
    - content: Pattern description
    - confidence: Pattern strength/frequency
    - provenance: Source debates/interactions

    This enables patterns to be queried and used across the system.
    """

    def __init__(self, mound: "KnowledgeMound"):
        """
        Initialize Pattern bridge.

        Args:
            mound: KnowledgeMound to store patterns in
        """
        self.mound = mound

    async def store_pattern(
        self,
        pattern_type: str,
        description: str,
        occurrences: int = 1,
        confidence: float = 0.5,
        source_ids: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Store a detected pattern as a KnowledgeNode.

        Args:
            pattern_type: Type of pattern (critique, debate, consensus, etc.)
            description: Human-readable pattern description
            occurrences: Number of times pattern was observed
            confidence: Pattern confidence/strength
            source_ids: IDs of source debates/interactions
            metadata: Additional pattern metadata

        Returns:
            Node ID
        """
        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        content = f"[{pattern_type}] {description}\nObserved {occurrences} time(s)"

        provenance = ProvenanceChain(
            source_type=ProvenanceType.INFERENCE,  # Patterns are inferred from data
            source_id=f"pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            transformations=[
                {
                    "type": "extraction",
                    "details": {
                        "pattern_type": pattern_type,
                        "occurrences": occurrences,
                        "source_ids": source_ids or [],
                        "metadata": metadata or {},
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        )

        # Adjust tier based on occurrences
        if occurrences >= 10:
            tier = MemoryTier.SLOW  # Well-established pattern
        elif occurrences >= 3:
            tier = MemoryTier.MEDIUM  # Emerging pattern
        else:
            tier = MemoryTier.FAST  # New pattern

        # Surprise based on how novel this pattern is
        surprise_score = max(0.1, 1.0 - (occurrences / 20))  # Less surprise for common patterns

        node = KnowledgeNode(
            node_type="pattern",
            content=content,
            confidence=confidence,
            provenance=provenance,
            tier=tier,
            workspace_id=self.mound._workspace_id,
            surprise_score=surprise_score,
            derived_from=source_ids or [],
        )

        node_id = await self.mound.add_node(node)
        logger.info(f"Stored pattern as node: {node_id} (type: {pattern_type})")
        return node_id

    async def store_critique_pattern(
        self,
        pattern_description: str,
        agent_name: str,
        frequency: int = 1,
        effectiveness: float = 0.5,
    ) -> str:
        """
        Store a critique pattern from an agent.

        Args:
            pattern_description: Description of the critique pattern
            agent_name: Name of the agent exhibiting the pattern
            frequency: How often the pattern appears
            effectiveness: How effective the critique is

        Returns:
            Node ID
        """
        return await self.store_pattern(
            pattern_type="critique",
            description=f"Agent '{agent_name}': {pattern_description}",
            occurrences=frequency,
            confidence=effectiveness,
            metadata={
                "agent": agent_name,
                "effectiveness": effectiveness,
            },
        )

    async def store_debate_pattern(
        self,
        pattern_description: str,
        debate_ids: list[str],
        consensus_rate: float = 0.5,
    ) -> str:
        """
        Store a debate pattern.

        Args:
            pattern_description: Description of the debate pattern
            debate_ids: IDs of debates where pattern was observed
            consensus_rate: Rate of consensus in debates with this pattern

        Returns:
            Node ID
        """
        return await self.store_pattern(
            pattern_type="debate",
            description=pattern_description,
            occurrences=len(debate_ids),
            confidence=consensus_rate,
            source_ids=debate_ids,
            metadata={
                "consensus_rate": consensus_rate,
            },
        )


class KnowledgeBridgeHub:
    """
    Central hub for all knowledge bridges.

    Provides a single point of access to connect various Aragora
    systems to the KnowledgeMound.

    Usage:
        hub = KnowledgeBridgeHub(mound)

        # Access specific bridges
        hub.meta_learner.capture_adjustment(...)
        hub.evidence.store_evidence(...)
        hub.patterns.store_pattern(...)
    """

    def __init__(self, mound: "KnowledgeMound"):
        """
        Initialize the bridge hub.

        Args:
            mound: KnowledgeMound to connect bridges to
        """
        self.mound = mound
        self._meta_learner: Optional[MetaLearnerBridge] = None
        self._evidence: Optional[EvidenceBridge] = None
        self._patterns: Optional[PatternBridge] = None

    @property
    def meta_learner(self) -> MetaLearnerBridge:
        """Get MetaLearner bridge (lazy initialization)."""
        if self._meta_learner is None:
            self._meta_learner = MetaLearnerBridge(self.mound)
        return self._meta_learner

    @property
    def evidence(self) -> EvidenceBridge:
        """Get Evidence bridge (lazy initialization)."""
        if self._evidence is None:
            self._evidence = EvidenceBridge(self.mound)
        return self._evidence

    @property
    def patterns(self) -> PatternBridge:
        """Get Pattern bridge (lazy initialization)."""
        if self._patterns is None:
            self._patterns = PatternBridge(self.mound)
        return self._patterns
