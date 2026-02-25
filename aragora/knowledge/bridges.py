"""
Knowledge Mound Integration Bridges.

Connects stranded features to the unified KnowledgeMound:
- MetaLearner -> pattern nodes (hyperparameter adjustments, efficiency metrics)
- Evidence Collector -> evidence nodes (external data, citations)
- Pattern Extractor -> pattern nodes (detected patterns from debates)

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
import inspect
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from aragora.learning.meta import HyperparameterState, LearningMetrics

    # Evidence type may not exist in collector
    CollectorEvidence = Any

logger = logging.getLogger(__name__)


class KnowledgeMoundProtocol(Protocol):
    """Protocol defining the interface bridges need from KnowledgeMound.

    This protocol enables proper type checking without circular imports
    and avoids the CRUDProtocol self-type mismatch issue.
    """

    @property
    def workspace_id(self) -> str:
        """Get the default workspace ID."""
        ...

    async def add_node(self, node: Any) -> str:
        """Add a KnowledgeNode to the mound and return its ID."""
        ...


__all__ = [
    "MetaLearnerBridge",
    "EvidenceBridge",
    "PatternBridge",
    "ToolAuditBridge",
    "KnowledgeBridgeHub",
    "PipelineBridge",
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

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        """
        Initialize MetaLearner bridge.

        Args:
            mound: KnowledgeMound instance to store patterns in
        """
        self.mound = mound

    async def capture_adjustment(
        self,
        metrics: LearningMetrics,
        adjustments: dict[str, Any],
        hyperparams: HyperparameterState,
        cycle_number: int = 0,
    ) -> str | None:
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
            node_type="fact",  # Meta-learning patterns stored as facts
            content=content,
            confidence=confidence,
            provenance=provenance,
            tier=MemoryTier.MEDIUM,  # Meta-learning patterns are medium-term
            workspace_id=self.mound.workspace_id,
            surprise_score=0.5,  # Neutral surprise for meta-learning
        )

        # add_node returns node_id string
        node_id = await self.mound.add_node(node)
        logger.info("Captured meta-learning adjustment as pattern node: %s", node_id)
        return node_id

    async def capture_learning_summary(
        self,
        summary: dict[str, Any],
    ) -> str | None:
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
            node_type="fact",  # Pattern summaries stored as facts
            content=content,
            confidence=base_confidence,
            provenance=provenance,
            tier=MemoryTier.SLOW,  # Summaries are long-term
            workspace_id=self.mound.workspace_id,
        )

        node_id = await self.mound.add_node(node)
        logger.info("Captured meta-learning summary as pattern node: %s", node_id)
        return str(node_id)


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

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        """
        Initialize Evidence bridge.

        Args:
            mound: KnowledgeMound instance to store evidence in
        """
        self.mound = mound

    async def store_evidence(
        self,
        content: str,
        source: str,
        evidence_type: str = "citation",
        supports_claim: bool = True,
        strength: float = 0.5,
        metadata: dict[str, Any] | None = None,
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
            workspace_id=self.mound.workspace_id,
        )

        node_id: str = await self.mound.add_node(node)
        logger.info("Stored evidence as node: %s (source: %s)", node_id, source)
        return node_id

    async def store_from_collector_evidence(
        self,
        evidence: CollectorEvidence,
        claim_node_id: str | None = None,
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
                        "metadata": evidence.metadata if hasattr(evidence, "metadata") else {},
                    },
                    "timestamp": (
                        evidence.timestamp
                        if hasattr(evidence, "timestamp")
                        else datetime.now().isoformat()
                    ),
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
            workspace_id=self.mound.workspace_id,
            supports=supports,
            contradicts=contradicts,
        )

        node_id: str = await self.mound.add_node(node)
        logger.info("Stored collector evidence as node: %s", node_id)
        return node_id


class ToolAuditBridge:
    """
    Bridge between vertical tool calls and KnowledgeMound.

    Captures vertical tool invocation metadata as evidence nodes to
    enable long-term auditability and cross-debate reuse.
    """

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        self.mound = mound

    async def record_tool_call(
        self,
        *,
        vertical_id: str,
        tool_name: str,
        agent_name: str,
        connector_type: str | None,
        parameters: dict[str, Any],
        status: str,
        policy: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> str | None:
        """Record a vertical tool invocation as a KnowledgeNode."""
        if not tool_name:
            return None

        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        parameter_keys = list(parameters.keys())
        parameter_types = {key: type(value).__name__ for key, value in parameters.items()}
        result_summary = _summarize_tool_result(result)

        content_lines = [
            f"Vertical tool invocation: {vertical_id}.{tool_name}",
            f"Status: {status}",
        ]
        if connector_type:
            content_lines.append(f"Connector: {connector_type}")
        if result_summary:
            content_lines.append(f"Summary: {result_summary}")
        content = "\n".join(content_lines)

        provenance = ProvenanceChain(
            source_type=ProvenanceType.AGENT,
            source_id=f"vertical_tool:{vertical_id}.{tool_name}",
            agent_id=agent_name,
        )
        provenance.add_transformation(
            "tool_invocation",
            agent_id=agent_name,
            details={
                "vertical_id": vertical_id,
                "tool": tool_name,
                "status": status,
                "connector_type": connector_type,
                "parameter_keys": parameter_keys,
                "parameter_types": parameter_types,
                "policy": policy or {},
                "result_summary": result_summary,
            },
        )

        confidence = 0.75 if status == "allowed" else 0.4

        node = KnowledgeNode(
            node_type="evidence",
            content=content,
            confidence=confidence,
            provenance=provenance,
            tier=MemoryTier.MEDIUM,
            workspace_id=self.mound.workspace_id,
        )

        node_id = await self.mound.add_node(node)
        logger.info("Recorded vertical tool invocation to KM: %s", node_id)
        return node_id


def _summarize_tool_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize a tool result payload without storing full content."""
    if not result:
        return {}

    summary: dict[str, Any] = {}
    for key in ("count", "mode", "status", "decision", "query", "error"):
        if key in result and result[key] is not None:
            summary[key] = result[key]

    for key in (
        "results",
        "articles",
        "papers",
        "cases",
        "statutes",
        "guidelines",
        "filings",
        "matches",
        "codes",
        "interactions",
    ):
        value = result.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)

    if "drug_info" in result and result.get("drug_info"):
        summary["drug_info"] = "present"

    return summary


class PatternBridge:
    """
    Bridge between Pattern Extractor and KnowledgeMound.

    Converts extracted patterns into KnowledgeNodes:
    - node_type: "fact" (with pattern info in content/provenance)
    - content: Pattern description
    - confidence: Pattern strength/frequency
    - provenance: Source debates/interactions

    This enables patterns to be queried and used across the system.
    """

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        """
        Initialize Pattern bridge.

        Args:
            mound: KnowledgeMound instance to store patterns in
        """
        self.mound = mound

    async def store_pattern(
        self,
        pattern_type: str,
        description: str,
        occurrences: int = 1,
        confidence: float = 0.5,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
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

        # Note: "pattern" is not a valid NodeType, use "fact" for detected patterns
        # The pattern_type is stored in the content and provenance for categorization
        node = KnowledgeNode(
            node_type="fact",
            content=content,
            confidence=confidence,
            provenance=provenance,
            tier=tier,
            workspace_id=self.mound.workspace_id,
            surprise_score=surprise_score,
            derived_from=source_ids or [],
        )

        node_id: str = await self.mound.add_node(node)
        logger.info("Stored pattern as node: %s (type: %s)", node_id, pattern_type)
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

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        """
        Initialize the bridge hub.

        Args:
            mound: KnowledgeMound instance to connect bridges to
        """
        self.mound = mound
        self._meta_learner: MetaLearnerBridge | None = None
        self._evidence: EvidenceBridge | None = None
        self._patterns: PatternBridge | None = None
        self._tool_audit: ToolAuditBridge | None = None

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

    @property
    def tool_audit(self) -> ToolAuditBridge:
        """Get ToolAudit bridge (lazy initialization)."""
        if self._tool_audit is None:
            self._tool_audit = ToolAuditBridge(self.mound)
        return self._tool_audit

    async def synthesize_for_debate(
        self,
        topic: str,
        domain: str = "general",
        max_items: int = 10,
    ) -> str:
        """Synthesize cross-adapter knowledge relevant to a debate topic.

        Queries multiple KM adapters (Consensus, Evidence, Performance, Pulse,
        Belief, Compliance) and returns a unified context block for injection
        into debate prompts. This is the missing cross-adapter synthesis layer
        that turns 33 write-heavy adapters into a coherent read-back system.

        Args:
            topic: The debate topic to search for
            domain: Domain context for filtering
            max_items: Maximum items to return across all adapters

        Returns:
            Formatted markdown context string for prompt injection
        """
        sections: list[str] = []
        items_remaining = max_items

        # Query adapters in priority order (most decision-relevant first)
        adapter_queries = [
            ("consensus", "search_by_topic", "Past Consensus Decisions"),
            ("debate", "search_by_topic", "Related Debate Outcomes"),
            ("evidence", "search_by_topic", "Evidence & Citations"),
            ("insight", "search_by_topic", "Organizational Insights"),
            ("compliance", "search_by_topic", "Compliance Considerations"),
        ]

        for adapter_name, method_name, section_title in adapter_queries:
            if items_remaining <= 0:
                break

            try:
                adapter = self._get_adapter(adapter_name)
                if adapter is None:
                    continue

                search_fn = getattr(adapter, method_name, None)
                if search_fn is None:
                    continue

                # Most adapters accept (topic, limit) or (query, limit).
                # Some adapters expose async search methods; await when needed.
                results = search_fn(topic, limit=min(3, items_remaining))
                if inspect.isawaitable(results):
                    results = await results
                if not results:
                    continue

                items = []
                for r in results[: min(3, items_remaining)]:
                    # Handle various result formats
                    if isinstance(r, dict):
                        content = r.get("content", r.get("summary", str(r)))
                        confidence = r.get("confidence", "")
                        conf_str = (
                            f" ({confidence:.0%})" if isinstance(confidence, (int, float)) else ""
                        )
                        items.append(f"- {content[:200]}{conf_str}")
                    elif hasattr(r, "content"):
                        conf = getattr(r, "confidence", "")
                        conf_str = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
                        items.append(f"- {str(r.content)[:200]}{conf_str}")
                    else:
                        items.append(f"- {str(r)[:200]}")

                if items:
                    sections.append(f"### {section_title}\n" + "\n".join(items))
                    items_remaining -= len(items)

            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                logger.debug("KM synthesis: %s adapter query failed: %s", adapter_name, e)
                continue

        if not sections:
            return ""

        return "## ORGANIZATIONAL KNOWLEDGE (Cross-Adapter Synthesis)\n\n" + "\n\n".join(sections)

    def _get_adapter(self, adapter_name: str) -> Any:
        """Get a KM adapter by name, lazily importing from the factory."""
        try:
            from aragora.knowledge.mound.adapters.factory import get_adapter

            return get_adapter(adapter_name, self.mound)
        except (ImportError, KeyError, AttributeError, TypeError) as e:
            logger.debug("Could not get adapter %s: %s", adapter_name, e)
            return None


class PipelineBridge:
    """Bridge between IdeaToExecutionPipeline and KnowledgeMound.

    Stores completed pipeline runs as KM nodes and queries for
    precedent runs on similar topics. This enables cross-run learning
    where future pipelines benefit from past outcomes.
    """

    def __init__(self, mound: KnowledgeMoundProtocol) -> None:
        self.mound = mound

    async def store_pipeline_run(
        self,
        pipeline_id: str,
        topic: str,
        stages_completed: int,
        duration: float,
        goal_count: int = 0,
        task_count: int = 0,
        receipt: dict[str, Any] | None = None,
    ) -> str | None:
        """Persist a completed pipeline run as a KM node.

        Returns:
            Node ID if stored, None on failure.
        """
        content_parts = [
            f"Pipeline run: {pipeline_id}",
            f"Topic: {topic}",
            f"Stages completed: {stages_completed}/4",
            f"Duration: {duration:.1f}s",
            f"Goals: {goal_count}, Tasks: {task_count}",
        ]
        content = "\n".join(content_parts)

        try:
            from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
            from aragora.memory.tier_manager import MemoryTier

            provenance = ProvenanceChain(
                source_type=ProvenanceType.AGENT,
                source_id=f"pipeline_{pipeline_id}",
            )

            node = KnowledgeNode(
                node_type="fact",
                content=content,
                confidence=min(1.0, stages_completed / 4.0),
                provenance=provenance,
                tier=MemoryTier.SLOW,
                workspace_id=self.mound.workspace_id,
                metadata={
                    "pipeline_id": pipeline_id,
                    "topic": topic,
                    "stages_completed": stages_completed,
                    "duration": duration,
                    "goal_count": goal_count,
                    "task_count": task_count,
                    **({"receipt_hash": receipt.get("hash", "")} if receipt else {}),
                },
            )

            node_id = await self.mound.add_node(node)
            logger.info("Stored pipeline run %s as KM node: %s", pipeline_id, node_id)
            return str(node_id)
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Failed to store pipeline run: %s", exc)
            return None

    async def query_precedents(
        self,
        topic: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search KM for past pipeline runs on similar topics.

        Returns:
            List of precedent dicts with pipeline_id, topic, stages_completed, etc.
        """
        try:
            results = await self.mound.search(  # type: ignore[attr-defined]
                query=topic,
                node_type="fact",
                limit=limit,
            )
            precedents = []
            for node in results:
                meta = getattr(node, "metadata", {}) or {}
                if meta.get("pipeline_id"):
                    precedents.append(meta)
            return precedents
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Precedent query failed: %s", exc)
            return []
