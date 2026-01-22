"""
Staleness detection for Knowledge Mound.

Identifies knowledge that may be outdated and needs revalidation,
based on age, contradictions, new evidence, and consensus changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aragora.knowledge.mound.types import (
    StalenessCheck,
    StalenessReason,
)
from aragora.knowledge.unified.types import RelationshipType

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


# Default staleness thresholds by tier
TIER_AGE_THRESHOLDS = {
    "fast": timedelta(hours=1),
    "medium": timedelta(days=1),
    "slow": timedelta(days=7),
    "glacial": timedelta(days=30),
}


@dataclass
class StalenessConfig:
    """Configuration for staleness detection."""

    # Age-based staleness
    age_thresholds: Dict[str, timedelta] = None
    age_weight: float = 0.4

    # Contradiction-based staleness
    contradiction_weight: float = 0.3
    min_contradictions_for_stale: int = 1

    # New evidence staleness
    new_evidence_weight: float = 0.2
    evidence_recency_window: timedelta = timedelta(days=7)

    # Consensus change staleness
    consensus_change_weight: float = 0.1

    # Auto-revalidation
    auto_revalidation_threshold: float = 0.8
    revalidation_batch_size: int = 10

    def __post_init__(self):
        if self.age_thresholds is None:
            self.age_thresholds = TIER_AGE_THRESHOLDS.copy()


class StalenessDetector:
    """
    Detects and manages stale knowledge in the Knowledge Mound.

    Staleness is computed based on multiple factors:
    1. Age relative to tier half-life
    2. Number of contradicting items added since last validation
    3. New evidence that may affect validity
    4. Consensus changes in related debates
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        config: Optional[StalenessConfig] = None,
        age_threshold: Optional[timedelta] = None,
        event_emitter: Optional[Any] = None,
    ):
        """
        Initialize staleness detector.

        Args:
            mound: Reference to the Knowledge Mound
            config: Staleness configuration
            age_threshold: Override default age threshold
            event_emitter: Optional event emitter for cross-subsystem integration
        """
        self._mound = mound
        self._config = config or StalenessConfig()
        self._event_emitter = event_emitter

        if age_threshold:
            # Apply override to all tiers
            for tier in self._config.age_thresholds:
                self._config.age_thresholds[tier] = age_threshold

    async def compute_staleness(self, node_id: str) -> StalenessCheck:
        """
        Compute staleness score for a knowledge node.

        Args:
            node_id: ID of the node to check

        Returns:
            StalenessCheck with score and reasons
        """
        node = await self._mound.get(node_id)  # type: ignore[misc]
        if not node:
            return StalenessCheck(
                node_id=node_id,
                staleness_score=0.0,
                reasons=[],
                revalidation_recommended=False,
            )

        reasons: List[StalenessReason] = []
        scores: Dict[str, float] = {}

        # 1. Age-based staleness
        age_score = self._compute_age_score(node)
        if age_score > 0.3:
            reasons.append(StalenessReason.AGE)
        scores["age"] = age_score

        # 2. Contradiction-based staleness
        contradiction_score = await self._compute_contradiction_score(node_id)
        if contradiction_score > 0.3:
            reasons.append(StalenessReason.CONTRADICTION)
        scores["contradiction"] = contradiction_score

        # 3. New evidence staleness
        evidence_score = await self._compute_evidence_score(node_id)
        if evidence_score > 0.3:
            reasons.append(StalenessReason.NEW_EVIDENCE)
        scores["evidence"] = evidence_score

        # 4. Consensus change staleness
        consensus_score = await self._compute_consensus_score(node_id)
        if consensus_score > 0.3:
            reasons.append(StalenessReason.CONSENSUS_CHANGE)
        scores["consensus"] = consensus_score

        # Compute weighted total
        total_score = (
            scores["age"] * self._config.age_weight
            + scores["contradiction"] * self._config.contradiction_weight
            + scores["evidence"] * self._config.new_evidence_weight
            + scores["consensus"] * self._config.consensus_change_weight
        )

        # Clamp to [0, 1]
        total_score = max(0.0, min(1.0, total_score))

        result = StalenessCheck(
            node_id=node_id,
            staleness_score=total_score,
            reasons=reasons,
            last_checked_at=datetime.now(),
            revalidation_recommended=total_score >= self._config.auto_revalidation_threshold,
            evidence={
                "scores": scores,
                "node_updated_at": node.updated_at.isoformat() if node.updated_at else None,
                "tier": node.metadata.get("tier", "slow"),
            },
        )

        # Emit event if staleness exceeds threshold
        if total_score >= self._config.auto_revalidation_threshold:
            self._emit_staleness_event(result, node)

        return result

    def _compute_age_score(self, node: Any) -> float:
        """Compute age-based staleness score."""
        if not node.updated_at:
            return 0.5  # Unknown age, moderate staleness

        tier = node.metadata.get("tier", "slow")
        threshold = self._config.age_thresholds.get(tier, timedelta(days=7))

        age = datetime.now() - node.updated_at
        if age <= timedelta(0):
            return 0.0

        # Score increases linearly up to 2x threshold
        max_age = threshold * 2
        score = min(1.0, age.total_seconds() / max_age.total_seconds())

        return score

    async def _compute_contradiction_score(self, node_id: str) -> float:
        """Compute contradiction-based staleness score."""
        try:
            # Find contradicting nodes added after this node
            node = await self._mound.get(node_id)  # type: ignore[misc]
            if not node:
                return 0.0

            # Query for contradicting relationships
            result = await self._mound.query_graph(
                start_id=node_id,
                relationship_types=[RelationshipType.CONTRADICTS],
                depth=1,
            )

            # Count contradictions added after this node was last updated
            recent_contradictions = 0
            for edge in result.edges:
                # Check if contradiction is newer
                target_node = await self._mound.get(edge.target_id)  # type: ignore[misc]
                if target_node and target_node.created_at > node.updated_at:
                    recent_contradictions += 1

            if recent_contradictions >= self._config.min_contradictions_for_stale:
                return min(1.0, recent_contradictions / 3.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Failed to compute contradiction score: {e}")
            return 0.0

    async def _compute_evidence_score(self, node_id: str) -> float:
        """Compute new evidence-based staleness score."""
        try:
            node = await self._mound.get(node_id)  # type: ignore[misc]
            if not node:
                return 0.0

            # Query for supporting evidence added recently
            result = await self._mound.query_graph(
                start_id=node_id,
                relationship_types=[RelationshipType.SUPPORTS, RelationshipType.DERIVED_FROM],
                depth=1,
            )

            # Count new evidence
            cutoff = datetime.now() - self._config.evidence_recency_window
            new_evidence = 0

            for linked_node in result.nodes:
                if linked_node.id != node_id and linked_node.created_at > cutoff:
                    new_evidence += 1

            # More new evidence = potentially stale (needs review)
            if new_evidence > 0:
                return min(1.0, new_evidence / 5.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Failed to compute evidence score: {e}")
            return 0.0

    async def _compute_consensus_score(self, node_id: str) -> float:
        """Compute consensus change-based staleness score."""
        try:
            node = await self._mound.get(node_id)  # type: ignore[misc]
            if not node:
                return 0.0

            # Check if this is a consensus-derived node
            if node.source.value != "consensus":
                return 0.0

            # Check if the originating debate has had follow-up debates
            debate_id = node.metadata.get("debate_id")
            if not debate_id:
                return 0.0

            # Query for superseding consensus
            superseding = await self._mound.query_graph(
                start_id=node_id,
                relationship_types=[RelationshipType.SUPERSEDES],
                depth=1,
            )

            if superseding.edges:
                return 0.8  # Has been superseded

            return 0.0

        except Exception as e:
            logger.warning(f"Failed to compute consensus score: {e}")
            return 0.0

    def _emit_staleness_event(self, check: StalenessCheck, node: Any) -> None:
        """Emit KNOWLEDGE_STALE event for cross-subsystem integration."""
        if self._event_emitter is None:
            return

        try:
            from aragora.events.types import StreamEvent, StreamEventType

            event = StreamEvent(
                type=StreamEventType.KNOWLEDGE_STALE,
                data={
                    "node_id": check.node_id,
                    "staleness_score": check.staleness_score,
                    "reasons": [r.value for r in check.reasons],
                    "revalidation_recommended": check.revalidation_recommended,
                    "tier": node.metadata.get("tier", "slow") if node else "unknown",
                    "content_preview": (
                        node.content[:200] if node and hasattr(node, "content") else None
                    ),
                    "workspace_id": (node.metadata.get("workspace_id") if node else None),
                },
            )

            if hasattr(self._event_emitter, "emit"):
                self._event_emitter.emit(event)
            elif hasattr(self._event_emitter, "publish"):
                self._event_emitter.publish(event)
            elif callable(self._event_emitter):
                self._event_emitter(event)

            logger.debug(
                f"Emitted KNOWLEDGE_STALE: node={check.node_id}, "
                f"score={check.staleness_score:.2f}, reasons={check.reasons}"
            )

        except ImportError:
            pass  # Events module not available
        except Exception as e:
            logger.warning(f"Failed to emit staleness event: {e}")

    async def get_stale_nodes(
        self,
        workspace_id: str,
        threshold: float = 0.5,
        limit: int = 100,
    ) -> List[StalenessCheck]:
        """
        Get knowledge nodes that may need revalidation.

        Args:
            workspace_id: Workspace to check
            threshold: Minimum staleness score to include
            limit: Maximum number of results

        Returns:
            List of StalenessCheck results, sorted by staleness
        """
        # Get all nodes for workspace (paginated)
        all_checks: List[StalenessCheck] = []

        # Query nodes ordered by last update (oldest first)
        nodes = await self._mound.query_nodes(  # type: ignore[misc]
            workspace_id=workspace_id,
            limit=limit * 2,  # Over-fetch since some may not be stale
        )

        for node in nodes:
            check = await self.compute_staleness(node.id)
            if check.staleness_score >= threshold:
                all_checks.append(check)

            if len(all_checks) >= limit:
                break

        # Sort by staleness score descending
        all_checks.sort(key=lambda x: x.staleness_score, reverse=True)

        return all_checks[:limit]

    async def batch_check_staleness(
        self,
        node_ids: List[str],
    ) -> List[StalenessCheck]:
        """
        Check staleness for a batch of nodes.

        Args:
            node_ids: List of node IDs to check

        Returns:
            List of StalenessCheck results
        """
        results = []
        for node_id in node_ids:
            check = await self.compute_staleness(node_id)
            results.append(check)
        return results

    async def update_staleness_scores(
        self,
        workspace_id: str,
        batch_size: int = 100,
    ) -> int:
        """
        Update staleness scores for all nodes in a workspace.

        This should be run periodically as a background job.

        Args:
            workspace_id: Workspace to update
            batch_size: Number of nodes to process per batch

        Returns:
            Number of nodes updated
        """
        updated = 0
        offset = 0

        while True:
            # Get batch of nodes
            nodes = await self._mound.query_nodes(  # type: ignore[misc]
                workspace_id=workspace_id,
                limit=batch_size,
                offset=offset,
            )

            if not nodes:
                break

            # Compute and update staleness for each
            for node in nodes:
                check = await self.compute_staleness(node.id)
                await self._mound.update(  # type: ignore[misc]
                    node.id,
                    {"staleness_score": check.staleness_score},
                )
                updated += 1

            offset += batch_size

            # Safety limit
            if updated >= 10000:
                logger.warning(f"Staleness update capped at {updated} nodes")
                break

        logger.info(f"Updated staleness scores for {updated} nodes in {workspace_id}")
        return updated
