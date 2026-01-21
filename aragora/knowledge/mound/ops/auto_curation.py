"""
Auto-Curation operations for Knowledge Mound (Phase 4).

Provides intelligent, automated knowledge maintenance:
- Quality-based scoring and ranking
- Automated tier optimization (promote/demote)
- Scheduled deduplication and pruning
- Usage-driven relevance tracking
- Debate-outcome feedback integration

Usage:
    from aragora.knowledge.mound import get_knowledge_mound

    mound = get_knowledge_mound()

    # Configure curation policy
    policy = CurationPolicy(
        workspace_id="my_workspace",
        quality_threshold=0.6,
        promotion_threshold=0.85,
    )
    await mound.set_curation_policy(policy)

    # Run curation (manually or via scheduler)
    result = await mound.run_curation("my_workspace")
    print(f"Promoted: {result.promoted_count}, Demoted: {result.demoted_count}")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CurationAction(str, Enum):
    """Actions that auto-curation can take."""

    PROMOTE = "promote"  # Move to faster tier
    DEMOTE = "demote"  # Move to slower tier
    MERGE = "merge"  # Merge with similar item
    ARCHIVE = "archive"  # Move to archive/glacial
    REFRESH = "refresh"  # Trigger revalidation
    FLAG = "flag"  # Mark for manual review


class TierLevel(str, Enum):
    """Knowledge tier levels in order of access speed."""

    HOT = "hot"  # Frequently accessed, fast retrieval
    WARM = "warm"  # Moderately accessed
    COLD = "cold"  # Infrequently accessed
    GLACIAL = "glacial"  # Archive tier


# Tier priority for promotion/demotion logic
TIER_ORDER = [TierLevel.HOT, TierLevel.WARM, TierLevel.COLD, TierLevel.GLACIAL]


@dataclass
class QualityScore:
    """Composite quality score for a knowledge item."""

    node_id: str
    overall_score: float  # 0-1 composite score

    # Component scores (0-1 each)
    freshness_score: float  # Based on staleness (inverted)
    confidence_score: float  # From original confidence + verification
    usage_score: float  # Based on retrieval frequency
    relevance_score: float  # Based on debate/query feedback
    relationship_score: float  # Based on graph connectivity

    # Metadata
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    debate_uses: int = 0  # Times used in debates
    retrieval_count: int = 0  # Total retrievals
    days_since_last_use: Optional[int] = None

    @property
    def recommendation(self) -> CurationAction:
        """Get recommended action based on score."""
        if self.overall_score >= 0.85:
            return CurationAction.PROMOTE
        elif self.overall_score < 0.3:
            return CurationAction.ARCHIVE
        elif self.overall_score < 0.5:
            return CurationAction.DEMOTE
        elif self.freshness_score < 0.3:
            return CurationAction.REFRESH
        return CurationAction.FLAG  # Default to review


@dataclass
class CurationPolicy:
    """Policy defining auto-curation behavior for a workspace."""

    workspace_id: str
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    name: str = "default"

    # Quality thresholds
    quality_threshold: float = 0.5  # Minimum quality to keep
    promotion_threshold: float = 0.85  # Promote if score exceeds
    demotion_threshold: float = 0.35  # Demote if score below
    archive_threshold: float = 0.2  # Archive if score below

    # Staleness settings
    refresh_staleness_threshold: float = 0.7  # Trigger revalidation above
    max_staleness_before_archive: float = 0.95

    # Usage settings
    usage_window_days: int = 30  # Window for usage analysis
    min_retrievals_for_promotion: int = 5  # Need at least N uses to promote
    debate_use_weight: float = 2.0  # Debates count double vs. regular retrieval

    # Score weights (must sum to 1.0)
    freshness_weight: float = 0.25
    confidence_weight: float = 0.20
    usage_weight: float = 0.25
    relevance_weight: float = 0.20
    relationship_weight: float = 0.10

    # Scheduling
    auto_curate: bool = True
    schedule_cron: Optional[str] = "0 3 * * *"  # 3am daily default
    max_items_per_run: int = 500

    # Dedup integration
    run_dedup: bool = True
    dedup_similarity_threshold: float = 0.92

    # Pruning integration
    run_pruning: bool = True
    prune_after_curation: bool = True

    def validate(self) -> bool:
        """Validate policy settings."""
        weights = [
            self.freshness_weight,
            self.confidence_weight,
            self.usage_weight,
            self.relevance_weight,
            self.relationship_weight,
        ]
        if abs(sum(weights) - 1.0) > 0.001:
            logger.warning(f"Policy weights don't sum to 1.0: {sum(weights)}")
            return False
        return True


@dataclass
class CurationCandidate:
    """An item being evaluated for curation."""

    node_id: str
    content_preview: str
    current_tier: str
    quality_score: QualityScore
    recommended_action: CurationAction
    target_tier: Optional[str] = None
    merge_target_id: Optional[str] = None


@dataclass
class CurationResult:
    """Result of a curation run."""

    workspace_id: str
    executed_at: datetime
    policy_id: str
    duration_ms: float

    # Analysis
    items_analyzed: int
    avg_quality_score: float

    # Actions taken
    promoted_count: int = 0
    demoted_count: int = 0
    archived_count: int = 0
    merged_count: int = 0
    refreshed_count: int = 0
    flagged_count: int = 0

    # Details
    promoted_ids: list[str] = field(default_factory=list)
    demoted_ids: list[str] = field(default_factory=list)
    archived_ids: list[str] = field(default_factory=list)
    merged_ids: list[str] = field(default_factory=list)
    flagged_ids: list[str] = field(default_factory=list)

    # Dedup/Prune integration
    dedup_clusters_merged: int = 0
    items_pruned: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        """Total number of actions taken."""
        return (
            self.promoted_count
            + self.demoted_count
            + self.archived_count
            + self.merged_count
            + self.refreshed_count
            + self.flagged_count
        )


@dataclass
class CurationHistory:
    """Historical record of curation operations."""

    history_id: str
    workspace_id: str
    executed_at: datetime
    policy_id: str
    result_summary: Dict[str, Any]
    items_affected: int
    trigger: str  # "scheduled", "manual", "threshold"


class AutoCurationMixin:
    """Mixin providing auto-curation operations for Knowledge Mound."""

    # Policies stored per workspace
    _curation_policies: Dict[str, CurationPolicy] = {}
    _curation_history: List[CurationHistory] = []

    async def set_curation_policy(
        self,
        policy: CurationPolicy,
    ) -> None:
        """Set the curation policy for a workspace.

        Args:
            policy: Curation policy to apply
        """
        if not policy.validate():
            raise ValueError("Invalid curation policy: weights must sum to 1.0")

        self._curation_policies[policy.workspace_id] = policy
        logger.info(f"Set curation policy '{policy.name}' for workspace {policy.workspace_id}")

    async def get_curation_policy(
        self,
        workspace_id: str,
    ) -> Optional[CurationPolicy]:
        """Get the curation policy for a workspace."""
        return self._curation_policies.get(workspace_id)

    async def calculate_quality_score(
        self,
        node_id: str,
        workspace_id: str,
        policy: Optional[CurationPolicy] = None,
    ) -> QualityScore:
        """Calculate composite quality score for a knowledge item.

        Args:
            node_id: Node to score
            workspace_id: Workspace context
            policy: Policy to use (or default)

        Returns:
            Quality score with component breakdown
        """
        policy = (
            policy
            or self._curation_policies.get(workspace_id)
            or CurationPolicy(workspace_id=workspace_id)
        )

        # Get node data using adapter method
        node = await self._get_node_by_id(node_id)  # type: ignore[attr-defined]
        if not node:
            raise ValueError(f"Node {node_id} not found")

        # Calculate freshness (inverse of staleness)
        staleness = await self._get_staleness_score(node_id, workspace_id)  # type: ignore[attr-defined]
        freshness_score = max(0.0, 1.0 - staleness)

        # Get confidence from node metadata
        confidence_score = node.get("confidence", 0.5)

        # Calculate usage score
        retrieval_count = node.get("retrieval_count", 0)
        debate_uses = node.get("debate_uses", 0)
        last_retrieved = node.get("last_retrieved_at")

        # Weighted usage with debate emphasis
        weighted_uses = retrieval_count + (debate_uses * policy.debate_use_weight)
        # Normalize: 10+ uses = 1.0, 0 uses = 0.0
        usage_score = min(1.0, weighted_uses / 10.0)

        # Calculate days since last use
        days_since_last_use = None
        if last_retrieved:
            if isinstance(last_retrieved, str):
                last_retrieved = datetime.fromisoformat(last_retrieved.replace("Z", "+00:00"))
            days_since_last_use = (datetime.now(timezone.utc) - last_retrieved).days
            # Decay usage score based on recency
            if days_since_last_use > policy.usage_window_days:
                usage_score *= 0.5  # Halve if outside window

        # Calculate relevance from debate feedback
        positive_feedback = node.get("positive_feedback", 0)
        negative_feedback = node.get("negative_feedback", 0)
        total_feedback = positive_feedback + negative_feedback
        if total_feedback > 0:
            relevance_score = positive_feedback / total_feedback
        else:
            relevance_score = 0.5  # Neutral if no feedback

        # Calculate relationship score (graph connectivity)
        relationship_count = node.get("relationship_count", 0)
        relationship_score = min(1.0, relationship_count / 5.0)  # 5+ relationships = 1.0

        # Compute weighted overall score
        overall_score = (
            freshness_score * policy.freshness_weight
            + confidence_score * policy.confidence_weight
            + usage_score * policy.usage_weight
            + relevance_score * policy.relevance_weight
            + relationship_score * policy.relationship_weight
        )

        return QualityScore(
            node_id=node_id,
            overall_score=overall_score,
            freshness_score=freshness_score,
            confidence_score=confidence_score,
            usage_score=usage_score,
            relevance_score=relevance_score,
            relationship_score=relationship_score,
            debate_uses=debate_uses,
            retrieval_count=retrieval_count,
            days_since_last_use=days_since_last_use,
        )

    async def get_curation_candidates(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[CurationCandidate]:
        """Get items that are candidates for curation actions.

        Args:
            workspace_id: Workspace to analyze
            limit: Maximum candidates to return

        Returns:
            List of candidates sorted by quality score (lowest first)
        """
        policy = self._curation_policies.get(workspace_id) or CurationPolicy(
            workspace_id=workspace_id
        )

        candidates = []

        # Get all nodes for workspace
        nodes = await self._get_nodes_for_workspace(  # type: ignore[attr-defined]
            workspace_id=workspace_id,
            limit=policy.max_items_per_run,
        )

        for node in nodes:
            try:
                score = await self.calculate_quality_score(
                    node_id=node["id"],
                    workspace_id=workspace_id,
                    policy=policy,
                )

                # Determine recommended action and target tier
                action = score.recommendation
                current_tier = node.get("tier", "warm")
                target_tier = None

                if action == CurationAction.PROMOTE:
                    target_tier = self._get_higher_tier(current_tier)
                elif action == CurationAction.DEMOTE:
                    target_tier = self._get_lower_tier(current_tier)
                elif action == CurationAction.ARCHIVE:
                    target_tier = TierLevel.GLACIAL.value

                candidate = CurationCandidate(
                    node_id=node["id"],
                    content_preview=node.get("content", "")[:100],
                    current_tier=current_tier,
                    quality_score=score,
                    recommended_action=action,
                    target_tier=target_tier,
                )
                candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Failed to score node {node.get('id')}: {e}")
                continue

        # Sort by quality score (lowest first - most needing action)
        candidates.sort(key=lambda c: c.quality_score.overall_score)

        return candidates[:limit]

    def _get_higher_tier(self, current_tier: str) -> Optional[str]:
        """Get the next higher (faster) tier."""
        tier_values = [t.value for t in TIER_ORDER]
        try:
            idx = tier_values.index(current_tier)
            if idx > 0:
                return tier_values[idx - 1]
        except ValueError:
            pass
        return None

    def _get_lower_tier(self, current_tier: str) -> Optional[str]:
        """Get the next lower (slower) tier."""
        tier_values = [t.value for t in TIER_ORDER]
        try:
            idx = tier_values.index(current_tier)
            if idx < len(tier_values) - 1:
                return tier_values[idx + 1]
        except ValueError:
            pass
        return None

    async def run_curation(
        self,
        workspace_id: str,
        dry_run: bool = False,
    ) -> CurationResult:
        """Run auto-curation for a workspace.

        Args:
            workspace_id: Workspace to curate
            dry_run: If True, analyze but don't take actions

        Returns:
            Curation result with actions taken
        """
        start_time = datetime.now(timezone.utc)
        policy = self._curation_policies.get(workspace_id) or CurationPolicy(
            workspace_id=workspace_id
        )

        result = CurationResult(
            workspace_id=workspace_id,
            executed_at=start_time,
            policy_id=policy.policy_id,
            duration_ms=0,
            items_analyzed=0,
            avg_quality_score=0.0,
        )

        try:
            # Get candidates
            candidates = await self.get_curation_candidates(
                workspace_id=workspace_id,
                limit=policy.max_items_per_run,
            )
            result.items_analyzed = len(candidates)

            if candidates:
                result.avg_quality_score = sum(
                    c.quality_score.overall_score for c in candidates
                ) / len(candidates)

            if dry_run:
                # Just return analysis without taking actions
                for candidate in candidates:
                    if candidate.recommended_action == CurationAction.PROMOTE:
                        result.promoted_count += 1
                        result.promoted_ids.append(candidate.node_id)
                    elif candidate.recommended_action == CurationAction.DEMOTE:
                        result.demoted_count += 1
                        result.demoted_ids.append(candidate.node_id)
                    elif candidate.recommended_action == CurationAction.ARCHIVE:
                        result.archived_count += 1
                        result.archived_ids.append(candidate.node_id)
                    elif candidate.recommended_action == CurationAction.FLAG:
                        result.flagged_count += 1
                        result.flagged_ids.append(candidate.node_id)
                return result

            # Execute curation actions
            for candidate in candidates:
                try:
                    action = candidate.recommended_action
                    score = candidate.quality_score

                    if action == CurationAction.PROMOTE and candidate.target_tier:
                        if score.retrieval_count >= policy.min_retrievals_for_promotion:
                            await self._move_to_tier(  # type: ignore[attr-defined]
                                node_id=candidate.node_id,
                                target_tier=candidate.target_tier,
                            )
                            result.promoted_count += 1
                            result.promoted_ids.append(candidate.node_id)

                    elif action == CurationAction.DEMOTE and candidate.target_tier:
                        await self._move_to_tier(  # type: ignore[attr-defined]
                            node_id=candidate.node_id,
                            target_tier=candidate.target_tier,
                        )
                        result.demoted_count += 1
                        result.demoted_ids.append(candidate.node_id)

                    elif action == CurationAction.ARCHIVE:
                        await self._move_to_tier(  # type: ignore[attr-defined]
                            node_id=candidate.node_id,
                            target_tier=TierLevel.GLACIAL.value,
                        )
                        result.archived_count += 1
                        result.archived_ids.append(candidate.node_id)

                    elif action == CurationAction.REFRESH:
                        await self._schedule_revalidation(  # type: ignore[attr-defined]
                            node_id=candidate.node_id,
                            priority="high",
                        )
                        result.refreshed_count += 1

                    elif action == CurationAction.FLAG:
                        await self._flag_for_review(  # type: ignore[attr-defined]
                            node_id=candidate.node_id,
                            reason="auto_curation_review",
                        )
                        result.flagged_count += 1
                        result.flagged_ids.append(candidate.node_id)

                except Exception as e:
                    result.errors.append(f"Action failed for {candidate.node_id}: {e}")
                    logger.warning(f"Curation action failed: {e}")

            # Run dedup if enabled
            if policy.run_dedup:
                try:
                    clusters = await self.find_duplicates(  # type: ignore[attr-defined]
                        workspace_id=workspace_id,
                        similarity_threshold=policy.dedup_similarity_threshold,
                    )
                    for cluster in clusters:
                        if cluster.recommended_action == "merge":
                            merge_result = await self.merge_duplicates(  # type: ignore[attr-defined]
                                cluster_id=cluster.cluster_id,
                            )
                            result.merged_count += len(merge_result.merged_node_ids)
                            result.merged_ids.extend(merge_result.merged_node_ids)
                            result.dedup_clusters_merged += 1
                except Exception as e:
                    result.errors.append(f"Dedup failed: {e}")
                    logger.warning(f"Dedup during curation failed: {e}")

            # Run pruning if enabled
            if policy.run_pruning and policy.prune_after_curation:
                try:
                    prune_result = await self.prune_workspace(  # type: ignore[attr-defined]
                        workspace_id=workspace_id,
                    )
                    result.items_pruned = prune_result.items_pruned
                except Exception as e:
                    result.errors.append(f"Pruning failed: {e}")
                    logger.warning(f"Pruning during curation failed: {e}")

        except Exception as e:
            result.errors.append(f"Curation failed: {e}")
            logger.error(f"Curation run failed for {workspace_id}: {e}")

        # Record timing
        end_time = datetime.now(timezone.utc)
        result.duration_ms = (end_time - start_time).total_seconds() * 1000

        # Record history
        history = CurationHistory(
            history_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            executed_at=start_time,
            policy_id=policy.policy_id,
            result_summary={
                "items_analyzed": result.items_analyzed,
                "total_actions": result.total_actions,
                "avg_quality": result.avg_quality_score,
                "errors": len(result.errors),
            },
            items_affected=result.total_actions,
            trigger="manual" if dry_run else "scheduled",
        )
        self._curation_history.append(history)

        logger.info(
            f"Curation completed for {workspace_id}: "
            f"analyzed={result.items_analyzed}, actions={result.total_actions}, "
            f"duration={result.duration_ms:.1f}ms"
        )

        return result

    async def get_curation_history(
        self,
        workspace_id: str,
        limit: int = 20,
    ) -> List[CurationHistory]:
        """Get curation history for a workspace.

        Args:
            workspace_id: Workspace to get history for
            limit: Maximum records to return

        Returns:
            List of history records, most recent first
        """
        history = [h for h in self._curation_history if h.workspace_id == workspace_id]
        history.sort(key=lambda h: h.executed_at, reverse=True)
        return history[:limit]

    async def get_workspace_quality_summary(
        self,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """Get quality summary for a workspace.

        Returns:
            Summary with tier distribution, avg scores, recommendations
        """
        candidates = await self.get_curation_candidates(
            workspace_id=workspace_id,
            limit=500,
        )

        if not candidates:
            return {
                "workspace_id": workspace_id,
                "total_items": 0,
                "avg_quality": 0.0,
                "tier_distribution": {},
                "action_recommendations": {},
                "needs_attention": False,
            }

        # Calculate stats
        tier_counts: Dict[str, int] = {}
        action_counts: Dict[str, int] = {}
        total_quality = 0.0

        for c in candidates:
            tier_counts[c.current_tier] = tier_counts.get(c.current_tier, 0) + 1
            action_counts[c.recommended_action.value] = (
                action_counts.get(c.recommended_action.value, 0) + 1
            )
            total_quality += c.quality_score.overall_score

        avg_quality = total_quality / len(candidates)

        # Determine if attention needed
        attention_actions = (
            action_counts.get(CurationAction.ARCHIVE.value, 0)
            + action_counts.get(CurationAction.FLAG.value, 0)
            + action_counts.get(CurationAction.REFRESH.value, 0)
        )
        needs_attention = attention_actions > len(candidates) * 0.3

        return {
            "workspace_id": workspace_id,
            "total_items": len(candidates),
            "avg_quality": round(avg_quality, 3),
            "tier_distribution": tier_counts,
            "action_recommendations": action_counts,
            "needs_attention": needs_attention,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }


# Exports
__all__ = [
    "AutoCurationMixin",
    "CurationPolicy",
    "CurationCandidate",
    "CurationResult",
    "CurationHistory",
    "CurationAction",
    "QualityScore",
    "TierLevel",
]
