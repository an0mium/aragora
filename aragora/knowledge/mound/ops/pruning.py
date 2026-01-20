"""
Pruning operations for Knowledge Mound.

Provides automated and manual pruning of stale, low-quality,
or redundant knowledge items:
- Policy-based automatic pruning
- Batch prune operations
- Pruning history and audit trail
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.core import KnowledgeMoundCore


class PruningAction(str, Enum):
    """Actions that can be taken on prunable items."""

    ARCHIVE = "archive"
    DELETE = "delete"
    DEMOTE = "demote"  # Move to lower tier
    FLAG = "flag"  # Mark for review


@dataclass
class PruningPolicy:
    """Policy defining when and how to prune items."""

    policy_id: str
    workspace_id: str
    name: str
    enabled: bool = True

    # Staleness-based pruning
    staleness_threshold: float = 0.9  # Prune if staleness > threshold
    min_age_days: int = 30  # Don't prune items younger than this

    # Confidence-based pruning
    min_confidence: float = 0.0  # Prune if confidence < threshold (0 = disabled)
    confidence_decay_rate: float = 0.0  # Daily decay (0 = disabled)

    # Usage-based pruning
    min_retrieval_count: int = 0  # Prune if retrieved < threshold (0 = disabled)
    usage_window_days: int = 90  # Window for usage analysis

    # Actions
    action: PruningAction = PruningAction.ARCHIVE
    tier_exceptions: list[str] = field(
        default_factory=lambda: ["glacial"]
    )  # Don't prune these tiers

    # Schedule
    auto_prune: bool = False
    schedule_cron: Optional[str] = None  # e.g., "0 2 * * *" for 2am daily


@dataclass
class PrunableItem:
    """An item that matches pruning criteria."""

    node_id: str
    content_preview: str
    staleness_score: float
    confidence: float
    retrieval_count: int
    last_retrieved_at: Optional[datetime]
    tier: str
    created_at: datetime
    prune_reason: str
    recommended_action: PruningAction


@dataclass
class PruneResult:
    """Result of a pruning operation."""

    workspace_id: str
    executed_at: datetime
    policy_id: Optional[str]
    items_analyzed: int
    items_pruned: int
    items_archived: int
    items_deleted: int
    items_demoted: int
    items_flagged: int
    pruned_item_ids: list[str]
    errors: list[str] = field(default_factory=list)


@dataclass
class PruneHistory:
    """Historical record of pruning operations."""

    history_id: str
    workspace_id: str
    executed_at: datetime
    policy_id: Optional[str]
    action: PruningAction
    items_pruned: int
    pruned_item_ids: list[str]
    reason: str
    executed_by: Optional[str]


class PruningOperationsMixin:
    """Mixin providing pruning operations for Knowledge Mound."""

    async def get_prunable_items(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        staleness_threshold: float = 0.9,
        min_age_days: int = 30,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[PrunableItem]:
        """Find items that are candidates for pruning.

        Args:
            workspace_id: Workspace to analyze
            staleness_threshold: Minimum staleness score
            min_age_days: Minimum age to be prunable
            min_confidence: Maximum confidence to be prunable (0 = disabled)
            limit: Maximum items to return

        Returns:
            List of prunable items with recommendations
        """
        prunable = []

        # Get stale nodes
        stale_nodes = await self._staleness_detector.get_stale_nodes(
            workspace_id=workspace_id,
            threshold=staleness_threshold,
            limit=limit * 2,  # Get extra to filter
        )

        min_created = datetime.now() - timedelta(days=min_age_days)

        for node in stale_nodes:
            # Check age requirement
            if hasattr(node, "created_at") and node.created_at > min_created:
                continue

            # Determine prune reason
            reasons = []
            if node.staleness_score >= staleness_threshold:
                reasons.append(f"staleness={node.staleness_score:.2f}")
            if (
                min_confidence > 0
                and hasattr(node, "confidence")
                and node.confidence < min_confidence
            ):
                reasons.append(f"low_confidence={node.confidence:.2f}")

            # Determine recommended action based on tier
            tier = getattr(node, "tier", "medium")
            if tier == "glacial":
                action = PruningAction.FLAG  # Don't auto-prune important items
            elif tier == "slow":
                action = PruningAction.DEMOTE
            else:
                action = PruningAction.ARCHIVE

            prunable.append(
                PrunableItem(
                    node_id=node.id,
                    content_preview=node.content[:200] if hasattr(node, "content") else "",
                    staleness_score=node.staleness_score,
                    confidence=getattr(node, "confidence", 0.5),
                    retrieval_count=getattr(node, "retrieval_count", 0),
                    last_retrieved_at=getattr(node, "last_retrieved_at", None),
                    tier=tier,
                    created_at=getattr(node, "created_at", datetime.now()),
                    prune_reason=", ".join(reasons),
                    recommended_action=action,
                )
            )

            if len(prunable) >= limit:
                break

        return prunable

    async def prune_items(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        item_ids: list[str],
        action: PruningAction = PruningAction.ARCHIVE,
        reason: str = "manual_prune",
        executed_by: Optional[str] = None,
    ) -> PruneResult:
        """Prune specific items.

        Args:
            workspace_id: Workspace containing items
            item_ids: IDs of items to prune
            action: What action to take
            reason: Reason for pruning
            executed_by: User who initiated the prune

        Returns:
            Result of the pruning operation
        """
        archived = 0
        deleted = 0
        demoted = 0
        flagged = 0
        errors = []
        pruned_ids = []

        for node_id in item_ids:
            try:
                if action == PruningAction.ARCHIVE:
                    await self._store.archive_node(
                        node_id=node_id,
                        workspace_id=workspace_id,
                        reason=reason,
                    )
                    archived += 1
                    pruned_ids.append(node_id)

                elif action == PruningAction.DELETE:
                    await self._store.delete_node(
                        node_id=node_id,
                        workspace_id=workspace_id,
                    )
                    deleted += 1
                    pruned_ids.append(node_id)

                elif action == PruningAction.DEMOTE:
                    # Move to lower tier
                    node = await self._store.get_node(node_id, workspace_id)
                    tier_order = ["fast", "medium", "slow", "glacial"]
                    current_idx = tier_order.index(node.tier) if node.tier in tier_order else 0
                    new_tier = tier_order[min(current_idx + 1, len(tier_order) - 1)]
                    await self._store.update_node(
                        node_id=node_id,
                        workspace_id=workspace_id,
                        tier=new_tier,
                    )
                    demoted += 1
                    pruned_ids.append(node_id)

                elif action == PruningAction.FLAG:
                    # Mark for review
                    await self._store.update_node(
                        node_id=node_id,
                        workspace_id=workspace_id,
                        metadata={"flagged_for_review": True, "flagged_reason": reason},
                    )
                    flagged += 1
                    pruned_ids.append(node_id)

            except Exception as e:
                errors.append(f"Failed to prune {node_id}: {e!s}")

        # Record history
        await self._record_prune_history(
            workspace_id=workspace_id,
            action=action,
            item_ids=pruned_ids,
            reason=reason,
            executed_by=executed_by,
        )

        return PruneResult(
            workspace_id=workspace_id,
            executed_at=datetime.now(),
            policy_id=None,
            items_analyzed=len(item_ids),
            items_pruned=len(pruned_ids),
            items_archived=archived,
            items_deleted=deleted,
            items_demoted=demoted,
            items_flagged=flagged,
            pruned_item_ids=pruned_ids,
            errors=errors,
        )

    async def auto_prune(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        policy: PruningPolicy,
        dry_run: bool = True,
    ) -> PruneResult:
        """Automatically prune items based on policy.

        Args:
            workspace_id: Workspace to prune
            policy: Pruning policy to apply
            dry_run: If True, report what would be pruned without doing it

        Returns:
            Result of the pruning operation
        """
        if not policy.enabled:
            return PruneResult(
                workspace_id=workspace_id,
                executed_at=datetime.now(),
                policy_id=policy.policy_id,
                items_analyzed=0,
                items_pruned=0,
                items_archived=0,
                items_deleted=0,
                items_demoted=0,
                items_flagged=0,
                pruned_item_ids=[],
                errors=["Policy is disabled"],
            )

        # Find prunable items
        prunable = await self.get_prunable_items(
            workspace_id=workspace_id,
            staleness_threshold=policy.staleness_threshold,
            min_age_days=policy.min_age_days,
            min_confidence=policy.min_confidence,
            limit=500,
        )

        # Filter out tier exceptions
        prunable = [p for p in prunable if p.tier not in policy.tier_exceptions]

        if dry_run:
            return PruneResult(
                workspace_id=workspace_id,
                executed_at=datetime.now(),
                policy_id=policy.policy_id,
                items_analyzed=len(prunable),
                items_pruned=0,
                items_archived=0,
                items_deleted=0,
                items_demoted=0,
                items_flagged=0,
                pruned_item_ids=[],
                errors=[f"DRY RUN: Would prune {len(prunable)} items"],
            )

        # Execute pruning
        return await self.prune_items(
            workspace_id=workspace_id,
            item_ids=[p.node_id for p in prunable],
            action=policy.action,
            reason=f"auto_prune_policy_{policy.policy_id}",
            executed_by="system",
        )

    async def get_prune_history(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> list[PruneHistory]:
        """Get pruning history for a workspace.

        Args:
            workspace_id: Workspace to query
            limit: Maximum entries to return
            since: Only return entries after this time

        Returns:
            List of pruning history entries
        """
        return await self._store.get_prune_history(
            workspace_id=workspace_id,
            limit=limit,
            since=since,
        )

    async def restore_pruned_item(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        node_id: str,
    ) -> bool:
        """Restore an archived item.

        Args:
            workspace_id: Workspace containing the item
            node_id: ID of the archived item

        Returns:
            True if restored, False if not found or already active
        """
        return await self._store.restore_archived_node(
            node_id=node_id,
            workspace_id=workspace_id,
        )

    async def apply_confidence_decay(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        decay_rate: float = 0.01,
        min_confidence: float = 0.1,
    ) -> int:
        """Apply time-based confidence decay to knowledge items.

        Reduces confidence of items that haven't been validated
        or retrieved recently.

        Args:
            workspace_id: Workspace to process
            decay_rate: Daily decay rate (0.01 = 1% per day)
            min_confidence: Minimum confidence floor

        Returns:
            Number of items updated
        """
        updated = 0

        # Get items that haven't been validated recently
        nodes = await self._store.get_nodes_for_workspace(
            workspace_id=workspace_id,
            limit=1000,
        )

        for node in nodes:
            if not hasattr(node, "last_validated_at"):
                continue

            days_since_validation = (datetime.now() - node.last_validated_at).days
            if days_since_validation <= 0:
                continue

            # Calculate decayed confidence
            decay = decay_rate * days_since_validation
            new_confidence = max(node.confidence - decay, min_confidence)

            if new_confidence < node.confidence:
                await self._store.update_node(
                    node_id=node.id,
                    workspace_id=workspace_id,
                    confidence=new_confidence,
                )
                updated += 1

        return updated

    async def _record_prune_history(
        self: "KnowledgeMoundCore",
        workspace_id: str,
        action: PruningAction,
        item_ids: list[str],
        reason: str,
        executed_by: Optional[str],
    ) -> None:
        """Record a pruning operation in history."""
        history = PruneHistory(
            history_id=f"prune_{datetime.now().timestamp()}",
            workspace_id=workspace_id,
            executed_at=datetime.now(),
            policy_id=None,
            action=action,
            items_pruned=len(item_ids),
            pruned_item_ids=item_ids,
            reason=reason,
            executed_by=executed_by,
        )
        await self._store.save_prune_history(history)
