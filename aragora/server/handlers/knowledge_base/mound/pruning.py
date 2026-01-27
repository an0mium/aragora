"""
Pruning operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for managing knowledge quality through pruning:
- GET /api/knowledge/mound/pruning/items - Get prunable items
- POST /api/knowledge/mound/pruning/execute - Prune specified items
- POST /api/knowledge/mound/pruning/auto - Run auto-prune with policy
- GET /api/knowledge/mound/pruning/history - Get pruning history
- POST /api/knowledge/mound/pruning/restore - Restore archived item
- POST /api/knowledge/mound/pruning/decay - Apply confidence decay
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional


from aragora.rbac.decorators import require_permission

from ...base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class PruningOperationsMixin:
    """Mixin providing pruning API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=30)
    async def get_prunable_items(
        self,
        workspace_id: str,
        staleness_threshold: float = 0.9,
        min_age_days: int = 30,
        limit: int = 100,
    ) -> HandlerResult:
        """
        Get items eligible for pruning.

        GET /api/knowledge/mound/pruning/items?workspace_id=...&staleness_threshold=0.9

        Args:
            workspace_id: Workspace to analyze
            staleness_threshold: Minimum staleness score (0.0-1.0)
            min_age_days: Minimum item age in days
            limit: Maximum items to return

        Returns:
            List of PrunableItem objects
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            items = await mound.get_prunable_items(
                workspace_id=workspace_id,
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                limit=limit,
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "staleness_threshold": staleness_threshold,
                    "min_age_days": min_age_days,
                    "items_found": len(items),
                    "items": [
                        {
                            "node_id": item.node_id,
                            "content_preview": item.content_preview,
                            "staleness_score": item.staleness_score,
                            "confidence": item.confidence,
                            "retrieval_count": item.retrieval_count,
                            "last_retrieved_at": (
                                item.last_retrieved_at.isoformat()
                                if item.last_retrieved_at
                                else None
                            ),
                            "tier": item.tier,
                            "created_at": item.created_at.isoformat(),
                            "prune_reason": item.prune_reason,
                            "recommended_action": item.recommended_action.value,
                        }
                        for item in items
                    ],
                }
            )
        except Exception as e:
            logger.error(f"Error getting prunable items: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=10)
    async def execute_prune(
        self,
        workspace_id: str,
        item_ids: List[str],
        action: str = "archive",
        reason: str = "manual_prune",
    ) -> HandlerResult:
        """
        Execute pruning on specified items.

        POST /api/knowledge/mound/pruning/execute
        {
            "workspace_id": "...",
            "item_ids": ["id1", "id2"],
            "action": "archive",  // archive, delete, demote, flag
            "reason": "manual_prune"
        }

        Args:
            workspace_id: Workspace containing items
            item_ids: List of node IDs to prune
            action: Pruning action (archive, delete, demote, flag)
            reason: Reason for pruning

        Returns:
            PruneResult with counts
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id or not item_ids:
            return error_response("workspace_id and item_ids are required", status=400)

        from aragora.knowledge.mound.ops.pruning import PruningAction

        try:
            prune_action = PruningAction(action)
        except ValueError:
            valid_actions = [a.value for a in PruningAction]
            return error_response(
                f"Invalid action '{action}'. Valid: {valid_actions}",
                status=400,
            )

        try:
            result = await mound.prune_items(
                workspace_id=workspace_id,
                item_ids=item_ids,
                action=prune_action,
                reason=reason,
            )

            return json_response(
                {
                    "success": True,
                    "workspace_id": result.workspace_id,
                    "executed_at": result.executed_at.isoformat(),
                    "items_analyzed": result.items_analyzed,
                    "items_pruned": result.items_pruned,
                    "items_archived": result.items_archived,
                    "items_deleted": result.items_deleted,
                    "items_demoted": result.items_demoted,
                    "items_flagged": result.items_flagged,
                    "pruned_item_ids": result.pruned_item_ids,
                    "errors": result.errors if hasattr(result, "errors") else [],
                }
            )
        except Exception as e:
            logger.error(f"Error executing prune: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=5)
    async def auto_prune(
        self,
        workspace_id: str,
        policy_id: Optional[str] = None,
        staleness_threshold: float = 0.9,
        min_age_days: int = 30,
        action: str = "archive",
        dry_run: bool = True,
    ) -> HandlerResult:
        """
        Run automatic pruning with a policy.

        POST /api/knowledge/mound/pruning/auto
        {
            "workspace_id": "...",
            "staleness_threshold": 0.9,
            "min_age_days": 30,
            "action": "archive",
            "dry_run": true
        }

        Args:
            workspace_id: Workspace to process
            policy_id: Optional existing policy ID
            staleness_threshold: Staleness threshold for pruning
            min_age_days: Minimum age in days
            action: Default action for prunable items
            dry_run: If true, report only without making changes

        Returns:
            PruneResult with counts
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        from aragora.knowledge.mound.ops.pruning import PruningAction, PruningPolicy

        try:
            prune_action = PruningAction(action)
        except ValueError:
            valid_actions = [a.value for a in PruningAction]
            return error_response(
                f"Invalid action '{action}'. Valid: {valid_actions}",
                status=400,
            )

        try:
            policy = PruningPolicy(
                policy_id=policy_id or f"auto_{workspace_id}_{datetime.now().timestamp()}",
                workspace_id=workspace_id,
                name="Auto Prune Policy",
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                action=prune_action,
                enabled=True,
            )

            result = await mound.auto_prune(
                workspace_id=workspace_id,
                policy=policy,
                dry_run=dry_run,
            )

            return json_response(
                {
                    "success": True,
                    "workspace_id": result.workspace_id,
                    "policy_id": (
                        result.policy_id if hasattr(result, "policy_id") else policy.policy_id
                    ),
                    "dry_run": dry_run,
                    "executed_at": result.executed_at.isoformat(),
                    "items_analyzed": result.items_analyzed,
                    "items_pruned": result.items_pruned,
                    "items_archived": result.items_archived,
                    "items_deleted": result.items_deleted,
                    "items_demoted": result.items_demoted,
                    "items_flagged": result.items_flagged,
                    "errors": result.errors if hasattr(result, "errors") else [],
                }
            )
        except Exception as e:
            logger.error(f"Error in auto-prune: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=30)
    async def get_prune_history(
        self,
        workspace_id: str,
        limit: int = 50,
        since: Optional[str] = None,
    ) -> HandlerResult:
        """
        Get pruning history for a workspace.

        GET /api/knowledge/mound/pruning/history?workspace_id=...&limit=50

        Args:
            workspace_id: Workspace to query
            limit: Maximum entries to return
            since: ISO datetime to filter from

        Returns:
            List of PruneHistory entries
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            since_dt = datetime.fromisoformat(since) if since else None
        except ValueError:
            return error_response("Invalid 'since' datetime format", status=400)

        try:
            history = await mound.get_prune_history(
                workspace_id=workspace_id,
                limit=limit,
                since=since_dt,
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "entries": [
                        {
                            "history_id": h.history_id,
                            "executed_at": h.executed_at.isoformat(),
                            "policy_id": h.policy_id,
                            "action": (
                                h.action.value if hasattr(h.action, "value") else str(h.action)
                            ),
                            "items_pruned": h.items_pruned,
                            "pruned_item_ids": h.pruned_item_ids,
                            "reason": h.reason,
                            "executed_by": h.executed_by,
                        }
                        for h in history
                    ],
                }
            )
        except Exception as e:
            logger.error(f"Error getting prune history: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=20)
    async def restore_pruned_item(
        self,
        workspace_id: str,
        node_id: str,
    ) -> HandlerResult:
        """
        Restore an archived/pruned item.

        POST /api/knowledge/mound/pruning/restore
        {
            "workspace_id": "...",
            "node_id": "..."
        }

        Args:
            workspace_id: Workspace containing the item
            node_id: Node ID to restore

        Returns:
            Success status
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id or not node_id:
            return error_response("workspace_id and node_id are required", status=400)

        try:
            success = await mound.restore_pruned_item(
                workspace_id=workspace_id,
                node_id=node_id,
            )

            if success:
                return json_response(
                    {
                        "success": True,
                        "workspace_id": workspace_id,
                        "node_id": node_id,
                        "message": "Item restored successfully",
                    }
                )
            else:
                return error_response(
                    f"Could not restore item {node_id}. It may not exist or was deleted.",
                    status=404,
                )
        except Exception as e:
            logger.error(f"Error restoring pruned item: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:admin")
    @rate_limit(requests_per_minute=5)
    async def apply_confidence_decay(
        self,
        workspace_id: str,
        decay_rate: float = 0.01,
        min_confidence: float = 0.1,
    ) -> HandlerResult:
        """
        Apply confidence decay to knowledge items.

        POST /api/knowledge/mound/pruning/decay
        {
            "workspace_id": "...",
            "decay_rate": 0.01,
            "min_confidence": 0.1
        }

        Args:
            workspace_id: Workspace to process
            decay_rate: Decay rate per day (0.01 = 1%)
            min_confidence: Minimum confidence floor

        Returns:
            Number of items decayed
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        if not 0 < decay_rate < 1:
            return error_response("decay_rate must be between 0 and 1", status=400)

        if not 0 < min_confidence < 1:
            return error_response("min_confidence must be between 0 and 1", status=400)

        try:
            items_decayed = await mound.apply_confidence_decay(
                workspace_id=workspace_id,
                decay_rate=decay_rate,
                min_confidence=min_confidence,
            )

            return json_response(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "decay_rate": decay_rate,
                    "min_confidence": min_confidence,
                    "items_decayed": items_decayed,
                }
            )
        except Exception as e:
            logger.error(f"Error applying confidence decay: {e}")
            return error_response(safe_error_message(e), status=500)
