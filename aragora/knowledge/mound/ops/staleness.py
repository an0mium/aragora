"""
Staleness Operations Mixin for Knowledge Mound.

Provides staleness detection and revalidation:
- get_stale_knowledge: Identify stale nodes
- mark_validated: Update validation status
- schedule_revalidation: Queue nodes for review
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem, MoundConfig, StalenessCheck

logger = logging.getLogger(__name__)


class StalenessProtocol(Protocol):
    """Protocol defining expected interface for Staleness mixin."""

    config: "MoundConfig"
    workspace_id: str
    _staleness_detector: Optional[Any]
    _initialized: bool

    def _ensure_initialized(self) -> None: ...
    async def update(self, node_id: str, updates: Dict[str, Any]) -> Optional["KnowledgeItem"]: ...


class StalenessOperationsMixin:
    """Mixin providing staleness management for KnowledgeMound."""

    async def get_stale_knowledge(
        self: StalenessProtocol,
        threshold: float = 0.5,
        limit: int = 100,
        workspace_id: Optional[str] = None,
    ) -> List["StalenessCheck"]:
        """Get knowledge items that may be stale."""
        self._ensure_initialized()

        if not self._staleness_detector:
            return []

        ws_id = workspace_id or self.workspace_id
        return await self._staleness_detector.get_stale_nodes(
            workspace_id=ws_id,
            threshold=threshold,
            limit=limit,
        )

    async def mark_validated(
        self: StalenessProtocol,
        node_id: str,
        validator: str,
        confidence: Optional[float] = None,
    ) -> None:
        """Mark a knowledge node as validated."""
        self._ensure_initialized()

        updates: Dict[str, Any] = {
            "validation_status": "majority_agreed",  # Valid ValidationStatus value
            "last_validated_at": datetime.now().isoformat(),
            "staleness_score": 0.0,
        }
        if confidence is not None:
            updates["confidence"] = confidence

        await self.update(node_id, updates)

    async def schedule_revalidation(
        self: StalenessProtocol,
        node_ids: List[str],
        priority: str = "low",
    ) -> List[str]:
        """
        Schedule nodes for revalidation via the control plane.

        Creates tasks in the control plane task queue for each node
        that needs revalidation. Workers will pick up these tasks
        and run revalidation debates/checks.

        Args:
            node_ids: List of node IDs to revalidate
            priority: Task priority ("low", "normal", "high")

        Returns:
            List of created task IDs
        """
        self._ensure_initialized()

        task_ids = []
        now = datetime.now().isoformat()

        for node_id in node_ids:
            # Mark node as needing revalidation
            await self.update(node_id, {"revalidation_requested": True})

            # Create control plane task
            task_id = f"reval_{uuid.uuid4().hex[:12]}"
            task = {
                "id": task_id,
                "type": "knowledge_revalidation",
                "priority": priority,
                "status": "pending",
                "node_id": node_id,
                "workspace_id": self.workspace_id,
                "created_at": now,
                "metadata": {
                    "source": "knowledge_mound",
                    "action": "revalidate",
                },
            }

            # Add to control plane queue
            try:
                from aragora.server.handlers.features.control_plane import _task_queue

                _task_queue.append(task)
                task_ids.append(task_id)
                logger.debug(f"Scheduled revalidation task {task_id} for node {node_id}")
            except ImportError:
                # Control plane not available, just log
                logger.warning(
                    f"Control plane not available, revalidation for {node_id} marked but not queued"
                )
                task_ids.append(f"pending_{node_id}")

        logger.info(f"Scheduled {len(task_ids)} revalidation tasks with priority={priority}")
        return task_ids
