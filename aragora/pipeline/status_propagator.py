"""Bidirectional Status Propagation for the Pipeline.

When execution succeeds or fails, propagates status backward through
the provenance chain so upstream ideas/goals show real-time execution state.

Usage:
    propagator = StatusPropagator(graph)
    propagator.propagate_status(node_id, "succeeded", PipelineStage.ORCHESTRATION)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.canvas.stages import PipelineStage
from aragora.pipeline.universal_node import UniversalGraph, UniversalNode

logger = logging.getLogger(__name__)

# Valid execution statuses
EXECUTION_STATUSES = {"pending", "in_progress", "succeeded", "failed", "partial"}


def _aggregate_status(child_statuses: list[str]) -> str:
    """Compute parent status from child statuses.

    Rules:
    - All children succeeded → "succeeded"
    - Any child failed → "partial" (if some succeeded) or "failed" (if all failed)
    - Any child in_progress → "in_progress"
    - Otherwise → "pending"
    """
    if not child_statuses:
        return "pending"

    status_set = set(child_statuses)

    if status_set == {"succeeded"}:
        return "succeeded"
    if status_set == {"failed"}:
        return "failed"
    if "in_progress" in status_set:
        return "in_progress"
    if "succeeded" in status_set and "failed" in status_set:
        return "partial"
    if "succeeded" in status_set:
        return "partial"
    return "pending"


class StatusPropagator:
    """Propagates execution status backward through the provenance chain.

    When a node's status changes, walks parent_ids to update ancestor
    nodes with aggregated child status.
    """

    def __init__(
        self,
        graph: UniversalGraph,
        emit_event: Any | None = None,
    ):
        self.graph = graph
        self._emit = emit_event  # Optional WebSocket event emitter

    def propagate_status(
        self,
        node_id: str,
        status: str,
        stage: PipelineStage | None = None,
    ) -> list[str]:
        """Set status on a node and propagate backward through provenance.

        Args:
            node_id: The node whose status changed.
            status: New execution status.
            stage: Optional stage hint (for logging).

        Returns:
            List of node IDs that were updated.
        """
        if status not in EXECUTION_STATUSES:
            logger.warning("Invalid status '%s', ignoring", status)
            return []

        node = self.graph.nodes.get(node_id)
        if node is None:
            logger.warning("Node %s not found in graph", node_id)
            return []

        # Set status on the target node
        node.metadata["execution_status"] = status
        updated = [node_id]

        # Emit event for frontend
        self._emit_status_event(node_id, status, node.stage)

        # Walk backward through parent_ids
        visited: set[str] = {node_id}
        queue = list(node.parent_ids)

        while queue:
            parent_id = queue.pop(0)
            if parent_id in visited:
                continue
            visited.add(parent_id)

            parent = self.graph.nodes.get(parent_id)
            if parent is None:
                continue

            # Gather all children of this parent
            child_statuses = self._get_child_statuses(parent_id)
            aggregated = _aggregate_status(child_statuses)

            old_status = parent.metadata.get("execution_status", "pending")
            if aggregated != old_status:
                parent.metadata["execution_status"] = aggregated
                updated.append(parent_id)
                self._emit_status_event(parent_id, aggregated, parent.stage)

                # Continue propagating upward
                queue.extend(parent.parent_ids)

        if len(updated) > 1:
            logger.info(
                "status_propagated source=%s status=%s affected=%d",
                node_id,
                status,
                len(updated),
            )

        return updated

    def _get_child_statuses(self, parent_id: str) -> list[str]:
        """Get execution statuses of all nodes that have parent_id as a parent."""
        statuses: list[str] = []
        for node in self.graph.nodes.values():
            if parent_id in node.parent_ids:
                status = node.metadata.get("execution_status", "pending")
                statuses.append(status)
        return statuses

    def _emit_status_event(
        self,
        node_id: str,
        status: str,
        stage: PipelineStage | None,
    ) -> None:
        """Emit a WebSocket event for status changes."""
        if self._emit is None:
            return

        try:
            self._emit(
                "pipeline:status_changed",
                {
                    "node_id": node_id,
                    "stage": stage.value if stage else None,
                    "status": status,
                },
            )
        except (RuntimeError, TypeError, AttributeError) as e:
            logger.debug("Status event emission failed: %s", e)


__all__ = [
    "StatusPropagator",
    "EXECUTION_STATUSES",
]
