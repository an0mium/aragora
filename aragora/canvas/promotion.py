"""
Idea → Goal Promotion Bridge.

Promotes selected Stage 1 (Ideas) nodes into Stage 2 (Goals) nodes,
creating ProvenanceLink entries with SHA-256 content hashes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .models import Canvas, CanvasEdge, CanvasNode, CanvasNodeType, EdgeType, Position, Size
from .stages import (
    NODE_TYPE_COLORS,
    PipelineStage,
    ProvenanceLink,
    content_hash,
)

logger = logging.getLogger(__name__)

# Maps idea node types → appropriate goal node types
IDEA_TO_GOAL_TYPE: dict[str, str] = {
    "concept": "goal",
    "hypothesis": "strategy",
    "question": "risk",
    "observation": "milestone",
    "insight": "metric",
    "constraint": "risk",
    "assumption": "risk",
    "cluster": "goal",
    "evidence": "milestone",
}


def promote_ideas_to_goals(
    source_canvas: Canvas,
    node_ids: list[str],
    user_id: str,
) -> tuple[Canvas, list[ProvenanceLink]]:
    """Promote selected idea nodes to a new Stage 2 (Goals) canvas.

    Args:
        source_canvas: The Stage 1 ideas canvas containing the nodes.
        node_ids: IDs of nodes to promote.
        user_id: User performing the promotion.

    Returns:
        Tuple of (new goals canvas, provenance links).
    """
    goals_canvas = Canvas(
        id=f"goals-{uuid.uuid4().hex[:8]}",
        name=f"Goals from {source_canvas.name}",
        owner_id=source_canvas.owner_id,
        workspace_id=source_canvas.workspace_id,
        metadata={
            "stage": PipelineStage.GOALS.value,
            "source_canvas_id": source_canvas.id,
            "promoted_by": user_id,
        },
    )

    provenance_links: list[ProvenanceLink] = []
    y_offset = 0

    for node_id in node_ids:
        source_node = source_canvas.get_node(node_id)
        if source_node is None:
            logger.warning("Node %s not found in canvas %s", node_id, source_canvas.id)
            continue

        idea_type = source_node.data.get("idea_type", "concept")
        goal_type = IDEA_TO_GOAL_TYPE.get(idea_type, "goal")
        color = NODE_TYPE_COLORS.get(goal_type, "#34d399")

        goal_node_id = f"goal-{uuid.uuid4().hex[:8]}"

        goal_node = CanvasNode(
            id=goal_node_id,
            node_type=CanvasNodeType.DECISION,
            position=Position(x=100, y=y_offset),
            size=Size(250, 100),
            label=source_node.label,
            data={
                "stage": PipelineStage.GOALS.value,
                "goal_type": goal_type,
                "description": source_node.data.get("body", source_node.data.get("full_content", "")),
                "priority": "medium",
                "source_idea_type": idea_type,
                "source_node_id": node_id,
                "source_canvas_id": source_canvas.id,
                "content_hash": content_hash(source_node.label),
                "rf_type": "goalNode",
            },
            style={"backgroundColor": color, "borderRadius": "12px"},
        )
        goals_canvas.nodes[goal_node_id] = goal_node
        y_offset += 140

        # Mark source node as promoted
        source_node.data["promoted_to_goal_id"] = goal_node_id
        source_node.data["promoted_to_canvas_id"] = goals_canvas.id

        # Create provenance link
        link = ProvenanceLink(
            source_node_id=node_id,
            source_stage=PipelineStage.IDEAS,
            target_node_id=goal_node_id,
            target_stage=PipelineStage.GOALS,
            content_hash=content_hash(source_node.label),
            method="manual_promotion",
        )
        provenance_links.append(link)

    logger.info(
        "Promoted %d idea nodes to goals canvas %s",
        len(provenance_links),
        goals_canvas.id,
    )

    return goals_canvas, provenance_links


__all__ = ["promote_ideas_to_goals", "IDEA_TO_GOAL_TYPE"]
