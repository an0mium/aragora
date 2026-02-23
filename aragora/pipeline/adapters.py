"""
Bidirectional adapters between existing Aragora models and UniversalNode.

Converts CanvasNode, GoalNode, and argument dicts into UniversalNode
and back, enabling interop between the legacy per-stage models and the
unified graph layer.
"""

from __future__ import annotations

import uuid
from typing import Any

from aragora.canvas.models import (
    Canvas,
    CanvasEdge,
    CanvasNode,
    CanvasNodeType,
    EdgeType,
    Position,
    Size,
)
from aragora.canvas.stages import (
    GoalNodeType,
    PipelineStage,
    StageEdgeType,
    content_hash,
)
from aragora.goals.extractor import GoalNode
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)


# ── CanvasNode ↔ UniversalNode ──────────────────────────────────────────

def from_canvas_node(node: CanvasNode, stage: PipelineStage) -> UniversalNode:
    """Convert a CanvasNode to a UniversalNode."""
    subtype = node.data.get("idea_type", node.data.get("subtype", "concept"))
    return UniversalNode(
        id=node.id,
        stage=stage,
        node_subtype=subtype,
        label=node.label,
        description=node.data.get("full_content", node.data.get("description", "")),
        position_x=node.position.x,
        position_y=node.position.y,
        width=node.size.width,
        height=node.size.height,
        content_hash=node.data.get("content_hash", content_hash(node.label)),
        status="active",
        confidence=float(node.data.get("confidence", 0)),
        data={k: v for k, v in node.data.items()
              if k not in ("idea_type", "subtype", "full_content", "description",
                           "content_hash", "confidence")},
        style=node.style,
        metadata={"source_type": "canvas_node", "canvas_node_type": node.node_type.value},
        created_at=node.created_at.timestamp(),
        updated_at=node.updated_at.timestamp(),
    )


def to_canvas_node(unode: UniversalNode) -> CanvasNode:
    """Convert a UniversalNode back to a CanvasNode."""
    # Map stage+subtype to CanvasNodeType
    node_type_map: dict[str, CanvasNodeType] = {
        "concept": CanvasNodeType.KNOWLEDGE,
        "cluster": CanvasNodeType.GROUP,
        "question": CanvasNodeType.TEXT,
        "insight": CanvasNodeType.KNOWLEDGE,
        "evidence": CanvasNodeType.EVIDENCE,
        "goal": CanvasNodeType.DECISION,
        "task": CanvasNodeType.WORKFLOW,
        "agent_task": CanvasNodeType.AGENT,
        "debate": CanvasNodeType.DEBATE,
        "human_gate": CanvasNodeType.INPUT,
    }
    canvas_type = node_type_map.get(unode.node_subtype, CanvasNodeType.TEXT)

    # Restore from metadata if the original type was saved
    original_type = unode.metadata.get("canvas_node_type")
    if original_type:
        try:
            canvas_type = CanvasNodeType(original_type)
        except ValueError:
            pass

    data = {
        "idea_type": unode.node_subtype,
        "content_hash": unode.content_hash,
        "stage": unode.stage.value,
        **unode.data,
    }
    if unode.description:
        data["full_content"] = unode.description
    if unode.confidence:
        data["confidence"] = unode.confidence

    return CanvasNode(
        id=unode.id,
        node_type=canvas_type,
        position=Position(x=unode.position_x, y=unode.position_y),
        size=Size(width=unode.width, height=unode.height),
        label=unode.label,
        data=data,
        style=unode.style,
    )


# ── GoalNode ↔ UniversalNode ───────────────────────────────────────────

def from_goal_node(goal: GoalNode) -> UniversalNode:
    """Convert a GoalNode to a UniversalNode."""
    return UniversalNode(
        id=goal.id,
        stage=PipelineStage.GOALS,
        node_subtype=goal.goal_type.value,
        label=goal.title,
        description=goal.description,
        content_hash=content_hash(goal.title + goal.description),
        parent_ids=list(goal.source_idea_ids),
        source_stage=PipelineStage.IDEAS if goal.source_idea_ids else None,
        status="active",
        confidence=goal.confidence,
        data={
            "priority": goal.priority,
            "measurable": goal.measurable,
        },
        metadata={
            "source_type": "goal_node",
            "dependencies": goal.dependencies,
            **goal.metadata,
        },
    )


def to_goal_node(unode: UniversalNode) -> GoalNode:
    """Convert a UniversalNode back to a GoalNode."""
    try:
        goal_type = GoalNodeType(unode.node_subtype)
    except ValueError:
        goal_type = GoalNodeType.GOAL

    return GoalNode(
        id=unode.id,
        title=unode.label,
        description=unode.description,
        goal_type=goal_type,
        priority=unode.data.get("priority", "medium"),
        measurable=unode.data.get("measurable", ""),
        dependencies=unode.metadata.get("dependencies", []),
        source_idea_ids=list(unode.parent_ids),
        confidence=unode.confidence,
        metadata={k: v for k, v in unode.metadata.items()
                  if k not in ("source_type", "dependencies")},
    )


# ── Argument dict → UniversalNode ──────────────────────────────────────

def from_argument_node(
    node: dict[str, Any],
    stage: PipelineStage = PipelineStage.IDEAS,
) -> UniversalNode:
    """Convert an ArgumentCartographer node dict to a UniversalNode."""
    label = node.get("label", node.get("content", ""))
    node_type = node.get("node_type", node.get("type", "concept"))
    return UniversalNode(
        id=node.get("id", str(uuid.uuid4())),
        stage=stage,
        node_subtype=node_type if node_type in ("concept", "cluster", "question",
                                                  "insight", "evidence", "assumption",
                                                  "constraint") else "concept",
        label=label,
        description=node.get("description", ""),
        content_hash=content_hash(label),
        confidence=float(node.get("weight", node.get("confidence", 0))),
        data={k: v for k, v in node.items()
              if k not in ("id", "label", "content", "node_type", "type",
                           "description", "weight", "confidence")},
        metadata={"source_type": "argument_node"},
    )


# ── Bulk conversions ────────────────────────────────────────────────────

def canvas_to_universal_graph(
    canvas: Canvas,
    stage: PipelineStage,
) -> UniversalGraph:
    """Convert an entire Canvas to a UniversalGraph."""
    graph = UniversalGraph(
        id=canvas.id,
        name=canvas.name,
        owner_id=canvas.owner_id,
        workspace_id=canvas.workspace_id,
        metadata=canvas.metadata,
        created_at=canvas.created_at.timestamp(),
        updated_at=canvas.updated_at.timestamp(),
    )

    # Convert nodes
    for node in canvas.nodes.values():
        unode = from_canvas_node(node, stage)
        graph.nodes[unode.id] = unode

    # Convert edges
    _edge_type_map: dict[str, StageEdgeType] = {
        "default": StageEdgeType.RELATES_TO,
        "data_flow": StageEdgeType.FOLLOWS,
        "control_flow": StageEdgeType.FOLLOWS,
        "reference": StageEdgeType.RELATES_TO,
        "dependency": StageEdgeType.REQUIRES,
        "critique": StageEdgeType.REFUTES,
        "support": StageEdgeType.SUPPORTS,
    }

    for edge in canvas.edges.values():
        edge_type = _edge_type_map.get(edge.edge_type.value, StageEdgeType.RELATES_TO)
        uedge = UniversalEdge(
            id=edge.id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_type=edge_type,
            label=edge.label,
            data=edge.data,
            created_at=edge.created_at.timestamp(),
        )
        graph.add_edge(uedge)

    return graph


def universal_graph_to_canvas(
    graph: UniversalGraph,
    stage: PipelineStage,
) -> Canvas:
    """Convert a UniversalGraph (filtered to one stage) back to a Canvas."""
    canvas = Canvas(
        id=graph.id,
        name=graph.name,
        owner_id=graph.owner_id,
        workspace_id=graph.workspace_id,
        metadata=graph.metadata,
    )

    # Convert nodes for the requested stage
    stage_nodes = graph.get_stage(stage)
    for unode in stage_nodes:
        cnode = to_canvas_node(unode)
        canvas.nodes[cnode.id] = cnode

    # Convert edges where both endpoints are in this stage
    node_ids = {n.id for n in stage_nodes}
    _reverse_edge_map: dict[StageEdgeType, EdgeType] = {
        StageEdgeType.RELATES_TO: EdgeType.DEFAULT,
        StageEdgeType.FOLLOWS: EdgeType.CONTROL_FLOW,
        StageEdgeType.REQUIRES: EdgeType.DEPENDENCY,
        StageEdgeType.SUPPORTS: EdgeType.SUPPORT,
        StageEdgeType.REFUTES: EdgeType.CRITIQUE,
    }

    for uedge in graph.edges.values():
        if uedge.source_id in node_ids and uedge.target_id in node_ids:
            edge_type = _reverse_edge_map.get(uedge.edge_type, EdgeType.DEFAULT)
            cedge = CanvasEdge(
                id=uedge.id,
                source_id=uedge.source_id,
                target_id=uedge.target_id,
                edge_type=edge_type,
                label=uedge.label,
                data=uedge.data,
            )
            canvas.edges[cedge.id] = cedge

    return canvas


__all__ = [
    "from_canvas_node",
    "to_canvas_node",
    "from_goal_node",
    "to_goal_node",
    "from_argument_node",
    "canvas_to_universal_graph",
    "universal_graph_to_canvas",
]
