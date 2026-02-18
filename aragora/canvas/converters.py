"""
Stage Converters for the Idea-to-Execution Pipeline.

Converts existing Aragora structures into Canvas nodes/edges with
stage-aware metadata, producing React Flow-compatible JSON output.

Converters:
- ArgumentCartographer → Stage 1 (Ideas) canvas
- GoalGraph → Stage 2 (Goals) canvas
- WorkflowDefinition → Stage 3 (Actions) canvas
- Arena execution plan → Stage 4 (Orchestration) canvas
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any

from .models import Canvas, CanvasEdge, CanvasNode, CanvasNodeType, EdgeType, Position, Size
from .stages import (
    NODE_TYPE_COLORS,
    STAGE_COLORS,
    PipelineStage,
    ProvenanceLink,
    StageEdgeType,
    content_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# React Flow JSON format
# =============================================================================


def to_react_flow(canvas: Canvas) -> dict[str, Any]:
    """Convert a Canvas to React Flow-compatible JSON.

    React Flow expects:
    - nodes: [{id, type, position: {x, y}, data: {...}, style: {...}}]
    - edges: [{id, source, target, type, label, animated, style: {...}}]
    """
    rf_nodes = []
    for node in canvas.nodes.values():
        rf_node: dict[str, Any] = {
            "id": node.id,
            "type": node.data.get("rf_type", "default"),
            "position": node.position.to_dict(),
            "data": {
                "label": node.label,
                "nodeType": node.node_type.value,
                **node.data,
            },
        }
        if node.style:
            rf_node["style"] = node.style
        if node.size.width != 200 or node.size.height != 100:
            rf_node["style"] = {
                **(rf_node.get("style") or {}),
                "width": node.size.width,
                "height": node.size.height,
            }
        if node.parent_id:
            rf_node["parentNode"] = node.parent_id
            rf_node["extent"] = "parent"
        rf_nodes.append(rf_node)

    rf_edges = []
    for edge in canvas.edges.values():
        rf_edge: dict[str, Any] = {
            "id": edge.id,
            "source": edge.source_id,
            "target": edge.target_id,
            "type": _edge_type_to_rf(edge.edge_type),
        }
        if edge.label:
            rf_edge["label"] = edge.label
        if edge.animated:
            rf_edge["animated"] = True
        if edge.style:
            rf_edge["style"] = edge.style
        if edge.data:
            rf_edge["data"] = edge.data
        rf_edges.append(rf_edge)

    return {
        "nodes": rf_nodes,
        "edges": rf_edges,
        "metadata": {
            "canvas_id": canvas.id,
            "canvas_name": canvas.name,
            "stage": canvas.metadata.get("stage", "unknown"),
            **canvas.metadata,
        },
    }


def _edge_type_to_rf(edge_type: EdgeType) -> str:
    """Map canvas EdgeType to React Flow edge type."""
    mapping = {
        EdgeType.DEFAULT: "default",
        EdgeType.DATA_FLOW: "smoothstep",
        EdgeType.CONTROL_FLOW: "step",
        EdgeType.REFERENCE: "straight",
        EdgeType.DEPENDENCY: "smoothstep",
        EdgeType.CRITIQUE: "bezier",
        EdgeType.SUPPORT: "bezier",
    }
    return mapping.get(edge_type, "default")


# =============================================================================
# Stage 1: ArgumentCartographer → Ideas Canvas
# =============================================================================


def debate_to_ideas_canvas(
    cartographer_data: dict[str, Any],
    canvas_id: str | None = None,
    canvas_name: str = "Idea Map",
) -> Canvas:
    """Convert ArgumentCartographer export to a Stage 1 Ideas canvas.

    Maps debate node types to idea node types:
    - PROPOSAL → concept
    - CRITIQUE → question (challenges the idea)
    - EVIDENCE → evidence
    - CONCESSION → insight (agreement reveals shared understanding)
    - REBUTTAL → assumption (defending a position)
    - VOTE → insight
    - CONSENSUS → cluster (agreed-upon idea group)

    Args:
        cartographer_data: Output from ArgumentCartographer.to_dict()
        canvas_id: Optional canvas ID
        canvas_name: Canvas name

    Returns:
        Canvas with Stage 1 idea nodes and relationship edges
    """
    canvas = Canvas(
        id=canvas_id or f"ideas-{uuid.uuid4().hex[:8]}",
        name=canvas_name,
        metadata={"stage": PipelineStage.IDEAS.value, "source": "debate"},
    )

    # Map debate node types to idea node types
    type_map = {
        "proposal": "concept",
        "critique": "question",
        "evidence": "evidence",
        "concession": "insight",
        "rebuttal": "assumption",
        "vote": "insight",
        "consensus": "cluster",
    }

    nodes_data = cartographer_data.get("nodes", [])
    edges_data = cartographer_data.get("edges", [])

    # Layout nodes in a force-directed-like radial layout
    positions = _radial_layout(len(nodes_data))

    for i, node_data in enumerate(nodes_data):
        node_type_str = node_data.get("type", "proposal").lower()
        idea_type = type_map.get(node_type_str, "concept")
        color = NODE_TYPE_COLORS.get(idea_type, "#818cf8")

        pos = positions[i] if i < len(positions) else Position(0, 0)
        summary = node_data.get("summary", node_data.get("content", "")[:80])

        canvas_node = CanvasNode(
            id=node_data.get("id", f"idea-{i}"),
            node_type=CanvasNodeType.KNOWLEDGE,
            position=pos,
            size=Size(220, 80),
            label=summary,
            data={
                "stage": PipelineStage.IDEAS.value,
                "idea_type": idea_type,
                "agent": node_data.get("agent", ""),
                "round": node_data.get("round_num", 0),
                "full_content": node_data.get("full_content", summary),
                "content_hash": content_hash(summary),
                "rf_type": "ideaNode",
            },
            style={"backgroundColor": color, "borderRadius": "8px"},
        )
        canvas.nodes[canvas_node.id] = canvas_node

    # Map debate edges to idea relationship edges
    edge_map = {
        "supports": EdgeType.SUPPORT,
        "refutes": EdgeType.CRITIQUE,
        "modifies": EdgeType.REFERENCE,
        "responds_to": EdgeType.REFERENCE,
        "concedes_to": EdgeType.SUPPORT,
    }

    for edge_data in edges_data:
        relation = edge_data.get("relation", "responds_to").lower()
        edge_type = edge_map.get(relation, EdgeType.REFERENCE)

        canvas_edge = CanvasEdge(
            id=f"e-{edge_data.get('source_id', '')}-{edge_data.get('target_id', '')}",
            source_id=edge_data.get("source_id", ""),
            target_id=edge_data.get("target_id", ""),
            edge_type=edge_type,
            label=relation.replace("_", " "),
            data={
                "stage": PipelineStage.IDEAS.value,
                "stage_edge_type": relation,
            },
            animated=relation in ("supports", "refutes"),
        )
        canvas.edges[canvas_edge.id] = canvas_edge

    return canvas


# =============================================================================
# Stage 2: Goals Canvas (from extracted goals)
# =============================================================================


def goals_to_canvas(
    goals: list[dict[str, Any]],
    provenance: list[ProvenanceLink] | None = None,
    canvas_id: str | None = None,
    canvas_name: str = "Goal Map",
) -> Canvas:
    """Convert extracted goals into a Stage 2 Goals canvas.

    Args:
        goals: List of goal dicts with keys:
            - id, type (goal/principle/strategy/milestone/metric/risk),
              title, description, priority, dependencies (list of goal IDs)
        provenance: Links back to Stage 1 idea nodes
        canvas_id: Optional canvas ID
        canvas_name: Canvas name

    Returns:
        Canvas with Stage 2 goal nodes and dependency edges
    """
    canvas = Canvas(
        id=canvas_id or f"goals-{uuid.uuid4().hex[:8]}",
        name=canvas_name,
        metadata={"stage": PipelineStage.GOALS.value},
    )

    # Layout goals in a hierarchical top-down layout
    positions = _hierarchical_layout(goals)

    for i, goal in enumerate(goals):
        goal_type = goal.get("type", "goal")
        color = NODE_TYPE_COLORS.get(goal_type, "#34d399")
        pos = positions.get(goal.get("id", f"g-{i}"), Position(i * 280, 0))

        canvas_node = CanvasNode(
            id=goal.get("id", f"goal-{i}"),
            node_type=CanvasNodeType.DECISION,
            position=pos,
            size=Size(250, 100),
            label=goal.get("title", "Untitled Goal"),
            data={
                "stage": PipelineStage.GOALS.value,
                "goal_type": goal_type,
                "description": goal.get("description", ""),
                "priority": goal.get("priority", "medium"),
                "content_hash": content_hash(goal.get("title", "")),
                "rf_type": "goalNode",
            },
            style={"backgroundColor": color, "borderRadius": "12px"},
        )
        canvas.nodes[canvas_node.id] = canvas_node

    # Add dependency edges between goals
    for goal in goals:
        goal_id = goal.get("id", "")
        for dep_id in goal.get("dependencies", []):
            canvas.edges[f"e-{dep_id}-{goal_id}"] = CanvasEdge(
                id=f"e-{dep_id}-{goal_id}",
                source_id=dep_id,
                target_id=goal_id,
                edge_type=EdgeType.DEPENDENCY,
                label="requires",
                data={"stage": PipelineStage.GOALS.value},
            )

    # Add provenance edges linking back to ideas
    if provenance:
        for link in provenance:
            canvas.edges[f"prov-{link.source_node_id}-{link.target_node_id}"] = CanvasEdge(
                id=f"prov-{link.source_node_id}-{link.target_node_id}",
                source_id=link.source_node_id,
                target_id=link.target_node_id,
                edge_type=EdgeType.REFERENCE,
                label="derived from",
                data={
                    "provenance": True,
                    "content_hash": link.content_hash,
                    "method": link.method,
                },
                style={"strokeDasharray": "5 5", "opacity": 0.5},
            )

    return canvas


# =============================================================================
# Stage 3: WorkflowDefinition → Actions Canvas
# =============================================================================


def workflow_to_actions_canvas(
    workflow_data: dict[str, Any],
    provenance: list[ProvenanceLink] | None = None,
    canvas_id: str | None = None,
    canvas_name: str = "Action Plan",
) -> Canvas:
    """Convert a WorkflowDefinition dict into a Stage 3 Actions canvas.

    Args:
        workflow_data: WorkflowDefinition.to_dict() output with steps and transitions
        canvas_id: Optional canvas ID
        canvas_name: Canvas name

    Returns:
        Canvas with Stage 3 action nodes and flow edges
    """
    canvas = Canvas(
        id=canvas_id or f"actions-{uuid.uuid4().hex[:8]}",
        name=canvas_name,
        metadata={"stage": PipelineStage.ACTIONS.value},
    )

    steps = workflow_data.get("steps", [])
    transitions = workflow_data.get("transitions", [])

    # Use visual positions if available, otherwise auto-layout
    for i, step in enumerate(steps):
        visual = step.get("visual", {})
        pos_data = visual.get("position", {})
        pos = Position(
            x=pos_data.get("x", i * 300),
            y=pos_data.get("y", (i % 3) * 150),
        )

        step_type = step.get("step_type", "task")
        color = NODE_TYPE_COLORS.get(
            "checkpoint" if step_type == "human_checkpoint" else "task",
            "#fbbf24",
        )

        canvas_node = CanvasNode(
            id=step.get("id", f"step-{i}"),
            node_type=CanvasNodeType.WORKFLOW,
            position=pos,
            size=Size(240, 90),
            label=step.get("name", f"Step {i + 1}"),
            data={
                "stage": PipelineStage.ACTIONS.value,
                "step_type": step_type,
                "phase": step.get("phase", ""),
                "source_goal_id": step.get("source_goal_id", ""),
                "description": step.get("description", ""),
                "timeout": step.get("timeout_seconds", 0),
                "retries": step.get("retries", 0),
                "optional": step.get("optional", False),
                "content_hash": content_hash(step.get("name", "")),
                "rf_type": "actionNode",
            },
            style={"backgroundColor": color, "borderRadius": "6px"},
        )
        canvas.nodes[canvas_node.id] = canvas_node

    # Add transition edges
    for trans in transitions:
        edge_id = trans.get("id", f"t-{trans.get('from_step')}-{trans.get('to_step')}")
        canvas.edges[edge_id] = CanvasEdge(
            id=edge_id,
            source_id=trans.get("from_step", ""),
            target_id=trans.get("to_step", ""),
            edge_type=EdgeType.CONTROL_FLOW,
            label=trans.get("label", trans.get("condition", "")),
            data={
                "stage": PipelineStage.ACTIONS.value,
                "condition": trans.get("condition", ""),
                "priority": trans.get("priority", 0),
            },
            animated=True,
        )

    return canvas


# =============================================================================
# Stage 4: Orchestration Canvas (from execution plan)
# =============================================================================


def execution_to_orchestration_canvas(
    execution_plan: dict[str, Any],
    canvas_id: str | None = None,
    canvas_name: str = "Orchestration Plan",
) -> Canvas:
    """Convert an execution plan into a Stage 4 Orchestration canvas.

    Args:
        execution_plan: Dict with keys:
            - agents: [{id, name, type, capabilities}]
            - tasks: [{id, name, assigned_agent, depends_on, type}]
        canvas_id: Optional canvas ID
        canvas_name: Canvas name

    Returns:
        Canvas with Stage 4 orchestration nodes and execution edges
    """
    canvas = Canvas(
        id=canvas_id or f"orch-{uuid.uuid4().hex[:8]}",
        name=canvas_name,
        metadata={"stage": PipelineStage.ORCHESTRATION.value},
    )

    agents = execution_plan.get("agents", [])
    tasks = execution_plan.get("tasks", [])

    # Layout: agents as a row at top, tasks below grouped by agent
    for i, agent in enumerate(agents):
        canvas.nodes[agent["id"]] = CanvasNode(
            id=agent["id"],
            node_type=CanvasNodeType.AGENT,
            position=Position(x=i * 300, y=0),
            size=Size(200, 60),
            label=agent.get("name", agent["id"]),
            data={
                "stage": PipelineStage.ORCHESTRATION.value,
                "orch_type": "agent",
                "agent_type": agent.get("type", "unknown"),
                "capabilities": agent.get("capabilities", []),
                "rf_type": "orchestrationNode",
            },
            style={
                "backgroundColor": NODE_TYPE_COLORS.get("agent_task", "#f472b6"),
                "borderRadius": "50%",
            },
        )

    # Build agent column lookup for layout
    agent_col: dict[str, int] = {}
    for i, agent in enumerate(agents):
        agent_col[agent["id"]] = i

    for j, task in enumerate(tasks):
        assigned = task.get("assigned_agent", "")
        col = agent_col.get(assigned, j % max(len(agents), 1))
        task_type = task.get("type", "agent_task")
        color = NODE_TYPE_COLORS.get(task_type, "#f472b6")

        canvas.nodes[task["id"]] = CanvasNode(
            id=task["id"],
            node_type=CanvasNodeType.WORKFLOW,
            position=Position(x=col * 300, y=100 + (j // max(len(agents), 1)) * 120),
            size=Size(240, 80),
            label=task.get("name", f"Task {j + 1}"),
            data={
                "stage": PipelineStage.ORCHESTRATION.value,
                "orch_type": task_type,
                "assigned_agent": assigned,
                "content_hash": content_hash(task.get("name", "")),
                "rf_type": "orchestrationNode",
            },
            style={"backgroundColor": color, "borderRadius": "6px"},
        )

        # Edge from agent to task (assignment)
        if assigned and assigned in agent_col:
            canvas.edges[f"assign-{assigned}-{task['id']}"] = CanvasEdge(
                id=f"assign-{assigned}-{task['id']}",
                source_id=assigned,
                target_id=task["id"],
                edge_type=EdgeType.CONTROL_FLOW,
                label="executes",
                style={"strokeDasharray": "3 3"},
            )

        # Dependency edges between tasks
        for dep_id in task.get("depends_on", []):
            canvas.edges[f"dep-{dep_id}-{task['id']}"] = CanvasEdge(
                id=f"dep-{dep_id}-{task['id']}",
                source_id=dep_id,
                target_id=task["id"],
                edge_type=EdgeType.DEPENDENCY,
                label="blocks",
                animated=True,
            )

    return canvas


# =============================================================================
# Layout Helpers
# =============================================================================


def _radial_layout(count: int, center_x: float = 400, center_y: float = 300,
                   radius: float = 250) -> list[Position]:
    """Arrange nodes in a radial layout."""
    if count == 0:
        return []
    if count == 1:
        return [Position(center_x, center_y)]

    positions = []
    for i in range(count):
        angle = (2 * math.pi * i) / count - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions.append(Position(round(x, 1), round(y, 1)))
    return positions


def _hierarchical_layout(
    items: list[dict[str, Any]],
    x_spacing: float = 280,
    y_spacing: float = 150,
) -> dict[str, Position]:
    """Arrange items in a hierarchical layout based on dependencies.

    Items with no dependencies go at the top. Items depending on others
    are placed below their dependencies.
    """
    positions: dict[str, Position] = {}
    dep_map: dict[str, list[str]] = {}
    for item in items:
        item_id = item.get("id", "")
        dep_map[item_id] = item.get("dependencies", [])

    # Compute depth (longest dependency chain)
    depth_cache: dict[str, int] = {}

    def get_depth(item_id: str) -> int:
        if item_id in depth_cache:
            return depth_cache[item_id]
        deps = dep_map.get(item_id, [])
        if not deps:
            depth_cache[item_id] = 0
            return 0
        d = 1 + max(get_depth(dep) for dep in deps if dep in dep_map)
        depth_cache[item_id] = d
        return d

    for item in items:
        get_depth(item.get("id", ""))

    # Group by depth level, assign positions
    levels: dict[int, list[str]] = {}
    for item_id, depth in depth_cache.items():
        levels.setdefault(depth, []).append(item_id)

    for depth, ids in sorted(levels.items()):
        for col, item_id in enumerate(ids):
            positions[item_id] = Position(
                x=round(col * x_spacing, 1),
                y=round(depth * y_spacing, 1),
            )

    return positions
