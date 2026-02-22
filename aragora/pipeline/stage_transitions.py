"""
Stage transition functions for the Universal Pipeline.

Promotes nodes from one pipeline stage to the next, generating
provenance links and cross-stage edges along the way.

Each function reads source nodes, creates target-stage nodes with
parent_ids provenance, and records a StageTransition on the graph.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from aragora.canvas.stages import (
    PipelineStage,
    ProvenanceLink,
    StageEdgeType,
    StageTransition,
    content_hash,
)
from aragora.pipeline.adapters import from_goal_node
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)

logger = logging.getLogger(__name__)


def promote_node(
    graph: UniversalGraph,
    node_id: str,
    target_stage: PipelineStage,
    new_subtype: str,
    new_label: str = "",
) -> UniversalNode:
    """Generic single-node promotion with provenance chain.

    Creates a new node in target_stage derived from node_id.
    """
    source = graph.nodes.get(node_id)
    if source is None:
        raise ValueError(f"Source node {node_id} not found in graph")

    label = new_label or source.label
    new_id = f"{target_stage.value}-{uuid.uuid4().hex[:8]}"

    new_node = UniversalNode(
        id=new_id,
        stage=target_stage,
        node_subtype=new_subtype,
        label=label,
        description=source.description,
        content_hash=content_hash(label + source.description),
        previous_hash=source.content_hash,
        parent_ids=[source.id],
        source_stage=source.stage,
        confidence=source.confidence,
        data=dict(source.data),
        metadata={"promoted_from": source.id},
    )
    graph.add_node(new_node)

    # Cross-stage edge
    edge_type = _promotion_edge_type(source.stage, target_stage)
    edge = UniversalEdge(
        id=f"edge-{uuid.uuid4().hex[:8]}",
        source_id=source.id,
        target_id=new_node.id,
        edge_type=edge_type,
        label=edge_type.value,
    )
    graph.add_edge(edge)

    return new_node


def ideas_to_goals(
    graph: UniversalGraph,
    idea_node_ids: list[str],
    extractor: Any | None = None,
) -> list[UniversalNode]:
    """Promote idea nodes to goal nodes.

    If a GoalExtractor is provided, uses it for AI-assisted synthesis.
    Otherwise, does a 1:1 structural promotion.
    """
    if extractor is not None:
        return _ideas_to_goals_with_extractor(graph, idea_node_ids, extractor)

    created: list[UniversalNode] = []
    provenance_links: list[ProvenanceLink] = []

    for idea_id in idea_node_ids:
        source = graph.nodes.get(idea_id)
        if source is None or source.stage != PipelineStage.IDEAS:
            continue

        goal_subtype = _idea_to_goal_subtype(source.node_subtype)
        goal_label = f"Achieve: {source.label}" if not source.label.startswith("Achieve") else source.label

        goal_node = UniversalNode(
            id=f"goal-{uuid.uuid4().hex[:8]}",
            stage=PipelineStage.GOALS,
            node_subtype=goal_subtype,
            label=goal_label,
            description=source.description,
            content_hash=content_hash(goal_label + source.description),
            previous_hash=source.content_hash,
            parent_ids=[source.id],
            source_stage=PipelineStage.IDEAS,
            confidence=source.confidence,
            data=dict(source.data),
            metadata={"promoted_from": source.id},
        )
        graph.add_node(goal_node)
        created.append(goal_node)

        # Cross-stage edge
        edge = UniversalEdge(
            id=f"edge-{uuid.uuid4().hex[:8]}",
            source_id=source.id,
            target_id=goal_node.id,
            edge_type=StageEdgeType.DERIVED_FROM,
            label="derived_from",
        )
        graph.add_edge(edge)

        provenance_links.append(ProvenanceLink(
            source_node_id=source.id,
            source_stage=PipelineStage.IDEAS,
            target_node_id=goal_node.id,
            target_stage=PipelineStage.GOALS,
            content_hash=source.content_hash,
            method="structural_promotion",
        ))

    # Record transition
    if created:
        transition = StageTransition(
            id=f"trans-ideas-goals-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(n.confidence for n in created) / len(created) if created else 0,
            ai_rationale=f"Promoted {len(created)} idea nodes to goals",
        )
        graph.transitions.append(transition)

    return created


def _ideas_to_goals_with_extractor(
    graph: UniversalGraph,
    idea_node_ids: list[str],
    extractor: Any,
) -> list[UniversalNode]:
    """Use GoalExtractor for richer idea→goal synthesis."""
    # Build a minimal canvas dict for the extractor
    nodes_data = []
    for nid in idea_node_ids:
        node = graph.nodes.get(nid)
        if node and node.stage == PipelineStage.IDEAS:
            nodes_data.append({
                "id": node.id,
                "label": node.label,
                "data": {
                    "idea_type": node.node_subtype,
                    "full_content": node.description or node.label,
                    **node.data,
                },
            })

    if not nodes_data:
        return []

    canvas_data = {"nodes": nodes_data, "edges": []}
    goal_graph = extractor.extract_from_ideas(canvas_data)

    created: list[UniversalNode] = []
    for goal in goal_graph.goals:
        unode = from_goal_node(goal)
        graph.add_node(unode)
        created.append(unode)

        # Cross-stage edges
        for src_id in goal.source_idea_ids:
            if src_id in graph.nodes:
                edge = UniversalEdge(
                    id=f"edge-{uuid.uuid4().hex[:8]}",
                    source_id=src_id,
                    target_id=unode.id,
                    edge_type=StageEdgeType.DERIVED_FROM,
                    label="derived_from",
                )
                graph.add_edge(edge)

    if goal_graph.transition:
        graph.transitions.append(goal_graph.transition)

    return created


def goals_to_actions(
    graph: UniversalGraph,
    goal_node_ids: list[str],
    meta_planner: Any | None = None,
) -> list[UniversalNode]:
    """Derive action/task nodes from goal nodes.

    If a MetaPlanner is provided, uses it to prioritize and enrich the
    decomposition with debate-driven rationale.  Falls back to structural
    decomposition when MetaPlanner is None or raises.
    """
    if meta_planner is not None:
        try:
            return _goals_to_actions_with_planner(graph, goal_node_ids, meta_planner)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.warning(
                "MetaPlanner enrichment failed, falling back to structural decomposition: %s",
                exc,
                exc_info=True,
            )

    return _goals_to_actions_structural(graph, goal_node_ids)


def _goals_to_actions_with_planner(
    graph: UniversalGraph,
    goal_node_ids: list[str],
    meta_planner: Any,
) -> list[UniversalNode]:
    """Use MetaPlanner for prioritized goal→action decomposition."""
    # Collect goal descriptions for the planner
    goal_descriptions: list[str] = []
    valid_ids: list[str] = []
    for goal_id in goal_node_ids:
        source = graph.nodes.get(goal_id)
        if source is not None and source.stage == PipelineStage.GOALS:
            goal_descriptions.append(source.description or source.label)
            valid_ids.append(goal_id)

    if not valid_ids:
        return []

    objective = "; ".join(goal_descriptions)

    # prioritize_work is async — run synchronously
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            prioritized = pool.submit(
                asyncio.run, meta_planner.prioritize_work(objective=objective)
            ).result()
    else:
        prioritized = asyncio.run(meta_planner.prioritize_work(objective=objective))

    # Build a priority map from MetaPlanner results (keyed by description similarity)
    priority_map: dict[str, Any] = {}
    for pg in prioritized:
        priority_map[pg.description.lower()] = pg

    created: list[UniversalNode] = []
    provenance_links: list[ProvenanceLink] = []

    for goal_id in valid_ids:
        source = graph.nodes[goal_id]
        action_subtype = _goal_to_action_subtype(source.node_subtype)
        action_label = source.label.replace("Achieve: ", "").replace("Maintain: ", "")

        # Try to match this goal to a MetaPlanner result for enrichment
        matched_pg = _match_prioritized_goal(source, priority_map)

        if matched_pg is not None:
            priority_val = matched_pg.estimated_impact
            rationale = matched_pg.rationale
            planner_priority = matched_pg.priority
        else:
            priority_val = source.data.get("priority", "medium")
            rationale = ""
            planner_priority = None

        metadata: dict[str, Any] = {"promoted_from": source.id}
        if rationale:
            metadata["meta_planner_rationale"] = rationale
        if planner_priority is not None:
            metadata["meta_planner_priority"] = planner_priority

        action_node = UniversalNode(
            id=f"action-{uuid.uuid4().hex[:8]}",
            stage=PipelineStage.ACTIONS,
            node_subtype=action_subtype,
            label=action_label,
            description=source.description,
            content_hash=content_hash(action_label + source.description),
            previous_hash=source.content_hash,
            parent_ids=[source.id],
            source_stage=PipelineStage.GOALS,
            confidence=source.confidence * 0.9,
            data={
                "priority": priority_val,
                "source_goal_id": source.id,
            },
            metadata=metadata,
        )
        graph.add_node(action_node)
        created.append(action_node)

        edge = UniversalEdge(
            id=f"edge-{uuid.uuid4().hex[:8]}",
            source_id=source.id,
            target_id=action_node.id,
            edge_type=StageEdgeType.IMPLEMENTS,
            label="implements",
        )
        graph.add_edge(edge)

        provenance_links.append(ProvenanceLink(
            source_node_id=source.id,
            source_stage=PipelineStage.GOALS,
            target_node_id=action_node.id,
            target_stage=PipelineStage.ACTIONS,
            content_hash=source.content_hash,
            method="meta_planner_decomposition",
        ))

    # Sort created nodes by MetaPlanner priority when available
    created.sort(
        key=lambda n: n.metadata.get("meta_planner_priority", 999),
    )

    if created:
        transition = StageTransition(
            id=f"trans-goals-actions-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.GOALS,
            to_stage=PipelineStage.ACTIONS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(n.confidence for n in created) / len(created),
            ai_rationale=(
                f"MetaPlanner-prioritized decomposition of {len(created)} goals"
            ),
        )
        graph.transitions.append(transition)

    return created


def _match_prioritized_goal(
    source: UniversalNode,
    priority_map: dict[str, Any],
) -> Any | None:
    """Best-effort match a goal node to a MetaPlanner PrioritizedGoal."""
    desc_lower = (source.description or source.label).lower()
    # Exact description match
    if desc_lower in priority_map:
        return priority_map[desc_lower]
    # Substring containment
    for key, pg in priority_map.items():
        if key in desc_lower or desc_lower in key:
            return pg
    return None


def _goals_to_actions_structural(
    graph: UniversalGraph,
    goal_node_ids: list[str],
) -> list[UniversalNode]:
    """Structural goal→action decomposition (original logic)."""
    created: list[UniversalNode] = []
    provenance_links: list[ProvenanceLink] = []

    for goal_id in goal_node_ids:
        source = graph.nodes.get(goal_id)
        if source is None or source.stage != PipelineStage.GOALS:
            continue

        action_subtype = _goal_to_action_subtype(source.node_subtype)
        action_label = source.label.replace("Achieve: ", "").replace("Maintain: ", "")

        action_node = UniversalNode(
            id=f"action-{uuid.uuid4().hex[:8]}",
            stage=PipelineStage.ACTIONS,
            node_subtype=action_subtype,
            label=action_label,
            description=source.description,
            content_hash=content_hash(action_label + source.description),
            previous_hash=source.content_hash,
            parent_ids=[source.id],
            source_stage=PipelineStage.GOALS,
            confidence=source.confidence * 0.9,
            data={
                "priority": source.data.get("priority", "medium"),
                "source_goal_id": source.id,
            },
            metadata={"promoted_from": source.id},
        )
        graph.add_node(action_node)
        created.append(action_node)

        edge = UniversalEdge(
            id=f"edge-{uuid.uuid4().hex[:8]}",
            source_id=source.id,
            target_id=action_node.id,
            edge_type=StageEdgeType.IMPLEMENTS,
            label="implements",
        )
        graph.add_edge(edge)

        provenance_links.append(ProvenanceLink(
            source_node_id=source.id,
            source_stage=PipelineStage.GOALS,
            target_node_id=action_node.id,
            target_stage=PipelineStage.ACTIONS,
            content_hash=source.content_hash,
            method="goal_decomposition",
        ))

    if created:
        transition = StageTransition(
            id=f"trans-goals-actions-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.GOALS,
            to_stage=PipelineStage.ACTIONS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(n.confidence for n in created) / len(created) if created else 0,
            ai_rationale=f"Decomposed {len(created)} goals into action tasks",
        )
        graph.transitions.append(transition)

    return created


def actions_to_orchestration(
    graph: UniversalGraph,
    action_node_ids: list[str],
) -> list[UniversalNode]:
    """Create orchestration nodes from action items."""
    created: list[UniversalNode] = []
    provenance_links: list[ProvenanceLink] = []

    for action_id in action_node_ids:
        source = graph.nodes.get(action_id)
        if source is None or source.stage != PipelineStage.ACTIONS:
            continue

        orch_subtype = _action_to_orch_subtype(source.node_subtype)
        agent_type = _assign_agent_type(source)

        orch_node = UniversalNode(
            id=f"orch-{uuid.uuid4().hex[:8]}",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype=orch_subtype,
            label=source.label,
            description=source.description,
            content_hash=content_hash(source.label + source.description),
            previous_hash=source.content_hash,
            parent_ids=[source.id],
            source_stage=PipelineStage.ACTIONS,
            confidence=source.confidence * 0.85,
            data={
                "agent_type": agent_type,
                "source_action_id": source.id,
                "priority": source.data.get("priority", "medium"),
            },
            metadata={"promoted_from": source.id},
        )
        graph.add_node(orch_node)
        created.append(orch_node)

        edge = UniversalEdge(
            id=f"edge-{uuid.uuid4().hex[:8]}",
            source_id=source.id,
            target_id=orch_node.id,
            edge_type=StageEdgeType.EXECUTES,
            label="executes",
        )
        graph.add_edge(edge)

        provenance_links.append(ProvenanceLink(
            source_node_id=source.id,
            source_stage=PipelineStage.ACTIONS,
            target_node_id=orch_node.id,
            target_stage=PipelineStage.ORCHESTRATION,
            content_hash=source.content_hash,
            method="agent_assignment",
        ))

    if created:
        transition = StageTransition(
            id=f"trans-actions-orch-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.ACTIONS,
            to_stage=PipelineStage.ORCHESTRATION,
            provenance=provenance_links,
            status="pending",
            confidence=sum(n.confidence for n in created) / len(created) if created else 0,
            ai_rationale=(
                f"Assigned {len(created)} action tasks to orchestration agents"
            ),
        )
        graph.transitions.append(transition)

    return created


# ── Mapping helpers ─────────────────────────────────────────────────────

def _promotion_edge_type(
    from_stage: PipelineStage, to_stage: PipelineStage
) -> StageEdgeType:
    if from_stage == PipelineStage.IDEAS and to_stage == PipelineStage.GOALS:
        return StageEdgeType.DERIVED_FROM
    if from_stage == PipelineStage.GOALS and to_stage == PipelineStage.ACTIONS:
        return StageEdgeType.IMPLEMENTS
    if from_stage == PipelineStage.ACTIONS and to_stage == PipelineStage.ORCHESTRATION:
        return StageEdgeType.EXECUTES
    return StageEdgeType.DERIVED_FROM


def _idea_to_goal_subtype(idea_subtype: str) -> str:
    return {
        "concept": "goal",
        "cluster": "goal",
        "question": "milestone",
        "insight": "strategy",
        "evidence": "metric",
        "assumption": "risk",
        "constraint": "principle",
        "observation": "metric",
        "hypothesis": "strategy",
    }.get(idea_subtype, "goal")


def _goal_to_action_subtype(goal_subtype: str) -> str:
    return {
        "goal": "task",
        "principle": "checkpoint",
        "strategy": "epic",
        "milestone": "checkpoint",
        "metric": "deliverable",
        "risk": "checkpoint",
    }.get(goal_subtype, "task")


def _action_to_orch_subtype(action_subtype: str) -> str:
    return {
        "task": "agent_task",
        "epic": "parallel_fan",
        "checkpoint": "human_gate",
        "deliverable": "verification",
        "dependency": "merge",
    }.get(action_subtype, "agent_task")


def _assign_agent_type(action_node: UniversalNode) -> str:
    """Pick an agent archetype based on action content."""
    subtype = action_node.node_subtype
    if subtype == "checkpoint":
        return "reviewer"
    if subtype == "deliverable":
        return "verifier"
    label_lower = action_node.label.lower()
    if any(w in label_lower for w in ("implement", "build", "code", "create")):
        return "implementer"
    if any(w in label_lower for w in ("review", "verify", "test", "check")):
        return "reviewer"
    return "analyst"


def suggest_transitions(
    graph: UniversalGraph,
    stage: PipelineStage,
) -> list[dict[str, Any]]:
    """Suggest candidate transitions from the given stage to the next.

    Analyzes nodes at *stage* and returns a list of suggestions with
    confidence scores indicating how ready each node is for promotion.

    Returns a list of dicts::

        {
            "node_id": str,
            "node_label": str,
            "from_stage": str,        # e.g. "ideas"
            "to_stage": str,           # e.g. "goals"
            "confidence": float,       # 0.0 – 1.0
            "reason": str,
        }
    """
    next_stage = _next_stage(stage)
    if next_stage is None:
        return []

    nodes = graph.get_stage(stage)
    if not nodes:
        return []

    # Check which nodes already have children in the next stage
    promoted_ids: set[str] = set()
    for node in graph.get_stage(next_stage):
        promoted_ids.update(node.parent_ids)

    suggestions: list[dict[str, Any]] = []
    for node in nodes:
        # Skip nodes that were already promoted
        if node.id in promoted_ids:
            continue

        confidence = _transition_confidence(node, stage, graph)
        if confidence < 0.1:
            continue

        suggestions.append({
            "node_id": node.id,
            "node_label": node.label,
            "from_stage": stage.value,
            "to_stage": next_stage.value,
            "confidence": round(confidence, 2),
            "reason": _transition_reason(node, stage, confidence),
        })

    suggestions.sort(key=lambda s: s["confidence"], reverse=True)
    return suggestions


def _next_stage(stage: PipelineStage) -> PipelineStage | None:
    """Return the next stage in the pipeline, or None for orchestration."""
    order = [
        PipelineStage.IDEAS,
        PipelineStage.GOALS,
        PipelineStage.ACTIONS,
        PipelineStage.ORCHESTRATION,
    ]
    try:
        idx = order.index(stage)
    except ValueError:
        return None
    return order[idx + 1] if idx + 1 < len(order) else None


def _transition_confidence(
    node: UniversalNode,
    stage: PipelineStage,
    graph: UniversalGraph,
) -> float:
    """Heuristic confidence score for promoting a node."""
    score = 0.3  # Base score for any active node

    # Nodes with descriptions are more ready
    if node.description and len(node.description) > 10:
        score += 0.2

    # Higher-confidence nodes are more ready
    if node.confidence > 0.5:
        score += 0.15
    elif node.confidence > 0.3:
        score += 0.1

    # Nodes with inbound edges (i.e. connected to other nodes) are more mature
    inbound = sum(
        1 for e in graph.edges.values() if e.target_id == node.id
    )
    if inbound > 0:
        score += min(0.15, inbound * 0.05)

    # Active nodes are promotable; archived/rejected are not
    if node.status in ("archived", "rejected"):
        return 0.0
    if node.status == "completed":
        score += 0.2

    return min(1.0, score)


def _transition_reason(
    node: UniversalNode,
    stage: PipelineStage,
    confidence: float,
) -> str:
    """Generate a human-readable reason for the transition suggestion."""
    next_s = _next_stage(stage)
    next_label = next_s.value.title() if next_s else "next stage"

    if confidence >= 0.7:
        return f"Strong candidate for {next_label} — well-defined with supporting context"
    if confidence >= 0.5:
        return f"Ready for promotion to {next_label}"
    return f"May benefit from further refinement before promoting to {next_label}"


__all__ = [
    "promote_node",
    "ideas_to_goals",
    "goals_to_actions",
    "actions_to_orchestration",
    "suggest_transitions",
]
