"""
Universal Node Schema for the Idea-to-Execution Pipeline.

Provides a single node/edge/graph model that spans all four pipeline stages
(Ideas, Goals, Actions, Orchestration), enabling cross-stage provenance
queries, unified persistence, and a single React Flow canvas.

Existing stage-specific models (CanvasNode, GoalNode, etc.) remain canonical
for their domains — UniversalNode is a projection layer that unifies them
for DAG operations and persistence.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aragora.canvas.stages import (
    ActionNodeType,
    GoalNodeType,
    IdeaNodeType,
    NODE_TYPE_COLORS,
    OrchestrationNodeType,
    PipelineStage,
    PrincipleNodeType,
    ProvenanceLink,
    STAGE_COLORS,
    StageEdgeType,
    StageTransition,
    content_hash,
)

# Valid subtypes per stage
_VALID_SUBTYPES: dict[PipelineStage, set[str]] = {
    PipelineStage.IDEAS: {e.value for e in IdeaNodeType},
    PipelineStage.PRINCIPLES: {e.value for e in PrincipleNodeType},
    PipelineStage.GOALS: {e.value for e in GoalNodeType},
    PipelineStage.ACTIONS: {e.value for e in ActionNodeType},
    PipelineStage.ORCHESTRATION: {e.value for e in OrchestrationNodeType},
}


@dataclass
class UniversalNode:
    """A node that can represent any pipeline stage's node type."""

    id: str
    stage: PipelineStage
    node_subtype: str  # e.g. "concept", "goal", "task", "agent_task"
    label: str
    description: str = ""
    # Position/visual
    position_x: float = 0.0
    position_y: float = 0.0
    width: float = 200.0
    height: float = 100.0
    # Provenance
    content_hash: str = ""
    previous_hash: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    source_stage: PipelineStage | None = None
    # Status
    status: str = "active"  # active, completed, archived, rejected
    execution_status: str | None = None  # pending, in_progress, succeeded, failed, partial
    confidence: float = 0.0
    # Data
    data: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.label + self.description)

    def validate_subtype(self) -> bool:
        """Check that node_subtype is valid for the current stage."""
        valid = _VALID_SUBTYPES.get(self.stage, set())
        return self.node_subtype in valid

    def to_react_flow_node(self) -> dict[str, Any]:
        """Export as a React Flow node dict."""
        color = NODE_TYPE_COLORS.get(self.node_subtype, "#94a3b8")
        stage_color = STAGE_COLORS.get(self.stage, {})
        return {
            "id": self.id,
            "type": f"{self.stage.value}Node",
            "position": {"x": self.position_x, "y": self.position_y},
            "data": {
                "label": self.label,
                "description": self.description,
                "stage": self.stage.value,
                "subtype": self.node_subtype,
                "status": self.status,
                "confidence": self.confidence,
                "color": color,
                "stageColor": stage_color.get("primary", color),
                **self.data,
            },
            "style": {
                "width": self.width,
                "height": self.height,
                "backgroundColor": color + "20",
                "borderColor": color,
                **self.style,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage.value,
            "node_subtype": self.node_subtype,
            "label": self.label,
            "description": self.description,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "width": self.width,
            "height": self.height,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
            "parent_ids": self.parent_ids,
            "source_stage": self.source_stage.value if self.source_stage else None,
            "status": self.status,
            "confidence": self.confidence,
            "data": self.data,
            "style": self.style,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalNode:
        source_stage = data.get("source_stage")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            stage=PipelineStage(data["stage"]),
            node_subtype=data["node_subtype"],
            label=data.get("label", ""),
            description=data.get("description", ""),
            position_x=float(data.get("position_x", 0)),
            position_y=float(data.get("position_y", 0)),
            width=float(data.get("width", 200)),
            height=float(data.get("height", 100)),
            content_hash=data.get("content_hash", ""),
            previous_hash=data.get("previous_hash"),
            parent_ids=data.get("parent_ids", []),
            source_stage=PipelineStage(source_stage) if source_stage else None,
            status=data.get("status", "active"),
            confidence=float(data.get("confidence", 0)),
            data=data.get("data", {}),
            style=data.get("style", {}),
            metadata=data.get("metadata", {}),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


@dataclass
class UniversalEdge:
    """An edge connecting two UniversalNodes, possibly across stages."""

    id: str
    source_id: str
    target_id: str
    edge_type: StageEdgeType
    label: str = ""
    weight: float = 1.0
    cross_stage: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_react_flow_edge(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "type": "smoothstep" if self.cross_stage else "default",
            "label": self.label or self.edge_type.value,
            "animated": self.cross_stage,
            "data": {
                "edgeType": self.edge_type.value,
                "weight": self.weight,
                "crossStage": self.cross_stage,
                **self.data,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "label": self.label,
            "weight": self.weight,
            "cross_stage": self.cross_stage,
            "data": self.data,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalEdge:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=StageEdgeType(data["edge_type"]),
            label=data.get("label", ""),
            weight=float(data.get("weight", 1.0)),
            cross_stage=bool(data.get("cross_stage", False)),
            data=data.get("data", {}),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class UniversalGraph:
    """Container for a full pipeline graph spanning all four stages."""

    id: str
    name: str = "Untitled Pipeline"
    nodes: dict[str, UniversalNode] = field(default_factory=dict)
    edges: dict[str, UniversalEdge] = field(default_factory=dict)
    transitions: list[StageTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None
    workspace_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- Mutation helpers ---------------------------------------------------

    def add_node(self, node: UniversalNode) -> None:
        self.nodes[node.id] = node
        self.updated_at = time.time()

    def add_edge(self, edge: UniversalEdge) -> None:
        # Validate that source and target exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return
        # Detect cross-stage edges
        src_stage = self.nodes[edge.source_id].stage
        tgt_stage = self.nodes[edge.target_id].stage
        edge.cross_stage = src_stage != tgt_stage
        self.edges[edge.id] = edge
        self.updated_at = time.time()

    def remove_node(self, node_id: str) -> UniversalNode | None:
        node = self.nodes.pop(node_id, None)
        if node:
            edges_to_remove = [
                eid
                for eid, e in self.edges.items()
                if e.source_id == node_id or e.target_id == node_id
            ]
            for eid in edges_to_remove:
                self.edges.pop(eid, None)
            self.updated_at = time.time()
        return node

    def remove_edge(self, edge_id: str) -> UniversalEdge | None:
        edge = self.edges.pop(edge_id, None)
        if edge:
            self.updated_at = time.time()
        return edge

    # -- Query helpers ------------------------------------------------------

    def get_stage(self, stage: PipelineStage) -> list[UniversalNode]:
        return [n for n in self.nodes.values() if n.stage == stage]

    def get_cross_stage_edges(self) -> list[UniversalEdge]:
        return [e for e in self.edges.values() if e.cross_stage]

    def get_provenance_chain(self, node_id: str) -> list[UniversalNode]:
        """Walk parent_ids recursively to build a provenance chain."""
        visited: set[str] = set()
        chain: list[UniversalNode] = []
        self._walk_provenance(node_id, visited, chain)
        return chain

    def _walk_provenance(self, node_id: str, visited: set[str], chain: list[UniversalNode]) -> None:
        if node_id in visited or node_id not in self.nodes:
            return
        visited.add(node_id)
        node = self.nodes[node_id]
        chain.append(node)
        for parent_id in node.parent_ids:
            self._walk_provenance(parent_id, visited, chain)

    def integrity_hash(self) -> str:
        """Merkle-like hash of all node content_hash values."""
        hashes = sorted(n.content_hash for n in self.nodes.values())
        combined = ":".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # -- Export helpers -----------------------------------------------------

    def to_react_flow(self, stage_filter: PipelineStage | None = None) -> dict[str, Any]:
        """Export as React Flow JSON, optionally filtered to one stage."""
        nodes = self.nodes.values()
        if stage_filter is not None:
            nodes = [n for n in nodes if n.stage == stage_filter]
        else:
            nodes = list(nodes)

        node_ids = {n.id for n in nodes}
        edges = [
            e for e in self.edges.values() if e.source_id in node_ids and e.target_id in node_ids
        ]

        return {
            "nodes": [n.to_react_flow_node() for n in nodes],
            "edges": [e.to_react_flow_edge() for e in edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "transitions": [t.to_dict() for t in self.transitions],
            "metadata": self.metadata,
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalGraph:
        graph = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Pipeline"),
            metadata=data.get("metadata", {}),
            owner_id=data.get("owner_id"),
            workspace_id=data.get("workspace_id"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
        for nd in data.get("nodes", []):
            node = UniversalNode.from_dict(nd)
            graph.nodes[node.id] = node
        for ed in data.get("edges", []):
            edge = UniversalEdge.from_dict(ed)
            graph.edges[edge.id] = edge
        for td in data.get("transitions", []):
            graph.transitions.append(
                StageTransition(
                    id=td["id"],
                    from_stage=PipelineStage(td["from_stage"]),
                    to_stage=PipelineStage(td["to_stage"]),
                    provenance=[
                        ProvenanceLink(
                            source_node_id=p["source_node_id"],
                            source_stage=PipelineStage(p["source_stage"]),
                            target_node_id=p["target_node_id"],
                            target_stage=PipelineStage(p["target_stage"]),
                            content_hash=p["content_hash"],
                            timestamp=p.get("timestamp", 0),
                            method=p.get("method", ""),
                        )
                        for p in td.get("provenance", [])
                    ],
                    status=td.get("status", "pending"),
                    confidence=td.get("confidence", 0),
                    ai_rationale=td.get("ai_rationale", ""),
                    human_notes=td.get("human_notes", ""),
                    created_at=td.get("created_at", 0),
                    reviewed_at=td.get("reviewed_at"),
                )
            )
        return graph


__all__ = [
    "UniversalNode",
    "UniversalEdge",
    "UniversalGraph",
]
