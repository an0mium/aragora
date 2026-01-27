"""
Canvas Data Models.

Defines the data structures for the Live Canvas system:
- Canvas: The main canvas container
- CanvasNode: Nodes on the canvas (agents, debates, knowledge, etc.)
- CanvasEdge: Connections between nodes
- CanvasEvent: Events for real-time updates
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CanvasNodeType(Enum):
    """Types of nodes that can appear on a canvas."""

    AGENT = "agent"  # AI agent
    DEBATE = "debate"  # Active debate
    KNOWLEDGE = "knowledge"  # KM node
    CONNECTOR = "connector"  # External integration
    BROWSER = "browser"  # Browser automation
    WORKFLOW = "workflow"  # Workflow step
    INPUT = "input"  # User input field
    OUTPUT = "output"  # Result display
    TEXT = "text"  # Text annotation
    IMAGE = "image"  # Image display
    GROUP = "group"  # Group container
    DECISION = "decision"  # Decision node
    EVIDENCE = "evidence"  # Evidence item


class CanvasEventType(Enum):
    """Types of canvas events."""

    # Connection events
    CONNECT = "canvas:connect"
    DISCONNECT = "canvas:disconnect"

    # State events
    STATE = "canvas:state"
    SYNC = "canvas:sync"

    # Canvas events
    CANVAS_UPDATE = "canvas:update"

    # Node events
    NODE_CREATE = "canvas:node:create"
    NODE_UPDATE = "canvas:node:update"
    NODE_MOVE = "canvas:node:move"
    NODE_RESIZE = "canvas:node:resize"
    NODE_DELETE = "canvas:node:delete"
    NODE_SELECT = "canvas:node:select"

    # Edge events
    EDGE_CREATE = "canvas:edge:create"
    EDGE_UPDATE = "canvas:edge:update"
    EDGE_DELETE = "canvas:edge:delete"

    # Action events
    ACTION = "canvas:action"
    ACTION_RESULT = "canvas:action:result"

    # Agent events
    AGENT_MESSAGE = "canvas:agent:message"
    AGENT_THINKING = "canvas:agent:thinking"
    AGENT_COMPLETE = "canvas:agent:complete"

    # Debate events
    DEBATE_START = "canvas:debate:start"
    DEBATE_ROUND = "canvas:debate:round"
    DEBATE_VOTE = "canvas:debate:vote"
    DEBATE_CONSENSUS = "canvas:debate:consensus"
    DEBATE_END = "canvas:debate:end"

    # Error events
    ERROR = "canvas:error"


@dataclass
class Position:
    """2D position on the canvas."""

    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        return cls(x=float(data.get("x", 0)), y=float(data.get("y", 0)))


@dataclass
class Size:
    """Dimensions of a node."""

    width: float = 200.0
    height: float = 100.0

    def to_dict(self) -> Dict[str, float]:
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Size:
        return cls(
            width=float(data.get("width", 200)),
            height=float(data.get("height", 100)),
        )


@dataclass
class CanvasNode:
    """
    A node on the canvas.

    Represents an entity like an agent, debate, knowledge item, etc.
    """

    id: str
    node_type: CanvasNodeType
    position: Position = field(default_factory=Position)
    size: Size = field(default_factory=Size)
    label: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    locked: bool = False
    selected: bool = False
    parent_id: Optional[str] = None  # For grouped nodes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "label": self.label,
            "data": self.data,
            "style": self.style,
            "locked": self.locked,
            "selected": self.selected,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasNode:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            node_type=CanvasNodeType(data.get("type", "text")),
            position=Position.from_dict(data.get("position", {})),
            size=Size.from_dict(data.get("size", {})),
            label=data.get("label", ""),
            data=data.get("data", {}),
            style=data.get("style", {}),
            locked=data.get("locked", False),
            selected=data.get("selected", False),
            parent_id=data.get("parent_id"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.now(timezone.utc)
            ),
        )

    def move(self, x: float, y: float) -> None:
        """Move the node to a new position."""
        self.position = Position(x, y)
        self.updated_at = datetime.now(timezone.utc)

    def resize(self, width: float, height: float) -> None:
        """Resize the node."""
        self.size = Size(width, height)
        self.updated_at = datetime.now(timezone.utc)

    def update_data(self, **kwargs: Any) -> None:
        """Update node data."""
        self.data.update(kwargs)
        self.updated_at = datetime.now(timezone.utc)


class EdgeType(Enum):
    """Types of edges between nodes."""

    DEFAULT = "default"
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    REFERENCE = "reference"
    DEPENDENCY = "dependency"
    CRITIQUE = "critique"
    SUPPORT = "support"


@dataclass
class CanvasEdge:
    """
    An edge connecting two nodes.

    Represents a relationship between nodes.
    """

    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DEFAULT
    label: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    animated: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "label": self.label,
            "data": self.data,
            "style": self.style,
            "animated": self.animated,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasEdge:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data.get("source", data.get("source_id", "")),
            target_id=data.get("target", data.get("target_id", "")),
            edge_type=EdgeType(data.get("type", "default")),
            label=data.get("label", ""),
            data=data.get("data", {}),
            style=data.get("style", {}),
            animated=data.get("animated", False),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            ),
        )


@dataclass
class CanvasEvent:
    """
    An event on the canvas.

    Used for real-time updates via WebSocket.
    """

    event_type: CanvasEventType
    canvas_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    node_id: Optional[str] = None
    edge_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "canvas_id": self.canvas_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "user_id": self.user_id,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanvasEvent:
        return cls(
            event_type=CanvasEventType(data.get("type", "canvas:error")),
            canvas_id=data.get("canvas_id", ""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc)
            ),
            data=data.get("data", {}),
            user_id=data.get("user_id"),
            node_id=data.get("node_id"),
            edge_id=data.get("edge_id"),
        )


@dataclass
class Canvas:
    """
    The main canvas container.

    Holds all nodes and edges, and manages canvas state.
    """

    id: str
    name: str = "Untitled Canvas"
    nodes: Dict[str, CanvasNode] = field(default_factory=dict)
    edges: Dict[str, CanvasEdge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_id": self.owner_id,
            "workspace_id": self.workspace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Canvas:
        canvas = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Canvas"),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.now(timezone.utc)
            ),
            owner_id=data.get("owner_id"),
            workspace_id=data.get("workspace_id"),
        )

        # Load nodes
        for node_data in data.get("nodes", []):
            node = CanvasNode.from_dict(node_data)
            canvas.nodes[node.id] = node

        # Load edges
        for edge_data in data.get("edges", []):
            edge = CanvasEdge.from_dict(edge_data)
            canvas.edges[edge.id] = edge

        return canvas

    def add_node(
        self,
        node_type: CanvasNodeType,
        position: Optional[Position] = None,
        label: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CanvasNode:
        """Add a new node to the canvas."""
        node_id = str(uuid.uuid4())
        node = CanvasNode(
            id=node_id,
            node_type=node_type,
            position=position or Position(),
            label=label,
            data=data or {},
            **kwargs,
        )
        self.nodes[node_id] = node
        self.updated_at = datetime.now(timezone.utc)
        return node

    def remove_node(self, node_id: str) -> Optional[CanvasNode]:
        """Remove a node and its connected edges."""
        node = self.nodes.pop(node_id, None)
        if node:
            # Remove connected edges
            edges_to_remove = [
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for edge_id in edges_to_remove:
                self.edges.pop(edge_id, None)
            self.updated_at = datetime.now(timezone.utc)
        return node

    def get_node(self, node_id: str) -> Optional[CanvasNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEFAULT,
        label: str = "",
        **kwargs: Any,
    ) -> Optional[CanvasEdge]:
        """Add an edge between two nodes."""
        # Validate nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge_id = str(uuid.uuid4())
        edge = CanvasEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label,
            **kwargs,
        )
        self.edges[edge_id] = edge
        self.updated_at = datetime.now(timezone.utc)
        return edge

    def remove_edge(self, edge_id: str) -> Optional[CanvasEdge]:
        """Remove an edge."""
        edge = self.edges.pop(edge_id, None)
        if edge:
            self.updated_at = datetime.now(timezone.utc)
        return edge

    def get_edge(self, edge_id: str) -> Optional[CanvasEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)

    def get_connected_nodes(self, node_id: str) -> List[CanvasNode]:
        """Get all nodes connected to a given node."""
        connected_ids = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                connected_ids.add(edge.target_id)
            elif edge.target_id == node_id:
                connected_ids.add(edge.source_id)
        return [self.nodes[nid] for nid in connected_ids if nid in self.nodes]

    def get_edges_for_node(self, node_id: str) -> List[CanvasEdge]:
        """Get all edges connected to a node."""
        return [
            edge
            for edge in self.edges.values()
            if edge.source_id == node_id or edge.target_id == node_id
        ]

    def get_nodes_by_type(self, node_type: CanvasNodeType) -> List[CanvasNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.updated_at = datetime.now(timezone.utc)


__all__ = [
    "CanvasNodeType",
    "CanvasEventType",
    "EdgeType",
    "Position",
    "Size",
    "CanvasNode",
    "CanvasEdge",
    "CanvasEvent",
    "Canvas",
]
