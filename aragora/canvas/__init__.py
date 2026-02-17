"""
Live Canvas System.

Interactive visual workspace for:
- Live debate visualization
- Drag-and-drop workflow building
- Real-time agent interaction
- Collaborative knowledge mapping

Usage:
    from aragora.canvas import Canvas, CanvasNode, CanvasEdge

    canvas = Canvas(id="my-canvas")
    node = canvas.add_node(CanvasNodeType.AGENT, position=Position(100, 100))
    edge = canvas.add_edge(node.id, other_node.id)
"""

from .models import (
    Canvas,
    CanvasEdge,
    CanvasEvent,
    CanvasEventType,
    CanvasNode,
    CanvasNodeType,
    EdgeType,
    Position,
    Size,
)
from .renderer import CanvasRenderer
from .manager import CanvasStateManager, get_canvas_manager
from .stages import PipelineStage, ProvenanceLink, StageTransition
from .converters import (
    debate_to_ideas_canvas,
    goals_to_canvas,
    workflow_to_actions_canvas,
    execution_to_orchestration_canvas,
    to_react_flow,
)
from .primitives import (
    AlertPrimitive,
    AlertSeverity,
    ButtonPrimitive,
    ButtonVariant,
    CardPrimitive,
    ChartDataPoint,
    ChartPrimitive,
    ChartSeries,
    ChartType,
    FormField,
    FormPrimitive,
    ProgressPrimitive,
    SelectOption,
    SelectPrimitive,
    TableColumn,
    TablePrimitive,
)

__all__ = [
    # Core canvas
    "Canvas",
    "CanvasNode",
    "CanvasEdge",
    "CanvasNodeType",
    "CanvasEventType",
    "CanvasEvent",
    "EdgeType",
    "Position",
    "Size",
    "CanvasRenderer",
    "CanvasStateManager",
    "get_canvas_manager",
    # A2UI Primitives
    "AlertPrimitive",
    "AlertSeverity",
    "ButtonPrimitive",
    "ButtonVariant",
    "CardPrimitive",
    "ChartDataPoint",
    "ChartPrimitive",
    "ChartSeries",
    "ChartType",
    "FormField",
    "FormPrimitive",
    "ProgressPrimitive",
    "SelectOption",
    "SelectPrimitive",
    "TableColumn",
    "TablePrimitive",
    # Pipeline stages
    "PipelineStage",
    "ProvenanceLink",
    "StageTransition",
    # Converters
    "debate_to_ideas_canvas",
    "goals_to_canvas",
    "workflow_to_actions_canvas",
    "execution_to_orchestration_canvas",
    "to_react_flow",
]
