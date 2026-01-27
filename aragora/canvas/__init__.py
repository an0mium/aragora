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
]
