"""
Canvas Renderer.

Provides visual rendering of canvas state for various output formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .models import Canvas


class CanvasRenderer:
    """
    Renders canvas state to various formats.

    Supports:
    - JSON serialization for web clients
    - SVG export for static images
    - Mermaid diagram generation
    """

    def __init__(self, canvas: Optional["Canvas"] = None):
        """Initialize renderer with optional canvas."""
        self._canvas = canvas

    def set_canvas(self, canvas: "Canvas") -> None:
        """Set the canvas to render."""
        self._canvas = canvas

    def to_json(self) -> Dict[str, Any]:
        """Export canvas state as JSON for web clients."""
        if not self._canvas:
            return {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}

        return {
            "id": self._canvas.id,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.node_type.value,
                    "position": {"x": node.position.x, "y": node.position.y},
                    "data": node.data,
                    "label": node.label,
                }
                for node in self._canvas.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "label": edge.label,
                    "data": edge.data,
                }
                for edge in self._canvas.edges.values()
            ],
            "viewport": self._canvas.metadata.get("viewport", {"x": 0, "y": 0, "zoom": 1}),
        }

    def to_svg(self, width: int = 800, height: int = 600) -> str:
        """Export canvas as SVG image."""
        if not self._canvas:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
            )

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            "<style>",
            ".node { fill: #f0f0f0; stroke: #333; stroke-width: 2; }",
            ".edge { stroke: #666; stroke-width: 1; fill: none; }",
            ".label { font-family: sans-serif; font-size: 12px; }",
            "</style>",
        ]

        # Render edges first (so they appear behind nodes)
        for edge in self._canvas.edges.values():
            source = self._canvas.nodes.get(edge.source_id)
            target = self._canvas.nodes.get(edge.target_id)
            if source and target:
                svg_parts.append(
                    f'<line class="edge" '
                    f'x1="{source.position.x}" y1="{source.position.y}" '
                    f'x2="{target.position.x}" y2="{target.position.y}" />'
                )

        # Render nodes
        for node in self._canvas.nodes.values():
            svg_parts.append(
                f'<rect class="node" '
                f'x="{node.position.x - 40}" y="{node.position.y - 20}" '
                f'width="80" height="40" rx="5" />'
            )
            svg_parts.append(
                f'<text class="label" '
                f'x="{node.position.x}" y="{node.position.y + 5}" '
                f'text-anchor="middle">{node.node_type.value}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def to_mermaid(self) -> str:
        """Export canvas as Mermaid diagram."""
        if not self._canvas:
            return "graph LR"

        lines = ["graph LR"]

        # Add nodes
        for node in self._canvas.nodes.values():
            node_label = node.label or node.data.get("label", node.node_type.value)
            lines.append(f"    {node.id}[{node_label}]")

        # Add edges
        for edge in self._canvas.edges.values():
            if edge.label:
                lines.append(f"    {edge.source_id} -->|{edge.label}| {edge.target_id}")
            else:
                lines.append(f"    {edge.source_id} --> {edge.target_id}")

        return "\n".join(lines)
