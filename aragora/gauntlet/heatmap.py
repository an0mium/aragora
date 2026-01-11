"""
Risk Heatmap - Visual risk aggregation.

Provides a category x severity breakdown of findings
for dashboard visualization.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .result import GauntletResult, SeverityLevel


@dataclass
class HeatmapCell:
    """A single cell in the risk heatmap."""

    category: str
    severity: str
    count: int
    vulnerabilities: list[str] = field(default_factory=list)  # Vulnerability IDs

    @property
    def intensity(self) -> float:
        """Calculate intensity for coloring (0-1)."""
        # Log scale for better visualization
        if self.count == 0:
            return 0.0
        import math
        return min(1.0, math.log10(self.count + 1) / 2)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "count": self.count,
            "intensity": self.intensity,
            "vulnerabilities": self.vulnerabilities,
        }


@dataclass
class RiskHeatmap:
    """
    Risk heatmap aggregating findings by category and severity.

    Provides data structure for visual risk display in dashboards.
    """

    cells: list[HeatmapCell] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    severities: list[str] = field(default_factory=lambda: ["critical", "high", "medium", "low"])

    # Summary statistics
    total_findings: int = 0
    highest_risk_category: Optional[str] = None
    highest_risk_severity: Optional[str] = None

    @classmethod
    def from_result(cls, result: GauntletResult) -> "RiskHeatmap":
        """Create heatmap from GauntletResult."""
        # Collect all categories
        categories = set()
        for vuln in result.vulnerabilities:
            categories.add(vuln.category)
        categories = sorted(categories)

        severities = ["critical", "high", "medium", "low"]

        # Build cell matrix
        cells = []
        category_totals: dict[str, int] = {}
        severity_totals: dict[str, int] = {}

        for category in categories:
            category_totals[category] = 0
            for severity in severities:
                vulns = [
                    v for v in result.vulnerabilities
                    if v.category == category and v.severity.value == severity
                ]
                count = len(vulns)
                category_totals[category] += count
                severity_totals[severity] = severity_totals.get(severity, 0) + count

                cells.append(HeatmapCell(
                    category=category,
                    severity=severity,
                    count=count,
                    vulnerabilities=[v.id for v in vulns],
                ))

        # Find highest risk
        highest_category = max(category_totals, key=category_totals.get) if category_totals else None
        highest_severity = "critical" if severity_totals.get("critical", 0) > 0 else (
            "high" if severity_totals.get("high", 0) > 0 else None
        )

        return cls(
            cells=cells,
            categories=list(categories),
            severities=severities,
            total_findings=result.risk_summary.total,
            highest_risk_category=highest_category,
            highest_risk_severity=highest_severity,
        )

    def get_cell(self, category: str, severity: str) -> Optional[HeatmapCell]:
        """Get a specific cell."""
        for cell in self.cells:
            if cell.category == category and cell.severity == severity:
                return cell
        return None

    def get_category_total(self, category: str) -> int:
        """Get total findings for a category."""
        return sum(c.count for c in self.cells if c.category == category)

    def get_severity_total(self, severity: str) -> int:
        """Get total findings for a severity."""
        return sum(c.count for c in self.cells if c.severity == severity)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cells": [c.to_dict() for c in self.cells],
            "categories": self.categories,
            "severities": self.severities,
            "total_findings": self.total_findings,
            "highest_risk_category": self.highest_risk_category,
            "highest_risk_severity": self.highest_risk_severity,
            "matrix": self._to_matrix(),
        }

    def _to_matrix(self) -> list[list[int]]:
        """Convert to 2D matrix for visualization."""
        matrix = []
        for category in self.categories:
            row = []
            for severity in self.severities:
                cell = self.get_cell(category, severity)
                row.append(cell.count if cell else 0)
            matrix.append(row)
        return matrix

    def to_json(self) -> str:
        """Export as JSON."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    def to_svg(self, width: int = 600, height: int = 400) -> str:
        """Generate SVG visualization."""
        if not self.categories:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No data</text></svg>'

        cell_width = min(100, (width - 150) // len(self.severities))
        cell_height = min(60, (height - 100) // len(self.categories))

        # Color scale
        colors = {
            "critical": "#dc2626",  # Red
            "high": "#ea580c",       # Orange
            "medium": "#eab308",     # Yellow
            "low": "#22c55e",        # Green
        }

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<style>',
            '  .label { font-family: sans-serif; font-size: 12px; }',
            '  .count { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: white; }',
            '  .header { font-family: sans-serif; font-size: 11px; font-weight: bold; }',
            '</style>',
        ]

        # Header row (severities)
        x_offset = 120
        for i, severity in enumerate(self.severities):
            x = x_offset + i * cell_width + cell_width // 2
            svg_parts.append(
                f'<text x="{x}" y="30" text-anchor="middle" class="header">{severity.upper()}</text>'
            )

        # Category rows
        y_offset = 50
        for i, category in enumerate(self.categories):
            y = y_offset + i * cell_height

            # Category label
            svg_parts.append(
                f'<text x="10" y="{y + cell_height // 2 + 4}" class="label">{category[:15]}</text>'
            )

            # Cells
            for j, severity in enumerate(self.severities):
                x = x_offset + j * cell_width
                cell = self.get_cell(category, severity)
                count = cell.count if cell else 0

                # Calculate opacity based on count
                opacity = min(1.0, 0.2 + count * 0.2) if count > 0 else 0.1
                color = colors.get(severity, "#94a3b8")

                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_width - 2}" height="{cell_height - 2}" '
                    f'fill="{color}" fill-opacity="{opacity}" rx="4"/>'
                )

                if count > 0:
                    svg_parts.append(
                        f'<text x="{x + cell_width // 2}" y="{y + cell_height // 2 + 5}" '
                        f'text-anchor="middle" class="count">{count}</text>'
                    )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def to_ascii(self) -> str:
        """Generate ASCII table representation."""
        if not self.categories:
            return "No findings to display"

        # Header
        header = "Category".ljust(20) + " | " + " | ".join(s[:4].upper().center(6) for s in self.severities)
        separator = "-" * len(header)

        lines = [header, separator]

        for category in self.categories:
            row_parts = [category[:20].ljust(20)]
            for severity in self.severities:
                cell = self.get_cell(category, severity)
                count = cell.count if cell else 0
                row_parts.append(str(count).center(6))
            lines.append(" | ".join(row_parts))

        lines.append(separator)

        # Totals
        total_parts = ["TOTAL".ljust(20)]
        for severity in self.severities:
            total = self.get_severity_total(severity)
            total_parts.append(str(total).center(6))
        lines.append(" | ".join(total_parts))

        return "\n".join(lines)
