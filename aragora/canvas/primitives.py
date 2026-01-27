"""
Canvas A2UI Primitives - Agent-to-User Interface Components.

Provides rich UI primitives that agents can use to build interactive
interfaces for users. These primitives enable:

- Interactive buttons with callbacks
- Form inputs with validation
- Data visualizations (charts, tables)
- Progress indicators for long operations
- Alert/notification displays

Usage:
    from aragora.canvas.primitives import (
        ButtonPrimitive,
        FormPrimitive,
        ChartPrimitive,
        ProgressPrimitive,
    )

    # Create a button
    button = ButtonPrimitive(
        label="Run Analysis",
        action="run_analysis",
        variant="primary",
    )

    # Create a progress indicator
    progress = ProgressPrimitive(
        value=45,
        max_value=100,
        label="Processing...",
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ButtonVariant(str, Enum):
    """Visual variants for buttons."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    GHOST = "ghost"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class ChartType(str, Enum):
    """Types of charts."""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"


@dataclass
class ButtonPrimitive:
    """
    Interactive button with action callback.

    Used to create clickable buttons in the canvas that trigger
    agent actions or user interactions.
    """

    label: str
    action: str  # Action identifier for callback
    variant: ButtonVariant = ButtonVariant.PRIMARY
    disabled: bool = False
    loading: bool = False
    icon: Optional[str] = None
    tooltip: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "button",
            "label": self.label,
            "action": self.action,
            "variant": self.variant.value,
            "disabled": self.disabled,
            "loading": self.loading,
            "icon": self.icon,
            "tooltip": self.tooltip,
            "metadata": self.metadata,
        }


@dataclass
class FormField:
    """A single field in a form."""

    name: str
    field_type: str  # text, number, email, select, checkbox, textarea
    label: str
    required: bool = False
    default_value: Any = None
    placeholder: Optional[str] = None
    options: List[Dict[str, str]] = field(default_factory=list)  # For select
    validation: Optional[str] = None  # Regex pattern

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type,
            "label": self.label,
            "required": self.required,
            "default_value": self.default_value,
            "placeholder": self.placeholder,
            "options": self.options,
            "validation": self.validation,
        }


@dataclass
class FormPrimitive:
    """
    Form with multiple input fields.

    Enables agents to collect structured input from users.
    """

    fields: List[FormField]
    submit_action: str
    submit_label: str = "Submit"
    cancel_label: Optional[str] = "Cancel"
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "form",
            "fields": [f.to_dict() for f in self.fields],
            "submit_action": self.submit_action,
            "submit_label": self.submit_label,
            "cancel_label": self.cancel_label,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class SelectOption:
    """An option in a select control."""

    value: str
    label: str
    disabled: bool = False


@dataclass
class SelectPrimitive:
    """
    Dropdown selection control.

    Allows users to choose from a list of options.
    """

    options: List[SelectOption]
    action: str
    label: Optional[str] = None
    selected_value: Optional[str] = None
    placeholder: str = "Select..."
    multiple: bool = False
    searchable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "select",
            "options": [
                {"value": o.value, "label": o.label, "disabled": o.disabled} for o in self.options
            ],
            "action": self.action,
            "label": self.label,
            "selected_value": self.selected_value,
            "placeholder": self.placeholder,
            "multiple": self.multiple,
            "searchable": self.searchable,
            "metadata": self.metadata,
        }


@dataclass
class ChartDataPoint:
    """A data point in a chart."""

    x: Any  # X-axis value
    y: float  # Y-axis value
    label: Optional[str] = None


@dataclass
class ChartSeries:
    """A series of data points in a chart."""

    name: str
    data: List[ChartDataPoint]
    color: Optional[str] = None


@dataclass
class ChartPrimitive:
    """
    Data visualization chart.

    Supports line, bar, pie, scatter, and area charts.
    """

    chart_type: ChartType
    series: List[ChartSeries]
    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    width: int = 400
    height: int = 300
    show_legend: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "chart",
            "chart_type": self.chart_type.value,
            "series": [
                {
                    "name": s.name,
                    "data": [{"x": p.x, "y": p.y, "label": p.label} for p in s.data],
                    "color": s.color,
                }
                for s in self.series
            ],
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "width": self.width,
            "height": self.height,
            "show_legend": self.show_legend,
            "metadata": self.metadata,
        }


@dataclass
class ProgressPrimitive:
    """
    Progress indicator for long-running operations.

    Shows completion percentage and optional status message.
    """

    value: float  # Current progress (0-100)
    max_value: float = 100.0
    label: Optional[str] = None
    show_percentage: bool = True
    animated: bool = True
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        """Get progress as percentage."""
        if self.max_value <= 0:
            return 0.0
        return min(100.0, (self.value / self.max_value) * 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "progress",
            "value": self.value,
            "max_value": self.max_value,
            "percentage": self.percentage,
            "label": self.label,
            "show_percentage": self.show_percentage,
            "animated": self.animated,
            "color": self.color,
            "metadata": self.metadata,
        }


@dataclass
class AlertPrimitive:
    """
    Alert/notification display.

    Shows informational, success, warning, or error messages.
    """

    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    title: Optional[str] = None
    dismissible: bool = True
    auto_dismiss_seconds: Optional[int] = None
    action: Optional[str] = None  # Action for dismiss callback
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "alert",
            "message": self.message,
            "severity": self.severity.value,
            "title": self.title,
            "dismissible": self.dismissible,
            "auto_dismiss_seconds": self.auto_dismiss_seconds,
            "action": self.action,
            "metadata": self.metadata,
        }


@dataclass
class CardPrimitive:
    """
    Information card display.

    Displays structured information in a card format.
    """

    title: str
    content: str
    subtitle: Optional[str] = None
    image_url: Optional[str] = None
    actions: List[ButtonPrimitive] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "card",
            "title": self.title,
            "content": self.content,
            "subtitle": self.subtitle,
            "image_url": self.image_url,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata,
        }


@dataclass
class TableColumn:
    """A column in a table."""

    key: str
    header: str
    sortable: bool = False
    width: Optional[int] = None


@dataclass
class TablePrimitive:
    """
    Data table display.

    Shows structured data in a tabular format.
    """

    columns: List[TableColumn]
    rows: List[Dict[str, Any]]
    title: Optional[str] = None
    sortable: bool = True
    paginated: bool = False
    page_size: int = 10
    selectable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "table",
            "columns": [
                {
                    "key": c.key,
                    "header": c.header,
                    "sortable": c.sortable,
                    "width": c.width,
                }
                for c in self.columns
            ],
            "rows": self.rows,
            "title": self.title,
            "sortable": self.sortable,
            "paginated": self.paginated,
            "page_size": self.page_size,
            "selectable": self.selectable,
            "metadata": self.metadata,
        }


# Export all primitives
__all__ = [
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
