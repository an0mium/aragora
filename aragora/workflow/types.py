"""
Type definitions for the Workflow Engine.

The Workflow Engine generalizes Aragora's PhaseExecutor to support
arbitrary multi-step workflows with conditional transitions, parallel
execution, and checkpointing.

Extended for Phase 2: Visual Workflow Builder with:
- Visual layout metadata for drag-drop canvas
- YAML/JSON serialization
- Industry-specific workflow templates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import yaml


class StepStatus(Enum):
    """Status of a workflow step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    WAITING = "waiting"  # Waiting for parallel steps or conditions


class ExecutionPattern(Enum):
    """Patterns for step execution."""

    SEQUENTIAL = "sequential"  # Steps run one after another
    PARALLEL = "parallel"  # Steps run concurrently (hive-mind)
    CONDITIONAL = "conditional"  # Step runs based on condition
    LOOP = "loop"  # Step repeats until condition met


class NodeCategory(Enum):
    """Categories of visual workflow nodes."""

    AGENT = "agent"  # AI agent execution
    TASK = "task"  # Generic task/action
    CONTROL = "control"  # Control flow (decision, loop, parallel)
    MEMORY = "memory"  # Knowledge Mound read/write
    HUMAN = "human"  # Human checkpoint/approval
    DEBATE = "debate"  # Aragora debate execution
    INTEGRATION = "integration"  # External system integration


class EdgeType(Enum):
    """Types of edges in workflow graph."""

    DATA_FLOW = "data_flow"  # Normal data flow
    CONDITIONAL = "conditional"  # Conditional branch
    ERROR = "error"  # Error handling path
    APPROVAL = "approval"  # Human approval path
    REJECTION = "rejection"  # Human rejection path


# =============================================================================
# Visual Layout Types for Workflow Builder
# =============================================================================


@dataclass
class Position:
    """2D position on the workflow canvas."""

    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Position":
        """Create from dictionary."""
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0))


@dataclass
class NodeSize:
    """Size of a workflow node."""

    width: float = 200.0
    height: float = 100.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "NodeSize":
        """Create from dictionary."""
        return cls(width=data.get("width", 200.0), height=data.get("height", 100.0))


@dataclass
class VisualNodeData:
    """Visual metadata for a workflow node (React Flow integration)."""

    position: Position = field(default_factory=Position)
    size: NodeSize = field(default_factory=NodeSize)
    category: NodeCategory = NodeCategory.TASK
    color: str = "#4a5568"  # Default gray
    icon: str = ""  # Optional icon identifier
    collapsed: bool = False  # Whether node is collapsed in canvas
    selected: bool = False  # Whether node is currently selected
    dragging: bool = False  # Whether node is being dragged
    z_index: int = 0  # Stacking order

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "category": self.category.value,
            "color": self.color,
            "icon": self.icon,
            "collapsed": self.collapsed,
            "selected": self.selected,
            "dragging": self.dragging,
            "z_index": self.z_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualNodeData":
        """Create from dictionary."""
        return cls(
            position=Position.from_dict(data.get("position", {})),
            size=NodeSize.from_dict(data.get("size", {})),
            category=NodeCategory(data.get("category", "task")),
            color=data.get("color", "#4a5568"),
            icon=data.get("icon", ""),
            collapsed=data.get("collapsed", False),
            selected=data.get("selected", False),
            dragging=data.get("dragging", False),
            z_index=data.get("z_index", 0),
        )


@dataclass
class VisualEdgeData:
    """Visual metadata for a workflow edge."""

    edge_type: EdgeType = EdgeType.DATA_FLOW
    label: str = ""
    animated: bool = False  # Animated edge (for active flow)
    color: str = "#718096"  # Default edge color
    stroke_width: float = 2.0
    source_handle: str = ""  # Source connection point
    target_handle: str = ""  # Target connection point

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_type": self.edge_type.value,
            "label": self.label,
            "animated": self.animated,
            "color": self.color,
            "stroke_width": self.stroke_width,
            "source_handle": self.source_handle,
            "target_handle": self.target_handle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualEdgeData":
        """Create from dictionary."""
        return cls(
            edge_type=EdgeType(data.get("edge_type", "data_flow")),
            label=data.get("label", ""),
            animated=data.get("animated", False),
            color=data.get("color", "#718096"),
            stroke_width=data.get("stroke_width", 2.0),
            source_handle=data.get("source_handle", ""),
            target_handle=data.get("target_handle", ""),
        )


@dataclass
class CanvasSettings:
    """Settings for the workflow canvas."""

    width: float = 4000.0  # Canvas width
    height: float = 3000.0  # Canvas height
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    grid_size: float = 20.0  # Snap-to-grid size
    snap_to_grid: bool = True
    show_minimap: bool = True
    show_controls: bool = True
    background_color: str = "#f7fafc"
    grid_color: str = "#e2e8f0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "grid_size": self.grid_size,
            "snap_to_grid": self.snap_to_grid,
            "show_minimap": self.show_minimap,
            "show_controls": self.show_controls,
            "background_color": self.background_color,
            "grid_color": self.grid_color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasSettings":
        """Create from dictionary."""
        return cls(
            width=data.get("width", 4000.0),
            height=data.get("height", 3000.0),
            zoom=data.get("zoom", 1.0),
            pan_x=data.get("pan_x", 0.0),
            pan_y=data.get("pan_y", 0.0),
            grid_size=data.get("grid_size", 20.0),
            snap_to_grid=data.get("snap_to_grid", True),
            show_minimap=data.get("show_minimap", True),
            show_controls=data.get("show_controls", True),
            background_color=data.get("background_color", "#f7fafc"),
            grid_color=data.get("grid_color", "#e2e8f0"),
        )


@dataclass
class StepResult:
    """Result from a single workflow step execution."""

    step_id: str
    step_name: str
    status: StepStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    @property
    def success(self) -> bool:
        """Check if step completed successfully."""
        return self.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)


@dataclass
class WorkflowResult:
    """Result from full workflow execution."""

    workflow_id: str
    definition_id: str
    success: bool
    steps: List[StepResult]
    total_duration_ms: float
    final_output: Any = None
    error: Optional[str] = None
    checkpoints_created: int = 0

    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """Get result for a specific step."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None


@dataclass
class StepDefinition:
    """Definition of a workflow step with visual metadata."""

    id: str
    name: str
    step_type: str  # Name of the step implementation class
    config: Dict[str, Any] = field(default_factory=dict)
    execution_pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL
    timeout_seconds: float = 120.0
    retries: int = 0
    optional: bool = False  # If True, workflow continues on failure
    next_steps: List[str] = field(default_factory=list)  # Default transitions

    # Visual metadata for workflow builder
    visual: VisualNodeData = field(default_factory=VisualNodeData)
    description: str = ""  # Human-readable description
    inputs: Dict[str, str] = field(default_factory=dict)  # Input parameter specs
    outputs: Dict[str, str] = field(default_factory=dict)  # Output parameter specs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type,
            "config": self.config,
            "execution_pattern": self.execution_pattern.value,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "optional": self.optional,
            "next_steps": self.next_steps,
            "visual": self.visual.to_dict(),
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepDefinition":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            step_type=data["step_type"],
            config=data.get("config", {}),
            execution_pattern=ExecutionPattern(data.get("execution_pattern", "sequential")),
            timeout_seconds=data.get("timeout_seconds", 120.0),
            retries=data.get("retries", 0),
            optional=data.get("optional", False),
            next_steps=data.get("next_steps", []),
            visual=VisualNodeData.from_dict(data.get("visual", {})),
            description=data.get("description", ""),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
        )


@dataclass
class TransitionRule:
    """Conditional transition between steps with visual metadata."""

    id: str
    from_step: str
    to_step: str
    condition: str  # Python expression evaluated against step output
    priority: int = 0  # Higher priority rules evaluated first

    # Visual metadata for workflow builder
    visual: VisualEdgeData = field(default_factory=VisualEdgeData)
    label: str = ""  # Display label for the edge

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_step": self.from_step,
            "to_step": self.to_step,
            "condition": self.condition,
            "priority": self.priority,
            "visual": self.visual.to_dict(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionRule":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            from_step=data["from_step"],
            to_step=data["to_step"],
            condition=data["condition"],
            priority=data.get("priority", 0),
            visual=VisualEdgeData.from_dict(data.get("visual", {})),
            label=data.get("label", ""),
        )


class WorkflowCategory(Enum):
    """Industry categories for workflow templates."""

    GENERAL = "general"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    CODE = "code"
    ACADEMIC = "academic"
    COMPLIANCE = "compliance"


@dataclass
class WorkflowDefinition:
    """
    Complete definition of a workflow with visual builder support.

    Workflows are defined declaratively and can be stored as JSON/YAML
    for configuration-driven execution. Extended for Phase 2 with:
    - Canvas settings for visual layout
    - Industry categorization for templates
    - YAML serialization support
    """

    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: List[StepDefinition] = field(default_factory=list)
    transitions: List[TransitionRule] = field(default_factory=list)
    entry_step: Optional[str] = None  # First step to execute
    inputs: Dict[str, str] = field(default_factory=dict)  # Input parameter definitions
    outputs: Dict[str, str] = field(default_factory=dict)  # Output parameter definitions
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Visual workflow builder metadata
    canvas: CanvasSettings = field(default_factory=CanvasSettings)
    category: WorkflowCategory = WorkflowCategory.GENERAL
    tags: List[str] = field(default_factory=list)
    icon: str = ""  # Workflow icon for template gallery
    thumbnail: str = ""  # Thumbnail image URL for template gallery

    # Template metadata
    is_template: bool = False
    template_id: Optional[str] = None  # ID of template this was created from
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: str = ""
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        """Set default entry step if not specified."""
        if self.entry_step is None and self.steps:
            self.entry_step = self.steps[0].id

    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get step definition by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_transitions_from(self, step_id: str) -> List[TransitionRule]:
        """Get all transitions from a step, sorted by priority."""
        transitions = [t for t in self.transitions if t.from_step == step_id]
        return sorted(transitions, key=lambda t: -t.priority)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "transitions": [t.to_dict() for t in self.transitions],
            "entry_step": self.entry_step,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "canvas": self.canvas.to_dict(),
            "category": self.category.value,
            "tags": self.tags,
            "icon": self.icon,
            "thumbnail": self.thumbnail,
            "is_template": self.is_template,
            "template_id": self.template_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """Create from dictionary."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            steps=[StepDefinition.from_dict(s) for s in data.get("steps", [])],
            transitions=[TransitionRule.from_dict(t) for t in data.get("transitions", [])],
            entry_step=data.get("entry_step"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            metadata=data.get("metadata", {}),
            canvas=CanvasSettings.from_dict(data.get("canvas", {})),
            category=WorkflowCategory(data.get("category", "general")),
            tags=data.get("tags", []),
            icon=data.get("icon", ""),
            thumbnail=data.get("thumbnail", ""),
            is_template=data.get("is_template", False),
            template_id=data.get("template_id"),
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            created_by=data.get("created_by", ""),
            tenant_id=data.get("tenant_id", "default"),
        )

    def to_yaml(self) -> str:
        """Serialize workflow to YAML format."""
        return yaml.safe_dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "WorkflowDefinition":
        """Create workflow from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def clone(self, new_id: Optional[str] = None, new_name: Optional[str] = None) -> "WorkflowDefinition":
        """Create a copy of this workflow with optional new ID/name."""
        import uuid

        data = self.to_dict()
        data["id"] = new_id or f"wf_{uuid.uuid4().hex[:12]}"
        data["name"] = new_name or f"{self.name} (Copy)"
        data["is_template"] = False
        data["template_id"] = self.id if self.is_template else self.template_id
        data["created_at"] = None
        data["updated_at"] = None
        return WorkflowDefinition.from_dict(data)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate workflow definition. Returns (is_valid, errors)."""
        errors = []

        if not self.id:
            errors.append("Workflow ID is required")
        if not self.name:
            errors.append("Workflow name is required")
        if not self.steps:
            errors.append("Workflow must have at least one step")

        # Validate entry step exists
        if self.entry_step:
            if not self.get_step(self.entry_step):
                errors.append(f"Entry step '{self.entry_step}' not found")

        # Validate transitions reference existing steps
        step_ids = {s.id for s in self.steps}
        for transition in self.transitions:
            if transition.from_step not in step_ids:
                errors.append(f"Transition from unknown step: {transition.from_step}")
            if transition.to_step not in step_ids:
                errors.append(f"Transition to unknown step: {transition.to_step}")

        # Validate next_steps references
        for step in self.steps:
            for next_step in step.next_steps:
                if next_step not in step_ids:
                    errors.append(f"Step '{step.id}' references unknown next step: {next_step}")

        return len(errors) == 0, errors


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""

    id: str
    workflow_id: str
    definition_id: str
    current_step: str
    completed_steps: List[str]
    step_outputs: Dict[str, Any]
    context_state: Dict[str, Any]
    created_at: datetime
    checksum: str = ""  # For integrity verification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "definition_id": self.definition_id,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "step_outputs": self.step_outputs,
            "context_state": self.context_state,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    # Timeout settings
    total_timeout_seconds: float = 3600.0  # 1 hour default
    step_timeout_seconds: float = 120.0  # Per-step timeout

    # Execution behavior
    stop_on_failure: bool = True
    skip_optional_on_timeout: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval_steps: int = 1  # Checkpoint every N steps

    # Tracing and metrics
    enable_tracing: bool = True
    trace_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    metrics_callback: Optional[Callable[[str, float], None]] = None

    # Parallel execution
    max_parallel_steps: int = 10
    parallel_timeout_seconds: float = 300.0


# Type aliases
StepType = Literal[
    "phase", "agent", "parallel", "conditional", "loop", "custom",
    # Phase 2: New step types for workflow builder
    "human_checkpoint", "memory_read", "memory_write", "debate", "decision", "task",
]

# Default colors for node categories (for React Flow)
NODE_CATEGORY_COLORS = {
    NodeCategory.AGENT: "#4299e1",  # Blue
    NodeCategory.TASK: "#48bb78",  # Green
    NodeCategory.CONTROL: "#ed8936",  # Orange
    NodeCategory.MEMORY: "#9f7aea",  # Purple
    NodeCategory.HUMAN: "#f56565",  # Red
    NodeCategory.DEBATE: "#38b2ac",  # Teal
    NodeCategory.INTEGRATION: "#667eea",  # Indigo
}


__all__ = [
    # Enums
    "StepStatus",
    "ExecutionPattern",
    "NodeCategory",
    "EdgeType",
    "WorkflowCategory",
    # Visual types
    "Position",
    "NodeSize",
    "VisualNodeData",
    "VisualEdgeData",
    "CanvasSettings",
    # Core types
    "StepResult",
    "WorkflowResult",
    "StepDefinition",
    "TransitionRule",
    "WorkflowDefinition",
    "WorkflowCheckpoint",
    "WorkflowConfig",
    # Type aliases
    "StepType",
    "NODE_CATEGORY_COLORS",
]
