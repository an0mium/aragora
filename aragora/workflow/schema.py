"""
Workflow Schema Validation for YAML/JSON workflows.

Provides comprehensive validation for workflow definitions loaded from
YAML or JSON files. Uses Pydantic models for type safety and detailed
error messages.

Features:
- Validate workflow structure and step configurations
- Check step type compatibility
- Validate transition rules and conditions
- Detect cycles and unreachable steps
- Validate resource limits
- Template variable validation

Usage:
    from aragora.workflow.schema import (
        validate_workflow,
        WorkflowSchema,
        ValidationResult,
    )

    # Validate a workflow definition
    result = validate_workflow(workflow_dict)
    if not result.valid:
        for error in result.errors:
            print(f"Error: {error}")

    # Or use the schema directly
    schema = WorkflowSchema(**workflow_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import yaml

# Try to use Pydantic v2, fall back to basic validation
try:
    from pydantic import BaseModel, Field, field_validator, model_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc,assignment]


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """A validation message with severity and location."""

    severity: ValidationSeverity
    message: str
    path: str = ""  # JSON path to the error location
    code: str = ""  # Error code for programmatic handling

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        location = f" at {self.path}" if self.path else ""
        return f"{prefix}{location}: {self.message}"


@dataclass
class ValidationResult:
    """Result of workflow validation."""

    valid: bool
    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationMessage]:
        """Get only error messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Get only warning messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.WARNING]

    def add_error(self, message: str, path: str = "", code: str = "") -> None:
        """Add an error message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                message=message,
                path=path,
                code=code,
            )
        )
        self.valid = False

    def add_warning(self, message: str, path: str = "", code: str = "") -> None:
        """Add a warning message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                message=message,
                path=path,
                code=code,
            )
        )

    def add_info(self, message: str, path: str = "", code: str = "") -> None:
        """Add an info message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.INFO,
                message=message,
                path=path,
                code=code,
            )
        )


# Valid step types
VALID_STEP_TYPES = {
    "agent",
    "parallel",
    "conditional",
    "loop",
    "human_checkpoint",
    "memory_read",
    "memory_write",
    "debate",
    "quick_debate",
    "decision",
    "switch",
    "task",
}

# Step type specific required config keys
STEP_TYPE_REQUIRED_CONFIG = {
    "agent": [],  # agent_type and prompt_template optional
    "debate": [],  # topic optional, has defaults
    "task": ["task_type"],
    "human_checkpoint": [],
    "memory_read": [],
    "memory_write": [],
    "decision": [],
    "switch": ["cases"],
}

# Valid task types for TaskStep
VALID_TASK_TYPES = {
    "function",
    "http",
    "transform",
    "validate",
    "aggregate",
}


class WorkflowValidator:
    """
    Validates workflow definitions.

    Performs structural and semantic validation:
    - Required fields present
    - Valid step types
    - Valid transitions
    - No orphan steps
    - No cycles (optional)
    - Resource limits within bounds
    """

    def __init__(
        self,
        allow_cycles: bool = True,
        max_steps: int = 100,
        max_transitions: int = 500,
    ):
        self.allow_cycles = allow_cycles
        self.max_steps = max_steps
        self.max_transitions = max_transitions

    def validate(self, workflow: Dict[str, Any]) -> ValidationResult:
        """
        Validate a workflow definition dictionary.

        Args:
            workflow: Workflow definition as dictionary

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)

        # Validate top-level structure
        self._validate_structure(workflow, result)
        if not result.valid:
            return result

        # Validate steps
        self._validate_steps(workflow, result)

        # Validate transitions
        self._validate_transitions(workflow, result)

        # Validate graph properties
        self._validate_graph(workflow, result)

        # Validate resource limits if present
        if "limits" in workflow:
            self._validate_limits(workflow["limits"], result)

        return result

    def _validate_structure(self, workflow: Dict[str, Any], result: ValidationResult) -> None:
        """Validate top-level workflow structure."""
        # Required fields
        if not workflow.get("id"):
            result.add_error("Workflow ID is required", "id", "MISSING_ID")

        if not workflow.get("name"):
            result.add_error("Workflow name is required", "name", "MISSING_NAME")

        if not workflow.get("steps"):
            result.add_error("Workflow must have at least one step", "steps", "NO_STEPS")
            return

        if not isinstance(workflow.get("steps"), list):
            result.add_error("Steps must be a list", "steps", "INVALID_STEPS_TYPE")
            return

        # Check limits
        if len(workflow.get("steps", [])) > self.max_steps:
            result.add_error(
                f"Too many steps: {len(workflow['steps'])} > {self.max_steps}",
                "steps",
                "TOO_MANY_STEPS",
            )

        if len(workflow.get("transitions", [])) > self.max_transitions:
            result.add_error(
                f"Too many transitions: {len(workflow['transitions'])} > {self.max_transitions}",
                "transitions",
                "TOO_MANY_TRANSITIONS",
            )

        # Validate entry_step
        entry_step = workflow.get("entry_step")
        if entry_step:
            step_ids = {s.get("id") for s in workflow.get("steps", [])}
            if entry_step not in step_ids:
                result.add_error(
                    f"Entry step '{entry_step}' not found in steps",
                    "entry_step",
                    "INVALID_ENTRY_STEP",
                )

    def _validate_steps(self, workflow: Dict[str, Any], result: ValidationResult) -> None:
        """Validate step definitions."""
        step_ids = set()

        for i, step in enumerate(workflow.get("steps", [])):
            path = f"steps[{i}]"

            # Check required fields
            if not step.get("id"):
                result.add_error(f"Step at index {i} missing ID", path, "MISSING_STEP_ID")
                continue

            step_id = step["id"]

            # Check for duplicate IDs
            if step_id in step_ids:
                result.add_error(
                    f"Duplicate step ID: {step_id}",
                    f"{path}.id",
                    "DUPLICATE_STEP_ID",
                )
            step_ids.add(step_id)

            # Check step name
            if not step.get("name"):
                result.add_warning(
                    f"Step '{step_id}' has no name",
                    f"{path}.name",
                    "MISSING_STEP_NAME",
                )

            # Validate step type
            step_type = step.get("step_type", "task")
            if step_type not in VALID_STEP_TYPES:
                result.add_error(
                    f"Invalid step type '{step_type}' for step '{step_id}'",
                    f"{path}.step_type",
                    "INVALID_STEP_TYPE",
                )

            # Validate step-specific config
            self._validate_step_config(step, path, result)

            # Validate next_steps references
            for next_id in step.get("next_steps", []):
                # Will be validated in graph validation
                pass

    def _validate_step_config(
        self,
        step: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate step-specific configuration."""
        step_type = step.get("step_type", "task")
        config = step.get("config", {})
        step_id = step.get("id", "unknown")

        # Check required config keys
        required = STEP_TYPE_REQUIRED_CONFIG.get(step_type, [])
        for key in required:
            if key not in config:
                result.add_error(
                    f"Step '{step_id}' ({step_type}) missing required config: {key}",
                    f"{path}.config.{key}",
                    "MISSING_REQUIRED_CONFIG",
                )

        # Task-specific validation
        if step_type == "task":
            task_type = config.get("task_type")
            if task_type and task_type not in VALID_TASK_TYPES:
                result.add_warning(
                    f"Unknown task type '{task_type}' for step '{step_id}'",
                    f"{path}.config.task_type",
                    "UNKNOWN_TASK_TYPE",
                )

            # HTTP task validation
            if task_type == "http":
                if not config.get("url"):
                    result.add_error(
                        f"HTTP task '{step_id}' missing URL",
                        f"{path}.config.url",
                        "MISSING_HTTP_URL",
                    )

        # Agent step validation
        if step_type == "agent":
            if not config.get("agent_type") and not config.get("prompt_template"):
                result.add_warning(
                    f"Agent step '{step_id}' has no agent_type or prompt_template",
                    f"{path}.config",
                    "INCOMPLETE_AGENT_CONFIG",
                )

        # Debate step validation
        if step_type in ("debate", "quick_debate"):
            if not config.get("topic") and not config.get("question"):
                result.add_warning(
                    f"Debate step '{step_id}' has no topic/question",
                    f"{path}.config",
                    "MISSING_DEBATE_TOPIC",
                )

        # Validate timeout
        timeout = step.get("timeout_seconds", 120)
        if timeout <= 0:
            result.add_error(
                f"Step '{step_id}' has invalid timeout: {timeout}",
                f"{path}.timeout_seconds",
                "INVALID_TIMEOUT",
            )
        elif timeout > 3600:
            result.add_warning(
                f"Step '{step_id}' has very long timeout: {timeout}s",
                f"{path}.timeout_seconds",
                "LONG_TIMEOUT",
            )

    def _validate_transitions(self, workflow: Dict[str, Any], result: ValidationResult) -> None:
        """Validate transition rules."""
        step_ids = {s.get("id") for s in workflow.get("steps", [])}
        transition_ids = set()

        for i, transition in enumerate(workflow.get("transitions", [])):
            path = f"transitions[{i}]"

            # Check required fields
            tr_id = transition.get("id", f"tr_{i}")
            if tr_id in transition_ids:
                result.add_warning(
                    f"Duplicate transition ID: {tr_id}",
                    f"{path}.id",
                    "DUPLICATE_TRANSITION_ID",
                )
            transition_ids.add(tr_id)

            from_step = transition.get("from_step")
            to_step = transition.get("to_step")

            if not from_step:
                result.add_error(
                    f"Transition {tr_id} missing from_step",
                    f"{path}.from_step",
                    "MISSING_FROM_STEP",
                )
            elif from_step not in step_ids:
                result.add_error(
                    f"Transition {tr_id} references unknown from_step: {from_step}",
                    f"{path}.from_step",
                    "UNKNOWN_FROM_STEP",
                )

            if not to_step:
                result.add_error(
                    f"Transition {tr_id} missing to_step",
                    f"{path}.to_step",
                    "MISSING_TO_STEP",
                )
            elif to_step not in step_ids:
                result.add_error(
                    f"Transition {tr_id} references unknown to_step: {to_step}",
                    f"{path}.to_step",
                    "UNKNOWN_TO_STEP",
                )

            # Validate condition syntax
            condition = transition.get("condition", "True")
            if condition:
                self._validate_condition(condition, path, result)

    def _validate_condition(
        self,
        condition: str,
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate a condition expression."""
        # Basic syntax check
        try:
            compile(condition, "<condition>", "eval")
        except SyntaxError as e:
            result.add_error(
                f"Invalid condition syntax: {e}",
                f"{path}.condition",
                "INVALID_CONDITION_SYNTAX",
            )
            return

        # Check for dangerous patterns
        dangerous = ["__", "import", "exec", "eval", "open", "file"]
        for pattern in dangerous:
            if pattern in condition:
                result.add_warning(
                    f"Potentially unsafe pattern in condition: {pattern}",
                    f"{path}.condition",
                    "UNSAFE_CONDITION",
                )

    def _validate_graph(self, workflow: Dict[str, Any], result: ValidationResult) -> None:
        """Validate workflow graph properties."""
        steps = workflow.get("steps", [])
        transitions = workflow.get("transitions", [])

        if not steps:
            return

        step_ids = {s.get("id") for s in steps}
        entry_step = workflow.get("entry_step") or steps[0].get("id")

        # Build adjacency list
        graph: Dict[str, Set[str]] = {s.get("id"): set() for s in steps}

        for step in steps:
            step_id = step.get("id")
            for next_id in step.get("next_steps", []):
                if next_id in step_ids:
                    graph[step_id].add(next_id)
                else:
                    result.add_error(
                        f"Step '{step_id}' references unknown next_step: {next_id}",
                        "steps[?].next_steps",
                        "UNKNOWN_NEXT_STEP",
                    )

        for transition in transitions:
            from_step = transition.get("from_step")
            to_step = transition.get("to_step")
            if from_step in graph and to_step in step_ids:
                graph[from_step].add(to_step)

        # Check for unreachable steps
        reachable = self._find_reachable(entry_step, graph)
        unreachable = step_ids - reachable
        if unreachable:
            result.add_warning(
                f"Unreachable steps: {', '.join(unreachable)}",
                "steps",
                "UNREACHABLE_STEPS",
            )

        # Check for cycles if not allowed
        if not self.allow_cycles:
            cycle = self._find_cycle(entry_step, graph)
            if cycle:
                result.add_error(
                    f"Cycle detected: {' -> '.join(cycle)}",
                    "transitions",
                    "CYCLE_DETECTED",
                )

    def _find_reachable(self, start: str, graph: Dict[str, Set[str]]) -> Set[str]:
        """Find all reachable nodes from start."""
        reachable = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(graph.get(node, set()))

        return reachable

    def _find_cycle(self, start: str, graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """Find a cycle in the graph, if any."""
        visited: set[str] = set()
        path: List[str] = []
        path_set: set[str] = set()

        def dfs(node: str) -> Optional[List[str]]:
            if node in path_set:
                # Found cycle
                idx = path.index(node)
                return path[idx:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for neighbor in graph.get(node, set()):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            path.pop()
            path_set.remove(node)
            return None

        return dfs(start)

    def _validate_limits(self, limits: Dict[str, Any], result: ValidationResult) -> None:
        """Validate resource limits."""
        path = "limits"

        # Max tokens
        max_tokens = limits.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                result.add_error(
                    f"Invalid max_tokens: {max_tokens}",
                    f"{path}.max_tokens",
                    "INVALID_MAX_TOKENS",
                )
            elif max_tokens > 1000000:
                result.add_warning(
                    f"Very high max_tokens: {max_tokens}",
                    f"{path}.max_tokens",
                    "HIGH_MAX_TOKENS",
                )

        # Max cost
        max_cost = limits.get("max_cost_usd")
        if max_cost is not None:
            if not isinstance(max_cost, (int, float)) or max_cost <= 0:
                result.add_error(
                    f"Invalid max_cost_usd: {max_cost}",
                    f"{path}.max_cost_usd",
                    "INVALID_MAX_COST",
                )
            elif max_cost > 100:
                result.add_warning(
                    f"High max_cost_usd: ${max_cost}",
                    f"{path}.max_cost_usd",
                    "HIGH_MAX_COST",
                )

        # Timeout
        timeout = limits.get("timeout_seconds")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                result.add_error(
                    f"Invalid timeout_seconds: {timeout}",
                    f"{path}.timeout_seconds",
                    "INVALID_TIMEOUT",
                )


def validate_workflow(workflow: Union[Dict[str, Any], str]) -> ValidationResult:
    """
    Validate a workflow definition.

    Args:
        workflow: Workflow as dict, YAML string, or JSON string

    Returns:
        ValidationResult with errors and warnings
    """
    # Parse if string
    workflow_dict: Dict[str, Any]
    if isinstance(workflow, str):
        try:
            # Try YAML first (superset of JSON)
            parsed = yaml.safe_load(workflow)
            if not isinstance(parsed, dict):
                result = ValidationResult(valid=False)
                result.add_error("Workflow must be a YAML/JSON object", "", "PARSE_ERROR")
                return result
            workflow_dict = parsed
        except yaml.YAMLError as e:
            result = ValidationResult(valid=False)
            result.add_error(f"Invalid YAML/JSON: {e}", "", "PARSE_ERROR")
            return result
    else:
        workflow_dict = workflow

    validator = WorkflowValidator()
    return validator.validate(workflow_dict)


def validate_workflow_file(path: str) -> ValidationResult:
    """
    Validate a workflow from a file.

    Args:
        path: Path to YAML or JSON file

    Returns:
        ValidationResult with errors and warnings
    """
    try:
        with open(path, "r") as f:
            content = f.read()
        return validate_workflow(content)
    except FileNotFoundError:
        result = ValidationResult(valid=False)
        result.add_error(f"File not found: {path}", "", "FILE_NOT_FOUND")
        return result
    except IOError as e:
        result = ValidationResult(valid=False)
        result.add_error(f"Error reading file: {e}", "", "READ_ERROR")
        return result


# Pydantic schemas for stricter validation (if available)
if PYDANTIC_AVAILABLE:

    class StepConfigSchema(BaseModel):
        """Schema for step configuration."""

        class Config:
            extra = "allow"

    class VisualNodeSchema(BaseModel):
        """Schema for visual node data."""

        position: Optional[Dict[str, float]] = None
        size: Optional[Dict[str, float]] = None
        category: Optional[str] = None
        color: Optional[str] = None

        class Config:
            extra = "allow"

    class StepSchema(BaseModel):
        """Schema for workflow step."""

        id: str = Field(..., min_length=1)
        name: str = Field(default="")
        step_type: str = Field(default="task")
        config: Dict[str, Any] = Field(default_factory=dict)
        execution_pattern: str = Field(default="sequential")
        timeout_seconds: float = Field(default=120.0, gt=0)
        retries: int = Field(default=0, ge=0)
        optional: bool = Field(default=False)
        next_steps: List[str] = Field(default_factory=list)
        visual: Optional[Dict[str, Any]] = None
        description: str = Field(default="")

        @field_validator("step_type")
        @classmethod
        def validate_step_type(cls, v):
            if v not in VALID_STEP_TYPES:
                raise ValueError(f"Invalid step type: {v}")
            return v

    class TransitionSchema(BaseModel):
        """Schema for transition rule."""

        id: str = Field(default="")
        from_step: str = Field(..., min_length=1)
        to_step: str = Field(..., min_length=1)
        condition: str = Field(default="True")
        priority: int = Field(default=0)
        label: str = Field(default="")

    class ResourceLimitsSchema(BaseModel):
        """Schema for resource limits."""

        max_tokens: int = Field(default=100000, gt=0)
        max_cost_usd: float = Field(default=10.0, gt=0)
        timeout_seconds: float = Field(default=600.0, gt=0)
        max_parallel_agents: int = Field(default=5, gt=0)
        max_retries: int = Field(default=3, ge=0)

    class WorkflowSchema(BaseModel):
        """Schema for complete workflow definition."""

        id: str = Field(..., min_length=1)
        name: str = Field(..., min_length=1)
        description: str = Field(default="")
        version: str = Field(default="1.0.0")
        steps: List[StepSchema] = Field(..., min_length=1)
        transitions: List[TransitionSchema] = Field(default_factory=list)
        entry_step: Optional[str] = None
        inputs: Dict[str, str] = Field(default_factory=dict)
        outputs: Dict[str, str] = Field(default_factory=dict)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        limits: Optional[ResourceLimitsSchema] = None
        category: str = Field(default="general")
        tags: List[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_references(self):
            """Validate step references."""
            step_ids = {s.id for s in self.steps}

            # Validate entry_step
            if self.entry_step and self.entry_step not in step_ids:
                raise ValueError(f"Entry step '{self.entry_step}' not found")

            # Validate transitions
            for t in self.transitions:
                if t.from_step not in step_ids:
                    raise ValueError(f"Transition from unknown step: {t.from_step}")
                if t.to_step not in step_ids:
                    raise ValueError(f"Transition to unknown step: {t.to_step}")

            # Validate next_steps
            for step in self.steps:
                for next_id in step.next_steps:
                    if next_id not in step_ids:
                        raise ValueError(f"Step '{step.id}' references unknown: {next_id}")

            return self


__all__ = [
    "ValidationSeverity",
    "ValidationMessage",
    "ValidationResult",
    "WorkflowValidator",
    "validate_workflow",
    "validate_workflow_file",
    "VALID_STEP_TYPES",
    "VALID_TASK_TYPES",
]

if PYDANTIC_AVAILABLE:
    __all__.extend(
        [
            "WorkflowSchema",
            "StepSchema",
            "TransitionSchema",
            "ResourceLimitsSchema",
        ]
    )
