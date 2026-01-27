"""
Hook Configuration Types.

Dataclasses for representing hook configurations loaded from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "HookConfig",
    "ActionConfig",
    "ConditionConfig",
]


@dataclass
class ConditionConfig:
    """
    Configuration for a hook condition.

    Conditions determine when a hook should fire based on
    the trigger context values.

    Attributes:
        field: The field path to evaluate (supports dot notation)
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, contains, etc.)
        value: The value to compare against
        negate: If True, invert the condition result
    """

    field: str
    operator: str
    value: Any
    negate: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionConfig":
        """Create ConditionConfig from dictionary."""
        return cls(
            field=data["field"],
            operator=data.get("operator", "eq"),
            value=data.get("value"),
            negate=data.get("negate", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "negate": self.negate,
        }


@dataclass
class ActionConfig:
    """
    Configuration for a hook action.

    Specifies what handler to call when hook conditions are met.

    Attributes:
        handler: Fully qualified handler path (e.g., 'aragora.hooks.builtin.notify')
        args: Arguments to pass to the handler
        async_execution: Whether to execute asynchronously
        timeout: Optional timeout in seconds
    """

    handler: str
    args: dict[str, Any] = field(default_factory=dict)
    async_execution: bool = True
    timeout: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionConfig":
        """Create ActionConfig from dictionary."""
        return cls(
            handler=data["handler"],
            args=data.get("args", {}),
            async_execution=data.get("async_execution", True),
            timeout=data.get("timeout"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "handler": self.handler,
            "args": self.args,
            "async_execution": self.async_execution,
        }
        if self.timeout is not None:
            result["timeout"] = self.timeout
        return result


@dataclass
class HookConfig:
    """
    Complete hook configuration.

    Represents a single hook definition loaded from YAML.

    Attributes:
        name: Unique identifier for this hook
        trigger: Hook type/event to trigger on (e.g., 'post_debate', 'on_finding')
        action: The action to execute
        conditions: List of conditions (all must pass for hook to fire)
        priority: Execution priority (critical, high, normal, low, cleanup)
        enabled: Whether the hook is active
        one_shot: If True, unregister after first execution
        description: Human-readable description
        tags: Optional tags for categorization
    """

    name: str
    trigger: str
    action: ActionConfig
    conditions: list[ConditionConfig] = field(default_factory=list)
    priority: str = "normal"
    enabled: bool = True
    one_shot: bool = False
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Source tracking
    source_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Optional[str] = None) -> "HookConfig":
        """Create HookConfig from dictionary."""
        # Parse action
        action_data = data.get("action", {})
        if isinstance(action_data, str):
            # Simple handler string
            action = ActionConfig(handler=action_data)
        else:
            action = ActionConfig.from_dict(action_data)

        # Parse conditions
        conditions = []
        for cond_data in data.get("conditions", []):
            conditions.append(ConditionConfig.from_dict(cond_data))

        return cls(
            name=data["name"],
            trigger=data["trigger"],
            action=action,
            conditions=conditions,
            priority=data.get("priority", "normal"),
            enabled=data.get("enabled", True),
            one_shot=data.get("one_shot", False),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            source_file=source_file,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "trigger": self.trigger,
            "action": self.action.to_dict(),
            "priority": self.priority,
            "enabled": self.enabled,
        }

        if self.conditions:
            result["conditions"] = [c.to_dict() for c in self.conditions]

        if self.one_shot:
            result["one_shot"] = True

        if self.description:
            result["description"] = self.description

        if self.tags:
            result["tags"] = self.tags

        return result

    def __hash__(self) -> int:
        """Make hashable by name."""
        return hash(self.name)
