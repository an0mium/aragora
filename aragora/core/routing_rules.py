"""
Routing Rules Engine for Aragora Control Plane.

Provides conditional routing of deliberation decisions to various channels
based on configurable rules and conditions.

Example:
    >>> from aragora.core.routing_rules import RoutingRule, Condition, Action
    >>> rule = RoutingRule(
    ...     id="low-confidence-escalate",
    ...     name="Escalate Low Confidence Decisions",
    ...     conditions=[
    ...         Condition(field="confidence", operator="lt", value=0.7)
    ...     ],
    ...     actions=[
    ...         Action(type="escalate_to", target="security-team")
    ...     ],
    ...     priority=1,
    ...     enabled=True
    ... )
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal


class ConditionOperator(str, Enum):
    """Operators for condition evaluation."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex match
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class ActionType(str, Enum):
    """Types of actions that can be triggered by rules."""

    ROUTE_TO_CHANNEL = "route_to_channel"
    ESCALATE_TO = "escalate_to"
    NOTIFY = "notify"
    TAG = "tag"
    SET_PRIORITY = "set_priority"
    DELAY = "delay"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Condition:
    """
    A condition that evaluates against deliberation context.

    Attributes:
        field: The field to evaluate (e.g., "confidence", "topic", "agent_count")
        operator: The comparison operator
        value: The value to compare against
        case_sensitive: Whether string comparisons are case-sensitive
    """

    field: str
    operator: ConditionOperator | str
    value: Any
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.operator, str):
            self.operator = ConditionOperator(self.operator)

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate this condition against a context dictionary.

        Args:
            context: Dictionary containing deliberation data

        Returns:
            True if condition is met, False otherwise
        """
        # Handle nested field access (e.g., "consensus.confidence")
        field_value = self._get_nested_value(context, self.field)

        # Handle existence checks
        if self.operator == ConditionOperator.EXISTS:
            return field_value is not None
        if self.operator == ConditionOperator.NOT_EXISTS:
            return field_value is None

        # If field doesn't exist, condition fails (except for not_exists)
        if field_value is None:
            return False

        # Normalize strings for case-insensitive comparison
        compare_value = self.value
        if isinstance(field_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            if isinstance(compare_value, str):
                compare_value = compare_value.lower()

        # Evaluate based on operator
        match self.operator:
            case ConditionOperator.EQUALS:
                return bool(field_value == compare_value)
            case ConditionOperator.NOT_EQUALS:
                return bool(field_value != compare_value)
            case ConditionOperator.GREATER_THAN:
                return bool(field_value > compare_value)
            case ConditionOperator.GREATER_THAN_OR_EQUAL:
                return bool(field_value >= compare_value)
            case ConditionOperator.LESS_THAN:
                return bool(field_value < compare_value)
            case ConditionOperator.LESS_THAN_OR_EQUAL:
                return bool(field_value <= compare_value)
            case ConditionOperator.CONTAINS:
                if isinstance(field_value, str):
                    return str(compare_value) in field_value
                if isinstance(field_value, (list, tuple)):
                    return compare_value in field_value
                return False
            case ConditionOperator.NOT_CONTAINS:
                if isinstance(field_value, str):
                    return str(compare_value) not in field_value
                if isinstance(field_value, (list, tuple)):
                    return compare_value not in field_value
                return True
            case ConditionOperator.STARTS_WITH:
                return isinstance(field_value, str) and field_value.startswith(str(compare_value))
            case ConditionOperator.ENDS_WITH:
                return isinstance(field_value, str) and field_value.endswith(str(compare_value))
            case ConditionOperator.MATCHES:
                if not isinstance(field_value, str):
                    return False
                pattern = str(compare_value)
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(pattern, field_value, flags))
            case ConditionOperator.IN:
                if isinstance(compare_value, (list, tuple)):
                    return field_value in compare_value
                return False
            case ConditionOperator.NOT_IN:
                if isinstance(compare_value, (list, tuple)):
                    return field_value not in compare_value
                return True
            case _:
                return False

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from a nested dictionary using dot notation."""
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    def to_dict(self) -> dict[str, Any]:
        """Serialize condition to dictionary."""
        return {
            "field": self.field,
            "operator": (
                self.operator.value
                if isinstance(self.operator, ConditionOperator)
                else self.operator
            ),
            "value": self.value,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        """Deserialize condition from dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"],
            case_sensitive=data.get("case_sensitive", False),
        )


@dataclass
class Action:
    """
    An action to be executed when a rule matches.

    Attributes:
        type: The type of action
        target: The target for the action (channel, user, webhook URL, etc.)
        params: Additional parameters for the action
    """

    type: ActionType | str
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = ActionType(self.type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize action to dictionary."""
        return {
            "type": self.type.value if isinstance(self.type, ActionType) else self.type,
            "target": self.target,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Deserialize action from dictionary."""
        return cls(
            type=data["type"],
            target=data.get("target"),
            params=data.get("params", {}),
        )


@dataclass
class RoutingRule:
    """
    A routing rule that matches conditions and executes actions.

    Attributes:
        id: Unique identifier for the rule
        name: Human-readable name
        description: Detailed description of what the rule does
        conditions: List of conditions that must all be met (AND logic)
        actions: List of actions to execute when conditions are met
        priority: Higher priority rules are evaluated first
        enabled: Whether the rule is active
        created_at: When the rule was created
        updated_at: When the rule was last updated
        created_by: User who created the rule
        match_mode: "all" requires all conditions, "any" requires at least one
        stop_processing: If True, stop processing further rules after this one matches
        tags: Optional tags for organization
    """

    id: str
    name: str
    conditions: list[Condition]
    actions: list[Action]
    priority: int = 0
    enabled: bool = True
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str | None = None
    match_mode: Literal["all", "any"] = "all"
    stop_processing: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str,
        conditions: list[Condition],
        actions: list[Action],
        **kwargs: Any,
    ) -> RoutingRule:
        """Create a new routing rule with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            conditions=conditions,
            actions=actions,
            **kwargs,
        )

    def matches(self, context: dict[str, Any]) -> bool:
        """
        Check if this rule matches the given context.

        Args:
            context: Dictionary containing deliberation data

        Returns:
            True if the rule matches, False otherwise
        """
        if not self.enabled:
            return False

        if not self.conditions:
            return True  # No conditions means always match

        if self.match_mode == "all":
            return all(cond.evaluate(context) for cond in self.conditions)
        else:  # "any"
            return any(cond.evaluate(context) for cond in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize rule to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "actions": [a.to_dict() for a in self.actions],
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "match_mode": self.match_mode,
            "stop_processing": self.stop_processing,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingRule:
        """Deserialize rule from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.utcnow()
            ),
            created_by=data.get("created_by"),
            match_mode=data.get("match_mode", "all"),
            stop_processing=data.get("stop_processing", False),
            tags=data.get("tags", []),
        )


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a routing rule."""

    rule: RoutingRule
    matched: bool
    actions: list[Action]
    execution_time_ms: float


class RoutingRulesEngine:
    """
    Engine for evaluating routing rules against deliberation context.

    The engine processes rules in priority order and collects all matching
    actions to be executed.

    Example:
        >>> engine = RoutingRulesEngine()
        >>> engine.add_rule(rule1)
        >>> engine.add_rule(rule2)
        >>> actions = engine.evaluate({"confidence": 0.5, "topic": "security"})
    """

    def __init__(self) -> None:
        self._rules: dict[str, RoutingRule] = {}
        self._action_handlers: dict[ActionType, Callable[[Action, dict[str, Any]], None]] = {}

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a rule to the engine."""
        self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> RoutingRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(
        self, enabled_only: bool = False, tags: list[str] | None = None
    ) -> list[RoutingRule]:
        """
        List all rules, optionally filtered.

        Args:
            enabled_only: Only return enabled rules
            tags: Only return rules with these tags

        Returns:
            List of matching rules sorted by priority (descending)
        """
        rules = list(self._rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        if tags:
            rules = [r for r in rules if any(t in r.tags for t in tags)]

        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def register_action_handler(
        self, action_type: ActionType, handler: Callable[[Action, dict[str, Any]], None]
    ) -> None:
        """
        Register a handler for an action type.

        Args:
            action_type: The action type to handle
            handler: Function that takes (action, context) and executes the action
        """
        self._action_handlers[action_type] = handler

    def evaluate(
        self, context: dict[str, Any], execute_actions: bool = False
    ) -> list[RuleEvaluationResult]:
        """
        Evaluate all rules against the context.

        Args:
            context: Dictionary containing deliberation data
            execute_actions: If True, execute the matched actions

        Returns:
            List of evaluation results
        """
        import time

        results: list[RuleEvaluationResult] = []
        rules = self.list_rules(enabled_only=True)

        for rule in rules:
            start_time = time.perf_counter()
            matched = rule.matches(context)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            result = RuleEvaluationResult(
                rule=rule,
                matched=matched,
                actions=rule.actions if matched else [],
                execution_time_ms=execution_time_ms,
            )
            results.append(result)

            if matched:
                if execute_actions:
                    self._execute_actions(rule.actions, context)

                if rule.stop_processing:
                    break

        return results

    def get_matching_actions(self, context: dict[str, Any]) -> list[Action]:
        """
        Get all actions that should be executed for the given context.

        Args:
            context: Dictionary containing deliberation data

        Returns:
            List of actions to execute
        """
        results = self.evaluate(context, execute_actions=False)
        return [action for result in results if result.matched for action in result.actions]

    def _execute_actions(self, actions: list[Action], context: dict[str, Any]) -> None:
        """Execute a list of actions."""
        for action in actions:
            action_type = (
                action.type if isinstance(action.type, ActionType) else ActionType(action.type)
            )
            handler = self._action_handlers.get(action_type)
            if handler:
                handler(action, context)


# Predefined rule templates for common use cases
RULE_TEMPLATES = {
    "low_confidence_escalate": RoutingRule.create(
        name="Escalate Low Confidence Decisions",
        description="Escalate decisions with confidence below 70% for human review",
        conditions=[Condition(field="confidence", operator="lt", value=0.7)],
        actions=[
            Action(type=ActionType.REQUIRE_APPROVAL, target="default"),
            Action(
                type=ActionType.NOTIFY,
                target="admin",
                params={"message": "Low confidence decision needs review"},
            ),
        ],
        priority=100,
        tags=["confidence", "escalation"],
    ),
    "security_topic_route": RoutingRule.create(
        name="Route Security Topics",
        description="Route security-related decisions to the security team channel",
        conditions=[
            Condition(field="topic", operator="contains", value="security"),
        ],
        actions=[
            Action(type=ActionType.ROUTE_TO_CHANNEL, target="security-team"),
            Action(type=ActionType.TAG, target="security"),
        ],
        priority=90,
        tags=["security", "routing"],
    ),
    "agent_dissent_escalate": RoutingRule.create(
        name="Escalate Agent Dissent",
        description="Escalate when agents have significant disagreement",
        conditions=[
            Condition(field="dissent_ratio", operator="gt", value=0.3),
        ],
        actions=[
            Action(type=ActionType.ESCALATE_TO, target="team-lead"),
            Action(
                type=ActionType.LOG,
                params={"level": "warning", "message": "High agent dissent detected"},
            ),
        ],
        priority=80,
        tags=["dissent", "escalation"],
    ),
    "high_priority_fast_track": RoutingRule.create(
        name="Fast Track High Priority",
        description="Fast track high priority decisions",
        conditions=[
            Condition(field="priority", operator="eq", value="critical"),
        ],
        actions=[
            Action(type=ActionType.SET_PRIORITY, target="urgent"),
            Action(
                type=ActionType.NOTIFY, target="all", params={"message": "Critical decision ready"}
            ),
        ],
        priority=95,
        stop_processing=True,
        tags=["priority", "fast-track"],
    ),
}
