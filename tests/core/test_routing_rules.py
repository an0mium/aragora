"""
Tests for core routing rules module.

Tests cover:
- ConditionOperator enum
- ActionType enum
- Condition dataclass and evaluation
- Action dataclass
- RoutingRule dataclass and matching
- RuleEvaluationResult dataclass
- RoutingRulesEngine class
- RULE_TEMPLATES predefined rules
"""

import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock

from aragora.core.routing_rules import (
    ConditionOperator,
    ActionType,
    Condition,
    Action,
    RoutingRule,
    RuleEvaluationResult,
    RoutingRulesEngine,
    RULE_TEMPLATES,
)


# =============================================================================
# ConditionOperator Tests
# =============================================================================


class TestConditionOperator:
    """Tests for ConditionOperator enum."""

    def test_all_operators_defined(self):
        """All expected operators are defined."""
        assert ConditionOperator.EQUALS.value == "eq"
        assert ConditionOperator.NOT_EQUALS.value == "neq"
        assert ConditionOperator.GREATER_THAN.value == "gt"
        assert ConditionOperator.GREATER_THAN_OR_EQUAL.value == "gte"
        assert ConditionOperator.LESS_THAN.value == "lt"
        assert ConditionOperator.LESS_THAN_OR_EQUAL.value == "lte"
        assert ConditionOperator.CONTAINS.value == "contains"
        assert ConditionOperator.NOT_CONTAINS.value == "not_contains"
        assert ConditionOperator.STARTS_WITH.value == "starts_with"
        assert ConditionOperator.ENDS_WITH.value == "ends_with"
        assert ConditionOperator.MATCHES.value == "matches"
        assert ConditionOperator.IN.value == "in"
        assert ConditionOperator.NOT_IN.value == "not_in"
        assert ConditionOperator.EXISTS.value == "exists"
        assert ConditionOperator.NOT_EXISTS.value == "not_exists"

    def test_operator_count(self):
        """Enum has expected number of operators."""
        assert len(ConditionOperator) == 15

    def test_can_create_from_string(self):
        """Can create operator from string value."""
        assert ConditionOperator("eq") == ConditionOperator.EQUALS
        assert ConditionOperator("gt") == ConditionOperator.GREATER_THAN


# =============================================================================
# ActionType Tests
# =============================================================================


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_action_types_defined(self):
        """All expected action types are defined."""
        assert ActionType.ROUTE_TO_CHANNEL.value == "route_to_channel"
        assert ActionType.ESCALATE_TO.value == "escalate_to"
        assert ActionType.NOTIFY.value == "notify"
        assert ActionType.TAG.value == "tag"
        assert ActionType.SET_PRIORITY.value == "set_priority"
        assert ActionType.DELAY.value == "delay"
        assert ActionType.BLOCK.value == "block"
        assert ActionType.REQUIRE_APPROVAL.value == "require_approval"
        assert ActionType.WEBHOOK.value == "webhook"
        assert ActionType.LOG.value == "log"

    def test_action_type_count(self):
        """Enum has expected number of action types."""
        assert len(ActionType) == 10


# =============================================================================
# Condition Tests
# =============================================================================


class TestCondition:
    """Tests for Condition dataclass."""

    def test_create_condition(self):
        """Can create a condition with required fields."""
        condition = Condition(
            field="confidence",
            operator=ConditionOperator.LESS_THAN,
            value=0.7,
        )
        assert condition.field == "confidence"
        assert condition.operator == ConditionOperator.LESS_THAN
        assert condition.value == 0.7
        assert condition.case_sensitive is False

    def test_create_with_string_operator(self):
        """Operator string is converted to enum."""
        condition = Condition(
            field="status",
            operator="eq",
            value="active",
        )
        assert condition.operator == ConditionOperator.EQUALS

    def test_case_sensitive_flag(self):
        """Case sensitive flag can be set."""
        condition = Condition(
            field="name",
            operator="eq",
            value="Test",
            case_sensitive=True,
        )
        assert condition.case_sensitive is True

    # -------------------------------------------------------------------------
    # Equality Operators
    # -------------------------------------------------------------------------

    def test_evaluate_equals_true(self):
        """EQUALS returns True when values match."""
        condition = Condition(field="status", operator="eq", value="active")
        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "ACTIVE"}) is True  # case insensitive

    def test_evaluate_equals_false(self):
        """EQUALS returns False when values differ."""
        condition = Condition(field="status", operator="eq", value="active")
        assert condition.evaluate({"status": "inactive"}) is False

    def test_evaluate_equals_case_sensitive(self):
        """EQUALS respects case_sensitive flag."""
        condition = Condition(
            field="status",
            operator="eq",
            value="Active",
            case_sensitive=True,
        )
        assert condition.evaluate({"status": "Active"}) is True
        assert condition.evaluate({"status": "active"}) is False

    def test_evaluate_not_equals_true(self):
        """NOT_EQUALS returns True when values differ."""
        condition = Condition(field="status", operator="neq", value="inactive")
        assert condition.evaluate({"status": "active"}) is True

    def test_evaluate_not_equals_false(self):
        """NOT_EQUALS returns False when values match."""
        condition = Condition(field="status", operator="neq", value="active")
        assert condition.evaluate({"status": "active"}) is False

    # -------------------------------------------------------------------------
    # Comparison Operators
    # -------------------------------------------------------------------------

    def test_evaluate_greater_than(self):
        """GREATER_THAN compares numeric values."""
        condition = Condition(field="confidence", operator="gt", value=0.7)
        assert condition.evaluate({"confidence": 0.8}) is True
        assert condition.evaluate({"confidence": 0.7}) is False
        assert condition.evaluate({"confidence": 0.6}) is False

    def test_evaluate_greater_than_or_equal(self):
        """GREATER_THAN_OR_EQUAL compares numeric values."""
        condition = Condition(field="confidence", operator="gte", value=0.7)
        assert condition.evaluate({"confidence": 0.8}) is True
        assert condition.evaluate({"confidence": 0.7}) is True
        assert condition.evaluate({"confidence": 0.6}) is False

    def test_evaluate_less_than(self):
        """LESS_THAN compares numeric values."""
        condition = Condition(field="confidence", operator="lt", value=0.7)
        assert condition.evaluate({"confidence": 0.6}) is True
        assert condition.evaluate({"confidence": 0.7}) is False
        assert condition.evaluate({"confidence": 0.8}) is False

    def test_evaluate_less_than_or_equal(self):
        """LESS_THAN_OR_EQUAL compares numeric values."""
        condition = Condition(field="confidence", operator="lte", value=0.7)
        assert condition.evaluate({"confidence": 0.6}) is True
        assert condition.evaluate({"confidence": 0.7}) is True
        assert condition.evaluate({"confidence": 0.8}) is False

    # -------------------------------------------------------------------------
    # String Operators
    # -------------------------------------------------------------------------

    def test_evaluate_contains_string(self):
        """CONTAINS checks if string contains substring."""
        condition = Condition(field="topic", operator="contains", value="security")
        assert condition.evaluate({"topic": "API security review"}) is True
        assert condition.evaluate({"topic": "code review"}) is False

    def test_evaluate_contains_list(self):
        """CONTAINS checks if list contains element."""
        condition = Condition(field="tags", operator="contains", value="urgent")
        assert condition.evaluate({"tags": ["urgent", "security"]}) is True
        assert condition.evaluate({"tags": ["low", "routine"]}) is False

    def test_evaluate_not_contains_string(self):
        """NOT_CONTAINS checks if string does not contain substring."""
        condition = Condition(field="topic", operator="not_contains", value="test")
        assert condition.evaluate({"topic": "production issue"}) is True
        assert condition.evaluate({"topic": "test environment"}) is False

    def test_evaluate_not_contains_list(self):
        """NOT_CONTAINS checks if list does not contain element."""
        condition = Condition(field="tags", operator="not_contains", value="test")
        assert condition.evaluate({"tags": ["production", "urgent"]}) is True
        assert condition.evaluate({"tags": ["test", "dev"]}) is False

    def test_evaluate_starts_with(self):
        """STARTS_WITH checks string prefix."""
        condition = Condition(field="topic", operator="starts_with", value="sec")
        assert condition.evaluate({"topic": "security review"}) is True
        assert condition.evaluate({"topic": "code security"}) is False

    def test_evaluate_ends_with(self):
        """ENDS_WITH checks string suffix."""
        condition = Condition(field="filename", operator="ends_with", value=".py")
        assert condition.evaluate({"filename": "test.py"}) is True
        assert condition.evaluate({"filename": "test.js"}) is False

    def test_evaluate_matches_regex(self):
        """MATCHES evaluates regex pattern."""
        condition = Condition(field="version", operator="matches", value=r"\d+\.\d+\.\d+")
        assert condition.evaluate({"version": "1.2.3"}) is True
        assert condition.evaluate({"version": "v1.2.3"}) is True
        assert condition.evaluate({"version": "latest"}) is False

    def test_evaluate_matches_case_insensitive(self):
        """MATCHES regex is case insensitive by default."""
        condition = Condition(field="name", operator="matches", value=r"test.*")
        assert condition.evaluate({"name": "TestCase"}) is True
        assert condition.evaluate({"name": "TESTING"}) is True

    def test_evaluate_matches_case_sensitive(self):
        """MATCHES regex respects case_sensitive flag."""
        condition = Condition(
            field="name",
            operator="matches",
            value=r"Test.*",
            case_sensitive=True,
        )
        assert condition.evaluate({"name": "TestCase"}) is True
        assert condition.evaluate({"name": "testcase"}) is False

    # -------------------------------------------------------------------------
    # Set Membership Operators
    # -------------------------------------------------------------------------

    def test_evaluate_in(self):
        """IN checks if value is in list."""
        condition = Condition(field="status", operator="in", value=["active", "pending"])
        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "pending"}) is True
        assert condition.evaluate({"status": "closed"}) is False

    def test_evaluate_in_not_list(self):
        """IN returns False when compare value is not a list."""
        condition = Condition(field="status", operator="in", value="active")
        assert condition.evaluate({"status": "active"}) is False

    def test_evaluate_not_in(self):
        """NOT_IN checks if value is not in list."""
        condition = Condition(field="status", operator="not_in", value=["closed", "cancelled"])
        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "closed"}) is False

    def test_evaluate_not_in_not_list(self):
        """NOT_IN returns True when compare value is not a list."""
        condition = Condition(field="status", operator="not_in", value="closed")
        assert condition.evaluate({"status": "active"}) is True

    # -------------------------------------------------------------------------
    # Existence Operators
    # -------------------------------------------------------------------------

    def test_evaluate_exists(self):
        """EXISTS checks if field is present and not None."""
        condition = Condition(field="metadata", operator="exists", value=None)
        assert condition.evaluate({"metadata": {"key": "value"}}) is True
        assert condition.evaluate({"metadata": None}) is False
        assert condition.evaluate({}) is False

    def test_evaluate_not_exists(self):
        """NOT_EXISTS checks if field is absent or None."""
        condition = Condition(field="metadata", operator="not_exists", value=None)
        assert condition.evaluate({}) is True
        assert condition.evaluate({"metadata": None}) is True
        assert condition.evaluate({"metadata": {"key": "value"}}) is False

    # -------------------------------------------------------------------------
    # Nested Field Access
    # -------------------------------------------------------------------------

    def test_evaluate_nested_field(self):
        """Evaluates nested fields using dot notation."""
        condition = Condition(field="consensus.confidence", operator="gt", value=0.8)
        context = {"consensus": {"confidence": 0.9, "reached": True}}
        assert condition.evaluate(context) is True

    def test_evaluate_nested_field_missing(self):
        """Returns False when nested field is missing."""
        condition = Condition(field="consensus.confidence", operator="gt", value=0.8)
        context = {"consensus": {"reached": True}}
        assert condition.evaluate(context) is False

    def test_evaluate_nested_field_parent_missing(self):
        """Returns False when parent of nested field is missing."""
        condition = Condition(field="consensus.confidence", operator="gt", value=0.8)
        context = {"other": {"data": True}}
        assert condition.evaluate(context) is False

    def test_evaluate_deeply_nested_field(self):
        """Evaluates deeply nested fields."""
        condition = Condition(field="result.data.value", operator="eq", value=42)
        context = {"result": {"data": {"value": 42}}}
        assert condition.evaluate(context) is True

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_evaluate_missing_field(self):
        """Returns False when field is missing (except for existence checks)."""
        condition = Condition(field="missing", operator="eq", value="something")
        assert condition.evaluate({}) is False

    def test_evaluate_none_field_value(self):
        """Handles None field values correctly."""
        condition = Condition(field="value", operator="eq", value=None)
        # None field value returns False for equality (field doesn't exist)
        assert condition.evaluate({"value": None}) is False

    def test_evaluate_numeric_strings(self):
        """Handles numeric vs string comparisons."""
        condition = Condition(field="count", operator="gt", value=5)
        assert condition.evaluate({"count": 10}) is True
        # String "10" compares as string, not number
        # This tests the actual behavior (string comparison)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def test_to_dict(self):
        """Serializes condition to dictionary."""
        condition = Condition(
            field="confidence",
            operator=ConditionOperator.LESS_THAN,
            value=0.7,
            case_sensitive=True,
        )
        data = condition.to_dict()

        assert data["field"] == "confidence"
        assert data["operator"] == "lt"
        assert data["value"] == 0.7
        assert data["case_sensitive"] is True

    def test_from_dict(self):
        """Creates condition from dictionary."""
        data = {
            "field": "status",
            "operator": "eq",
            "value": "active",
            "case_sensitive": False,
        }
        condition = Condition.from_dict(data)

        assert condition.field == "status"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value == "active"
        assert condition.case_sensitive is False

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = Condition(
            field="dissent_ratio",
            operator="gte",
            value=0.3,
            case_sensitive=True,
        )
        data = original.to_dict()
        restored = Condition.from_dict(data)

        assert restored.field == original.field
        assert restored.operator == original.operator
        assert restored.value == original.value
        assert restored.case_sensitive == original.case_sensitive


# =============================================================================
# Action Tests
# =============================================================================


class TestAction:
    """Tests for Action dataclass."""

    def test_create_action(self):
        """Can create an action with required fields."""
        action = Action(type=ActionType.NOTIFY, target="admin")
        assert action.type == ActionType.NOTIFY
        assert action.target == "admin"
        assert action.params == {}

    def test_create_with_string_type(self):
        """Type string is converted to enum."""
        action = Action(type="notify", target="admin")
        assert action.type == ActionType.NOTIFY

    def test_create_with_params(self):
        """Can create action with additional params."""
        action = Action(
            type=ActionType.WEBHOOK,
            target="https://example.com/hook",
            params={"method": "POST", "headers": {"Authorization": "Bearer token"}},
        )
        assert action.params["method"] == "POST"
        assert "Authorization" in action.params["headers"]

    def test_to_dict(self):
        """Serializes action to dictionary."""
        action = Action(
            type=ActionType.ESCALATE_TO,
            target="security-team",
            params={"priority": "high"},
        )
        data = action.to_dict()

        assert data["type"] == "escalate_to"
        assert data["target"] == "security-team"
        assert data["params"]["priority"] == "high"

    def test_from_dict(self):
        """Creates action from dictionary."""
        data = {
            "type": "route_to_channel",
            "target": "general",
            "params": {"notify_users": True},
        }
        action = Action.from_dict(data)

        assert action.type == ActionType.ROUTE_TO_CHANNEL
        assert action.target == "general"
        assert action.params["notify_users"] is True

    def test_from_dict_defaults(self):
        """from_dict handles missing optional fields."""
        data = {"type": "block"}
        action = Action.from_dict(data)

        assert action.type == ActionType.BLOCK
        assert action.target is None
        assert action.params == {}

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = Action(
            type=ActionType.LOG,
            target=None,
            params={"level": "warning", "message": "Test log"},
        )
        data = original.to_dict()
        restored = Action.from_dict(data)

        assert restored.type == original.type
        assert restored.target == original.target
        assert restored.params == original.params


# =============================================================================
# RoutingRule Tests
# =============================================================================


class TestRoutingRule:
    """Tests for RoutingRule dataclass."""

    def test_create_rule(self):
        """Can create a rule with required fields."""
        rule = RoutingRule(
            id="rule-1",
            name="Test Rule",
            conditions=[Condition(field="confidence", operator="lt", value=0.7)],
            actions=[Action(type=ActionType.NOTIFY, target="admin")],
        )
        assert rule.id == "rule-1"
        assert rule.name == "Test Rule"
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1

    def test_create_rule_factory(self):
        """RoutingRule.create generates unique IDs."""
        rule1 = RoutingRule.create(
            name="Rule 1",
            conditions=[],
            actions=[],
        )
        rule2 = RoutingRule.create(
            name="Rule 2",
            conditions=[],
            actions=[],
        )
        assert rule1.id != rule2.id
        assert len(rule1.id) == 36  # UUID format

    def test_default_values(self):
        """Default values are sensible."""
        rule = RoutingRule(
            id="rule-1",
            name="Test",
            conditions=[],
            actions=[],
        )
        assert rule.priority == 0
        assert rule.enabled is True
        assert rule.description == ""
        assert rule.match_mode == "all"
        assert rule.stop_processing is False
        assert rule.tags == []
        assert rule.created_by is None

    def test_custom_settings(self):
        """Custom settings can be specified."""
        rule = RoutingRule(
            id="rule-1",
            name="High Priority Rule",
            conditions=[],
            actions=[],
            priority=100,
            enabled=False,
            description="A test rule",
            match_mode="any",
            stop_processing=True,
            tags=["test", "security"],
            created_by="admin",
        )
        assert rule.priority == 100
        assert rule.enabled is False
        assert rule.description == "A test rule"
        assert rule.match_mode == "any"
        assert rule.stop_processing is True
        assert rule.tags == ["test", "security"]
        assert rule.created_by == "admin"

    # -------------------------------------------------------------------------
    # Rule Matching
    # -------------------------------------------------------------------------

    def test_matches_disabled_rule(self):
        """Disabled rules never match."""
        rule = RoutingRule(
            id="rule-1",
            name="Disabled Rule",
            conditions=[Condition(field="confidence", operator="lt", value=0.7)],
            actions=[],
            enabled=False,
        )
        assert rule.matches({"confidence": 0.5}) is False

    def test_matches_no_conditions(self):
        """Rule with no conditions always matches."""
        rule = RoutingRule(
            id="rule-1",
            name="Always Match",
            conditions=[],
            actions=[],
            enabled=True,
        )
        assert rule.matches({}) is True
        assert rule.matches({"any": "data"}) is True

    def test_matches_single_condition(self):
        """Rule with single condition matches correctly."""
        rule = RoutingRule(
            id="rule-1",
            name="Low Confidence",
            conditions=[Condition(field="confidence", operator="lt", value=0.7)],
            actions=[],
        )
        assert rule.matches({"confidence": 0.5}) is True
        assert rule.matches({"confidence": 0.8}) is False

    def test_matches_all_mode(self):
        """match_mode='all' requires all conditions to match."""
        rule = RoutingRule(
            id="rule-1",
            name="Multiple Conditions",
            conditions=[
                Condition(field="confidence", operator="lt", value=0.7),
                Condition(field="status", operator="eq", value="pending"),
            ],
            actions=[],
            match_mode="all",
        )
        # Both conditions match
        assert rule.matches({"confidence": 0.5, "status": "pending"}) is True
        # Only first condition matches
        assert rule.matches({"confidence": 0.5, "status": "active"}) is False
        # Only second condition matches
        assert rule.matches({"confidence": 0.8, "status": "pending"}) is False

    def test_matches_any_mode(self):
        """match_mode='any' requires at least one condition to match."""
        rule = RoutingRule(
            id="rule-1",
            name="Any Match",
            conditions=[
                Condition(field="confidence", operator="lt", value=0.7),
                Condition(field="priority", operator="eq", value="critical"),
            ],
            actions=[],
            match_mode="any",
        )
        # First condition matches
        assert rule.matches({"confidence": 0.5, "priority": "normal"}) is True
        # Second condition matches
        assert rule.matches({"confidence": 0.8, "priority": "critical"}) is True
        # Both match
        assert rule.matches({"confidence": 0.5, "priority": "critical"}) is True
        # Neither matches
        assert rule.matches({"confidence": 0.8, "priority": "normal"}) is False

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def test_to_dict(self):
        """Serializes rule to dictionary."""
        rule = RoutingRule(
            id="rule-1",
            name="Test Rule",
            description="A test rule",
            conditions=[Condition(field="confidence", operator="lt", value=0.7)],
            actions=[Action(type=ActionType.NOTIFY, target="admin")],
            priority=50,
            enabled=True,
            tags=["test"],
        )
        data = rule.to_dict()

        assert data["id"] == "rule-1"
        assert data["name"] == "Test Rule"
        assert data["description"] == "A test rule"
        assert len(data["conditions"]) == 1
        assert len(data["actions"]) == 1
        assert data["priority"] == 50
        assert data["enabled"] is True
        assert data["tags"] == ["test"]
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self):
        """Creates rule from dictionary."""
        data = {
            "id": "rule-2",
            "name": "Imported Rule",
            "description": "An imported rule",
            "conditions": [{"field": "status", "operator": "eq", "value": "active"}],
            "actions": [{"type": "tag", "target": "reviewed"}],
            "priority": 75,
            "enabled": False,
            "match_mode": "any",
            "stop_processing": True,
            "tags": ["imported"],
        }
        rule = RoutingRule.from_dict(data)

        assert rule.id == "rule-2"
        assert rule.name == "Imported Rule"
        assert rule.description == "An imported rule"
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == "status"
        assert len(rule.actions) == 1
        assert rule.actions[0].type == ActionType.TAG
        assert rule.priority == 75
        assert rule.enabled is False
        assert rule.match_mode == "any"
        assert rule.stop_processing is True

    def test_from_dict_with_dates(self):
        """from_dict parses ISO format dates."""
        data = {
            "id": "rule-3",
            "name": "Dated Rule",
            "conditions": [],
            "actions": [],
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-16T14:00:00",
        }
        rule = RoutingRule.from_dict(data)

        assert isinstance(rule.created_at, datetime)
        assert rule.created_at.year == 2024
        assert rule.created_at.month == 1
        assert rule.created_at.day == 15

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = RoutingRule.create(
            name="Roundtrip Test",
            conditions=[
                Condition(field="confidence", operator="lt", value=0.7),
                Condition(field="topic", operator="contains", value="security"),
            ],
            actions=[
                Action(type=ActionType.ESCALATE_TO, target="security-team"),
                Action(type=ActionType.TAG, target="security"),
            ],
            priority=90,
            description="Test rule for roundtrip",
            match_mode="all",
            stop_processing=True,
            tags=["test", "roundtrip"],
        )
        data = original.to_dict()
        restored = RoutingRule.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.conditions) == len(original.conditions)
        assert len(restored.actions) == len(original.actions)
        assert restored.priority == original.priority
        assert restored.match_mode == original.match_mode
        assert restored.stop_processing == original.stop_processing
        assert restored.tags == original.tags


# =============================================================================
# RuleEvaluationResult Tests
# =============================================================================


class TestRuleEvaluationResult:
    """Tests for RuleEvaluationResult dataclass."""

    def test_create_result(self):
        """Can create evaluation result."""
        rule = RoutingRule(
            id="rule-1",
            name="Test",
            conditions=[],
            actions=[Action(type=ActionType.NOTIFY, target="admin")],
        )
        result = RuleEvaluationResult(
            rule=rule,
            matched=True,
            actions=rule.actions,
            execution_time_ms=1.5,
        )

        assert result.rule == rule
        assert result.matched is True
        assert len(result.actions) == 1
        assert result.execution_time_ms == 1.5

    def test_no_actions_on_no_match(self):
        """No actions when rule doesn't match."""
        rule = RoutingRule(
            id="rule-1",
            name="Test",
            conditions=[],
            actions=[Action(type=ActionType.NOTIFY, target="admin")],
        )
        result = RuleEvaluationResult(
            rule=rule,
            matched=False,
            actions=[],
            execution_time_ms=0.5,
        )

        assert result.matched is False
        assert result.actions == []


# =============================================================================
# RoutingRulesEngine Tests
# =============================================================================


class TestRoutingRulesEngine:
    """Tests for RoutingRulesEngine class."""

    def test_init_empty(self):
        """Engine initializes with no rules."""
        engine = RoutingRulesEngine()
        assert engine.list_rules() == []

    def test_add_rule(self):
        """Can add rules to engine."""
        engine = RoutingRulesEngine()
        rule = RoutingRule(
            id="rule-1",
            name="Test",
            conditions=[],
            actions=[],
        )
        engine.add_rule(rule)

        assert len(engine.list_rules()) == 1
        assert engine.get_rule("rule-1") == rule

    def test_add_multiple_rules(self):
        """Can add multiple rules."""
        engine = RoutingRulesEngine()
        rule1 = RoutingRule(id="rule-1", name="First", conditions=[], actions=[])
        rule2 = RoutingRule(id="rule-2", name="Second", conditions=[], actions=[])

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        assert len(engine.list_rules()) == 2

    def test_remove_rule(self):
        """Can remove rules from engine."""
        engine = RoutingRulesEngine()
        rule = RoutingRule(id="rule-1", name="Test", conditions=[], actions=[])
        engine.add_rule(rule)

        result = engine.remove_rule("rule-1")

        assert result is True
        assert engine.get_rule("rule-1") is None
        assert len(engine.list_rules()) == 0

    def test_remove_nonexistent_rule(self):
        """Removing nonexistent rule returns False."""
        engine = RoutingRulesEngine()
        result = engine.remove_rule("nonexistent")
        assert result is False

    def test_get_rule(self):
        """Can retrieve rule by ID."""
        engine = RoutingRulesEngine()
        rule = RoutingRule(id="rule-1", name="Test", conditions=[], actions=[])
        engine.add_rule(rule)

        retrieved = engine.get_rule("rule-1")
        assert retrieved == rule

    def test_get_rule_not_found(self):
        """Returns None for unknown rule ID."""
        engine = RoutingRulesEngine()
        assert engine.get_rule("unknown") is None

    # -------------------------------------------------------------------------
    # list_rules
    # -------------------------------------------------------------------------

    def test_list_rules_sorted_by_priority(self):
        """Rules are sorted by priority (descending)."""
        engine = RoutingRulesEngine()
        engine.add_rule(RoutingRule(id="low", name="Low", conditions=[], actions=[], priority=10))
        engine.add_rule(
            RoutingRule(id="high", name="High", conditions=[], actions=[], priority=100)
        )
        engine.add_rule(
            RoutingRule(id="med", name="Medium", conditions=[], actions=[], priority=50)
        )

        rules = engine.list_rules()

        assert rules[0].id == "high"
        assert rules[1].id == "med"
        assert rules[2].id == "low"

    def test_list_rules_enabled_only(self):
        """Can filter to only enabled rules."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(id="enabled", name="Enabled", conditions=[], actions=[], enabled=True)
        )
        engine.add_rule(
            RoutingRule(id="disabled", name="Disabled", conditions=[], actions=[], enabled=False)
        )

        rules = engine.list_rules(enabled_only=True)

        assert len(rules) == 1
        assert rules[0].id == "enabled"

    def test_list_rules_by_tags(self):
        """Can filter rules by tags."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="security", name="Security", conditions=[], actions=[], tags=["security"]
            )
        )
        engine.add_rule(
            RoutingRule(id="billing", name="Billing", conditions=[], actions=[], tags=["billing"])
        )
        engine.add_rule(
            RoutingRule(
                id="both", name="Both", conditions=[], actions=[], tags=["security", "billing"]
            )
        )

        security_rules = engine.list_rules(tags=["security"])

        assert len(security_rules) == 2
        rule_ids = {r.id for r in security_rules}
        assert "security" in rule_ids
        assert "both" in rule_ids

    def test_list_rules_combined_filters(self):
        """Can combine enabled_only and tags filters."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(id="r1", name="R1", conditions=[], actions=[], enabled=True, tags=["test"])
        )
        engine.add_rule(
            RoutingRule(id="r2", name="R2", conditions=[], actions=[], enabled=False, tags=["test"])
        )
        engine.add_rule(
            RoutingRule(id="r3", name="R3", conditions=[], actions=[], enabled=True, tags=["other"])
        )

        rules = engine.list_rules(enabled_only=True, tags=["test"])

        assert len(rules) == 1
        assert rules[0].id == "r1"

    # -------------------------------------------------------------------------
    # Action Handler Registration
    # -------------------------------------------------------------------------

    def test_register_action_handler(self):
        """Can register action handlers."""
        engine = RoutingRulesEngine()
        handler = MagicMock()

        engine.register_action_handler(ActionType.NOTIFY, handler)

        # Handler is registered (internal state)
        assert ActionType.NOTIFY in engine._action_handlers

    # -------------------------------------------------------------------------
    # evaluate
    # -------------------------------------------------------------------------

    def test_evaluate_no_rules(self):
        """Evaluate returns empty list when no rules."""
        engine = RoutingRulesEngine()
        results = engine.evaluate({"confidence": 0.5})
        assert results == []

    def test_evaluate_matching_rule(self):
        """Evaluate identifies matching rules."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Low Confidence",
                conditions=[Condition(field="confidence", operator="lt", value=0.7)],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
            )
        )

        results = engine.evaluate({"confidence": 0.5})

        assert len(results) == 1
        assert results[0].matched is True
        assert len(results[0].actions) == 1

    def test_evaluate_non_matching_rule(self):
        """Evaluate records non-matching rules."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Low Confidence",
                conditions=[Condition(field="confidence", operator="lt", value=0.7)],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
            )
        )

        results = engine.evaluate({"confidence": 0.9})

        assert len(results) == 1
        assert results[0].matched is False
        assert results[0].actions == []

    def test_evaluate_multiple_rules(self):
        """Evaluate processes multiple rules."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Rule 1",
                conditions=[Condition(field="confidence", operator="lt", value=0.7)],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
                priority=100,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="rule-2",
                name="Rule 2",
                conditions=[Condition(field="topic", operator="contains", value="security")],
                actions=[Action(type=ActionType.TAG, target="security")],
                priority=50,
            )
        )

        # Context matches both rules
        results = engine.evaluate({"confidence": 0.5, "topic": "security review"})

        assert len(results) == 2
        assert results[0].rule.id == "rule-1"  # Higher priority first
        assert results[0].matched is True
        assert results[1].rule.id == "rule-2"
        assert results[1].matched is True

    def test_evaluate_stop_processing(self):
        """Evaluate respects stop_processing flag."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Stop Here",
                conditions=[Condition(field="confidence", operator="lt", value=0.7)],
                actions=[Action(type=ActionType.BLOCK)],
                priority=100,
                stop_processing=True,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="rule-2",
                name="Never Reached",
                conditions=[],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
                priority=50,
            )
        )

        results = engine.evaluate({"confidence": 0.5})

        # Only first rule evaluated (it matched and has stop_processing)
        assert len(results) == 1
        assert results[0].rule.id == "rule-1"

    def test_evaluate_stop_processing_not_matched(self):
        """stop_processing only stops if rule matched."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Won't Match",
                conditions=[Condition(field="confidence", operator="lt", value=0.3)],
                actions=[Action(type=ActionType.BLOCK)],
                priority=100,
                stop_processing=True,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="rule-2",
                name="Will Be Reached",
                conditions=[],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
                priority=50,
            )
        )

        results = engine.evaluate({"confidence": 0.5})

        # Both rules evaluated because first didn't match
        assert len(results) == 2
        assert results[0].matched is False
        assert results[1].matched is True

    def test_evaluate_execution_time_recorded(self):
        """Evaluate records execution time for each rule."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Test",
                conditions=[Condition(field="data", operator="eq", value="test")],
                actions=[],
            )
        )

        results = engine.evaluate({"data": "test"})

        assert results[0].execution_time_ms >= 0

    def test_evaluate_execute_actions(self):
        """Evaluate can execute actions when requested."""
        engine = RoutingRulesEngine()
        handler = MagicMock()
        engine.register_action_handler(ActionType.NOTIFY, handler)

        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Notify Rule",
                conditions=[],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
            )
        )

        engine.evaluate({"data": "test"}, execute_actions=True)

        handler.assert_called_once()

    def test_evaluate_no_execute_actions_by_default(self):
        """Evaluate does not execute actions by default."""
        engine = RoutingRulesEngine()
        handler = MagicMock()
        engine.register_action_handler(ActionType.NOTIFY, handler)

        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Notify Rule",
                conditions=[],
                actions=[Action(type=ActionType.NOTIFY, target="admin")],
            )
        )

        engine.evaluate({"data": "test"})  # execute_actions=False

        handler.assert_not_called()

    # -------------------------------------------------------------------------
    # get_matching_actions
    # -------------------------------------------------------------------------

    def test_get_matching_actions(self):
        """get_matching_actions returns all actions from matching rules."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Rule 1",
                conditions=[Condition(field="confidence", operator="lt", value=0.7)],
                actions=[
                    Action(type=ActionType.NOTIFY, target="admin"),
                    Action(type=ActionType.TAG, target="low-confidence"),
                ],
                priority=100,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="rule-2",
                name="Rule 2",
                conditions=[Condition(field="topic", operator="contains", value="security")],
                actions=[Action(type=ActionType.ESCALATE_TO, target="security-team")],
                priority=50,
            )
        )

        actions = engine.get_matching_actions(
            {
                "confidence": 0.5,
                "topic": "security review",
            }
        )

        assert len(actions) == 3
        action_types = {a.type for a in actions}
        assert ActionType.NOTIFY in action_types
        assert ActionType.TAG in action_types
        assert ActionType.ESCALATE_TO in action_types

    def test_get_matching_actions_empty(self):
        """get_matching_actions returns empty list when no matches."""
        engine = RoutingRulesEngine()
        engine.add_rule(
            RoutingRule(
                id="rule-1",
                name="Rule 1",
                conditions=[Condition(field="confidence", operator="lt", value=0.3)],
                actions=[Action(type=ActionType.BLOCK)],
            )
        )

        actions = engine.get_matching_actions({"confidence": 0.8})

        assert actions == []


# =============================================================================
# RULE_TEMPLATES Tests
# =============================================================================


class TestRuleTemplates:
    """Tests for predefined rule templates."""

    def test_templates_exist(self):
        """All expected templates are defined."""
        assert "low_confidence_escalate" in RULE_TEMPLATES
        assert "security_topic_route" in RULE_TEMPLATES
        assert "agent_dissent_escalate" in RULE_TEMPLATES
        assert "high_priority_fast_track" in RULE_TEMPLATES

    def test_low_confidence_escalate(self):
        """Low confidence escalate template is valid."""
        rule = RULE_TEMPLATES["low_confidence_escalate"]

        assert "confidence" in rule.name.lower() or "low" in rule.name.lower()
        assert len(rule.conditions) >= 1
        assert len(rule.actions) >= 1

        # Should match low confidence
        assert rule.matches({"confidence": 0.5}) is True
        assert rule.matches({"confidence": 0.8}) is False

    def test_security_topic_route(self):
        """Security topic route template is valid."""
        rule = RULE_TEMPLATES["security_topic_route"]

        assert len(rule.conditions) >= 1
        assert len(rule.actions) >= 1

        # Should match security topics
        assert rule.matches({"topic": "API security review"}) is True
        assert rule.matches({"topic": "code refactoring"}) is False

    def test_agent_dissent_escalate(self):
        """Agent dissent escalate template is valid."""
        rule = RULE_TEMPLATES["agent_dissent_escalate"]

        assert len(rule.conditions) >= 1
        assert len(rule.actions) >= 1

        # Should match high dissent
        assert rule.matches({"dissent_ratio": 0.5}) is True
        assert rule.matches({"dissent_ratio": 0.1}) is False

    def test_high_priority_fast_track(self):
        """High priority fast track template is valid."""
        rule = RULE_TEMPLATES["high_priority_fast_track"]

        assert len(rule.conditions) >= 1
        assert len(rule.actions) >= 1
        assert rule.stop_processing is True

        # Should match critical priority
        assert rule.matches({"priority": "critical"}) is True
        assert rule.matches({"priority": "normal"}) is False

    def test_templates_have_tags(self):
        """All templates have appropriate tags."""
        for name, rule in RULE_TEMPLATES.items():
            assert len(rule.tags) > 0, f"Template {name} should have tags"

    def test_templates_have_priorities(self):
        """All templates have non-zero priorities."""
        for name, rule in RULE_TEMPLATES.items():
            assert rule.priority > 0, f"Template {name} should have priority > 0"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRoutingRulesIntegration:
    """Integration tests for the routing rules system."""

    def test_full_workflow(self):
        """Tests a complete routing workflow."""
        # Create engine
        engine = RoutingRulesEngine()

        # Register action handlers
        logged_messages = []

        def log_handler(action: Action, context: dict):
            logged_messages.append(f"LOG: {action.params.get('message', 'no message')}")

        def notify_handler(action: Action, context: dict):
            logged_messages.append(f"NOTIFY {action.target}: {action.params.get('message', '')}")

        engine.register_action_handler(ActionType.LOG, log_handler)
        engine.register_action_handler(ActionType.NOTIFY, notify_handler)

        # Add rules
        engine.add_rule(
            RoutingRule.create(
                name="Critical Alert",
                conditions=[
                    Condition(field="confidence", operator="lt", value=0.5),
                    Condition(field="topic", operator="contains", value="security"),
                ],
                actions=[
                    Action(
                        type=ActionType.NOTIFY,
                        target="security-team",
                        params={"message": "Critical security issue"},
                    ),
                    Action(
                        type=ActionType.LOG,
                        params={"level": "critical", "message": "Security alert triggered"},
                    ),
                ],
                priority=100,
                match_mode="all",
            )
        )

        engine.add_rule(
            RoutingRule.create(
                name="Low Confidence Warning",
                conditions=[Condition(field="confidence", operator="lt", value=0.6)],
                actions=[
                    Action(type=ActionType.LOG, params={"message": "Low confidence detected"}),
                ],
                priority=50,
            )
        )

        # Evaluate context that matches both rules
        context = {
            "confidence": 0.4,
            "topic": "API security vulnerability",
            "timestamp": "2024-01-15T10:00:00",
        }

        results = engine.evaluate(context, execute_actions=True)

        # Both rules should match
        assert len(results) == 2
        assert all(r.matched for r in results)

        # Actions should have been executed
        assert len(logged_messages) == 3  # 2 from first rule, 1 from second
        assert any("security-team" in msg for msg in logged_messages)
        assert any("Low confidence" in msg for msg in logged_messages)

    def test_rule_priority_ordering(self):
        """Tests that rules are evaluated in priority order."""
        engine = RoutingRulesEngine()
        evaluation_order = []

        def tracking_handler(action: Action, context: dict):
            evaluation_order.append(action.target)

        engine.register_action_handler(ActionType.TAG, tracking_handler)

        # Add rules with different priorities
        engine.add_rule(
            RoutingRule(
                id="low",
                name="Low Priority",
                conditions=[],
                actions=[Action(type=ActionType.TAG, target="low")],
                priority=10,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="high",
                name="High Priority",
                conditions=[],
                actions=[Action(type=ActionType.TAG, target="high")],
                priority=100,
            )
        )
        engine.add_rule(
            RoutingRule(
                id="medium",
                name="Medium Priority",
                conditions=[],
                actions=[Action(type=ActionType.TAG, target="medium")],
                priority=50,
            )
        )

        engine.evaluate({}, execute_actions=True)

        # Actions should be executed in priority order (high to low)
        assert evaluation_order == ["high", "medium", "low"]

    def test_complex_conditions(self):
        """Tests complex condition combinations."""
        engine = RoutingRulesEngine()

        engine.add_rule(
            RoutingRule.create(
                name="Complex Rule",
                conditions=[
                    Condition(field="result.confidence", operator="lt", value=0.7),
                    Condition(field="result.consensus.reached", operator="eq", value=True),
                    Condition(field="tags", operator="contains", value="production"),
                    Condition(field="requester", operator="not_in", value=["bot", "system"]),
                ],
                actions=[Action(type=ActionType.REQUIRE_APPROVAL)],
                match_mode="all",
            )
        )

        # Context that matches all conditions
        matching_context = {
            "result": {
                "confidence": 0.5,
                "consensus": {"reached": True},
            },
            "tags": ["production", "release"],
            "requester": "user123",
        }

        # Context that fails one condition
        non_matching_context = {
            "result": {
                "confidence": 0.5,
                "consensus": {"reached": True},
            },
            "tags": ["staging"],  # Doesn't contain "production"
            "requester": "user123",
        }

        matching_results = engine.evaluate(matching_context)
        non_matching_results = engine.evaluate(non_matching_context)

        assert matching_results[0].matched is True
        assert non_matching_results[0].matched is False
