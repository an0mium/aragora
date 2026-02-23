"""Tests for Shared Inbox data models (aragora/server/handlers/shared_inbox/models.py).

Covers all enums, dataclasses, and serialization methods:

Enums:
- MessageStatus: OPEN, ASSIGNED, IN_PROGRESS, WAITING, RESOLVED, CLOSED
- RuleConditionField: FROM, TO, SUBJECT, BODY, LABELS, PRIORITY, SENDER_DOMAIN
- RuleConditionOperator: CONTAINS, EQUALS, STARTS_WITH, ENDS_WITH, MATCHES, GREATER_THAN, LESS_THAN
- RuleActionType: ASSIGN, LABEL, ESCALATE, ARCHIVE, NOTIFY, FORWARD

Dataclasses:
- RuleCondition: field, operator, value; to_dict(), from_dict()
- RuleAction: type, target, params; to_dict(), from_dict()
- RoutingRule: full rule with conditions/actions; to_dict(), from_dict()
- SharedInboxMessage: message with collaboration metadata; to_dict()
- SharedInbox: inbox configuration; to_dict()

Test categories per class:
- Construction / defaults
- to_dict() round-trip fidelity
- from_dict() happy path and defaults
- from_dict() edge cases (missing optional fields, extra keys)
- Enum value matching (str mixin)
- Error cases (invalid enum values, missing required fields)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from aragora.server.handlers.shared_inbox.models import (
    MessageStatus,
    RuleActionType,
    RuleConditionField,
    RuleConditionOperator,
    RuleAction,
    RuleCondition,
    RoutingRule,
    SharedInbox,
    SharedInboxMessage,
)


# ===========================================================================
# MessageStatus Enum
# ===========================================================================


class TestMessageStatus:
    """Tests for the MessageStatus enum."""

    def test_all_values(self):
        """All six status values are defined."""
        assert MessageStatus.OPEN.value == "open"
        assert MessageStatus.ASSIGNED.value == "assigned"
        assert MessageStatus.IN_PROGRESS.value == "in_progress"
        assert MessageStatus.WAITING.value == "waiting"
        assert MessageStatus.RESOLVED.value == "resolved"
        assert MessageStatus.CLOSED.value == "closed"

    def test_member_count(self):
        """Exactly six members exist."""
        assert len(MessageStatus) == 6

    def test_str_mixin(self):
        """MessageStatus is a str enum -- comparing to its value works."""
        assert MessageStatus.OPEN == "open"
        assert MessageStatus.CLOSED == "closed"

    def test_construction_from_value(self):
        """Can construct from string value."""
        assert MessageStatus("open") is MessageStatus.OPEN
        assert MessageStatus("in_progress") is MessageStatus.IN_PROGRESS

    def test_invalid_value_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            MessageStatus("invalid_status")

    def test_membership(self):
        """All values are members of the enum."""
        values = {"open", "assigned", "in_progress", "waiting", "resolved", "closed"}
        assert {m.value for m in MessageStatus} == values

    def test_identity(self):
        """Enum members are singletons."""
        assert MessageStatus("open") is MessageStatus("open")


# ===========================================================================
# RuleConditionField Enum
# ===========================================================================


class TestRuleConditionField:
    """Tests for the RuleConditionField enum."""

    def test_all_values(self):
        assert RuleConditionField.FROM.value == "from"
        assert RuleConditionField.TO.value == "to"
        assert RuleConditionField.SUBJECT.value == "subject"
        assert RuleConditionField.BODY.value == "body"
        assert RuleConditionField.LABELS.value == "labels"
        assert RuleConditionField.PRIORITY.value == "priority"
        assert RuleConditionField.SENDER_DOMAIN.value == "sender_domain"

    def test_member_count(self):
        assert len(RuleConditionField) == 7

    def test_str_mixin(self):
        assert RuleConditionField.FROM == "from"
        assert RuleConditionField.SENDER_DOMAIN == "sender_domain"

    def test_construction_from_value(self):
        assert RuleConditionField("subject") is RuleConditionField.SUBJECT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RuleConditionField("nonexistent")


# ===========================================================================
# RuleConditionOperator Enum
# ===========================================================================


class TestRuleConditionOperator:
    """Tests for the RuleConditionOperator enum."""

    def test_all_values(self):
        assert RuleConditionOperator.CONTAINS.value == "contains"
        assert RuleConditionOperator.EQUALS.value == "equals"
        assert RuleConditionOperator.STARTS_WITH.value == "starts_with"
        assert RuleConditionOperator.ENDS_WITH.value == "ends_with"
        assert RuleConditionOperator.MATCHES.value == "matches"
        assert RuleConditionOperator.GREATER_THAN.value == "greater_than"
        assert RuleConditionOperator.LESS_THAN.value == "less_than"

    def test_member_count(self):
        assert len(RuleConditionOperator) == 7

    def test_str_mixin(self):
        assert RuleConditionOperator.CONTAINS == "contains"

    def test_construction_from_value(self):
        assert RuleConditionOperator("starts_with") is RuleConditionOperator.STARTS_WITH

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RuleConditionOperator("not_a_valid_op")


# ===========================================================================
# RuleActionType Enum
# ===========================================================================


class TestRuleActionType:
    """Tests for the RuleActionType enum."""

    def test_all_values(self):
        assert RuleActionType.ASSIGN.value == "assign"
        assert RuleActionType.LABEL.value == "label"
        assert RuleActionType.ESCALATE.value == "escalate"
        assert RuleActionType.ARCHIVE.value == "archive"
        assert RuleActionType.NOTIFY.value == "notify"
        assert RuleActionType.FORWARD.value == "forward"

    def test_member_count(self):
        assert len(RuleActionType) == 6

    def test_str_mixin(self):
        assert RuleActionType.ASSIGN == "assign"

    def test_construction_from_value(self):
        assert RuleActionType("forward") is RuleActionType.FORWARD

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RuleActionType("explode")


# ===========================================================================
# RuleCondition Dataclass
# ===========================================================================


class TestRuleCondition:
    """Tests for the RuleCondition dataclass."""

    def test_construction(self):
        cond = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="urgent",
        )
        assert cond.field is RuleConditionField.SUBJECT
        assert cond.operator is RuleConditionOperator.CONTAINS
        assert cond.value == "urgent"

    def test_to_dict(self):
        cond = RuleCondition(
            field=RuleConditionField.FROM,
            operator=RuleConditionOperator.EQUALS,
            value="boss@example.com",
        )
        d = cond.to_dict()
        assert d == {
            "field": "from",
            "operator": "equals",
            "value": "boss@example.com",
        }

    def test_from_dict_happy(self):
        data = {"field": "body", "operator": "matches", "value": "^error.*"}
        cond = RuleCondition.from_dict(data)
        assert cond.field is RuleConditionField.BODY
        assert cond.operator is RuleConditionOperator.MATCHES
        assert cond.value == "^error.*"

    def test_round_trip(self):
        """to_dict -> from_dict produces an equivalent object."""
        original = RuleCondition(
            field=RuleConditionField.SENDER_DOMAIN,
            operator=RuleConditionOperator.ENDS_WITH,
            value=".gov",
        )
        rebuilt = RuleCondition.from_dict(original.to_dict())
        assert rebuilt.field == original.field
        assert rebuilt.operator == original.operator
        assert rebuilt.value == original.value

    def test_from_dict_invalid_field_raises(self):
        with pytest.raises(ValueError):
            RuleCondition.from_dict(
                {"field": "nope", "operator": "contains", "value": "x"}
            )

    def test_from_dict_invalid_operator_raises(self):
        with pytest.raises(ValueError):
            RuleCondition.from_dict(
                {"field": "subject", "operator": "nope", "value": "x"}
            )

    def test_from_dict_missing_field_raises(self):
        with pytest.raises(KeyError):
            RuleCondition.from_dict({"operator": "contains", "value": "x"})

    def test_from_dict_missing_operator_raises(self):
        with pytest.raises(KeyError):
            RuleCondition.from_dict({"field": "subject", "value": "x"})

    def test_from_dict_missing_value_raises(self):
        with pytest.raises(KeyError):
            RuleCondition.from_dict({"field": "subject", "operator": "contains"})

    def test_empty_value_allowed(self):
        """Empty string value is valid (no constraint in the model)."""
        cond = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="",
        )
        assert cond.value == ""
        d = cond.to_dict()
        assert d["value"] == ""

    def test_all_field_operator_combinations(self):
        """All fields and operators can be used together."""
        for fld in RuleConditionField:
            for op in RuleConditionOperator:
                cond = RuleCondition(field=fld, operator=op, value="test")
                d = cond.to_dict()
                assert d["field"] == fld.value
                assert d["operator"] == op.value

    def test_from_dict_extra_keys_ignored(self):
        """Extra keys in the dict do not cause errors (dict access is explicit)."""
        data = {
            "field": "subject",
            "operator": "contains",
            "value": "urgent",
            "extra_key": "should be ignored",
        }
        cond = RuleCondition.from_dict(data)
        assert cond.value == "urgent"


# ===========================================================================
# RuleAction Dataclass
# ===========================================================================


class TestRuleAction:
    """Tests for the RuleAction dataclass."""

    def test_construction_with_all_fields(self):
        action = RuleAction(
            type=RuleActionType.ASSIGN,
            target="user-123",
            params={"priority": "high"},
        )
        assert action.type is RuleActionType.ASSIGN
        assert action.target == "user-123"
        assert action.params == {"priority": "high"}

    def test_defaults(self):
        action = RuleAction(type=RuleActionType.ARCHIVE)
        assert action.target is None
        assert action.params == {}

    def test_to_dict(self):
        action = RuleAction(
            type=RuleActionType.NOTIFY,
            target="slack-channel",
            params={"message": "New email"},
        )
        d = action.to_dict()
        assert d == {
            "type": "notify",
            "target": "slack-channel",
            "params": {"message": "New email"},
        }

    def test_to_dict_none_target(self):
        action = RuleAction(type=RuleActionType.ARCHIVE)
        d = action.to_dict()
        assert d["target"] is None
        assert d["params"] == {}

    def test_from_dict_happy(self):
        data = {
            "type": "forward",
            "target": "other-inbox",
            "params": {"cc": "admin@co.com"},
        }
        action = RuleAction.from_dict(data)
        assert action.type is RuleActionType.FORWARD
        assert action.target == "other-inbox"
        assert action.params == {"cc": "admin@co.com"}

    def test_from_dict_defaults(self):
        """Missing target and params get defaults."""
        data = {"type": "archive"}
        action = RuleAction.from_dict(data)
        assert action.target is None
        assert action.params == {}

    def test_round_trip(self):
        original = RuleAction(
            type=RuleActionType.ESCALATE,
            target="manager-team",
            params={"urgency": "critical"},
        )
        rebuilt = RuleAction.from_dict(original.to_dict())
        assert rebuilt.type == original.type
        assert rebuilt.target == original.target
        assert rebuilt.params == original.params

    def test_from_dict_invalid_type_raises(self):
        with pytest.raises(ValueError):
            RuleAction.from_dict({"type": "self_destruct"})

    def test_from_dict_missing_type_raises(self):
        with pytest.raises(KeyError):
            RuleAction.from_dict({"target": "someone"})

    def test_all_action_types(self):
        """All action types can be serialized and deserialized."""
        for at in RuleActionType:
            action = RuleAction(type=at, target="t")
            d = action.to_dict()
            assert d["type"] == at.value
            rebuilt = RuleAction.from_dict(d)
            assert rebuilt.type is at

    def test_empty_params(self):
        action = RuleAction(type=RuleActionType.LABEL, params={})
        assert action.to_dict()["params"] == {}

    def test_params_default_factory_independence(self):
        """Each instance gets its own params dict (default_factory)."""
        a1 = RuleAction(type=RuleActionType.LABEL)
        a2 = RuleAction(type=RuleActionType.LABEL)
        a1.params["key"] = "value"
        assert "key" not in a2.params


# ===========================================================================
# RoutingRule Dataclass
# ===========================================================================


def _make_condition(**overrides: Any) -> RuleCondition:
    """Build a valid RuleCondition with optional overrides."""
    defaults = {
        "field": RuleConditionField.SUBJECT,
        "operator": RuleConditionOperator.CONTAINS,
        "value": "urgent",
    }
    defaults.update(overrides)
    return RuleCondition(**defaults)


def _make_action(**overrides: Any) -> RuleAction:
    """Build a valid RuleAction with optional overrides."""
    defaults = {"type": RuleActionType.ASSIGN, "target": "user-1"}
    defaults.update(overrides)
    return RuleAction(**defaults)


def _make_routing_rule(**overrides: Any) -> RoutingRule:
    """Build a valid RoutingRule with optional overrides."""
    defaults: dict[str, Any] = {
        "id": "rule-001",
        "name": "Urgent emails",
        "workspace_id": "ws-001",
        "conditions": [_make_condition()],
        "condition_logic": "AND",
        "actions": [_make_action()],
    }
    defaults.update(overrides)
    return RoutingRule(**defaults)


class TestRoutingRule:
    """Tests for the RoutingRule dataclass."""

    def test_construction_minimal(self):
        rule = _make_routing_rule()
        assert rule.id == "rule-001"
        assert rule.name == "Urgent emails"
        assert rule.workspace_id == "ws-001"
        assert len(rule.conditions) == 1
        assert rule.condition_logic == "AND"
        assert len(rule.actions) == 1

    def test_defaults(self):
        rule = _make_routing_rule()
        assert rule.priority == 5
        assert rule.enabled is True
        assert rule.description is None
        assert rule.created_by is None
        assert rule.stats == {}
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_custom_priority(self):
        rule = _make_routing_rule(priority=1)
        assert rule.priority == 1

    def test_disabled_rule(self):
        rule = _make_routing_rule(enabled=False)
        assert rule.enabled is False

    def test_to_dict_structure(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        rule = RoutingRule(
            id="rule-100",
            name="Test Rule",
            workspace_id="ws-200",
            conditions=[_make_condition()],
            condition_logic="OR",
            actions=[_make_action()],
            priority=3,
            enabled=False,
            description="A test rule",
            created_at=now,
            updated_at=now,
            created_by="admin-1",
            stats={"matched": 42},
        )
        d = rule.to_dict()
        assert d["id"] == "rule-100"
        assert d["name"] == "Test Rule"
        assert d["workspace_id"] == "ws-200"
        assert d["condition_logic"] == "OR"
        assert d["priority"] == 3
        assert d["enabled"] is False
        assert d["description"] == "A test rule"
        assert d["created_at"] == now.isoformat()
        assert d["updated_at"] == now.isoformat()
        assert d["created_by"] == "admin-1"
        assert d["stats"] == {"matched": 42}
        # Nested conditions and actions are dicts
        assert len(d["conditions"]) == 1
        assert d["conditions"][0]["field"] == "subject"
        assert len(d["actions"]) == 1
        assert d["actions"][0]["type"] == "assign"

    def test_to_dict_timestamps_are_iso_strings(self):
        rule = _make_routing_rule()
        d = rule.to_dict()
        # Should be parseable ISO format
        datetime.fromisoformat(d["created_at"])
        datetime.fromisoformat(d["updated_at"])

    def test_from_dict_happy(self):
        now_str = "2025-06-15T12:00:00+00:00"
        data = {
            "id": "rule-500",
            "name": "From dict test",
            "workspace_id": "ws-300",
            "conditions": [
                {"field": "from", "operator": "equals", "value": "ceo@co.com"}
            ],
            "condition_logic": "AND",
            "actions": [{"type": "escalate", "target": "mgr-team"}],
            "priority": 1,
            "enabled": True,
            "description": "Important",
            "created_at": now_str,
            "updated_at": now_str,
            "created_by": "system",
            "stats": {"runs": 10},
        }
        rule = RoutingRule.from_dict(data)
        assert rule.id == "rule-500"
        assert rule.name == "From dict test"
        assert rule.workspace_id == "ws-300"
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field is RuleConditionField.FROM
        assert rule.condition_logic == "AND"
        assert len(rule.actions) == 1
        assert rule.actions[0].type is RuleActionType.ESCALATE
        assert rule.priority == 1
        assert rule.enabled is True
        assert rule.description == "Important"
        assert rule.created_by == "system"
        assert rule.stats == {"runs": 10}
        assert rule.created_at == datetime.fromisoformat(now_str)

    def test_from_dict_defaults(self):
        """Missing optional fields use sensible defaults."""
        data = {
            "id": "rule-min",
            "name": "Minimal",
            "workspace_id": "ws-1",
        }
        rule = RoutingRule.from_dict(data)
        assert rule.conditions == []
        assert rule.condition_logic == "AND"
        assert rule.actions == []
        assert rule.priority == 5
        assert rule.enabled is True
        assert rule.description is None
        assert rule.created_by is None
        assert rule.stats == {}
        # Timestamps should be auto-generated
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_from_dict_no_timestamps(self):
        """When timestamps are absent, defaults are generated."""
        data = {
            "id": "rule-no-ts",
            "name": "No timestamps",
            "workspace_id": "ws-1",
        }
        rule = RoutingRule.from_dict(data)
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_from_dict_null_timestamps(self):
        """Explicit None timestamps also fall back to now()."""
        data = {
            "id": "rule-null-ts",
            "name": "Null timestamps",
            "workspace_id": "ws-1",
            "created_at": None,
            "updated_at": None,
        }
        rule = RoutingRule.from_dict(data)
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_from_dict_empty_string_timestamp(self):
        """Empty string timestamp also falls back to now() (falsy)."""
        data = {
            "id": "rule-empty-ts",
            "name": "Empty timestamps",
            "workspace_id": "ws-1",
            "created_at": "",
            "updated_at": "",
        }
        rule = RoutingRule.from_dict(data)
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_round_trip(self):
        now = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        original = RoutingRule(
            id="rule-rt",
            name="Round Trip",
            workspace_id="ws-rt",
            conditions=[
                _make_condition(field=RuleConditionField.FROM),
                _make_condition(
                    field=RuleConditionField.BODY,
                    operator=RuleConditionOperator.MATCHES,
                    value=".*password.*",
                ),
            ],
            condition_logic="OR",
            actions=[
                _make_action(type=RuleActionType.LABEL, target="security"),
                _make_action(type=RuleActionType.NOTIFY, target="sec-team"),
            ],
            priority=1,
            enabled=True,
            description="Security rule",
            created_at=now,
            updated_at=now,
            created_by="admin",
            stats={"matched": 5, "last_run": "2025-01-02"},
        )
        rebuilt = RoutingRule.from_dict(original.to_dict())
        assert rebuilt.id == original.id
        assert rebuilt.name == original.name
        assert rebuilt.workspace_id == original.workspace_id
        assert len(rebuilt.conditions) == len(original.conditions)
        for orig_c, new_c in zip(original.conditions, rebuilt.conditions):
            assert new_c.field == orig_c.field
            assert new_c.operator == orig_c.operator
            assert new_c.value == orig_c.value
        assert rebuilt.condition_logic == original.condition_logic
        assert len(rebuilt.actions) == len(original.actions)
        for orig_a, new_a in zip(original.actions, rebuilt.actions):
            assert new_a.type == orig_a.type
            assert new_a.target == orig_a.target
        assert rebuilt.priority == original.priority
        assert rebuilt.enabled == original.enabled
        assert rebuilt.description == original.description
        assert rebuilt.created_at == original.created_at
        assert rebuilt.updated_at == original.updated_at
        assert rebuilt.created_by == original.created_by
        assert rebuilt.stats == original.stats

    def test_from_dict_missing_id_raises(self):
        with pytest.raises(KeyError):
            RoutingRule.from_dict({"name": "No ID", "workspace_id": "ws"})

    def test_from_dict_missing_name_raises(self):
        with pytest.raises(KeyError):
            RoutingRule.from_dict({"id": "r1", "workspace_id": "ws"})

    def test_from_dict_missing_workspace_id_raises(self):
        with pytest.raises(KeyError):
            RoutingRule.from_dict({"id": "r1", "name": "X"})

    def test_multiple_conditions(self):
        rule = _make_routing_rule(
            conditions=[
                _make_condition(field=RuleConditionField.SUBJECT),
                _make_condition(field=RuleConditionField.FROM),
                _make_condition(field=RuleConditionField.PRIORITY),
            ]
        )
        assert len(rule.conditions) == 3
        d = rule.to_dict()
        assert len(d["conditions"]) == 3

    def test_multiple_actions(self):
        rule = _make_routing_rule(
            actions=[
                _make_action(type=RuleActionType.ASSIGN),
                _make_action(type=RuleActionType.LABEL, target="vip"),
                _make_action(type=RuleActionType.NOTIFY, target="admin"),
            ]
        )
        assert len(rule.actions) == 3
        d = rule.to_dict()
        assert len(d["actions"]) == 3

    def test_empty_conditions_and_actions(self):
        rule = _make_routing_rule(conditions=[], actions=[])
        d = rule.to_dict()
        assert d["conditions"] == []
        assert d["actions"] == []

    def test_stats_default_factory_independence(self):
        """Each instance gets its own stats dict."""
        r1 = _make_routing_rule()
        r2 = _make_routing_rule()
        r1.stats["counter"] = 1
        assert "counter" not in r2.stats

    def test_condition_logic_or(self):
        rule = _make_routing_rule(condition_logic="OR")
        assert rule.condition_logic == "OR"
        d = rule.to_dict()
        assert d["condition_logic"] == "OR"

    def test_from_dict_condition_logic_default(self):
        data = {"id": "r1", "name": "X", "workspace_id": "ws"}
        rule = RoutingRule.from_dict(data)
        assert rule.condition_logic == "AND"

    def test_from_dict_extra_keys_ignored(self):
        data = {
            "id": "r1",
            "name": "X",
            "workspace_id": "ws",
            "unknown_field": "should not crash",
        }
        rule = RoutingRule.from_dict(data)
        assert rule.id == "r1"

    def test_priority_zero(self):
        rule = _make_routing_rule(priority=0)
        assert rule.priority == 0
        assert rule.to_dict()["priority"] == 0

    def test_negative_priority(self):
        rule = _make_routing_rule(priority=-1)
        assert rule.priority == -1

    def test_high_priority(self):
        rule = _make_routing_rule(priority=100)
        assert rule.priority == 100


# ===========================================================================
# SharedInboxMessage Dataclass
# ===========================================================================


def _make_message(**overrides: Any) -> SharedInboxMessage:
    """Build a valid SharedInboxMessage with optional overrides."""
    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    defaults: dict[str, Any] = {
        "id": "msg-001",
        "inbox_id": "inbox-001",
        "email_id": "email-001",
        "subject": "Hello",
        "from_address": "sender@example.com",
        "to_addresses": ["team@example.com"],
        "snippet": "This is a test email...",
        "received_at": now,
    }
    defaults.update(overrides)
    return SharedInboxMessage(**defaults)


class TestSharedInboxMessage:
    """Tests for the SharedInboxMessage dataclass."""

    def test_construction_minimal(self):
        msg = _make_message()
        assert msg.id == "msg-001"
        assert msg.inbox_id == "inbox-001"
        assert msg.email_id == "email-001"
        assert msg.subject == "Hello"
        assert msg.from_address == "sender@example.com"
        assert msg.to_addresses == ["team@example.com"]
        assert msg.snippet == "This is a test email..."

    def test_defaults(self):
        msg = _make_message()
        assert msg.status is MessageStatus.OPEN
        assert msg.assigned_to is None
        assert msg.assigned_at is None
        assert msg.tags == []
        assert msg.priority is None
        assert msg.notes == []
        assert msg.thread_id is None
        assert msg.sla_deadline is None
        assert msg.resolved_at is None
        assert msg.resolved_by is None

    def test_assigned_message(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        msg = _make_message(
            status=MessageStatus.ASSIGNED,
            assigned_to="user-42",
            assigned_at=now,
        )
        assert msg.status is MessageStatus.ASSIGNED
        assert msg.assigned_to == "user-42"
        assert msg.assigned_at == now

    def test_resolved_message(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        msg = _make_message(
            status=MessageStatus.RESOLVED,
            resolved_at=now,
            resolved_by="user-99",
        )
        assert msg.status is MessageStatus.RESOLVED
        assert msg.resolved_at == now
        assert msg.resolved_by == "user-99"

    def test_to_dict_all_none_optionals(self):
        msg = _make_message()
        d = msg.to_dict()
        assert d["id"] == "msg-001"
        assert d["inbox_id"] == "inbox-001"
        assert d["email_id"] == "email-001"
        assert d["subject"] == "Hello"
        assert d["from_address"] == "sender@example.com"
        assert d["to_addresses"] == ["team@example.com"]
        assert d["snippet"] == "This is a test email..."
        assert d["received_at"] == "2025-06-15T12:00:00+00:00"
        assert d["status"] == "open"
        assert d["assigned_to"] is None
        assert d["assigned_at"] is None
        assert d["tags"] == []
        assert d["priority"] is None
        assert d["notes"] == []
        assert d["thread_id"] is None
        assert d["sla_deadline"] is None
        assert d["resolved_at"] is None
        assert d["resolved_by"] is None

    def test_to_dict_with_all_fields(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        deadline = datetime(2025, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
        msg = SharedInboxMessage(
            id="msg-full",
            inbox_id="inbox-full",
            email_id="email-full",
            subject="Full message",
            from_address="a@b.com",
            to_addresses=["c@d.com", "e@f.com"],
            snippet="snippet text",
            received_at=now,
            status=MessageStatus.IN_PROGRESS,
            assigned_to="user-5",
            assigned_at=now,
            tags=["vip", "urgent"],
            priority="high",
            notes=[{"author": "user-5", "text": "Working on it"}],
            thread_id="thread-100",
            sla_deadline=deadline,
            resolved_at=now,
            resolved_by="user-5",
        )
        d = msg.to_dict()
        assert d["status"] == "in_progress"
        assert d["assigned_to"] == "user-5"
        assert d["assigned_at"] == now.isoformat()
        assert d["tags"] == ["vip", "urgent"]
        assert d["priority"] == "high"
        assert d["notes"] == [{"author": "user-5", "text": "Working on it"}]
        assert d["thread_id"] == "thread-100"
        assert d["sla_deadline"] == deadline.isoformat()
        assert d["resolved_at"] == now.isoformat()
        assert d["resolved_by"] == "user-5"

    def test_to_dict_datetime_formats(self):
        now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        msg = _make_message(
            received_at=now,
            assigned_at=now,
            sla_deadline=now,
            resolved_at=now,
        )
        d = msg.to_dict()
        for key in ["received_at", "assigned_at", "sla_deadline", "resolved_at"]:
            assert isinstance(d[key], str)
            parsed = datetime.fromisoformat(d[key])
            assert parsed == now

    def test_multiple_to_addresses(self):
        msg = _make_message(to_addresses=["a@b.com", "c@d.com", "e@f.com"])
        assert len(msg.to_addresses) == 3
        d = msg.to_dict()
        assert d["to_addresses"] == ["a@b.com", "c@d.com", "e@f.com"]

    def test_empty_to_addresses(self):
        msg = _make_message(to_addresses=[])
        assert msg.to_addresses == []
        assert msg.to_dict()["to_addresses"] == []

    def test_tags_default_factory_independence(self):
        m1 = _make_message()
        m2 = _make_message()
        m1.tags.append("tagged")
        assert "tagged" not in m2.tags

    def test_notes_default_factory_independence(self):
        m1 = _make_message()
        m2 = _make_message()
        m1.notes.append({"text": "note"})
        assert len(m2.notes) == 0

    def test_all_statuses(self):
        """Message can have any MessageStatus."""
        for status in MessageStatus:
            msg = _make_message(status=status)
            assert msg.status is status
            assert msg.to_dict()["status"] == status.value

    def test_empty_subject(self):
        msg = _make_message(subject="")
        assert msg.subject == ""
        assert msg.to_dict()["subject"] == ""

    def test_empty_snippet(self):
        msg = _make_message(snippet="")
        assert msg.snippet == ""

    def test_unicode_content(self):
        msg = _make_message(
            subject="Urgent: compte rendu",
            snippet="Bonjour, voici le rapport...",
            from_address="user@example.fr",
        )
        d = msg.to_dict()
        assert d["subject"] == "Urgent: compte rendu"
        assert d["snippet"] == "Bonjour, voici le rapport..."


# ===========================================================================
# SharedInbox Dataclass
# ===========================================================================


def _make_inbox(**overrides: Any) -> SharedInbox:
    """Build a valid SharedInbox with optional overrides."""
    defaults: dict[str, Any] = {
        "id": "inbox-001",
        "workspace_id": "ws-001",
        "name": "Support Inbox",
    }
    defaults.update(overrides)
    return SharedInbox(**defaults)


class TestSharedInbox:
    """Tests for the SharedInbox dataclass."""

    def test_construction_minimal(self):
        inbox = _make_inbox()
        assert inbox.id == "inbox-001"
        assert inbox.workspace_id == "ws-001"
        assert inbox.name == "Support Inbox"

    def test_defaults(self):
        inbox = _make_inbox()
        assert inbox.description is None
        assert inbox.email_address is None
        assert inbox.connector_type is None
        assert inbox.team_members == []
        assert inbox.admins == []
        assert inbox.settings == {}
        assert isinstance(inbox.created_at, datetime)
        assert isinstance(inbox.updated_at, datetime)
        assert inbox.created_by is None
        assert inbox.message_count == 0
        assert inbox.unread_count == 0

    def test_full_construction(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        inbox = SharedInbox(
            id="inbox-full",
            workspace_id="ws-full",
            name="Sales Inbox",
            description="For sales team",
            email_address="sales@company.com",
            connector_type="gmail",
            team_members=["user-1", "user-2"],
            admins=["admin-1"],
            settings={"auto_assign": True, "sla_hours": 24},
            created_at=now,
            updated_at=now,
            created_by="admin-1",
            message_count=150,
            unread_count=12,
        )
        assert inbox.description == "For sales team"
        assert inbox.email_address == "sales@company.com"
        assert inbox.connector_type == "gmail"
        assert inbox.team_members == ["user-1", "user-2"]
        assert inbox.admins == ["admin-1"]
        assert inbox.settings == {"auto_assign": True, "sla_hours": 24}
        assert inbox.message_count == 150
        assert inbox.unread_count == 12

    def test_to_dict_minimal(self):
        inbox = _make_inbox()
        d = inbox.to_dict()
        assert d["id"] == "inbox-001"
        assert d["workspace_id"] == "ws-001"
        assert d["name"] == "Support Inbox"
        assert d["description"] is None
        assert d["email_address"] is None
        assert d["connector_type"] is None
        assert d["team_members"] == []
        assert d["admins"] == []
        assert d["settings"] == {}
        assert d["created_by"] is None
        assert d["message_count"] == 0
        assert d["unread_count"] == 0

    def test_to_dict_full(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        inbox = SharedInbox(
            id="inbox-d",
            workspace_id="ws-d",
            name="Dev Inbox",
            description="Development",
            email_address="dev@co.com",
            connector_type="outlook",
            team_members=["u1", "u2", "u3"],
            admins=["a1"],
            settings={"feature_x": True},
            created_at=now,
            updated_at=now,
            created_by="a1",
            message_count=500,
            unread_count=50,
        )
        d = inbox.to_dict()
        assert d["description"] == "Development"
        assert d["email_address"] == "dev@co.com"
        assert d["connector_type"] == "outlook"
        assert d["team_members"] == ["u1", "u2", "u3"]
        assert d["admins"] == ["a1"]
        assert d["settings"] == {"feature_x": True}
        assert d["created_at"] == now.isoformat()
        assert d["updated_at"] == now.isoformat()
        assert d["created_by"] == "a1"
        assert d["message_count"] == 500
        assert d["unread_count"] == 50

    def test_to_dict_timestamps_are_iso(self):
        inbox = _make_inbox()
        d = inbox.to_dict()
        datetime.fromisoformat(d["created_at"])
        datetime.fromisoformat(d["updated_at"])

    def test_team_members_default_factory_independence(self):
        i1 = _make_inbox()
        i2 = _make_inbox()
        i1.team_members.append("user-x")
        assert "user-x" not in i2.team_members

    def test_admins_default_factory_independence(self):
        i1 = _make_inbox()
        i2 = _make_inbox()
        i1.admins.append("admin-x")
        assert "admin-x" not in i2.admins

    def test_settings_default_factory_independence(self):
        i1 = _make_inbox()
        i2 = _make_inbox()
        i1.settings["key"] = "val"
        assert "key" not in i2.settings

    def test_connector_types(self):
        """Various connector types are just strings, no restriction."""
        for ct in ["gmail", "outlook", "imap", "exchange", None]:
            inbox = _make_inbox(connector_type=ct)
            assert inbox.connector_type == ct

    def test_zero_counts(self):
        inbox = _make_inbox(message_count=0, unread_count=0)
        d = inbox.to_dict()
        assert d["message_count"] == 0
        assert d["unread_count"] == 0

    def test_large_counts(self):
        inbox = _make_inbox(message_count=1_000_000, unread_count=500_000)
        d = inbox.to_dict()
        assert d["message_count"] == 1_000_000
        assert d["unread_count"] == 500_000

    def test_empty_name(self):
        inbox = _make_inbox(name="")
        assert inbox.name == ""
        assert inbox.to_dict()["name"] == ""

    def test_unicode_name(self):
        inbox = _make_inbox(name="Support-Equipe")
        assert inbox.to_dict()["name"] == "Support-Equipe"

    def test_multiple_team_members(self):
        members = [f"user-{i}" for i in range(20)]
        inbox = _make_inbox(team_members=members)
        assert len(inbox.team_members) == 20
        assert inbox.to_dict()["team_members"] == members

    def test_nested_settings(self):
        settings = {
            "auto_assign": True,
            "sla": {"hours": 24, "escalate_to": "manager"},
            "notifications": ["email", "slack"],
        }
        inbox = _make_inbox(settings=settings)
        d = inbox.to_dict()
        assert d["settings"] == settings
        assert d["settings"]["sla"]["hours"] == 24


# ===========================================================================
# Cross-cutting / Integration
# ===========================================================================


class TestCrossCutting:
    """Tests that span multiple model classes or test interplay."""

    def test_routing_rule_with_all_condition_fields(self):
        """A rule can use every RuleConditionField."""
        conditions = [
            RuleCondition(field=f, operator=RuleConditionOperator.CONTAINS, value="x")
            for f in RuleConditionField
        ]
        rule = _make_routing_rule(conditions=conditions)
        assert len(rule.conditions) == 7
        d = rule.to_dict()
        fields_in_dict = {c["field"] for c in d["conditions"]}
        assert fields_in_dict == {f.value for f in RuleConditionField}

    def test_routing_rule_with_all_action_types(self):
        """A rule can use every RuleActionType."""
        actions = [RuleAction(type=t, target="target") for t in RuleActionType]
        rule = _make_routing_rule(actions=actions)
        assert len(rule.actions) == 6
        d = rule.to_dict()
        types_in_dict = {a["type"] for a in d["actions"]}
        assert types_in_dict == {t.value for t in RuleActionType}

    def test_routing_rule_with_all_operators(self):
        """A rule can use every RuleConditionOperator."""
        conditions = [
            RuleCondition(field=RuleConditionField.SUBJECT, operator=op, value="v")
            for op in RuleConditionOperator
        ]
        rule = _make_routing_rule(conditions=conditions)
        d = rule.to_dict()
        ops_in_dict = {c["operator"] for c in d["conditions"]}
        assert ops_in_dict == {op.value for op in RuleConditionOperator}

    def test_message_status_matches_expected_lifecycle(self):
        """Statuses cover the expected message lifecycle."""
        lifecycle = ["open", "assigned", "in_progress", "waiting", "resolved", "closed"]
        for status_str in lifecycle:
            s = MessageStatus(status_str)
            assert s.value == status_str

    def test_enum_string_comparison(self):
        """All str enums support direct string comparison."""
        assert MessageStatus.OPEN == "open"
        assert RuleConditionField.SUBJECT == "subject"
        assert RuleConditionOperator.EQUALS == "equals"
        assert RuleActionType.FORWARD == "forward"

    def test_from_dict_preserves_action_params(self):
        """Action params survive a full round-trip through RoutingRule."""
        rule = _make_routing_rule(
            actions=[
                RuleAction(
                    type=RuleActionType.NOTIFY,
                    target="slack",
                    params={"channel": "#alerts", "mention": "@oncall"},
                )
            ]
        )
        rebuilt = RoutingRule.from_dict(rule.to_dict())
        assert rebuilt.actions[0].params == {"channel": "#alerts", "mention": "@oncall"}

    def test_routing_rule_enabled_false_round_trip(self):
        """Disabled flag survives round-trip."""
        rule = _make_routing_rule(enabled=False)
        rebuilt = RoutingRule.from_dict(rule.to_dict())
        assert rebuilt.enabled is False

    def test_routing_rule_from_dict_priority_zero(self):
        """Priority 0 is preserved (not treated as falsy default)."""
        data = {
            "id": "r",
            "name": "N",
            "workspace_id": "ws",
            "priority": 0,
        }
        rule = RoutingRule.from_dict(data)
        # Note: from_dict uses data.get("priority", 5) so 0 should be preserved
        assert rule.priority == 0

    def test_routing_rule_from_dict_enabled_false(self):
        """Enabled=False is preserved (not treated as falsy default)."""
        data = {
            "id": "r",
            "name": "N",
            "workspace_id": "ws",
            "enabled": False,
        }
        rule = RoutingRule.from_dict(data)
        assert rule.enabled is False
