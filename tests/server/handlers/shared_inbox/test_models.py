"""Tests for shared inbox data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.server.handlers.shared_inbox.models import (
    MessageStatus,
    RoutingRule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
    SharedInbox,
    SharedInboxMessage,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestMessageStatus:
    def test_values(self):
        assert MessageStatus.OPEN.value == "open"
        assert MessageStatus.ASSIGNED.value == "assigned"
        assert MessageStatus.IN_PROGRESS.value == "in_progress"
        assert MessageStatus.WAITING.value == "waiting"
        assert MessageStatus.RESOLVED.value == "resolved"
        assert MessageStatus.CLOSED.value == "closed"

    def test_is_str_enum(self):
        assert isinstance(MessageStatus.OPEN, str)
        assert MessageStatus.OPEN == "open"

    def test_member_count(self):
        assert len(MessageStatus) == 6


class TestRuleConditionField:
    def test_values(self):
        assert RuleConditionField.FROM.value == "from"
        assert RuleConditionField.TO.value == "to"
        assert RuleConditionField.SUBJECT.value == "subject"
        assert RuleConditionField.BODY.value == "body"
        assert RuleConditionField.LABELS.value == "labels"
        assert RuleConditionField.PRIORITY.value == "priority"
        assert RuleConditionField.SENDER_DOMAIN.value == "sender_domain"

    def test_member_count(self):
        assert len(RuleConditionField) == 7


class TestRuleConditionOperator:
    def test_values(self):
        assert RuleConditionOperator.CONTAINS.value == "contains"
        assert RuleConditionOperator.EQUALS.value == "equals"
        assert RuleConditionOperator.STARTS_WITH.value == "starts_with"
        assert RuleConditionOperator.ENDS_WITH.value == "ends_with"
        assert RuleConditionOperator.MATCHES.value == "matches"
        assert RuleConditionOperator.GREATER_THAN.value == "greater_than"
        assert RuleConditionOperator.LESS_THAN.value == "less_than"

    def test_member_count(self):
        assert len(RuleConditionOperator) == 7


class TestRuleActionType:
    def test_values(self):
        assert RuleActionType.ASSIGN.value == "assign"
        assert RuleActionType.LABEL.value == "label"
        assert RuleActionType.ESCALATE.value == "escalate"
        assert RuleActionType.ARCHIVE.value == "archive"
        assert RuleActionType.NOTIFY.value == "notify"
        assert RuleActionType.FORWARD.value == "forward"

    def test_member_count(self):
        assert len(RuleActionType) == 6


# =============================================================================
# RuleCondition Tests
# =============================================================================


class TestRuleCondition:
    def test_to_dict(self):
        cond = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="urgent",
        )
        d = cond.to_dict()
        assert d == {"field": "subject", "operator": "contains", "value": "urgent"}

    def test_from_dict(self):
        data = {"field": "from", "operator": "equals", "value": "boss@corp.com"}
        cond = RuleCondition.from_dict(data)
        assert cond.field == RuleConditionField.FROM
        assert cond.operator == RuleConditionOperator.EQUALS
        assert cond.value == "boss@corp.com"

    def test_roundtrip(self):
        original = RuleCondition(
            field=RuleConditionField.SENDER_DOMAIN,
            operator=RuleConditionOperator.ENDS_WITH,
            value=".gov",
        )
        restored = RuleCondition.from_dict(original.to_dict())
        assert restored.field == original.field
        assert restored.operator == original.operator
        assert restored.value == original.value

    def test_from_dict_invalid_field_raises(self):
        with pytest.raises(ValueError):
            RuleCondition.from_dict({"field": "invalid", "operator": "contains", "value": "x"})

    def test_from_dict_invalid_operator_raises(self):
        with pytest.raises(ValueError):
            RuleCondition.from_dict({"field": "subject", "operator": "invalid", "value": "x"})


# =============================================================================
# RuleAction Tests
# =============================================================================


class TestRuleAction:
    def test_to_dict(self):
        action = RuleAction(
            type=RuleActionType.ASSIGN,
            target="user-42",
            params={"notify": "true"},
        )
        d = action.to_dict()
        assert d == {
            "type": "assign",
            "target": "user-42",
            "params": {"notify": "true"},
        }

    def test_from_dict(self):
        data = {"type": "label", "target": "important", "params": {"color": "red"}}
        action = RuleAction.from_dict(data)
        assert action.type == RuleActionType.LABEL
        assert action.target == "important"
        assert action.params == {"color": "red"}

    def test_defaults(self):
        action = RuleAction(type=RuleActionType.ARCHIVE)
        assert action.target is None
        assert action.params == {}

    def test_from_dict_defaults(self):
        data = {"type": "archive"}
        action = RuleAction.from_dict(data)
        assert action.target is None
        assert action.params == {}

    def test_roundtrip(self):
        original = RuleAction(
            type=RuleActionType.FORWARD,
            target="team@company.com",
            params={"cc": "manager@company.com"},
        )
        restored = RuleAction.from_dict(original.to_dict())
        assert restored.type == original.type
        assert restored.target == original.target
        assert restored.params == original.params


# =============================================================================
# RoutingRule Tests
# =============================================================================


class TestRoutingRule:
    @pytest.fixture
    def sample_rule(self):
        return RoutingRule(
            id="rule-1",
            name="Urgent Escalation",
            workspace_id="ws-1",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="URGENT",
                ),
            ],
            condition_logic="AND",
            actions=[
                RuleAction(type=RuleActionType.ESCALATE, target="manager"),
            ],
            priority=1,
            enabled=True,
            description="Escalate urgent emails",
            created_by="admin",
        )

    def test_to_dict(self, sample_rule):
        d = sample_rule.to_dict()
        assert d["id"] == "rule-1"
        assert d["name"] == "Urgent Escalation"
        assert d["workspace_id"] == "ws-1"
        assert d["condition_logic"] == "AND"
        assert d["priority"] == 1
        assert d["enabled"] is True
        assert d["description"] == "Escalate urgent emails"
        assert d["created_by"] == "admin"
        assert len(d["conditions"]) == 1
        assert d["conditions"][0]["value"] == "URGENT"
        assert len(d["actions"]) == 1
        assert d["actions"][0]["type"] == "escalate"
        # Timestamps are ISO strings
        assert isinstance(d["created_at"], str)
        assert isinstance(d["updated_at"], str)

    def test_from_dict_roundtrip(self, sample_rule):
        d = sample_rule.to_dict()
        restored = RoutingRule.from_dict(d)
        assert restored.id == sample_rule.id
        assert restored.name == sample_rule.name
        assert restored.workspace_id == sample_rule.workspace_id
        assert restored.condition_logic == sample_rule.condition_logic
        assert restored.priority == sample_rule.priority
        assert restored.enabled == sample_rule.enabled
        assert len(restored.conditions) == 1
        assert len(restored.actions) == 1

    def test_from_dict_defaults(self):
        data = {
            "id": "rule-2",
            "name": "Simple",
            "workspace_id": "ws-1",
            "conditions": [],
            "actions": [],
        }
        rule = RoutingRule.from_dict(data)
        assert rule.condition_logic == "AND"
        assert rule.priority == 5
        assert rule.enabled is True
        assert rule.description is None
        assert rule.created_by is None
        assert rule.stats == {}

    def test_defaults(self):
        rule = RoutingRule(
            id="r",
            name="n",
            workspace_id="w",
            conditions=[],
            condition_logic="OR",
            actions=[],
        )
        assert rule.priority == 5
        assert rule.enabled is True
        assert rule.description is None
        assert rule.stats == {}


# =============================================================================
# SharedInboxMessage Tests
# =============================================================================


class TestSharedInboxMessage:
    @pytest.fixture
    def sample_message(self):
        return SharedInboxMessage(
            id="msg-1",
            inbox_id="inbox-1",
            email_id="email-abc",
            subject="Test subject",
            from_address="sender@test.com",
            to_addresses=["team@company.com"],
            snippet="Hello world...",
            received_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            status=MessageStatus.ASSIGNED,
            assigned_to="user-42",
            assigned_at=datetime(2026, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
            tags=["important", "client"],
            priority="high",
            notes=[{"author": "user-42", "text": "Working on it"}],
            thread_id="thread-xyz",
        )

    def test_to_dict(self, sample_message):
        d = sample_message.to_dict()
        assert d["id"] == "msg-1"
        assert d["inbox_id"] == "inbox-1"
        assert d["email_id"] == "email-abc"
        assert d["subject"] == "Test subject"
        assert d["from_address"] == "sender@test.com"
        assert d["to_addresses"] == ["team@company.com"]
        assert d["snippet"] == "Hello world..."
        assert d["status"] == "assigned"
        assert d["assigned_to"] == "user-42"
        assert d["tags"] == ["important", "client"]
        assert d["priority"] == "high"
        assert d["thread_id"] == "thread-xyz"
        # Datetimes serialized as ISO strings
        assert "2026-01-15" in d["received_at"]
        assert "2026-01-15" in d["assigned_at"]

    def test_defaults(self):
        msg = SharedInboxMessage(
            id="m",
            inbox_id="i",
            email_id="e",
            subject="s",
            from_address="f@t.com",
            to_addresses=[],
            snippet="",
            received_at=datetime.now(timezone.utc),
        )
        assert msg.status == MessageStatus.OPEN
        assert msg.assigned_to is None
        assert msg.assigned_at is None
        assert msg.tags == []
        assert msg.priority is None
        assert msg.notes == []
        assert msg.thread_id is None
        assert msg.sla_deadline is None
        assert msg.resolved_at is None
        assert msg.resolved_by is None

    def test_to_dict_none_fields(self):
        msg = SharedInboxMessage(
            id="m",
            inbox_id="i",
            email_id="e",
            subject="s",
            from_address="f@t.com",
            to_addresses=[],
            snippet="",
            received_at=datetime.now(timezone.utc),
        )
        d = msg.to_dict()
        assert d["assigned_at"] is None
        assert d["sla_deadline"] is None
        assert d["resolved_at"] is None
        assert d["resolved_by"] is None


# =============================================================================
# SharedInbox Tests
# =============================================================================


class TestSharedInbox:
    @pytest.fixture
    def sample_inbox(self):
        return SharedInbox(
            id="inbox-1",
            workspace_id="ws-1",
            name="Support Inbox",
            description="Customer support",
            email_address="support@company.com",
            connector_type="gmail",
            team_members=["user-1", "user-2"],
            admins=["admin-1"],
            settings={"auto_assign": True},
            created_by="admin-1",
            message_count=42,
            unread_count=5,
        )

    def test_to_dict(self, sample_inbox):
        d = sample_inbox.to_dict()
        assert d["id"] == "inbox-1"
        assert d["workspace_id"] == "ws-1"
        assert d["name"] == "Support Inbox"
        assert d["description"] == "Customer support"
        assert d["email_address"] == "support@company.com"
        assert d["connector_type"] == "gmail"
        assert d["team_members"] == ["user-1", "user-2"]
        assert d["admins"] == ["admin-1"]
        assert d["settings"] == {"auto_assign": True}
        assert d["created_by"] == "admin-1"
        assert d["message_count"] == 42
        assert d["unread_count"] == 5
        assert isinstance(d["created_at"], str)
        assert isinstance(d["updated_at"], str)

    def test_defaults(self):
        inbox = SharedInbox(id="i", workspace_id="w", name="n")
        assert inbox.description is None
        assert inbox.email_address is None
        assert inbox.connector_type is None
        assert inbox.team_members == []
        assert inbox.admins == []
        assert inbox.settings == {}
        assert inbox.created_by is None
        assert inbox.message_count == 0
        assert inbox.unread_count == 0
