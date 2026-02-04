"""
Tests for aragora.server.handlers.shared_inbox.rules_engine module.

Tests cover:
1. _evaluate_rule - rule matching logic
2. get_matching_rules_for_email - async rule fetching
3. apply_routing_rules_to_message - rule application
4. evaluate_rule_for_test - test evaluation helper
5. Different condition operators (contains, equals, starts_with, ends_with, matches)
6. AND/OR condition logic
7. Timeout protection for regex
8. Message field extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockMessage:
    """Mock message for rule evaluation testing."""

    from_address: str = "sender@example.com"
    to_addresses: list[str] = field(default_factory=lambda: ["recipient@example.com"])
    subject: str = "Test Subject"
    snippet: str = "Test body content"
    priority: str | None = None
    status: MagicMock = field(default_factory=lambda: MagicMock(value="open"))
    assigned_to: str | None = None
    assigned_at: datetime | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not hasattr(self.status, "value"):
            self.status = MagicMock(value="open")


@dataclass
class MockCondition:
    """Mock rule condition."""

    field: MagicMock
    operator: MagicMock
    value: str

    def __init__(self, field_name: str, operator_name: str, value: str):
        self.field = MagicMock()
        self.field.name = field_name
        self.field.__eq__ = lambda self, other: (
            hasattr(other, "name") and self.name == other.name
        ) or self.name == str(other)
        # Make the field match enum comparisons
        self.field.value = field_name

        self.operator = MagicMock()
        self.operator.name = operator_name
        self.operator.__eq__ = lambda self, other: (
            hasattr(other, "name") and self.name == other.name
        ) or self.name == str(other)
        self.operator.value = operator_name

        self.value = value


@dataclass
class MockRoutingRule:
    """Mock routing rule."""

    id: str = "rule_test123"
    name: str = "Test Rule"
    workspace_id: str = "ws_test123"
    inbox_id: str | None = None
    conditions: list = field(default_factory=list)
    condition_logic: str = "AND"
    actions: list = field(default_factory=list)
    enabled: bool = True
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "inbox_id": self.inbox_id,
            "conditions": [
                {"field": c.field.value, "operator": c.operator.value, "value": c.value}
                for c in self.conditions
            ],
            "condition_logic": self.condition_logic,
            "actions": self.actions,
            "enabled": self.enabled,
            "priority": self.priority,
        }


# =============================================================================
# Mock Enums
# =============================================================================


class MockRuleConditionField:
    """Mock enum for rule condition fields."""

    FROM = MagicMock(name="FROM", value="from")
    TO = MagicMock(name="TO", value="to")
    SUBJECT = MagicMock(name="SUBJECT", value="subject")
    SENDER_DOMAIN = MagicMock(name="SENDER_DOMAIN", value="sender_domain")
    PRIORITY = MagicMock(name="PRIORITY", value="priority")


class MockRuleConditionOperator:
    """Mock enum for rule condition operators."""

    CONTAINS = MagicMock(name="CONTAINS", value="contains")
    EQUALS = MagicMock(name="EQUALS", value="equals")
    STARTS_WITH = MagicMock(name="STARTS_WITH", value="starts_with")
    ENDS_WITH = MagicMock(name="ENDS_WITH", value="ends_with")
    MATCHES = MagicMock(name="MATCHES", value="matches")


class MockMessageStatus:
    """Mock enum for message status."""

    OPEN = MagicMock(name="OPEN", value="open")
    ASSIGNED = MagicMock(name="ASSIGNED", value="assigned")
    CLOSED = MagicMock(name="CLOSED", value="closed")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_message():
    """Create a mock message."""
    return MockMessage()


@pytest.fixture
def mock_rule():
    """Create a mock routing rule."""
    return MockRoutingRule()


# =============================================================================
# Test Rule Evaluation (_evaluate_rule)
# =============================================================================


class TestEvaluateRule:
    """Tests for _evaluate_rule function."""

    def test_evaluate_rule_contains_match(self):
        """Test CONTAINS operator matches correctly."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="URGENT: Please review this")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="urgent",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Urgent Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_contains_no_match(self):
        """Test CONTAINS operator returns False when no match."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="Normal subject")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="urgent",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Urgent Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is False

    def test_evaluate_rule_equals_match(self):
        """Test EQUALS operator matches correctly."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(from_address="admin@example.com")
        condition = RuleCondition(
            field=RuleConditionField.FROM,
            operator=RuleConditionOperator.EQUALS,
            value="admin@example.com",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Admin Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_starts_with_match(self):
        """Test STARTS_WITH operator matches correctly."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="[ALERT] System notification")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.STARTS_WITH,
            value="[alert]",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Alert Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_ends_with_match(self):
        """Test ENDS_WITH operator matches correctly."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="Meeting reminder - EOD")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.ENDS_WITH,
            value="eod",
        )
        rule = RoutingRule(
            id="rule_1",
            name="EOD Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_matches_regex(self):
        """Test MATCHES operator with valid regex."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="Invoice #12345 for review")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.MATCHES,
            value=r"invoice\s*#\d+",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Invoice Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_sender_domain_extraction(self):
        """Test sender domain field extraction."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(from_address="user@important-client.com")
        condition = RuleCondition(
            field=RuleConditionField.SENDER_DOMAIN,
            operator=RuleConditionOperator.EQUALS,
            value="important-client.com",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Client Domain Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_to_addresses(self):
        """Test TO field matches across multiple addresses."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(to_addresses=["support@example.com", "billing@example.com"])
        condition = RuleCondition(
            field=RuleConditionField.TO,
            operator=RuleConditionOperator.CONTAINS,
            value="billing",
        )
        rule = RoutingRule(
            id="rule_1",
            name="Billing Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_priority_field(self):
        """Test priority field matching."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(priority="high")
        condition = RuleCondition(
            field=RuleConditionField.PRIORITY,
            operator=RuleConditionOperator.EQUALS,
            value="high",
        )
        rule = RoutingRule(
            id="rule_1",
            name="High Priority Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_and_logic_all_match(self):
        """Test AND logic requires all conditions to match."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(
            from_address="admin@company.com",
            subject="[URGENT] Action Required",
        )
        conditions = [
            RuleCondition(
                field=RuleConditionField.FROM,
                operator=RuleConditionOperator.CONTAINS,
                value="admin",
            ),
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",
            ),
        ]
        rule = RoutingRule(
            id="rule_1",
            name="Admin Urgent Rule",
            workspace_id="ws_1",
            conditions=conditions,
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_and_logic_partial_match(self):
        """Test AND logic returns False when not all conditions match."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(
            from_address="admin@company.com",
            subject="Normal subject",  # No "urgent" keyword
        )
        conditions = [
            RuleCondition(
                field=RuleConditionField.FROM,
                operator=RuleConditionOperator.CONTAINS,
                value="admin",
            ),
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",
            ),
        ]
        rule = RoutingRule(
            id="rule_1",
            name="Admin Urgent Rule",
            workspace_id="ws_1",
            conditions=conditions,
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is False

    def test_evaluate_rule_or_logic_any_match(self):
        """Test OR logic requires at least one condition to match."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(
            from_address="regular@company.com",
            subject="[URGENT] Please respond",
        )
        conditions = [
            RuleCondition(
                field=RuleConditionField.FROM,
                operator=RuleConditionOperator.CONTAINS,
                value="admin",  # Does not match
            ),
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",  # Matches
            ),
        ]
        rule = RoutingRule(
            id="rule_1",
            name="Alert Rule",
            workspace_id="ws_1",
            conditions=conditions,
            condition_logic="OR",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True

    def test_evaluate_rule_or_logic_no_match(self):
        """Test OR logic returns False when no conditions match."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(
            from_address="regular@company.com",
            subject="Normal subject",
        )
        conditions = [
            RuleCondition(
                field=RuleConditionField.FROM,
                operator=RuleConditionOperator.CONTAINS,
                value="admin",  # Does not match
            ),
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",  # Does not match
            ),
        ]
        rule = RoutingRule(
            id="rule_1",
            name="Alert Rule",
            workspace_id="ws_1",
            conditions=conditions,
            condition_logic="OR",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is False

    def test_evaluate_rule_empty_conditions(self):
        """Test rule with no conditions returns False."""
        from aragora.server.handlers.shared_inbox.models import RoutingRule
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage()
        rule = RoutingRule(
            id="rule_1",
            name="Empty Rule",
            workspace_id="ws_1",
            conditions=[],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is False

    def test_evaluate_rule_case_insensitive(self):
        """Test rule matching is case insensitive."""
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )
        from aragora.server.handlers.shared_inbox.rules_engine import _evaluate_rule

        message = MockMessage(subject="URGENT: Review needed")
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="Urgent",  # Different case
        )
        rule = RoutingRule(
            id="rule_1",
            name="Case Test Rule",
            workspace_id="ws_1",
            conditions=[condition],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        result = _evaluate_rule(rule, message)
        assert result is True


# =============================================================================
# Test Get Matching Rules For Email
# =============================================================================


class TestGetMatchingRulesForEmail:
    """Tests for get_matching_rules_for_email function."""

    @pytest.mark.asyncio
    async def test_get_matching_rules_empty_inbox(self):
        """Test getting rules for inbox with no rules."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            get_matching_rules_for_email,
        )

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
            return_value=None,
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._routing_rules",
                {},
            ):
                result = await get_matching_rules_for_email(
                    inbox_id="inbox_1",
                    email_data={
                        "from_address": "sender@example.com",
                        "to_addresses": ["recipient@example.com"],
                        "subject": "Test",
                    },
                )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_matching_rules_from_store(self):
        """Test getting rules from RulesStore."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            get_matching_rules_for_email,
        )

        mock_store = MagicMock()
        mock_store.get_matching_rules.return_value = [
            {"id": "rule_1", "name": "Rule 1", "priority": 1}
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._routing_rules",
                {},
            ):
                with patch(
                    "aragora.server.handlers.shared_inbox.rules_engine._storage_lock",
                    MagicMock(),
                ):
                    result = await get_matching_rules_for_email(
                        inbox_id="inbox_1",
                        email_data={
                            "from_address": "sender@example.com",
                            "to_addresses": ["recipient@example.com"],
                            "subject": "Test",
                        },
                    )

        assert len(result) == 1
        assert result[0]["id"] == "rule_1"

    @pytest.mark.asyncio
    async def test_get_matching_rules_store_fallback(self):
        """Test fallback to in-memory when store fails."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            get_matching_rules_for_email,
        )

        mock_store = MagicMock()
        mock_store.get_matching_rules.side_effect = RuntimeError("Store error")

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._routing_rules",
                {},
            ):
                with patch(
                    "aragora.server.handlers.shared_inbox.rules_engine._storage_lock",
                    MagicMock(),
                ):
                    result = await get_matching_rules_for_email(
                        inbox_id="inbox_1",
                        email_data={
                            "from_address": "sender@example.com",
                            "to_addresses": ["recipient@example.com"],
                            "subject": "Test",
                        },
                    )

        # Should return empty list (fallback worked)
        assert result == []


# =============================================================================
# Test Apply Routing Rules to Message
# =============================================================================


class TestApplyRoutingRulesToMessage:
    """Tests for apply_routing_rules_to_message function."""

    @pytest.mark.asyncio
    async def test_apply_rules_no_matching_rules(self):
        """Test applying rules when no rules match."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )

        message = MockMessage()

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=[]),
        ):
            result = await apply_routing_rules_to_message(
                inbox_id="inbox_1",
                message=message,
                workspace_id="ws_1",
            )

        assert result["applied"] is False
        assert result["rules_matched"] == 0
        assert result["actions"] == []

    @pytest.mark.asyncio
    async def test_apply_rules_assign_action(self):
        """Test applying assign action."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )
        from aragora.server.handlers.shared_inbox.models import MessageStatus

        message = MockMessage()
        message.status = MessageStatus.OPEN

        matching_rules = [
            {
                "id": "rule_1",
                "actions": [{"type": "assign", "target": "user_123"}],
            }
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=matching_rules),
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
                return_value=None,
            ):
                result = await apply_routing_rules_to_message(
                    inbox_id="inbox_1",
                    message=message,
                    workspace_id="ws_1",
                )

        assert result["applied"] is True
        assert result["rules_matched"] == 1
        assert {"type": "assign", "target": "user_123"} in result["actions"]
        assert message.assigned_to == "user_123"

    @pytest.mark.asyncio
    async def test_apply_rules_label_action(self):
        """Test applying label action."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )

        message = MockMessage(tags=[])

        matching_rules = [
            {
                "id": "rule_1",
                "actions": [{"type": "label", "target": "important"}],
            }
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=matching_rules),
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
                return_value=None,
            ):
                result = await apply_routing_rules_to_message(
                    inbox_id="inbox_1",
                    message=message,
                    workspace_id="ws_1",
                )

        assert result["applied"] is True
        assert {"type": "label", "target": "important"} in result["actions"]
        assert "important" in message.tags

    @pytest.mark.asyncio
    async def test_apply_rules_escalate_action(self):
        """Test applying escalate action."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )

        message = MockMessage(priority="normal")

        matching_rules = [
            {
                "id": "rule_1",
                "actions": [{"type": "escalate"}],
            }
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=matching_rules),
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
                return_value=None,
            ):
                result = await apply_routing_rules_to_message(
                    inbox_id="inbox_1",
                    message=message,
                    workspace_id="ws_1",
                )

        assert result["applied"] is True
        assert {"type": "escalate"} in result["actions"]
        assert message.priority == "high"

    @pytest.mark.asyncio
    async def test_apply_rules_archive_action(self):
        """Test applying archive action."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )
        from aragora.server.handlers.shared_inbox.models import MessageStatus

        message = MockMessage()
        message.status = MessageStatus.OPEN

        matching_rules = [
            {
                "id": "rule_1",
                "actions": [{"type": "archive"}],
            }
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=matching_rules),
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
                return_value=None,
            ):
                result = await apply_routing_rules_to_message(
                    inbox_id="inbox_1",
                    message=message,
                    workspace_id="ws_1",
                )

        assert result["applied"] is True
        assert {"type": "archive"} in result["actions"]
        assert message.status == MessageStatus.CLOSED

    @pytest.mark.asyncio
    async def test_apply_rules_multiple_actions(self):
        """Test applying multiple actions from one rule."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            apply_routing_rules_to_message,
        )
        from aragora.server.handlers.shared_inbox.models import MessageStatus

        message = MockMessage(tags=[])
        message.status = MessageStatus.OPEN

        matching_rules = [
            {
                "id": "rule_1",
                "actions": [
                    {"type": "assign", "target": "user_123"},
                    {"type": "label", "target": "vip"},
                    {"type": "escalate"},
                ],
            }
        ]

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine.get_matching_rules_for_email",
            AsyncMock(return_value=matching_rules),
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._get_rules_store",
                return_value=None,
            ):
                result = await apply_routing_rules_to_message(
                    inbox_id="inbox_1",
                    message=message,
                    workspace_id="ws_1",
                )

        assert result["applied"] is True
        assert result["rules_matched"] == 1
        assert len(result["actions"]) == 3
        assert message.assigned_to == "user_123"
        assert "vip" in message.tags
        assert message.priority == "high"


# =============================================================================
# Test Evaluate Rule for Test
# =============================================================================


class TestEvaluateRuleForTest:
    """Tests for evaluate_rule_for_test function."""

    def test_evaluate_rule_for_test_empty_inbox(self):
        """Test evaluation with no messages returns 0."""
        from aragora.server.handlers.shared_inbox.rules_engine import (
            evaluate_rule_for_test,
        )
        from aragora.server.handlers.shared_inbox.models import (
            RuleCondition,
            RuleConditionField,
            RuleConditionOperator,
            RoutingRule,
        )

        rule = RoutingRule(
            id="rule_1",
            name="Test Rule",
            workspace_id="ws_1",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="test",
                )
            ],
            condition_logic="AND",
            actions=[],
            enabled=True,
        )

        with patch(
            "aragora.server.handlers.shared_inbox.rules_engine._inbox_messages",
            {},
        ):
            with patch(
                "aragora.server.handlers.shared_inbox.rules_engine._shared_inboxes",
                {},
            ):
                with patch(
                    "aragora.server.handlers.shared_inbox.rules_engine._storage_lock",
                    MagicMock(),
                ):
                    result = evaluate_rule_for_test(rule, "ws_1")

        assert result == 0


__all__ = [
    "TestEvaluateRule",
    "TestGetMatchingRulesForEmail",
    "TestApplyRoutingRulesToMessage",
    "TestEvaluateRuleForTest",
]
