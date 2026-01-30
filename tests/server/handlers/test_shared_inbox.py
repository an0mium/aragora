"""
Tests for the shared inbox handler.

Tests:
- Shared inbox CRUD operations
- Message assignment and status
- Routing rules creation and evaluation
- Thread safety and data integrity
- Error handling
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from aragora.server.handlers.shared_inbox import (
    SharedInboxHandler,
    SharedInbox,
    SharedInboxMessage,
    MessageStatus,
    RoutingRule,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
    RuleAction,
    RuleActionType,
    handle_create_shared_inbox,
    handle_list_shared_inboxes,
    handle_get_shared_inbox,
    handle_get_inbox_messages,
    handle_assign_message,
    handle_update_message_status,
    handle_add_message_tag,
    handle_create_routing_rule,
    handle_list_routing_rules,
    handle_update_routing_rule,
    handle_delete_routing_rule,
    handle_test_routing_rule,
    apply_routing_rules_to_message,
    _shared_inboxes,
    _inbox_messages,
    _routing_rules,
    _storage_lock,
)


@pytest.fixture
def shared_inbox_handler():
    """Create a shared inbox handler with mocked dependencies."""
    ctx = {"storage": None, "auth_context": MagicMock(user_id="test_user")}
    handler = SharedInboxHandler(ctx)
    return handler


@pytest.fixture
def clean_inbox_state():
    """Clean up inbox state before/after tests and mock stores."""
    # Clear in-memory state before test
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()

    # Mock the store functions to return None (forces in-memory fallback)
    with (
        patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None),
        patch("aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None),
        patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store", return_value=None
        ),
    ):
        yield

    # Clear state after test
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()


@pytest.fixture
def sample_inbox(clean_inbox_state):
    """Create a sample inbox for testing."""
    now = datetime.now(timezone.utc)
    inbox = SharedInbox(
        id="inbox_test123",
        workspace_id="ws_test",
        name="Test Inbox",
        description="A test inbox",
        email_address="test@example.com",
        connector_type="gmail",
        team_members=["user1", "user2"],
        admins=["admin1"],
        settings={"auto_assign": True},
        created_at=now,
        updated_at=now,
        created_by="admin1",
    )
    with _storage_lock:
        _shared_inboxes[inbox.id] = inbox
        _inbox_messages[inbox.id] = {}
    return inbox


@pytest.fixture
def sample_message(sample_inbox):
    """Create a sample message in the sample inbox."""
    now = datetime.now(timezone.utc)
    message = SharedInboxMessage(
        id="msg_test456",
        inbox_id=sample_inbox.id,
        email_id="email_123",
        subject="Test Subject",
        from_address="sender@example.com",
        to_addresses=["test@example.com"],
        snippet="This is a test email...",
        received_at=now,
        status=MessageStatus.OPEN,
    )
    with _storage_lock:
        _inbox_messages[sample_inbox.id][message.id] = message
    return message


@pytest.fixture
def sample_routing_rule(clean_inbox_state):
    """Create a sample routing rule."""
    now = datetime.now(timezone.utc)
    rule = RoutingRule(
        id="rule_test789",
        name="Test Rule",
        workspace_id="ws_test",
        conditions=[
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",
            )
        ],
        condition_logic="AND",
        actions=[
            RuleAction(
                type=RuleActionType.LABEL,
                target="urgent",
            )
        ],
        priority=1,
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    with _storage_lock:
        _routing_rules[rule.id] = rule
    return rule


# =============================================================================
# Data Model Tests
# =============================================================================


class TestMessageStatus:
    """Tests for MessageStatus enum."""

    def test_status_values(self):
        """All expected status values should be defined."""
        assert MessageStatus.OPEN == "open"
        assert MessageStatus.ASSIGNED == "assigned"
        assert MessageStatus.IN_PROGRESS == "in_progress"
        assert MessageStatus.WAITING == "waiting"
        assert MessageStatus.RESOLVED == "resolved"
        assert MessageStatus.CLOSED == "closed"


class TestRuleConditionField:
    """Tests for RuleConditionField enum."""

    def test_field_values(self):
        """All expected field values should be defined."""
        assert RuleConditionField.FROM == "from"
        assert RuleConditionField.TO == "to"
        assert RuleConditionField.SUBJECT == "subject"
        assert RuleConditionField.BODY == "body"
        assert RuleConditionField.LABELS == "labels"
        assert RuleConditionField.PRIORITY == "priority"
        assert RuleConditionField.SENDER_DOMAIN == "sender_domain"


class TestRuleConditionOperator:
    """Tests for RuleConditionOperator enum."""

    def test_operator_values(self):
        """All expected operator values should be defined."""
        assert RuleConditionOperator.CONTAINS == "contains"
        assert RuleConditionOperator.EQUALS == "equals"
        assert RuleConditionOperator.STARTS_WITH == "starts_with"
        assert RuleConditionOperator.ENDS_WITH == "ends_with"
        assert RuleConditionOperator.MATCHES == "matches"


class TestRuleActionType:
    """Tests for RuleActionType enum."""

    def test_action_values(self):
        """All expected action values should be defined."""
        assert RuleActionType.ASSIGN == "assign"
        assert RuleActionType.LABEL == "label"
        assert RuleActionType.ESCALATE == "escalate"
        assert RuleActionType.ARCHIVE == "archive"
        assert RuleActionType.NOTIFY == "notify"
        assert RuleActionType.FORWARD == "forward"


class TestRuleCondition:
    """Tests for RuleCondition dataclass."""

    def test_to_dict(self):
        """Condition should serialize to dict correctly."""
        condition = RuleCondition(
            field=RuleConditionField.SUBJECT,
            operator=RuleConditionOperator.CONTAINS,
            value="test",
        )
        result = condition.to_dict()
        assert result["field"] == "subject"
        assert result["operator"] == "contains"
        assert result["value"] == "test"

    def test_from_dict(self):
        """Condition should deserialize from dict correctly."""
        data = {
            "field": "subject",
            "operator": "contains",
            "value": "test",
        }
        condition = RuleCondition.from_dict(data)
        assert condition.field == RuleConditionField.SUBJECT
        assert condition.operator == RuleConditionOperator.CONTAINS
        assert condition.value == "test"


class TestRuleAction:
    """Tests for RuleAction dataclass."""

    def test_to_dict(self):
        """Action should serialize to dict correctly."""
        action = RuleAction(
            type=RuleActionType.ASSIGN,
            target="user1",
            params={"priority": "high"},
        )
        result = action.to_dict()
        assert result["type"] == "assign"
        assert result["target"] == "user1"
        assert result["params"] == {"priority": "high"}

    def test_from_dict(self):
        """Action should deserialize from dict correctly."""
        data = {
            "type": "assign",
            "target": "user1",
            "params": {"priority": "high"},
        }
        action = RuleAction.from_dict(data)
        assert action.type == RuleActionType.ASSIGN
        assert action.target == "user1"
        assert action.params == {"priority": "high"}

    def test_from_dict_defaults(self):
        """Action should handle missing optional fields."""
        data = {"type": "label"}
        action = RuleAction.from_dict(data)
        assert action.type == RuleActionType.LABEL
        assert action.target is None
        assert action.params == {}


class TestRoutingRule:
    """Tests for RoutingRule dataclass."""

    def test_to_dict(self):
        """Rule should serialize to dict correctly."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_123",
            name="Test Rule",
            workspace_id="ws_test",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                )
            ],
            condition_logic="AND",
            actions=[RuleAction(type=RuleActionType.LABEL, target="urgent")],
            priority=1,
            enabled=True,
            description="Test description",
            created_at=now,
            updated_at=now,
        )
        result = rule.to_dict()
        assert result["id"] == "rule_123"
        assert result["name"] == "Test Rule"
        assert result["workspace_id"] == "ws_test"
        assert len(result["conditions"]) == 1
        assert len(result["actions"]) == 1
        assert result["priority"] == 1
        assert result["enabled"] is True

    def test_from_dict(self):
        """Rule should deserialize from dict correctly."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "rule_123",
            "name": "Test Rule",
            "workspace_id": "ws_test",
            "conditions": [{"field": "subject", "operator": "contains", "value": "urgent"}],
            "actions": [{"type": "label", "target": "urgent"}],
            "condition_logic": "AND",
            "priority": 1,
            "enabled": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        rule = RoutingRule.from_dict(data)
        assert rule.id == "rule_123"
        assert rule.name == "Test Rule"
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == RuleConditionField.SUBJECT


class TestSharedInboxMessage:
    """Tests for SharedInboxMessage dataclass."""

    def test_to_dict(self):
        """Message should serialize to dict correctly."""
        now = datetime.now(timezone.utc)
        message = SharedInboxMessage(
            id="msg_123",
            inbox_id="inbox_123",
            email_id="email_123",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            snippet="Test snippet",
            received_at=now,
            status=MessageStatus.OPEN,
            tags=["tag1", "tag2"],
        )
        result = message.to_dict()
        assert result["id"] == "msg_123"
        assert result["subject"] == "Test Subject"
        assert result["status"] == "open"
        assert result["tags"] == ["tag1", "tag2"]


class TestSharedInbox:
    """Tests for SharedInbox dataclass."""

    def test_to_dict(self):
        """Inbox should serialize to dict correctly."""
        now = datetime.now(timezone.utc)
        inbox = SharedInbox(
            id="inbox_123",
            workspace_id="ws_test",
            name="Test Inbox",
            description="Test description",
            email_address="test@example.com",
            team_members=["user1", "user2"],
            admins=["admin1"],
            created_at=now,
            updated_at=now,
        )
        result = inbox.to_dict()
        assert result["id"] == "inbox_123"
        assert result["name"] == "Test Inbox"
        assert result["team_members"] == ["user1", "user2"]


# =============================================================================
# Handler Class Tests
# =============================================================================


class TestSharedInboxHandler:
    """Tests for SharedInboxHandler class."""

    def test_handler_has_routes(self, shared_inbox_handler):
        """Handler should define routes."""
        assert hasattr(shared_inbox_handler, "ROUTES")
        assert len(shared_inbox_handler.ROUTES) > 0

    def test_handler_has_route_prefixes(self, shared_inbox_handler):
        """Handler should define route prefixes."""
        assert hasattr(shared_inbox_handler, "ROUTE_PREFIXES")
        assert len(shared_inbox_handler.ROUTE_PREFIXES) > 0

    def test_can_handle_inbox_route(self, shared_inbox_handler):
        """Handler should recognize inbox routes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/shared") is True

    def test_can_handle_routing_rules_route(self, shared_inbox_handler):
        """Handler should recognize routing rules routes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/routing/rules") is True

    def test_can_handle_inbox_id_route(self, shared_inbox_handler):
        """Handler should recognize inbox ID routes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/shared/inbox_123") is True

    def test_can_handle_rule_id_route(self, shared_inbox_handler):
        """Handler should recognize rule ID routes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/routing/rules/rule_123") is True

    def test_cannot_handle_unknown_route(self, shared_inbox_handler):
        """Handler should reject unknown routes."""
        assert shared_inbox_handler.can_handle("/api/v1/unknown") is False
        assert shared_inbox_handler.can_handle("/api/v1/debates") is False

    def test_handle_returns_none_for_base(self, shared_inbox_handler):
        """Base handle method should return None."""
        result = shared_inbox_handler.handle("/api/v1/inbox/shared", {}, None)
        assert result is None


# =============================================================================
# Inbox CRUD Handler Tests
# =============================================================================


class TestCreateSharedInbox:
    """Tests for create shared inbox handler."""

    @pytest.mark.asyncio
    async def test_create_inbox_success(self, clean_inbox_state):
        """Should create inbox successfully."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_create_shared_inbox(
                workspace_id="ws_test",
                name="New Inbox",
                description="Test description",
                email_address="new@example.com",
                team_members=["user1"],
                admins=["admin1"],
            )

        assert result["success"] is True
        assert "inbox" in result
        assert result["inbox"]["name"] == "New Inbox"
        assert result["inbox"]["workspace_id"] == "ws_test"

    @pytest.mark.asyncio
    async def test_create_inbox_generates_id(self, clean_inbox_state):
        """Created inbox should have generated ID."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_create_shared_inbox(
                workspace_id="ws_test",
                name="New Inbox",
            )

        assert result["success"] is True
        assert result["inbox"]["id"].startswith("inbox_")

    @pytest.mark.asyncio
    async def test_create_inbox_stores_in_memory(self, clean_inbox_state):
        """Created inbox should be stored in memory."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_create_shared_inbox(
                workspace_id="ws_test",
                name="New Inbox",
            )

        inbox_id = result["inbox"]["id"]
        assert inbox_id in _shared_inboxes


class TestListSharedInboxes:
    """Tests for list shared inboxes handler."""

    @pytest.mark.asyncio
    async def test_list_inboxes_empty(self, clean_inbox_state):
        """Should return empty list when no inboxes."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_list_shared_inboxes(workspace_id="ws_test")

        assert result["success"] is True
        assert result["inboxes"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_inboxes_returns_matching(self, sample_inbox):
        """Should return inboxes matching workspace."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_list_shared_inboxes(workspace_id="ws_test")

        assert result["success"] is True
        assert len(result["inboxes"]) == 1
        assert result["inboxes"][0]["id"] == sample_inbox.id

    @pytest.mark.asyncio
    async def test_list_inboxes_filters_by_user(self, sample_inbox):
        """Should filter inboxes by user membership."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            # User who is a member
            result = await handle_list_shared_inboxes(workspace_id="ws_test", user_id="user1")
            assert len(result["inboxes"]) == 1

            # User who is not a member
            result = await handle_list_shared_inboxes(
                workspace_id="ws_test", user_id="unknown_user"
            )
            assert len(result["inboxes"]) == 0


class TestGetSharedInbox:
    """Tests for get shared inbox handler."""

    @pytest.mark.asyncio
    async def test_get_inbox_success(self, sample_inbox):
        """Should return inbox details."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["inbox"]["id"] == sample_inbox.id
        assert result["inbox"]["name"] == "Test Inbox"

    @pytest.mark.asyncio
    async def test_get_inbox_not_found(self, clean_inbox_state):
        """Should return error for non-existent inbox."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestGetInboxMessages:
    """Tests for get inbox messages handler."""

    @pytest.mark.asyncio
    async def test_get_messages_success(self, sample_inbox, sample_message):
        """Should return messages in inbox."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_inbox_messages(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == sample_message.id

    @pytest.mark.asyncio
    async def test_get_messages_empty(self, sample_inbox):
        """Should return empty list when no messages."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_inbox_messages(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_get_messages_filters_by_status(self, sample_inbox, sample_message):
        """Should filter messages by status."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            # Filter by matching status
            result = await handle_get_inbox_messages(inbox_id=sample_inbox.id, status="open")
            assert len(result["messages"]) == 1

            # Filter by non-matching status
            result = await handle_get_inbox_messages(inbox_id=sample_inbox.id, status="closed")
            assert len(result["messages"]) == 0

    @pytest.mark.asyncio
    async def test_get_messages_pagination(self, sample_inbox, sample_message):
        """Should support pagination."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_inbox_messages(inbox_id=sample_inbox.id, limit=10, offset=0)

        assert result["success"] is True
        assert "limit" in result
        assert "offset" in result


# =============================================================================
# Message Assignment Tests
# =============================================================================


class TestAssignMessage:
    """Tests for assign message handler."""

    @pytest.mark.asyncio
    async def test_assign_message_success(self, sample_inbox, sample_message):
        """Should assign message to user."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            with patch(
                "aragora.server.handlers._shared_inbox_handler._log_activity", return_value=None
            ):
                result = await handle_assign_message(
                    inbox_id=sample_inbox.id,
                    message_id=sample_message.id,
                    assigned_to="user1",
                    assigned_by="admin1",
                )

        assert result["success"] is True
        assert result["message"]["assigned_to"] == "user1"
        assert result["message"]["status"] == MessageStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_assign_message_not_found(self, sample_inbox):
        """Should return error for non-existent message."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_assign_message(
                inbox_id=sample_inbox.id,
                message_id="nonexistent",
                assigned_to="user1",
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestUpdateMessageStatus:
    """Tests for update message status handler."""

    @pytest.mark.asyncio
    async def test_update_status_success(self, sample_inbox, sample_message):
        """Should update message status."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            with patch(
                "aragora.server.handlers._shared_inbox_handler._log_activity", return_value=None
            ):
                result = await handle_update_message_status(
                    inbox_id=sample_inbox.id,
                    message_id=sample_message.id,
                    status="in_progress",
                )

        assert result["success"] is True
        assert result["message"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_update_status_to_resolved(self, sample_inbox, sample_message):
        """Should set resolved timestamp when status is resolved."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            with patch(
                "aragora.server.handlers._shared_inbox_handler._log_activity", return_value=None
            ):
                result = await handle_update_message_status(
                    inbox_id=sample_inbox.id,
                    message_id=sample_message.id,
                    status="resolved",
                    updated_by="user1",
                )

        assert result["success"] is True
        assert result["message"]["status"] == "resolved"
        assert result["message"]["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_update_status_invalid(self, sample_inbox, sample_message):
        """Should return error for invalid status."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_update_message_status(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                status="invalid_status",
            )

        assert result["success"] is False


class TestAddMessageTag:
    """Tests for add message tag handler."""

    @pytest.mark.asyncio
    async def test_add_tag_success(self, sample_inbox, sample_message):
        """Should add tag to message."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag="important",
            )

        assert result["success"] is True
        assert "important" in result["message"]["tags"]

    @pytest.mark.asyncio
    async def test_add_duplicate_tag(self, sample_inbox, sample_message):
        """Should not add duplicate tag."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            # Add tag first time
            await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag="important",
            )
            # Add same tag again
            result = await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag="important",
            )

        assert result["success"] is True
        # Should only have one "important" tag
        assert result["message"]["tags"].count("important") == 1


# =============================================================================
# Routing Rules Tests
# =============================================================================


class TestCreateRoutingRule:
    """Tests for create routing rule handler."""

    @pytest.mark.asyncio
    async def test_create_rule_success(self, clean_inbox_state):
        """Should create routing rule successfully."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_create_routing_rule(
                workspace_id="ws_test",
                name="Test Rule",
                conditions=[{"field": "subject", "operator": "contains", "value": "urgent"}],
                actions=[{"type": "label", "target": "urgent"}],
            )

        assert result["success"] is True
        assert "rule" in result
        assert result["rule"]["name"] == "Test Rule"

    @pytest.mark.asyncio
    async def test_create_rule_generates_id(self, clean_inbox_state):
        """Created rule should have generated ID."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_create_routing_rule(
                workspace_id="ws_test",
                name="Test Rule",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "test"}],
            )

        assert result["success"] is True
        assert result["rule"]["id"].startswith("rule_")

    @pytest.mark.asyncio
    async def test_create_rule_with_priority(self, clean_inbox_state):
        """Should create rule with specified priority."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_create_routing_rule(
                workspace_id="ws_test",
                name="High Priority Rule",
                conditions=[{"field": "subject", "operator": "contains", "value": "urgent"}],
                actions=[{"type": "label", "target": "urgent"}],
                priority=1,
            )

        assert result["success"] is True
        assert result["rule"]["priority"] == 1


class TestListRoutingRules:
    """Tests for list routing rules handler."""

    @pytest.mark.asyncio
    async def test_list_rules_empty(self, clean_inbox_state):
        """Should return empty list when no rules."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_list_routing_rules(workspace_id="ws_test")

        assert result["success"] is True
        assert result["rules"] == []

    @pytest.mark.asyncio
    async def test_list_rules_returns_matching(self, sample_routing_rule):
        """Should return rules matching workspace."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_list_routing_rules(workspace_id="ws_test")

        assert result["success"] is True
        assert len(result["rules"]) == 1
        assert result["rules"][0]["id"] == sample_routing_rule.id

    @pytest.mark.asyncio
    async def test_list_rules_filters_enabled(self, sample_routing_rule):
        """Should filter by enabled status."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_list_routing_rules(workspace_id="ws_test", enabled_only=True)

        assert result["success"] is True
        assert len(result["rules"]) == 1


class TestUpdateRoutingRule:
    """Tests for update routing rule handler."""

    @pytest.mark.asyncio
    async def test_update_rule_success(self, sample_routing_rule):
        """Should update routing rule."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_update_routing_rule(
                rule_id=sample_routing_rule.id,
                updates={"name": "Updated Rule Name"},
            )

        assert result["success"] is True
        assert result["rule"]["name"] == "Updated Rule Name"

    @pytest.mark.asyncio
    async def test_update_rule_enabled(self, sample_routing_rule):
        """Should update rule enabled status."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_update_routing_rule(
                rule_id=sample_routing_rule.id,
                updates={"enabled": False},
            )

        assert result["success"] is True
        assert result["rule"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_update_rule_not_found(self, clean_inbox_state):
        """Should return error for non-existent rule."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_update_routing_rule(
                rule_id="nonexistent",
                updates={"name": "Test"},
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestDeleteRoutingRule:
    """Tests for delete routing rule handler."""

    @pytest.mark.asyncio
    async def test_delete_rule_success(self, sample_routing_rule):
        """Should delete routing rule."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_delete_routing_rule(rule_id=sample_routing_rule.id)

        assert result["success"] is True
        assert sample_routing_rule.id not in _routing_rules

    @pytest.mark.asyncio
    async def test_delete_rule_not_found(self, clean_inbox_state):
        """Should return error for non-existent rule."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await handle_delete_routing_rule(rule_id="nonexistent")

        assert result["success"] is False


class TestTestRoutingRule:
    """Tests for test routing rule handler."""

    @pytest.mark.asyncio
    async def test_test_rule_success(self, sample_routing_rule):
        """Should test routing rule."""
        result = await handle_test_routing_rule(
            rule_id=sample_routing_rule.id,
            workspace_id="ws_test",
        )

        assert result["success"] is True
        assert "rule_id" in result
        assert "match_count" in result
        assert "rule" in result

    @pytest.mark.asyncio
    async def test_test_rule_not_found(self, clean_inbox_state):
        """Should return error for non-existent rule."""
        result = await handle_test_routing_rule(
            rule_id="nonexistent",
            workspace_id="ws_test",
        )

        assert result["success"] is False


# =============================================================================
# Routing Rule Application Tests
# =============================================================================


class TestApplyRoutingRules:
    """Tests for apply_routing_rules_to_message function."""

    @pytest.mark.asyncio
    async def test_apply_rules_no_match(self, sample_inbox, sample_message, sample_routing_rule):
        """Should not apply rules when no conditions match."""
        # Update message to have non-matching subject
        sample_message.subject = "Normal email"  # Doesn't contain "urgent"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id="ws_test",
        )

        assert result["applied"] is False
        assert result["rules_matched"] == 0

    @pytest.mark.asyncio
    async def test_apply_rules_with_match(self, sample_inbox, sample_message, sample_routing_rule):
        """Should apply rules when conditions match."""
        # Update message to have matching subject
        sample_message.subject = "This is urgent!"  # Contains "urgent"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id="ws_test",
        )

        assert result["applied"] is True
        assert result["rules_matched"] >= 1


# =============================================================================
# Handler Method Tests (HTTP layer)
# =============================================================================


class TestHandlerPostSharedInbox:
    """Tests for POST /api/v1/inbox/shared handler method."""

    @pytest.mark.asyncio
    async def test_post_inbox_requires_workspace_id(self, shared_inbox_handler):
        """Should require workspace_id."""
        result = await shared_inbox_handler.handle_post_shared_inbox({"name": "Test Inbox"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_inbox_requires_name(self, shared_inbox_handler):
        """Should require name."""
        result = await shared_inbox_handler.handle_post_shared_inbox({"workspace_id": "ws_test"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_inbox_success(self, shared_inbox_handler, clean_inbox_state):
        """Should create inbox successfully."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await shared_inbox_handler.handle_post_shared_inbox(
                {"workspace_id": "ws_test", "name": "Test Inbox"}
            )
        assert result.status_code == 200


class TestHandlerGetSharedInboxes:
    """Tests for GET /api/v1/inbox/shared handler method."""

    @pytest.mark.asyncio
    async def test_get_inboxes_requires_workspace_id(self, shared_inbox_handler):
        """Should require workspace_id."""
        result = await shared_inbox_handler.handle_get_shared_inboxes({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_inboxes_success(self, shared_inbox_handler, clean_inbox_state):
        """Should return inboxes successfully."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await shared_inbox_handler.handle_get_shared_inboxes(
                {"workspace_id": "ws_test"}
            )
        assert result.status_code == 200


class TestHandlerPostRoutingRule:
    """Tests for POST /api/v1/inbox/routing/rules handler method."""

    @pytest.mark.asyncio
    async def test_post_rule_requires_fields(self, shared_inbox_handler):
        """Should require all required fields."""
        result = await shared_inbox_handler.handle_post_routing_rule({"name": "Test Rule"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_rule_success(self, shared_inbox_handler, clean_inbox_state):
        """Should create rule successfully."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await shared_inbox_handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws_test",
                    "name": "Test Rule",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "urgent"}],
                    "actions": [{"type": "label", "target": "urgent"}],
                }
            )
        assert result.status_code == 200


class TestHandlerGetRoutingRules:
    """Tests for GET /api/v1/inbox/routing/rules handler method."""

    @pytest.mark.asyncio
    async def test_get_rules_requires_workspace_id(self, shared_inbox_handler):
        """Should require workspace_id."""
        result = await shared_inbox_handler.handle_get_routing_rules({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_rules_success(self, shared_inbox_handler, clean_inbox_state):
        """Should return rules successfully."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            result = await shared_inbox_handler.handle_get_routing_rules(
                {"workspace_id": "ws_test"}
            )
        assert result.status_code == 200


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in shared inbox handlers."""

    @pytest.mark.asyncio
    async def test_handles_store_exception(self, clean_inbox_state):
        """Should handle store exceptions gracefully."""
        mock_store = MagicMock()
        # Use RuntimeError (handled by the handler) rather than bare Exception
        mock_store.create_shared_inbox.side_effect = RuntimeError("Store error")

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_store", return_value=mock_store
        ):
            result = await handle_create_shared_inbox(
                workspace_id="ws_test",
                name="Test Inbox",
            )

        # Should succeed (in-memory fallback) even when store fails
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handles_invalid_status(self, sample_inbox, sample_message):
        """Should handle invalid status gracefully."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_update_message_status(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                status="invalid_status",
            )

        assert result["success"] is False


class TestThreadSafety:
    """Tests for thread safety in shared inbox operations."""

    @pytest.mark.asyncio
    async def test_concurrent_inbox_creation(self, clean_inbox_state):
        """Should handle concurrent inbox creation."""
        import asyncio

        async def create_inbox(i):
            with patch(
                "aragora.server.handlers._shared_inbox_handler._get_store", return_value=None
            ):
                return await handle_create_shared_inbox(
                    workspace_id="ws_test",
                    name=f"Inbox {i}",
                )

        results = await asyncio.gather(*[create_inbox(i) for i in range(5)])

        # All should succeed
        assert all(r["success"] for r in results)
        # All should have unique IDs
        ids = [r["inbox"]["id"] for r in results]
        assert len(set(ids)) == 5


# =============================================================================
# Participant Handling Tests
# =============================================================================


class TestParticipantHandling:
    """Tests for team member and admin participant handling."""

    @pytest.mark.asyncio
    async def test_inbox_team_members_list(self, sample_inbox):
        """Should store and return team members correctly."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert "user1" in result["inbox"]["team_members"]
        assert "user2" in result["inbox"]["team_members"]

    @pytest.mark.asyncio
    async def test_inbox_admins_list(self, sample_inbox):
        """Should store and return admins correctly."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert "admin1" in result["inbox"]["admins"]

    @pytest.mark.asyncio
    async def test_admin_can_access_inbox(self, sample_inbox):
        """Admin should be able to access the inbox."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_list_shared_inboxes(
                workspace_id="ws_test",
                user_id="admin1",
            )

        assert result["success"] is True
        assert len(result["inboxes"]) == 1

    @pytest.mark.asyncio
    async def test_create_inbox_without_team_members(self, clean_inbox_state):
        """Should create inbox with empty team members list."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_create_shared_inbox(
                workspace_id="ws_test",
                name="Empty Team Inbox",
            )

        assert result["success"] is True
        assert result["inbox"]["team_members"] == []
        assert result["inbox"]["admins"] == []


# =============================================================================
# Message Routing Tests
# =============================================================================


class TestMessageRouting:
    """Tests for message routing with different conditions."""

    @pytest.fixture
    def routing_rule_from_domain(self, clean_inbox_state):
        """Create a routing rule that matches sender domain."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_domain",
            name="Internal Domain Rule",
            workspace_id="ws_test",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SENDER_DOMAIN,
                    operator=RuleConditionOperator.EQUALS,
                    value="company.com",
                )
            ],
            condition_logic="AND",
            actions=[
                RuleAction(type=RuleActionType.LABEL, target="internal"),
            ],
            priority=2,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with _storage_lock:
            _routing_rules[rule.id] = rule
        return rule

    @pytest.fixture
    def routing_rule_assign(self, clean_inbox_state):
        """Create a routing rule that assigns to a user."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_assign",
            name="Assign to Support",
            workspace_id="ws_test",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="support",
                )
            ],
            condition_logic="AND",
            actions=[
                RuleAction(type=RuleActionType.ASSIGN, target="support_user"),
            ],
            priority=1,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with _storage_lock:
            _routing_rules[rule.id] = rule
        return rule

    @pytest.mark.asyncio
    async def test_route_by_sender_domain(self, sample_inbox, routing_rule_from_domain):
        """Should route messages based on sender domain."""
        now = datetime.now(timezone.utc)
        message = SharedInboxMessage(
            id="msg_domain_test",
            inbox_id=sample_inbox.id,
            email_id="email_domain",
            subject="Internal Update",
            from_address="employee@company.com",
            to_addresses=["inbox@example.com"],
            snippet="Internal update...",
            received_at=now,
        )
        with _storage_lock:
            _inbox_messages[sample_inbox.id][message.id] = message

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=message,
            workspace_id="ws_test",
        )

        assert result["applied"] is True
        assert "internal" in message.tags

    @pytest.mark.asyncio
    async def test_route_with_assign_action(self, sample_inbox, routing_rule_assign):
        """Should assign message to user based on routing rule."""
        now = datetime.now(timezone.utc)
        message = SharedInboxMessage(
            id="msg_assign_test",
            inbox_id=sample_inbox.id,
            email_id="email_support",
            subject="Need support help",
            from_address="customer@example.com",
            to_addresses=["inbox@example.com"],
            snippet="I need support...",
            received_at=now,
        )
        with _storage_lock:
            _inbox_messages[sample_inbox.id][message.id] = message

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=message,
            workspace_id="ws_test",
        )

        assert result["applied"] is True
        assert message.assigned_to == "support_user"
        assert message.status == MessageStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_route_with_or_condition(self, sample_inbox, clean_inbox_state):
        """Should match messages using OR condition logic."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_or",
            name="Urgent OR High Priority",
            workspace_id="ws_test",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                ),
                RuleCondition(
                    field=RuleConditionField.PRIORITY,
                    operator=RuleConditionOperator.EQUALS,
                    value="high",
                ),
            ],
            condition_logic="OR",
            actions=[
                RuleAction(type=RuleActionType.LABEL, target="priority"),
            ],
            priority=1,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with _storage_lock:
            _routing_rules[rule.id] = rule

        # Message matches first condition only
        message = SharedInboxMessage(
            id="msg_or_test",
            inbox_id=sample_inbox.id,
            email_id="email_or",
            subject="This is urgent!",
            from_address="user@example.com",
            to_addresses=["inbox@example.com"],
            snippet="...",
            received_at=now,
            priority="normal",  # Second condition doesn't match
        )
        with _storage_lock:
            _inbox_messages[sample_inbox.id][message.id] = message

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=message,
            workspace_id="ws_test",
        )

        assert result["applied"] is True
        assert "priority" in message.tags


# =============================================================================
# Additional Error Handling Tests
# =============================================================================


class TestExtendedErrorHandling:
    """Extended error handling tests for edge cases."""

    @pytest.mark.asyncio
    async def test_assign_to_inbox_not_found(self, clean_inbox_state):
        """Should fail to assign message when inbox doesn't exist."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_assign_message(
                inbox_id="nonexistent_inbox",
                message_id="msg_123",
                assigned_to="user1",
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_add_tag_to_nonexistent_message(self, sample_inbox):
        """Should fail to add tag when message doesn't exist."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id="nonexistent_msg",
                tag="important",
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_message(self, sample_inbox):
        """Should fail to update status when message doesn't exist."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_update_message_status(
                inbox_id=sample_inbox.id,
                message_id="nonexistent_msg",
                status="resolved",
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Handler Methods Validation Tests
# =============================================================================


class TestHandlerMethodValidation:
    """Tests for handler method parameter validation."""

    @pytest.mark.asyncio
    async def test_post_assign_requires_assigned_to(self, shared_inbox_handler, sample_inbox):
        """POST assign should require assigned_to field."""
        result = await shared_inbox_handler.handle_post_assign_message(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_status_requires_status(self, shared_inbox_handler, sample_inbox):
        """POST status should require status field."""
        result = await shared_inbox_handler.handle_post_update_status(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_tag_requires_tag(self, shared_inbox_handler, sample_inbox):
        """POST tag should require tag field."""
        result = await shared_inbox_handler.handle_post_add_tag(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_test_rule_requires_workspace_id(self, shared_inbox_handler):
        """POST test rule should require workspace_id."""
        result = await shared_inbox_handler.handle_post_test_routing_rule(
            data={},
            rule_id="rule_123",
        )
        assert result.status_code == 400


# =============================================================================
# Inbox Message Counts Tests
# =============================================================================


class TestInboxMessageCounts:
    """Tests for inbox message and unread counts."""

    @pytest.mark.asyncio
    async def test_inbox_message_count_updated(self, sample_inbox, sample_message):
        """Should update message count when getting inbox."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["inbox"]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_inbox_unread_count_updated(self, sample_inbox, sample_message):
        """Should update unread count based on open messages."""
        # sample_message has status=OPEN
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["inbox"]["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_unread_count_decreases_when_assigned(self, sample_inbox, sample_message):
        """Unread count should decrease when message is assigned."""
        with patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None):
            with patch(
                "aragora.server.handlers._shared_inbox_handler._log_activity", return_value=None
            ):
                # Assign the message (changes status from OPEN to ASSIGNED)
                await handle_assign_message(
                    inbox_id=sample_inbox.id,
                    message_id=sample_message.id,
                    assigned_to="user1",
                )

                # Check inbox counts
                result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["inbox"]["unread_count"] == 0


# =============================================================================
# Rule Priority Tests
# =============================================================================


class TestRulePriority:
    """Tests for routing rule priority ordering."""

    @pytest.mark.asyncio
    async def test_rules_sorted_by_priority(self, clean_inbox_state):
        """Rules should be returned sorted by priority."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store", return_value=None
        ):
            # Create rules with different priorities
            await handle_create_routing_rule(
                workspace_id="ws_test",
                name="Low Priority",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "low"}],
                priority=10,
            )
            await handle_create_routing_rule(
                workspace_id="ws_test",
                name="High Priority",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "high"}],
                priority=1,
            )
            await handle_create_routing_rule(
                workspace_id="ws_test",
                name="Medium Priority",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "medium"}],
                priority=5,
            )

            result = await handle_list_routing_rules(workspace_id="ws_test")

        assert result["success"] is True
        priorities = [r["priority"] for r in result["rules"]]
        assert priorities == sorted(priorities)  # Should be ascending


# =============================================================================
# Handler User ID Extraction Tests
# =============================================================================


class TestHandlerUserIdExtraction:
    """Tests for user ID extraction from handler context."""

    def test_get_user_id_from_context(self, shared_inbox_handler):
        """Should extract user ID from auth context."""
        user_id = shared_inbox_handler._get_user_id()
        assert user_id == "test_user"

    def test_get_user_id_fallback_to_default(self):
        """Should fallback to 'default' when no auth context."""
        ctx = {}
        handler = SharedInboxHandler(ctx)
        user_id = handler._get_user_id()
        assert user_id == "default"

    def test_get_user_id_with_none_context(self):
        """Should handle None auth context gracefully."""
        ctx = {"auth_context": None}
        handler = SharedInboxHandler(ctx)
        user_id = handler._get_user_id()
        assert user_id == "default"
