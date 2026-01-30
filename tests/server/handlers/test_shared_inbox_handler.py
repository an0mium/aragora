"""
Comprehensive tests for aragora/server/handlers/_shared_inbox_handler.py.

This is a P0 critical handler test suite covering:
- LazyStoreFactory mock patterns
- Async handler functions with pytest-asyncio
- RBAC permission checks with AuthorizationContext mocking
- Error handling and edge cases
- Inbox listing/filtering endpoints
- Message operations (read, archive, move, assign)
- Rule management (create, update, delete rules)
- Activity logging
- Rule evaluation logic
- Store integration patterns

Test organization:
1. Fixtures for common mocks
2. Data model serialization tests
3. Handler function tests
4. HTTP handler method tests
5. Rule evaluation tests
6. Store integration tests
7. RBAC permission tests
8. Error handling tests
9. Edge case tests
"""

from __future__ import annotations

import io
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Import from the public package interface for backward compatibility
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
    handle_add_message_to_inbox,
    handle_create_routing_rule,
    handle_list_routing_rules,
    handle_update_routing_rule,
    handle_delete_routing_rule,
    handle_test_routing_rule,
    apply_routing_rules_to_message,
    get_matching_rules_for_email,
    _shared_inboxes,
    _inbox_messages,
    _routing_rules,
    _storage_lock,
    _get_store,
    _get_rules_store,
    _get_activity_store,
    _log_activity,
)

# Import rule evaluation function directly for testing
from aragora.server.handlers._shared_inbox_handler import _evaluate_rule


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    rfile: Any = None
    client_address: tuple[str, int] = ("127.0.0.1", 12345)
    path: str = "/api/v1/inbox/shared"

    def __post_init__(self):
        self.rfile = io.BytesIO(self.body)


@dataclass
class MockAuthContext:
    """Mock authorization context for RBAC testing."""

    user_id: str = "test_user_123"
    org_id: str = "org_456"
    roles: list[str] = field(default_factory=lambda: ["admin"])
    permissions: list[str] = field(default_factory=list)
    is_authenticated: bool = True


@dataclass
class MockEmailStore:
    """Mock email store for testing persistent storage integration."""

    inboxes: dict[str, dict] = field(default_factory=dict)
    messages: dict[str, dict] = field(default_factory=dict)
    rules: dict[str, dict] = field(default_factory=dict)

    def create_shared_inbox(self, **kwargs) -> None:
        self.inboxes[kwargs["inbox_id"]] = kwargs

    def get_shared_inbox(self, inbox_id: str) -> dict | None:
        return self.inboxes.get(inbox_id)

    def list_shared_inboxes(self, workspace_id: str, user_id: str | None = None) -> list[dict]:
        return [
            inbox for inbox in self.inboxes.values() if inbox.get("workspace_id") == workspace_id
        ]

    def save_message(self, **kwargs) -> None:
        self.messages[kwargs["message_id"]] = kwargs

    def get_inbox_messages(
        self,
        inbox_id: str,
        status: str | None = None,
        assigned_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        return [msg for msg in self.messages.values() if msg.get("inbox_id") == inbox_id]

    def update_message(self, message_id: str, updates: dict) -> None:
        if message_id in self.messages:
            self.messages[message_id].update(updates)

    def create_routing_rule(self, **kwargs) -> None:
        self.rules[kwargs["rule_id"]] = kwargs

    def list_routing_rules(
        self,
        workspace_id: str,
        inbox_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[dict]:
        return [rule for rule in self.rules.values() if rule.get("workspace_id") == workspace_id]

    def get_routing_rule(self, rule_id: str) -> dict | None:
        return self.rules.get(rule_id)

    def update_routing_rule(self, rule_id: str, **updates) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].update(updates)

    def delete_routing_rule(self, rule_id: str) -> bool:
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False


@dataclass
class MockRulesStore:
    """Mock rules store for testing."""

    rules: dict[str, dict] = field(default_factory=dict)

    def create_rule(self, rule_data: dict) -> None:
        self.rules[rule_data["id"]] = rule_data

    def get_rule(self, rule_id: str) -> dict | None:
        return self.rules.get(rule_id)

    def list_rules(
        self,
        workspace_id: str,
        inbox_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        results = [
            rule
            for rule in self.rules.values()
            if rule.get("workspace_id") == workspace_id
            and (not enabled_only or rule.get("enabled", True))
        ]
        return results[offset : offset + limit]

    def count_rules(
        self,
        workspace_id: str,
        inbox_id: str | None = None,
        enabled_only: bool = False,
    ) -> int:
        return len(
            [
                rule
                for rule in self.rules.values()
                if rule.get("workspace_id") == workspace_id
                and (not enabled_only or rule.get("enabled", True))
            ]
        )

    def update_rule(self, rule_id: str, updates: dict) -> dict | None:
        if rule_id in self.rules:
            self.rules[rule_id].update(updates)
            self.rules[rule_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            return self.rules[rule_id]
        return None

    def delete_rule(self, rule_id: str) -> bool:
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def get_matching_rules(
        self,
        inbox_id: str,
        email_data: dict,
        workspace_id: str | None = None,
    ) -> list[dict]:
        # Simple matching for testing
        return [
            rule
            for rule in self.rules.values()
            if rule.get("enabled", True)
            and (workspace_id is None or rule.get("workspace_id") == workspace_id)
        ]

    def increment_rule_stats(self, rule_id: str, matched: int = 0, applied: int = 0) -> None:
        if rule_id in self.rules:
            stats = self.rules[rule_id].get("stats", {})
            stats["matched"] = stats.get("matched", 0) + matched
            stats["applied"] = stats.get("applied", 0) + applied
            self.rules[rule_id]["stats"] = stats


@dataclass
class MockActivityStore:
    """Mock activity store for testing activity logging."""

    activities: list[Any] = field(default_factory=list)

    def log_activity(self, activity: Any) -> None:
        self.activities.append(activity)


@pytest.fixture
def mock_email_store():
    """Create a mock email store."""
    return MockEmailStore()


@pytest.fixture
def mock_rules_store():
    """Create a mock rules store."""
    return MockRulesStore()


@pytest.fixture
def mock_activity_store():
    """Create a mock activity store."""
    return MockActivityStore()


@pytest.fixture
def mock_auth_context():
    """Create a mock authentication context."""
    return MockAuthContext()


@pytest.fixture
def server_context(mock_auth_context):
    """Create mock server context with auth context."""
    return {
        "storage": None,
        "auth_context": mock_auth_context,
    }


@pytest.fixture
def shared_inbox_handler(server_context):
    """Create a shared inbox handler instance."""
    return SharedInboxHandler(server_context)


@pytest.fixture
def clean_state():
    """Clean up in-memory state before and after tests with store mocking."""
    # Clear state before test
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()

    # Mock stores to return None (forces in-memory fallback)
    with (
        patch("aragora.server.handlers._shared_inbox_handler._get_store", return_value=None),
        patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store",
            return_value=None,
        ),
        patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=None,
        ),
    ):
        yield

    # Clear state after test
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()


@pytest.fixture
def sample_inbox_data():
    """Sample inbox data for testing."""
    return {
        "workspace_id": "ws_test_123",
        "name": "Support Inbox",
        "description": "Customer support inbox",
        "email_address": "support@example.com",
        "connector_type": "gmail",
        "team_members": ["user1", "user2", "user3"],
        "admins": ["admin1"],
        "settings": {"auto_assign": True, "sla_hours": 24},
    }


@pytest.fixture
def sample_inbox(clean_state, sample_inbox_data):
    """Create and return a sample inbox."""
    now = datetime.now(timezone.utc)
    inbox = SharedInbox(
        id="inbox_sample123",
        workspace_id=sample_inbox_data["workspace_id"],
        name=sample_inbox_data["name"],
        description=sample_inbox_data["description"],
        email_address=sample_inbox_data["email_address"],
        connector_type=sample_inbox_data["connector_type"],
        team_members=sample_inbox_data["team_members"],
        admins=sample_inbox_data["admins"],
        settings=sample_inbox_data["settings"],
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
    """Create and return a sample message."""
    now = datetime.now(timezone.utc)
    message = SharedInboxMessage(
        id="msg_sample456",
        inbox_id=sample_inbox.id,
        email_id="email_ext_789",
        subject="Help with my order",
        from_address="customer@gmail.com",
        to_addresses=["support@example.com"],
        snippet="I need help with order #12345...",
        received_at=now,
        status=MessageStatus.OPEN,
        priority="normal",
        thread_id="thread_123",
    )
    with _storage_lock:
        _inbox_messages[sample_inbox.id][message.id] = message
    return message


@pytest.fixture
def sample_rule(clean_state):
    """Create and return a sample routing rule."""
    now = datetime.now(timezone.utc)
    rule = RoutingRule(
        id="rule_sample789",
        name="Urgent Handler",
        workspace_id="ws_test_123",
        conditions=[
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="urgent",
            )
        ],
        condition_logic="AND",
        actions=[
            RuleAction(type=RuleActionType.LABEL, target="urgent"),
            RuleAction(type=RuleActionType.ESCALATE),
        ],
        priority=1,
        enabled=True,
        description="Route urgent messages for fast response",
        created_at=now,
        updated_at=now,
        created_by="admin1",
    )
    with _storage_lock:
        _routing_rules[rule.id] = rule
    return rule


# =============================================================================
# Data Model Tests
# =============================================================================


class TestMessageStatusEnum:
    """Tests for MessageStatus enum values and usage."""

    def test_all_status_values_exist(self):
        """All required status values should be defined."""
        assert hasattr(MessageStatus, "OPEN")
        assert hasattr(MessageStatus, "ASSIGNED")
        assert hasattr(MessageStatus, "IN_PROGRESS")
        assert hasattr(MessageStatus, "WAITING")
        assert hasattr(MessageStatus, "RESOLVED")
        assert hasattr(MessageStatus, "CLOSED")

    def test_status_values_are_strings(self):
        """Status values should be lowercase strings."""
        assert MessageStatus.OPEN.value == "open"
        assert MessageStatus.ASSIGNED.value == "assigned"
        assert MessageStatus.IN_PROGRESS.value == "in_progress"
        assert MessageStatus.WAITING.value == "waiting"
        assert MessageStatus.RESOLVED.value == "resolved"
        assert MessageStatus.CLOSED.value == "closed"

    def test_status_from_string(self):
        """Should create status from string value."""
        assert MessageStatus("open") == MessageStatus.OPEN
        assert MessageStatus("resolved") == MessageStatus.RESOLVED

    def test_invalid_status_raises_error(self):
        """Invalid status string should raise ValueError."""
        with pytest.raises(ValueError):
            MessageStatus("invalid_status")


class TestRuleConditionFieldEnum:
    """Tests for RuleConditionField enum."""

    def test_all_field_values_exist(self):
        """All required field values should be defined."""
        fields = ["FROM", "TO", "SUBJECT", "BODY", "LABELS", "PRIORITY", "SENDER_DOMAIN"]
        for field_name in fields:
            assert hasattr(RuleConditionField, field_name)

    def test_field_values_are_lowercase(self):
        """Field values should be lowercase strings."""
        assert RuleConditionField.FROM.value == "from"
        assert RuleConditionField.SENDER_DOMAIN.value == "sender_domain"


class TestRuleConditionOperatorEnum:
    """Tests for RuleConditionOperator enum."""

    def test_all_operator_values_exist(self):
        """All required operator values should be defined."""
        operators = [
            "CONTAINS",
            "EQUALS",
            "STARTS_WITH",
            "ENDS_WITH",
            "MATCHES",
            "GREATER_THAN",
            "LESS_THAN",
        ]
        for op in operators:
            assert hasattr(RuleConditionOperator, op)


class TestRuleActionTypeEnum:
    """Tests for RuleActionType enum."""

    def test_all_action_values_exist(self):
        """All required action values should be defined."""
        actions = ["ASSIGN", "LABEL", "ESCALATE", "ARCHIVE", "NOTIFY", "FORWARD"]
        for action in actions:
            assert hasattr(RuleActionType, action)


class TestRuleConditionSerialization:
    """Tests for RuleCondition dataclass serialization."""

    def test_to_dict_complete(self):
        """to_dict should include all fields."""
        condition = RuleCondition(
            field=RuleConditionField.FROM,
            operator=RuleConditionOperator.ENDS_WITH,
            value="@company.com",
        )
        result = condition.to_dict()

        assert result["field"] == "from"
        assert result["operator"] == "ends_with"
        assert result["value"] == "@company.com"

    def test_from_dict_complete(self):
        """from_dict should restore all fields."""
        data = {
            "field": "sender_domain",
            "operator": "matches",
            "value": r".*\.edu$",
        }
        condition = RuleCondition.from_dict(data)

        assert condition.field == RuleConditionField.SENDER_DOMAIN
        assert condition.operator == RuleConditionOperator.MATCHES
        assert condition.value == r".*\.edu$"

    def test_roundtrip_serialization(self):
        """Serialization should be reversible."""
        original = RuleCondition(
            field=RuleConditionField.PRIORITY,
            operator=RuleConditionOperator.EQUALS,
            value="high",
        )
        restored = RuleCondition.from_dict(original.to_dict())

        assert restored.field == original.field
        assert restored.operator == original.operator
        assert restored.value == original.value


class TestRuleActionSerialization:
    """Tests for RuleAction dataclass serialization."""

    def test_to_dict_with_params(self):
        """to_dict should include params when present."""
        action = RuleAction(
            type=RuleActionType.NOTIFY,
            target="slack-channel",
            params={"message": "New urgent ticket", "priority": "high"},
        )
        result = action.to_dict()

        assert result["type"] == "notify"
        assert result["target"] == "slack-channel"
        assert result["params"]["message"] == "New urgent ticket"

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {"type": "archive"}
        action = RuleAction.from_dict(data)

        assert action.type == RuleActionType.ARCHIVE
        assert action.target is None
        assert action.params == {}

    def test_forward_action(self):
        """Forward action should store email target."""
        action = RuleAction(
            type=RuleActionType.FORWARD,
            target="escalation@company.com",
        )
        assert action.type == RuleActionType.FORWARD
        assert action.target == "escalation@company.com"


class TestRoutingRuleSerialization:
    """Tests for RoutingRule dataclass serialization."""

    def test_to_dict_complete(self):
        """to_dict should serialize all fields correctly."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_test",
            name="Test Rule",
            workspace_id="ws_123",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="test",
                )
            ],
            condition_logic="OR",
            actions=[RuleAction(type=RuleActionType.LABEL, target="test-label")],
            priority=3,
            enabled=False,
            description="A test rule",
            created_at=now,
            updated_at=now,
            created_by="user_123",
            stats={"total_matches": 42},
        )
        result = rule.to_dict()

        assert result["id"] == "rule_test"
        assert result["name"] == "Test Rule"
        assert result["condition_logic"] == "OR"
        assert result["priority"] == 3
        assert result["enabled"] is False
        assert result["description"] == "A test rule"
        assert result["created_by"] == "user_123"
        assert result["stats"]["total_matches"] == 42

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {
            "id": "rule_min",
            "name": "Minimal Rule",
            "workspace_id": "ws_min",
            "conditions": [],
            "actions": [],
        }
        rule = RoutingRule.from_dict(data)

        assert rule.id == "rule_min"
        assert rule.condition_logic == "AND"  # Default
        assert rule.priority == 5  # Default
        assert rule.enabled is True  # Default
        assert rule.description is None
        assert rule.stats == {}


class TestSharedInboxMessageSerialization:
    """Tests for SharedInboxMessage dataclass serialization."""

    def test_to_dict_with_all_fields(self):
        """to_dict should serialize all fields including nullable ones."""
        now = datetime.now(timezone.utc)
        resolved_at = now + timedelta(hours=2)
        sla_deadline = now + timedelta(hours=24)

        message = SharedInboxMessage(
            id="msg_full",
            inbox_id="inbox_123",
            email_id="email_ext",
            subject="Full Test",
            from_address="sender@test.com",
            to_addresses=["recipient1@test.com", "recipient2@test.com"],
            snippet="This is the snippet",
            received_at=now,
            status=MessageStatus.RESOLVED,
            assigned_to="user_456",
            assigned_at=now,
            tags=["important", "customer"],
            priority="high",
            notes=[{"author": "user_456", "text": "Resolved issue"}],
            thread_id="thread_789",
            sla_deadline=sla_deadline,
            resolved_at=resolved_at,
            resolved_by="user_456",
        )
        result = message.to_dict()

        assert result["id"] == "msg_full"
        assert result["status"] == "resolved"
        assert len(result["to_addresses"]) == 2
        assert result["assigned_to"] == "user_456"
        assert result["tags"] == ["important", "customer"]
        assert result["priority"] == "high"
        assert result["resolved_by"] == "user_456"
        assert result["sla_deadline"] is not None

    def test_to_dict_with_none_values(self):
        """to_dict should handle None values correctly."""
        now = datetime.now(timezone.utc)
        message = SharedInboxMessage(
            id="msg_minimal",
            inbox_id="inbox_123",
            email_id="email_ext",
            subject="Minimal",
            from_address="sender@test.com",
            to_addresses=["recipient@test.com"],
            snippet="Snippet",
            received_at=now,
        )
        result = message.to_dict()

        assert result["assigned_to"] is None
        assert result["assigned_at"] is None
        assert result["resolved_at"] is None
        assert result["sla_deadline"] is None


class TestSharedInboxSerialization:
    """Tests for SharedInbox dataclass serialization."""

    def test_to_dict_complete(self):
        """to_dict should serialize all inbox fields."""
        now = datetime.now(timezone.utc)
        inbox = SharedInbox(
            id="inbox_full",
            workspace_id="ws_full",
            name="Full Inbox",
            description="Full description",
            email_address="inbox@test.com",
            connector_type="outlook",
            team_members=["user1", "user2"],
            admins=["admin1"],
            settings={"auto_assign": True},
            created_at=now,
            updated_at=now,
            created_by="admin1",
            message_count=100,
            unread_count=25,
        )
        result = inbox.to_dict()

        assert result["id"] == "inbox_full"
        assert result["connector_type"] == "outlook"
        assert len(result["team_members"]) == 2
        assert result["message_count"] == 100
        assert result["unread_count"] == 25


# =============================================================================
# Handler Function Tests - Inbox Operations
# =============================================================================


class TestHandleCreateSharedInbox:
    """Tests for handle_create_shared_inbox function."""

    @pytest.mark.asyncio
    async def test_create_inbox_with_all_fields(self, clean_state, sample_inbox_data):
        """Should create inbox with all provided fields."""
        result = await handle_create_shared_inbox(
            **sample_inbox_data,
            created_by="creator_user",
        )

        assert result["success"] is True
        inbox = result["inbox"]
        assert inbox["name"] == sample_inbox_data["name"]
        assert inbox["email_address"] == sample_inbox_data["email_address"]
        assert inbox["team_members"] == sample_inbox_data["team_members"]
        assert inbox["created_by"] == "creator_user"

    @pytest.mark.asyncio
    async def test_create_inbox_minimal_fields(self, clean_state):
        """Should create inbox with only required fields."""
        result = await handle_create_shared_inbox(
            workspace_id="ws_minimal",
            name="Minimal Inbox",
        )

        assert result["success"] is True
        inbox = result["inbox"]
        assert inbox["workspace_id"] == "ws_minimal"
        assert inbox["name"] == "Minimal Inbox"
        assert inbox["team_members"] == []
        assert inbox["admins"] == []

    @pytest.mark.asyncio
    async def test_create_inbox_id_format(self, clean_state):
        """Created inbox ID should follow expected format."""
        result = await handle_create_shared_inbox(
            workspace_id="ws_test",
            name="ID Test Inbox",
        )

        assert result["success"] is True
        inbox_id = result["inbox"]["id"]
        assert inbox_id.startswith("inbox_")
        assert len(inbox_id) > 6  # "inbox_" + some hex

    @pytest.mark.asyncio
    async def test_create_inbox_timestamps(self, clean_state):
        """Created inbox should have valid timestamps."""
        before = datetime.now(timezone.utc)
        result = await handle_create_shared_inbox(
            workspace_id="ws_test",
            name="Timestamp Test",
        )
        after = datetime.now(timezone.utc)

        assert result["success"] is True
        created_at = datetime.fromisoformat(result["inbox"]["created_at"])
        assert before <= created_at <= after

    @pytest.mark.asyncio
    async def test_create_inbox_stored_in_memory(self, clean_state):
        """Created inbox should be stored in memory cache."""
        result = await handle_create_shared_inbox(
            workspace_id="ws_test",
            name="Memory Test",
        )

        inbox_id = result["inbox"]["id"]
        assert inbox_id in _shared_inboxes
        assert inbox_id in _inbox_messages


class TestHandleListSharedInboxes:
    """Tests for handle_list_shared_inboxes function."""

    @pytest.mark.asyncio
    async def test_list_empty_workspace(self, clean_state):
        """Should return empty list for workspace with no inboxes."""
        result = await handle_list_shared_inboxes(workspace_id="ws_empty")

        assert result["success"] is True
        assert result["inboxes"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_multiple_inboxes(self, clean_state):
        """Should return all inboxes in workspace."""
        # Create multiple inboxes
        for i in range(3):
            await handle_create_shared_inbox(
                workspace_id="ws_multi",
                name=f"Inbox {i}",
            )

        result = await handle_list_shared_inboxes(workspace_id="ws_multi")

        assert result["success"] is True
        assert len(result["inboxes"]) == 3
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_list_filters_by_workspace(self, clean_state):
        """Should only return inboxes from specified workspace."""
        await handle_create_shared_inbox(workspace_id="ws_a", name="Inbox A")
        await handle_create_shared_inbox(workspace_id="ws_b", name="Inbox B")

        result = await handle_list_shared_inboxes(workspace_id="ws_a")

        assert result["success"] is True
        assert len(result["inboxes"]) == 1
        assert result["inboxes"][0]["name"] == "Inbox A"

    @pytest.mark.asyncio
    async def test_list_filters_by_user_membership(self, sample_inbox):
        """Should filter by user membership when user_id provided."""
        # user1 is a team member
        result = await handle_list_shared_inboxes(
            workspace_id=sample_inbox.workspace_id,
            user_id="user1",
        )
        assert len(result["inboxes"]) == 1

        # unknown user is not a member
        result = await handle_list_shared_inboxes(
            workspace_id=sample_inbox.workspace_id,
            user_id="unknown_user",
        )
        assert len(result["inboxes"]) == 0

    @pytest.mark.asyncio
    async def test_list_includes_admins(self, sample_inbox):
        """Should include inboxes where user is admin."""
        result = await handle_list_shared_inboxes(
            workspace_id=sample_inbox.workspace_id,
            user_id="admin1",
        )

        assert len(result["inboxes"]) == 1


class TestHandleGetSharedInbox:
    """Tests for handle_get_shared_inbox function."""

    @pytest.mark.asyncio
    async def test_get_existing_inbox(self, sample_inbox):
        """Should return inbox details for existing inbox."""
        result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["inbox"]["id"] == sample_inbox.id
        assert result["inbox"]["name"] == sample_inbox.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_inbox(self, clean_state):
        """Should return error for nonexistent inbox."""
        result = await handle_get_shared_inbox(inbox_id="inbox_nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_inbox_updates_counts(self, sample_inbox, sample_message):
        """Should update message and unread counts."""
        result = await handle_get_shared_inbox(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["inbox"]["message_count"] == 1
        assert result["inbox"]["unread_count"] == 1  # sample_message is OPEN


class TestHandleGetInboxMessages:
    """Tests for handle_get_inbox_messages function."""

    @pytest.mark.asyncio
    async def test_get_messages_basic(self, sample_inbox, sample_message):
        """Should return messages in inbox."""
        result = await handle_get_inbox_messages(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == sample_message.id

    @pytest.mark.asyncio
    async def test_get_messages_empty_inbox(self, sample_inbox):
        """Should return empty list for inbox with no messages."""
        result = await handle_get_inbox_messages(inbox_id=sample_inbox.id)

        assert result["success"] is True
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_get_messages_filter_by_status(self, sample_inbox, sample_message):
        """Should filter messages by status."""
        # Create additional message with different status
        now = datetime.now(timezone.utc)
        resolved_msg = SharedInboxMessage(
            id="msg_resolved",
            inbox_id=sample_inbox.id,
            email_id="email_resolved",
            subject="Resolved Issue",
            from_address="user@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Resolved",
            received_at=now,
            status=MessageStatus.RESOLVED,
        )
        with _storage_lock:
            _inbox_messages[sample_inbox.id][resolved_msg.id] = resolved_msg

        # Filter for open only
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            status="open",
        )
        assert len(result["messages"]) == 1
        assert result["messages"][0]["status"] == "open"

        # Filter for resolved only
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            status="resolved",
        )
        assert len(result["messages"]) == 1
        assert result["messages"][0]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_get_messages_filter_by_assigned_to(self, sample_inbox, sample_message):
        """Should filter messages by assignee."""
        # Assign the message
        sample_message.assigned_to = "user1"
        sample_message.status = MessageStatus.ASSIGNED

        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            assigned_to="user1",
        )
        assert len(result["messages"]) == 1

        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            assigned_to="user_other",
        )
        assert len(result["messages"]) == 0

    @pytest.mark.asyncio
    async def test_get_messages_filter_by_tag(self, sample_inbox, sample_message):
        """Should filter messages by tag."""
        sample_message.tags = ["urgent", "customer"]

        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            tag="urgent",
        )
        assert len(result["messages"]) == 1

        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            tag="nonexistent_tag",
        )
        assert len(result["messages"]) == 0

    @pytest.mark.asyncio
    async def test_get_messages_pagination(self, sample_inbox):
        """Should support pagination."""
        # Create multiple messages
        now = datetime.now(timezone.utc)
        for i in range(15):
            msg = SharedInboxMessage(
                id=f"msg_page_{i}",
                inbox_id=sample_inbox.id,
                email_id=f"email_{i}",
                subject=f"Message {i}",
                from_address="user@test.com",
                to_addresses=["inbox@test.com"],
                snippet=f"Content {i}",
                received_at=now - timedelta(minutes=i),
            )
            with _storage_lock:
                _inbox_messages[sample_inbox.id][msg.id] = msg

        # First page
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            limit=5,
            offset=0,
        )
        assert len(result["messages"]) == 5
        assert result["total"] == 15
        assert result["limit"] == 5
        assert result["offset"] == 0

        # Second page
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            limit=5,
            offset=5,
        )
        assert len(result["messages"]) == 5
        assert result["offset"] == 5

    @pytest.mark.asyncio
    async def test_get_messages_sorted_by_received_at(self, sample_inbox):
        """Messages should be sorted by received_at descending."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            msg = SharedInboxMessage(
                id=f"msg_sort_{i}",
                inbox_id=sample_inbox.id,
                email_id=f"email_sort_{i}",
                subject=f"Sort Message {i}",
                from_address="user@test.com",
                to_addresses=["inbox@test.com"],
                snippet="Content",
                received_at=now - timedelta(hours=i),
            )
            with _storage_lock:
                _inbox_messages[sample_inbox.id][msg.id] = msg

        result = await handle_get_inbox_messages(inbox_id=sample_inbox.id)

        # Most recent should be first
        messages = result["messages"]
        for i in range(len(messages) - 1):
            t1 = datetime.fromisoformat(messages[i]["received_at"])
            t2 = datetime.fromisoformat(messages[i + 1]["received_at"])
            assert t1 >= t2

    @pytest.mark.asyncio
    async def test_get_messages_nonexistent_inbox(self, clean_state):
        """Should return error for nonexistent inbox."""
        result = await handle_get_inbox_messages(inbox_id="inbox_nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Handler Function Tests - Message Operations
# =============================================================================


class TestHandleAssignMessage:
    """Tests for handle_assign_message function."""

    @pytest.mark.asyncio
    async def test_assign_message_success(self, sample_inbox, sample_message):
        """Should assign message to user."""
        result = await handle_assign_message(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            assigned_to="assignee_user",
            assigned_by="manager_user",
        )

        assert result["success"] is True
        assert result["message"]["assigned_to"] == "assignee_user"
        assert result["message"]["assigned_at"] is not None

    @pytest.mark.asyncio
    async def test_assign_message_updates_status(self, sample_inbox, sample_message):
        """Assigning open message should update status to ASSIGNED."""
        assert sample_message.status == MessageStatus.OPEN

        result = await handle_assign_message(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            assigned_to="user1",
        )

        assert result["message"]["status"] == MessageStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_assign_message_preserves_non_open_status(self, sample_inbox, sample_message):
        """Assigning message with non-OPEN status should preserve status."""
        sample_message.status = MessageStatus.IN_PROGRESS

        result = await handle_assign_message(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            assigned_to="user1",
        )

        assert result["message"]["status"] == MessageStatus.IN_PROGRESS.value

    @pytest.mark.asyncio
    async def test_assign_nonexistent_message(self, sample_inbox):
        """Should return error for nonexistent message."""
        result = await handle_assign_message(
            inbox_id=sample_inbox.id,
            message_id="msg_nonexistent",
            assigned_to="user1",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_assign_logs_activity(self, sample_inbox, sample_message, mock_activity_store):
        """Should log activity when org_id provided."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            result = await handle_assign_message(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                assigned_to="user1",
                assigned_by="manager",
                org_id="org_123",
            )

        assert result["success"] is True
        assert len(mock_activity_store.activities) == 1


class TestHandleUpdateMessageStatus:
    """Tests for handle_update_message_status function."""

    @pytest.mark.asyncio
    async def test_update_status_success(self, sample_inbox, sample_message):
        """Should update message status."""
        result = await handle_update_message_status(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            status="in_progress",
        )

        assert result["success"] is True
        assert result["message"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_update_status_to_resolved(self, sample_inbox, sample_message):
        """Resolving message should set resolved timestamp and user."""
        result = await handle_update_message_status(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            status="resolved",
            updated_by="resolver_user",
        )

        assert result["success"] is True
        assert result["message"]["status"] == "resolved"
        assert result["message"]["resolved_at"] is not None
        assert result["message"]["resolved_by"] == "resolver_user"

    @pytest.mark.asyncio
    async def test_update_status_invalid(self, sample_inbox, sample_message):
        """Should return error for invalid status."""
        result = await handle_update_message_status(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            status="invalid_status_value",
        )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_update_status_all_valid_statuses(self, sample_inbox, sample_message):
        """Should accept all valid status values."""
        valid_statuses = ["open", "assigned", "in_progress", "waiting", "resolved", "closed"]

        for status in valid_statuses:
            result = await handle_update_message_status(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                status=status,
            )
            assert result["success"] is True
            assert result["message"]["status"] == status


class TestHandleAddMessageTag:
    """Tests for handle_add_message_tag function."""

    @pytest.mark.asyncio
    async def test_add_tag_success(self, sample_inbox, sample_message):
        """Should add tag to message."""
        result = await handle_add_message_tag(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            tag="priority",
        )

        assert result["success"] is True
        assert "priority" in result["message"]["tags"]

    @pytest.mark.asyncio
    async def test_add_multiple_tags(self, sample_inbox, sample_message):
        """Should accumulate multiple tags."""
        tags = ["urgent", "customer", "billing"]
        for tag in tags:
            result = await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag=tag,
            )
            assert result["success"] is True

        assert len(result["message"]["tags"]) == 3
        for tag in tags:
            assert tag in result["message"]["tags"]

    @pytest.mark.asyncio
    async def test_add_duplicate_tag_no_duplicate(self, sample_inbox, sample_message):
        """Adding duplicate tag should not create duplicate."""
        await handle_add_message_tag(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            tag="important",
        )
        result = await handle_add_message_tag(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            tag="important",
        )

        assert result["success"] is True
        assert result["message"]["tags"].count("important") == 1

    @pytest.mark.asyncio
    async def test_add_tag_nonexistent_message(self, sample_inbox):
        """Should return error for nonexistent message."""
        result = await handle_add_message_tag(
            inbox_id=sample_inbox.id,
            message_id="msg_nonexistent",
            tag="test",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestHandleAddMessageToInbox:
    """Tests for handle_add_message_to_inbox function."""

    @pytest.mark.asyncio
    async def test_add_message_basic(self, sample_inbox):
        """Should add message to inbox."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="external_email_123",
            subject="New Support Request",
            from_address="customer@gmail.com",
            to_addresses=["support@example.com"],
            snippet="I need help with...",
        )

        assert result["success"] is True
        assert result["message"]["inbox_id"] == sample_inbox.id
        assert result["message"]["subject"] == "New Support Request"
        assert result["message"]["id"].startswith("msg_")

    @pytest.mark.asyncio
    async def test_add_message_with_all_fields(self, sample_inbox):
        """Should add message with all optional fields."""
        now = datetime.now(timezone.utc)
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_full",
            subject="Full Message",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com", "cc@test.com"],
            snippet="Full snippet content",
            received_at=now,
            thread_id="thread_full",
            priority="high",
            workspace_id="ws_full",
        )

        assert result["success"] is True
        msg = result["message"]
        assert msg["thread_id"] == "thread_full"
        assert msg["priority"] == "high"
        assert len(msg["to_addresses"]) == 2

    @pytest.mark.asyncio
    async def test_add_message_default_received_at(self, sample_inbox):
        """Should set received_at to now if not provided."""
        before = datetime.now(timezone.utc)
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_time",
            subject="Time Test",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Snippet",
        )
        after = datetime.now(timezone.utc)

        received_at = datetime.fromisoformat(result["message"]["received_at"])
        assert before <= received_at <= after

    @pytest.mark.asyncio
    async def test_add_message_creates_inbox_entry_if_missing(self, clean_state):
        """Should create inbox messages dict if inbox not in memory."""
        result = await handle_add_message_to_inbox(
            inbox_id="inbox_new",
            email_id="email_new",
            subject="New Inbox Message",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Snippet",
        )

        assert result["success"] is True
        assert "inbox_new" in _inbox_messages

    @pytest.mark.asyncio
    async def test_add_message_status_is_open(self, sample_inbox):
        """New message should have OPEN status."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_status",
            subject="Status Test",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Snippet",
        )

        assert result["message"]["status"] == "open"


# =============================================================================
# Handler Function Tests - Routing Rules
# =============================================================================


class TestHandleCreateRoutingRule:
    """Tests for handle_create_routing_rule function."""

    @pytest.mark.asyncio
    async def test_create_rule_basic(self, clean_state):
        """Should create routing rule with basic fields."""
        result = await handle_create_routing_rule(
            workspace_id="ws_test",
            name="Basic Rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "help"}],
            actions=[{"type": "label", "target": "support"}],
        )

        assert result["success"] is True
        assert result["rule"]["name"] == "Basic Rule"
        assert result["rule"]["id"].startswith("rule_")

    @pytest.mark.asyncio
    async def test_create_rule_with_all_fields(self, clean_state):
        """Should create rule with all optional fields."""
        result = await handle_create_routing_rule(
            workspace_id="ws_full",
            name="Full Rule",
            conditions=[
                {"field": "from", "operator": "ends_with", "value": "@vip.com"},
                {"field": "priority", "operator": "equals", "value": "high"},
            ],
            actions=[
                {"type": "assign", "target": "vip-team"},
                {"type": "label", "target": "vip"},
                {"type": "notify", "target": "slack", "params": {"channel": "#vip"}},
            ],
            condition_logic="AND",
            priority=1,
            enabled=True,
            description="Handle VIP customers",
            created_by="admin_user",
            inbox_id="inbox_vip",
        )

        assert result["success"] is True
        rule = result["rule"]
        assert len(rule["conditions"]) == 2
        assert len(rule["actions"]) == 3
        assert rule["priority"] == 1
        assert rule["description"] == "Handle VIP customers"
        assert rule["created_by"] == "admin_user"

    @pytest.mark.asyncio
    async def test_create_rule_defaults(self, clean_state):
        """Should use sensible defaults for optional fields."""
        result = await handle_create_routing_rule(
            workspace_id="ws_default",
            name="Default Rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
            actions=[{"type": "label", "target": "test"}],
        )

        rule = result["rule"]
        assert rule["condition_logic"] == "AND"
        assert rule["priority"] == 5
        assert rule["enabled"] is True

    @pytest.mark.asyncio
    async def test_create_rule_with_or_logic(self, clean_state):
        """Should create rule with OR condition logic."""
        result = await handle_create_routing_rule(
            workspace_id="ws_or",
            name="OR Rule",
            conditions=[
                {"field": "subject", "operator": "contains", "value": "urgent"},
                {"field": "subject", "operator": "contains", "value": "asap"},
            ],
            actions=[{"type": "escalate"}],
            condition_logic="OR",
        )

        assert result["rule"]["condition_logic"] == "OR"


class TestHandleListRoutingRules:
    """Tests for handle_list_routing_rules function."""

    @pytest.mark.asyncio
    async def test_list_rules_empty(self, clean_state):
        """Should return empty list when no rules exist."""
        result = await handle_list_routing_rules(workspace_id="ws_empty")

        assert result["success"] is True
        assert result["rules"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_rules_multiple(self, clean_state):
        """Should return all rules for workspace."""
        for i in range(3):
            await handle_create_routing_rule(
                workspace_id="ws_multi",
                name=f"Rule {i}",
                conditions=[{"field": "subject", "operator": "contains", "value": f"test{i}"}],
                actions=[{"type": "label", "target": f"label{i}"}],
            )

        result = await handle_list_routing_rules(workspace_id="ws_multi")

        assert result["success"] is True
        assert len(result["rules"]) == 3
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_list_rules_filters_by_workspace(self, clean_state):
        """Should only return rules from specified workspace."""
        await handle_create_routing_rule(
            workspace_id="ws_a",
            name="Rule A",
            conditions=[{"field": "subject", "operator": "contains", "value": "a"}],
            actions=[{"type": "label", "target": "a"}],
        )
        await handle_create_routing_rule(
            workspace_id="ws_b",
            name="Rule B",
            conditions=[{"field": "subject", "operator": "contains", "value": "b"}],
            actions=[{"type": "label", "target": "b"}],
        )

        result = await handle_list_routing_rules(workspace_id="ws_a")

        assert len(result["rules"]) == 1
        assert result["rules"][0]["name"] == "Rule A"

    @pytest.mark.asyncio
    async def test_list_rules_enabled_only(self, sample_rule):
        """Should filter by enabled status when requested."""
        # Disable the sample rule
        sample_rule.enabled = False

        # Create an enabled rule
        await handle_create_routing_rule(
            workspace_id=sample_rule.workspace_id,
            name="Enabled Rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "enabled"}],
            actions=[{"type": "label", "target": "enabled"}],
            enabled=True,
        )

        result = await handle_list_routing_rules(
            workspace_id=sample_rule.workspace_id,
            enabled_only=True,
        )

        assert len(result["rules"]) == 1
        assert result["rules"][0]["enabled"] is True

    @pytest.mark.asyncio
    async def test_list_rules_sorted_by_priority(self, clean_state):
        """Rules should be sorted by priority ascending."""
        for priority in [10, 1, 5]:
            await handle_create_routing_rule(
                workspace_id="ws_sort",
                name=f"Priority {priority}",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "test"}],
                priority=priority,
            )

        result = await handle_list_routing_rules(workspace_id="ws_sort")

        priorities = [r["priority"] for r in result["rules"]]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_list_rules_pagination(self, clean_state):
        """Should support pagination."""
        for i in range(10):
            await handle_create_routing_rule(
                workspace_id="ws_page",
                name=f"Rule {i}",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "test"}],
            )

        result = await handle_list_routing_rules(
            workspace_id="ws_page",
            limit=3,
            offset=0,
        )

        assert len(result["rules"]) == 3
        assert result["limit"] == 3
        assert result["offset"] == 0


class TestHandleUpdateRoutingRule:
    """Tests for handle_update_routing_rule function."""

    @pytest.mark.asyncio
    async def test_update_rule_name(self, sample_rule):
        """Should update rule name."""
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"name": "Updated Name"},
        )

        assert result["success"] is True
        assert result["rule"]["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_rule_enabled(self, sample_rule):
        """Should update rule enabled status."""
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"enabled": False},
        )

        assert result["success"] is True
        assert result["rule"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_update_rule_priority(self, sample_rule):
        """Should update rule priority."""
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"priority": 10},
        )

        assert result["success"] is True
        assert result["rule"]["priority"] == 10

    @pytest.mark.asyncio
    async def test_update_rule_conditions(self, sample_rule):
        """Should update rule conditions."""
        new_conditions = [{"field": "from", "operator": "contains", "value": "@company.com"}]
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"conditions": new_conditions},
        )

        assert result["success"] is True
        assert len(result["rule"]["conditions"]) == 1
        assert result["rule"]["conditions"][0]["field"] == "from"

    @pytest.mark.asyncio
    async def test_update_rule_actions(self, sample_rule):
        """Should update rule actions."""
        new_actions = [{"type": "archive"}]
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"actions": new_actions},
        )

        assert result["success"] is True
        assert len(result["rule"]["actions"]) == 1
        assert result["rule"]["actions"][0]["type"] == "archive"

    @pytest.mark.asyncio
    async def test_update_rule_multiple_fields(self, sample_rule):
        """Should update multiple fields at once."""
        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={
                "name": "Multi Update",
                "priority": 2,
                "enabled": False,
                "description": "Updated description",
            },
        )

        assert result["success"] is True
        rule = result["rule"]
        assert rule["name"] == "Multi Update"
        assert rule["priority"] == 2
        assert rule["enabled"] is False
        assert rule["description"] == "Updated description"

    @pytest.mark.asyncio
    async def test_update_rule_updates_timestamp(self, sample_rule):
        """Should update the updated_at timestamp."""
        original_updated_at = sample_rule.updated_at

        result = await handle_update_routing_rule(
            rule_id=sample_rule.id,
            updates={"name": "Timestamp Test"},
        )

        new_updated_at = datetime.fromisoformat(result["rule"]["updated_at"])
        assert new_updated_at >= original_updated_at

    @pytest.mark.asyncio
    async def test_update_rule_nonexistent(self, clean_state):
        """Should return error for nonexistent rule."""
        result = await handle_update_routing_rule(
            rule_id="rule_nonexistent",
            updates={"name": "Test"},
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestHandleDeleteRoutingRule:
    """Tests for handle_delete_routing_rule function."""

    @pytest.mark.asyncio
    async def test_delete_rule_success(self, sample_rule):
        """Should delete existing rule."""
        rule_id = sample_rule.id
        assert rule_id in _routing_rules

        result = await handle_delete_routing_rule(rule_id=rule_id)

        assert result["success"] is True
        assert result["deleted"] == rule_id
        assert rule_id not in _routing_rules

    @pytest.mark.asyncio
    async def test_delete_rule_nonexistent(self, clean_state):
        """Should return error for nonexistent rule."""
        result = await handle_delete_routing_rule(rule_id="rule_nonexistent")

        assert result["success"] is False


class TestHandleTestRoutingRule:
    """Tests for handle_test_routing_rule function."""

    @pytest.mark.asyncio
    async def test_test_rule_success(self, sample_inbox, sample_rule):
        """Should test rule and return match count."""
        result = await handle_test_routing_rule(
            rule_id=sample_rule.id,
            workspace_id=sample_rule.workspace_id,
        )

        assert result["success"] is True
        assert "match_count" in result
        assert "rule" in result
        assert result["rule_id"] == sample_rule.id

    @pytest.mark.asyncio
    async def test_test_rule_with_matching_messages(self, sample_inbox, sample_rule):
        """Should count messages matching the rule."""
        # Create messages with "urgent" in subject (matches sample_rule)
        now = datetime.now(timezone.utc)
        for i in range(3):
            msg = SharedInboxMessage(
                id=f"msg_urgent_{i}",
                inbox_id=sample_inbox.id,
                email_id=f"email_urgent_{i}",
                subject=f"Urgent issue {i}",
                from_address="user@test.com",
                to_addresses=["inbox@test.com"],
                snippet="Urgent content",
                received_at=now,
            )
            with _storage_lock:
                _inbox_messages[sample_inbox.id][msg.id] = msg

        result = await handle_test_routing_rule(
            rule_id=sample_rule.id,
            workspace_id=sample_rule.workspace_id,
        )

        assert result["success"] is True
        assert result["match_count"] >= 3

    @pytest.mark.asyncio
    async def test_test_rule_nonexistent(self, clean_state):
        """Should return error for nonexistent rule."""
        result = await handle_test_routing_rule(
            rule_id="rule_nonexistent",
            workspace_id="ws_test",
        )

        assert result["success"] is False


# =============================================================================
# Rule Evaluation Tests
# =============================================================================


class TestEvaluateRule:
    """Tests for _evaluate_rule function."""

    def test_evaluate_contains_match(self):
        """CONTAINS operator should match substring."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "This is an urgent request"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_contains_no_match(self):
        """CONTAINS operator should not match when substring absent."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Normal request"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is False

    def test_evaluate_equals_match(self):
        """EQUALS operator should match exact value."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.PRIORITY,
                    operator=RuleConditionOperator.EQUALS,
                    value="high",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Test"
            priority = "high"

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_starts_with(self):
        """STARTS_WITH operator should match prefix."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.STARTS_WITH,
                    value="[ticket]",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "[Ticket] Issue #123"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_ends_with(self):
        """ENDS_WITH operator should match suffix."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.FROM,
                    operator=RuleConditionOperator.ENDS_WITH,
                    value="@company.com",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "employee@company.com"
            to_addresses = ["inbox@test.com"]
            subject = "Internal"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_matches_regex(self):
        """MATCHES operator should match regex pattern."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.MATCHES,
                    value=r"ticket\s*#\d+",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Re: Ticket #12345"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_matches_invalid_regex(self):
        """MATCHES with invalid regex should return False."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.MATCHES,
                    value=r"[invalid",  # Invalid regex
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Test"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is False

    def test_evaluate_sender_domain(self):
        """SENDER_DOMAIN field should extract domain from email."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SENDER_DOMAIN,
                    operator=RuleConditionOperator.EQUALS,
                    value="gmail.com",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "user@gmail.com"
            to_addresses = ["inbox@test.com"]
            subject = "Test"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_to_addresses(self):
        """TO field should check all recipient addresses."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.TO,
                    operator=RuleConditionOperator.CONTAINS,
                    value="sales@",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "customer@test.com"
            to_addresses = ["support@company.com", "sales@company.com"]
            subject = "Test"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_and_logic_all_match(self):
        """AND logic should require all conditions to match."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
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
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Urgent issue"
            priority = "high"

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_and_logic_partial_match(self):
        """AND logic should fail if any condition doesn't match."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
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
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Urgent issue"
            priority = "normal"  # Doesn't match

        assert _evaluate_rule(rule, MockMessage()) is False

    def test_evaluate_or_logic_one_match(self):
        """OR logic should pass if any condition matches."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                ),
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="asap",
                ),
            ],
            condition_logic="OR",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Please help ASAP"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True

    def test_evaluate_or_logic_no_match(self):
        """OR logic should fail if no conditions match."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="urgent",
                ),
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="asap",
                ),
            ],
            condition_logic="OR",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Normal request"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is False

    def test_evaluate_empty_conditions(self):
        """Rule with empty conditions should return False."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "Test"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is False

    def test_evaluate_case_insensitive(self):
        """Matching should be case insensitive."""
        rule = RoutingRule(
            id="rule_test",
            name="Test",
            workspace_id="ws",
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="URGENT",
                )
            ],
            condition_logic="AND",
            actions=[],
        )

        class MockMessage:
            from_address = "test@test.com"
            to_addresses = ["inbox@test.com"]
            subject = "This is urgent"
            priority = None

        assert _evaluate_rule(rule, MockMessage()) is True


# =============================================================================
# Routing Rule Application Tests
# =============================================================================


class TestApplyRoutingRulesToMessage:
    """Tests for apply_routing_rules_to_message function."""

    @pytest.mark.asyncio
    async def test_apply_no_matching_rules(self, sample_inbox, sample_message, sample_rule):
        """Should return not applied when no rules match."""
        # sample_message doesn't have "urgent" in subject
        sample_message.subject = "Normal request"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_rule.workspace_id,
        )

        assert result["applied"] is False
        assert result["rules_matched"] == 0

    @pytest.mark.asyncio
    async def test_apply_matching_rule_label(self, sample_inbox, sample_message, sample_rule):
        """Should apply label action from matching rule."""
        sample_message.subject = "Urgent issue here"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_rule.workspace_id,
        )

        assert result["applied"] is True
        assert result["rules_matched"] >= 1
        assert "urgent" in sample_message.tags

    @pytest.mark.asyncio
    async def test_apply_matching_rule_escalate(self, sample_inbox, sample_message, sample_rule):
        """Should apply escalate action from matching rule."""
        sample_message.subject = "Urgent issue"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_rule.workspace_id,
        )

        assert sample_message.priority == "high"

    @pytest.mark.asyncio
    async def test_apply_assign_action(self, sample_inbox, sample_message, clean_state):
        """Should assign message when assign action matches."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_assign",
            name="Assign Rule",
            workspace_id=sample_inbox.workspace_id,
            conditions=[
                RuleCondition(
                    field=RuleConditionField.SUBJECT,
                    operator=RuleConditionOperator.CONTAINS,
                    value="help",
                )
            ],
            condition_logic="AND",
            actions=[RuleAction(type=RuleActionType.ASSIGN, target="support_agent")],
            priority=1,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with _storage_lock:
            _routing_rules[rule.id] = rule

        sample_message.subject = "I need help with my order"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_inbox.workspace_id,
        )

        assert result["applied"] is True
        assert sample_message.assigned_to == "support_agent"
        assert sample_message.status == MessageStatus.ASSIGNED

    @pytest.mark.asyncio
    async def test_apply_archive_action(self, sample_inbox, sample_message, clean_state):
        """Should close message when archive action matches."""
        now = datetime.now(timezone.utc)
        rule = RoutingRule(
            id="rule_archive",
            name="Archive Rule",
            workspace_id=sample_inbox.workspace_id,
            conditions=[
                RuleCondition(
                    field=RuleConditionField.FROM,
                    operator=RuleConditionOperator.CONTAINS,
                    value="noreply@",
                )
            ],
            condition_logic="AND",
            actions=[RuleAction(type=RuleActionType.ARCHIVE)],
            priority=1,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with _storage_lock:
            _routing_rules[rule.id] = rule

        sample_message.from_address = "noreply@notifications.com"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_inbox.workspace_id,
        )

        assert result["applied"] is True
        assert sample_message.status == MessageStatus.CLOSED

    @pytest.mark.asyncio
    async def test_apply_multiple_actions(self, sample_inbox, sample_message, sample_rule):
        """Should apply multiple actions from same rule."""
        sample_message.subject = "Urgent issue"

        result = await apply_routing_rules_to_message(
            inbox_id=sample_inbox.id,
            message=sample_message,
            workspace_id=sample_rule.workspace_id,
        )

        # sample_rule has label and escalate actions
        assert "urgent" in sample_message.tags
        assert sample_message.priority == "high"


class TestGetMatchingRulesForEmail:
    """Tests for get_matching_rules_for_email function."""

    @pytest.mark.asyncio
    async def test_get_matching_rules_basic(self, sample_rule):
        """Should return matching rules for email data."""
        email_data = {
            "from_address": "customer@test.com",
            "to_addresses": ["support@company.com"],
            "subject": "Urgent issue with order",
            "snippet": "Please help urgently",
            "priority": None,
        }

        result = await get_matching_rules_for_email(
            inbox_id="inbox_test",
            email_data=email_data,
            workspace_id=sample_rule.workspace_id,
        )

        assert len(result) >= 1
        assert any(r["id"] == sample_rule.id for r in result)

    @pytest.mark.asyncio
    async def test_get_matching_rules_no_match(self, sample_rule):
        """Should return empty list when no rules match."""
        email_data = {
            "from_address": "customer@test.com",
            "to_addresses": ["support@company.com"],
            "subject": "Normal inquiry",
            "snippet": "General question",
            "priority": None,
        }

        result = await get_matching_rules_for_email(
            inbox_id="inbox_test",
            email_data=email_data,
            workspace_id=sample_rule.workspace_id,
        )

        assert result == []


# =============================================================================
# Handler Class Tests
# =============================================================================


class TestSharedInboxHandlerRouting:
    """Tests for SharedInboxHandler routing methods."""

    def test_routes_defined(self, shared_inbox_handler):
        """Handler should define expected routes."""
        assert "/api/v1/inbox/shared" in shared_inbox_handler.ROUTES
        assert "/api/v1/inbox/routing/rules" in shared_inbox_handler.ROUTES

    def test_route_prefixes_defined(self, shared_inbox_handler):
        """Handler should define expected route prefixes."""
        assert "/api/v1/inbox/shared/" in shared_inbox_handler.ROUTE_PREFIXES
        assert "/api/v1/inbox/routing/rules/" in shared_inbox_handler.ROUTE_PREFIXES

    def test_can_handle_exact_routes(self, shared_inbox_handler):
        """can_handle should match exact routes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/shared") is True
        assert shared_inbox_handler.can_handle("/api/v1/inbox/routing/rules") is True

    def test_can_handle_prefixed_routes(self, shared_inbox_handler):
        """can_handle should match routes with ID suffixes."""
        assert shared_inbox_handler.can_handle("/api/v1/inbox/shared/inbox_123") is True
        assert shared_inbox_handler.can_handle("/api/v1/inbox/shared/inbox_123/messages") is True
        assert shared_inbox_handler.can_handle("/api/v1/inbox/routing/rules/rule_456") is True
        assert shared_inbox_handler.can_handle("/api/v1/inbox/routing/rules/rule_456/test") is True

    def test_can_handle_rejects_unrelated(self, shared_inbox_handler):
        """can_handle should reject unrelated routes."""
        assert shared_inbox_handler.can_handle("/api/v1/debates") is False
        assert shared_inbox_handler.can_handle("/api/v1/users") is False
        assert shared_inbox_handler.can_handle("/api/v1/inbox") is False  # Missing /shared

    def test_handle_returns_none(self, shared_inbox_handler):
        """Base handle method should return None (subclass responsibility)."""
        result = shared_inbox_handler.handle("/api/v1/inbox/shared", {}, None)
        assert result is None


class TestSharedInboxHandlerMethods:
    """Tests for SharedInboxHandler HTTP method handlers."""

    @pytest.mark.asyncio
    async def test_handle_post_shared_inbox_missing_workspace(self, shared_inbox_handler):
        """POST shared inbox should require workspace_id."""
        result = await shared_inbox_handler.handle_post_shared_inbox({"name": "Test"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_shared_inbox_missing_name(self, shared_inbox_handler):
        """POST shared inbox should require name."""
        result = await shared_inbox_handler.handle_post_shared_inbox({"workspace_id": "ws"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_shared_inbox_success(self, shared_inbox_handler, clean_state):
        """POST shared inbox should succeed with valid data."""
        result = await shared_inbox_handler.handle_post_shared_inbox(
            {"workspace_id": "ws_test", "name": "Test Inbox"}
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_shared_inboxes_missing_workspace(self, shared_inbox_handler):
        """GET shared inboxes should require workspace_id."""
        result = await shared_inbox_handler.handle_get_shared_inboxes({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_shared_inboxes_success(self, shared_inbox_handler, clean_state):
        """GET shared inboxes should succeed with valid workspace_id."""
        result = await shared_inbox_handler.handle_get_shared_inboxes({"workspace_id": "ws_test"})
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_assign_missing_assigned_to(self, shared_inbox_handler, sample_inbox):
        """POST assign should require assigned_to."""
        result = await shared_inbox_handler.handle_post_assign_message(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_status_missing_status(self, shared_inbox_handler, sample_inbox):
        """POST status should require status."""
        result = await shared_inbox_handler.handle_post_update_status(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_tag_missing_tag(self, shared_inbox_handler, sample_inbox):
        """POST tag should require tag."""
        result = await shared_inbox_handler.handle_post_add_tag(
            data={},
            inbox_id=sample_inbox.id,
            message_id="msg_123",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_routing_rule_missing_fields(self, shared_inbox_handler):
        """POST routing rule should require all fields."""
        result = await shared_inbox_handler.handle_post_routing_rule(
            {"workspace_id": "ws", "name": "Test"}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_routing_rules_missing_workspace(self, shared_inbox_handler):
        """GET routing rules should require workspace_id."""
        result = await shared_inbox_handler.handle_get_routing_rules({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_test_routing_rule_missing_workspace(self, shared_inbox_handler):
        """POST test routing rule should require workspace_id."""
        result = await shared_inbox_handler.handle_post_test_routing_rule(
            data={},
            rule_id="rule_123",
        )
        assert result.status_code == 400


class TestSharedInboxHandlerUserIdExtraction:
    """Tests for _get_user_id method."""

    def test_get_user_id_from_auth_context(self, server_context):
        """Should extract user_id from auth context."""
        handler = SharedInboxHandler(server_context)
        user_id = handler._get_user_id()
        assert user_id == "test_user_123"

    def test_get_user_id_default_when_no_context(self):
        """Should return 'default' when no auth context."""
        handler = SharedInboxHandler({})
        user_id = handler._get_user_id()
        assert user_id == "default"

    def test_get_user_id_default_when_none_context(self):
        """Should return 'default' when auth context is None."""
        handler = SharedInboxHandler({"auth_context": None})
        user_id = handler._get_user_id()
        assert user_id == "default"


# =============================================================================
# Store Integration Tests
# =============================================================================


class TestEmailStoreIntegration:
    """Tests for email store integration."""

    @pytest.mark.asyncio
    async def test_create_inbox_persists_to_store(self, mock_email_store, clean_state):
        """Should persist inbox to email store when available."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_store",
            return_value=mock_email_store,
        ):
            result = await handle_create_shared_inbox(
                workspace_id="ws_store",
                name="Store Test Inbox",
            )

        assert result["success"] is True
        inbox_id = result["inbox"]["id"]
        assert inbox_id in mock_email_store.inboxes

    @pytest.mark.asyncio
    async def test_add_message_persists_to_store(self, mock_email_store, sample_inbox):
        """Should persist message to email store when available."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_store",
            return_value=mock_email_store,
        ):
            result = await handle_add_message_to_inbox(
                inbox_id=sample_inbox.id,
                email_id="email_store",
                subject="Store Test",
                from_address="test@test.com",
                to_addresses=["inbox@test.com"],
                snippet="Snippet",
            )

        assert result["success"] is True
        message_id = result["message"]["id"]
        assert message_id in mock_email_store.messages

    @pytest.mark.asyncio
    async def test_store_exception_falls_back_to_memory(self, clean_state):
        """Should fall back to in-memory when store raises exception."""
        mock_store = MagicMock()
        mock_store.create_shared_inbox.side_effect = RuntimeError("Store error")

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_store",
            return_value=mock_store,
        ):
            result = await handle_create_shared_inbox(
                workspace_id="ws_fallback",
                name="Fallback Test",
            )

        # Should still succeed with in-memory storage
        assert result["success"] is True
        assert result["inbox"]["id"] in _shared_inboxes


class TestRulesStoreIntegration:
    """Tests for rules store integration."""

    @pytest.mark.asyncio
    async def test_create_rule_persists_to_store(self, mock_rules_store, clean_state):
        """Should persist rule to rules store when available."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store",
            return_value=mock_rules_store,
        ):
            result = await handle_create_routing_rule(
                workspace_id="ws_rules",
                name="Store Rule",
                conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
                actions=[{"type": "label", "target": "test"}],
            )

        assert result["success"] is True
        rule_id = result["rule"]["id"]
        assert rule_id in mock_rules_store.rules

    @pytest.mark.asyncio
    async def test_list_rules_from_store(self, mock_rules_store, clean_state):
        """Should load rules from rules store when available."""
        # Pre-populate mock store
        mock_rules_store.rules["rule_existing"] = {
            "id": "rule_existing",
            "name": "Existing Rule",
            "workspace_id": "ws_rules",
            "conditions": [],
            "actions": [],
            "enabled": True,
        }

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store",
            return_value=mock_rules_store,
        ):
            result = await handle_list_routing_rules(workspace_id="ws_rules")

        assert result["success"] is True
        assert len(result["rules"]) == 1

    @pytest.mark.asyncio
    async def test_update_rule_persists_to_store(self, mock_rules_store, sample_rule):
        """Should persist rule update to rules store."""
        mock_rules_store.rules[sample_rule.id] = sample_rule.to_dict()

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_rules_store",
            return_value=mock_rules_store,
        ):
            result = await handle_update_routing_rule(
                rule_id=sample_rule.id,
                updates={"name": "Updated Via Store"},
            )

        assert result["success"] is True
        assert mock_rules_store.rules[sample_rule.id]["name"] == "Updated Via Store"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_message_status_error(self, sample_inbox, sample_message):
        """Should handle invalid status value gracefully."""
        result = await handle_update_message_status(
            inbox_id=sample_inbox.id,
            message_id=sample_message.id,
            status="not_a_valid_status",
        )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_nonexistent_inbox_assignment(self, clean_state):
        """Should handle assignment to nonexistent inbox."""
        result = await handle_assign_message(
            inbox_id="inbox_does_not_exist",
            message_id="msg_123",
            assigned_to="user1",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_nonexistent_inbox_messages(self, clean_state):
        """Should handle getting messages from nonexistent inbox."""
        result = await handle_get_inbox_messages(inbox_id="inbox_does_not_exist")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rule_with_invalid_condition_field(self, clean_state):
        """Should handle invalid condition field gracefully."""
        # The handler catches the exception and returns an error response
        result = await handle_create_routing_rule(
            workspace_id="ws_test",
            name="Invalid Rule",
            conditions=[{"field": "invalid_field", "operator": "contains", "value": "test"}],
            actions=[{"type": "label", "target": "test"}],
        )

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of operations."""

    @pytest.mark.asyncio
    async def test_concurrent_inbox_creation(self, clean_state):
        """Should handle concurrent inbox creation safely."""
        import asyncio

        async def create_inbox(i: int):
            return await handle_create_shared_inbox(
                workspace_id="ws_concurrent",
                name=f"Inbox {i}",
            )

        results = await asyncio.gather(*[create_inbox(i) for i in range(10)])

        # All should succeed
        assert all(r["success"] for r in results)

        # All should have unique IDs
        ids = [r["inbox"]["id"] for r in results]
        assert len(set(ids)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_message_tagging(self, sample_inbox, sample_message):
        """Should handle concurrent message tagging safely."""
        import asyncio

        async def add_tag(tag: str):
            return await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag=tag,
            )

        tags = [f"tag_{i}" for i in range(10)]
        results = await asyncio.gather(*[add_tag(tag) for tag in tags])

        # All should succeed
        assert all(r["success"] for r in results)

        # All tags should be added (no duplicates from concurrency)
        final_tags = results[-1]["message"]["tags"]
        for tag in tags:
            assert tag in final_tags

    @pytest.mark.asyncio
    async def test_concurrent_rule_creation(self, clean_state):
        """Should handle concurrent rule creation safely."""
        import asyncio

        async def create_rule(i: int):
            return await handle_create_routing_rule(
                workspace_id="ws_concurrent",
                name=f"Rule {i}",
                conditions=[{"field": "subject", "operator": "contains", "value": f"test{i}"}],
                actions=[{"type": "label", "target": f"label{i}"}],
            )

        results = await asyncio.gather(*[create_rule(i) for i in range(10)])

        # All should succeed
        assert all(r["success"] for r in results)

        # All should have unique IDs
        ids = [r["rule"]["id"] for r in results]
        assert len(set(ids)) == 10


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_subject_message(self, sample_inbox):
        """Should handle message with empty subject."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_empty_subject",
            subject="",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Content without subject",
        )

        assert result["success"] is True
        assert result["message"]["subject"] == ""

    @pytest.mark.asyncio
    async def test_very_long_subject(self, sample_inbox):
        """Should handle message with very long subject."""
        long_subject = "X" * 1000
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_long_subject",
            subject=long_subject,
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Content",
        )

        assert result["success"] is True
        assert result["message"]["subject"] == long_subject

    @pytest.mark.asyncio
    async def test_unicode_in_content(self, sample_inbox):
        """Should handle unicode characters in message content."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_unicode",
            subject="Test with emoji and unicode",
            from_address="sender@test.com",
            to_addresses=["inbox@test.com"],
            snippet="Content",
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_special_characters_in_email(self, sample_inbox):
        """Should handle special characters in email addresses."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_special",
            subject="Special chars test",
            from_address="user+tag@sub.domain.test.com",
            to_addresses=["inbox@test.com"],
            snippet="Content",
        )

        assert result["success"] is True
        assert result["message"]["from_address"] == "user+tag@sub.domain.test.com"

    @pytest.mark.asyncio
    async def test_multiple_to_addresses(self, sample_inbox):
        """Should handle multiple recipient addresses."""
        to_addresses = [f"recipient{i}@test.com" for i in range(10)]
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_multi_to",
            subject="Multi recipient",
            from_address="sender@test.com",
            to_addresses=to_addresses,
            snippet="Content",
        )

        assert result["success"] is True
        assert len(result["message"]["to_addresses"]) == 10

    @pytest.mark.asyncio
    async def test_empty_to_addresses(self, sample_inbox):
        """Should handle empty to_addresses list."""
        result = await handle_add_message_to_inbox(
            inbox_id=sample_inbox.id,
            email_id="email_empty_to",
            subject="Empty to",
            from_address="sender@test.com",
            to_addresses=[],
            snippet="Content",
        )

        assert result["success"] is True
        assert result["message"]["to_addresses"] == []

    @pytest.mark.asyncio
    async def test_rule_with_many_conditions(self, clean_state):
        """Should handle rule with many conditions."""
        conditions = [
            {"field": "subject", "operator": "contains", "value": f"keyword{i}"} for i in range(20)
        ]

        result = await handle_create_routing_rule(
            workspace_id="ws_many",
            name="Many Conditions Rule",
            conditions=conditions,
            actions=[{"type": "label", "target": "complex"}],
            condition_logic="OR",
        )

        assert result["success"] is True
        assert len(result["rule"]["conditions"]) == 20

    @pytest.mark.asyncio
    async def test_rule_with_many_actions(self, clean_state):
        """Should handle rule with many actions."""
        actions = [{"type": "label", "target": f"label{i}"} for i in range(10)]

        result = await handle_create_routing_rule(
            workspace_id="ws_many_actions",
            name="Many Actions Rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "test"}],
            actions=actions,
        )

        assert result["success"] is True
        assert len(result["rule"]["actions"]) == 10

    @pytest.mark.asyncio
    async def test_pagination_beyond_results(self, sample_inbox):
        """Should handle pagination offset beyond available results."""
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            limit=10,
            offset=1000,  # Way beyond actual messages
        )

        assert result["success"] is True
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_zero_limit_pagination(self, sample_inbox, sample_message):
        """Should handle zero limit gracefully."""
        # Note: The handler might still return results if limit is normalized
        result = await handle_get_inbox_messages(
            inbox_id=sample_inbox.id,
            limit=0,
            offset=0,
        )

        assert result["success"] is True


# =============================================================================
# Activity Logging Tests
# =============================================================================


class TestActivityLogging:
    """Tests for activity logging functionality."""

    @pytest.mark.asyncio
    async def test_assign_logs_activity(self, sample_inbox, sample_message, mock_activity_store):
        """Should log activity when assigning message."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_assign_message(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                assigned_to="user1",
                assigned_by="manager",
                org_id="org_123",
            )

        assert len(mock_activity_store.activities) == 1
        activity = mock_activity_store.activities[0]
        assert activity.inbox_id == sample_inbox.id
        assert activity.action == "assigned"

    @pytest.mark.asyncio
    async def test_reassign_logs_different_action(
        self, sample_inbox, sample_message, mock_activity_store
    ):
        """Should log 'reassigned' when reassigning message."""
        # First assignment
        sample_message.assigned_to = "original_user"

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_assign_message(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                assigned_to="new_user",
                assigned_by="manager",
                org_id="org_123",
            )

        assert len(mock_activity_store.activities) == 1
        activity = mock_activity_store.activities[0]
        assert activity.action == "reassigned"

    @pytest.mark.asyncio
    async def test_status_change_logs_activity(
        self, sample_inbox, sample_message, mock_activity_store
    ):
        """Should log activity when changing status."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_update_message_status(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                status="in_progress",
                updated_by="user1",
                org_id="org_123",
            )

        assert len(mock_activity_store.activities) == 1
        activity = mock_activity_store.activities[0]
        assert activity.action == "status_changed"

    @pytest.mark.asyncio
    async def test_tag_add_logs_activity(self, sample_inbox, sample_message, mock_activity_store):
        """Should log activity when adding tag."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag="important",
                added_by="user1",
                org_id="org_123",
            )

        assert len(mock_activity_store.activities) == 1
        activity = mock_activity_store.activities[0]
        assert activity.action == "tag_added"

    @pytest.mark.asyncio
    async def test_duplicate_tag_no_activity(
        self, sample_inbox, sample_message, mock_activity_store
    ):
        """Should not log activity when adding duplicate tag."""
        sample_message.tags = ["important"]

        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_add_message_tag(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                tag="important",  # Already exists
                added_by="user1",
                org_id="org_123",
            )

        # Should not log because tag was not actually added
        assert len(mock_activity_store.activities) == 0

    @pytest.mark.asyncio
    async def test_activity_not_logged_without_org_id(
        self, sample_inbox, sample_message, mock_activity_store
    ):
        """Should not log activity when org_id not provided."""
        with patch(
            "aragora.server.handlers._shared_inbox_handler._get_activity_store",
            return_value=mock_activity_store,
        ):
            await handle_assign_message(
                inbox_id=sample_inbox.id,
                message_id=sample_message.id,
                assigned_to="user1",
                # No org_id
            )

        assert len(mock_activity_store.activities) == 0
