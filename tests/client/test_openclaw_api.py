"""
Tests for OpenClaw API resource.

Tests cover:
- Session management (create, get, list, end)
- Action execution (execute_action, execute_shell, execute_file_read, etc.)
- Policy management (get_policy_rules, add_rule, remove_rule)
- Approval workflows (list_pending_approvals, approve_action, deny_approval)
- Audit trail queries (query_audit with various filters)
- Stats retrieval (get_stats)
- Parsing helpers (_parse_session, _parse_action_result, _parse_rule, etc.)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from aragora.client.resources.openclaw import (
    ActionResult,
    AuditRecord,
    OpenClawAPI,
    OpenClawSession,
    PendingApproval,
    PolicyRule,
    ProxyStats,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock()
    return client


@pytest.fixture
def api(mock_client: MagicMock) -> OpenClawAPI:
    """Create an OpenClawAPI with mock client."""
    return OpenClawAPI(mock_client)


# ============================================================================
# Session Management Tests
# ============================================================================


class TestSessionManagement:
    """Tests for session create, get, list, and end operations."""

    def test_create_session(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test creating a new proxy session with required parameters."""
        mock_client._post.return_value = {
            "session_id": "sess-1",
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "workspace_id": "/workspace",
            "roles": ["developer"],
            "status": "active",
            "action_count": 0,
        }

        session = api.create_session(user_id="user-1", tenant_id="tenant-1", roles=["developer"])

        assert isinstance(session, OpenClawSession)
        assert session.session_id == "sess-1"
        assert session.user_id == "user-1"
        assert session.tenant_id == "tenant-1"
        assert session.roles == ["developer"]
        assert session.status == "active"
        mock_client._post.assert_called_once_with(
            "/api/v1/openclaw/sessions",
            json={
                "user_id": "user-1",
                "tenant_id": "tenant-1",
                "workspace_id": "/workspace",
                "roles": ["developer"],
            },
        )

    def test_create_session_with_metadata(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test creating a session with optional metadata."""
        mock_client._post.return_value = {
            "session_id": "sess-2",
            "user_id": "user-2",
            "tenant_id": "tenant-2",
            "workspace_id": "/custom/path",
            "roles": ["admin"],
            "status": "active",
            "metadata": {"source": "ci"},
        }

        session = api.create_session(
            user_id="user-2",
            tenant_id="tenant-2",
            workspace_id="/custom/path",
            roles=["admin"],
            metadata={"source": "ci"},
        )

        assert session.session_id == "sess-2"
        assert session.workspace_id == "/custom/path"
        assert session.metadata == {"source": "ci"}
        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["workspace_id"] == "/custom/path"
        assert call_payload["metadata"] == {"source": "ci"}

    def test_get_session(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test retrieving an existing session by ID."""
        mock_client._get.return_value = {
            "session_id": "sess-get",
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "workspace_id": "/workspace",
            "roles": ["viewer"],
            "status": "active",
            "action_count": 5,
            "created_at": "2025-06-15T10:00:00",
        }

        session = api.get_session("sess-get")

        assert session.session_id == "sess-get"
        assert session.action_count == 5
        assert session.created_at == datetime.fromisoformat("2025-06-15T10:00:00")
        mock_client._get.assert_called_once_with("/api/v1/openclaw/sessions/sess-get")

    def test_list_sessions_no_filters(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test listing sessions without any filters."""
        mock_client._get.return_value = {
            "sessions": [
                {
                    "session_id": "sess-a",
                    "user_id": "user-1",
                    "tenant_id": "tenant-1",
                    "workspace_id": "/workspace",
                    "status": "active",
                },
                {
                    "session_id": "sess-b",
                    "user_id": "user-2",
                    "tenant_id": "tenant-1",
                    "workspace_id": "/workspace",
                    "status": "ended",
                },
            ]
        }

        sessions = api.list_sessions()

        assert len(sessions) == 2
        assert sessions[0].session_id == "sess-a"
        assert sessions[1].session_id == "sess-b"
        mock_client._get.assert_called_once_with("/api/v1/openclaw/sessions", params={"limit": 50})

    def test_list_sessions_with_filters(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test listing sessions with tenant_id, status, and limit filters."""
        mock_client._get.return_value = {"sessions": []}

        api.list_sessions(tenant_id="tenant-x", status="active", limit=10)

        mock_client._get.assert_called_once_with(
            "/api/v1/openclaw/sessions",
            params={"limit": 10, "tenant_id": "tenant-x", "status": "active"},
        )

    def test_end_session_success(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test ending a session returns True on success."""
        mock_client._post.return_value = {"success": True}

        result = api.end_session("sess-end")

        assert result is True
        mock_client._post.assert_called_once_with("/api/v1/openclaw/sessions/sess-end/end")

    def test_end_session_failure(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test ending a session returns False when server indicates failure."""
        mock_client._post.return_value = {"success": False}

        result = api.end_session("sess-bad")

        assert result is False


# ============================================================================
# Action Execution Tests
# ============================================================================


class TestActionExecution:
    """Tests for action execution methods."""

    def test_execute_action(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test executing a generic action through the proxy."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-1",
            "decision": "allow",
            "result": {"output": "hello world"},
            "execution_time_ms": 42.5,
            "audit_id": "aud-1",
        }

        result = api.execute_action("sess-1", "shell", command="echo hello world")

        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.action_id == "act-1"
        assert result.decision == "allow"
        assert result.result == {"output": "hello world"}
        assert result.execution_time_ms == 42.5
        assert result.audit_id == "aud-1"
        mock_client._post.assert_called_once_with(
            "/api/v1/openclaw/actions",
            json={
                "session_id": "sess-1",
                "action_type": "shell",
                "input_data": {"command": "echo hello world"},
            },
        )

    def test_execute_action_denied(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test executing an action that is denied by policy."""
        mock_client._post.return_value = {
            "success": False,
            "action_id": "act-denied",
            "decision": "deny",
            "error": "Action not permitted by policy",
        }

        result = api.execute_action("sess-1", "shell", command="rm -rf /")

        assert result.success is False
        assert result.decision == "deny"
        assert result.error == "Action not permitted by policy"

    def test_execute_action_requires_approval(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test executing an action that requires human approval."""
        mock_client._post.return_value = {
            "success": False,
            "action_id": "act-pending",
            "decision": "require_approval",
            "requires_approval": True,
            "approval_id": "appr-100",
        }

        result = api.execute_action("sess-1", "file_write", path="/etc/config", content="data")

        assert result.success is False
        assert result.decision == "require_approval"
        assert result.requires_approval is True
        assert result.approval_id == "appr-100"

    def test_execute_shell(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test execute_shell delegates to execute_action with correct params."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-sh",
            "decision": "allow",
            "result": "output",
        }

        result = api.execute_shell("sess-1", "ls -la")

        assert result.success is True
        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["action_type"] == "shell"
        assert call_payload["input_data"] == {"command": "ls -la"}

    def test_execute_file_read(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test execute_file_read delegates with file_read action type."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-fr",
            "decision": "allow",
            "result": "file contents here",
        }

        result = api.execute_file_read("sess-1", "/tmp/test.txt")

        assert result.success is True
        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["action_type"] == "file_read"
        assert call_payload["input_data"] == {"path": "/tmp/test.txt"}

    def test_execute_file_write(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test execute_file_write delegates with file_write action type and content."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-fw",
            "decision": "allow",
        }

        result = api.execute_file_write("sess-1", "/tmp/out.txt", "new content")

        assert result.success is True
        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["action_type"] == "file_write"
        assert call_payload["input_data"] == {"path": "/tmp/out.txt", "content": "new content"}

    def test_execute_browser(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test execute_browser delegates with browser action type."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-br",
            "decision": "allow",
            "result": {"title": "Example"},
        }

        result = api.execute_browser("sess-1", "https://example.com", action="navigate")

        assert result.success is True
        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["action_type"] == "browser"
        assert call_payload["input_data"] == {"url": "https://example.com", "action": "navigate"}

    def test_execute_browser_default_action(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test execute_browser uses navigate as default action."""
        mock_client._post.return_value = {
            "success": True,
            "action_id": "act-br-def",
            "decision": "allow",
        }

        api.execute_browser("sess-1", "https://example.com")

        call_payload = mock_client._post.call_args[1]["json"]
        assert call_payload["input_data"]["action"] == "navigate"


# ============================================================================
# Policy Management Tests
# ============================================================================


class TestPolicyManagement:
    """Tests for policy rule CRUD operations."""

    def test_get_policy_rules(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test retrieving all active policy rules."""
        mock_client._get.return_value = {
            "rules": [
                {
                    "name": "block-rm",
                    "action_types": ["shell"],
                    "decision": "deny",
                    "priority": 100,
                    "description": "Block destructive shell commands",
                    "enabled": True,
                    "config": {"pattern": "rm -rf"},
                },
                {
                    "name": "allow-read",
                    "action_types": ["file_read"],
                    "decision": "allow",
                    "priority": 50,
                },
            ]
        }

        rules = api.get_policy_rules()

        assert len(rules) == 2
        assert isinstance(rules[0], PolicyRule)
        assert rules[0].name == "block-rm"
        assert rules[0].action_types == ["shell"]
        assert rules[0].decision == "deny"
        assert rules[0].priority == 100
        assert rules[0].config == {"pattern": "rm -rf"}
        assert rules[1].name == "allow-read"
        mock_client._get.assert_called_once_with("/api/v1/openclaw/policy/rules")

    def test_get_policy_rules_empty(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test retrieving policy rules when none exist."""
        mock_client._get.return_value = {"rules": []}

        rules = api.get_policy_rules()

        assert rules == []

    def test_add_rule(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test adding a new policy rule."""
        mock_client._post.return_value = {
            "name": "require-approval-write",
            "action_types": ["file_write"],
            "decision": "require_approval",
            "priority": 75,
            "description": "Require approval for file writes",
            "enabled": True,
            "config": {},
        }

        rule = PolicyRule(
            name="require-approval-write",
            action_types=["file_write"],
            decision="require_approval",
            priority=75,
            description="Require approval for file writes",
        )
        result = api.add_rule(rule)

        assert isinstance(result, PolicyRule)
        assert result.name == "require-approval-write"
        assert result.decision == "require_approval"
        mock_client._post.assert_called_once_with(
            "/api/v1/openclaw/policy/rules",
            json={
                "name": "require-approval-write",
                "action_types": ["file_write"],
                "decision": "require_approval",
                "priority": 75,
                "description": "Require approval for file writes",
                "enabled": True,
                "config": {},
            },
        )

    def test_remove_rule_success(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test removing a policy rule successfully."""
        mock_client._delete.return_value = {"success": True}

        result = api.remove_rule("block-rm")

        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/openclaw/policy/rules/block-rm")

    def test_remove_rule_failure(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test removing a rule that does not exist returns False."""
        mock_client._delete.return_value = {"success": False}

        result = api.remove_rule("nonexistent-rule")

        assert result is False


# ============================================================================
# Approval Workflow Tests
# ============================================================================


class TestApprovalWorkflows:
    """Tests for approval listing, approval, and denial operations."""

    def test_list_pending_approvals(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test listing pending approval requests."""
        mock_client._get.return_value = {
            "approvals": [
                {
                    "approval_id": "appr-1",
                    "session_id": "sess-1",
                    "user_id": "user-1",
                    "tenant_id": "tenant-1",
                    "action_type": "file_write",
                    "action_params": {"path": "/etc/config"},
                    "status": "pending",
                    "created_at": "2025-06-15T10:00:00",
                    "expires_at": "2025-06-15T11:00:00",
                },
            ]
        }

        approvals = api.list_pending_approvals(tenant_id="tenant-1")

        assert len(approvals) == 1
        assert isinstance(approvals[0], PendingApproval)
        assert approvals[0].approval_id == "appr-1"
        assert approvals[0].action_type == "file_write"
        assert approvals[0].status == "pending"
        assert approvals[0].created_at == datetime.fromisoformat("2025-06-15T10:00:00")
        assert approvals[0].expires_at == datetime.fromisoformat("2025-06-15T11:00:00")
        mock_client._get.assert_called_once_with(
            "/api/v1/openclaw/approvals",
            params={"limit": 50, "tenant_id": "tenant-1"},
        )

    def test_list_pending_approvals_with_limit(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test listing approvals with a custom limit."""
        mock_client._get.return_value = {"approvals": []}

        api.list_pending_approvals(limit=5)

        mock_client._get.assert_called_once_with(
            "/api/v1/openclaw/approvals",
            params={"limit": 5},
        )

    def test_approve_action(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test approving a pending action."""
        mock_client._post.return_value = {"success": True}

        result = api.approve_action("appr-1", approver_id="admin-1", reason="Looks safe")

        assert result is True
        mock_client._post.assert_called_once_with(
            "/api/v1/openclaw/approvals/appr-1/approve",
            json={"approver_id": "admin-1", "reason": "Looks safe"},
        )

    def test_deny_approval(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test denying a pending approval."""
        mock_client._post.return_value = {"success": True}

        result = api.deny_approval("appr-2", approver_id="admin-1", reason="Too risky")

        assert result is True
        mock_client._post.assert_called_once_with(
            "/api/v1/openclaw/approvals/appr-2/deny",
            json={"approver_id": "admin-1", "reason": "Too risky"},
        )

    def test_deny_approval_failure(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test deny returns False when operation fails."""
        mock_client._post.return_value = {"success": False}

        result = api.deny_approval("appr-invalid", approver_id="admin-1")

        assert result is False


# ============================================================================
# Audit Query Tests
# ============================================================================


class TestAuditQueries:
    """Tests for audit trail query operations."""

    def test_query_audit_no_filters(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test querying audit records with no filters."""
        mock_client._get.return_value = {
            "records": [
                {
                    "record_id": "rec-1",
                    "event_type": "action_executed",
                    "timestamp": 1718444400.0,
                    "user_id": "user-1",
                    "session_id": "sess-1",
                    "action_type": "shell",
                    "success": True,
                    "metadata": {"command": "ls"},
                },
            ]
        }

        records = api.query_audit()

        assert len(records) == 1
        assert isinstance(records[0], AuditRecord)
        assert records[0].record_id == "rec-1"
        assert records[0].event_type == "action_executed"
        assert records[0].timestamp == 1718444400.0
        assert records[0].user_id == "user-1"
        assert records[0].success is True
        assert records[0].metadata == {"command": "ls"}
        mock_client._get.assert_called_once_with(
            "/api/v1/openclaw/audit",
            params={"limit": 50},
        )

    def test_query_audit_with_all_filters(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test querying audit records with all available filters."""
        mock_client._get.return_value = {"records": []}

        api.query_audit(
            user_id="user-x",
            session_id="sess-x",
            event_type="session_created",
            tenant_id="tenant-x",
            limit=10,
        )

        mock_client._get.assert_called_once_with(
            "/api/v1/openclaw/audit",
            params={
                "limit": 10,
                "user_id": "user-x",
                "session_id": "sess-x",
                "event_type": "session_created",
                "tenant_id": "tenant-x",
            },
        )

    def test_query_audit_failed_action(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test querying audit records that include failed actions."""
        mock_client._get.return_value = {
            "records": [
                {
                    "record_id": "rec-fail",
                    "event_type": "action_denied",
                    "timestamp": 1718445000.0,
                    "action_type": "shell",
                    "success": False,
                    "error": "Policy violation",
                },
            ]
        }

        records = api.query_audit(event_type="action_denied")

        assert len(records) == 1
        assert records[0].success is False
        assert records[0].error == "Policy violation"


# ============================================================================
# Stats Tests
# ============================================================================


class TestStats:
    """Tests for proxy stats retrieval."""

    def test_get_stats(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test retrieving proxy statistics."""
        mock_client._get.return_value = {
            "active_sessions": 12,
            "actions_allowed": 340,
            "actions_denied": 15,
            "pending_approvals": 3,
            "policy_rules": 8,
        }

        stats = api.get_stats()

        assert isinstance(stats, ProxyStats)
        assert stats.active_sessions == 12
        assert stats.actions_allowed == 340
        assert stats.actions_denied == 15
        assert stats.pending_approvals == 3
        assert stats.policy_rules == 8
        mock_client._get.assert_called_once_with("/api/v1/openclaw/stats")

    def test_get_stats_defaults(self, api: OpenClawAPI, mock_client: MagicMock):
        """Test that missing stats fields default to zero."""
        mock_client._get.return_value = {}

        stats = api.get_stats()

        assert stats.active_sessions == 0
        assert stats.actions_allowed == 0
        assert stats.actions_denied == 0
        assert stats.pending_approvals == 0
        assert stats.policy_rules == 0


# ============================================================================
# Parsing Helper Tests
# ============================================================================


class TestParsingHelpers:
    """Tests for the static parsing helper methods."""

    def test_parse_session_full(self):
        """Test _parse_session with all fields present."""
        data = {
            "session_id": "sess-parse",
            "user_id": "user-parse",
            "tenant_id": "tenant-parse",
            "workspace_id": "/custom",
            "roles": ["admin", "developer"],
            "status": "ended",
            "action_count": 42,
            "created_at": "2025-03-10T08:30:00",
            "last_activity": "2025-03-10T09:15:00",
            "metadata": {"env": "production"},
        }

        session = OpenClawAPI._parse_session(data)

        assert session.session_id == "sess-parse"
        assert session.workspace_id == "/custom"
        assert session.roles == ["admin", "developer"]
        assert session.status == "ended"
        assert session.action_count == 42
        assert session.created_at == datetime.fromisoformat("2025-03-10T08:30:00")
        assert session.last_activity == datetime.fromisoformat("2025-03-10T09:15:00")
        assert session.metadata == {"env": "production"}

    def test_parse_session_minimal(self):
        """Test _parse_session with minimal data uses defaults."""
        session = OpenClawAPI._parse_session({})

        assert session.session_id == ""
        assert session.user_id == ""
        assert session.workspace_id == "/workspace"
        assert session.roles == []
        assert session.status == "active"
        assert session.action_count == 0
        assert session.created_at is None
        assert session.last_activity is None
        assert session.metadata == {}

    def test_parse_session_invalid_datetime(self):
        """Test _parse_session gracefully handles invalid datetime strings."""
        data = {
            "session_id": "sess-bad-date",
            "user_id": "u",
            "tenant_id": "t",
            "created_at": "not-a-date",
            "last_activity": "also-not-a-date",
        }

        session = OpenClawAPI._parse_session(data)

        assert session.created_at is None
        assert session.last_activity is None

    def test_parse_action_result_full(self):
        """Test _parse_action_result with all fields present."""
        data = {
            "success": True,
            "action_id": "act-parse",
            "decision": "allow",
            "result": {"output": "data"},
            "error": None,
            "execution_time_ms": 123.4,
            "audit_id": "aud-parse",
            "requires_approval": False,
            "approval_id": None,
        }

        result = OpenClawAPI._parse_action_result(data)

        assert result.success is True
        assert result.action_id == "act-parse"
        assert result.decision == "allow"
        assert result.result == {"output": "data"}
        assert result.error is None
        assert result.execution_time_ms == 123.4
        assert result.audit_id == "aud-parse"
        assert result.requires_approval is False

    def test_parse_action_result_defaults(self):
        """Test _parse_action_result defaults for missing fields."""
        result = OpenClawAPI._parse_action_result({})

        assert result.success is False
        assert result.action_id == ""
        assert result.decision == "deny"
        assert result.result is None
        assert result.execution_time_ms == 0.0
        assert result.requires_approval is False

    def test_parse_rule(self):
        """Test _parse_rule with complete data."""
        data = {
            "name": "test-rule",
            "action_types": ["shell", "browser"],
            "decision": "allow",
            "priority": 10,
            "description": "A test rule",
            "enabled": False,
            "config": {"max_retries": 3},
        }

        rule = OpenClawAPI._parse_rule(data)

        assert rule.name == "test-rule"
        assert rule.action_types == ["shell", "browser"]
        assert rule.decision == "allow"
        assert rule.priority == 10
        assert rule.description == "A test rule"
        assert rule.enabled is False
        assert rule.config == {"max_retries": 3}

    def test_parse_rule_defaults(self):
        """Test _parse_rule defaults for missing fields."""
        rule = OpenClawAPI._parse_rule({})

        assert rule.name == ""
        assert rule.action_types == []
        assert rule.decision == "deny"
        assert rule.priority == 0
        assert rule.description == ""
        assert rule.enabled is True
        assert rule.config == {}

    def test_parse_approval_with_dates(self):
        """Test _parse_approval parses datetime fields correctly."""
        data = {
            "approval_id": "appr-parse",
            "session_id": "sess-1",
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "action_type": "browser",
            "action_params": {"url": "https://example.com"},
            "status": "approved",
            "created_at": "2025-06-15T10:00:00",
            "expires_at": "2025-06-15T11:00:00",
        }

        approval = OpenClawAPI._parse_approval(data)

        assert approval.approval_id == "appr-parse"
        assert approval.action_params == {"url": "https://example.com"}
        assert approval.status == "approved"
        assert approval.created_at == datetime.fromisoformat("2025-06-15T10:00:00")
        assert approval.expires_at == datetime.fromisoformat("2025-06-15T11:00:00")

    def test_parse_approval_invalid_dates(self):
        """Test _parse_approval handles invalid date strings gracefully."""
        data = {
            "approval_id": "appr-bad",
            "session_id": "s",
            "user_id": "u",
            "tenant_id": "t",
            "action_type": "shell",
            "created_at": "garbage",
            "expires_at": "also-garbage",
        }

        approval = OpenClawAPI._parse_approval(data)

        assert approval.created_at is None
        assert approval.expires_at is None

    def test_parse_audit_record(self):
        """Test _parse_audit_record with complete data."""
        data = {
            "record_id": "rec-parse",
            "event_type": "session_ended",
            "timestamp": 1718450000.0,
            "user_id": "user-1",
            "session_id": "sess-1",
            "tenant_id": "tenant-1",
            "action_type": None,
            "success": True,
            "error": None,
            "metadata": {"reason": "timeout"},
        }

        record = OpenClawAPI._parse_audit_record(data)

        assert record.record_id == "rec-parse"
        assert record.event_type == "session_ended"
        assert record.timestamp == 1718450000.0
        assert record.user_id == "user-1"
        assert record.action_type is None
        assert record.metadata == {"reason": "timeout"}

    def test_parse_audit_record_defaults(self):
        """Test _parse_audit_record defaults for missing fields."""
        record = OpenClawAPI._parse_audit_record({})

        assert record.record_id == ""
        assert record.event_type == ""
        assert record.timestamp == 0.0
        assert record.user_id is None
        assert record.session_id is None
        assert record.success is True
        assert record.error is None
        assert record.metadata == {}
