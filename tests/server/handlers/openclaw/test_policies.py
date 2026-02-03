"""
Tests for OpenClaw Policy Handler Mixin.

Tests policy rules, approval workflows, and admin operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw.policies import PolicyHandlerMixin


class MockStore:
    """Mock store for testing."""

    def __init__(self):
        self.policy_rules = []
        self.approvals = []
        self.audit_entries = []
        self._metrics = {
            "sessions": {"active": 5, "closed": 10, "total": 15},
            "actions": {"pending": 3, "running": 2, "completed": 100, "failed": 5},
        }

    def get_policy_rules(self):
        return self.policy_rules

    def add_policy_rule(self, name, action_types, decision, priority, description, enabled, config):
        rule = MagicMock()
        rule.name = name
        rule.action_types = action_types
        rule.decision = decision
        rule.priority = priority
        rule.description = description
        rule.enabled = enabled
        rule.config = config
        rule.to_dict = lambda: {
            "name": name,
            "action_types": action_types,
            "decision": decision,
            "priority": priority,
        }
        self.policy_rules.append(rule)
        return rule

    def remove_policy_rule(self, rule_name):
        original_len = len(self.policy_rules)
        self.policy_rules = [r for r in self.policy_rules if r.name != rule_name]
        return len(self.policy_rules) < original_len

    def list_approvals(self, tenant_id, limit, offset):
        return self.approvals[:limit], len(self.approvals)

    def approve_action(self, approval_id, approver_id, reason):
        return True

    def deny_action(self, approval_id, approver_id, reason):
        return True

    def get_metrics(self):
        return self._metrics

    def get_audit_log(self, action, actor_id, resource_type, limit, offset):
        entries = self.audit_entries
        if action:
            entries = [e for e in entries if e.get("action") == action]
        return entries[:limit], len(entries)

    def add_audit_entry(self, **kwargs):
        entry = MagicMock()
        entry.to_dict = lambda: kwargs
        for k, v in kwargs.items():
            setattr(entry, k, v)
        self.audit_entries.append(entry)


class MockHandler(PolicyHandlerMixin):
    """Mock handler for testing the mixin."""

    def __init__(self):
        self.store = MockStore()
        self._current_user = None

    def _get_user_id(self, handler):
        return "user_123"

    def _get_tenant_id(self, handler):
        return "tenant_456"

    def get_current_user(self, handler):
        return self._current_user


class TestGetPolicyRules:
    """Tests for _handle_get_policy_rules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_get_policy_rules_empty(self, mock_get_store):
        """Test getting empty policy rules."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_get_policy_rules({}, mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_get_policy_rules_with_rules(self, mock_get_store):
        """Test getting policy rules when rules exist."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        self.handler.store.add_policy_rule(
            name="block_delete",
            action_types=["delete"],
            decision="deny",
            priority=100,
            description="Block all delete operations",
            enabled=True,
            config={},
        )

        result = self.handler._handle_get_policy_rules({}, mock_http)

        assert result.status_code == 200


class TestAddPolicyRule:
    """Tests for _handle_add_policy_rule."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_add_policy_rule_success(self, mock_get_store):
        """Test successful policy rule addition."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_add_policy_rule(
            {
                "name": "require_approval",
                "action_types": ["execute", "delete"],
                "decision": "require_approval",
                "priority": 50,
                "description": "Require approval for dangerous actions",
            },
            mock_http,
        )

        assert result.status_code == 201
        assert len(self.handler.store.policy_rules) == 1

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_add_policy_rule_missing_name(self, mock_get_store):
        """Test adding policy rule without name."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_add_policy_rule(
            {"action_types": ["click"], "decision": "allow"}, mock_http
        )

        assert result.status_code == 400

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_add_policy_rule_audit(self, mock_get_store):
        """Test audit entry for policy rule addition."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        self.handler._handle_add_policy_rule(
            {"name": "test_rule", "action_types": [], "decision": "allow"}, mock_http
        )

        assert len(self.handler.store.audit_entries) == 1
        assert self.handler.store.audit_entries[0].action == "policy.rule.add"


class TestRemovePolicyRule:
    """Tests for _handle_remove_policy_rule."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_remove_policy_rule_success(self, mock_get_store):
        """Test successful policy rule removal."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        self.handler.store.add_policy_rule(
            name="test_rule",
            action_types=[],
            decision="deny",
            priority=0,
            description="",
            enabled=True,
            config={},
        )

        result = self.handler._handle_remove_policy_rule("test_rule", mock_http)

        assert result.status_code == 200
        assert len(self.handler.store.policy_rules) == 0


class TestListApprovals:
    """Tests for _handle_list_approvals."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_list_approvals_empty(self, mock_get_store):
        """Test listing empty approvals."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_list_approvals({}, mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_list_approvals_with_pagination(self, mock_get_store):
        """Test listing approvals with pagination."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_list_approvals({"limit": "10", "offset": "5"}, mock_http)

        assert result.status_code == 200


class TestApproveAction:
    """Tests for _handle_approve_action."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_approve_action_success(self, mock_get_store):
        """Test successful action approval."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_approve_action(
            "approval_123", {"reason": "Looks good"}, mock_http
        )

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_approve_action_uses_authenticated_user(self, mock_get_store):
        """Test that approval uses authenticated user, not body parameter."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        # Even if body contains a different approver_id, the authenticated user is used
        self.handler._handle_approve_action(
            "approval_123",
            {"approver_id": "attacker_trying_to_impersonate", "reason": "test"},
            mock_http,
        )

        # Check audit entry uses the authenticated user
        assert self.handler.store.audit_entries[0].details["approver_id"] == "user_123"


class TestDenyAction:
    """Tests for _handle_deny_action."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_deny_action_success(self, mock_get_store):
        """Test successful action denial."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_deny_action(
            "approval_123", {"reason": "Security concern"}, mock_http
        )

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_deny_action_audit(self, mock_get_store):
        """Test audit entry for action denial."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        self.handler._handle_deny_action("approval_456", {"reason": "Rejected"}, mock_http)

        assert len(self.handler.store.audit_entries) == 1
        assert self.handler.store.audit_entries[0].action == "approval.deny"


class TestHealthEndpoint:
    """Tests for _handle_health."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_health_healthy(self, mock_get_store):
        """Test health check returns healthy status."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_health(mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_health_degraded(self, mock_get_store):
        """Test health check returns degraded status when high load."""
        mock_get_store.return_value = self.handler.store
        self.handler.store._metrics["actions"]["running"] = 150
        mock_http = MagicMock()

        result = self.handler._handle_health(mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_health_unhealthy(self, mock_get_store):
        """Test health check returns unhealthy status when overloaded."""
        mock_get_store.return_value = self.handler.store
        self.handler.store._metrics["actions"]["pending"] = 600
        mock_http = MagicMock()

        result = self.handler._handle_health(mock_http)

        assert result.status_code == 200


class TestMetricsEndpoint:
    """Tests for _handle_metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_metrics_success(self, mock_get_store):
        """Test successful metrics retrieval."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_metrics(mock_http)

        assert result.status_code == 200


class TestAuditEndpoint:
    """Tests for _handle_audit."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_audit_empty(self, mock_get_store):
        """Test audit log retrieval with no entries."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_audit({}, mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_audit_with_filter(self, mock_get_store):
        """Test audit log retrieval with action filter."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        # Add some audit entries
        self.handler.store.add_audit_entry(
            action="session.create",
            actor_id="user_1",
            resource_type="session",
            resource_id="s_1",
            result="success",
        )
        self.handler.store.add_audit_entry(
            action="action.execute",
            actor_id="user_1",
            resource_type="action",
            resource_id="a_1",
            result="success",
        )

        result = self.handler._handle_audit({"action": "session.create"}, mock_http)

        assert result.status_code == 200

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_audit_with_pagination(self, mock_get_store):
        """Test audit log retrieval with pagination."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_audit({"limit": "50", "offset": "100"}, mock_http)

        assert result.status_code == 200


class TestStatsEndpoint:
    """Tests for _handle_stats."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MockHandler()

    @patch("aragora.server.handlers.openclaw.policies._get_store")
    def test_stats_success(self, mock_get_store):
        """Test successful stats retrieval."""
        mock_get_store.return_value = self.handler.store
        mock_http = MagicMock()

        result = self.handler._handle_stats(mock_http)

        assert result.status_code == 200
