"""Comprehensive tests for OpenClaw PolicyHandlerMixin.

Covers all handler methods defined in
aragora/server/handlers/openclaw/policies.py:

Policy rule handlers:
- _handle_get_policy_rules   (GET    /api/v1/openclaw/policy/rules)
- _handle_add_policy_rule    (POST   /api/v1/openclaw/policy/rules)
- _handle_remove_policy_rule (DELETE /api/v1/openclaw/policy/rules/:name)

Approval handlers:
- _handle_list_approvals     (GET    /api/v1/openclaw/approvals)
- _handle_approve_action     (POST   /api/v1/openclaw/approvals/:id/approve)
- _handle_deny_action        (POST   /api/v1/openclaw/approvals/:id/deny)

Admin handlers:
- _handle_health             (GET    /api/v1/openclaw/health)
- _handle_metrics            (GET    /api/v1/openclaw/metrics)
- _handle_audit              (GET    /api/v1/openclaw/audit)
- _handle_stats              (GET    /api/v1/openclaw/stats)

Test categories:
- Happy paths for every endpoint
- Store fallback (missing methods)
- Validation errors (missing fields)
- Error handling (exceptions -> 500/503)
- Query parameter parsing (filters, pagination)
- Audit logging side effects
- Security (approver_id forced to authenticated user)
- Legacy path support
- Edge cases (empty results, to_dict conversion)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw.gateway import OpenClawGatewayHandler
from aragora.server.handlers.openclaw.models import AuditEntry
from aragora.server.handlers.openclaw.policies import PolicyHandlerMixin
from aragora.server.handlers.openclaw.store import OpenClawGatewayStore


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict[str, Any]:
    """Decode a HandlerResult body to dict."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        return json.loads(result.body)
    return result


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return 0


class MockHTTPHandler:
    """Minimal mock HTTP handler for BaseHandler methods."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.rfile = MagicMock()
        self.command = method
        self.client_address = ("127.0.0.1", 54321)
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }


class _MockUser:
    """Minimal mock user for get_current_user."""

    def __init__(self, user_id="test-user-001", org_id="test-org-001", role="admin"):
        self.user_id = user_id
        self.org_id = org_id
        self.role = role


class _MockPolicyRule:
    """Mock policy rule object with to_dict support."""

    def __init__(self, name, action_types=None, decision="deny", priority=0):
        self.name = name
        self.action_types = action_types or []
        self.decision = decision
        self.priority = priority

    def to_dict(self):
        return {
            "name": self.name,
            "action_types": self.action_types,
            "decision": self.decision,
            "priority": self.priority,
        }


class _MockApproval:
    """Mock approval object with to_dict support."""

    def __init__(self, approval_id, status="pending"):
        self.id = approval_id
        self.status = status

    def to_dict(self):
        return {"id": self.id, "status": self.status}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def store():
    """Fresh in-memory OpenClaw store for each test."""
    return OpenClawGatewayStore()


@pytest.fixture()
def mock_user():
    """Default mock user returned by get_current_user."""
    return _MockUser()


@pytest.fixture()
def handler(store, mock_user):
    """OpenClawGatewayHandler with _get_store and get_current_user patched."""
    with patch(
        "aragora.server.handlers.openclaw.orchestrator._get_store",
        return_value=store,
    ), patch(
        "aragora.server.handlers.openclaw.credentials._get_store",
        return_value=store,
    ), patch(
        "aragora.server.handlers.openclaw.policies._get_store",
        return_value=store,
    ):
        h = OpenClawGatewayHandler(server_context={})
        h.get_current_user = lambda handler: mock_user
        yield h


@pytest.fixture()
def mock_http():
    """Factory for MockHTTPHandler."""

    def _make(body: dict | None = None, method: str = "GET") -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method=method)

    return _make


# ============================================================================
# GET Policy Rules
# ============================================================================


class TestGetPolicyRules:
    """Tests for _handle_get_policy_rules (GET /policy/rules)."""

    def test_get_policy_rules_empty_store(self, handler, mock_http):
        """Empty store returns empty list with total=0."""
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["rules"] == []
        assert body["total"] == 0

    def test_get_policy_rules_store_has_method(self, handler, mock_http, store):
        """When store has get_policy_rules method, its results are returned."""
        rule = _MockPolicyRule("block_shell", ["shell_exec"], "deny", 10)
        store.get_policy_rules = MagicMock(return_value=[rule])
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["rules"][0]["name"] == "block_shell"
        assert body["rules"][0]["decision"] == "deny"
        assert body["rules"][0]["priority"] == 10

    def test_get_policy_rules_multiple_rules(self, handler, mock_http, store):
        """Multiple rules are all returned."""
        rules = [
            _MockPolicyRule("rule1", ["type_a"], "deny", 1),
            _MockPolicyRule("rule2", ["type_b"], "allow", 2),
            _MockPolicyRule("rule3", ["type_c"], "deny", 3),
        ]
        store.get_policy_rules = MagicMock(return_value=rules)
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        names = [r["name"] for r in body["rules"]]
        assert names == ["rule1", "rule2", "rule3"]

    def test_get_policy_rules_without_to_dict(self, handler, mock_http, store):
        """Rules without to_dict are returned as-is (dict)."""
        raw_rules = [
            {"name": "raw_rule", "decision": "deny"},
        ]
        store.get_policy_rules = MagicMock(return_value=raw_rules)
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["rules"][0]["name"] == "raw_rule"

    def test_get_policy_rules_store_without_method(self, handler, mock_http, store):
        """Store without get_policy_rules returns empty list."""
        # Default OpenClawGatewayStore doesn't have get_policy_rules
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["rules"] == []
        assert body["total"] == 0

    def test_get_policy_rules_store_error_returns_500(self, handler, mock_http, store):
        """Store exception returns 500."""
        store.get_policy_rules = MagicMock(side_effect=ValueError("parse error"))
        result = handler._handle_get_policy_rules({}, mock_http())
        assert _status(result) == 500

    def test_get_policy_rules_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.get_policy_rules = MagicMock(side_effect=KeyError("missing_key"))
        result = handler._handle_get_policy_rules({}, mock_http())
        assert _status(result) == 500

    def test_get_policy_rules_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.get_policy_rules = MagicMock(side_effect=OSError("disk failure"))
        result = handler._handle_get_policy_rules({}, mock_http())
        assert _status(result) == 500

    def test_get_policy_rules_via_legacy_path(self, handler, mock_http, store):
        """Legacy /api/gateway/openclaw/policy/rules works."""
        store.get_policy_rules = MagicMock(return_value=[])
        result = handler.handle("/api/gateway/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        assert _body(result)["rules"] == []


# ============================================================================
# Add Policy Rule (POST)
# ============================================================================


class TestAddPolicyRule:
    """Tests for _handle_add_policy_rule (POST /policy/rules)."""

    def test_add_policy_rule_success(self, handler, mock_http, store):
        """Successfully add a policy rule."""
        body = {
            "name": "new_rule",
            "action_types": ["shell_exec"],
            "decision": "deny",
            "priority": 5,
            "description": "Block shell execution",
            "enabled": True,
            "config": {"timeout": 30},
        }
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "new_rule"
        assert resp["decision"] == "deny"
        assert resp["priority"] == 5

    def test_add_policy_rule_minimal_body(self, handler, mock_http, store):
        """Only name is required; defaults fill in the rest."""
        body = {"name": "minimal_rule"}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "minimal_rule"
        assert resp["action_types"] == []
        assert resp["decision"] == "deny"
        assert resp["priority"] == 0
        assert resp["description"] == ""
        assert resp["enabled"] is True
        assert resp["config"] == {}

    def test_add_policy_rule_missing_name_returns_400(self, handler, mock_http):
        """Missing name field returns 400."""
        body = {"decision": "deny"}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400
        resp = _body(result)
        assert "name" in resp.get("error", "").lower()

    def test_add_policy_rule_empty_name_returns_400(self, handler, mock_http):
        """Empty name string returns 400."""
        body = {"name": ""}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_add_policy_rule_null_name_returns_400(self, handler, mock_http):
        """Null name returns 400."""
        body = {"name": None}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_add_policy_rule_with_store_method(self, handler, mock_http, store):
        """When store has add_policy_rule, it's used."""
        mock_rule = _MockPolicyRule("stored_rule", ["type_a"], "allow", 10)
        store.add_policy_rule = MagicMock(return_value=mock_rule)
        body = {"name": "stored_rule", "action_types": ["type_a"], "decision": "allow", "priority": 10}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "stored_rule"
        store.add_policy_rule.assert_called_once()

    def test_add_policy_rule_store_without_method(self, handler, mock_http, store):
        """Without add_policy_rule, a plain dict fallback is used."""
        body = {"name": "fallback_rule", "decision": "allow", "priority": 3}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "fallback_rule"
        assert resp["decision"] == "allow"
        assert resp["priority"] == 3

    def test_add_policy_rule_creates_audit_entry(self, handler, mock_http, store):
        """Adding a rule creates an audit entry."""
        body = {"name": "audited_rule", "action_types": ["test"], "decision": "deny"}
        handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        entries, total = store.get_audit_log(action="policy.rule.add")
        assert total >= 1
        entry = entries[0]
        assert entry.action == "policy.rule.add"
        assert entry.result == "success"
        assert entry.resource_type == "policy_rule"
        assert entry.resource_id == "audited_rule"
        assert entry.details["decision"] == "deny"
        assert entry.details["action_types"] == ["test"]

    def test_add_policy_rule_audit_actor_is_user(self, handler, mock_http, store, mock_user):
        """Audit entry records the authenticated user as actor."""
        body = {"name": "tracked_rule"}
        handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        entries, _ = store.get_audit_log(action="policy.rule.add")
        assert entries[0].actor_id == mock_user.user_id

    def test_add_policy_rule_value_error_returns_500(self, handler, mock_http, store):
        """ValueError from store returns 500."""
        store.add_policy_rule = MagicMock(side_effect=ValueError("bad data"))
        body = {"name": "error_rule"}
        result = handler._handle_add_policy_rule(body, mock_http(method="POST"))
        assert _status(result) == 500

    def test_add_policy_rule_type_error_returns_500(self, handler, mock_http, store):
        """TypeError from store returns 500."""
        store.add_policy_rule = MagicMock(side_effect=TypeError("bad type"))
        body = {"name": "error_rule"}
        result = handler._handle_add_policy_rule(body, mock_http(method="POST"))
        assert _status(result) == 500

    def test_add_policy_rule_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.add_policy_rule = MagicMock(side_effect=OSError("io error"))
        body = {"name": "error_rule"}
        result = handler._handle_add_policy_rule(body, mock_http(method="POST"))
        assert _status(result) == 500

    def test_add_policy_rule_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.add_policy_rule = MagicMock(side_effect=KeyError("missing"))
        body = {"name": "error_rule"}
        result = handler._handle_add_policy_rule(body, mock_http(method="POST"))
        assert _status(result) == 500

    def test_add_policy_rule_allow_decision(self, handler, mock_http, store):
        """Rule with allow decision is stored correctly."""
        body = {"name": "allow_rule", "decision": "allow"}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        assert _body(result)["decision"] == "allow"

    def test_add_policy_rule_custom_config(self, handler, mock_http, store):
        """Custom config is passed through to the result."""
        body = {"name": "config_rule", "config": {"max_retries": 3, "timeout": 60}}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["config"] == {"max_retries": 3, "timeout": 60}

    def test_add_policy_rule_via_legacy_path(self, handler, mock_http, store):
        """Legacy path /api/gateway/openclaw/policy/rules works for POST."""
        body = {"name": "legacy_rule"}
        result = handler.handle_post(
            "/api/gateway/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201


# ============================================================================
# Remove Policy Rule (DELETE)
# ============================================================================


class TestRemovePolicyRule:
    """Tests for _handle_remove_policy_rule (DELETE /policy/rules/:name)."""

    def test_remove_policy_rule_success(self, handler, mock_http, store):
        """Successfully remove a policy rule."""
        store.remove_policy_rule = MagicMock(return_value=True)
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/block_shell", {}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["name"] == "block_shell"

    def test_remove_policy_rule_not_found(self, handler, mock_http, store):
        """Removing a non-existent rule returns success=False from store."""
        store.remove_policy_rule = MagicMock(return_value=False)
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/nonexistent", {}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False

    def test_remove_policy_rule_store_without_method(self, handler, mock_http, store):
        """Without remove_policy_rule method, returns success=True fallback."""
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/any_rule", {}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_remove_policy_rule_creates_audit_entry(self, handler, mock_http, store):
        """Removing a rule creates an audit entry."""
        store.remove_policy_rule = MagicMock(return_value=True)
        handler.handle_delete(
            "/api/v1/openclaw/policy/rules/audited_rule", {}, mock_http()
        )
        entries, total = store.get_audit_log(action="policy.rule.remove")
        assert total >= 1
        entry = entries[0]
        assert entry.action == "policy.rule.remove"
        assert entry.result == "success"
        assert entry.resource_type == "policy_rule"
        assert entry.resource_id == "audited_rule"

    def test_remove_policy_rule_audit_actor(self, handler, mock_http, store, mock_user):
        """Audit entry records authenticated user."""
        store.remove_policy_rule = MagicMock(return_value=True)
        handler.handle_delete(
            "/api/v1/openclaw/policy/rules/tracked_rule", {}, mock_http()
        )
        entries, _ = store.get_audit_log(action="policy.rule.remove")
        assert entries[0].actor_id == mock_user.user_id

    def test_remove_policy_rule_value_error_returns_500(self, handler, mock_http, store):
        """ValueError from store returns 500."""
        store.remove_policy_rule = MagicMock(side_effect=ValueError("bad data"))
        result = handler._handle_remove_policy_rule("bad_rule", mock_http())
        assert _status(result) == 500

    def test_remove_policy_rule_type_error_returns_500(self, handler, mock_http, store):
        """TypeError from store returns 500."""
        store.remove_policy_rule = MagicMock(side_effect=TypeError("bad type"))
        result = handler._handle_remove_policy_rule("bad_rule", mock_http())
        assert _status(result) == 500

    def test_remove_policy_rule_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.remove_policy_rule = MagicMock(side_effect=OSError("disk failure"))
        result = handler._handle_remove_policy_rule("bad_rule", mock_http())
        assert _status(result) == 500

    def test_remove_policy_rule_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.remove_policy_rule = MagicMock(side_effect=KeyError("missing"))
        result = handler._handle_remove_policy_rule("bad_rule", mock_http())
        assert _status(result) == 500

    def test_remove_policy_rule_via_legacy_path(self, handler, mock_http, store):
        """Legacy delete path works."""
        store.remove_policy_rule = MagicMock(return_value=True)
        result = handler.handle_delete(
            "/api/gateway/openclaw/policy/rules/legacy_rule", {}, mock_http()
        )
        assert _status(result) == 200
        assert _body(result)["name"] == "legacy_rule"


# ============================================================================
# List Approvals (GET)
# ============================================================================


class TestListApprovals:
    """Tests for _handle_list_approvals (GET /approvals)."""

    def test_list_approvals_empty(self, handler, mock_http):
        """Empty store returns empty approvals list."""
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["approvals"] == []
        assert body["total"] == 0
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_approvals_with_results(self, handler, mock_http, store):
        """Store with list_approvals returns results."""
        approvals = [_MockApproval("app-1", "pending"), _MockApproval("app-2", "approved")]
        store.list_approvals = MagicMock(return_value=(approvals, 2))
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["approvals"]) == 2
        assert body["approvals"][0]["id"] == "app-1"
        assert body["approvals"][1]["id"] == "app-2"

    def test_list_approvals_without_to_dict(self, handler, mock_http, store):
        """Approvals without to_dict are returned as-is."""
        raw_approvals = [{"id": "raw-1", "status": "pending"}]
        store.list_approvals = MagicMock(return_value=(raw_approvals, 1))
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["approvals"][0]["id"] == "raw-1"

    def test_list_approvals_store_without_method(self, handler, mock_http, store):
        """Store without list_approvals returns empty list."""
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["approvals"] == []
        assert body["total"] == 0

    def test_list_approvals_custom_limit(self, handler, mock_http, store):
        """Custom limit query parameter is used."""
        store.list_approvals = MagicMock(return_value=([], 0))
        result = handler.handle(
            "/api/v1/openclaw/approvals", {"limit": "10"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 10

    def test_list_approvals_custom_offset(self, handler, mock_http, store):
        """Custom offset query parameter is used."""
        store.list_approvals = MagicMock(return_value=([], 0))
        result = handler.handle(
            "/api/v1/openclaw/approvals", {"offset": "25"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["offset"] == 25

    def test_list_approvals_passes_tenant_id(self, handler, mock_http, store, mock_user):
        """Tenant ID from user is passed to store."""
        store.list_approvals = MagicMock(return_value=([], 0))
        handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        store.list_approvals.assert_called_once_with(
            tenant_id=mock_user.org_id,
            limit=50,
            offset=0,
        )

    def test_list_approvals_value_error_returns_500(self, handler, mock_http, store):
        """ValueError from store returns 500."""
        store.list_approvals = MagicMock(side_effect=ValueError("bad"))
        result = handler._handle_list_approvals({}, mock_http())
        assert _status(result) == 500

    def test_list_approvals_type_error_returns_500(self, handler, mock_http, store):
        """TypeError from store returns 500."""
        store.list_approvals = MagicMock(side_effect=TypeError("bad"))
        result = handler._handle_list_approvals({}, mock_http())
        assert _status(result) == 500

    def test_list_approvals_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.list_approvals = MagicMock(side_effect=OSError("io"))
        result = handler._handle_list_approvals({}, mock_http())
        assert _status(result) == 500

    def test_list_approvals_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.list_approvals = MagicMock(side_effect=KeyError("k"))
        result = handler._handle_list_approvals({}, mock_http())
        assert _status(result) == 500


# ============================================================================
# Approve Action (POST)
# ============================================================================


class TestApproveAction:
    """Tests for _handle_approve_action (POST /approvals/:id/approve)."""

    def test_approve_action_success(self, handler, mock_http, store):
        """Successfully approve a pending action."""
        store.approve_action = MagicMock(return_value=True)
        body = {"reason": "Looks good"}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-123/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True
        assert resp["approval_id"] == "app-123"

    def test_approve_action_without_reason(self, handler, mock_http, store):
        """Approving without reason defaults to empty string."""
        store.approve_action = MagicMock(return_value=True)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-456/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True

    def test_approve_action_store_returns_false(self, handler, mock_http, store):
        """When store returns False, success is False."""
        store.approve_action = MagicMock(return_value=False)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-789/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is False

    def test_approve_action_store_returns_non_bool(self, handler, mock_http, store):
        """When store returns non-bool, success defaults to True."""
        store.approve_action = MagicMock(return_value="ok")
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-abc/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True

    def test_approve_action_store_without_method(self, handler, mock_http, store):
        """Without approve_action method, success defaults to True."""
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-def/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_approve_action_creates_audit_entry(self, handler, mock_http, store):
        """Approving creates an audit entry."""
        store.approve_action = MagicMock(return_value=True)
        body = {"reason": "All checks passed"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-audit/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        entries, total = store.get_audit_log(action="approval.approve")
        assert total >= 1
        entry = entries[0]
        assert entry.action == "approval.approve"
        assert entry.result == "success"
        assert entry.resource_type == "approval"
        assert entry.resource_id == "app-audit"
        assert entry.details["reason"] == "All checks passed"

    def test_approve_action_security_uses_authenticated_user(
        self, handler, mock_http, store, mock_user
    ):
        """Security: approver_id MUST be the authenticated user, not the body."""
        store.approve_action = MagicMock(return_value=True)
        body = {"approver_id": "attacker-user-999", "reason": "impersonation attempt"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-sec/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        # Verify the store was called with the authenticated user, not the body value
        store.approve_action.assert_called_once_with(
            approval_id="app-sec",
            approver_id=mock_user.user_id,
            reason="impersonation attempt",
        )

    def test_approve_action_audit_records_authenticated_approver(
        self, handler, mock_http, store, mock_user
    ):
        """Audit entry records the authenticated user as approver."""
        store.approve_action = MagicMock(return_value=True)
        body = {"approver_id": "fake_user", "reason": "test"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-verify/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        entries, _ = store.get_audit_log(action="approval.approve")
        assert entries[0].actor_id == mock_user.user_id
        assert entries[0].details["approver_id"] == mock_user.user_id

    def test_approve_action_value_error_returns_500(self, handler, mock_http, store):
        """ValueError from store returns 500."""
        store.approve_action = MagicMock(side_effect=ValueError("bad"))
        result = handler._handle_approve_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_approve_action_type_error_returns_500(self, handler, mock_http, store):
        """TypeError from store returns 500."""
        store.approve_action = MagicMock(side_effect=TypeError("bad"))
        result = handler._handle_approve_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_approve_action_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.approve_action = MagicMock(side_effect=OSError("io"))
        result = handler._handle_approve_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_approve_action_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.approve_action = MagicMock(side_effect=KeyError("k"))
        result = handler._handle_approve_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500


# ============================================================================
# Deny Action (POST)
# ============================================================================


class TestDenyAction:
    """Tests for _handle_deny_action (POST /approvals/:id/deny)."""

    def test_deny_action_success(self, handler, mock_http, store):
        """Successfully deny a pending action."""
        store.deny_action = MagicMock(return_value=True)
        body = {"reason": "Policy violation"}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-deny/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True
        assert resp["approval_id"] == "app-deny"

    def test_deny_action_without_reason(self, handler, mock_http, store):
        """Denying without reason defaults to empty string."""
        store.deny_action = MagicMock(return_value=True)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-noreason/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_deny_action_store_returns_false(self, handler, mock_http, store):
        """When store returns False, success is False."""
        store.deny_action = MagicMock(return_value=False)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-false/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        assert _body(result)["success"] is False

    def test_deny_action_store_returns_non_bool(self, handler, mock_http, store):
        """When store returns non-bool, success defaults to True."""
        store.deny_action = MagicMock(return_value="denied")
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-str/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_deny_action_store_without_method(self, handler, mock_http, store):
        """Without deny_action method, success defaults to True."""
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/app-no-method/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_deny_action_creates_audit_entry(self, handler, mock_http, store):
        """Denying creates an audit entry."""
        store.deny_action = MagicMock(return_value=True)
        body = {"reason": "Does not meet criteria"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-audit-deny/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        entries, total = store.get_audit_log(action="approval.deny")
        assert total >= 1
        entry = entries[0]
        assert entry.action == "approval.deny"
        assert entry.result == "success"
        assert entry.resource_type == "approval"
        assert entry.resource_id == "app-audit-deny"
        assert entry.details["reason"] == "Does not meet criteria"

    def test_deny_action_security_uses_authenticated_user(
        self, handler, mock_http, store, mock_user
    ):
        """Security: approver_id MUST be the authenticated user, not the body."""
        store.deny_action = MagicMock(return_value=True)
        body = {"approver_id": "attacker-user-999", "reason": "impersonation"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-sec-deny/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        store.deny_action.assert_called_once_with(
            approval_id="app-sec-deny",
            approver_id=mock_user.user_id,
            reason="impersonation",
        )

    def test_deny_action_audit_records_authenticated_approver(
        self, handler, mock_http, store, mock_user
    ):
        """Audit entry records the authenticated user."""
        store.deny_action = MagicMock(return_value=True)
        body = {"approver_id": "fake_user", "reason": "test"}
        handler.handle_post(
            "/api/v1/openclaw/approvals/app-verify-deny/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        entries, _ = store.get_audit_log(action="approval.deny")
        assert entries[0].actor_id == mock_user.user_id
        assert entries[0].details["approver_id"] == mock_user.user_id

    def test_deny_action_value_error_returns_500(self, handler, mock_http, store):
        """ValueError from store returns 500."""
        store.deny_action = MagicMock(side_effect=ValueError("bad"))
        result = handler._handle_deny_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_deny_action_type_error_returns_500(self, handler, mock_http, store):
        """TypeError from store returns 500."""
        store.deny_action = MagicMock(side_effect=TypeError("bad"))
        result = handler._handle_deny_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_deny_action_os_error_returns_500(self, handler, mock_http, store):
        """OSError from store returns 500."""
        store.deny_action = MagicMock(side_effect=OSError("io"))
        result = handler._handle_deny_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500

    def test_deny_action_key_error_returns_500(self, handler, mock_http, store):
        """KeyError from store returns 500."""
        store.deny_action = MagicMock(side_effect=KeyError("k"))
        result = handler._handle_deny_action("app-err", {}, mock_http(method="POST"))
        assert _status(result) == 500


# ============================================================================
# Health (GET)
# ============================================================================


class TestHealth:
    """Tests for _handle_health (GET /health)."""

    def test_health_healthy(self, handler, mock_http, store):
        """Health endpoint returns healthy for normal metrics."""
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["healthy"] is True
        assert "timestamp" in body

    def test_health_degraded_when_running_actions_high(self, handler, mock_http, store):
        """Status is degraded when running actions exceed 100."""
        # Create 101 running actions
        from aragora.server.handlers.openclaw.models import ActionStatus
        session = store.create_session(user_id="test-user-001")
        for i in range(101):
            action = store.create_action(
                session_id=session.id,
                action_type="test",
                input_data={"i": i},
            )
            store.update_action(action.id, status=ActionStatus.RUNNING)
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"
        assert body["healthy"] is True

    def test_health_unhealthy_when_pending_actions_high(self, handler, mock_http, store):
        """Status is unhealthy when pending actions exceed 500."""
        session = store.create_session(user_id="test-user-001")
        for i in range(501):
            store.create_action(
                session_id=session.id,
                action_type="test",
                input_data={"i": i},
            )
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["healthy"] is False

    def test_health_timestamp_is_iso_format(self, handler, mock_http):
        """Timestamp is a valid ISO format string."""
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        body = _body(result)
        ts = datetime.fromisoformat(body["timestamp"])
        assert ts.tzinfo is not None  # Must be timezone-aware

    def test_health_error_returns_503(self, handler, mock_http):
        """Store error returns 503 with error status."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = RuntimeError("store down")
            result = handler._handle_health(mock_http())
            assert _status(result) == 503
            body = _body(result)
            assert body["status"] == "error"
            assert body["healthy"] is False
            assert "timestamp" in body

    def test_health_value_error_returns_503(self, handler, mock_http):
        """ValueError from store returns 503."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = ValueError("bad")
            result = handler._handle_health(mock_http())
            assert _status(result) == 503

    def test_health_type_error_returns_503(self, handler, mock_http):
        """TypeError from store returns 503."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = TypeError("bad")
            result = handler._handle_health(mock_http())
            assert _status(result) == 503

    def test_health_os_error_returns_503(self, handler, mock_http):
        """OSError from store returns 503."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = OSError("io")
            result = handler._handle_health(mock_http())
            assert _status(result) == 503

    def test_health_key_error_returns_503(self, handler, mock_http):
        """KeyError from store returns 503."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = KeyError("k")
            result = handler._handle_health(mock_http())
            assert _status(result) == 503

    def test_health_via_legacy_path(self, handler, mock_http):
        """Legacy /api/gateway/openclaw/health works."""
        result = handler.handle("/api/gateway/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        assert _body(result)["healthy"] is True

    def test_health_does_not_expose_counts(self, handler, mock_http, store):
        """Health response does NOT expose session/action counts (security)."""
        store.create_session(user_id="test-user-001")
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        body = _body(result)
        # Only status, healthy, timestamp should be present
        assert "sessions" not in body
        assert "actions" not in body
        assert "credentials" not in body


# ============================================================================
# Metrics (GET)
# ============================================================================


class TestMetrics:
    """Tests for _handle_metrics (GET /metrics)."""

    def test_metrics_success(self, handler, mock_http, store):
        """Metrics endpoint returns store metrics with timestamp."""
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body
        assert "actions" in body
        assert "credentials" in body
        assert "timestamp" in body

    def test_metrics_includes_session_counts(self, handler, mock_http, store):
        """Metrics include session total and active counts."""
        store.create_session(user_id="test-user-001")
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        body = _body(result)
        assert body["sessions"]["total"] >= 1
        assert body["sessions"]["active"] >= 1

    def test_metrics_includes_action_counts(self, handler, mock_http, store):
        """Metrics include action counts."""
        session = store.create_session(user_id="test-user-001")
        store.create_action(session_id=session.id, action_type="test", input_data={})
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        body = _body(result)
        assert body["actions"]["total"] >= 1
        assert body["actions"]["pending"] >= 1

    def test_metrics_timestamp_is_iso(self, handler, mock_http, store):
        """Timestamp in metrics is valid ISO format."""
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        body = _body(result)
        ts = datetime.fromisoformat(body["timestamp"])
        assert ts.tzinfo is not None

    def test_metrics_error_returns_500(self, handler, mock_http):
        """Store error returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = ValueError("bad")
            result = handler._handle_metrics(mock_http())
            assert _status(result) == 500

    def test_metrics_key_error_returns_500(self, handler, mock_http):
        """KeyError from store returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = KeyError("k")
            result = handler._handle_metrics(mock_http())
            assert _status(result) == 500

    def test_metrics_via_legacy_path(self, handler, mock_http, store):
        """Legacy /api/gateway/openclaw/metrics works."""
        result = handler.handle("/api/gateway/openclaw/metrics", {}, mock_http())
        assert _status(result) == 200
        assert "sessions" in _body(result)


# ============================================================================
# Audit Log (GET)
# ============================================================================


class TestAudit:
    """Tests for _handle_audit (GET /audit)."""

    def test_audit_empty(self, handler, mock_http, store):
        """Empty audit log returns empty results."""
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["limit"] == 100
        assert body["offset"] == 0

    def test_audit_with_entries(self, handler, mock_http, store):
        """Audit endpoint returns audit entries."""
        store.add_audit_entry(
            action="test.action",
            actor_id="user-1",
            resource_type="test",
            resource_id="res-1",
            result="success",
        )
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1
        assert body["entries"][0]["action"] == "test.action"

    def test_audit_action_filter(self, handler, mock_http, store):
        """Audit entries can be filtered by action."""
        store.add_audit_entry(action="a.create", actor_id="u1", resource_type="a")
        store.add_audit_entry(action="b.delete", actor_id="u1", resource_type="b")
        result = handler.handle(
            "/api/v1/openclaw/audit", {"action": "a.create"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["entries"][0]["action"] == "a.create"

    def test_audit_actor_filter(self, handler, mock_http, store):
        """Audit entries can be filtered by actor_id."""
        store.add_audit_entry(action="test", actor_id="user-1", resource_type="r")
        store.add_audit_entry(action="test", actor_id="user-2", resource_type="r")
        result = handler.handle(
            "/api/v1/openclaw/audit", {"actor_id": "user-1"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["entries"][0]["actor_id"] == "user-1"

    def test_audit_resource_type_filter(self, handler, mock_http, store):
        """Audit entries can be filtered by resource_type."""
        store.add_audit_entry(action="t", actor_id="u", resource_type="session")
        store.add_audit_entry(action="t", actor_id="u", resource_type="credential")
        result = handler.handle(
            "/api/v1/openclaw/audit", {"resource_type": "session"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["entries"][0]["resource_type"] == "session"

    def test_audit_pagination_limit(self, handler, mock_http, store):
        """Audit respects limit parameter."""
        for i in range(5):
            store.add_audit_entry(action=f"a{i}", actor_id="u", resource_type="r")
        result = handler.handle(
            "/api/v1/openclaw/audit", {"limit": "2"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["entries"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2

    def test_audit_pagination_offset(self, handler, mock_http, store):
        """Audit respects offset parameter."""
        for i in range(5):
            store.add_audit_entry(action=f"a{i}", actor_id="u", resource_type="r")
        result = handler.handle(
            "/api/v1/openclaw/audit", {"offset": "3"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["offset"] == 3
        assert len(body["entries"]) == 2  # 5 total, offset 3 = entries 3,4

    def test_audit_entry_fields(self, handler, mock_http, store):
        """Audit entry has all expected fields."""
        store.add_audit_entry(
            action="full.test",
            actor_id="actor-1",
            resource_type="type-1",
            resource_id="res-1",
            result="success",
            details={"key": "value"},
        )
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        entry = _body(result)["entries"][0]
        expected_fields = {
            "id", "timestamp", "action", "actor_id",
            "resource_type", "resource_id", "result", "details",
        }
        assert expected_fields.issubset(set(entry.keys()))

    def test_audit_error_returns_500(self, handler, mock_http):
        """Store error returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_audit_log.side_effect = ValueError("bad")
            result = handler._handle_audit({}, mock_http())
            assert _status(result) == 500

    def test_audit_type_error_returns_500(self, handler, mock_http):
        """TypeError from store returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_audit_log.side_effect = TypeError("t")
            result = handler._handle_audit({}, mock_http())
            assert _status(result) == 500

    def test_audit_via_legacy_path(self, handler, mock_http, store):
        """Legacy /api/gateway/openclaw/audit works."""
        store.add_audit_entry(action="legacy", actor_id="u", resource_type="r")
        result = handler.handle("/api/gateway/openclaw/audit", {}, mock_http())
        assert _status(result) == 200
        assert _body(result)["total"] >= 1


# ============================================================================
# Stats (GET)
# ============================================================================


class TestStats:
    """Tests for _handle_stats (GET /stats)."""

    def test_stats_success(self, handler, mock_http, store):
        """Stats endpoint returns expected fields."""
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "active_sessions" in body
        assert "actions_allowed" in body
        assert "actions_denied" in body
        assert "pending_approvals" in body
        assert "policy_rules" in body
        assert "timestamp" in body

    def test_stats_active_sessions(self, handler, mock_http, store):
        """Stats reflects active session count."""
        store.create_session(user_id="test-user-001")
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        body = _body(result)
        assert body["active_sessions"] >= 1

    def test_stats_actions_allowed_from_completed(self, handler, mock_http, store):
        """actions_allowed maps to metrics['actions']['completed'] (may be 0 if key absent)."""
        # The in-memory store metrics don't expose a top-level "completed" key,
        # so this is 0 by default. If a custom store returns it, it works.
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        body = _body(result)
        assert body["actions_allowed"] == 0  # default path

    def test_stats_pending_from_metrics(self, handler, mock_http, store):
        """Pending approvals reflects pending actions count."""
        session = store.create_session(user_id="test-user-001")
        store.create_action(session_id=session.id, action_type="test", input_data={})
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        body = _body(result)
        assert body["pending_approvals"] >= 1

    def test_stats_actions_denied_default_zero(self, handler, mock_http, store):
        """actions_denied maps to metrics['actions']['failed'] (0 when absent)."""
        # In-memory store metrics don't expose "failed" at top level,
        # so the default .get() returns 0.
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        body = _body(result)
        assert body["actions_denied"] == 0

    def test_stats_policy_rules_default_zero(self, handler, mock_http, store):
        """Policy rules count is always 0 in current implementation."""
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _body(result)["policy_rules"] == 0

    def test_stats_timestamp_is_iso(self, handler, mock_http, store):
        """Stats timestamp is valid ISO format."""
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        ts = datetime.fromisoformat(_body(result)["timestamp"])
        assert ts.tzinfo is not None

    def test_stats_error_returns_500(self, handler, mock_http):
        """Store error returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = ValueError("bad")
            result = handler._handle_stats(mock_http())
            assert _status(result) == 500

    def test_stats_key_error_returns_500(self, handler, mock_http):
        """KeyError from store returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = KeyError("k")
            result = handler._handle_stats(mock_http())
            assert _status(result) == 500

    def test_stats_os_error_returns_500(self, handler, mock_http):
        """OSError from store returns 500."""
        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
        ) as mock_store:
            mock_store.return_value.get_metrics.side_effect = OSError("io")
            result = handler._handle_stats(mock_http())
            assert _status(result) == 500


# ============================================================================
# Mixin Base Class Tests
# ============================================================================


class TestPolicyHandlerMixinBase:
    """Tests for PolicyHandlerMixin used in isolation."""

    def test_mixin_inherits_from_openclaw_mixin_base(self):
        """PolicyHandlerMixin inherits from OpenClawMixinBase."""
        from aragora.server.handlers.openclaw._base import OpenClawMixinBase
        assert issubclass(PolicyHandlerMixin, OpenClawMixinBase)

    def test_mixin_exports(self):
        """Module exports PolicyHandlerMixin."""
        from aragora.server.handlers.openclaw.policies import __all__
        assert "PolicyHandlerMixin" in __all__


# ============================================================================
# Path Routing Tests
# ============================================================================


class TestPathRouting:
    """Tests for path normalization and routing."""

    def test_v1_shorthand_policy_rules(self, handler, mock_http, store):
        """GET /api/v1/openclaw/policy/rules is routed correctly."""
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200

    def test_v1_shorthand_approvals(self, handler, mock_http, store):
        """GET /api/v1/openclaw/approvals is routed correctly."""
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200

    def test_v1_shorthand_health(self, handler, mock_http, store):
        """GET /api/v1/openclaw/health is routed correctly."""
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200

    def test_v1_shorthand_metrics(self, handler, mock_http, store):
        """GET /api/v1/openclaw/metrics is routed correctly."""
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        assert _status(result) == 200

    def test_v1_shorthand_audit(self, handler, mock_http, store):
        """GET /api/v1/openclaw/audit is routed correctly."""
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 200

    def test_v1_shorthand_stats(self, handler, mock_http, store):
        """GET /api/v1/openclaw/stats is routed correctly."""
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _status(result) == 200

    def test_post_add_rule_via_v1(self, handler, mock_http, store):
        """POST /api/v1/openclaw/policy/rules adds a rule."""
        body = {"name": "v1_rule"}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201

    def test_post_approve_via_v1(self, handler, mock_http, store):
        """POST /api/v1/openclaw/approvals/:id/approve works."""
        store.approve_action = MagicMock(return_value=True)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/id-1/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200

    def test_post_deny_via_v1(self, handler, mock_http, store):
        """POST /api/v1/openclaw/approvals/:id/deny works."""
        store.deny_action = MagicMock(return_value=True)
        body = {}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/id-2/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200

    def test_delete_rule_via_v1(self, handler, mock_http, store):
        """DELETE /api/v1/openclaw/policy/rules/:name works."""
        store.remove_policy_rule = MagicMock(return_value=True)
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/my_rule", {}, mock_http()
        )
        assert _status(result) == 200

    def test_can_handle_v1_openclaw(self, handler):
        """can_handle returns True for /api/v1/openclaw/ paths."""
        assert handler.can_handle("/api/v1/openclaw/policy/rules") is True

    def test_can_handle_gateway_openclaw(self, handler):
        """can_handle returns True for /api/gateway/openclaw/ paths."""
        assert handler.can_handle("/api/gateway/openclaw/policy/rules") is True

    def test_can_handle_v1_gateway_openclaw(self, handler):
        """can_handle returns True for /api/v1/gateway/openclaw/ paths."""
        assert handler.can_handle("/api/v1/gateway/openclaw/health") is True

    def test_can_handle_unrelated_path(self, handler):
        """can_handle returns False for unrelated paths."""
        assert handler.can_handle("/api/v1/debates") is False
