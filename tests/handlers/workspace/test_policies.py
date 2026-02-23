"""Tests for workspace retention policy handler (WorkspacePoliciesMixin).

Tests the retention policy management endpoints:
- GET    /api/v1/retention/policies                    - List retention policies
- POST   /api/v1/retention/policies                    - Create retention policy
- GET    /api/v1/retention/policies/{policy_id}        - Get a retention policy
- PUT    /api/v1/retention/policies/{policy_id}        - Update a retention policy
- DELETE /api/v1/retention/policies/{policy_id}        - Delete a retention policy
- POST   /api/v1/retention/policies/{policy_id}/execute - Execute a retention policy
- GET    /api/v1/retention/expiring                    - Get items expiring soon
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult."""
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    body = _body(result)
    return body.get("error", "")


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


class MockRetentionAction(str, Enum):
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"


class MockAuditAction(str, Enum):
    MODIFY_POLICY = "modify_policy"
    EXECUTE_RETENTION = "execute_retention"


class MockAuditOutcome(str, Enum):
    SUCCESS = "success"


def _make_mock_policy(
    *,
    policy_id: str = "pol-001",
    name: str = "Default Policy",
    description: str = "Test policy",
    retention_days: int = 90,
    action: MockRetentionAction = MockRetentionAction.DELETE,
    enabled: bool = True,
    applies_to: list[str] | None = None,
    workspace_ids: list[str] | None = None,
    grace_period_days: int = 7,
    notify_before_days: int = 3,
    exclude_sensitivity_levels: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    created_at: datetime | None = None,
    last_run: datetime | None = None,
) -> MagicMock:
    """Create a mock retention policy object."""
    policy = MagicMock()
    policy.id = policy_id
    policy.name = name
    policy.description = description
    policy.retention_days = retention_days
    policy.action = action
    policy.enabled = enabled
    policy.applies_to = applies_to or ["documents", "findings", "sessions"]
    policy.workspace_ids = workspace_ids
    policy.grace_period_days = grace_period_days
    policy.notify_before_days = notify_before_days
    policy.exclude_sensitivity_levels = exclude_sensitivity_levels or []
    policy.exclude_tags = exclude_tags or []
    policy.created_at = created_at or datetime.now(timezone.utc)
    policy.last_run = last_run
    return policy


def _make_mock_report(
    *,
    items_deleted: int = 5,
    items_evaluated: int = 100,
) -> MagicMock:
    """Create a mock execution report."""
    report = MagicMock()
    report.items_deleted = items_deleted
    report.items_evaluated = items_evaluated
    report.to_dict.return_value = {
        "items_deleted": items_deleted,
        "items_evaluated": items_evaluated,
        "policy_id": "pol-001",
    }
    return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def mock_workspace_module():
    """Patch the workspace_module imported via _mod().

    Returns a MagicMock that provides all the symbols the policies mixin
    accesses through ``_mod()``.
    """
    m = MagicMock()

    # Auth context returned by extract_user_from_request
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "test-user-001"
    m.extract_user_from_request.return_value = auth_ctx

    # Permission constants
    m.PERM_RETENTION_READ = "retention:read"
    m.PERM_RETENTION_WRITE = "retention:write"
    m.PERM_RETENTION_DELETE = "retention:delete"
    m.PERM_RETENTION_EXECUTE = "retention:execute"

    # RetentionAction enum -- use the real one for correct value parsing
    m.RetentionAction = MockRetentionAction

    # Audit types
    m.AuditAction = MagicMock()
    m.AuditAction.MODIFY_POLICY = "modify_policy"
    m.AuditAction.EXECUTE_RETENTION = "execute_retention"
    m.AuditOutcome = MagicMock()
    m.AuditOutcome.SUCCESS = "success"
    m.Actor = MagicMock()
    m.Resource = MagicMock()

    # Cache infrastructure
    cache = MagicMock()
    cache.get.return_value = None  # Cache miss by default
    m._retention_policy_cache = cache
    m._invalidate_retention_cache = MagicMock()

    # Response helpers -- delegate to real implementations
    from aragora.server.handlers.base import json_response, error_response

    m.json_response = json_response
    m.error_response = error_response

    with patch(
        "aragora.server.handlers.workspace.policies._mod",
        return_value=m,
    ):
        yield m


@pytest.fixture
def handler(mock_workspace_module):
    """Create a WorkspacePoliciesMixin instance with mocked dependencies."""
    from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

    class _TestHandler(WorkspacePoliciesMixin):
        """Concrete handler combining mixin with mock infrastructure."""

        def __init__(self):
            self._mock_retention_manager = MagicMock()
            self._mock_user_store = MagicMock()
            self._mock_audit_log = MagicMock()
            self._mock_audit_log.log = AsyncMock()

        def _get_user_store(self):
            return self._mock_user_store

        def _get_retention_manager(self):
            return self._mock_retention_manager

        def _get_audit_log(self):
            return self._mock_audit_log

        def _run_async(self, coro):
            """Run coroutine synchronously for tests."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, coro).result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        def _check_rbac_permission(self, handler, perm, auth_ctx):
            return None  # Always allow in tests

        def read_json_body(self, handler):
            return getattr(handler, "_json_body", None)

    return _TestHandler()


@pytest.fixture
def make_handler_request():
    """Factory for creating mock HTTP handler objects."""

    def _make(
        path: str = "/api/v1/retention/policies",
        method: str = "GET",
        body: dict[str, Any] | None = None,
        query: str = "",
    ):
        h = MagicMock()
        full_path = f"{path}?{query}" if query else path
        h.path = full_path
        h.command = method
        h.headers = {"Content-Length": "0"}
        if body is not None:
            h._json_body = body
        else:
            h._json_body = None
        return h

    return _make


# ---------------------------------------------------------------------------
# GET /api/v1/retention/policies - List Retention Policies
# ---------------------------------------------------------------------------


class TestListPolicies:
    """Test the _handle_list_policies endpoint."""

    def test_list_policies_success(self, handler, make_handler_request, mock_workspace_module):
        policy = _make_mock_policy()
        handler._mock_retention_manager.list_policies.return_value = [policy]

        req = make_handler_request()
        result = handler._handle_list_policies(req, {"workspace_id": "ws-1"})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert len(body["policies"]) == 1
        assert body["policies"][0]["id"] == "pol-001"
        assert body["policies"][0]["name"] == "Default Policy"
        assert body["policies"][0]["retention_days"] == 90
        assert body["policies"][0]["action"] == "delete"
        assert body["policies"][0]["enabled"] is True

    def test_list_policies_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["policies"] == []

    def test_list_policies_multiple(self, handler, make_handler_request, mock_workspace_module):
        policies = [
            _make_mock_policy(policy_id="pol-001", name="Policy A"),
            _make_mock_policy(policy_id="pol-002", name="Policy B"),
            _make_mock_policy(policy_id="pol-003", name="Policy C"),
        ]
        handler._mock_retention_manager.list_policies.return_value = policies

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert len(body["policies"]) == 3
        ids = [p["id"] for p in body["policies"]]
        assert ids == ["pol-001", "pol-002", "pol-003"]

    def test_list_policies_with_workspace_filter(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        result = handler._handle_list_policies(req, {"workspace_id": "ws-42"})
        assert _status(result) == 200
        handler._mock_retention_manager.list_policies.assert_called_once_with(workspace_id="ws-42")

    def test_list_policies_no_workspace_filter(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 200
        handler._mock_retention_manager.list_policies.assert_called_once_with(workspace_id=None)

    def test_list_policies_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        """When cache has data, should return cached result without calling manager."""
        cached = {"policies": [{"id": "cached-pol"}], "total": 1}
        mock_workspace_module._retention_policy_cache.get.return_value = cached

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["policies"][0]["id"] == "cached-pol"
        handler._mock_retention_manager.list_policies.assert_not_called()

    def test_list_policies_cache_miss_stores_result(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """On cache miss, result should be stored in cache."""
        mock_workspace_module._retention_policy_cache.get.return_value = None
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        handler._handle_list_policies(req, {})
        mock_workspace_module._retention_policy_cache.set.assert_called_once()

    def test_list_policies_policy_with_last_run(
        self, handler, make_handler_request, mock_workspace_module
    ):
        last_run = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        policy = _make_mock_policy(last_run=last_run)
        handler._mock_retention_manager.list_policies.return_value = [policy]

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        body = _body(result)
        assert body["policies"][0]["last_run"] == last_run.isoformat()

    def test_list_policies_policy_without_last_run(
        self, handler, make_handler_request, mock_workspace_module
    ):
        policy = _make_mock_policy(last_run=None)
        handler._mock_retention_manager.list_policies.return_value = [policy]

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        body = _body(result)
        assert body["policies"][0]["last_run"] is None

    def test_list_policies_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 401

    def test_list_policies_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        assert _status(result) == 403

    def test_list_policies_applies_to_field(
        self, handler, make_handler_request, mock_workspace_module
    ):
        policy = _make_mock_policy(applies_to=["documents", "sessions"])
        handler._mock_retention_manager.list_policies.return_value = [policy]

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        body = _body(result)
        assert body["policies"][0]["applies_to"] == ["documents", "sessions"]

    def test_list_policies_description_field(
        self, handler, make_handler_request, mock_workspace_module
    ):
        policy = _make_mock_policy(description="Keep for compliance")
        handler._mock_retention_manager.list_policies.return_value = [policy]

        req = make_handler_request()
        result = handler._handle_list_policies(req, {})
        body = _body(result)
        assert body["policies"][0]["description"] == "Keep for compliance"


# ---------------------------------------------------------------------------
# POST /api/v1/retention/policies - Create Retention Policy
# ---------------------------------------------------------------------------


class TestCreatePolicy:
    """Test the _handle_create_policy endpoint."""

    def test_create_policy_success(self, handler, make_handler_request, mock_workspace_module):
        created_policy = _make_mock_policy(policy_id="new-pol", name="New Policy")
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(
            method="POST",
            body={"name": "New Policy", "retention_days": 60, "action": "delete"},
        )
        result = handler._handle_create_policy(req)
        assert _status(result) == 201
        body = _body(result)
        assert body["message"] == "Policy created successfully"
        assert body["policy"]["id"] == "new-pol"
        assert body["policy"]["name"] == "New Policy"

    def test_create_policy_minimal_body(self, handler, make_handler_request, mock_workspace_module):
        """Only name is required; retention_days, action default."""
        created_policy = _make_mock_policy(
            policy_id="pol-min",
            name="Min Policy",
            retention_days=90,
            action=MockRetentionAction.DELETE,
        )
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "Min Policy"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 201

        # Verify defaults were passed to manager
        call_kwargs = handler._mock_retention_manager.create_policy.call_args.kwargs
        assert call_kwargs["retention_days"] == 90
        assert call_kwargs["action"] == MockRetentionAction.DELETE

    def test_create_policy_all_fields(self, handler, make_handler_request, mock_workspace_module):
        created_policy = _make_mock_policy(policy_id="pol-full", name="Full Policy")
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(
            method="POST",
            body={
                "name": "Full Policy",
                "retention_days": 365,
                "action": "archive",
                "workspace_ids": ["ws-1", "ws-2"],
                "description": "Archive old items",
                "applies_to": ["documents"],
            },
        )
        result = handler._handle_create_policy(req)
        assert _status(result) == 201

        call_kwargs = handler._mock_retention_manager.create_policy.call_args.kwargs
        assert call_kwargs["name"] == "Full Policy"
        assert call_kwargs["retention_days"] == 365
        assert call_kwargs["action"] == MockRetentionAction.ARCHIVE
        assert call_kwargs["workspace_ids"] == ["ws-1", "ws-2"]
        assert call_kwargs["description"] == "Archive old items"
        assert call_kwargs["applies_to"] == ["documents"]

    def test_create_policy_missing_name(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="POST", body={"retention_days": 60})
        result = handler._handle_create_policy(req)
        assert _status(result) == 400
        assert "name" in _error(result).lower()

    def test_create_policy_empty_name(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="POST", body={"name": ""})
        result = handler._handle_create_policy(req)
        assert _status(result) == 400
        assert "name" in _error(result).lower()

    def test_create_policy_null_body(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="POST", body=None)
        req._json_body = None
        result = handler._handle_create_policy(req)
        assert _status(result) == 400
        assert "JSON" in _error(result)

    def test_create_policy_invalid_action(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(
            method="POST",
            body={"name": "Test", "action": "explode"},
        )
        result = handler._handle_create_policy(req)
        assert _status(result) == 400
        assert "Invalid action" in _error(result)
        assert "explode" in _error(result)

    def test_create_policy_action_delete(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy(action=MockRetentionAction.DELETE)
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P", "action": "delete"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 201
        assert _body(result)["policy"]["action"] == "delete"

    def test_create_policy_action_archive(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy(action=MockRetentionAction.ARCHIVE)
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P", "action": "archive"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 201
        assert _body(result)["policy"]["action"] == "archive"

    def test_create_policy_action_anonymize(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy(action=MockRetentionAction.ANONYMIZE)
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P", "action": "anonymize"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 201
        assert _body(result)["policy"]["action"] == "anonymize"

    def test_create_policy_action_notify(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy(action=MockRetentionAction.NOTIFY)
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P", "action": "notify"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 201
        assert _body(result)["policy"]["action"] == "notify"

    def test_create_policy_invalidates_cache(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)
        mock_workspace_module._invalidate_retention_cache.assert_called_once()

    def test_create_policy_audit_log(self, handler, make_handler_request, mock_workspace_module):
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)
        handler._mock_audit_log.log.assert_called_once()

    def test_create_policy_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="POST", body={"name": "P"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 401

    def test_create_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="POST", body={"name": "P"})
        result = handler._handle_create_policy(req)
        assert _status(result) == 403

    def test_create_policy_default_applies_to(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)

        call_kwargs = handler._mock_retention_manager.create_policy.call_args.kwargs
        assert call_kwargs["applies_to"] == ["documents", "findings", "sessions"]

    def test_create_policy_default_description_empty(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)

        call_kwargs = handler._mock_retention_manager.create_policy.call_args.kwargs
        assert call_kwargs["description"] == ""


# ---------------------------------------------------------------------------
# GET /api/v1/retention/policies/{policy_id} - Get Retention Policy
# ---------------------------------------------------------------------------


class TestGetPolicy:
    """Test the _handle_get_policy endpoint."""

    def test_get_policy_success(self, handler, make_handler_request, mock_workspace_module):
        policy = _make_mock_policy(
            policy_id="pol-42",
            name="Detailed Policy",
            workspace_ids=["ws-1"],
            grace_period_days=14,
            notify_before_days=5,
            exclude_sensitivity_levels=["top_secret"],
            exclude_tags=["keep-forever"],
        )
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-42")
        assert _status(result) == 200
        body = _body(result)
        p = body["policy"]
        assert p["id"] == "pol-42"
        assert p["name"] == "Detailed Policy"
        assert p["workspace_ids"] == ["ws-1"]
        assert p["grace_period_days"] == 14
        assert p["notify_before_days"] == 5
        assert p["exclude_sensitivity_levels"] == ["top_secret"]
        assert p["exclude_tags"] == ["keep-forever"]
        assert "created_at" in p

    def test_get_policy_not_found(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.get_policy.return_value = None

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-nonexistent")
        assert _status(result) == 404
        assert "not found" in _error(result).lower()

    def test_get_policy_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        cached = {"policy": {"id": "cached-pol", "name": "Cached"}}
        mock_workspace_module._retention_policy_cache.get.return_value = cached

        req = make_handler_request()
        result = handler._handle_get_policy(req, "cached-pol")
        assert _status(result) == 200
        body = _body(result)
        assert body["policy"]["id"] == "cached-pol"
        handler._mock_retention_manager.get_policy.assert_not_called()

    def test_get_policy_cache_miss_stores_result(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module._retention_policy_cache.get.return_value = None
        policy = _make_mock_policy()
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        handler._handle_get_policy(req, "pol-001")
        mock_workspace_module._retention_policy_cache.set.assert_called_once()
        # Verify cache key format
        call_args = mock_workspace_module._retention_policy_cache.set.call_args
        assert call_args[0][0] == "retention:pol-001"

    def test_get_policy_with_last_run(self, handler, make_handler_request, mock_workspace_module):
        last_run = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
        policy = _make_mock_policy(last_run=last_run)
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-001")
        body = _body(result)
        assert body["policy"]["last_run"] == last_run.isoformat()

    def test_get_policy_without_last_run(
        self, handler, make_handler_request, mock_workspace_module
    ):
        policy = _make_mock_policy(last_run=None)
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-001")
        body = _body(result)
        assert body["policy"]["last_run"] is None

    def test_get_policy_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-001")
        assert _status(result) == 401

    def test_get_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-001")
        assert _status(result) == 403

    def test_get_policy_detailed_fields(self, handler, make_handler_request, mock_workspace_module):
        """Verify all detail fields are returned."""
        policy = _make_mock_policy(
            enabled=False,
            description="Compliance archive",
            retention_days=365,
            action=MockRetentionAction.ARCHIVE,
        )
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        result = handler._handle_get_policy(req, "pol-001")
        body = _body(result)
        p = body["policy"]
        assert p["enabled"] is False
        assert p["description"] == "Compliance archive"
        assert p["retention_days"] == 365
        assert p["action"] == "archive"


# ---------------------------------------------------------------------------
# PUT /api/v1/retention/policies/{policy_id} - Update Retention Policy
# ---------------------------------------------------------------------------


class TestUpdatePolicy:
    """Test the _handle_update_policy endpoint."""

    def test_update_policy_success(self, handler, make_handler_request, mock_workspace_module):
        updated = _make_mock_policy(policy_id="pol-001", name="Updated", retention_days=180)
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(
            method="PUT",
            body={"name": "Updated", "retention_days": 180},
        )
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Policy updated successfully"
        assert body["policy"]["id"] == "pol-001"
        assert body["policy"]["name"] == "Updated"
        assert body["policy"]["retention_days"] == 180

    def test_update_policy_with_action(self, handler, make_handler_request, mock_workspace_module):
        updated = _make_mock_policy(action=MockRetentionAction.ARCHIVE)
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(
            method="PUT",
            body={"action": "archive"},
        )
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 200

        call_kwargs = handler._mock_retention_manager.update_policy.call_args.kwargs
        assert call_kwargs["action"] == MockRetentionAction.ARCHIVE

    def test_update_policy_invalid_action(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(
            method="PUT",
            body={"action": "nuke"},
        )
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 400
        assert "Invalid action" in _error(result)

    def test_update_policy_not_found(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.update_policy.side_effect = ValueError("not found")

        req = make_handler_request(
            method="PUT",
            body={"name": "X"},
        )
        result = handler._handle_update_policy(req, "pol-missing")
        assert _status(result) == 404

    def test_update_policy_null_body(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="PUT", body=None)
        req._json_body = None
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 400
        assert "JSON" in _error(result)

    def test_update_policy_invalidates_cache(
        self, handler, make_handler_request, mock_workspace_module
    ):
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(method="PUT", body={"name": "X"})
        handler._handle_update_policy(req, "pol-001")
        mock_workspace_module._invalidate_retention_cache.assert_called_once_with("pol-001")

    def test_update_policy_audit_log(self, handler, make_handler_request, mock_workspace_module):
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(method="PUT", body={"name": "X", "retention_days": 30})
        handler._handle_update_policy(req, "pol-001")
        handler._mock_audit_log.log.assert_called_once()

    def test_update_policy_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="PUT", body={"name": "X"})
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 401

    def test_update_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="PUT", body={"name": "X"})
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 403

    def test_update_policy_body_without_action_passes_through(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """When body has no action key, body passes through without enum conversion."""
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(
            method="PUT",
            body={"retention_days": 60, "enabled": False},
        )
        result = handler._handle_update_policy(req, "pol-001")
        assert _status(result) == 200

        call_kwargs = handler._mock_retention_manager.update_policy.call_args.kwargs
        assert call_kwargs["retention_days"] == 60
        assert call_kwargs["enabled"] is False
        assert "action" not in call_kwargs

    def test_update_policy_passes_policy_id(
        self, handler, make_handler_request, mock_workspace_module
    ):
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(method="PUT", body={"name": "X"})
        handler._handle_update_policy(req, "pol-specific")
        call_args = handler._mock_retention_manager.update_policy.call_args
        assert call_args[0][0] == "pol-specific"


# ---------------------------------------------------------------------------
# DELETE /api/v1/retention/policies/{policy_id} - Delete Retention Policy
# ---------------------------------------------------------------------------


class TestDeletePolicy:
    """Test the _handle_delete_policy endpoint."""

    def test_delete_policy_success(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_policy(req, "pol-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Policy deleted successfully"

    def test_delete_policy_calls_manager(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-xyz")
        handler._mock_retention_manager.delete_policy.assert_called_once_with("pol-xyz")

    def test_delete_policy_invalidates_cache(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-001")
        mock_workspace_module._invalidate_retention_cache.assert_called_once_with("pol-001")

    def test_delete_policy_audit_log(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-001")
        handler._mock_audit_log.log.assert_called_once()

    def test_delete_policy_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_policy(req, "pol-001")
        assert _status(result) == 401

    def test_delete_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_policy(req, "pol-001")
        assert _status(result) == 403

    def test_delete_policy_rbac_checks_delete_permission(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """Verify that delete uses PERM_RETENTION_DELETE, not WRITE."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac

        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-001")
        assert "retention:delete" in captured_perms


# ---------------------------------------------------------------------------
# POST /api/v1/retention/policies/{policy_id}/execute - Execute Policy
# ---------------------------------------------------------------------------


class TestExecutePolicy:
    """Test the _handle_execute_policy endpoint."""

    def test_execute_policy_success(self, handler, make_handler_request, mock_workspace_module):
        report = _make_mock_report(items_deleted=10, items_evaluated=200)
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["dry_run"] is False
        assert body["report"]["items_deleted"] == 10
        assert body["report"]["items_evaluated"] == 200

    def test_execute_policy_dry_run_true(
        self, handler, make_handler_request, mock_workspace_module
    ):
        report = _make_mock_report()
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {"dry_run": "true"})
        assert _status(result) == 200
        body = _body(result)
        assert body["dry_run"] is True

        # Verify dry_run was passed to manager
        handler._mock_retention_manager.execute_policy.assert_called_once_with(
            "pol-001", dry_run=True
        )

    def test_execute_policy_dry_run_false_default(
        self, handler, make_handler_request, mock_workspace_module
    ):
        report = _make_mock_report()
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        handler._handle_execute_policy(req, "pol-001", {})

        handler._mock_retention_manager.execute_policy.assert_called_once_with(
            "pol-001", dry_run=False
        )

    def test_execute_policy_dry_run_case_insensitive(
        self, handler, make_handler_request, mock_workspace_module
    ):
        report = _make_mock_report()
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {"dry_run": "True"})
        body = _body(result)
        assert body["dry_run"] is True

    def test_execute_policy_not_found(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.execute_policy = AsyncMock(
            side_effect=ValueError("not found")
        )

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-missing", {})
        assert _status(result) == 404

    def test_execute_policy_audit_log(self, handler, make_handler_request, mock_workspace_module):
        report = _make_mock_report(items_deleted=3, items_evaluated=50)
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        handler._handle_execute_policy(req, "pol-001", {})
        handler._mock_audit_log.log.assert_called_once()

    def test_execute_policy_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {})
        assert _status(result) == 401

    def test_execute_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {})
        assert _status(result) == 403

    def test_execute_policy_rbac_checks_execute_permission(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """Verify that execute uses PERM_RETENTION_EXECUTE."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        report = _make_mock_report()
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        handler._handle_execute_policy(req, "pol-001", {})
        assert "retention:execute" in captured_perms

    def test_execute_policy_dry_run_non_true_is_false(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """Any value other than 'true' for dry_run should be treated as false."""
        report = _make_mock_report()
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        result = handler._handle_execute_policy(req, "pol-001", {"dry_run": "yes"})
        body = _body(result)
        assert body["dry_run"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/retention/expiring - Get Expiring Items
# ---------------------------------------------------------------------------


class TestExpiringItems:
    """Test the _handle_expiring_items endpoint."""

    def test_expiring_items_success(self, handler, make_handler_request, mock_workspace_module):
        expiring_data = [
            {"id": "item-1", "expires_at": "2026-03-01T00:00:00Z"},
            {"id": "item-2", "expires_at": "2026-03-05T00:00:00Z"},
        ]
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=expiring_data)

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert body["days_ahead"] == 14  # default
        assert len(body["expiring"]) == 2

    def test_expiring_items_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=[])

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["expiring"] == []

    def test_expiring_items_custom_days(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=[])

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {"days": "30"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days_ahead"] == 30

        handler._mock_retention_manager.check_expiring_soon.assert_called_once_with(
            workspace_id=None, days=30
        )

    def test_expiring_items_with_workspace_filter(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=[])

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {"workspace_id": "ws-99"})
        assert _status(result) == 200

        handler._mock_retention_manager.check_expiring_soon.assert_called_once_with(
            workspace_id="ws-99", days=14
        )

    def test_expiring_items_with_both_params(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=[])

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {"workspace_id": "ws-1", "days": "7"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days_ahead"] == 7

        handler._mock_retention_manager.check_expiring_soon.assert_called_once_with(
            workspace_id="ws-1", days=7
        )

    def test_expiring_items_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {})
        assert _status(result) == 401

    def test_expiring_items_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {})
        assert _status(result) == 403

    def test_expiring_items_default_days_is_14(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.check_expiring_soon = AsyncMock(return_value=[])

        req = make_handler_request()
        result = handler._handle_expiring_items(req, {})
        body = _body(result)
        assert body["days_ahead"] == 14


# ---------------------------------------------------------------------------
# Cross-cutting concerns
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        from aragora.server.handlers.workspace import policies

        assert "WorkspacePoliciesMixin" in policies.__all__

    def test_all_count(self):
        from aragora.server.handlers.workspace import policies

        assert len(policies.__all__) == 1


class TestAuditDetails:
    """Test that audit log entries contain correct operation details."""

    def test_create_audit_contains_operation_create(
        self, handler, make_handler_request, mock_workspace_module
    ):
        created_policy = _make_mock_policy(name="Audit Test")
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "Audit Test"})
        handler._handle_create_policy(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["operation"] == "create"
        assert call_kwargs["details"]["name"] == "Audit Test"

    def test_update_audit_contains_changed_keys(
        self, handler, make_handler_request, mock_workspace_module
    ):
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(
            method="PUT",
            body={"name": "New Name", "retention_days": 30},
        )
        handler._handle_update_policy(req, "pol-001")

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["operation"] == "update"
        assert set(call_kwargs["details"]["changes"]) == {"name", "retention_days"}

    def test_delete_audit_contains_operation_delete(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-001")

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["operation"] == "delete"

    def test_execute_audit_contains_dry_run_and_counts(
        self, handler, make_handler_request, mock_workspace_module
    ):
        report = _make_mock_report(items_deleted=7, items_evaluated=42)
        handler._mock_retention_manager.execute_policy = AsyncMock(return_value=report)

        req = make_handler_request(method="POST")
        handler._handle_execute_policy(req, "pol-001", {"dry_run": "true"})

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["dry_run"] is True
        assert call_kwargs["details"]["items_deleted"] == 7
        assert call_kwargs["details"]["items_evaluated"] == 42

    def test_audit_actor_uses_user_id(self, handler, make_handler_request, mock_workspace_module):
        """Audit log actor should use the authenticated user's ID."""
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        # Actor is created via mock_workspace_module.Actor(id=auth_ctx.user_id, type="user")
        mock_workspace_module.Actor.assert_called_with(id="test-user-001", type="user")

    def test_audit_resource_uses_policy_id(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """Audit log resource should use the policy ID."""
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-target")

        mock_workspace_module.Resource.assert_called_with(id="pol-target", type="retention_policy")


class TestCacheKeyFormats:
    """Test that cache keys are formed correctly."""

    def test_list_cache_key_with_workspace(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        handler._handle_list_policies(req, {"workspace_id": "ws-42"})

        get_call = mock_workspace_module._retention_policy_cache.get.call_args
        assert get_call[0][0] == "retention:list:ws-42"

    def test_list_cache_key_without_workspace(
        self, handler, make_handler_request, mock_workspace_module
    ):
        handler._mock_retention_manager.list_policies.return_value = []

        req = make_handler_request()
        handler._handle_list_policies(req, {})

        get_call = mock_workspace_module._retention_policy_cache.get.call_args
        assert get_call[0][0] == "retention:list:all"

    def test_get_policy_cache_key(self, handler, make_handler_request, mock_workspace_module):
        policy = _make_mock_policy()
        handler._mock_retention_manager.get_policy.return_value = policy

        req = make_handler_request()
        handler._handle_get_policy(req, "pol-abc")

        get_call = mock_workspace_module._retention_policy_cache.get.call_args
        assert get_call[0][0] == "retention:pol-abc"


class TestCacheInvalidation:
    """Test cache invalidation behavior on write operations."""

    def test_create_invalidates_all(self, handler, make_handler_request, mock_workspace_module):
        created_policy = _make_mock_policy()
        handler._mock_retention_manager.create_policy.return_value = created_policy

        req = make_handler_request(method="POST", body={"name": "P"})
        handler._handle_create_policy(req)
        mock_workspace_module._invalidate_retention_cache.assert_called_once_with()

    def test_update_invalidates_specific_policy(
        self, handler, make_handler_request, mock_workspace_module
    ):
        updated = _make_mock_policy()
        handler._mock_retention_manager.update_policy.return_value = updated

        req = make_handler_request(method="PUT", body={"name": "X"})
        handler._handle_update_policy(req, "pol-42")
        mock_workspace_module._invalidate_retention_cache.assert_called_once_with("pol-42")

    def test_delete_invalidates_specific_policy(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_policy(req, "pol-42")
        mock_workspace_module._invalidate_retention_cache.assert_called_once_with("pol-42")
