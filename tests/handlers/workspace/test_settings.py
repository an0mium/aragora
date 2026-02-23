"""Tests for workspace settings handler (WorkspaceSettingsMixin).

Tests the classification and audit endpoints:
- POST /api/v1/classify                              - Classify content sensitivity
- GET  /api/v1/classify/policy/{level}               - Get policy for sensitivity level
- GET  /api/v1/audit/entries                          - Query audit log entries
- GET  /api/v1/audit/report                           - Generate compliance audit report
- GET  /api/v1/audit/verify                           - Verify audit log integrity
- GET  /api/v1/audit/actor/{actor_id}/history         - Get actor action history
- GET  /api/v1/audit/resource/{resource_id}/history   - Get resource action history
- GET  /api/v1/audit/denied                           - Get denied access attempts
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


class MockSensitivityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class MockAuditAction(str, Enum):
    CLASSIFY_DOCUMENT = "classify_document"
    GENERATE_REPORT = "generate_report"


class MockAuditOutcome(str, Enum):
    SUCCESS = "success"


def _make_classification_result(
    *,
    level: MockSensitivityLevel = MockSensitivityLevel.INTERNAL,
    confidence: float = 0.85,
) -> MagicMock:
    """Create a mock ClassificationResult."""
    result = MagicMock()
    result.level = level
    result.confidence = confidence
    result.to_dict.return_value = {
        "level": level.value,
        "confidence": confidence,
        "classified_at": datetime.now(timezone.utc).isoformat(),
        "pii_detected": False,
        "secrets_detected": False,
        "indicators_found": 0,
        "classification_method": "rule_based",
    }
    return result


def _make_audit_entry(
    *,
    entry_id: str = "entry-001",
    action: str = "read",
    outcome: str = "success",
    actor_id: str = "user-1",
    resource_id: str = "doc-1",
) -> MagicMock:
    """Create a mock AuditEntry."""
    entry = MagicMock()
    entry.id = entry_id
    entry.to_dict.return_value = {
        "id": entry_id,
        "action": action,
        "outcome": outcome,
        "actor": {"id": actor_id, "type": "user"},
        "resource": {"id": resource_id, "type": "document"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return entry


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

    Returns a MagicMock that provides all the symbols the settings mixin
    accesses through ``_mod()``.
    """
    m = MagicMock()

    # Auth context returned by extract_user_from_request
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "test-user-001"
    m.extract_user_from_request.return_value = auth_ctx

    # Permission constants
    m.PERM_CLASSIFY_READ = "classify:read"
    m.PERM_CLASSIFY_WRITE = "classify:write"
    m.PERM_AUDIT_READ = "audit:read"
    m.PERM_AUDIT_REPORT = "audit:report"
    m.PERM_AUDIT_VERIFY = "audit:verify"

    # SensitivityLevel enum
    m.SensitivityLevel = MockSensitivityLevel

    # Audit types
    m.AuditAction = MagicMock()
    m.AuditAction.CLASSIFY_DOCUMENT = "classify_document"
    m.AuditAction.GENERATE_REPORT = "generate_report"
    m.AuditOutcome = MagicMock()
    m.AuditOutcome.SUCCESS = "success"
    m.Actor = MagicMock()
    m.Resource = MagicMock()

    # Cache infrastructure
    cache = MagicMock()
    cache.get.return_value = None  # Cache miss by default
    m._audit_query_cache = cache

    # safe_query_int helper
    def _safe_query_int(query, key, default, min_val=1, max_val=100):
        try:
            raw = query.get(key)
            if raw is None:
                return default
            val = int(raw)
            return max(min_val, min(val, max_val))
        except (ValueError, TypeError):
            return default

    m.safe_query_int = _safe_query_int

    # Response helpers -- delegate to real implementations
    from aragora.server.handlers.base import json_response, error_response

    m.json_response = json_response
    m.error_response = error_response

    with patch(
        "aragora.server.handlers.workspace.settings._mod",
        return_value=m,
    ):
        yield m


@pytest.fixture
def handler(mock_workspace_module):
    """Create a WorkspaceSettingsMixin instance with mocked dependencies."""
    from aragora.server.handlers.workspace.settings import WorkspaceSettingsMixin

    class _TestHandler(WorkspaceSettingsMixin):
        """Concrete handler combining mixin with mock infrastructure."""

        def __init__(self):
            self._mock_classifier = MagicMock()
            self._mock_user_store = MagicMock()
            self._mock_audit_log = MagicMock()
            self._mock_audit_log.log = AsyncMock()
            self._mock_audit_log.query = AsyncMock(return_value=[])
            self._mock_audit_log.generate_compliance_report = AsyncMock(
                return_value={"report_id": "rpt-001", "summary": "OK"}
            )
            self._mock_audit_log.verify_integrity = AsyncMock(
                return_value=(True, [])
            )
            self._mock_audit_log.get_actor_history = AsyncMock(return_value=[])
            self._mock_audit_log.get_resource_history = AsyncMock(return_value=[])
            self._mock_audit_log.get_denied_access_attempts = AsyncMock(
                return_value=[]
            )

        def _get_user_store(self):
            return self._mock_user_store

        def _get_classifier(self):
            return self._mock_classifier

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
        path: str = "/api/v1/classify",
        method: str = "POST",
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


# ===========================================================================
# POST /api/v1/classify - Classify Content Sensitivity
# ===========================================================================


class TestClassifyContent:
    """Test the _handle_classify_content endpoint."""

    def test_classify_success(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "Some sensitive document"})
        result = handler._handle_classify_content(req)
        assert _status(result) == 200
        body = _body(result)
        assert "classification" in body
        assert body["classification"]["level"] == "internal"
        assert body["classification"]["confidence"] == 0.85

    def test_classify_with_document_id(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Data", "document_id": "doc-42"}
        )
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

        # Verify classifier was called with document_id
        call_kwargs = handler._mock_classifier.classify.call_args.kwargs
        assert call_kwargs["document_id"] == "doc-42"

    def test_classify_with_metadata(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        metadata = {"source": "email", "department": "legal"}
        req = make_handler_request(
            body={"content": "Data", "metadata": metadata}
        )
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

        call_kwargs = handler._mock_classifier.classify.call_args.kwargs
        assert call_kwargs["metadata"] == metadata

    def test_classify_missing_content(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(body={"document_id": "doc-1"})
        result = handler._handle_classify_content(req)
        assert _status(result) == 400
        assert "content" in _error(result).lower()

    def test_classify_empty_content(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(body={"content": ""})
        result = handler._handle_classify_content(req)
        assert _status(result) == 400
        assert "content" in _error(result).lower()

    def test_classify_null_body(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(body=None)
        req._json_body = None
        result = handler._handle_classify_content(req)
        assert _status(result) == 400
        assert "JSON" in _error(result)

    def test_classify_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(body={"content": "Data"})
        result = handler._handle_classify_content(req)
        assert _status(result) == 401

    def test_classify_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(body={"content": "Data"})
        result = handler._handle_classify_content(req)
        assert _status(result) == 403

    def test_classify_audit_log_with_document_id(self, handler, make_handler_request, mock_workspace_module):
        """When document_id is provided, audit log entry should be created."""
        classification = _make_classification_result(
            level=MockSensitivityLevel.CONFIDENTIAL, confidence=0.9
        )
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Data", "document_id": "doc-99"}
        )
        handler._handle_classify_content(req)
        handler._mock_audit_log.log.assert_called_once()

    def test_classify_no_audit_log_without_document_id(self, handler, make_handler_request, mock_workspace_module):
        """When no document_id is provided, audit log should NOT be written."""
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "Data"})
        handler._handle_classify_content(req)
        handler._mock_audit_log.log.assert_not_called()

    def test_classify_default_document_id(self, handler, make_handler_request, mock_workspace_module):
        """Default document_id is empty string, which is falsy -- no audit log."""
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "Data"})
        handler._handle_classify_content(req)

        call_kwargs = handler._mock_classifier.classify.call_args.kwargs
        assert call_kwargs["document_id"] == ""
        handler._mock_audit_log.log.assert_not_called()

    def test_classify_default_metadata(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "Data"})
        handler._handle_classify_content(req)

        call_kwargs = handler._mock_classifier.classify.call_args.kwargs
        assert call_kwargs["metadata"] == {}

    def test_classify_all_sensitivity_levels(self, handler, make_handler_request, mock_workspace_module):
        """Verify response works with all sensitivity levels."""
        for level in MockSensitivityLevel:
            classification = _make_classification_result(level=level)
            handler._mock_classifier.classify = AsyncMock(return_value=classification)

            req = make_handler_request(body={"content": "Data"})
            result = handler._handle_classify_content(req)
            assert _status(result) == 200
            body = _body(result)
            assert body["classification"]["level"] == level.value

    def test_classify_audit_actor_uses_user_id(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Data", "document_id": "doc-1"}
        )
        handler._handle_classify_content(req)
        mock_workspace_module.Actor.assert_called_with(id="test-user-001", type="user")

    def test_classify_audit_resource_sensitivity_level(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result(
            level=MockSensitivityLevel.RESTRICTED
        )
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Data", "document_id": "doc-sec"}
        )
        handler._handle_classify_content(req)

        # Verify Resource was created with sensitivity_level
        mock_workspace_module.Resource.assert_called_once()
        call_kwargs = mock_workspace_module.Resource.call_args.kwargs
        assert call_kwargs["sensitivity_level"] == "restricted"

    def test_classify_rbac_checks_write_permission(self, handler, make_handler_request, mock_workspace_module):
        """Verify that classify uses PERM_CLASSIFY_WRITE."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "Data"})
        handler._handle_classify_content(req)
        assert "classify:write" in captured_perms


# ===========================================================================
# GET /api/v1/classify/policy/{level} - Get Level Policy
# ===========================================================================


class TestGetLevelPolicy:
    """Test the _handle_get_level_policy endpoint."""

    def test_get_policy_success(self, handler, make_handler_request, mock_workspace_module):
        policy = {"encryption": True, "access_control": "rbac", "retention_days": 365}
        handler._mock_classifier.get_level_policy.return_value = policy

        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "confidential")
        assert _status(result) == 200
        body = _body(result)
        assert body["level"] == "confidential"
        assert body["policy"] == policy

    def test_get_policy_all_valid_levels(self, handler, make_handler_request, mock_workspace_module):
        """Each valid sensitivity level should return 200."""
        for level in MockSensitivityLevel:
            handler._mock_classifier.get_level_policy.return_value = {}

            req = make_handler_request(method="GET")
            result = handler._handle_get_level_policy(req, level.value)
            assert _status(result) == 200
            body = _body(result)
            assert body["level"] == level.value

    def test_get_policy_invalid_level(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "ultra_secret")
        assert _status(result) == 400
        assert "Invalid level" in _error(result)
        assert "ultra_secret" in _error(result)

    def test_get_policy_invalid_level_lists_valid(self, handler, make_handler_request, mock_workspace_module):
        """Error should list valid sensitivity levels."""
        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "bogus")
        error_msg = _error(result)
        assert "public" in error_msg
        assert "internal" in error_msg
        assert "confidential" in error_msg

    def test_get_policy_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "public")
        assert _status(result) == 401

    def test_get_policy_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "public")
        assert _status(result) == 403

    def test_get_policy_rbac_checks_read_permission(self, handler, make_handler_request, mock_workspace_module):
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        handler._mock_classifier.get_level_policy.return_value = {}

        req = make_handler_request(method="GET")
        handler._handle_get_level_policy(req, "public")
        assert "classify:read" in captured_perms

    def test_get_policy_case_sensitive_level(self, handler, make_handler_request, mock_workspace_module):
        """SensitivityLevel is case-sensitive; uppercase should fail."""
        req = make_handler_request(method="GET")
        result = handler._handle_get_level_policy(req, "PUBLIC")
        assert _status(result) == 400


# ===========================================================================
# GET /api/v1/audit/entries - Query Audit Log Entries
# ===========================================================================


class TestQueryAudit:
    """Test the _handle_query_audit endpoint."""

    def test_query_success_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["limit"] == 100  # default

    def test_query_with_entries(self, handler, make_handler_request, mock_workspace_module):
        entries = [
            _make_audit_entry(entry_id="e1"),
            _make_audit_entry(entry_id="e2"),
        ]
        handler._mock_audit_log.query = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["entries"]) == 2

    def test_query_with_date_filters(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-02-01T00:00:00+00:00",
        })
        assert _status(result) == 200

        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    def test_query_invalid_start_date(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"start_date": "not-a-date"})
        assert _status(result) == 400
        assert "ISO 8601" in _error(result)

    def test_query_invalid_end_date(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"end_date": "bad-date"})
        assert _status(result) == 400
        assert "ISO 8601" in _error(result)

    def test_query_with_actor_id(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"actor_id": "user-42"})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["actor_id"] == "user-42"

    def test_query_with_resource_id(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"resource_id": "res-99"})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["resource_id"] == "res-99"

    def test_query_with_workspace_id(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"workspace_id": "ws-1"})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-1"

    def test_query_with_action_filter(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"action": "read"})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["action"] is not None

    def test_query_with_outcome_filter(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"outcome": "success"})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["outcome"] is not None

    def test_query_custom_limit(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"limit": "50"})
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 50

    def test_query_limit_capped_at_1000(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"limit": "5000"})
        body = _body(result)
        assert body["limit"] == 1000

    def test_query_limit_min_1(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"limit": "0"})
        body = _body(result)
        assert body["limit"] == 1

    def test_query_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        cached = {"entries": [{"id": "cached-entry"}], "total": 1, "limit": 100}
        mock_workspace_module._audit_query_cache.get.return_value = cached

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"][0]["id"] == "cached-entry"
        handler._mock_audit_log.query.assert_not_called()

    def test_query_cache_miss_stores_result(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {})
        mock_workspace_module._audit_query_cache.set.assert_called_once()

    def test_query_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        assert _status(result) == 401

    def test_query_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        assert _status(result) == 403

    def test_query_rbac_checks_audit_read(self, handler, make_handler_request, mock_workspace_module):
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {})
        assert "audit:read" in captured_perms

    def test_query_no_filters_passes_defaults(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {})
        call_kwargs = handler._mock_audit_log.query.call_args.kwargs
        assert call_kwargs["start_date"] is None
        assert call_kwargs["end_date"] is None
        assert call_kwargs["actor_id"] is None
        assert call_kwargs["resource_id"] is None
        assert call_kwargs["workspace_id"] is None
        assert call_kwargs["action"] is None
        assert call_kwargs["outcome"] is None
        assert call_kwargs["limit"] == 100


# ===========================================================================
# GET /api/v1/audit/report - Generate Compliance Audit Report
# ===========================================================================


class TestAuditReport:
    """Test the _handle_audit_report endpoint."""

    def test_report_success(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-001", "summary": "All good"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        result = handler._handle_audit_report(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["report"]["report_id"] == "rpt-001"

    def test_report_with_date_range(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-002"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        result = handler._handle_audit_report(req, {
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
        })
        assert _status(result) == 200

        call_kwargs = handler._mock_audit_log.generate_compliance_report.call_args.kwargs
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    def test_report_with_workspace_id(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-003"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {"workspace_id": "ws-42"})
        call_kwargs = handler._mock_audit_log.generate_compliance_report.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-42"

    def test_report_with_format(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-004"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {"format": "csv"})
        call_kwargs = handler._mock_audit_log.generate_compliance_report.call_args.kwargs
        assert call_kwargs["format"] == "csv"

    def test_report_default_format_json(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-005"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {})
        call_kwargs = handler._mock_audit_log.generate_compliance_report.call_args.kwargs
        assert call_kwargs["format"] == "json"

    def test_report_invalid_date(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="GET")
        result = handler._handle_audit_report(req, {"start_date": "garbage"})
        assert _status(result) == 400
        assert "ISO 8601" in _error(result)

    def test_report_audit_log_logged(self, handler, make_handler_request, mock_workspace_module):
        """Generating a report should log to audit."""
        report = {"report_id": "rpt-006"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {})
        handler._mock_audit_log.log.assert_called_once()

    def test_report_audit_resource_uses_report_id(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-unique"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {})
        mock_workspace_module.Resource.assert_called_once()
        call_kwargs = mock_workspace_module.Resource.call_args.kwargs
        assert call_kwargs["id"] == "rpt-unique"
        assert call_kwargs["type"] == "compliance_report"

    def test_report_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_audit_report(req, {})
        assert _status(result) == 401

    def test_report_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_audit_report(req, {})
        assert _status(result) == 403

    def test_report_rbac_checks_report_permission(self, handler, make_handler_request, mock_workspace_module):
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        report = {"report_id": "rpt-007"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {})
        assert "audit:report" in captured_perms


# ===========================================================================
# GET /api/v1/audit/verify - Verify Audit Log Integrity
# ===========================================================================


class TestVerifyIntegrity:
    """Test the _handle_verify_integrity endpoint."""

    def test_verify_success_valid(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(True, [])
        )

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert body["errors"] == []
        assert body["error_count"] == 0
        assert "verified_at" in body

    def test_verify_with_errors(self, handler, make_handler_request, mock_workspace_module):
        errors = ["Checksum mismatch at entry-5", "Missing entry-6"]
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(False, errors)
        )

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert body["error_count"] == 2
        assert len(body["errors"]) == 2

    def test_verify_with_date_range(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(True, [])
        )

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
        })
        assert _status(result) == 200

        call_kwargs = handler._mock_audit_log.verify_integrity.call_args.kwargs
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    def test_verify_invalid_date(self, handler, make_handler_request, mock_workspace_module):
        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {"start_date": "nope"})
        assert _status(result) == 400
        assert "ISO 8601" in _error(result)

    def test_verify_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {})
        assert _status(result) == 401

    def test_verify_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {})
        assert _status(result) == 403

    def test_verify_rbac_checks_verify_permission(self, handler, make_handler_request, mock_workspace_module):
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(True, [])
        )

        req = make_handler_request(method="GET")
        handler._handle_verify_integrity(req, {})
        assert "audit:verify" in captured_perms

    def test_verify_verified_at_is_iso_format(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(True, [])
        )

        req = make_handler_request(method="GET")
        result = handler._handle_verify_integrity(req, {})
        body = _body(result)
        # Should be parseable as ISO datetime
        datetime.fromisoformat(body["verified_at"])

    def test_verify_no_date_filters(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.verify_integrity = AsyncMock(
            return_value=(True, [])
        )

        req = make_handler_request(method="GET")
        handler._handle_verify_integrity(req, {})
        call_kwargs = handler._mock_audit_log.verify_integrity.call_args.kwargs
        assert call_kwargs["start_date"] is None
        assert call_kwargs["end_date"] is None


# ===========================================================================
# GET /api/v1/audit/actor/{actor_id}/history - Actor History
# ===========================================================================


class TestActorHistory:
    """Test the _handle_actor_history endpoint."""

    def test_actor_history_success_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["actor_id"] == "user-42"
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["days"] == 30  # default

    def test_actor_history_with_entries(self, handler, make_handler_request, mock_workspace_module):
        entries = [
            _make_audit_entry(entry_id="e1", actor_id="user-42"),
            _make_audit_entry(entry_id="e2", actor_id="user-42"),
        ]
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["entries"]) == 2

    def test_actor_history_custom_days(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {"days": "7"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days"] == 7

        handler._mock_audit_log.get_actor_history.assert_called_once_with(
            actor_id="user-42", days=7
        )

    def test_actor_history_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        cached = {
            "actor_id": "user-42",
            "entries": [{"id": "cached"}],
            "total": 1,
            "days": 30,
        }
        mock_workspace_module._audit_query_cache.get.return_value = cached

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"][0]["id"] == "cached"
        handler._mock_audit_log.get_actor_history.assert_not_called()

    def test_actor_history_cache_miss_stores_result(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_actor_history(req, "user-42", {})
        mock_workspace_module._audit_query_cache.set.assert_called_once()

    def test_actor_history_cache_key_format(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_actor_history(req, "user-99", {"days": "14"})
        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:actor:user-99:days:14"

    def test_actor_history_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {})
        assert _status(result) == 401

    def test_actor_history_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-42", {})
        assert _status(result) == 403


# ===========================================================================
# GET /api/v1/audit/resource/{resource_id}/history - Resource History
# ===========================================================================


class TestResourceHistory:
    """Test the _handle_resource_history endpoint."""

    def test_resource_history_success_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["resource_id"] == "doc-42"
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["days"] == 30  # default

    def test_resource_history_with_entries(self, handler, make_handler_request, mock_workspace_module):
        entries = [
            _make_audit_entry(entry_id="e1", resource_id="doc-42"),
            _make_audit_entry(entry_id="e2", resource_id="doc-42"),
            _make_audit_entry(entry_id="e3", resource_id="doc-42"),
        ]
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3

    def test_resource_history_custom_days(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {"days": "60"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days"] == 60

        handler._mock_audit_log.get_resource_history.assert_called_once_with(
            resource_id="doc-42", days=60
        )

    def test_resource_history_days_clamped_max(self, handler, make_handler_request, mock_workspace_module):
        """Days should be clamped to max_val=365."""
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {"days": "9999"})
        body = _body(result)
        assert body["days"] == 365

    def test_resource_history_days_clamped_min(self, handler, make_handler_request, mock_workspace_module):
        """Days should be clamped to min_val=1."""
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {"days": "0"})
        body = _body(result)
        assert body["days"] == 1

    def test_resource_history_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        cached = {
            "resource_id": "doc-42",
            "entries": [{"id": "cached"}],
            "total": 1,
            "days": 30,
        }
        mock_workspace_module._audit_query_cache.get.return_value = cached

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"][0]["id"] == "cached"
        handler._mock_audit_log.get_resource_history.assert_not_called()

    def test_resource_history_cache_miss_stores_result(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_resource_history(req, "doc-42", {})
        mock_workspace_module._audit_query_cache.set.assert_called_once()

    def test_resource_history_cache_key_format(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_resource_history(req, "res-abc", {"days": "7"})
        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:resource:res-abc:days:7"

    def test_resource_history_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {})
        assert _status(result) == 401

    def test_resource_history_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-42", {})
        assert _status(result) == 403


# ===========================================================================
# GET /api/v1/audit/denied - Denied Access Attempts
# ===========================================================================


class TestDeniedAccess:
    """Test the _handle_denied_access endpoint."""

    def test_denied_success_empty(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["denied_attempts"] == []
        assert body["total"] == 0
        assert body["days"] == 7  # default

    def test_denied_with_entries(self, handler, make_handler_request, mock_workspace_module):
        entries = [
            _make_audit_entry(entry_id="d1", outcome="denied"),
            _make_audit_entry(entry_id="d2", outcome="denied"),
        ]
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(
            return_value=entries
        )

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["denied_attempts"]) == 2

    def test_denied_custom_days(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {"days": "14"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days"] == 14

        handler._mock_audit_log.get_denied_access_attempts.assert_called_once_with(days=14)

    def test_denied_days_clamped_max(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {"days": "999"})
        body = _body(result)
        assert body["days"] == 365

    def test_denied_days_clamped_min(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {"days": "-5"})
        body = _body(result)
        assert body["days"] == 1

    def test_denied_cache_hit(self, handler, make_handler_request, mock_workspace_module):
        cached = {"denied_attempts": [{"id": "cached-d"}], "total": 1, "days": 7}
        mock_workspace_module._audit_query_cache.get.return_value = cached

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["denied_attempts"][0]["id"] == "cached-d"
        handler._mock_audit_log.get_denied_access_attempts.assert_not_called()

    def test_denied_cache_miss_stores_result(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_denied_access(req, {})
        mock_workspace_module._audit_query_cache.set.assert_called_once()

    def test_denied_cache_key_format(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module._audit_query_cache.get.return_value = None
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_denied_access(req, {"days": "3"})
        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:denied:days:3"

    def test_denied_not_authenticated(self, handler, make_handler_request, mock_workspace_module):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        assert _status(result) == 401

    def test_denied_rbac_denied(self, handler, make_handler_request, mock_workspace_module):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        assert _status(result) == 403

    def test_denied_rbac_checks_audit_read(self, handler, make_handler_request, mock_workspace_module):
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_denied_access(req, {})
        assert "audit:read" in captured_perms


# ===========================================================================
# Cross-cutting concerns
# ===========================================================================


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        from aragora.server.handlers.workspace import settings

        assert "WorkspaceSettingsMixin" in settings.__all__

    def test_all_count(self):
        from aragora.server.handlers.workspace import settings

        assert len(settings.__all__) == 1


class TestCacheKeyFormats:
    """Test that cache keys are formed correctly for audit queries."""

    def test_query_cache_key_all_defaults(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {})

        get_call = mock_workspace_module._audit_query_cache.get.call_args
        key = get_call[0][0]
        assert key.startswith("audit:query:")
        assert "all" in key  # workspace_id is None -> "all"

    def test_query_cache_key_with_workspace(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_query_audit(req, {"workspace_id": "ws-42"})

        get_call = mock_workspace_module._audit_query_cache.get.call_args
        key = get_call[0][0]
        assert "ws-42" in key

    def test_actor_history_cache_key(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_actor_history(req, "actor-x", {})

        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:actor:actor-x:days:30"

    def test_resource_history_cache_key(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_resource_history(req, "res-y", {})

        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:resource:res-y:days:30"

    def test_denied_cache_key(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        handler._handle_denied_access(req, {})

        get_call = mock_workspace_module._audit_query_cache.get.call_args
        assert get_call[0][0] == "audit:denied:days:7"


class TestSecurityEdgeCases:
    """Test security-related edge cases."""

    def test_classify_content_with_script_tag(self, handler, make_handler_request, mock_workspace_module):
        """Ensure script tags in content are accepted (classification handles it)."""
        classification = _make_classification_result(
            level=MockSensitivityLevel.RESTRICTED
        )
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "<script>alert('xss')</script>"}
        )
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

    def test_classify_content_with_sql_injection(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "'; DROP TABLE users; --"}
        )
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

    def test_classify_very_long_content(self, handler, make_handler_request, mock_workspace_module):
        """Very long content should still be accepted."""
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(body={"content": "x" * 100_000})
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

    def test_classify_unicode_content(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result()
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Confidential data with unicode chars: \u00e9\u00e8\u00ea\u4e2d\u6587"}
        )
        result = handler._handle_classify_content(req)
        assert _status(result) == 200

    def test_query_audit_path_traversal_actor_id(self, handler, make_handler_request, mock_workspace_module):
        """Path traversal in actor_id should not cause issues."""
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "../../../etc/passwd", {})
        # The handler should still process it (validation at routing layer)
        assert _status(result) == 200

    def test_query_audit_empty_workspace_id(self, handler, make_handler_request, mock_workspace_module):
        handler._mock_audit_log.query = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {"workspace_id": ""})
        assert _status(result) == 200

    def test_denied_non_numeric_days(self, handler, make_handler_request, mock_workspace_module):
        """Non-numeric days should fallback to default via safe_query_int."""
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(return_value=[])

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {"days": "abc"})
        assert _status(result) == 200
        body = _body(result)
        assert body["days"] == 7  # default


class TestAuditLogDetails:
    """Test that audit log entries contain correct operation details."""

    def test_classify_audit_details(self, handler, make_handler_request, mock_workspace_module):
        classification = _make_classification_result(
            level=MockSensitivityLevel.CONFIDENTIAL, confidence=0.92
        )
        handler._mock_classifier.classify = AsyncMock(return_value=classification)

        req = make_handler_request(
            body={"content": "Secret", "document_id": "doc-detail"}
        )
        handler._handle_classify_content(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["level"] == "confidential"
        assert call_kwargs["details"]["confidence"] == 0.92

    def test_report_audit_details_workspace_id(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-audit"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {"workspace_id": "ws-detail"})

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["workspace_id"] == "ws-detail"

    def test_report_audit_details_no_workspace(self, handler, make_handler_request, mock_workspace_module):
        report = {"report_id": "rpt-nows"}
        handler._mock_audit_log.generate_compliance_report = AsyncMock(
            return_value=report
        )

        req = make_handler_request(method="GET")
        handler._handle_audit_report(req, {})

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["workspace_id"] is None


class TestEntryConversion:
    """Test batch conversion of entries to dicts."""

    def test_query_entries_to_dict(self, handler, make_handler_request, mock_workspace_module):
        entries = [
            _make_audit_entry(entry_id=f"e-{i}")
            for i in range(5)
        ]
        handler._mock_audit_log.query = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_query_audit(req, {})
        body = _body(result)
        assert len(body["entries"]) == 5
        for i, entry in enumerate(body["entries"]):
            assert entry["id"] == f"e-{i}"

    def test_actor_history_entries_to_dict(self, handler, make_handler_request, mock_workspace_module):
        entries = [_make_audit_entry(entry_id="ah-1"), _make_audit_entry(entry_id="ah-2")]
        handler._mock_audit_log.get_actor_history = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_actor_history(req, "user-1", {})
        body = _body(result)
        assert body["entries"][0]["id"] == "ah-1"
        assert body["entries"][1]["id"] == "ah-2"

    def test_resource_history_entries_to_dict(self, handler, make_handler_request, mock_workspace_module):
        entries = [_make_audit_entry(entry_id="rh-1")]
        handler._mock_audit_log.get_resource_history = AsyncMock(return_value=entries)

        req = make_handler_request(method="GET")
        result = handler._handle_resource_history(req, "doc-1", {})
        body = _body(result)
        assert body["entries"][0]["id"] == "rh-1"

    def test_denied_entries_to_dict(self, handler, make_handler_request, mock_workspace_module):
        entries = [_make_audit_entry(entry_id="deny-1")]
        handler._mock_audit_log.get_denied_access_attempts = AsyncMock(
            return_value=entries
        )

        req = make_handler_request(method="GET")
        result = handler._handle_denied_access(req, {})
        body = _body(result)
        assert body["denied_attempts"][0]["id"] == "deny-1"
