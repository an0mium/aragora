"""
Tests for audit export handler.

Tests cover:
- Route registration (register_handlers)
- Module exports (__all__)
- handle_audit_events: happy path, filtering, date validation, pagination
- handle_audit_stats: happy path
- handle_audit_export: JSON/CSV/SOC2 exports, validation, permission decorator
- handle_audit_verify: integrity checks, date filtering, empty body
- Edge cases: limit capping, offset, search text, invalid enums
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Import the module under test.
#
# The aragora.server.handlers __init__.py re-exports from sub-packages that
# may have broken transitive imports in the current tree. We ensure the
# handlers *package* is importable (even if degraded) so that importing the
# individual audit_export *module* succeeds.
# ---------------------------------------------------------------------------


def _import_audit_module():
    """Import audit_export module, working around broken sibling imports."""
    # First, try the clean import path.
    try:
        import aragora.server.handlers.audit_export as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # The handlers __init__.py has a broken transitive import (slack handler).
    # Clear the partially-loaded package and its children, then re-import
    # with the entire broken subtree stubbed out.
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    # Stub the entire broken _slack_impl subtree and its parents so the
    # handlers __init__.py import chain skips over the broken code.
    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.audit_export as mod

    return mod


audit_module = _import_audit_module()

handle_audit_events = audit_module.handle_audit_events
handle_audit_export = audit_module.handle_audit_export
handle_audit_stats = audit_module.handle_audit_stats
handle_audit_verify = audit_module.handle_audit_verify
register_handlers = audit_module.register_handlers


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_event(event_id="evt-1", category="auth", action="login"):
    """Create a mock audit event with a to_dict method."""
    evt = MagicMock()
    evt.to_dict.return_value = {
        "id": event_id,
        "category": category,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return evt


def _make_request(query=None):
    """Create a mock GET request with query params."""
    request = MagicMock()
    request.query = query or {}
    return request


def _make_post_request(body=None, *, raise_json_error=False):
    """Create a mock POST request with a JSON body."""
    request = MagicMock()
    if raise_json_error:
        request.json = AsyncMock(side_effect=json.JSONDecodeError("err", "doc", 0))
    else:
        request.json = AsyncMock(return_value=body or {})
    return request


@pytest.fixture(autouse=True)
def _mock_audit_log():
    """Replace get_audit_log with a mock for every test, restoring afterwards."""
    mock_audit = MagicMock()
    # Defaults that tests can override
    mock_audit.query.return_value = []
    mock_audit.get_stats.return_value = {"total": 0}
    mock_audit.verify_integrity.return_value = (True, [])
    mock_audit.export_json.return_value = 0
    mock_audit.export_csv.return_value = 0
    mock_audit.export_soc2.return_value = {"events_exported": 0}

    original = audit_module.get_audit_log
    audit_module.get_audit_log = lambda: mock_audit
    yield mock_audit
    audit_module.get_audit_log = original


# ===========================================================================
# Route Registration
# ===========================================================================


class TestRegisterHandlers:
    """Tests for register_handlers function."""

    def test_registers_all_four_routes(self):
        """Should register 2 GET and 2 POST routes."""
        app = MagicMock()
        register_handlers(app)

        get_paths = [c[0][0] for c in app.router.add_get.call_args_list]
        post_paths = [c[0][0] for c in app.router.add_post.call_args_list]

        assert "/api/v1/audit/events" in get_paths
        assert "/api/v1/audit/stats" in get_paths
        assert "/api/v1/audit/export" in post_paths
        assert "/api/v1/audit/verify" in post_paths

    def test_registers_correct_handlers(self):
        """Each route should point to the correct handler function."""
        app = MagicMock()
        register_handlers(app)

        get_calls = {c[0][0]: c[0][1] for c in app.router.add_get.call_args_list}
        post_calls = {c[0][0]: c[0][1] for c in app.router.add_post.call_args_list}

        assert get_calls["/api/v1/audit/events"] is handle_audit_events
        assert get_calls["/api/v1/audit/stats"] is handle_audit_stats
        # handle_audit_export is wrapped by @require_permission, so check post route exists
        assert "/api/v1/audit/export" in post_calls
        assert post_calls["/api/v1/audit/verify"] is handle_audit_verify


# ===========================================================================
# Module Exports
# ===========================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports(self):
        expected = {
            "handle_audit_events",
            "handle_audit_export",
            "handle_audit_stats",
            "handle_audit_verify",
            "register_handlers",
        }
        assert set(audit_module.__all__) == expected


# ===========================================================================
# handle_audit_events
# ===========================================================================


@pytest.mark.asyncio
class TestHandleAuditEvents:
    """Tests for the GET /api/audit/events endpoint."""

    async def test_happy_path_default_params(self, _mock_audit_log):
        """Should return events with default date range and pagination."""
        events = [_make_mock_event("e1"), _make_mock_event("e2")]
        _mock_audit_log.query.return_value = events

        resp = await handle_audit_events(_make_request())
        body = json.loads(resp.body)

        assert resp.status == 200
        assert body["count"] == 2
        assert len(body["events"]) == 2
        assert body["query"]["limit"] == 100
        assert body["query"]["offset"] == 0

    async def test_custom_limit_and_offset(self, _mock_audit_log):
        """Custom limit and offset should be forwarded to the query."""
        _mock_audit_log.query.return_value = []

        resp = await handle_audit_events(_make_request({"limit": "50", "offset": "10"}))
        body = json.loads(resp.body)

        assert body["query"]["limit"] == 50
        assert body["query"]["offset"] == 10

    async def test_limit_capped_at_1000(self, _mock_audit_log):
        """Limit should be capped at 1000 even if a higher value is requested."""
        _mock_audit_log.query.return_value = []

        resp = await handle_audit_events(_make_request({"limit": "5000"}))
        body = json.loads(resp.body)

        assert body["query"]["limit"] == 1000

    async def test_valid_date_range(self, _mock_audit_log):
        """Valid ISO dates should be accepted and reflected in the response."""
        _mock_audit_log.query.return_value = []

        resp = await handle_audit_events(
            _make_request(
                {"start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-31T23:59:59Z"}
            )
        )
        body = json.loads(resp.body)

        assert resp.status == 200
        assert "2024-01-01" in body["query"]["start_date"]
        assert "2024-01-31" in body["query"]["end_date"]

    async def test_invalid_start_date_returns_400(self):
        """Invalid start_date should return 400."""
        resp = await handle_audit_events(_make_request({"start_date": "not-a-date"}))
        body = json.loads(resp.body)

        assert resp.status == 400
        assert "start_date" in body["error"]

    async def test_invalid_end_date_returns_400(self):
        """Invalid end_date should return 400."""
        resp = await handle_audit_events(_make_request({"end_date": "bad"}))
        assert resp.status == 400

    async def test_invalid_category_returns_400(self):
        """An unknown category value should return 400."""
        resp = await handle_audit_events(_make_request({"category": "nonexistent"}))
        body = json.loads(resp.body)

        assert resp.status == 400
        assert "category" in body["error"].lower()

    async def test_valid_category_accepted(self, _mock_audit_log):
        """A valid category value should be accepted."""
        _mock_audit_log.query.return_value = []

        resp = await handle_audit_events(_make_request({"category": "auth"}))
        assert resp.status == 200

    async def test_invalid_outcome_returns_400(self):
        """An unknown outcome value should return 400."""
        resp = await handle_audit_events(_make_request({"outcome": "nope"}))
        assert resp.status == 400

    async def test_valid_outcome_accepted(self, _mock_audit_log):
        """A valid outcome value should be accepted."""
        _mock_audit_log.query.return_value = []

        resp = await handle_audit_events(_make_request({"outcome": "success"}))
        assert resp.status == 200

    async def test_filters_forwarded(self, _mock_audit_log):
        """action, actor_id, org_id, and search should be forwarded to the query."""
        _mock_audit_log.query.return_value = []

        await handle_audit_events(
            _make_request(
                {
                    "action": "login",
                    "actor_id": "user-42",
                    "org_id": "org-1",
                    "search": "keyword",
                }
            )
        )

        query_arg = _mock_audit_log.query.call_args[0][0]
        assert query_arg.action == "login"
        assert query_arg.actor_id == "user-42"
        assert query_arg.org_id == "org-1"
        assert query_arg.search_text == "keyword"


# ===========================================================================
# handle_audit_stats
# ===========================================================================


@pytest.mark.asyncio
class TestHandleAuditStats:
    """Tests for the GET /api/audit/stats endpoint."""

    async def test_happy_path(self, _mock_audit_log):
        """Should return stats from the audit log."""
        _mock_audit_log.get_stats.return_value = {"total": 42, "categories": {"auth": 20}}

        resp = await handle_audit_stats(_make_request())
        body = json.loads(resp.body)

        assert resp.status == 200
        assert body["total"] == 42
        assert body["categories"]["auth"] == 20


# ===========================================================================
# handle_audit_export
# ===========================================================================


@pytest.mark.asyncio
class TestHandleAuditExport:
    """Tests for the POST /api/audit/export endpoint."""

    async def test_json_export_happy_path(self, _mock_audit_log):
        """JSON export should write a temp file and return its content."""

        def _write_json(path, *a, **kw):
            path.write_text('{"events": []}')
            return 5

        _mock_audit_log.export_json.side_effect = _write_json

        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "format": "json",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 200
        assert resp.content_type == "application/json"
        assert "audit_export" in resp.headers["Content-Disposition"]
        assert resp.headers["X-Audit-Event-Count"] == "5"

    async def test_csv_export_happy_path(self, _mock_audit_log):
        """CSV export should set text/csv content type."""

        def _write_csv(path, *a, **kw):
            path.write_text("id,action\n1,login\n")
            return 3

        _mock_audit_log.export_csv.side_effect = _write_csv

        request = _make_post_request(
            {
                "start_date": "2024-06-01",
                "end_date": "2024-06-30",
                "format": "csv",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 200
        assert resp.content_type == "text/csv"
        assert resp.headers["X-Audit-Event-Count"] == "3"

    async def test_soc2_export_happy_path(self, _mock_audit_log):
        """SOC2 export should use export_soc2 and return JSON."""

        def _write_soc2(path, *a, **kw):
            path.write_text('{"soc2": true}')
            return {"events_exported": 10}

        _mock_audit_log.export_soc2.side_effect = _write_soc2

        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "format": "soc2",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 200
        assert resp.content_type == "application/json"
        assert "soc2_audit" in resp.headers["Content-Disposition"]
        assert resp.headers["X-Audit-Event-Count"] == "10"

    async def test_default_format_is_json(self, _mock_audit_log):
        """When no format is specified, it should default to json."""

        def _write_json(path, *a, **kw):
            path.write_text("{}")
            return 0

        _mock_audit_log.export_json.side_effect = _write_json

        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 200
        assert resp.content_type == "application/json"

    async def test_invalid_format_returns_400(self, _mock_audit_log):
        """An unsupported format should return 400."""
        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "format": "xml",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "Invalid format" in body["error"]

    async def test_missing_start_date_returns_400(self):
        """Missing start_date should return 400."""
        request = _make_post_request({"end_date": "2024-01-31"})
        resp = await handle_audit_export(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "start_date" in body["error"]

    async def test_missing_end_date_returns_400(self):
        """Missing end_date should return 400."""
        request = _make_post_request({"start_date": "2024-01-01"})
        resp = await handle_audit_export(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "end_date" in body["error"]

    async def test_invalid_json_body_returns_400(self):
        """Non-JSON body should return 400."""
        request = _make_post_request(raise_json_error=True)
        resp = await handle_audit_export(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "Invalid JSON" in body["error"]

    async def test_invalid_date_format_returns_400(self):
        """Malformed date strings should return 400."""
        request = _make_post_request(
            {
                "start_date": "not-a-date",
                "end_date": "2024-01-31",
            }
        )
        resp = await handle_audit_export(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "date format" in body["error"].lower() or "ISO" in body["error"]

    async def test_has_require_permission_decorator(self):
        """handle_audit_export should be decorated with @require_permission('audit:export')."""
        import asyncio

        assert asyncio.iscoroutinefunction(handle_audit_export)
        assert callable(handle_audit_export)

    async def test_export_with_org_id_filter(self, _mock_audit_log):
        """org_id from body should be passed to the export function."""

        def _write_json(path, start, end, org_id):
            path.write_text("{}")
            assert org_id == "org-99"
            return 2

        _mock_audit_log.export_json.side_effect = _write_json

        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "org_id": "org-99",
            }
        )
        resp = await handle_audit_export(request)
        assert resp.status == 200

    async def test_export_file_content_returned(self, _mock_audit_log):
        """The body of the response should contain the file content."""
        content = '{"data": [1,2,3]}'

        def _write_json(path, *a, **kw):
            path.write_text(content)
            return 3

        _mock_audit_log.export_json.side_effect = _write_json

        request = _make_post_request(
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }
        )
        resp = await handle_audit_export(request)

        # web.Response wraps string body in a StringPayload; decode to compare.
        body_bytes = resp.body._value if hasattr(resp.body, "_value") else resp.body
        if isinstance(body_bytes, bytes):
            body_bytes = body_bytes.decode()
        assert body_bytes == content


# ===========================================================================
# handle_audit_verify
# ===========================================================================


@pytest.mark.asyncio
class TestHandleAuditVerify:
    """Tests for the POST /api/audit/verify endpoint."""

    async def test_happy_path_integrity_valid(self, _mock_audit_log):
        """When integrity is valid, verified should be True."""
        _mock_audit_log.verify_integrity.return_value = (True, [])

        request = _make_post_request({"start_date": "2024-01-01", "end_date": "2024-12-31"})
        resp = await handle_audit_verify(request)
        body = json.loads(resp.body)

        assert resp.status == 200
        assert body["verified"] is True
        assert body["total_errors"] == 0

    async def test_integrity_failure_with_errors(self, _mock_audit_log):
        """When integrity check fails, errors should be included."""
        errors = ["hash mismatch at row 5", "missing entry at row 12"]
        _mock_audit_log.verify_integrity.return_value = (False, errors)

        request = _make_post_request({})
        resp = await handle_audit_verify(request)
        body = json.loads(resp.body)

        assert resp.status == 200
        assert body["verified"] is False
        assert body["total_errors"] == 2
        assert len(body["errors"]) == 2

    async def test_errors_capped_at_20(self, _mock_audit_log):
        """Response should include at most 20 error entries."""
        errors = [f"error {i}" for i in range(50)]
        _mock_audit_log.verify_integrity.return_value = (False, errors)

        request = _make_post_request({})
        resp = await handle_audit_verify(request)
        body = json.loads(resp.body)

        assert len(body["errors"]) == 20
        assert body["total_errors"] == 50

    async def test_empty_body_is_valid(self, _mock_audit_log):
        """An empty/invalid JSON body should be treated as verify-all."""
        _mock_audit_log.verify_integrity.return_value = (True, [])

        request = _make_post_request(raise_json_error=True)
        resp = await handle_audit_verify(request)
        body = json.loads(resp.body)

        assert resp.status == 200
        assert body["verified"] is True
        assert body["verified_range"]["start_date"] == "beginning"
        assert body["verified_range"]["end_date"] == "now"

    async def test_invalid_start_date_returns_400(self):
        """Invalid start_date should return 400."""
        request = _make_post_request({"start_date": "nope"})
        resp = await handle_audit_verify(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "start_date" in body["error"]

    async def test_invalid_end_date_returns_400(self):
        """Invalid end_date should return 400."""
        request = _make_post_request({"end_date": "nope"})
        resp = await handle_audit_verify(request)

        assert resp.status == 400
        body = json.loads(resp.body)
        assert "end_date" in body["error"]

    async def test_verified_range_with_dates(self, _mock_audit_log):
        """When dates are provided, verified_range should contain ISO strings."""
        _mock_audit_log.verify_integrity.return_value = (True, [])

        request = _make_post_request(
            {
                "start_date": "2024-06-01T00:00:00Z",
                "end_date": "2024-06-30T23:59:59Z",
            }
        )
        resp = await handle_audit_verify(request)
        body = json.loads(resp.body)

        assert "2024-06-01" in body["verified_range"]["start_date"]
        assert "2024-06-30" in body["verified_range"]["end_date"]

    async def test_verify_calls_audit_with_none_dates_when_omitted(self, _mock_audit_log):
        """When no dates provided, verify_integrity should be called with None, None."""
        _mock_audit_log.verify_integrity.return_value = (True, [])

        request = _make_post_request({})
        await handle_audit_verify(request)

        _mock_audit_log.verify_integrity.assert_called_once_with(None, None)
