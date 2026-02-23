"""Tests for workspace activity feed handler (WorkspaceActivityMixin).

Tests the activity feed endpoint:
- GET /api/v1/workspaces/{workspace_id}/activity - Get recent workspace activity

Covers:
- Successful event retrieval with various event types
- Pagination (limit/offset query params)
- Audit log unavailability (None, AttributeError, ImportError)
- Query errors (TypeError, ValueError, AttributeError, OSError)
- Event attribute resolution via getattr fallbacks
- The _describe_event helper for all 10 known event types + fallback
- The _parse_query helper for URL query string parsing
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult or infer from dict."""
    if isinstance(result, dict):
        return result.get("status", 200)
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult or return dict body."""
    if isinstance(result, dict):
        return result.get("body", result)
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    body = _body(result)
    return body.get("error", "")


# ---------------------------------------------------------------------------
# Mock event objects
# ---------------------------------------------------------------------------


class MockAuditEvent:
    """Mock audit event with standard attribute names."""

    def __init__(
        self,
        *,
        id: str = "evt-1",
        event_type: str = "member_joined",
        actor: str = "alice",
        target: str = "",
        description: str = "",
        timestamp: str = "2026-02-01T00:00:00Z",
        metadata: dict | None = None,
    ):
        self.id = id
        self.event_type = event_type
        self.actor = actor
        self.target = target
        self.description = description
        self.timestamp = timestamp
        self.metadata = metadata or {}


class AltAuditEvent:
    """Mock audit event using alternative attribute names (type/user_id/created_at/resource)."""

    def __init__(
        self,
        *,
        type: str = "debate_created",
        user_id: str = "bob",
        resource: str = "",
        created_at: str = "2026-02-02T12:00:00Z",
        metadata: dict | None = None,
    ):
        self.type = type
        self.user_id = user_id
        self.resource = resource
        self.created_at = created_at
        self.metadata = metadata or {}


class BareEvent:
    """Event with no standard attributes at all."""
    pass


# ---------------------------------------------------------------------------
# Ensure path_utils mock is available
# ---------------------------------------------------------------------------


def _mock_extract_path_param(path: str, segment_name: str) -> str | None:
    """Extract the value after a named segment in a URL path."""
    clean = path.split("?")[0]
    parts = clean.split("/")
    try:
        idx = parts.index(segment_name)
        return parts[idx + 1] if idx + 1 < len(parts) else None
    except ValueError:
        return None


@pytest.fixture(autouse=True)
def _patch_path_utils():
    """Ensure aragora.server.handlers.utils.path_utils is importable."""
    mod = types.ModuleType("aragora.server.handlers.utils.path_utils")
    mod.extract_path_param = _mock_extract_path_param
    sys.modules["aragora.server.handlers.utils.path_utils"] = mod
    yield
    sys.modules.pop("aragora.server.handlers.utils.path_utils", None)


# ---------------------------------------------------------------------------
# Rate limiter reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _get_raw_handler_fn():
    """Extract the innermost (unwrapped) handle_workspace_activity function."""
    from aragora.server.handlers.workspace.activity import WorkspaceActivityMixin

    fn = WorkspaceActivityMixin.handle_workspace_activity
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    # Also check closures for the actual function (handle_errors without args case)
    if hasattr(fn, "__closure__") and fn.__closure__:
        for cell in fn.__closure__:
            try:
                val = cell.cell_contents
                if callable(val) and getattr(val, "__name__", "") == "handle_workspace_activity":
                    inner = val
                    while hasattr(inner, "__wrapped__"):
                        inner = inner.__wrapped__
                    return inner
            except ValueError:
                pass
    return fn


@pytest.fixture
def raw_fn():
    """Return the unwrapped handle_workspace_activity function."""
    return _get_raw_handler_fn()


@pytest.fixture
def handler():
    """Create a WorkspaceActivityMixin instance with mocked dependencies."""
    from aragora.server.handlers.workspace.activity import WorkspaceActivityMixin

    class _TestHandler(WorkspaceActivityMixin):
        def __init__(self):
            self._mock_audit_log = MagicMock()
            self._mock_audit_log.query.return_value = []

        def _get_audit_log(self):
            return self._mock_audit_log

        def _run_async(self, coro):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, coro).result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        def read_json_body(self, handler):
            return getattr(handler, "_json_body", None)

    return _TestHandler()


@pytest.fixture
def make_request():
    """Factory for creating mock HTTP handler (request) objects."""

    def _make(
        path: str = "/api/v1/workspaces/ws-1/activity",
        method: str = "GET",
    ):
        h = MagicMock()
        h.path = path
        h.command = method
        h.headers = {"Content-Length": "0"}
        h.client_address = ("127.0.0.1", 12345)
        return h

    return _make


# ---------------------------------------------------------------------------
# Import module-level helpers for direct testing
# ---------------------------------------------------------------------------


@pytest.fixture
def describe_event():
    from aragora.server.handlers.workspace.activity import _describe_event
    return _describe_event


@pytest.fixture
def parse_query():
    from aragora.server.handlers.workspace.activity import _parse_query
    return _parse_query


# ===========================================================================
# _parse_query tests
# ===========================================================================


class TestParseQuery:
    """Tests for the _parse_query URL query string parser."""

    def test_no_query_string(self, parse_query):
        assert parse_query("/api/v1/workspaces/ws-1/activity") == {}

    def test_single_param(self, parse_query):
        result = parse_query("/path?limit=10")
        assert result == {"limit": "10"}

    def test_multiple_params(self, parse_query):
        result = parse_query("/path?limit=10&offset=20")
        assert result == {"limit": "10", "offset": "20"}

    def test_param_without_value(self, parse_query):
        """Params without = sign are silently ignored."""
        result = parse_query("/path?orphan&limit=5")
        assert result == {"limit": "5"}

    def test_empty_value(self, parse_query):
        result = parse_query("/path?key=")
        assert result == {"key": ""}

    def test_value_with_equals(self, parse_query):
        """Only splits on first = so values can contain =."""
        result = parse_query("/path?filter=a=b")
        assert result == {"filter": "a=b"}

    def test_empty_query_string(self, parse_query):
        result = parse_query("/path?")
        assert result == {}

    def test_multiple_question_marks(self, parse_query):
        """Only splits on first ?."""
        result = parse_query("/path?key=val?ue")
        assert result == {"key": "val?ue"}


# ===========================================================================
# _describe_event tests
# ===========================================================================


class TestDescribeEvent:
    """Tests for the _describe_event human-readable description generator."""

    def test_member_joined(self, describe_event):
        evt = MockAuditEvent(event_type="member_joined", actor="alice")
        assert describe_event(evt) == "alice joined the workspace"

    def test_member_removed(self, describe_event):
        evt = MockAuditEvent(event_type="member_removed", actor="bob")
        assert describe_event(evt) == "bob was removed from the workspace"

    def test_member_invited(self, describe_event):
        evt = MockAuditEvent(event_type="member_invited", actor="alice", target="charlie")
        assert describe_event(evt) == "alice invited charlie to the workspace"

    def test_role_changed(self, describe_event):
        evt = MockAuditEvent(event_type="role_changed", actor="admin")
        assert describe_event(evt) == "admin's role was changed"

    def test_debate_created(self, describe_event):
        evt = MockAuditEvent(event_type="debate_created", actor="dave")
        assert describe_event(evt) == "dave started a new debate"

    def test_debate_completed(self, describe_event):
        evt = MockAuditEvent(event_type="debate_completed")
        assert describe_event(evt) == "Debate completed"

    def test_settings_updated(self, describe_event):
        evt = MockAuditEvent(event_type="settings_updated", actor="eve")
        assert describe_event(evt) == "eve updated workspace settings"

    def test_invite_sent(self, describe_event):
        evt = MockAuditEvent(event_type="invite_sent", target="frank@example.com")
        assert describe_event(evt) == "Invite sent to frank@example.com"

    def test_invite_revoked(self, describe_event):
        evt = MockAuditEvent(event_type="invite_revoked", target="grace@example.com")
        assert describe_event(evt) == "Invite to grace@example.com was revoked"

    def test_policy_updated(self, describe_event):
        evt = MockAuditEvent(event_type="policy_updated", actor="hank")
        assert describe_event(evt) == "hank updated retention policies"

    def test_unknown_type_with_description_attr(self, describe_event):
        """Unknown event types use the event's own description attribute."""
        evt = MockAuditEvent(event_type="custom_action", actor="ivy", description="Custom thing happened")
        assert describe_event(evt) == "Custom thing happened"

    def test_unknown_type_without_description(self, describe_event):
        """Unknown event types without description use fallback format."""
        evt = MagicMock(spec=[])
        evt.event_type = "weird_thing"
        evt.actor = "joe"
        # No target, no description attributes
        result = describe_event(evt)
        assert result == "joe performed weird_thing"

    def test_alt_event_uses_type_attr(self, describe_event):
        """Events using 'type' instead of 'event_type' should still resolve."""
        evt = AltAuditEvent(type="debate_created", user_id="kim")
        assert describe_event(evt) == "kim started a new debate"

    def test_alt_event_uses_user_id_for_actor(self, describe_event):
        """Events with user_id instead of actor should still resolve."""
        evt = AltAuditEvent(type="member_joined", user_id="leo")
        assert describe_event(evt) == "leo joined the workspace"

    def test_alt_event_uses_resource_for_target(self, describe_event):
        """Events with resource instead of target should still resolve."""
        evt = AltAuditEvent(type="invite_sent", resource="mike@example.com")
        assert describe_event(evt) == "Invite sent to mike@example.com"

    def test_bare_event_defaults(self, describe_event):
        """Events with no recognized attributes use defaults."""
        evt = BareEvent()
        result = describe_event(evt)
        # actor defaults to "Someone", event_type defaults to ""
        assert "Someone" in result


# ===========================================================================
# handle_workspace_activity tests
# ===========================================================================


class TestHandleWorkspaceActivity:
    """Tests for the main handler method."""

    def test_returns_200_empty_events_no_audit_log(self, handler, raw_fn, make_request):
        """When audit log is None, returns empty events."""
        handler._get_audit_log = MagicMock(side_effect=AttributeError("no audit"))
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_returns_200_empty_events_import_error(self, handler, raw_fn, make_request):
        """When _get_audit_log raises ImportError, returns empty events."""
        handler._get_audit_log = MagicMock(side_effect=ImportError("no module"))
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_returns_200_empty_events_no_results(self, handler, raw_fn, make_request):
        """When audit log returns empty list, returns empty events."""
        handler._mock_audit_log.query.return_value = []
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_single_event_returned(self, handler, raw_fn, make_request):
        evt = MockAuditEvent(id="evt-1", event_type="member_joined", actor="alice")
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["events"]) == 1
        event = body["events"][0]
        assert event["id"] == "evt-1"
        assert event["type"] == "member_joined"
        assert event["actor"] == "alice"
        assert event["description"] == "alice joined the workspace"

    def test_multiple_events_returned(self, handler, raw_fn, make_request):
        events = [
            MockAuditEvent(id=f"evt-{i}", event_type="debate_created", actor=f"user-{i}")
            for i in range(5)
        ]
        handler._mock_audit_log.query.return_value = events
        req = make_request()
        result = raw_fn(handler, req)
        body = _body(result)
        assert len(body["events"]) == 5

    def test_workspace_id_extracted_from_path(self, handler, raw_fn, make_request):
        """The workspace_id should be extracted and passed to audit_log.query."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/my-workspace-42/activity")
        raw_fn(handler, req)
        handler._mock_audit_log.query.assert_called_once_with(
            workspace_id="my-workspace-42",
            limit=50,
            offset=0,
        )

    def test_default_limit_is_50(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request()
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 50

    def test_default_offset_is_0(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request()
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["offset"] == 0

    def test_custom_limit(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=25")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 25

    def test_custom_offset(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?offset=100")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["offset"] == 100

    def test_limit_capped_at_200(self, handler, raw_fn, make_request):
        """Limit should be capped to 200 even if a larger value is requested."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=999")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 200

    def test_limit_exactly_200(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=200")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 200

    def test_limit_just_below_200(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=199")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 199

    def test_combined_limit_and_offset(self, handler, raw_fn, make_request):
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=10&offset=30")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 30

    def test_event_id_fallback_to_hash(self, handler, raw_fn, make_request):
        """When event has no id, it falls back to hash of str(evt)."""
        evt = MagicMock(spec=[])
        evt.event_type = "member_joined"
        evt.actor = "alice"
        evt.timestamp = "2026-01-01"
        evt.metadata = {}
        # No .id attribute
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        # Should be the string representation of hash(str(evt))
        assert event["id"] == str(hash(str(evt)))

    def test_event_type_fallback_to_type(self, handler, raw_fn, make_request):
        """When event has no event_type but has type, use type."""
        evt = AltAuditEvent(type="debate_created", user_id="bob")
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["type"] == "debate_created"

    def test_event_type_fallback_to_unknown(self, handler, raw_fn, make_request):
        """When event has neither event_type nor type, use 'unknown'."""
        evt = BareEvent()
        evt.timestamp = "2026-01-01"
        evt.metadata = {}
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["type"] == "unknown"

    def test_actor_fallback_to_user_id(self, handler, raw_fn, make_request):
        """When event has no actor but has user_id, use user_id."""
        evt = AltAuditEvent(type="member_joined", user_id="charlie")
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["actor"] == "charlie"

    def test_actor_fallback_to_system(self, handler, raw_fn, make_request):
        """When event has neither actor nor user_id, default to 'system'."""
        evt = BareEvent()
        evt.timestamp = "2026-01-01"
        evt.metadata = {}
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["actor"] == "system"

    def test_timestamp_fallback_to_created_at(self, handler, raw_fn, make_request):
        """When event has no timestamp but has created_at, use created_at."""
        evt = AltAuditEvent(type="debate_created", created_at="2026-03-15T10:00:00Z")
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["timestamp"] == "2026-03-15T10:00:00Z"

    def test_timestamp_fallback_to_empty(self, handler, raw_fn, make_request):
        """When event has neither timestamp nor created_at, timestamp is empty string."""
        evt = BareEvent()
        evt.metadata = {}
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["timestamp"] == ""

    def test_metadata_preserved(self, handler, raw_fn, make_request):
        """Event metadata dict should be passed through."""
        evt = MockAuditEvent(metadata={"key": "value", "nested": {"a": 1}})
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["metadata"] == {"key": "value", "nested": {"a": 1}}

    def test_metadata_default_empty_dict(self, handler, raw_fn, make_request):
        """When event has no metadata, it defaults to empty dict."""
        evt = BareEvent()
        evt.timestamp = "2026-01-01"
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["metadata"] == {}

    def test_query_type_error_returns_empty(self, handler, raw_fn, make_request):
        """TypeError during query returns empty events."""
        handler._mock_audit_log.query.side_effect = TypeError("bad args")
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_query_value_error_returns_empty(self, handler, raw_fn, make_request):
        """ValueError during query returns empty events."""
        handler._mock_audit_log.query.side_effect = ValueError("invalid")
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_query_attribute_error_returns_empty(self, handler, raw_fn, make_request):
        """AttributeError during query returns empty events."""
        handler._mock_audit_log.query.side_effect = AttributeError("missing")
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_query_os_error_returns_empty(self, handler, raw_fn, make_request):
        """OSError during query returns empty events."""
        handler._mock_audit_log.query.side_effect = OSError("disk error")
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_audit_log_none_returns_empty(self, handler, raw_fn, make_request):
        """When _get_audit_log returns None, events are empty."""
        handler._get_audit_log = MagicMock(return_value=None)
        req = make_request()
        result = raw_fn(handler, req)
        assert _status(result) == 200
        assert _body(result)["events"] == []

    def test_different_workspace_ids(self, handler, raw_fn, make_request):
        """Different workspace IDs in path are correctly extracted."""
        handler._mock_audit_log.query.return_value = []
        for ws_id in ["ws-alpha", "workspace-123", "test"]:
            req = make_request(path=f"/api/v1/workspaces/{ws_id}/activity")
            raw_fn(handler, req)
            call_kwargs = handler._mock_audit_log.query.call_args[1]
            assert call_kwargs["workspace_id"] == ws_id

    def test_event_description_uses_describe_event(self, handler, raw_fn, make_request):
        """The description field should come from _describe_event."""
        evt = MockAuditEvent(event_type="settings_updated", actor="admin")
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["description"] == "admin updated workspace settings"

    def test_zero_limit(self, handler, raw_fn, make_request):
        """limit=0 is valid and should be passed through."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=0")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 0

    def test_limit_one(self, handler, raw_fn, make_request):
        """limit=1 is valid."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?limit=1")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["limit"] == 1

    def test_response_structure(self, handler, raw_fn, make_request):
        """Response should have status=200 and body with events list."""
        handler._mock_audit_log.query.return_value = []
        req = make_request()
        result = raw_fn(handler, req)
        assert result["status"] == 200
        assert "events" in result["body"]
        assert isinstance(result["body"]["events"], list)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for the activity handler."""

    def test_event_with_all_standard_fields(self, handler, raw_fn, make_request):
        """Fully populated standard event is serialized correctly."""
        evt = MockAuditEvent(
            id="evt-full",
            event_type="policy_updated",
            actor="admin-user",
            target="policy-1",
            timestamp="2026-02-20T15:30:00Z",
            metadata={"old_value": "7d", "new_value": "30d"},
        )
        handler._mock_audit_log.query.return_value = [evt]
        req = make_request()
        result = raw_fn(handler, req)
        event = _body(result)["events"][0]
        assert event["id"] == "evt-full"
        assert event["type"] == "policy_updated"
        assert event["actor"] == "admin-user"
        assert event["description"] == "admin-user updated retention policies"
        assert event["timestamp"] == "2026-02-20T15:30:00Z"
        assert event["metadata"] == {"old_value": "7d", "new_value": "30d"}

    def test_mixed_event_types(self, handler, raw_fn, make_request):
        """Multiple events of different types in a single response."""
        events = [
            MockAuditEvent(id="e1", event_type="member_joined", actor="a"),
            MockAuditEvent(id="e2", event_type="debate_created", actor="b"),
            MockAuditEvent(id="e3", event_type="settings_updated", actor="c"),
        ]
        handler._mock_audit_log.query.return_value = events
        req = make_request()
        result = raw_fn(handler, req)
        body_events = _body(result)["events"]
        assert len(body_events) == 3
        assert body_events[0]["type"] == "member_joined"
        assert body_events[1]["type"] == "debate_created"
        assert body_events[2]["type"] == "settings_updated"

    def test_event_iteration_error_midway(self, handler, raw_fn, make_request):
        """If iteration raises an error midway, the exception is caught."""
        class BrokenIterator:
            def __init__(self):
                self._count = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self._count == 0:
                    self._count += 1
                    return MockAuditEvent(id="ok-1")
                raise ValueError("iteration failed")

        handler._mock_audit_log.query.return_value = BrokenIterator()
        req = make_request()
        # The for loop inside the try block will catch ValueError
        result = raw_fn(handler, req)
        assert _status(result) == 200
        # The first event was appended before the error; the ValueError
        # is caught by the outer except block, but events already built
        # may or may not be returned depending on implementation.
        # The handler catches (TypeError, ValueError, AttributeError, OSError),
        # so the entire events list is discarded (empty) because the except
        # block runs and events stays as whatever was built before
        # Actually: events list was built in the try block, and the except
        # catches so we get whatever events were accumulated before the error
        # The events list is declared before the if block, so it keeps its value
        # but the for loop is inside the try, so when ValueError fires,
        # execution jumps to except, and whatever was appended stays.
        # However, events is initialized as [] before the if block, so
        # events accumulated before the error remain.
        # Actually re-reading: events is declared at line 83 as [],
        # then inside the try block events are appended. When ValueError fires,
        # the except catches it and does logger.debug. Then we fall through
        # to the return with whatever events we accumulated.
        body_events = _body(result)["events"]
        # The first event was successfully appended before the error
        assert len(body_events) == 1
        assert body_events[0]["id"] == "ok-1"

    def test_path_with_trailing_slash(self, handler, raw_fn, make_request):
        """Path with trailing slash still works."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity/")
        raw_fn(handler, req)
        # The extract_path_param mock should still find "ws-1"
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["workspace_id"] == "ws-1"

    def test_large_offset(self, handler, raw_fn, make_request):
        """Large offset values are passed through."""
        handler._mock_audit_log.query.return_value = []
        req = make_request(path="/api/v1/workspaces/ws-1/activity?offset=99999")
        raw_fn(handler, req)
        call_kwargs = handler._mock_audit_log.query.call_args[1]
        assert call_kwargs["offset"] == 99999
