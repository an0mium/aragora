"""Tests for aragora.events.handler_events."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import aragora.events.handler_events as mod
from aragora.events.handler_events import (
    APPROVED,
    COMPLETED,
    CREATED,
    DELETED,
    FAILED,
    QUERIED,
    REJECTED,
    STARTED,
    UPDATED,
    emit_handler_event,
)


@pytest.fixture(autouse=True)
def _reset_dispatcher_cache():
    """Reset the cached dispatcher availability flag between tests."""
    mod._dispatcher_available = None
    yield
    mod._dispatcher_available = None


# ------------------------------------------------------------------
# Core dispatch behaviour
# ------------------------------------------------------------------


def test_emit_handler_event_calls_dispatch():
    """emit_handler_event should call the underlying dispatch_event function."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("debates", "created", {"debate_id": "d1"})
    mock_dispatch.assert_called_once()
    event_type, data = mock_dispatch.call_args[0]
    assert event_type == "debates.created"


def test_emit_handler_event_includes_metadata():
    """Emitted data must contain handler, action, and a numeric timestamp."""
    mock_dispatch = MagicMock()
    before = time.time()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("knowledge", "updated")
    after = time.time()

    data = mock_dispatch.call_args[0][1]
    assert data["handler"] == "knowledge"
    assert data["action"] == "updated"
    assert before <= data["timestamp"] <= after


def test_emit_handler_event_includes_payload():
    """Payload dict entries should be merged into the event data."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("agents", "started", {"agent": "claude", "model": "opus"})

    data = mock_dispatch.call_args[0][1]
    assert data["agent"] == "claude"
    assert data["model"] == "opus"


def test_emit_handler_event_includes_user_id():
    """user_id keyword should appear in event data when supplied."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("debates", "created", user_id="u-42")

    data = mock_dispatch.call_args[0][1]
    assert data["user_id"] == "u-42"


def test_emit_handler_event_includes_resource_id():
    """resource_id keyword should appear in event data when supplied."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("knowledge", "deleted", resource_id="r-99")

    data = mock_dispatch.call_args[0][1]
    assert data["resource_id"] == "r-99"


def test_emit_handler_event_includes_trace_id():
    """When tracing middleware provides a trace_id it should be included."""
    mock_dispatch = MagicMock()
    with (
        patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch),
        patch(
            "aragora.events.handler_events.get_trace_id",
            create=True,
            return_value="trace-abc",
        ) as _mock_trace,
    ):
        # We need to patch the import inside the function
        fake_tracing = MagicMock()
        fake_tracing.get_trace_id.return_value = "trace-abc"
        with patch.dict("sys.modules", {"aragora.server.middleware.tracing": fake_tracing}):
            emit_handler_event("debates", "completed")

    data = mock_dispatch.call_args[0][1]
    assert data["trace_id"] == "trace-abc"


def test_emit_handler_event_no_trace_id_when_unavailable():
    """When tracing middleware is not importable, trace_id should be absent."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        # Remove the tracing module so the import fails
        import sys

        saved = sys.modules.pop("aragora.server.middleware.tracing", None)
        sys.modules["aragora.server.middleware.tracing"] = None  # force ImportError
        try:
            emit_handler_event("debates", "completed")
        finally:
            if saved is not None:
                sys.modules["aragora.server.middleware.tracing"] = saved
            else:
                sys.modules.pop("aragora.server.middleware.tracing", None)

    data = mock_dispatch.call_args[0][1]
    assert "trace_id" not in data


# ------------------------------------------------------------------
# Resilience
# ------------------------------------------------------------------


def test_emit_handler_event_noop_when_dispatcher_unavailable():
    """When the dispatcher cannot be imported, emit should silently no-op."""
    with patch.object(mod, "_get_dispatch_fn", return_value=None):
        # Should not raise
        emit_handler_event("debates", "created", {"id": "1"})


def test_emit_handler_event_swallows_dispatch_errors():
    """Runtime errors in dispatch_event must not propagate to the caller."""
    mock_dispatch = MagicMock(side_effect=RuntimeError("boom"))
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        # Should not raise
        emit_handler_event("debates", "failed", {"reason": "timeout"})


# ------------------------------------------------------------------
# Event type formatting
# ------------------------------------------------------------------


def test_event_type_format():
    """Event type should be formatted as 'handler.action'."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("billing", "approved")

    event_type = mock_dispatch.call_args[0][0]
    assert event_type == "billing.approved"


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


def test_action_constants_exist():
    """All standard action constants should be defined and be strings."""
    constants = [CREATED, UPDATED, DELETED, COMPLETED, FAILED, STARTED, APPROVED, REJECTED, QUERIED]
    assert len(constants) == 9
    for c in constants:
        assert isinstance(c, str)
        assert len(c) > 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_payload():
    """Passing None or empty dict as payload should not add extra keys."""
    mock_dispatch = MagicMock()
    with patch.object(mod, "_get_dispatch_fn", return_value=mock_dispatch):
        emit_handler_event("agents", "queried")

    data = mock_dispatch.call_args[0][1]
    # Only the standard keys should be present
    assert set(data.keys()) <= {"handler", "action", "timestamp", "trace_id"}


def test_dispatch_fn_cached_after_first_call():
    """After a successful import, _dispatcher_available should be True and cached."""
    mock_dispatch_event = MagicMock()
    with patch("aragora.events.dispatcher.dispatch_event", mock_dispatch_event, create=True):
        with patch.dict(
            "sys.modules",
            {"aragora.events.dispatcher": MagicMock(dispatch_event=mock_dispatch_event)},
        ):
            fn = mod._get_dispatch_fn()
            assert fn is not None
            assert mod._dispatcher_available is True

            # Second call should still return a function without re-importing
            fn2 = mod._get_dispatch_fn()
            assert fn2 is not None
