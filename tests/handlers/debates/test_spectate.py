"""Tests for debate spectator SSE handler.

Tests covering:
- get_active_collectors: returns the module-level registry
- push_spectator_event: fan-out to queues, queue-full handling, dead queue cleanup
- handle_spectate: returns status JSON with debate info, viewer count, SSE URL
- spectate_sse_generator: SSE frame generation, heartbeat, registration/cleanup
- _sse_frame: SSE frame formatting
- register_spectate_routes: route registration, Starlette path, fallback
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.debates.spectate import (
    _active_collectors,
    _sse_frame,
    get_active_collectors,
    handle_spectate,
    push_spectator_event,
    register_spectate_routes,
    spectate_sse_generator,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract JSON body from a HandlerResult."""
    if result is None:
        return {}
    raw = result.body
    if isinstance(raw, bytes):
        return json.loads(raw.decode())
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    if result is None:
        return 0
    return result.status_code


@pytest.fixture
def auth_context():
    """Create an admin authorization context."""
    return AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin"},
        permissions={"*"},
    )


@pytest.fixture(autouse=True)
def clear_collectors():
    """Ensure the global _active_collectors dict is clean before/after each test."""
    _active_collectors.clear()
    yield
    _active_collectors.clear()


# ===========================================================================
# get_active_collectors
# ===========================================================================


class TestGetActiveCollectors:
    """Tests for get_active_collectors()."""

    def test_returns_dict(self):
        """Returns a dict instance."""
        result = get_active_collectors()
        assert isinstance(result, dict)

    def test_returns_same_object_as_module_level(self):
        """Returns the exact same dict object as the module-level registry."""
        result = get_active_collectors()
        assert result is _active_collectors

    def test_initially_empty(self):
        """Registry is empty before any clients connect."""
        assert len(get_active_collectors()) == 0

    def test_reflects_mutations(self):
        """Mutations to the registry are visible via get_active_collectors."""
        q = asyncio.Queue()
        _active_collectors["debate-1"] = {q}
        assert "debate-1" in get_active_collectors()
        assert q in get_active_collectors()["debate-1"]


# ===========================================================================
# push_spectator_event
# ===========================================================================


class TestPushSpectatorEvent:
    """Tests for push_spectator_event()."""

    def test_no_collectors_returns_zero(self):
        """Returns 0 when no collectors exist for the debate."""
        count = push_spectator_event("nonexistent", "test_event")
        assert count == 0

    def test_push_to_single_client(self):
        """Pushes event to a single client queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        count = push_spectator_event("d1", "round_start")
        assert count == 1
        event = q.get_nowait()
        assert event["type"] == "round_start"

    def test_push_to_multiple_clients(self):
        """Fans out event to all watching clients."""
        q1: asyncio.Queue = asyncio.Queue(maxsize=10)
        q2: asyncio.Queue = asyncio.Queue(maxsize=10)
        q3: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q1, q2, q3}
        count = push_spectator_event("d1", "vote")
        assert count == 3
        for q in (q1, q2, q3):
            event = q.get_nowait()
            assert event["type"] == "vote"

    def test_event_has_timestamp(self):
        """Event includes a float timestamp."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        before = time.time()
        push_spectator_event("d1", "test")
        after = time.time()
        event = q.get_nowait()
        assert before <= event["timestamp"] <= after

    def test_event_agent_field(self):
        """Agent name is stored in the event."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "proposal", agent="claude")
        event = q.get_nowait()
        assert event["agent"] == "claude"

    def test_event_agent_empty_string_becomes_none(self):
        """Empty string agent becomes None."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "test", agent="")
        event = q.get_nowait()
        assert event["agent"] is None

    def test_event_details_field(self):
        """Details string is stored in the event."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "critique", details="Weak argument")
        event = q.get_nowait()
        assert event["details"] == "Weak argument"

    def test_event_details_empty_string_becomes_none(self):
        """Empty details becomes None."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "test", details="")
        event = q.get_nowait()
        assert event["details"] is None

    def test_event_metric_field(self):
        """Metric float is stored in the event."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "consensus", metric=0.85)
        event = q.get_nowait()
        assert event["metric"] == 0.85

    def test_event_metric_default_none(self):
        """Metric defaults to None when not provided."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "test")
        event = q.get_nowait()
        assert event["metric"] is None

    def test_event_round_number(self):
        """Round number is stored in the event."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "round_start", round_number=3)
        event = q.get_nowait()
        assert event["round"] == 3

    def test_event_round_number_default_none(self):
        """Round number defaults to None."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event("d1", "test")
        event = q.get_nowait()
        assert event["round"] is None

    def test_queue_full_drops_oldest(self):
        """When queue is full, drops oldest event and pushes new one."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        _active_collectors["d1"] = {q}
        # Fill the queue
        push_spectator_event("d1", "old_event")
        # Push again - should drop old and push new
        count = push_spectator_event("d1", "new_event")
        assert count == 1
        event = q.get_nowait()
        assert event["type"] == "new_event"

    def test_dead_queue_removed(self):
        """Queues that can't accept events are removed."""
        # Create a queue with maxsize=0 that will always be full
        # Actually, let's use a queue of size 1 and make both put_nowait calls fail
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        _active_collectors["d1"] = {q}
        # Fill the queue
        q.put_nowait({"type": "filler"})
        # Now patch get_nowait to raise QueueEmpty so the retry also fails
        original_get = q.get_nowait
        fail_count = 0

        def failing_get():
            nonlocal fail_count
            fail_count += 1
            raise asyncio.QueueEmpty()

        q.get_nowait = failing_get
        count = push_spectator_event("d1", "new_event")
        assert count == 0
        # Dead queue should be discarded
        assert q not in _active_collectors["d1"]

    def test_different_debates_isolated(self):
        """Events for one debate don't leak to another."""
        q1: asyncio.Queue = asyncio.Queue(maxsize=10)
        q2: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q1}
        _active_collectors["d2"] = {q2}
        push_spectator_event("d1", "event_for_d1")
        assert not q2.empty() is False or q2.qsize() == 0  # q2 should be empty
        assert q1.qsize() == 1
        assert q2.qsize() == 0

    def test_mixed_healthy_and_dead_queues(self):
        """Healthy queues receive events even when some are dead."""
        good_q: asyncio.Queue = asyncio.Queue(maxsize=10)
        dead_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        _active_collectors["d1"] = {good_q, dead_q}
        # Fill dead queue and make it unfixable
        dead_q.put_nowait({"type": "filler"})
        original_get = dead_q.get_nowait
        dead_q.get_nowait = lambda: (_ for _ in ()).throw(asyncio.QueueEmpty)
        count = push_spectator_event("d1", "test")
        # At least the good queue should receive the event
        assert good_q.qsize() == 1
        event = good_q.get_nowait()
        assert event["type"] == "test"

    def test_all_event_fields_present(self):
        """All expected fields are present in the pushed event."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        _active_collectors["d1"] = {q}
        push_spectator_event(
            "d1", "round_end", agent="gpt-4", details="Summary", metric=0.9, round_number=2
        )
        event = q.get_nowait()
        assert set(event.keys()) == {"type", "timestamp", "agent", "details", "metric", "round"}
        assert event["type"] == "round_end"
        assert event["agent"] == "gpt-4"
        assert event["details"] == "Summary"
        assert event["metric"] == 0.9
        assert event["round"] == 2


# ===========================================================================
# handle_spectate
# ===========================================================================


class TestHandleSpectate:
    """Tests for handle_spectate()."""

    @pytest.mark.asyncio
    async def test_returns_handler_result(self, auth_context):
        """Returns a HandlerResult."""
        result = await handle_spectate("debate-123", context=auth_context)
        assert isinstance(result, HandlerResult)

    @pytest.mark.asyncio
    async def test_status_200(self, auth_context):
        """Returns HTTP 200."""
        result = await handle_spectate("debate-123", context=auth_context)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_body_contains_debate_id(self, auth_context):
        """Response body includes the debate ID."""
        result = await handle_spectate("debate-abc", context=auth_context)
        body = _body(result)
        assert body["debate_id"] == "debate-abc"

    @pytest.mark.asyncio
    async def test_spectate_available_true(self, auth_context):
        """spectate_available is always True."""
        result = await handle_spectate("d1", context=auth_context)
        body = _body(result)
        assert body["spectate_available"] is True

    @pytest.mark.asyncio
    async def test_sse_url_includes_debate_id(self, auth_context):
        """SSE URL contains the debate ID."""
        result = await handle_spectate("debate-xyz", context=auth_context)
        body = _body(result)
        assert body["sse_url"] == "/api/v1/debates/debate-xyz/spectate"

    @pytest.mark.asyncio
    async def test_active_viewers_zero_when_none(self, auth_context):
        """active_viewers is 0 when no one is watching."""
        result = await handle_spectate("d1", context=auth_context)
        body = _body(result)
        assert body["active_viewers"] == 0

    @pytest.mark.asyncio
    async def test_active_viewers_counts_queues(self, auth_context):
        """active_viewers reflects the number of connected SSE clients."""
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        _active_collectors["d1"] = {q1, q2}
        result = await handle_spectate("d1", context=auth_context)
        body = _body(result)
        assert body["active_viewers"] == 2

    @pytest.mark.asyncio
    async def test_active_viewers_other_debate_not_counted(self, auth_context):
        """Viewers of a different debate are not counted."""
        _active_collectors["other-debate"] = {asyncio.Queue()}
        result = await handle_spectate("my-debate", context=auth_context)
        body = _body(result)
        assert body["active_viewers"] == 0


# ===========================================================================
# spectate_sse_generator
# ===========================================================================


class TestSpectateSSEGenerator:
    """Tests for spectate_sse_generator()."""

    @pytest.mark.asyncio
    async def test_first_yield_is_connected_frame(self):
        """First yielded frame is a 'connected' SSE event."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.05)
        frame = await gen.__anext__()
        assert frame.startswith("event: connected\n")
        data = json.loads(frame.split("data: ")[1].strip())
        assert data["debate_id"] == "d1"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_registers_queue_in_collectors(self):
        """Generator registers its queue in _active_collectors."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.05)
        await gen.__anext__()  # connected frame
        assert "d1" in _active_collectors
        assert len(_active_collectors["d1"]) == 1
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_unregisters_queue_on_close(self):
        """Queue is removed from collectors when generator is closed."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.05)
        await gen.__anext__()  # connected frame
        assert len(_active_collectors["d1"]) == 1
        await gen.aclose()
        # After close, the debate entry should be cleaned up entirely
        assert "d1" not in _active_collectors

    @pytest.mark.asyncio
    async def test_heartbeat_on_timeout(self):
        """Yields a heartbeat comment when no events arrive within heartbeat_interval."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.01)
        await gen.__anext__()  # connected frame
        # Wait for heartbeat (no events pushed)
        frame = await gen.__anext__()
        assert frame == ": heartbeat\n\n"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_yields_pushed_event(self):
        """Yields SSE frame when an event is pushed to the queue."""
        gen = spectate_sse_generator("d1", heartbeat_interval=5.0)
        await gen.__anext__()  # connected frame

        # Push an event
        push_spectator_event("d1", "vote", agent="claude")

        frame = await gen.__anext__()
        assert "event: vote\n" in frame
        data = json.loads(frame.split("data: ")[1].strip())
        assert data["type"] == "vote"
        assert data["agent"] == "claude"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_max_queue_size_respected(self):
        """Queue is created with the specified maxsize."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.05, max_queue_size=5)
        await gen.__anext__()  # connected frame
        # Grab the queue from collectors
        queues = list(_active_collectors["d1"])
        assert len(queues) == 1
        assert queues[0].maxsize == 5
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_multiple_generators_same_debate(self):
        """Multiple generators for the same debate each get their own queue."""
        gen1 = spectate_sse_generator("d1", heartbeat_interval=0.05)
        gen2 = spectate_sse_generator("d1", heartbeat_interval=0.05)
        await gen1.__anext__()
        await gen2.__anext__()
        assert len(_active_collectors["d1"]) == 2
        await gen1.aclose()
        # One remaining
        assert len(_active_collectors.get("d1", set())) == 1
        await gen2.aclose()
        assert "d1" not in _active_collectors

    @pytest.mark.asyncio
    async def test_close_does_not_error_if_collectors_cleared(self):
        """Closing the generator does not error if the debate was already cleaned up."""
        gen = spectate_sse_generator("d1", heartbeat_interval=0.05)
        await gen.__anext__()
        # Manually clear collectors before close
        _active_collectors.clear()
        # Should not raise
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_event_type_in_sse_frame(self):
        """The SSE event: line reflects the event type from the pushed event."""
        gen = spectate_sse_generator("d1", heartbeat_interval=5.0)
        await gen.__anext__()  # connected
        push_spectator_event("d1", "consensus")
        frame = await gen.__anext__()
        assert frame.startswith("event: consensus\n")
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_event_without_type_key_defaults(self):
        """If pushed event lacks a 'type' key, falls back to 'event'."""
        gen = spectate_sse_generator("d1", heartbeat_interval=5.0)
        await gen.__anext__()  # connected
        # Directly put an event without 'type' key
        queues = list(_active_collectors["d1"])
        queues[0].put_nowait({"data": "no type here"})
        frame = await gen.__anext__()
        assert frame.startswith("event: event\n")
        await gen.aclose()


# ===========================================================================
# _sse_frame
# ===========================================================================


class TestSSEFrame:
    """Tests for _sse_frame()."""

    def test_basic_format(self):
        """Produces correctly formatted SSE frame."""
        frame = _sse_frame("test", {"key": "value"})
        assert frame.startswith("event: test\n")
        assert "data: " in frame
        assert frame.endswith("\n\n")

    def test_data_is_valid_json(self):
        """The data line contains valid JSON."""
        frame = _sse_frame("event_type", {"a": 1, "b": "two"})
        data_line = frame.split("data: ")[1].rstrip("\n")
        parsed = json.loads(data_line)
        assert parsed["a"] == 1
        assert parsed["b"] == "two"

    def test_event_type_in_frame(self):
        """The event type appears in the event: line."""
        frame = _sse_frame("my_custom_event", {})
        assert "event: my_custom_event\n" in frame

    def test_non_serializable_uses_str(self):
        """Non-JSON-serializable objects are converted via str()."""
        # time objects are not JSON-serializable, but default=str handles them
        import datetime

        frame = _sse_frame("test", {"ts": datetime.datetime(2026, 1, 1)})
        data_line = frame.split("data: ")[1].rstrip("\n")
        parsed = json.loads(data_line)
        assert "2026" in parsed["ts"]

    def test_string_data(self):
        """Works with string data."""
        frame = _sse_frame("msg", "hello world")
        data_line = frame.split("data: ")[1].rstrip("\n")
        assert json.loads(data_line) == "hello world"

    def test_list_data(self):
        """Works with list data."""
        frame = _sse_frame("items", [1, 2, 3])
        data_line = frame.split("data: ")[1].rstrip("\n")
        assert json.loads(data_line) == [1, 2, 3]

    def test_null_data(self):
        """Works with None data."""
        frame = _sse_frame("empty", None)
        data_line = frame.split("data: ")[1].rstrip("\n")
        assert json.loads(data_line) is None


# ===========================================================================
# register_spectate_routes
# ===========================================================================


class TestRegisterSpectateRoutes:
    """Tests for register_spectate_routes()."""

    def test_adds_get_route(self):
        """Registers a GET route on the router."""
        router = MagicMock()
        register_spectate_routes(router)
        router.add_route.assert_called_once()
        args = router.add_route.call_args
        assert args[0][0] == "GET"
        assert args[0][1] == "/api/v1/debates/{debate_id}/spectate"

    def test_endpoint_is_callable(self):
        """The registered endpoint is a callable."""
        router = MagicMock()
        register_spectate_routes(router)
        endpoint = router.add_route.call_args[0][2]
        assert callable(endpoint)

    @pytest.mark.asyncio
    async def test_endpoint_with_starlette_returns_streaming_response(self):
        """When Starlette is available, returns a StreamingResponse."""
        router = MagicMock()
        register_spectate_routes(router)
        endpoint = router.add_route.call_args[0][2]

        mock_request = MagicMock()
        mock_request.path_params = {"debate_id": "d1"}

        try:
            from starlette.responses import StreamingResponse

            result = await endpoint(mock_request)
            assert isinstance(result, StreamingResponse)
            assert result.media_type == "text/event-stream"
            # Check headers
            assert result.headers.get("Cache-Control") == "no-cache"
            assert result.headers.get("X-Accel-Buffering") == "no"
        except ImportError:
            pytest.skip("Starlette not installed")

    @pytest.mark.asyncio
    async def test_endpoint_fallback_when_no_starlette(self):
        """When Starlette is not available, returns JSON fallback."""
        router = MagicMock()
        register_spectate_routes(router)
        endpoint = router.add_route.call_args[0][2]

        mock_request = MagicMock()
        mock_request.path_params = {"debate_id": "d1"}

        with patch.dict("sys.modules", {"starlette": None, "starlette.responses": None}):
            # We need to make the import fail inside the endpoint
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "starlette.responses":
                    raise ImportError("No starlette")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = await endpoint(mock_request)
                body = _body(result)
                assert body["debate_id"] == "d1"
                assert body["spectate_available"] is True
                assert body["message"] == "Connect via SSE client"

    @pytest.mark.asyncio
    async def test_endpoint_extracts_debate_id_from_path_params(self):
        """Endpoint reads debate_id from request.path_params."""
        router = MagicMock()
        register_spectate_routes(router)
        endpoint = router.add_route.call_args[0][2]

        mock_request = MagicMock()
        mock_request.path_params = {"debate_id": "test-debate-42"}

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "starlette.responses":
                raise ImportError("No starlette")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await endpoint(mock_request)
            body = _body(result)
            assert body["debate_id"] == "test-debate-42"
            assert "test-debate-42" in body["sse_url"]

    @pytest.mark.asyncio
    async def test_endpoint_missing_debate_id_uses_empty(self):
        """When path_params has no debate_id, defaults to empty string."""
        router = MagicMock()
        register_spectate_routes(router)
        endpoint = router.add_route.call_args[0][2]

        mock_request = MagicMock()
        mock_request.path_params = {}

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "starlette.responses":
                raise ImportError("No starlette")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await endpoint(mock_request)
            body = _body(result)
            assert body["debate_id"] == ""


# ===========================================================================
# Integration-style tests
# ===========================================================================


class TestIntegration:
    """Integration tests combining push and generator."""

    @pytest.mark.asyncio
    async def test_push_received_by_generator(self):
        """An event pushed while a generator is active is received by the generator."""
        gen = spectate_sse_generator("d1", heartbeat_interval=5.0)
        connected = await gen.__anext__()
        assert "connected" in connected

        push_spectator_event("d1", "debate_start", agent="system", details="Starting")
        frame = await gen.__anext__()
        data = json.loads(frame.split("data: ")[1].strip())
        assert data["type"] == "debate_start"
        assert data["agent"] == "system"
        assert data["details"] == "Starting"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_multiple_events_in_sequence(self):
        """Multiple pushed events are received in order."""
        gen = spectate_sse_generator("d1", heartbeat_interval=5.0)
        await gen.__anext__()  # connected

        push_spectator_event("d1", "round_start", round_number=1)
        push_spectator_event("d1", "proposal", agent="claude")
        push_spectator_event("d1", "critique", agent="gpt-4")

        f1 = await gen.__anext__()
        f2 = await gen.__anext__()
        f3 = await gen.__anext__()

        assert "round_start" in f1
        assert "proposal" in f2
        assert "critique" in f3
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_push_with_no_active_generators(self):
        """Pushing events with no active generators is a no-op returning 0."""
        count = push_spectator_event("d1", "orphan_event")
        assert count == 0

    @pytest.mark.asyncio
    async def test_push_returns_correct_client_count(self):
        """push_spectator_event returns the number of clients that received the event."""
        gen1 = spectate_sse_generator("d1", heartbeat_interval=5.0)
        gen2 = spectate_sse_generator("d1", heartbeat_interval=5.0)
        await gen1.__anext__()
        await gen2.__anext__()
        count = push_spectator_event("d1", "test")
        assert count == 2
        await gen1.aclose()
        await gen2.aclose()

    @pytest.mark.asyncio
    async def test_generator_cleanup_after_all_close(self):
        """After all generators close, the debate is fully cleaned from registry."""
        gen1 = spectate_sse_generator("d1", heartbeat_interval=0.05)
        gen2 = spectate_sse_generator("d1", heartbeat_interval=0.05)
        await gen1.__anext__()
        await gen2.__anext__()
        assert len(_active_collectors["d1"]) == 2
        await gen1.aclose()
        assert len(_active_collectors.get("d1", set())) == 1
        await gen2.aclose()
        assert "d1" not in _active_collectors
