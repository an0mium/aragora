"""
Tests for debate spectator SSE endpoint.

Tests the handlers in aragora/server/handlers/debates/spectate.py:
- GET /api/v1/debates/:id/spectate (SSE stream of real-time events)
- push_spectator_event fan-out to connected clients
- spectate_sse_generator lifecycle (connect, events, heartbeat, disconnect)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.debates.spectate import (
    _active_collectors,
    handle_spectate,
    push_spectator_event,
    spectate_sse_generator,
)


def _body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    raw = result["body"]
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    return raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_collectors():
    """Clear active collectors between tests."""
    _active_collectors.clear()
    yield
    _active_collectors.clear()


@pytest.fixture
def auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock(spec=AuthorizationContext)
    ctx.user_id = "user-123"
    ctx.org_id = "org-456"
    ctx.permissions = {"debates:read"}
    return ctx


# ---------------------------------------------------------------------------
# push_spectator_event
# ---------------------------------------------------------------------------


class TestPushSpectatorEvent:
    """Tests for push_spectator_event fan-out."""

    def test_push_no_collectors(self):
        """Returns 0 when no clients are watching."""
        assert push_spectator_event("debate-1", "proposal", agent="claude") == 0

    def test_push_to_single_client(self):
        """Pushes event to a single watching client."""
        q: asyncio.Queue = asyncio.Queue()
        _active_collectors["debate-1"] = {q}

        count = push_spectator_event("debate-1", "proposal", agent="claude", details="test")

        assert count == 1
        event = q.get_nowait()
        assert event["type"] == "proposal"
        assert event["agent"] == "claude"
        assert event["details"] == "test"
        assert "timestamp" in event

    def test_push_to_multiple_clients(self):
        """Fans out to all watching clients."""
        q1: asyncio.Queue = asyncio.Queue()
        q2: asyncio.Queue = asyncio.Queue()
        q3: asyncio.Queue = asyncio.Queue()
        _active_collectors["debate-1"] = {q1, q2, q3}

        count = push_spectator_event("debate-1", "vote", agent="gpt4")

        assert count == 3
        for q in (q1, q2, q3):
            event = q.get_nowait()
            assert event["type"] == "vote"

    def test_push_full_queue_drops_oldest(self):
        """When queue is full, drops oldest event and enqueues new one."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        q.put_nowait({"type": "old_event"})
        _active_collectors["debate-1"] = {q}

        count = push_spectator_event("debate-1", "critique", agent="gemini")

        assert count == 1
        event = q.get_nowait()
        assert event["type"] == "critique"

    def test_push_different_debates_isolated(self):
        """Events are scoped to their debate_id."""
        q1: asyncio.Queue = asyncio.Queue()
        q2: asyncio.Queue = asyncio.Queue()
        _active_collectors["debate-a"] = {q1}
        _active_collectors["debate-b"] = {q2}

        push_spectator_event("debate-a", "proposal")

        assert q2.empty() is not False  # q2 should be empty
        assert q2.qsize() == 0
        assert q1.qsize() == 1

    def test_push_includes_round_and_metric(self):
        """Round number and metric are included in event payload."""
        q: asyncio.Queue = asyncio.Queue()
        _active_collectors["debate-1"] = {q}

        push_spectator_event("debate-1", "convergence", metric=0.85, round_number=3)

        event = q.get_nowait()
        assert event["metric"] == 0.85
        assert event["round"] == 3


# ---------------------------------------------------------------------------
# spectate_sse_generator
# ---------------------------------------------------------------------------


class TestSpectateSSEGenerator:
    """Tests for the SSE generator lifecycle."""

    @pytest.mark.asyncio
    async def test_generator_sends_connected_event(self):
        """Generator yields a connected event first."""
        gen = spectate_sse_generator("debate-1", heartbeat_interval=0.1)
        frame = await gen.__anext__()
        assert "event: connected" in frame
        assert "debate-1" in frame
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_generator_registers_and_unregisters(self):
        """Generator registers queue on start, removes on close."""
        gen = spectate_sse_generator("debate-1", heartbeat_interval=0.1)

        # Consume connected frame to trigger registration
        await gen.__anext__()

        assert "debate-1" in _active_collectors
        assert len(_active_collectors["debate-1"]) == 1

        await gen.aclose()

        # After close, collector should be cleaned up
        assert "debate-1" not in _active_collectors

    @pytest.mark.asyncio
    async def test_generator_yields_pushed_events(self):
        """Events pushed via push_spectator_event appear in the generator."""
        gen = spectate_sse_generator("debate-1", heartbeat_interval=5.0)

        # Consume connected
        await gen.__anext__()

        # Push event from another coroutine
        push_spectator_event("debate-1", "proposal", agent="claude", details="hi")

        frame = await gen.__anext__()
        assert "event: proposal" in frame
        data = json.loads(frame.split("data: ")[1].split("\n")[0])
        assert data["agent"] == "claude"

        await gen.aclose()

    @pytest.mark.asyncio
    async def test_generator_sends_heartbeat_on_timeout(self):
        """Generator sends heartbeat comment when no events arrive."""
        gen = spectate_sse_generator("debate-1", heartbeat_interval=0.05)

        # Consume connected
        await gen.__anext__()

        # Wait for heartbeat
        frame = await gen.__anext__()
        assert frame == ": heartbeat\n\n"

        await gen.aclose()

    @pytest.mark.asyncio
    async def test_multiple_generators_same_debate(self):
        """Multiple clients can spectate the same debate."""
        gen1 = spectate_sse_generator("debate-1", heartbeat_interval=5.0)
        gen2 = spectate_sse_generator("debate-1", heartbeat_interval=5.0)

        await gen1.__anext__()  # connected
        await gen2.__anext__()  # connected

        assert len(_active_collectors["debate-1"]) == 2

        push_spectator_event("debate-1", "vote", agent="gpt4")

        frame1 = await gen1.__anext__()
        frame2 = await gen2.__anext__()
        assert "vote" in frame1
        assert "vote" in frame2

        await gen1.aclose()
        assert len(_active_collectors["debate-1"]) == 1

        await gen2.aclose()
        assert "debate-1" not in _active_collectors


# ---------------------------------------------------------------------------
# handle_spectate (non-streaming fallback)
# ---------------------------------------------------------------------------


class TestHandleSpectate:
    """Tests for handle_spectate handler."""

    @pytest.mark.asyncio
    async def test_spectate_status_no_viewers(self, auth_context):
        """Returns available with 0 viewers."""
        result = await handle_spectate("debate-1", auth_context)
        assert result["status"] == 200
        body = _body(result)
        assert body["spectate_available"] is True
        assert body["active_viewers"] == 0

    @pytest.mark.asyncio
    async def test_spectate_status_with_viewers(self, auth_context):
        """Returns correct viewer count."""
        _active_collectors["debate-1"] = {asyncio.Queue(), asyncio.Queue()}
        result = await handle_spectate("debate-1", auth_context)
        body = _body(result)
        assert body["active_viewers"] == 2
        assert body["debate_id"] == "debate-1"
        assert "/spectate" in body["sse_url"]
