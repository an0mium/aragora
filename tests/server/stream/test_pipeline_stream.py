"""Tests for PipelineStreamEmitter."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.stream.pipeline_stream import (
    PipelineStreamClient,
    PipelineStreamEmitter,
    get_pipeline_emitter,
    set_pipeline_emitter,
)
from aragora.events.types import StreamEventType


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def emitter():
    return PipelineStreamEmitter()


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.send_str = AsyncMock()
    return ws


@pytest.fixture
def mock_ws_factory():
    def factory():
        ws = AsyncMock()
        ws.send_str = AsyncMock()
        return ws

    return factory


# =========================================================================
# Client management
# =========================================================================


class TestClientManagement:
    def test_add_client(self, emitter, mock_ws):
        client_id = emitter.add_client(mock_ws, "pipe-1")
        assert client_id.startswith("pipe_")
        assert emitter.client_count == 1

    def test_remove_client(self, emitter, mock_ws):
        client_id = emitter.add_client(mock_ws, "pipe-1")
        emitter.remove_client(client_id)
        assert emitter.client_count == 0

    def test_remove_nonexistent_client(self, emitter):
        emitter.remove_client("nonexistent")
        assert emitter.client_count == 0

    def test_multiple_clients(self, emitter, mock_ws_factory):
        ws1 = mock_ws_factory()
        ws2 = mock_ws_factory()
        id1 = emitter.add_client(ws1, "pipe-1")
        id2 = emitter.add_client(ws2, "pipe-2")
        assert emitter.client_count == 2
        assert id1 != id2

    def test_client_with_subscriptions(self, emitter, mock_ws):
        client_id = emitter.add_client(
            mock_ws,
            "pipe-1",
            subscriptions={"pipeline_stage_started"},
        )
        assert emitter._clients[client_id].subscriptions == {"pipeline_stage_started"}


# =========================================================================
# Event emission
# =========================================================================


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_emit_to_matching_pipeline(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit("pipe-1", StreamEventType.PIPELINE_STARTED, {"test": True})
        mock_ws.send_str.assert_called_once()
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_started"
        assert sent["data"]["pipeline_id"] == "pipe-1"

    @pytest.mark.asyncio
    async def test_no_emit_to_different_pipeline(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit("pipe-2", StreamEventType.PIPELINE_STARTED, {"test": True})
        mock_ws.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscription_filter(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1", subscriptions={"pipeline_completed"})
        await emitter.emit("pipe-1", StreamEventType.PIPELINE_STARTED, {"test": True})
        mock_ws.send_str.assert_not_called()

        await emitter.emit("pipe-1", StreamEventType.PIPELINE_COMPLETED, {"test": True})
        mock_ws.send_str.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnected_client_cleanup(self, emitter, mock_ws):
        mock_ws.send_str.side_effect = ConnectionError("gone")
        emitter.add_client(mock_ws, "pipe-1")
        assert emitter.client_count == 1

        await emitter.emit("pipe-1", StreamEventType.PIPELINE_STARTED, {})
        assert emitter.client_count == 0


# =========================================================================
# Event history
# =========================================================================


class TestEventHistory:
    @pytest.mark.asyncio
    async def test_history_stored(self, emitter):
        await emitter.emit("pipe-1", StreamEventType.PIPELINE_STARTED, {"step": 1})
        await emitter.emit("pipe-1", StreamEventType.PIPELINE_COMPLETED, {"step": 2})

        history = emitter.get_history("pipe-1")
        assert len(history) == 2
        assert history[0]["type"] == "pipeline_started"
        assert history[1]["type"] == "pipeline_completed"

    @pytest.mark.asyncio
    async def test_history_per_pipeline(self, emitter):
        await emitter.emit("pipe-1", StreamEventType.PIPELINE_STARTED, {})
        await emitter.emit("pipe-2", StreamEventType.PIPELINE_FAILED, {})

        assert len(emitter.get_history("pipe-1")) == 1
        assert len(emitter.get_history("pipe-2")) == 1
        assert len(emitter.get_history("pipe-3")) == 0

    @pytest.mark.asyncio
    async def test_history_limit(self, emitter):
        for i in range(10):
            await emitter.emit("pipe-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"i": i})

        history = emitter.get_history("pipe-1", limit=3)
        assert len(history) == 3


# =========================================================================
# Convenience methods
# =========================================================================


class TestConvenienceMethods:
    @pytest.mark.asyncio
    async def test_emit_stage_started(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_stage_started("pipe-1", "ideation", {"key": "val"})
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["stage"] == "ideation"

    @pytest.mark.asyncio
    async def test_emit_stage_completed(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_stage_completed("pipe-1", "goals", {"count": 5})
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["stage"] == "goals"

    @pytest.mark.asyncio
    async def test_emit_goal_extracted(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_goal_extracted("pipe-1", {"id": "g1", "title": "Test"})
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["goal"]["id"] == "g1"

    @pytest.mark.asyncio
    async def test_emit_completed(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_completed("pipe-1", {"hash": "abc"})
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["receipt"]["hash"] == "abc"

    @pytest.mark.asyncio
    async def test_emit_failed(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_failed("pipe-1", "something broke")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["error"] == "something broke"

    @pytest.mark.asyncio
    async def test_emit_node_added(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_node_added("pipe-1", "ideas", "n1", "ideaNode", "Rate limiter")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_node_added"
        assert sent["data"]["stage"] == "ideas"
        assert sent["data"]["node_id"] == "n1"
        assert sent["data"]["node_type"] == "ideaNode"
        assert sent["data"]["label"] == "Rate limiter"
        assert "added_at" in sent["data"]

    @pytest.mark.asyncio
    async def test_emit_transition_pending(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "pipe-1")
        await emitter.emit_transition_pending(
            "pipe-1",
            "ideas",
            "goals",
            0.72,
            "Extracted 3 goals from 4 ideas",
        )
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_transition_pending"
        assert sent["data"]["from_stage"] == "ideas"
        assert sent["data"]["to_stage"] == "goals"
        assert sent["data"]["confidence"] == 0.72
        assert sent["data"]["ai_rationale"] == "Extracted 3 goals from 4 ideas"
        assert "pending_at" in sent["data"]


# =========================================================================
# Event callback adapter
# =========================================================================


class TestEventCallback:
    def test_as_event_callback_returns_callable(self, emitter):
        cb = emitter.as_event_callback("pipe-1")
        assert callable(cb)


# =========================================================================
# Global emitter
# =========================================================================


class TestGlobalEmitter:
    def test_get_creates_singleton(self):
        set_pipeline_emitter(None)  # Reset
        e1 = get_pipeline_emitter()
        e2 = get_pipeline_emitter()
        assert e1 is e2

    def test_set_overrides(self):
        custom = PipelineStreamEmitter()
        set_pipeline_emitter(custom)
        assert get_pipeline_emitter() is custom
        set_pipeline_emitter(None)  # Cleanup
