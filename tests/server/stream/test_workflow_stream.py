"""Tests for WorkflowStreamEmitter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from aragora.server.stream.workflow_stream import (
    WorkflowStreamClient,
    WorkflowStreamEmitter,
    get_workflow_emitter,
    set_workflow_emitter,
)
from aragora.events.types import StreamEventType


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def emitter():
    return WorkflowStreamEmitter()


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
# Initialization
# =========================================================================


class TestInitialization:
    def test_initial_state(self, emitter):
        assert emitter.client_count == 0
        assert emitter._counter == 0
        assert emitter._max_history == 500
        assert emitter._clients == {}
        assert emitter._event_history == {}


# =========================================================================
# Client management
# =========================================================================


class TestClientManagement:
    def test_add_client(self, emitter, mock_ws):
        client_id = emitter.add_client(mock_ws, "wf-1")
        assert client_id.startswith("wf_")
        assert emitter.client_count == 1

    def test_remove_client(self, emitter, mock_ws):
        client_id = emitter.add_client(mock_ws, "wf-1")
        emitter.remove_client(client_id)
        assert emitter.client_count == 0

    def test_remove_nonexistent_client(self, emitter):
        emitter.remove_client("nonexistent")
        assert emitter.client_count == 0

    def test_multiple_clients(self, emitter, mock_ws_factory):
        ws1 = mock_ws_factory()
        ws2 = mock_ws_factory()
        id1 = emitter.add_client(ws1, "wf-1")
        id2 = emitter.add_client(ws2, "wf-2")
        assert emitter.client_count == 2
        assert id1 != id2

    def test_client_stores_workflow_id(self, emitter, mock_ws):
        client_id = emitter.add_client(mock_ws, "wf-42")
        client = emitter._clients[client_id]
        assert isinstance(client, WorkflowStreamClient)
        assert client.workflow_id == "wf-42"
        assert client.ws is mock_ws
        assert client.connected_at > 0

    def test_add_client_increments_counter(self, emitter, mock_ws_factory):
        ws1 = mock_ws_factory()
        ws2 = mock_ws_factory()
        emitter.add_client(ws1, "wf-1")
        emitter.add_client(ws2, "wf-1")
        assert emitter._counter == 2


# =========================================================================
# Event emission
# =========================================================================


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_emit_to_matching_workflow(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"test": True})
        mock_ws.send_str.assert_called_once()
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_step_progress"
        assert sent["data"]["workflow_id"] == "wf-1"
        assert sent["data"]["test"] is True

    @pytest.mark.asyncio
    async def test_no_emit_to_different_workflow(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit("wf-2", StreamEventType.PIPELINE_STEP_PROGRESS, {"test": True})
        mock_ws.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_to_multiple_clients_same_workflow(self, emitter, mock_ws_factory):
        ws1 = mock_ws_factory()
        ws2 = mock_ws_factory()
        emitter.add_client(ws1, "wf-1")
        emitter.add_client(ws2, "wf-1")
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"x": 1})
        ws1.send_str.assert_called_once()
        ws2.send_str.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_filtering_mixed_workflows(self, emitter, mock_ws_factory):
        ws_a = mock_ws_factory()
        ws_b = mock_ws_factory()
        ws_c = mock_ws_factory()
        emitter.add_client(ws_a, "wf-1")
        emitter.add_client(ws_b, "wf-2")
        emitter.add_client(ws_c, "wf-1")

        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"step": "run"})

        ws_a.send_str.assert_called_once()
        ws_b.send_str.assert_not_called()
        ws_c.send_str.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnected_client_cleanup(self, emitter, mock_ws):
        mock_ws.send_str.side_effect = ConnectionError("gone")
        emitter.add_client(mock_ws, "wf-1")
        assert emitter.client_count == 1

        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {})
        assert emitter.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnected_os_error_cleanup(self, emitter, mock_ws):
        mock_ws.send_str.side_effect = OSError("broken pipe")
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {})
        assert emitter.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnected_runtime_error_cleanup(self, emitter, mock_ws):
        mock_ws.send_str.side_effect = RuntimeError("closed")
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {})
        assert emitter.client_count == 0

    @pytest.mark.asyncio
    async def test_healthy_clients_survive_disconnect_cleanup(self, emitter, mock_ws_factory):
        ws_good = mock_ws_factory()
        ws_bad = mock_ws_factory()
        ws_bad.send_str.side_effect = ConnectionError("gone")
        emitter.add_client(ws_good, "wf-1")
        emitter.add_client(ws_bad, "wf-1")
        assert emitter.client_count == 2

        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"ok": True})
        assert emitter.client_count == 1
        ws_good.send_str.assert_called_once()


# =========================================================================
# Event history
# =========================================================================


class TestEventHistory:
    @pytest.mark.asyncio
    async def test_history_stored(self, emitter):
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"step": 1})
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"step": 2})

        history = emitter.get_history("wf-1")
        assert len(history) == 2
        assert history[0]["data"]["step"] == 1
        assert history[1]["data"]["step"] == 2

    @pytest.mark.asyncio
    async def test_history_per_workflow(self, emitter):
        await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {})
        await emitter.emit("wf-2", StreamEventType.PIPELINE_STEP_PROGRESS, {})

        assert len(emitter.get_history("wf-1")) == 1
        assert len(emitter.get_history("wf-2")) == 1
        assert len(emitter.get_history("wf-3")) == 0

    @pytest.mark.asyncio
    async def test_history_limit(self, emitter):
        for i in range(10):
            await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"i": i})

        history = emitter.get_history("wf-1", limit=3)
        assert len(history) == 3
        # Should return the last 3 events
        assert history[0]["data"]["i"] == 7
        assert history[1]["data"]["i"] == 8
        assert history[2]["data"]["i"] == 9

    @pytest.mark.asyncio
    async def test_history_capping(self, emitter):
        emitter._max_history = 5
        for i in range(10):
            await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"i": i})

        # Internal storage should be capped at _max_history
        raw_history = emitter._event_history["wf-1"]
        assert len(raw_history) == 5
        # Should keep the last 5 events
        assert raw_history[0]["data"]["i"] == 5
        assert raw_history[-1]["data"]["i"] == 9

    @pytest.mark.asyncio
    async def test_history_default_limit(self, emitter):
        for i in range(150):
            await emitter.emit("wf-1", StreamEventType.PIPELINE_STEP_PROGRESS, {"i": i})

        history = emitter.get_history("wf-1")
        assert len(history) == 100  # default limit=100


# =========================================================================
# Convenience methods
# =========================================================================


class TestConvenienceMethods:
    @pytest.mark.asyncio
    async def test_emit_step_started(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit_step_started("wf-1", "s1", "validate_input")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_step_progress"
        assert sent["data"]["step_id"] == "s1"
        assert sent["data"]["step_name"] == "validate_input"
        assert sent["data"]["status"] == "started"
        assert sent["data"]["workflow_id"] == "wf-1"

    @pytest.mark.asyncio
    async def test_emit_step_completed(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit_step_completed("wf-1", "s2", "run_analysis", output={"score": 0.9})
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_step_progress"
        assert sent["data"]["step_id"] == "s2"
        assert sent["data"]["step_name"] == "run_analysis"
        assert sent["data"]["status"] == "completed"
        assert sent["data"]["output"] == {"score": 0.9}

    @pytest.mark.asyncio
    async def test_emit_step_completed_no_output(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit_step_completed("wf-1", "s2", "run_analysis")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["output"] == {}

    @pytest.mark.asyncio
    async def test_emit_step_failed(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit_step_failed("wf-1", "s3", "deploy", error="timeout exceeded")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["type"] == "pipeline_step_progress"
        assert sent["data"]["step_id"] == "s3"
        assert sent["data"]["step_name"] == "deploy"
        assert sent["data"]["status"] == "failed"
        assert sent["data"]["error"] == "timeout exceeded"

    @pytest.mark.asyncio
    async def test_emit_step_failed_no_error(self, emitter, mock_ws):
        emitter.add_client(mock_ws, "wf-1")
        await emitter.emit_step_failed("wf-1", "s3", "deploy")
        sent = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent["data"]["error"] == ""


# =========================================================================
# Global emitter (singleton)
# =========================================================================


class TestGlobalEmitter:
    def test_get_creates_singleton(self):
        set_workflow_emitter(None)  # Reset
        e1 = get_workflow_emitter()
        e2 = get_workflow_emitter()
        assert e1 is e2
        set_workflow_emitter(None)  # Cleanup

    def test_set_overrides(self):
        custom = WorkflowStreamEmitter()
        set_workflow_emitter(custom)
        assert get_workflow_emitter() is custom
        set_workflow_emitter(None)  # Cleanup

    def test_set_none_resets(self):
        set_workflow_emitter(WorkflowStreamEmitter())
        set_workflow_emitter(None)
        e1 = get_workflow_emitter()
        assert isinstance(e1, WorkflowStreamEmitter)
        set_workflow_emitter(None)  # Cleanup
