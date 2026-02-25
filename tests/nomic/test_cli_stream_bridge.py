"""Tests for CLIStreamBridge - bridging CLI events to WebSocket streams."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.cli_stream_bridge import (
    CLIStreamBridge,
    _NOMIC_PHASE_MAP,
    _PIPELINE_STAGE_MAP,
)


@pytest.fixture
def bridge():
    """Create a CLIStreamBridge with defaults for testing."""
    return CLIStreamBridge(
        nomic_port=0,
        pipeline_id="test-pipeline-001",
        print_to_stdout=False,
    )


@pytest.fixture
def bridge_with_stdout():
    """Create a CLIStreamBridge that prints to stdout."""
    return CLIStreamBridge(
        nomic_port=0,
        pipeline_id="test-pipeline-002",
        print_to_stdout=True,
    )


class TestBridgeInit:
    """Test bridge initialization."""

    def test_default_init(self):
        b = CLIStreamBridge()
        assert b.started is False
        assert b.pipeline_id.startswith("cli-")
        assert len(b.pipeline_id) == 12  # "cli-" + 8 hex chars

    def test_custom_pipeline_id(self, bridge):
        assert bridge.pipeline_id == "test-pipeline-001"

    def test_not_started_initially(self, bridge):
        assert bridge.started is False
        assert bridge._nomic_server is None
        assert bridge._pipeline_emitter is None


class TestBridgeLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_servers(self):
        """Test that start() creates the Nomic server and pipeline emitter."""
        mock_nomic = AsyncMock()
        mock_nomic.start = AsyncMock()
        mock_emitter = MagicMock()

        with patch(
            "aragora.nomic.cli_stream_bridge.CLIStreamBridge._create_nomic_server",
            return_value=mock_nomic,
            create=True,
        ):
            bridge = CLIStreamBridge(pipeline_id="test")
            # Patch the internal creation to use mocks
            with (
                patch(
                    "aragora.server.stream.nomic_loop_stream.NomicLoopStreamServer",
                    return_value=mock_nomic,
                ),
                patch(
                    "aragora.server.stream.pipeline_stream.get_pipeline_emitter",
                    return_value=mock_emitter,
                ),
            ):
                await bridge.start()
                assert bridge.started is True
                mock_nomic.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_graceful_on_import_error(self):
        """Test that start() handles missing websockets gracefully."""
        bridge = CLIStreamBridge(
            pipeline_id="test",
            enable_pipeline_emitter=False,
        )
        with patch(
            "aragora.server.stream.nomic_loop_stream.NomicLoopStreamServer",
            side_effect=ImportError("no websockets"),
        ):
            # Should not raise
            await bridge.start()
            assert bridge.started is True
            assert bridge._nomic_server is None

    @pytest.mark.asyncio
    async def test_start_graceful_on_os_error(self):
        """Test that start() handles port-in-use gracefully."""
        mock_server = AsyncMock()
        mock_server.start = AsyncMock(side_effect=OSError("port in use"))

        bridge = CLIStreamBridge(
            pipeline_id="test",
            enable_pipeline_emitter=False,
        )
        with patch(
            "aragora.server.stream.nomic_loop_stream.NomicLoopStreamServer",
            return_value=mock_server,
        ):
            await bridge.start()
            assert bridge.started is True
            assert bridge._nomic_server is None

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """Test that stop() shuts down servers."""
        mock_nomic = AsyncMock()
        mock_nomic.start = AsyncMock()
        mock_nomic.stop = AsyncMock()

        bridge = CLIStreamBridge(
            pipeline_id="test",
            enable_pipeline_emitter=False,
        )
        with patch(
            "aragora.server.stream.nomic_loop_stream.NomicLoopStreamServer",
            return_value=mock_nomic,
        ):
            await bridge.start()
            await bridge.stop()

        assert bridge.started is False
        assert bridge._nomic_server is None
        mock_nomic.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        """Test that calling start() twice does not restart."""
        bridge = CLIStreamBridge(
            pipeline_id="test",
            enable_nomic_server=False,
            enable_pipeline_emitter=False,
        )
        await bridge.start()
        first_start_time = bridge._start_time
        await bridge.start()
        assert bridge._start_time == first_start_time

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, bridge):
        """Test that stop() before start() does nothing."""
        await bridge.stop()
        assert bridge.started is False


class TestNomicEventEmission:
    """Test emitting events to NomicLoopStreamServer."""

    @pytest.mark.asyncio
    async def test_emit_phase_started(self, bridge):
        mock_server = AsyncMock()
        bridge._nomic_server = mock_server
        bridge._started = True

        await bridge.emit_nomic_event("debate", "started")
        mock_server.emit_phase_started.assert_awaited_once_with(
            phase="debate",
            cycle=0,
        )

    @pytest.mark.asyncio
    async def test_emit_phase_completed(self, bridge):
        mock_server = AsyncMock()
        bridge._nomic_server = mock_server
        bridge._started = True
        bridge._start_time = 100.0

        with patch("aragora.nomic.cli_stream_bridge.time") as mock_time:
            mock_time.time.return_value = 110.0
            await bridge.emit_nomic_event("implement", "completed", {"summary": "done"})

        mock_server.emit_phase_completed.assert_awaited_once()
        call_kwargs = mock_server.emit_phase_completed.call_args.kwargs
        assert call_kwargs["phase"] == "implement"
        assert call_kwargs["result_summary"] == "done"

    @pytest.mark.asyncio
    async def test_emit_phase_failed(self, bridge):
        mock_server = AsyncMock()
        bridge._nomic_server = mock_server
        bridge._started = True

        await bridge.emit_nomic_event("verify", "failed", {"error": "tests failed"})
        mock_server.emit_phase_failed.assert_awaited_once_with(
            phase="verify",
            cycle=0,
            error="tests failed",
        )

    @pytest.mark.asyncio
    async def test_emit_unknown_status_sends_log(self, bridge):
        mock_server = AsyncMock()
        bridge._nomic_server = mock_server
        bridge._started = True

        await bridge.emit_nomic_event("design", "paused")
        mock_server.emit_log_message.assert_awaited_once_with(
            level="info",
            message="design: paused",
            source="cli_bridge",
        )

    @pytest.mark.asyncio
    async def test_emit_nomic_noop_without_server(self, bridge):
        """No error when server is None."""
        bridge._nomic_server = None
        await bridge.emit_nomic_event("debate", "started")  # should not raise

    @pytest.mark.asyncio
    async def test_emit_nomic_catches_connection_error(self, bridge):
        mock_server = AsyncMock()
        mock_server.emit_phase_started = AsyncMock(side_effect=ConnectionError("disconnected"))
        bridge._nomic_server = mock_server

        # Should not raise
        await bridge.emit_nomic_event("debate", "started")


class TestPipelineEventEmission:
    """Test emitting events to PipelineStreamEmitter."""

    @pytest.mark.asyncio
    async def test_emit_stage_started(self, bridge):
        mock_emitter = AsyncMock()
        bridge._pipeline_emitter = mock_emitter

        await bridge.emit_pipeline_event("goals", "started", {"key": "val"})
        mock_emitter.emit_stage_started.assert_awaited_once_with(
            pipeline_id="test-pipeline-001",
            stage_name="goals",
            config={"key": "val"},
        )

    @pytest.mark.asyncio
    async def test_emit_stage_completed(self, bridge):
        mock_emitter = AsyncMock()
        bridge._pipeline_emitter = mock_emitter

        await bridge.emit_pipeline_event("workflows", "completed", {"count": 5})
        mock_emitter.emit_stage_completed.assert_awaited_once_with(
            pipeline_id="test-pipeline-001",
            stage_name="workflows",
            summary={"count": 5},
        )

    @pytest.mark.asyncio
    async def test_emit_failed(self, bridge):
        mock_emitter = AsyncMock()
        bridge._pipeline_emitter = mock_emitter

        await bridge.emit_pipeline_event("orchestration", "failed", {"error": "boom"})
        mock_emitter.emit_failed.assert_awaited_once_with(
            pipeline_id="test-pipeline-001",
            error="boom",
        )

    @pytest.mark.asyncio
    async def test_emit_progress(self, bridge):
        mock_emitter = AsyncMock()
        bridge._pipeline_emitter = mock_emitter

        await bridge.emit_pipeline_event("ideas", "progress", {"step": "extract", "progress": 0.5})
        mock_emitter.emit_step_progress.assert_awaited_once_with(
            pipeline_id="test-pipeline-001",
            step_name="extract",
            progress=0.5,
        )

    @pytest.mark.asyncio
    async def test_emit_pipeline_noop_without_emitter(self, bridge):
        bridge._pipeline_emitter = None
        await bridge.emit_pipeline_event("goals", "started")  # should not raise


class TestProgressCallback:
    """Test the progress callback compatibility with self_develop.py."""

    def test_callback_returns_callable(self, bridge):
        cb = bridge.as_progress_callback()
        assert callable(cb)

    def test_callback_prints_to_stdout(self, bridge_with_stdout, capsys):
        cb = bridge_with_stdout.as_progress_callback()
        cb("cycle_started", {"cycle_id": "abc-123"})
        captured = capsys.readouterr()
        assert "abc-123" in captured.out

    def test_callback_prints_planning_complete(self, bridge_with_stdout, capsys):
        cb = bridge_with_stdout.as_progress_callback()
        cb("planning_complete", {"goals": 5})
        captured = capsys.readouterr()
        assert "5 goals identified" in captured.out

    def test_callback_prints_decomposition_complete(self, bridge_with_stdout, capsys):
        cb = bridge_with_stdout.as_progress_callback()
        cb("decomposition_complete", {"subtasks": 12})
        captured = capsys.readouterr()
        assert "12 subtasks created" in captured.out

    def test_callback_prints_execution_complete(self, bridge_with_stdout, capsys):
        cb = bridge_with_stdout.as_progress_callback()
        cb("execution_complete", {"completed": 8, "failed": 2})
        captured = capsys.readouterr()
        assert "8 completed" in captured.out
        assert "2 failed" in captured.out

    def test_callback_prints_cycle_complete(self, bridge_with_stdout, capsys):
        cb = bridge_with_stdout.as_progress_callback()
        cb("cycle_complete", {"completed": 10, "failed": 1, "duration": 42.5})
        captured = capsys.readouterr()
        assert "42.5s" in captured.out

    def test_callback_suppresses_stdout_when_disabled(self, bridge, capsys):
        cb = bridge.as_progress_callback()
        cb("cycle_started", {"cycle_id": "abc-123"})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_callback_fires_events_in_running_loop(self, bridge):
        """Test that the callback emits events when an event loop is running."""
        mock_nomic = AsyncMock()
        mock_emitter = AsyncMock()
        bridge._nomic_server = mock_nomic
        bridge._pipeline_emitter = mock_emitter
        bridge._started = True

        cb = bridge.as_progress_callback()

        async def run():
            cb("cycle_started", {"cycle_id": "test"})
            # Let fire-and-forget tasks run
            await asyncio.sleep(0.05)

        asyncio.run(run())
        # Verify that nomic emission was attempted (phase "context" for cycle_started)
        assert mock_nomic.emit_phase_started.await_count > 0

    def test_callback_no_error_without_event_loop(self, bridge):
        """Test that callback works even without a running event loop."""
        bridge._started = True
        cb = bridge.as_progress_callback()
        # Should not raise even without an event loop
        cb("cycle_started", {"cycle_id": "test"})


class TestEventMapping:
    """Test that event name mappings are complete and correct."""

    def test_nomic_phase_map_covers_all_progress_events(self):
        expected_events = {
            "cycle_started",
            "planning_complete",
            "decomposition_complete",
            "risk_assessment_complete",
            "risk_blocked",
            "risk_review_needed",
            "execution_complete",
            "cycle_complete",
        }
        assert set(_NOMIC_PHASE_MAP.keys()) == expected_events

    def test_pipeline_stage_map_covers_all_progress_events(self):
        expected_events = {
            "cycle_started",
            "planning_complete",
            "decomposition_complete",
            "risk_assessment_complete",
            "risk_blocked",
            "risk_review_needed",
            "execution_complete",
            "cycle_complete",
        }
        assert set(_PIPELINE_STAGE_MAP.keys()) == expected_events

    def test_nomic_phases_are_valid(self):
        valid_phases = {"context", "debate", "design", "implement", "verify"}
        for phase in _NOMIC_PHASE_MAP.values():
            assert phase in valid_phases, f"Invalid Nomic phase: {phase}"

    def test_pipeline_stages_are_valid(self):
        valid_stages = {"ideas", "goals", "workflows", "orchestration"}
        for stage in _PIPELINE_STAGE_MAP.values():
            assert stage in valid_stages, f"Invalid pipeline stage: {stage}"

    def test_cycle_started_increments_cycle_number(self, bridge):
        """Test that cycle_started increments the internal cycle counter."""
        bridge._started = True
        assert bridge._cycle_number == 0
        bridge._emit_fire_and_forget("cycle_started", {"cycle_id": "1"})
        assert bridge._cycle_number == 1
        bridge._emit_fire_and_forget("cycle_started", {"cycle_id": "2"})
        assert bridge._cycle_number == 2
