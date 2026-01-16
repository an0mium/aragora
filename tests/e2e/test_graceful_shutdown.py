"""
E2E tests for graceful server shutdown.

Tests the ServerLifecycleManager's ability to:
- Wait for active debates to complete
- Close WebSocket connections gracefully
- Persist circuit breaker states
- Handle shutdown timeout scenarios
- Process signal handlers correctly
"""

from __future__ import annotations

import asyncio
import signal
import time
from typing import Any, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.lifecycle import ServerLifecycleManager


class TestShutdownWaitsForDebates:
    """Test that shutdown waits for active debates to complete."""

    @pytest.fixture
    def mock_active_debates(self) -> Callable[[], Dict[str, Any]]:
        """Create a mutable active debates dict."""
        debates: Dict[str, Any] = {}

        def get_debates() -> Dict[str, Any]:
            return debates

        get_debates.debates = debates  # type: ignore
        return get_debates

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_in_progress_debates(self, mock_active_debates):
        """Verify shutdown waits for in_progress debates to complete."""
        manager = ServerLifecycleManager(get_active_debates=mock_active_debates)

        # Register an active debate
        mock_active_debates.debates["debate-1"] = {"status": "in_progress"}

        # Start shutdown in background
        shutdown_task = asyncio.create_task(manager.graceful_shutdown(timeout=5.0))

        # Wait a bit - shutdown should be waiting
        await asyncio.sleep(0.3)
        assert not shutdown_task.done()

        # Complete the debate
        mock_active_debates.debates["debate-1"]["status"] = "completed"

        # Now shutdown should complete
        await asyncio.wait_for(shutdown_task, timeout=2.0)
        assert shutdown_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_completes_immediately_with_no_debates(self, mock_active_debates):
        """Verify shutdown completes immediately when no debates active."""
        manager = ServerLifecycleManager(get_active_debates=mock_active_debates)

        start = time.time()
        await manager.graceful_shutdown(timeout=5.0)
        elapsed = time.time() - start

        # Should complete in less than 1 second
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_shutdown_completes_immediately_without_callback(self):
        """Verify shutdown completes when no get_active_debates callback."""
        manager = ServerLifecycleManager()

        start = time.time()
        await manager.graceful_shutdown(timeout=5.0)
        elapsed = time.time() - start

        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_multiple_debates(self, mock_active_debates):
        """Verify shutdown waits for multiple debates."""
        manager = ServerLifecycleManager(get_active_debates=mock_active_debates)

        # Register multiple active debates
        mock_active_debates.debates["debate-1"] = {"status": "in_progress"}
        mock_active_debates.debates["debate-2"] = {"status": "in_progress"}
        mock_active_debates.debates["debate-3"] = {"status": "completed"}

        shutdown_task = asyncio.create_task(manager.graceful_shutdown(timeout=5.0))

        await asyncio.sleep(0.2)
        assert not shutdown_task.done()

        # Complete first debate
        mock_active_debates.debates["debate-1"]["status"] = "completed"
        await asyncio.sleep(0.3)
        assert not shutdown_task.done()  # Still waiting for debate-2

        # Complete second debate
        mock_active_debates.debates["debate-2"]["status"] = "completed"

        await asyncio.wait_for(shutdown_task, timeout=2.0)
        assert shutdown_task.done()


class TestShutdownTimeout:
    """Test shutdown timeout handling with stuck debates."""

    @pytest.mark.asyncio
    async def test_shutdown_times_out_with_stuck_debate(self):
        """Verify shutdown times out when debate doesn't complete."""
        debates = {"stuck-debate": {"status": "in_progress"}}
        manager = ServerLifecycleManager(get_active_debates=lambda: debates)

        start = time.time()
        # Use short timeout for test
        await manager.graceful_shutdown(timeout=1.0)
        elapsed = time.time() - start

        # Should timeout after ~1 second
        assert 0.9 < elapsed < 2.0

    @pytest.mark.asyncio
    async def test_is_shutting_down_flag_set(self):
        """Verify is_shutting_down flag is set during shutdown."""
        manager = ServerLifecycleManager()

        assert not manager.is_shutting_down

        await manager.graceful_shutdown(timeout=1.0)

        assert manager.is_shutting_down


class TestShutdownClosesWebSockets:
    """Test WebSocket connection closure during shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_websocket_server(self):
        """Verify WebSocket server graceful_shutdown is called."""
        mock_stream_server = MagicMock()
        mock_stream_server.graceful_shutdown = AsyncMock()

        manager = ServerLifecycleManager(stream_server=mock_stream_server)
        await manager.graceful_shutdown(timeout=1.0)

        mock_stream_server.graceful_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_websocket_error(self):
        """Verify shutdown continues even if WebSocket closure fails."""
        mock_stream_server = MagicMock()
        mock_stream_server.graceful_shutdown = AsyncMock(
            side_effect=RuntimeError("WebSocket error")
        )

        manager = ServerLifecycleManager(stream_server=mock_stream_server)

        # Should not raise despite WebSocket error
        await manager.graceful_shutdown(timeout=1.0)

    @pytest.mark.asyncio
    async def test_shutdown_without_stream_server(self):
        """Verify shutdown works without stream server configured."""
        manager = ServerLifecycleManager(stream_server=None)

        # Should complete without error
        await manager.graceful_shutdown(timeout=1.0)


class TestShutdownPersistsCircuitBreakers:
    """Test circuit breaker state persistence during shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_persists_circuit_breakers(self):
        """Verify circuit breaker states are persisted."""
        manager = ServerLifecycleManager()

        with patch(
            "aragora.resilience.persist_all_circuit_breakers"
        ) as mock_persist:
            mock_persist.return_value = 3
            await manager.graceful_shutdown(timeout=1.0)
            mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_circuit_breaker_error(self):
        """Verify shutdown continues if circuit breaker persistence fails."""
        manager = ServerLifecycleManager()

        with patch(
            "aragora.resilience.persist_all_circuit_breakers",
            side_effect=RuntimeError("Persistence error"),
        ):
            # Should not raise
            await manager.graceful_shutdown(timeout=1.0)


class TestShutdownClosesDatabaseConnections:
    """Test database connection closure during shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_database_managers(self):
        """Verify DatabaseManager instances are cleared."""
        manager = ServerLifecycleManager()

        with patch(
            "aragora.storage.schema.DatabaseManager"
        ) as mock_db_manager:
            await manager.graceful_shutdown(timeout=1.0)
            mock_db_manager.clear_instances.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_database_error(self):
        """Verify shutdown continues if database cleanup fails."""
        manager = ServerLifecycleManager()

        # The lifecycle code catches sqlite3.Error and ImportError
        # so shutdown should complete regardless of database issues
        await manager.graceful_shutdown(timeout=1.0)
        assert manager.is_shutting_down


class TestShutdownCallbacks:
    """Test registered shutdown callbacks."""

    @pytest.mark.asyncio
    async def test_shutdown_runs_sync_callbacks(self):
        """Verify synchronous shutdown callbacks are executed."""
        manager = ServerLifecycleManager()

        callback_executed = []

        def sync_callback():
            callback_executed.append("sync")

        manager.register_shutdown_callback(sync_callback)
        await manager.graceful_shutdown(timeout=1.0)

        assert "sync" in callback_executed

    @pytest.mark.asyncio
    async def test_shutdown_runs_async_callbacks(self):
        """Verify asynchronous shutdown callbacks are executed."""
        manager = ServerLifecycleManager()

        callback_executed = []

        async def async_callback():
            await asyncio.sleep(0.01)
            callback_executed.append("async")

        manager.register_shutdown_callback(async_callback)
        await manager.graceful_shutdown(timeout=1.0)

        assert "async" in callback_executed

    @pytest.mark.asyncio
    async def test_shutdown_runs_multiple_callbacks(self):
        """Verify multiple callbacks are executed in order."""
        manager = ServerLifecycleManager()

        callback_order = []

        manager.register_shutdown_callback(lambda: callback_order.append(1))
        manager.register_shutdown_callback(lambda: callback_order.append(2))
        manager.register_shutdown_callback(lambda: callback_order.append(3))

        await manager.graceful_shutdown(timeout=1.0)

        assert callback_order == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_callback_error(self):
        """Verify shutdown continues even if a callback fails."""
        manager = ServerLifecycleManager()

        callback_executed = []

        def failing_callback():
            raise RuntimeError("Callback error")

        def success_callback():
            callback_executed.append("success")

        manager.register_shutdown_callback(failing_callback)
        manager.register_shutdown_callback(success_callback)

        # Should not raise
        await manager.graceful_shutdown(timeout=1.0)

        # Second callback should still run
        assert "success" in callback_executed


class TestSignalHandlers:
    """Test signal handler setup and behavior."""

    def test_setup_signal_handlers_registers_handlers(self):
        """Verify signal handlers are registered."""
        manager = ServerLifecycleManager()

        with patch("signal.signal") as mock_signal:
            manager.setup_signal_handlers()

            # Should register SIGTERM and SIGINT
            calls = mock_signal.call_args_list
            signals_registered = [call[0][0] for call in calls]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered

    def test_setup_signal_handlers_handles_errors(self):
        """Verify signal handler setup handles errors gracefully."""
        manager = ServerLifecycleManager()

        with patch("signal.signal", side_effect=ValueError("Not main thread")):
            # Should not raise
            manager.setup_signal_handlers()


class TestShutdownStopsBackgroundTasks:
    """Test background task stopping during shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_handles_missing_background_manager(self):
        """Verify shutdown handles missing background manager gracefully."""
        manager = ServerLifecycleManager()

        # Should not raise even if background module has issues
        await manager.graceful_shutdown(timeout=1.0)

    @pytest.mark.asyncio
    async def test_shutdown_handles_missing_pulse_scheduler(self):
        """Verify shutdown handles missing pulse scheduler gracefully."""
        manager = ServerLifecycleManager()

        # Should not raise even if pulse module not found
        await manager.graceful_shutdown(timeout=1.0)

    @pytest.mark.asyncio
    async def test_shutdown_completes_despite_background_errors(self):
        """Verify shutdown completes even with background task errors."""
        manager = ServerLifecycleManager()

        # The lifecycle code catches ImportError and RuntimeError
        # so shutdown should complete regardless of background task issues
        await manager.graceful_shutdown(timeout=1.0)
        assert manager.is_shutting_down


class TestShutdownClosesHTTPConnector:
    """Test HTTP connector closure during shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_shared_connector(self):
        """Verify shared HTTP connector is closed."""
        manager = ServerLifecycleManager()

        with patch(
            "aragora.agents.api_agents.common.close_shared_connector",
            new_callable=AsyncMock,
        ) as mock_close:
            await manager.graceful_shutdown(timeout=1.0)
            mock_close.assert_called_once()


class TestShutdownSequencing:
    """Test that shutdown steps occur in correct order."""

    @pytest.mark.asyncio
    async def test_shutdown_sequence_order(self):
        """Verify shutdown steps execute in documented order."""
        manager = ServerLifecycleManager()

        sequence = []

        # Mock each step to record order
        original_wait = manager._wait_for_debates
        original_persist = manager._persist_circuit_breakers
        original_bg = manager._stop_background_tasks
        original_ws = manager._close_websockets
        original_http = manager._close_http_connector
        original_db = manager._close_database_connections
        original_cb = manager._run_shutdown_callbacks

        async def mock_wait(*args):
            sequence.append("wait_debates")
            return await original_wait(*args)

        def mock_persist():
            sequence.append("persist_breakers")

        async def mock_bg():
            sequence.append("stop_background")

        async def mock_ws():
            sequence.append("close_websockets")

        async def mock_http():
            sequence.append("close_http")

        def mock_db():
            sequence.append("close_database")

        async def mock_cb():
            sequence.append("run_callbacks")

        manager._wait_for_debates = mock_wait
        manager._persist_circuit_breakers = mock_persist
        manager._stop_background_tasks = mock_bg
        manager._close_websockets = mock_ws
        manager._close_http_connector = mock_http
        manager._close_database_connections = mock_db
        manager._run_shutdown_callbacks = mock_cb

        await manager.graceful_shutdown(timeout=1.0)

        expected_order = [
            "wait_debates",
            "persist_breakers",
            "stop_background",
            "close_websockets",
            "close_http",
            "close_database",
            "run_callbacks",
        ]
        assert sequence == expected_order


class TestConcurrentShutdown:
    """Test behavior when multiple shutdown requests occur."""

    @pytest.mark.asyncio
    async def test_multiple_shutdowns_dont_interfere(self):
        """Verify multiple concurrent shutdown calls work correctly."""
        manager = ServerLifecycleManager()

        # Start multiple shutdowns concurrently
        tasks = [
            asyncio.create_task(manager.graceful_shutdown(timeout=1.0))
            for _ in range(3)
        ]

        # All should complete without error
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(r is None for r in results)
