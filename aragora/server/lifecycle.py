"""
Server lifecycle management - startup/shutdown coordination.

This module handles:
- Graceful shutdown with resource cleanup
- Signal handler setup
- Component shutdown sequencing
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sqlite3
import time
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.stream.server import GauntletStreamServer

logger = logging.getLogger(__name__)


class ServerLifecycleManager:
    """Manages server startup and shutdown lifecycle.

    Coordinates graceful shutdown of all server components in the correct
    order to prevent data loss and connection leaks.

    Usage:
        lifecycle = ServerLifecycleManager(stream_server=ws_server)
        lifecycle.setup_signal_handlers()

        # During shutdown:
        await lifecycle.graceful_shutdown()
    """

    def __init__(
        self,
        stream_server: Optional["GauntletStreamServer"] = None,
        get_active_debates: Optional[Callable] = None,
    ):
        """Initialize lifecycle manager.

        Args:
            stream_server: WebSocket server to shut down
            get_active_debates: Function to get active debates dict
        """
        self.stream_server = stream_server
        self._get_active_debates = get_active_debates
        self._shutting_down = False
        self._shutdown_callbacks: list[Callable] = []

    @property
    def is_shutting_down(self) -> bool:
        """Check if server is in shutdown mode."""
        return self._shutting_down

    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to run during shutdown.

        Args:
            callback: Async or sync callable to run during shutdown
        """
        self._shutdown_callbacks.append(callback)

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            signame = signal.Signals(signum).name
            logger.info(f"Received {signame}, initiating graceful shutdown...")
            asyncio.create_task(self.graceful_shutdown())

        # Register handlers for common termination signals
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.debug("Signal handlers registered for graceful shutdown")
        except (ValueError, OSError) as e:
            # Signal handling may not work in all contexts (e.g., non-main thread)
            logger.debug(f"Could not register signal handlers: {e}")

    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shut down the server.

        Steps:
        1. Stop accepting new debates
        2. Wait for in-flight debates to complete (with timeout)
        3. Persist circuit breaker states
        4. Stop background tasks
        5. Close WebSocket connections
        6. Close shared HTTP connector
        7. Close database connections (connection pool cleanup)
        8. Run registered shutdown callbacks

        Args:
            timeout: Maximum seconds to wait for in-flight debates
        """
        logger.info("Starting graceful shutdown...")
        shutdown_start = time.time()

        # 1. Stop accepting new debates by setting flag
        self._shutting_down = True

        # 2. Wait for in-flight debates to complete
        await self._wait_for_debates(timeout)

        # 3. Persist circuit breaker states
        self._persist_circuit_breakers()

        # 4. Stop background tasks
        await self._stop_background_tasks()

        # 5. Close WebSocket connections
        await self._close_websockets()

        # 6. Close shared HTTP connector
        await self._close_http_connector()

        # 7. Close database connections
        self._close_database_connections()

        # 8. Run registered shutdown callbacks
        await self._run_shutdown_callbacks()

        elapsed = time.time() - shutdown_start
        logger.info(f"Graceful shutdown completed in {elapsed:.1f}s")

    async def _wait_for_debates(self, timeout: float) -> None:
        """Wait for in-flight debates to complete."""
        if not self._get_active_debates:
            return

        logger.info("Waiting for in-flight debates to complete...")
        active_debates = self._get_active_debates()
        if not active_debates:
            return

        in_progress = [
            d_id for d_id, d in active_debates.items()
            if d.get("status") == "in_progress"
        ]
        if not in_progress:
            return

        logger.info(f"Waiting for {len(in_progress)} in-flight debate(s)")
        wait_start = time.time()

        while time.time() - wait_start < timeout:
            active_debates = self._get_active_debates()
            still_running = sum(
                1 for d_id in in_progress
                if d_id in active_debates and
                active_debates.get(d_id, {}).get("status") == "in_progress"
            )
            if still_running == 0:
                logger.info("All in-flight debates completed")
                return
            await asyncio.sleep(1)

        logger.warning(f"Shutdown timeout reached with debates still running")

    def _persist_circuit_breakers(self) -> None:
        """Persist circuit breaker states."""
        try:
            from aragora.resilience import persist_all_circuit_breakers
            count = persist_all_circuit_breakers()
            if count > 0:
                logger.info(f"Persisted {count} circuit breaker state(s)")
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Failed to persist circuit breaker states: {e}")

    async def _stop_background_tasks(self) -> None:
        """Stop background task manager."""
        try:
            from aragora.server.background import get_background_manager
            background_mgr = get_background_manager()
            background_mgr.stop()
            logger.info("Background tasks stopped")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"Background task shutdown: {e}")

        # Stop pulse scheduler if running
        try:
            from aragora.server.handlers.pulse import get_pulse_scheduler
            scheduler = get_pulse_scheduler()
            if scheduler and scheduler.state.value != "stopped":
                await scheduler.stop(graceful=True)
                logger.info("Pulse scheduler stopped")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"Pulse scheduler shutdown: {e}")

    async def _close_websockets(self) -> None:
        """Close WebSocket connections."""
        if not self.stream_server:
            return

        try:
            await self.stream_server.graceful_shutdown()
            logger.info("WebSocket connections closed")
        except (OSError, RuntimeError, asyncio.CancelledError) as e:
            logger.warning(f"WebSocket shutdown error: {e}")

    async def _close_http_connector(self) -> None:
        """Close shared HTTP connector."""
        try:
            from aragora.agents.api_agents.common import close_shared_connector
            await close_shared_connector()
            logger.info("Shared HTTP connector closed")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Connector shutdown: {e}")

    def _close_database_connections(self) -> None:
        """Close database connections."""
        try:
            from aragora.storage.schema import DatabaseManager
            DatabaseManager.clear_instances()
            logger.info("Database connections closed")
        except (ImportError, sqlite3.Error) as e:
            logger.debug(f"Database shutdown: {e}")

    async def _run_shutdown_callbacks(self) -> None:
        """Run registered shutdown callbacks."""
        for callback in self._shutdown_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Shutdown callback error: {e}")


__all__ = ["ServerLifecycleManager"]
