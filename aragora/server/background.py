"""
Background task management for the Aragora server.

Provides periodic task execution for maintenance operations like:
- Memory tier cleanup (TTL enforcement)
- Stale debate cleanup
- Cache pruning
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a background task."""
    name: str
    interval_seconds: float
    callback: Callable[[], Any]
    enabled: bool = True
    run_on_startup: bool = False
    last_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0


class BackgroundTaskManager:
    """
    Manages periodic background tasks for server maintenance.

    Thread-safe manager that runs tasks at configured intervals.
    Integrates with the server lifecycle via start/stop methods.

    Usage:
        manager = BackgroundTaskManager()
        manager.register_task(
            name="memory_cleanup",
            interval_seconds=3600,  # 1 hour
            callback=cleanup_memory,
        )
        manager.start()
        # ... server runs ...
        manager.stop()
    """

    def __init__(self):
        self._tasks: Dict[str, TaskConfig] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register_task(
        self,
        name: str,
        interval_seconds: float,
        callback: Callable[[], Any],
        enabled: bool = True,
        run_on_startup: bool = False,
    ) -> None:
        """
        Register a periodic background task.

        Args:
            name: Unique task identifier
            interval_seconds: How often to run (in seconds)
            callback: Function to call (should be quick and non-blocking)
            enabled: Whether task is active
            run_on_startup: Run immediately when manager starts
        """
        with self._lock:
            self._tasks[name] = TaskConfig(
                name=name,
                interval_seconds=interval_seconds,
                callback=callback,
                enabled=enabled,
                run_on_startup=run_on_startup,
            )
            logger.info(
                "Registered background task: %s (interval=%ds, enabled=%s)",
                name, interval_seconds, enabled
            )

    def unregister_task(self, name: str) -> bool:
        """Remove a registered task."""
        with self._lock:
            if name in self._tasks:
                del self._tasks[name]
                logger.info("Unregistered background task: %s", name)
                return True
            return False

    def enable_task(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a task."""
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = enabled
                logger.info("Task %s enabled=%s", name, enabled)
                return True
            return False

    def start(self) -> None:
        """Start the background task runner."""
        if self._running:
            logger.warning("BackgroundTaskManager already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="BackgroundTaskManager",
        )
        self._thread.start()
        logger.info("BackgroundTaskManager started")

        # Run startup tasks
        with self._lock:
            for task in self._tasks.values():
                if task.enabled and task.run_on_startup:
                    self._execute_task(task)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background task runner."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        logger.info("BackgroundTaskManager stopped")

    def _run_loop(self) -> None:
        """Main loop that checks and runs tasks."""
        check_interval = 10.0  # Check every 10 seconds

        while not self._stop_event.is_set():
            now = time.time()

            with self._lock:
                tasks_to_run = []
                for task in self._tasks.values():
                    if not task.enabled:
                        continue
                    if task.last_run is None:
                        # Never run before - check if startup already ran it
                        if not task.run_on_startup or task.run_count == 0:
                            tasks_to_run.append(task)
                    elif now - task.last_run >= task.interval_seconds:
                        tasks_to_run.append(task)

            for task in tasks_to_run:
                self._execute_task(task)

            # Wait with interruptible sleep
            self._stop_event.wait(timeout=check_interval)

    def _execute_task(self, task: TaskConfig) -> None:
        """Execute a single task with error handling."""
        try:
            logger.debug("Running background task: %s", task.name)
            start = time.time()
            task.callback()
            duration = time.time() - start

            task.last_run = time.time()
            task.run_count += 1

            logger.info(
                "Background task %s completed (duration=%.2fs, runs=%d)",
                task.name, duration, task.run_count
            )
        except Exception as e:
            task.error_count += 1
            logger.error(
                "Background task %s failed (errors=%d): %s",
                task.name, task.error_count, e,
                exc_info=True
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about background tasks."""
        with self._lock:
            return {
                "running": self._running,
                "task_count": len(self._tasks),
                "tasks": {
                    name: {
                        "enabled": task.enabled,
                        "interval_seconds": task.interval_seconds,
                        "last_run": task.last_run,
                        "run_count": task.run_count,
                        "error_count": task.error_count,
                        "seconds_until_next": (
                            max(0, task.interval_seconds - (time.time() - task.last_run))
                            if task.last_run else 0
                        ),
                    }
                    for name, task in self._tasks.items()
                },
            }

    def run_task_now(self, name: str) -> bool:
        """Manually trigger a task to run immediately."""
        with self._lock:
            task = self._tasks.get(name)
            if not task:
                return False

        self._execute_task(task)
        return True


# Singleton instance
_background_manager: Optional[BackgroundTaskManager] = None
_manager_lock = threading.Lock()


def get_background_manager() -> BackgroundTaskManager:
    """Get the singleton BackgroundTaskManager instance."""
    global _background_manager

    with _manager_lock:
        if _background_manager is None:
            _background_manager = BackgroundTaskManager()

    return _background_manager


def setup_default_tasks(
    nomic_dir: Optional[str] = None,
    memory_instance: Optional[Any] = None,
    pressure_threshold: float = 0.8,
) -> None:
    """
    Register default background tasks.

    Called during server startup to set up maintenance tasks.

    Args:
        nomic_dir: Path to nomic directory (for memory cleanup)
        memory_instance: Optional shared ContinuumMemory instance to use.
            If provided, cleanup uses this instance instead of creating a new one.
            This enables pressure-aware cleanup on the actual server memory.
        pressure_threshold: Trigger cleanup when memory pressure exceeds this
            threshold (0.0-1.0). Default 0.8 means cleanup at 80% capacity.
    """
    manager = get_background_manager()

    # Store shared memory reference
    _shared_memory = memory_instance

    # Memory tier cleanup - runs every 6 hours
    def memory_cleanup_task():
        nonlocal _shared_memory
        try:
            # Use shared instance if provided
            if _shared_memory is not None:
                memory = _shared_memory
            else:
                from aragora.memory.continuum import ContinuumMemory
                from aragora.config import DB_MEMORY_PATH

                # Use provided path or default
                db_path = nomic_dir + "/continuum_memory.db" if nomic_dir else DB_MEMORY_PATH
                memory = ContinuumMemory(db_path=db_path)

            # Check memory pressure before cleanup
            pressure = memory.get_memory_pressure()
            if pressure < pressure_threshold:
                logger.debug(
                    "Memory pressure %.1f%% below threshold %.1f%%, skipping cleanup",
                    pressure * 100, pressure_threshold * 100
                )
                return

            logger.info(
                "Memory pressure %.1f%% exceeds threshold, running cleanup",
                pressure * 100
            )

            result = memory.cleanup_expired_memories(archive=True)

            if result["deleted"] > 0 or result["archived"] > 0:
                logger.info(
                    "Memory cleanup: archived=%d, deleted=%d",
                    result["archived"], result["deleted"]
                )
        except ImportError:
            logger.debug("ContinuumMemory not available, skipping cleanup")
        except Exception as e:
            logger.warning("Memory cleanup failed: %s", e)

    manager.register_task(
        name="memory_tier_cleanup",
        interval_seconds=6 * 3600,  # 6 hours
        callback=memory_cleanup_task,
        enabled=True,
        run_on_startup=False,  # Don't run on startup (let things settle first)
    )

    # Stale debate cleanup - runs every 30 minutes
    def stale_debate_cleanup():
        try:
            from aragora.server.state import get_state_manager
            state_mgr = get_state_manager()
            cleaned = state_mgr.cleanup_stale_debates(max_age_seconds=3600)  # 1 hour max
            if cleaned > 0:
                logger.info("Cleaned up %d stale debates", cleaned)
        except Exception as e:
            logger.warning("Stale debate cleanup failed: %s", e)

    manager.register_task(
        name="stale_debate_cleanup",
        interval_seconds=30 * 60,  # 30 minutes
        callback=stale_debate_cleanup,
        enabled=True,
        run_on_startup=False,
    )

    # Circuit breaker cleanup - runs every hour
    def circuit_breaker_cleanup():
        try:
            from aragora.resilience import prune_circuit_breakers
            cleaned = prune_circuit_breakers()
            if cleaned > 0:
                logger.info("Cleaned up %d stale circuit breakers", cleaned)
        except ImportError:
            pass  # resilience module may not be available
        except Exception as e:
            logger.warning("Circuit breaker cleanup failed: %s", e)

    manager.register_task(
        name="circuit_breaker_cleanup",
        interval_seconds=3600,  # 1 hour
        callback=circuit_breaker_cleanup,
        enabled=True,
        run_on_startup=False,
    )

    logger.info("Default background tasks registered")
