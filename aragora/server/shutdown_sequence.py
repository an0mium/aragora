"""
Graceful shutdown orchestration for the unified server.

Provides a structured, phase-based approach to server shutdown with:
- Individual phase isolation (errors in one phase don't affect others)
- Consistent logging and error handling
- Configurable timeouts per phase
- Easy extension for new shutdown requirements
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List

logger = logging.getLogger(__name__)


@dataclass
class ShutdownPhase:
    """A single phase in the shutdown sequence.

    Attributes:
        name: Human-readable name for logging
        execute: Async function to execute (or sync wrapped in coroutine)
        timeout: Maximum seconds for this phase
        critical: If True, log as warning on failure; else debug
    """

    name: str
    execute: Callable[[], Coroutine[Any, Any, None]]
    timeout: float = 5.0
    critical: bool = False


class ShutdownSequence:
    """Orchestrates multi-phase graceful shutdown.

    Executes shutdown phases in order, isolating failures and
    providing consistent logging throughout.

    Usage:
        sequence = ShutdownSequence()
        sequence.add_phase(ShutdownPhase(
            name="Stop background tasks",
            execute=stop_background_tasks,
            timeout=5.0,
        ))
        await sequence.execute_all()
    """

    def __init__(self):
        """Initialize an empty shutdown sequence."""
        self._phases: List[ShutdownPhase] = []
        self._completed: List[str] = []
        self._failed: List[str] = []

    def add_phase(self, phase: ShutdownPhase) -> "ShutdownSequence":
        """Add a phase to the shutdown sequence.

        Args:
            phase: The shutdown phase to add

        Returns:
            Self for method chaining
        """
        self._phases.append(phase)
        return self

    async def execute_all(self, overall_timeout: float = 30.0) -> dict:
        """Execute all shutdown phases in order.

        Args:
            overall_timeout: Maximum total time for all phases

        Returns:
            Dict with 'completed', 'failed', and 'elapsed' keys
        """
        start_time = time.time()
        logger.info(f"Starting graceful shutdown ({len(self._phases)} phases)...")

        for phase in self._phases:
            elapsed = time.time() - start_time
            if elapsed >= overall_timeout:
                logger.warning("Overall shutdown timeout reached, skipping remaining phases")
                break

            remaining = overall_timeout - elapsed
            phase_timeout = min(phase.timeout, remaining)

            await self._execute_phase(phase, phase_timeout)

        elapsed = time.time() - start_time
        logger.info(
            f"Graceful shutdown completed in {elapsed:.1f}s "
            f"({len(self._completed)} succeeded, {len(self._failed)} failed)"
        )

        return {
            "completed": self._completed.copy(),
            "failed": self._failed.copy(),
            "elapsed": elapsed,
        }

    async def _execute_phase(self, phase: ShutdownPhase, timeout: float) -> bool:
        """Execute a single shutdown phase with error isolation.

        Args:
            phase: The phase to execute
            timeout: Maximum seconds for this phase

        Returns:
            True if phase completed successfully
        """
        try:
            await asyncio.wait_for(phase.execute(), timeout=timeout)
            self._completed.append(phase.name)
            logger.debug(f"Shutdown phase completed: {phase.name}")
            return True

        except asyncio.TimeoutError:
            self._failed.append(phase.name)
            if phase.critical:
                logger.warning(f"Shutdown phase timed out: {phase.name}")
            else:
                logger.debug(f"Shutdown phase timed out: {phase.name}")
            return False

        except asyncio.CancelledError:
            self._failed.append(phase.name)
            logger.debug(f"Shutdown phase cancelled: {phase.name}")
            return False

        except Exception as e:
            self._failed.append(phase.name)
            if phase.critical:
                logger.warning(f"Shutdown phase failed: {phase.name}: {e}")
            else:
                logger.debug(f"Shutdown phase failed: {phase.name}: {e}")
            return False


def create_server_shutdown_sequence(server: Any) -> ShutdownSequence:
    """Create the standard shutdown sequence for UnifiedServer.

    Args:
        server: The UnifiedServer instance

    Returns:
        Configured ShutdownSequence ready to execute
    """
    sequence = ShutdownSequence()

    # Phase 1: Mark server as shutting down
    async def set_shutting_down():
        server._shutting_down = True

    sequence.add_phase(
        ShutdownPhase(
            name="Set shutdown flag",
            execute=set_shutting_down,
            timeout=1.0,
        )
    )

    # Phase 2: Wait for in-flight debates
    async def wait_for_debates():
        from aragora.server.debate_utils import get_active_debates

        active_debates = get_active_debates()
        if not active_debates:
            return

        in_progress = [
            d_id for d_id, d in active_debates.items() if d.get("status") == "in_progress"
        ]
        if not in_progress:
            return

        logger.info(f"Waiting for {len(in_progress)} in-flight debate(s)")
        wait_start = time.time()
        while time.time() - wait_start < 25.0:  # Leave buffer for other phases
            still_running = sum(
                1
                for d_id in in_progress
                if d_id in active_debates
                and active_debates.get(d_id, {}).get("status") == "in_progress"
            )
            if still_running == 0:
                logger.info("All in-flight debates completed")
                return
            await asyncio.sleep(1)

        logger.warning("Some debates still running after timeout")

    sequence.add_phase(
        ShutdownPhase(
            name="Wait for in-flight debates",
            execute=wait_for_debates,
            timeout=26.0,
            critical=True,
        )
    )

    # Phase 3: Persist circuit breaker states
    async def persist_circuit_breakers():
        from aragora.resilience import persist_all_circuit_breakers

        count = persist_all_circuit_breakers()
        if count > 0:
            logger.info(f"Persisted {count} circuit breaker state(s)")

    sequence.add_phase(
        ShutdownPhase(
            name="Persist circuit breakers",
            execute=persist_circuit_breakers,
            timeout=5.0,
        )
    )

    # Phase 4: Flush KM adapters and event batches
    async def flush_km_adapters():
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()

            # Flush event batches first
            flushed = manager.flush_all_batches()
            if flushed > 0:
                logger.info(f"Flushed {flushed} batched events")

            # Sync RankingAdapter state
            try:
                from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

                ranking_adapter = getattr(manager, "_ranking_adapter", None)
                if ranking_adapter is not None:
                    stats = ranking_adapter.get_stats()
                    if stats.get("total_expertise_records", 0) > 0:
                        logger.debug(
                            f"RankingAdapter has {stats['total_expertise_records']} expertise records"
                        )
            except ImportError:
                pass

            # Sync RlmAdapter state
            try:
                from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

                rlm_adapter = getattr(manager, "_rlm_adapter", None)
                if rlm_adapter is not None:
                    stats = rlm_adapter.get_stats()
                    if stats.get("total_patterns", 0) > 0:
                        logger.debug(
                            f"RlmAdapter has {stats['total_patterns']} compression patterns"
                        )
            except ImportError:
                pass

            logger.info("KM adapters and event batches flushed")

        except ImportError:
            pass  # Cross-subscriber module not available
        except Exception as e:
            logger.warning(f"KM adapter flush failed: {e}")

    sequence.add_phase(
        ShutdownPhase(
            name="Flush KM adapters",
            execute=flush_km_adapters,
            timeout=5.0,
        )
    )

    # Phase 5: Shutdown OpenTelemetry tracer
    async def shutdown_tracing():
        from aragora.observability.tracing import shutdown as shutdown_otel

        shutdown_otel()
        logger.info("OpenTelemetry tracer shutdown complete")

    sequence.add_phase(
        ShutdownPhase(
            name="Shutdown OpenTelemetry",
            execute=shutdown_tracing,
            timeout=5.0,
        )
    )

    # Phase 5: Stop background tasks
    async def stop_background_tasks():
        from aragora.server.background import get_background_manager

        background_mgr = get_background_manager()
        background_mgr.stop()
        logger.info("Background tasks stopped")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop background tasks",
            execute=stop_background_tasks,
            timeout=5.0,
        )
    )

    # Phase 6: Stop pulse scheduler
    async def stop_pulse_scheduler():
        from aragora.server.handlers.pulse import get_pulse_scheduler

        scheduler = get_pulse_scheduler()
        if scheduler and scheduler.state.value != "stopped":
            await scheduler.stop(graceful=True)
            logger.info("Pulse scheduler stopped")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop pulse scheduler",
            execute=stop_pulse_scheduler,
            timeout=5.0,
        )
    )

    # Phase 7: Stop state cleanup task
    async def stop_state_cleanup():
        from aragora.server.stream.state_manager import stop_cleanup_task

        stop_cleanup_task()
        logger.debug("State cleanup task stopped")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop state cleanup",
            execute=stop_state_cleanup,
            timeout=2.0,
        )
    )

    # Phase 8: Stop stuck debate watchdog
    async def stop_watchdog():
        if hasattr(server, "_watchdog_task") and server._watchdog_task:
            server._watchdog_task.cancel()
            try:
                await asyncio.wait_for(server._watchdog_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            logger.debug("Stuck debate watchdog stopped")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop watchdog",
            execute=stop_watchdog,
            timeout=3.0,
        )
    )

    # Phase 9: Shutdown Control Plane coordinator
    async def shutdown_control_plane():
        if hasattr(server, "_control_plane_coordinator") and server._control_plane_coordinator:
            await server._control_plane_coordinator.shutdown()
            logger.info("Control Plane coordinator shutdown complete")

    sequence.add_phase(
        ShutdownPhase(
            name="Shutdown Control Plane",
            execute=shutdown_control_plane,
            timeout=5.0,
        )
    )

    # Phase 10: Close debate stream WebSocket connections
    async def close_debate_websockets():
        if hasattr(server, "stream_server") and server.stream_server:
            await server.stream_server.graceful_shutdown()
            logger.info("Debate stream WebSocket connections closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close debate WebSockets",
            execute=close_debate_websockets,
            timeout=5.0,
            critical=True,
        )
    )

    # Phase 11: Close control plane WebSocket connections
    async def close_control_plane_websockets():
        if hasattr(server, "control_plane_stream") and server.control_plane_stream:
            await server.control_plane_stream.stop()
            logger.info("Control plane WebSocket connections closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close control plane WebSockets",
            execute=close_control_plane_websockets,
            timeout=5.0,
        )
    )

    # Phase 12: Close nomic loop WebSocket connections
    async def close_nomic_websockets():
        if hasattr(server, "nomic_loop_stream") and server.nomic_loop_stream:
            await server.nomic_loop_stream.stop()
            logger.info("Nomic loop WebSocket connections closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close nomic loop WebSockets",
            execute=close_nomic_websockets,
            timeout=5.0,
        )
    )

    # Phase 13: Close shared HTTP connector
    async def close_http_connector():
        from aragora.agents.api_agents.common import close_shared_connector

        await close_shared_connector()
        logger.info("Shared HTTP connector closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close HTTP connector",
            execute=close_http_connector,
            timeout=5.0,
        )
    )

    # Phase 14: Stop RBAC distributed cache
    async def stop_rbac_cache():
        try:
            from aragora.rbac.cache import get_rbac_cache, reset_rbac_cache

            cache = get_rbac_cache()
            if cache:
                cache.stop()
                reset_rbac_cache()  # Clear singleton for clean restart
                logger.debug("RBAC distributed cache stopped")
        except ImportError:
            pass  # RBAC module not available
        except Exception as e:
            logger.debug(f"RBAC cache stop skipped: {e}")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop RBAC cache",
            execute=stop_rbac_cache,
            timeout=2.0,
        )
    )

    # Phase 15: Close Redis connection pool
    async def close_redis():
        from aragora.server.redis_config import close_redis_pool

        close_redis_pool()
        logger.debug("Redis connection pool closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close Redis pool",
            execute=close_redis,
            timeout=2.0,
        )
    )

    # Phase 16: Stop auth cleanup thread
    async def stop_auth_cleanup():
        from aragora.server.auth import auth_config

        auth_config.stop_cleanup_thread()
        logger.debug("Auth cleanup thread stopped")

    sequence.add_phase(
        ShutdownPhase(
            name="Stop auth cleanup",
            execute=stop_auth_cleanup,
            timeout=2.0,
        )
    )

    # Phase 17: Close database connections
    async def close_databases():
        from aragora.storage.schema import DatabaseManager

        DatabaseManager.clear_instances()
        logger.info("Database connections closed")

    sequence.add_phase(
        ShutdownPhase(
            name="Close databases",
            execute=close_databases,
            timeout=5.0,
            critical=True,
        )
    )

    return sequence
