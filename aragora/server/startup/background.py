"""
Server startup background task initialization.

This module handles circuit breaker persistence, background tasks,
state cleanup, stuck debate watchdog, and pulse scheduler initialization.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def init_circuit_breaker_persistence(nomic_dir: Optional[Path]) -> int:
    """Initialize circuit breaker persistence.

    Args:
        nomic_dir: Path to nomic directory for database storage

    Returns:
        Number of circuit breaker states restored
    """
    try:
        from aragora.resilience import (
            init_circuit_breaker_persistence as _init_cb_persistence,
            load_circuit_breakers,
        )

        from aragora.persistence.db_config import get_nomic_dir

        data_dir = nomic_dir or get_nomic_dir()
        db_path = str(data_dir / "circuit_breaker.db")
        _init_cb_persistence(db_path)
        loaded = load_circuit_breakers()
        if loaded > 0:
            logger.info(f"Restored {loaded} circuit breaker states from disk")
        return loaded
    except (ImportError, OSError, RuntimeError) as e:
        logger.debug(f"Circuit breaker persistence not available: {e}")
        return 0


def init_background_tasks(nomic_dir: Optional[Path]) -> bool:
    """Initialize background task manager.

    Args:
        nomic_dir: Path to nomic directory

    Returns:
        True if background tasks were started, False otherwise
    """
    import os

    if os.environ.get("ARAGORA_DISABLE_BACKGROUND_TASKS") == "1":
        logger.info("Background tasks disabled via ARAGORA_DISABLE_BACKGROUND_TASKS")
        return False

    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "ARAGORA_TEST_ENABLE_BACKGROUND_TASKS"
    ):
        logger.info("Background tasks disabled during pytest run")
        return False

    try:
        from aragora.server.background import get_background_manager, setup_default_tasks

        nomic_path = str(nomic_dir) if nomic_dir else None
        setup_default_tasks(
            nomic_dir=nomic_path,
            memory_instance=None,  # Will use shared instance from get_continuum_memory()
        )
        background_mgr = get_background_manager()
        background_mgr.start()
        logger.info("Background task manager started")
        return True
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Failed to start background tasks: %s", e)
        return False


async def init_pulse_scheduler(stream_emitter: Optional[Any] = None) -> bool:
    """Initialize auto-start pulse scheduler if configured.

    Args:
        stream_emitter: Optional event emitter for debates

    Returns:
        True if scheduler was started, False otherwise
    """
    try:
        from aragora.config.legacy import (
            PULSE_SCHEDULER_AUTOSTART,
            PULSE_SCHEDULER_MAX_PER_HOUR,
            PULSE_SCHEDULER_POLL_INTERVAL,
        )

        if not PULSE_SCHEDULER_AUTOSTART:
            return False

        from aragora.server.handlers.pulse import get_pulse_scheduler

        scheduler = get_pulse_scheduler()
        if not scheduler:
            logger.warning("Pulse scheduler not available for autostart")
            return False

        # Update config from environment
        scheduler.update_config(
            {
                "poll_interval_seconds": PULSE_SCHEDULER_POLL_INTERVAL,
                "max_debates_per_hour": PULSE_SCHEDULER_MAX_PER_HOUR,
            }
        )

        # Set up debate creator callback
        async def auto_create_debate(topic_text: str, rounds: int, threshold: float):
            try:
                from aragora import Arena, DebateProtocol, Environment
                from aragora.agents import get_agents_by_names

                env = Environment(task=topic_text)
                agents = get_agents_by_names(["anthropic-api", "openai-api"])
                protocol = DebateProtocol(
                    rounds=rounds,
                    consensus="majority",
                    convergence_detection=False,
                    early_stopping=False,
                )
                if not agents:
                    return None
                arena = Arena.from_env(env, agents, protocol)
                result = await arena.run()
                return {
                    "debate_id": result.id,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                    "rounds_used": result.rounds_used,
                }
            except Exception as e:
                logger.error(f"Auto-scheduled debate failed: {e}")
                return None

        scheduler.set_debate_creator(auto_create_debate)
        asyncio.create_task(scheduler.start())
        logger.info("Pulse scheduler auto-started (PULSE_SCHEDULER_AUTOSTART=true)")
        return True

    except ImportError as e:
        logger.debug(f"Pulse scheduler autostart not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to auto-start pulse scheduler: {e}")
    return False


def init_state_cleanup_task() -> bool:
    """Start periodic state cleanup task.

    Returns:
        True if cleanup task was started, False otherwise
    """
    try:
        from aragora.server.stream.state_manager import (
            get_stream_state_manager,
            start_cleanup_task,
        )

        stream_state_manager = get_stream_state_manager()
        start_cleanup_task(stream_state_manager, interval_seconds=300)
        logger.debug("State cleanup task started (5 min interval)")
        return True
    except (ImportError, RuntimeError) as e:
        logger.debug(f"State cleanup task not started: {e}")
        return False


async def init_stuck_debate_watchdog() -> Optional[asyncio.Task]:
    """Start stuck debate watchdog.

    Returns:
        The watchdog task if started, None otherwise
    """
    try:
        from aragora.server.debate_utils import watchdog_stuck_debates

        task = asyncio.create_task(watchdog_stuck_debates())
        logger.info("Stuck debate watchdog started (10 min timeout)")
        return task
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Stuck debate watchdog not started: {e}")
        return None
