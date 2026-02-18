"""
Server startup background task initialization.

This module handles circuit breaker persistence, background tasks,
state cleanup, stuck debate watchdog, and pulse scheduler initialization.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def init_circuit_breaker_persistence(nomic_dir: Path | None) -> int:
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


def init_background_tasks(nomic_dir: Path | None) -> bool:
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


async def init_pulse_scheduler(stream_emitter: Any | None = None) -> bool:
    """Initialize auto-start pulse scheduler if configured.

    Args:
        stream_emitter: Optional event emitter for debates

    Returns:
        True if scheduler was started, False otherwise
    """
    try:
        import os

        PULSE_SCHEDULER_AUTOSTART = (
            os.environ.get("PULSE_SCHEDULER_AUTOSTART", "true").lower() == "true"
        )
        PULSE_SCHEDULER_MAX_PER_HOUR = int(os.environ.get("PULSE_SCHEDULER_MAX_PER_HOUR", "6"))
        PULSE_SCHEDULER_POLL_INTERVAL = int(os.environ.get("PULSE_SCHEDULER_POLL_INTERVAL", "300"))

        if not PULSE_SCHEDULER_AUTOSTART:
            return False

        from aragora.server.handlers.pulse import get_pulse_scheduler

        scheduler = get_pulse_scheduler()
        if not scheduler:
            logger.debug("Pulse scheduler not available for autostart")
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
            except (RuntimeError, ValueError, OSError, ImportError) as e:
                logger.error(f"Auto-scheduled debate failed: {e}")
                return None

        scheduler.set_debate_creator(auto_create_debate)
        task = asyncio.create_task(scheduler.start())
        task.add_done_callback(
            lambda t: logger.error("Pulse scheduler crashed: %s", t.exception())
            if not t.cancelled() and t.exception()
            else None
        )
        logger.info("Pulse scheduler auto-started (PULSE_SCHEDULER_AUTOSTART=true)")
        return True

    except ImportError as e:
        logger.debug(f"Pulse scheduler autostart not available: {e}")
    except (RuntimeError, OSError, ValueError, AttributeError) as e:
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


async def init_stuck_debate_watchdog() -> asyncio.Task | None:
    """Start stuck debate watchdog.

    Returns:
        The watchdog task if started, None otherwise
    """
    try:
        from aragora.server.debate_utils import watchdog_stuck_debates

        task = asyncio.create_task(watchdog_stuck_debates())
        task.add_done_callback(
            lambda t: logger.error("Stuck debate watchdog crashed: %s", t.exception())
            if not t.cancelled() and t.exception()
            else None
        )
        logger.info("Stuck debate watchdog started (10 min timeout)")
        return task
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Stuck debate watchdog not started: {e}")
        return None


async def init_titans_memory_sweep() -> asyncio.Task | None:
    """Start the Titans memory sweep loop as a background task.

    The sweep controller periodically evaluates memory items using
    RetentionGate (retain/forget/consolidate/demote) and fires
    reactive triggers on memory events.

    Environment:
        ARAGORA_MEMORY_SWEEP_ENABLED: "true" to enable (default: "false")
        ARAGORA_MEMORY_SWEEP_INTERVAL: Seconds between sweeps (default: 300)

    Returns:
        The sweep task if started, None otherwise
    """
    import os

    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "ARAGORA_TEST_ENABLE_BACKGROUND_TASKS"
    ):
        return None

    if os.environ.get("ARAGORA_MEMORY_SWEEP_ENABLED", "false").lower() not in (
        "true",
        "1",
        "yes",
    ):
        logger.debug("Titans memory sweep disabled (set ARAGORA_MEMORY_SWEEP_ENABLED=true)")
        return None

    try:
        from aragora.memory.titans_controller import TitansMemoryController
        from aragora.memory.triggers import MemoryTriggerEngine

        interval = int(os.environ.get("ARAGORA_MEMORY_SWEEP_INTERVAL", "300"))
        trigger_engine = MemoryTriggerEngine()
        controller = TitansMemoryController(trigger_engine=trigger_engine)

        task = asyncio.create_task(
            controller.run_sweep_loop(interval_seconds=float(interval))
        )
        task.add_done_callback(
            lambda t: logger.error("Titans memory sweep crashed: %s", t.exception())
            if not t.cancelled() and t.exception()
            else None
        )
        logger.info(
            "Titans memory sweep started (interval=%ds, triggers=%d)",
            interval,
            len(trigger_engine.list_triggers()),
        )
        return task

    except ImportError as e:
        logger.debug("Titans memory sweep not available: %s", e)
        return None
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Failed to start Titans memory sweep: %s", e)
        return None


async def init_slack_token_refresh_scheduler() -> asyncio.Task | None:
    """Start Slack token refresh scheduler.

    Proactively refreshes Slack OAuth tokens 1 hour before expiry
    to prevent service interruption.

    Returns:
        The scheduler task if started, None otherwise
    """
    import os

    # Check if Slack integration is enabled
    if not os.environ.get("SLACK_CLIENT_ID") or not os.environ.get("SLACK_CLIENT_SECRET"):
        logger.debug("Slack token refresh scheduler not started: missing client credentials")
        return None

    try:
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        store = get_slack_workspace_store()
        client_id = os.environ["SLACK_CLIENT_ID"]
        client_secret = os.environ["SLACK_CLIENT_SECRET"]

        async def _refresh_expiring_tokens():
            """Background task to refresh expiring Slack tokens."""
            refresh_interval = 3600  # Check every hour
            lookahead_hours = 1  # Refresh tokens expiring in the next hour

            while True:
                try:
                    # Get workspaces with tokens expiring soon
                    expiring = store.get_expiring_tokens(hours=lookahead_hours)

                    if expiring:
                        logger.info(f"Found {len(expiring)} Slack tokens expiring soon")

                    for workspace in expiring:
                        try:
                            result = await store.refresh_workspace_token(
                                workspace.workspace_id,
                                client_id,
                                client_secret,
                            )
                            if result:
                                logger.info(
                                    f"Refreshed Slack token for workspace {workspace.workspace_id}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to refresh Slack token for {workspace.workspace_id}"
                                )
                        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                            logger.error(
                                f"Error refreshing token for {workspace.workspace_id}: {e}"
                            )

                except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                    logger.error(f"Error in Slack token refresh scheduler: {e}")

                await asyncio.sleep(refresh_interval)

        task = asyncio.create_task(_refresh_expiring_tokens())
        task.add_done_callback(
            lambda t: logger.error("Slack token refresh scheduler crashed: %s", t.exception())
            if not t.cancelled() and t.exception()
            else None
        )
        logger.info("Slack token refresh scheduler started (1 hour interval)")
        return task

    except ImportError as e:
        logger.debug(f"Slack token refresh scheduler not available: {e}")
        return None
    except (RuntimeError, OSError) as e:
        logger.warning(f"Failed to start Slack token refresh scheduler: {e}")
        return None
