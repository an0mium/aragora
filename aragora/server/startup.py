"""
Server startup initialization tasks.

This module handles the startup sequence for the unified server,
including monitoring, tracing, background tasks, and schedulers.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def init_error_monitoring() -> bool:
    """Initialize error monitoring (Sentry).

    Returns:
        True if monitoring was enabled, False otherwise
    """
    try:
        from aragora.server.error_monitoring import init_monitoring

        if init_monitoring():
            logger.info("Error monitoring enabled (Sentry)")
            return True
    except ImportError:
        pass
    return False


async def init_opentelemetry() -> bool:
    """Initialize OpenTelemetry tracing.

    Returns:
        True if tracing was enabled, False otherwise
    """
    try:
        from aragora.observability.config import is_tracing_enabled
        from aragora.observability.tracing import get_tracer

        if is_tracing_enabled():
            get_tracer()  # Initialize tracer singleton
            logger.info("OpenTelemetry tracing enabled")
            return True
        else:
            logger.debug("OpenTelemetry tracing disabled (set OTEL_ENABLED=true to enable)")
    except ImportError as e:
        logger.debug(f"OpenTelemetry not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    return False


async def init_prometheus_metrics() -> bool:
    """Initialize Prometheus metrics server.

    Returns:
        True if metrics were enabled, False otherwise
    """
    try:
        from aragora.observability.config import is_metrics_enabled
        from aragora.observability.metrics import start_metrics_server

        if is_metrics_enabled():
            start_metrics_server()
            logger.info("Prometheus metrics server started")
            return True
    except ImportError as e:
        logger.debug(f"Prometheus metrics not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")
    return False


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

        data_dir = nomic_dir or Path(".data")
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


async def init_control_plane_coordinator() -> Optional[Any]:
    """Initialize the Control Plane coordinator.

    Creates and connects the ControlPlaneCoordinator which manages:
    - Agent registry (service discovery)
    - Task scheduler (distributed task execution)
    - Health monitor (agent health tracking)

    Returns:
        Connected ControlPlaneCoordinator, or None if initialization fails
    """
    try:
        from aragora.control_plane.coordinator import ControlPlaneCoordinator

        coordinator = await ControlPlaneCoordinator.create()
        logger.info("Control Plane coordinator initialized and connected")
        return coordinator
    except ImportError as e:
        logger.debug(f"Control Plane not available: {e}")
        return None
    except Exception as e:
        # Redis may not be available - this is OK for local development
        logger.warning(f"Control Plane coordinator not started (Redis may be unavailable): {e}")
        return None


async def init_shared_control_plane_state() -> bool:
    """Initialize the shared control plane state for the AgentDashboardHandler.

    Connects to Redis for multi-instance state sharing. Falls back to in-memory
    for single-instance deployments.

    Returns:
        True if Redis connected, False if using in-memory fallback
    """
    try:
        from aragora.control_plane.shared_state import get_shared_state, set_shared_state

        state = await get_shared_state(auto_connect=True)
        if state.is_persistent:
            logger.info("Shared control plane state connected to Redis (HA enabled)")
            return True
        else:
            logger.info("Shared control plane state using in-memory fallback (single-instance)")
            return False
    except ImportError as e:
        logger.debug(f"Shared control plane state not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Shared control plane state initialization failed: {e}")
        return False


async def init_tts_integration() -> bool:
    """Initialize TTS integration for live voice responses.

    Wires the TTS synthesis system to the event bus so that agent messages
    can automatically trigger voice synthesis for active voice sessions.

    Returns:
        True if TTS integration was initialized, False otherwise
    """
    try:
        from aragora.debate.event_bus import get_event_bus
        from aragora.server.stream.tts_integration import init_tts_integration as _init_tts

        # Initialize TTS integration with event bus
        event_bus = get_event_bus()
        integration = _init_tts(event_bus=event_bus)

        if integration.is_available:
            logger.info("TTS integration initialized (voice synthesis enabled)")
            return True
        else:
            logger.info("TTS integration initialized (voice synthesis unavailable)")
            return False

    except ImportError as e:
        logger.debug(f"TTS integration not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize TTS integration: {e}")

    return False


async def init_km_adapters() -> bool:
    """Initialize Knowledge Mound adapters from persisted state.

    Loads expertise profiles and compression patterns from KM to restore
    cross-debate learning state.

    Returns:
        True if adapters were initialized, False otherwise
    """
    try:
        from aragora.events.cross_subscribers import get_cross_subscriber_manager
        from aragora.knowledge.mound.adapters import RankingAdapter, RlmAdapter

        manager = get_cross_subscriber_manager()

        # Initialize RankingAdapter
        ranking_adapter = RankingAdapter()
        manager._ranking_adapter = ranking_adapter  # type: ignore[attr-defined]
        logger.debug("RankingAdapter initialized for KM integration")

        # Initialize RlmAdapter
        rlm_adapter = RlmAdapter()
        manager._rlm_adapter = rlm_adapter  # type: ignore[attr-defined]
        logger.debug("RlmAdapter initialized for KM integration")

        # Note: Actual KM state loading would happen here if KM backend is available
        # For now, adapters start fresh and accumulate state during debates

        logger.info("KM adapters initialized for cross-debate learning")
        return True

    except ImportError as e:
        logger.debug(f"KM adapters not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize KM adapters: {e}")

    return False


def init_workflow_checkpoint_persistence() -> bool:
    """Wire Knowledge Mound to workflow checkpoint persistence.

    This enables workflow checkpoints to be stored in KnowledgeMound rather
    than local files, providing durable persistence that survives server restarts
    and enables cross-instance checkpoint access.

    Returns:
        True if checkpoint persistence was wired to KnowledgeMound, False if
        falling back to file-based storage.
    """
    try:
        from aragora.knowledge.mound import get_knowledge_mound
        from aragora.workflow.checkpoint_store import set_default_knowledge_mound

        # Get the singleton Knowledge Mound instance
        mound = get_knowledge_mound()

        # Wire it to the checkpoint store
        set_default_knowledge_mound(mound)

        logger.info("Workflow checkpoint persistence wired to KnowledgeMound")
        return True

    except ImportError as e:
        logger.debug(f"KnowledgeMound not available for checkpoints: {e}")
    except Exception as e:
        logger.warning(f"Failed to wire checkpoint persistence: {e}")

    return False


async def run_startup_sequence(
    nomic_dir: Optional[Path] = None,
    stream_emitter: Optional[Any] = None,
) -> dict:
    """Run the full server startup sequence.

    Args:
        nomic_dir: Path to nomic directory
        stream_emitter: Optional event emitter for debates

    Returns:
        Dictionary with startup status for each component
    """
    status: dict[str, Any] = {
        "error_monitoring": False,
        "opentelemetry": False,
        "prometheus": False,
        "circuit_breakers": 0,
        "background_tasks": False,
        "pulse_scheduler": False,
        "state_cleanup": False,
        "watchdog_task": None,
        "control_plane_coordinator": None,
        "km_adapters": False,
        "workflow_checkpoint_persistence": False,
        "shared_control_plane_state": False,
        "tts_integration": False,
    }

    # Initialize in parallel where possible
    status["error_monitoring"] = await init_error_monitoring()
    status["opentelemetry"] = await init_opentelemetry()
    status["prometheus"] = await init_prometheus_metrics()

    # Sequential initialization for components with dependencies
    status["circuit_breakers"] = init_circuit_breaker_persistence(nomic_dir)
    status["background_tasks"] = init_background_tasks(nomic_dir)
    status["pulse_scheduler"] = await init_pulse_scheduler(stream_emitter)
    status["state_cleanup"] = init_state_cleanup_task()
    status["watchdog_task"] = await init_stuck_debate_watchdog()
    status["control_plane_coordinator"] = await init_control_plane_coordinator()
    status["shared_control_plane_state"] = await init_shared_control_plane_state()
    status["km_adapters"] = await init_km_adapters()
    status["workflow_checkpoint_persistence"] = init_workflow_checkpoint_persistence()
    status["tts_integration"] = await init_tts_integration()

    return status


__all__ = [
    "init_error_monitoring",
    "init_opentelemetry",
    "init_prometheus_metrics",
    "init_circuit_breaker_persistence",
    "init_background_tasks",
    "init_pulse_scheduler",
    "init_state_cleanup_task",
    "init_stuck_debate_watchdog",
    "init_control_plane_coordinator",
    "init_shared_control_plane_state",
    "init_tts_integration",
    "init_km_adapters",
    "init_workflow_checkpoint_persistence",
    "run_startup_sequence",
]
