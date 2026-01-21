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


def check_production_requirements() -> list[str]:
    """Check if production requirements are met.

    Returns:
        List of missing requirements (empty if all met)
    """
    import os

    missing = []
    env = os.environ.get("ARAGORA_ENV", "development")

    if env == "production":
        # Encryption key is required for production
        if not os.environ.get("ARAGORA_ENCRYPTION_KEY"):
            missing.append(
                "ARAGORA_ENCRYPTION_KEY required in production "
                "(32-byte hex string for AES-256 encryption)"
            )

        # Redis is required for durable state
        if not os.environ.get("REDIS_URL"):
            missing.append(
                "REDIS_URL required in production for debate origins "
                "and control plane state"
            )

        # PostgreSQL is recommended for governance store
        if not os.environ.get("DATABASE_URL"):
            logger.warning(
                "DATABASE_URL not set in production - using SQLite. "
                "PostgreSQL recommended for governance store."
            )

    return missing


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


async def init_persistent_task_queue() -> int:
    """Initialize persistent task queue with recovery of pending tasks.

    Creates the singleton PersistentTaskQueue and recovers any tasks that
    were pending/running when the server was previously stopped.

    Returns:
        Number of tasks recovered
    """
    try:
        from aragora.workflow.queue import (
            get_persistent_task_queue,
            PersistentTaskQueue,
        )

        # Get or create the singleton queue
        queue = get_persistent_task_queue()

        # Start the queue processor
        await queue.start()

        # Recover pending tasks from previous runs
        recovered = await queue.recover_tasks()

        # Schedule cleanup of old completed tasks (24h retention)
        import asyncio

        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run hourly
                try:
                    deleted = queue.delete_completed_tasks(older_than_hours=24)
                    if deleted > 0:
                        logger.debug(f"Cleaned up {deleted} old completed tasks")
                except Exception as e:
                    logger.warning(f"Task cleanup failed: {e}")

        asyncio.create_task(cleanup_loop())

        if recovered > 0:
            logger.info(f"Persistent task queue started, recovered {recovered} tasks")
        else:
            logger.info("Persistent task queue started")

        return recovered

    except ImportError as e:
        logger.debug(f"Persistent task queue not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize persistent task queue: {e}")

    return 0


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


def get_km_config_from_env() -> "MoundConfig":
    """Create MoundConfig from environment variables.

    Environment variables:
        KM_BACKEND: Backend type (sqlite, postgres, auto). Default: auto
        KM_POSTGRES_URL: PostgreSQL connection URL
        KM_POSTGRES_POOL_SIZE: Connection pool size (default: 10)
        KM_REDIS_URL: Redis connection URL for caching
        KM_REDIS_CACHE_TTL: Query cache TTL in seconds (default: 300)
        KM_SQLITE_PATH: SQLite database path
        KM_WEAVIATE_URL: Weaviate vector DB URL
        KM_WEAVIATE_API_KEY: Weaviate API key
        KM_WORKSPACE_ID: Default workspace ID
        KM_ENABLE_STALENESS: Enable staleness detection (default: true)
        KM_ENABLE_CULTURE: Enable culture accumulation (default: true)
        KM_ENABLE_DEDUP: Enable deduplication (default: true)
        KM_FEDERATION_NODE_NAME: This node's federation name
        KM_FEDERATION_ENDPOINT: This node's federation endpoint URL

    Returns:
        MoundConfig configured from environment
    """
    import os
    from aragora.knowledge.mound.types import MoundConfig, MoundBackend

    # Determine backend
    backend_str = os.environ.get("KM_BACKEND", "auto").lower()
    postgres_url = os.environ.get("KM_POSTGRES_URL", os.environ.get("DATABASE_URL", ""))
    sqlite_path = os.environ.get("KM_SQLITE_PATH", "")

    if backend_str == "postgres":
        backend = MoundBackend.POSTGRES
    elif backend_str == "sqlite":
        backend = MoundBackend.SQLITE
    elif backend_str == "auto":
        # Auto-select based on available credentials
        if postgres_url:
            backend = MoundBackend.POSTGRES
        else:
            backend = MoundBackend.SQLITE
    else:
        logger.warning(f"Unknown KM_BACKEND: {backend_str}, defaulting to SQLite")
        backend = MoundBackend.SQLITE

    # Helper for bool env vars
    def env_bool(key: str, default: bool = True) -> bool:
        val = os.environ.get(key, "").lower()
        if val in ("true", "1", "yes", "on"):
            return True
        elif val in ("false", "0", "no", "off"):
            return False
        return default

    # Helper for int env vars
    def env_int(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            return default

    # Helper for float env vars
    def env_float(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            return default

    config = MoundConfig(
        backend=backend,
        # PostgreSQL
        postgres_url=postgres_url or None,
        postgres_pool_size=env_int("KM_POSTGRES_POOL_SIZE", 10),
        postgres_pool_max_overflow=env_int("KM_POSTGRES_POOL_OVERFLOW", 5),
        # Redis
        redis_url=os.environ.get("KM_REDIS_URL", os.environ.get("REDIS_URL", "")) or None,
        redis_cache_ttl=env_int("KM_REDIS_CACHE_TTL", 300),
        redis_culture_ttl=env_int("KM_REDIS_CULTURE_TTL", 3600),
        # SQLite
        sqlite_path=sqlite_path or None,
        # Weaviate
        weaviate_url=os.environ.get("KM_WEAVIATE_URL", os.environ.get("WEAVIATE_URL", "")) or None,
        weaviate_api_key=os.environ.get("KM_WEAVIATE_API_KEY", os.environ.get("WEAVIATE_API_KEY", "")) or None,
        weaviate_collection=os.environ.get("KM_WEAVIATE_COLLECTION", "KnowledgeMound"),
        # Feature flags
        enable_staleness_detection=env_bool("KM_ENABLE_STALENESS", True),
        enable_culture_accumulator=env_bool("KM_ENABLE_CULTURE", True),
        enable_auto_revalidation=env_bool("KM_ENABLE_AUTO_REVALIDATION", False),
        enable_deduplication=env_bool("KM_ENABLE_DEDUP", True),
        enable_provenance_tracking=env_bool("KM_ENABLE_PROVENANCE", True),
        # Adapter flags
        enable_evidence_adapter=env_bool("KM_ENABLE_EVIDENCE_ADAPTER", True),
        enable_pulse_adapter=env_bool("KM_ENABLE_PULSE_ADAPTER", True),
        enable_insights_adapter=env_bool("KM_ENABLE_INSIGHTS_ADAPTER", True),
        enable_elo_adapter=env_bool("KM_ENABLE_ELO_ADAPTER", True),
        enable_belief_adapter=env_bool("KM_ENABLE_BELIEF_ADAPTER", True),
        enable_cost_adapter=env_bool("KM_ENABLE_COST_ADAPTER", False),  # Opt-in
        # Confidence thresholds
        evidence_min_reliability=env_float("KM_EVIDENCE_MIN_RELIABILITY", 0.6),
        pulse_min_quality=env_float("KM_PULSE_MIN_QUALITY", 0.6),
        insight_min_confidence=env_float("KM_INSIGHT_MIN_CONFIDENCE", 0.7),
        crux_min_score=env_float("KM_CRUX_MIN_SCORE", 0.3),
        belief_min_confidence=env_float("KM_BELIEF_MIN_CONFIDENCE", 0.8),
        # Multi-tenant
        default_workspace_id=os.environ.get("KM_WORKSPACE_ID", "default"),
        # Query settings
        default_query_limit=env_int("KM_DEFAULT_QUERY_LIMIT", 20),
    )

    logger.info(
        f"KM config: backend={backend.value}, "
        f"postgres={'configured' if config.postgres_url else 'none'}, "
        f"redis={'configured' if config.redis_url else 'none'}, "
        f"weaviate={'configured' if config.weaviate_url else 'none'}"
    )

    return config


async def init_knowledge_mound_from_env() -> bool:
    """Initialize the Knowledge Mound singleton from environment configuration.

    This should be called early in the startup sequence to ensure the KM
    is properly configured before other systems that depend on it.

    Returns:
        True if KM was initialized, False otherwise
    """
    try:
        from aragora.knowledge.mound import set_mound_config, get_knowledge_mound

        config = get_km_config_from_env()
        set_mound_config(config)

        # Initialize the mound
        mound = get_knowledge_mound()
        await mound.initialize()

        logger.info(f"Knowledge Mound initialized with {config.backend.value} backend")
        return True

    except ImportError as e:
        logger.debug(f"Knowledge Mound not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize Knowledge Mound from env: {e}")

    return False


async def init_km_adapters() -> bool:
    """Initialize Knowledge Mound adapters from persisted state.

    Loads expertise profiles and compression patterns from KM to restore
    cross-debate learning state. Also initializes:
    - AdapterFactory for auto-creating adapters from subsystems
    - WebSocket bridge for real-time KM event streaming
    - Global metrics for observability

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

        # Initialize global KM metrics
        try:
            from aragora.knowledge.mound.metrics import KMMetrics, set_metrics

            set_metrics(KMMetrics())
            logger.debug("KM global metrics initialized")
        except ImportError:
            pass

        # Initialize WebSocket bridge for KM events
        try:
            from aragora.knowledge.mound.websocket_bridge import create_km_bridge

            bridge = create_km_bridge()
            logger.debug("KM WebSocket bridge initialized")
        except ImportError:
            pass

        return True

    except ImportError as e:
        logger.debug(f"KM adapters not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize KM adapters: {e}")

    return False


def init_slo_webhooks() -> bool:
    """Initialize SLO violation webhook notifications.

    Connects SLO metric violations to the webhook dispatcher so that
    external alerting systems can be notified of performance degradation.

    Returns:
        True if SLO webhooks were initialized, False otherwise
    """
    try:
        from aragora.observability.metrics.slo import init_slo_webhooks as _init_slo_webhooks

        if _init_slo_webhooks():
            logger.info("SLO webhook notifications enabled")
            return True
        else:
            logger.debug("SLO webhooks not initialized (dispatcher not available)")
            return False

    except ImportError as e:
        logger.debug(f"SLO webhooks not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize SLO webhooks: {e}")

    return False


def init_webhook_dispatcher() -> bool:
    """Initialize the webhook dispatcher for outbound notifications.

    Loads webhook configurations from environment and starts the dispatcher.

    Returns:
        True if dispatcher was started, False otherwise
    """
    try:
        from aragora.integrations.webhooks import init_dispatcher

        dispatcher = init_dispatcher()
        if dispatcher:
            logger.info(f"Webhook dispatcher started with {len(dispatcher.configs)} endpoint(s)")
            return True
        else:
            logger.debug("No webhook configurations found, dispatcher not started")
            return False

    except ImportError as e:
        logger.debug(f"Webhook dispatcher not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize webhook dispatcher: {e}")

    return False


async def init_redis_state_backend() -> bool:
    """Initialize Redis-backed state management for horizontal scaling.

    Enables cross-instance debate state sharing and WebSocket broadcasting.
    Only initializes if ARAGORA_STATE_BACKEND=redis or ARAGORA_REDIS_URL is set.

    Environment Variables:
        ARAGORA_STATE_BACKEND: "redis" to enable, "memory" to disable (default)
        ARAGORA_REDIS_URL: Redis connection URL

    Returns:
        True if Redis state backend was initialized, False otherwise
    """
    import os

    # Check if Redis state is enabled
    state_backend = os.environ.get("ARAGORA_STATE_BACKEND", "memory").lower()
    redis_url = os.environ.get("ARAGORA_REDIS_URL", "")

    if state_backend != "redis" and not redis_url:
        logger.debug("Redis state backend disabled (set ARAGORA_STATE_BACKEND=redis to enable)")
        return False

    try:
        from aragora.server.redis_state import get_redis_state_manager

        manager = await get_redis_state_manager(auto_connect=True)
        if manager.is_connected:
            logger.info("Redis state backend initialized for horizontal scaling")
            return True
        else:
            logger.warning("Redis state backend failed to connect, falling back to in-memory")
            return False

    except ImportError as e:
        logger.debug(f"Redis state backend not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis state backend: {e}")

    return False


def init_gauntlet_run_recovery() -> int:
    """Recover stale gauntlet runs after server restart.

    Finds gauntlet runs that were pending/running when the server stopped
    and marks them as interrupted. Users can then view the status and
    optionally restart them.

    Returns:
        Number of stale runs recovered/marked as interrupted
    """
    try:
        from aragora.server.handlers.gauntlet import recover_stale_gauntlet_runs

        recovered = recover_stale_gauntlet_runs(max_age_seconds=7200)
        if recovered > 0:
            logger.info(f"Recovered {recovered} stale gauntlet runs from previous session")
        return recovered

    except ImportError as e:
        logger.debug(f"Gauntlet run recovery not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to recover stale gauntlet runs: {e}")

    return 0


async def init_durable_job_queue_recovery() -> int:
    """Recover interrupted jobs from the durable job queue.

    This is enabled by default. Set ARAGORA_DURABLE_GAUNTLET=0 to disable.

    Returns:
        Number of jobs recovered and re-enqueued
    """
    import os

    # Enabled by default - set to "0" to disable
    if os.environ.get("ARAGORA_DURABLE_GAUNTLET", "1").lower() in ("0", "false", "no"):
        logger.debug("Durable job queue recovery skipped (ARAGORA_DURABLE_GAUNTLET disabled)")
        return 0

    try:
        from aragora.queue.workers.gauntlet_worker import recover_interrupted_gauntlets

        recovered = await recover_interrupted_gauntlets()
        if recovered > 0:
            logger.info(f"Recovered {recovered} interrupted gauntlet jobs to durable queue")
        return recovered

    except ImportError as e:
        logger.debug(f"Durable job queue recovery not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to recover durable job queue: {e}")

    return 0


async def init_gauntlet_worker() -> bool:
    """Initialize and start the gauntlet job queue worker.

    Enabled by default. Set ARAGORA_DURABLE_GAUNTLET=0 to disable.

    Environment Variables:
        ARAGORA_DURABLE_GAUNTLET: "0" to disable (enabled by default)
        ARAGORA_GAUNTLET_WORKERS: Number of concurrent jobs (default: 3)

    Returns:
        True if worker was started, False otherwise
    """
    import os

    # Enabled by default - set to "0" to disable
    if os.environ.get("ARAGORA_DURABLE_GAUNTLET", "1").lower() in ("0", "false", "no"):
        logger.debug("Gauntlet worker not started (ARAGORA_DURABLE_GAUNTLET disabled)")
        return False

    try:
        from aragora.queue.workers.gauntlet_worker import GauntletWorker

        max_concurrent = int(os.environ.get("ARAGORA_GAUNTLET_WORKERS", "3"))
        worker = GauntletWorker(max_concurrent=max_concurrent)

        # Start worker in background
        import asyncio

        asyncio.create_task(worker.start())
        logger.info(f"Gauntlet worker started (max_concurrent={max_concurrent})")
        return True

    except ImportError as e:
        logger.debug(f"Gauntlet worker not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start gauntlet worker: {e}")

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
        "persistent_task_queue": 0,
        "webhook_dispatcher": False,
        "slo_webhooks": False,
        "gauntlet_runs_recovered": 0,
        "durable_jobs_recovered": 0,
        "gauntlet_worker": False,
        "redis_state_backend": False,
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
    status["persistent_task_queue"] = await init_persistent_task_queue()

    # Initialize webhooks (dispatcher must be initialized before SLO webhooks)
    status["webhook_dispatcher"] = init_webhook_dispatcher()
    status["slo_webhooks"] = init_slo_webhooks()

    # Recover stale gauntlet runs from previous session
    status["gauntlet_runs_recovered"] = init_gauntlet_run_recovery()

    # Recover and re-enqueue interrupted jobs from durable queue (if enabled)
    status["durable_jobs_recovered"] = await init_durable_job_queue_recovery()

    # Start gauntlet worker for durable queue processing (if enabled)
    status["gauntlet_worker"] = await init_gauntlet_worker()

    # Initialize Redis state backend for horizontal scaling
    status["redis_state_backend"] = await init_redis_state_backend()

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
    "init_persistent_task_queue",
    "init_km_adapters",
    "init_workflow_checkpoint_persistence",
    "init_webhook_dispatcher",
    "init_slo_webhooks",
    "init_gauntlet_run_recovery",
    "init_durable_job_queue_recovery",
    "init_gauntlet_worker",
    "init_redis_state_backend",
    "run_startup_sequence",
]
