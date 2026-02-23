"""
Server startup Knowledge Mound initialization.

This module handles TTS integration, Knowledge Mound configuration,
initialization, and adapter setup.
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import MoundConfig


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
        logger.debug("TTS integration not available: %s", e)
    except (ValueError, TypeError, RuntimeError, OSError, AttributeError) as e:
        logger.warning("Failed to initialize TTS integration: %s", e)

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
        logger.warning("Unknown KM_BACKEND: %s, defaulting to SQLite", backend_str)
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
        weaviate_api_key=os.environ.get(
            "KM_WEAVIATE_API_KEY", os.environ.get("WEAVIATE_API_KEY", "")
        )
        or None,
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
        "KM config: backend=%s, postgres=%s, redis=%s, weaviate=%s",
        backend.value,
        "configured" if config.postgres_url else "none",
        "configured" if config.redis_url else "none",
        "configured" if config.weaviate_url else "none",
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

        logger.info("Knowledge Mound initialized with %s backend", config.backend.value)
        return True

    except ImportError as e:
        logger.debug("Knowledge Mound not available: %s", e)
    except (ValueError, TypeError, RuntimeError, OSError, ConnectionError) as e:
        logger.warning("Failed to initialize Knowledge Mound from env: %s", e)

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
        # RankingAdapter is instantiable despite inheriting abstract base
        ranking_adapter = RankingAdapter()  # type: ignore[abstract]
        setattr(manager, "_ranking_adapter", ranking_adapter)
        logger.debug("RankingAdapter initialized for KM integration")

        # Initialize RlmAdapter
        rlm_adapter = RlmAdapter()
        setattr(manager, "_rlm_adapter", rlm_adapter)
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

            create_km_bridge()
            logger.debug("KM WebSocket bridge initialized")
        except ImportError:
            pass

        return True

    except ImportError as e:
        logger.debug("KM adapters not available: %s", e)
    except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as e:
        logger.warning("Failed to initialize KM adapters: %s", e)

    return False
