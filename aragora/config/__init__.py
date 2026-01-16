"""
Aragora Configuration Package.

This package provides both validated Pydantic settings (new) and
module-level constants (legacy) for configuration.

New code should use the Pydantic settings:
    from aragora.config.settings import get_settings
    settings = get_settings()
    timeout = settings.database.timeout_seconds

Legacy constants are still available for backward compatibility:
    from aragora.config import DEFAULT_RATE_LIMIT

See settings.py for the full validated configuration schema.
"""

# Re-export Pydantic settings for convenient access
# Re-export legacy constants for backward compatibility
from .legacy import (  # noqa: F401
    AGENT_TIMEOUT_SECONDS,
    BELIEF_CONVERGENCE_THRESHOLD,
    # Belief Network
    BELIEF_MAX_ITERATIONS,
    CACHE_TTL_AGENT_FLIPS,
    CACHE_TTL_AGENT_H2H,
    # Cache TTLs - Agent Data
    CACHE_TTL_AGENT_PROFILE,
    CACHE_TTL_AGENT_REPUTATION,
    CACHE_TTL_ALL_REPUTATIONS,
    # Cache TTLs - Analytics
    CACHE_TTL_ANALYTICS,
    CACHE_TTL_ANALYTICS_DEBATES,
    CACHE_TTL_ANALYTICS_MEMORY,
    CACHE_TTL_ANALYTICS_RANKING,
    CACHE_TTL_ARCHIVE_STATS,
    CACHE_TTL_CALIBRATION_LB,
    # Cache TTLs - Consensus
    CACHE_TTL_CONSENSUS,
    CACHE_TTL_CONSENSUS_SETTLED,
    CACHE_TTL_CONSENSUS_SIMILAR,
    CACHE_TTL_CONSENSUS_STATS,
    CACHE_TTL_CONTRARIAN_VIEWS,
    CACHE_TTL_CRITIQUE_PATTERNS,
    CACHE_TTL_CRITIQUE_STATS,
    # Cache TTLs - Dashboard
    CACHE_TTL_DASHBOARD_DEBATES,
    # Cache TTLs - Embeddings
    CACHE_TTL_EMBEDDINGS,
    CACHE_TTL_FLIPS_RECENT,
    CACHE_TTL_FLIPS_SUMMARY,
    CACHE_TTL_LB_INTROSPECTION,
    CACHE_TTL_LB_MATCHES,
    CACHE_TTL_LB_RANKINGS,
    CACHE_TTL_LB_REPUTATION,
    CACHE_TTL_LB_STATS,
    CACHE_TTL_LB_TEAMS,
    # Cache TTLs - Leaderboard & Rankings
    CACHE_TTL_LEADERBOARD,
    CACHE_TTL_LEARNING_EVOLUTION,
    CACHE_TTL_META_LEARNING,
    # Cache TTLs - Generic
    CACHE_TTL_METHOD,
    CACHE_TTL_QUERY,
    CACHE_TTL_RECENT_DISSENTS,
    CACHE_TTL_RECENT_MATCHES,
    # Cache TTLs - Memory & Learning
    CACHE_TTL_REPLAYS_LIST,
    CACHE_TTL_RISK_WARNINGS,
    CROSS_EXAMINATION_DEPTH,
    DB_CALIBRATION_PATH,
    DB_CONSENSUS_PATH,
    # Database Paths (legacy)
    DB_ELO_PATH,
    DB_GENESIS_PATH,
    DB_INSIGHTS_PATH,
    DB_LAB_PATH,
    DB_MEMORY_PATH,
    DB_MODE,
    DB_PERSONAS_PATH,
    DB_POSITIONS_PATH,
    # Database
    DB_TIMEOUT_SECONDS,
    DEBATE_TIMEOUT_SECONDS,
    # Deep Audit
    DEEP_AUDIT_ROUNDS,
    # Agents
    DEFAULT_AGENTS,
    DEFAULT_CONSENSUS,
    DEFAULT_DEBATE_LANGUAGE,
    DEFAULT_PAGINATION,
    # Rate Limiting
    DEFAULT_RATE_LIMIT,
    # Debate Defaults
    DEFAULT_ROUNDS,
    # Storage
    DEFAULT_STORAGE_DIR,
    ELO_CALIBRATION_MIN_COUNT,
    # ELO System
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ENFORCE_RESPONSE_LANGUAGE,
    HEARTBEAT_INTERVAL_SECONDS,
    INTER_REQUEST_DELAY_SECONDS,
    IP_RATE_LIMIT,
    # State Management Limits
    MAX_ACTIVE_DEBATES,
    MAX_ACTIVE_LOOPS,
    # Debate Limits
    MAX_AGENTS_PER_DEBATE,
    # API Limits
    MAX_API_LIMIT,
    MAX_CONCURRENT_CRITIQUES,
    MAX_CONCURRENT_DEBATES,
    MAX_CONCURRENT_REVISIONS,
    MAX_CONCURRENT_STREAMING,
    MAX_CONTENT_LENGTH,
    MAX_DEBATE_STATES,
    MAX_EVENT_QUEUE_SIZE,
    MAX_LOG_BYTES,
    MAX_QUESTION_LENGTH,
    MAX_REPLAY_QUEUE_SIZE,
    MAX_ROUNDS,
    # Evidence Collection
    MAX_SNIPPETS_PER_CONNECTOR,
    MAX_TOTAL_SNIPPETS,
    NOMIC_DIR,
    OPENROUTER_INTER_REQUEST_DELAY,
    RISK_THRESHOLD,
    SHAREABLE_LINK_TTL,
    SNIPPET_MAX_LENGTH,
    SSL_CERT_PATH,
    # SSL/TLS
    SSL_ENABLED,
    SSL_KEY_PATH,
    STREAM_BATCH_SIZE,
    STREAM_DRAIN_INTERVAL_MS,
    STREAMING_CAPABLE_AGENTS,
    # Authentication
    TOKEN_TTL_SECONDS,
    USER_EVENT_QUEUE_SIZE,
    WS_HEARTBEAT_INTERVAL,
    # WebSocket
    WS_MAX_MESSAGE_SIZE,
    ConfigurationError,
    # Helper functions
    get_api_key,
    resolve_db_path,
    validate_configuration,
)
from .settings import (
    # Constants
    ALLOWED_AGENT_TYPES,
    AgentSettings,
    APILimitSettings,
    # Nested settings classes (for type hints)
    AuthSettings,
    BeliefSettings,
    CacheSettings,
    DatabaseSettings,
    DebateSettings,
    EloSettings,
    EvidenceSettings,
    RateLimitSettings,
    # Main settings class and accessor
    Settings,
    SSLSettings,
    StorageSettings,
    WebSocketSettings,
    get_settings,
    reset_settings,
)

# Re-export stability markers for honest epistemics
from .stability import (
    FeatureStabilityInfo,
    Stability,
    get_feature_info,
    get_feature_stability,
    get_stability_badge,
    get_stability_color,
    list_features_by_stability,
    stability_marker,
)

__all__ = [
    # Main settings
    "Settings",
    "get_settings",
    "reset_settings",
    # Nested settings classes
    "AuthSettings",
    "RateLimitSettings",
    "APILimitSettings",
    "DebateSettings",
    "AgentSettings",
    "CacheSettings",
    "DatabaseSettings",
    "WebSocketSettings",
    "EloSettings",
    "BeliefSettings",
    "SSLSettings",
    "StorageSettings",
    "EvidenceSettings",
    # Stability markers
    "Stability",
    "FeatureStabilityInfo",
    "stability_marker",
    "get_feature_stability",
    "get_feature_info",
    "list_features_by_stability",
    "get_stability_badge",
    "get_stability_color",
    # Constants
    "ALLOWED_AGENT_TYPES",
]
