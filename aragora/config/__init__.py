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
from .settings import (
    # Main settings class and accessor
    Settings,
    get_settings,
    reset_settings,
    # Nested settings classes (for type hints)
    AuthSettings,
    RateLimitSettings,
    APILimitSettings,
    DebateSettings,
    AgentSettings,
    CacheSettings,
    DatabaseSettings,
    WebSocketSettings,
    EloSettings,
    BeliefSettings,
    SSLSettings,
    StorageSettings,
    EvidenceSettings,
    # Constants
    ALLOWED_AGENT_TYPES,
)

# Re-export stability markers for honest epistemics
from .stability import (
    Stability,
    FeatureStabilityInfo,
    stability_marker,
    get_feature_stability,
    get_feature_info,
    list_features_by_stability,
    get_stability_badge,
    get_stability_color,
)

# Re-export legacy constants for backward compatibility
from .legacy import (
    # Helper functions
    get_api_key,
    validate_configuration,
    ConfigurationError,
    # Authentication
    TOKEN_TTL_SECONDS,
    SHAREABLE_LINK_TTL,
    # Rate Limiting
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    # API Limits
    MAX_API_LIMIT,
    DEFAULT_PAGINATION,
    MAX_CONTENT_LENGTH,
    MAX_QUESTION_LENGTH,
    # Debate Defaults
    DEFAULT_ROUNDS,
    MAX_ROUNDS,
    DEFAULT_CONSENSUS,
    DEBATE_TIMEOUT_SECONDS,
    AGENT_TIMEOUT_SECONDS,
    MAX_CONCURRENT_CRITIQUES,
    MAX_CONCURRENT_REVISIONS,
    # Agents
    DEFAULT_AGENTS,
    STREAMING_CAPABLE_AGENTS,
    # Cache TTLs - Leaderboard & Rankings
    CACHE_TTL_LEADERBOARD,
    CACHE_TTL_LB_RANKINGS,
    CACHE_TTL_LB_MATCHES,
    CACHE_TTL_LB_REPUTATION,
    CACHE_TTL_LB_TEAMS,
    CACHE_TTL_LB_STATS,
    CACHE_TTL_LB_INTROSPECTION,
    # Cache TTLs - Agent Data
    CACHE_TTL_AGENT_PROFILE,
    CACHE_TTL_AGENT_H2H,
    CACHE_TTL_AGENT_FLIPS,
    CACHE_TTL_AGENT_REPUTATION,
    CACHE_TTL_RECENT_MATCHES,
    CACHE_TTL_CALIBRATION_LB,
    CACHE_TTL_FLIPS_RECENT,
    CACHE_TTL_FLIPS_SUMMARY,
    # Cache TTLs - Analytics
    CACHE_TTL_ANALYTICS,
    CACHE_TTL_ANALYTICS_RANKING,
    CACHE_TTL_ANALYTICS_DEBATES,
    CACHE_TTL_ANALYTICS_MEMORY,
    # Cache TTLs - Consensus
    CACHE_TTL_CONSENSUS,
    CACHE_TTL_CONSENSUS_SIMILAR,
    CACHE_TTL_CONSENSUS_SETTLED,
    CACHE_TTL_CONSENSUS_STATS,
    CACHE_TTL_RECENT_DISSENTS,
    CACHE_TTL_CONTRARIAN_VIEWS,
    CACHE_TTL_RISK_WARNINGS,
    # Cache TTLs - Memory & Learning
    CACHE_TTL_REPLAYS_LIST,
    CACHE_TTL_LEARNING_EVOLUTION,
    CACHE_TTL_META_LEARNING,
    CACHE_TTL_CRITIQUE_PATTERNS,
    CACHE_TTL_CRITIQUE_STATS,
    CACHE_TTL_ARCHIVE_STATS,
    CACHE_TTL_ALL_REPUTATIONS,
    # Cache TTLs - Dashboard
    CACHE_TTL_DASHBOARD_DEBATES,
    # Cache TTLs - Embeddings
    CACHE_TTL_EMBEDDINGS,
    # Cache TTLs - Generic
    CACHE_TTL_METHOD,
    CACHE_TTL_QUERY,
    # WebSocket
    WS_MAX_MESSAGE_SIZE,
    WS_HEARTBEAT_INTERVAL,
    # Storage
    DEFAULT_STORAGE_DIR,
    MAX_LOG_BYTES,
    # Database
    DB_TIMEOUT_SECONDS,
    DB_MODE,
    NOMIC_DIR,
    # Database Paths (legacy)
    DB_ELO_PATH,
    DB_MEMORY_PATH,
    DB_INSIGHTS_PATH,
    DB_CONSENSUS_PATH,
    DB_CALIBRATION_PATH,
    DB_LAB_PATH,
    DB_PERSONAS_PATH,
    DB_POSITIONS_PATH,
    DB_GENESIS_PATH,
    # Evidence Collection
    MAX_SNIPPETS_PER_CONNECTOR,
    MAX_TOTAL_SNIPPETS,
    SNIPPET_MAX_LENGTH,
    # Deep Audit
    DEEP_AUDIT_ROUNDS,
    CROSS_EXAMINATION_DEPTH,
    RISK_THRESHOLD,
    # ELO System
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_CALIBRATION_MIN_COUNT,
    # Debate Limits
    MAX_AGENTS_PER_DEBATE,
    MAX_CONCURRENT_DEBATES,
    USER_EVENT_QUEUE_SIZE,
    # State Management Limits
    MAX_ACTIVE_DEBATES,
    MAX_ACTIVE_LOOPS,
    MAX_DEBATE_STATES,
    MAX_EVENT_QUEUE_SIZE,
    MAX_REPLAY_QUEUE_SIZE,
    # Belief Network
    BELIEF_MAX_ITERATIONS,
    BELIEF_CONVERGENCE_THRESHOLD,
    # SSL/TLS
    SSL_ENABLED,
    SSL_CERT_PATH,
    SSL_KEY_PATH,
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
