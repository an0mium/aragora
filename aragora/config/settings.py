"""
Pydantic-based configuration settings for Aragora.

This module provides validated, type-safe configuration using Pydantic.
All settings can be overridden via environment variables with the ARAGORA_ prefix.

Usage:
    from aragora.config.settings import get_settings

    settings = get_settings()
    print(settings.database.timeout_seconds)
    print(settings.rate_limit.default_limit)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """Authentication configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    token_ttl: int = Field(default=3600, ge=60, le=86400, alias="ARAGORA_TOKEN_TTL")
    shareable_link_ttl: int = Field(default=3600, ge=60, le=604800, alias="ARAGORA_SHAREABLE_LINK_TTL")
    # Rate limit tracking limits (prevent memory exhaustion)
    max_tracked_entries: int = Field(
        default=10000, ge=100, le=1000000,
        description="Max entries in rate limit tracking to prevent memory exhaustion"
    )
    max_revoked_tokens: int = Field(
        default=10000, ge=100, le=1000000,
        description="Max revoked tokens to store"
    )
    revoked_token_ttl: int = Field(
        default=86400, ge=3600, le=604800,
        description="How long to keep revoked tokens (seconds)"
    )
    rate_limit_window: int = Field(
        default=60, ge=10, le=3600,
        description="Rate limit window in seconds"
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    default_limit: int = Field(default=60, ge=1, le=10000, alias="ARAGORA_RATE_LIMIT")
    ip_rate_limit: int = Field(default=120, ge=1, le=10000, alias="ARAGORA_IP_RATE_LIMIT")
    burst_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, alias="ARAGORA_BURST_MULTIPLIER")

    # Redis configuration for persistent rate limiting
    redis_url: Optional[str] = Field(default=None, alias="ARAGORA_REDIS_URL")
    redis_key_prefix: str = Field(default="aragora:ratelimit:", alias="ARAGORA_REDIS_KEY_PREFIX")
    redis_ttl_seconds: int = Field(default=120, ge=60, le=3600, alias="ARAGORA_REDIS_TTL")


class APILimitSettings(BaseSettings):
    """API limits configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    max_api_limit: int = Field(default=100, ge=1, le=1000, alias="ARAGORA_MAX_API_LIMIT")
    default_pagination: int = Field(default=20, ge=1, le=100, alias="ARAGORA_DEFAULT_PAGINATION")
    max_content_length: int = Field(default=100 * 1024 * 1024, ge=1024, alias="ARAGORA_MAX_CONTENT_LENGTH")
    max_question_length: int = Field(default=10000, ge=100, le=100000, alias="ARAGORA_MAX_QUESTION_LENGTH")


class DebateSettings(BaseSettings):
    """Debate configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    default_rounds: int = Field(default=3, ge=1, le=20, alias="ARAGORA_DEFAULT_ROUNDS")
    max_rounds: int = Field(default=10, ge=1, le=50, alias="ARAGORA_MAX_ROUNDS")
    default_consensus: str = Field(default="hybrid", alias="ARAGORA_DEFAULT_CONSENSUS")
    timeout_seconds: int = Field(default=600, ge=30, le=7200, alias="ARAGORA_DEBATE_TIMEOUT")
    max_agents_per_debate: int = Field(default=10, ge=2, le=50, alias="ARAGORA_MAX_AGENTS_PER_DEBATE")
    max_concurrent_debates: int = Field(default=10, ge=1, le=100, alias="ARAGORA_MAX_CONCURRENT_DEBATES")
    user_event_queue_size: int = Field(default=10000, ge=100, le=100000, alias="ARAGORA_USER_EVENT_QUEUE_SIZE")

    @field_validator("default_consensus")
    @classmethod
    def validate_consensus(cls, v: str) -> str:
        valid = {"unanimous", "majority", "supermajority", "hybrid"}
        if v.lower() not in valid:
            raise ValueError(f"Consensus must be one of {valid}")
        return v.lower()


class AgentSettings(BaseSettings):
    """Agent configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    default_agents: str = Field(
        default="grok,anthropic-api,openai-api,deepseek,gemini",
        alias="ARAGORA_DEFAULT_AGENTS"
    )
    streaming_agents: str = Field(
        default="grok,anthropic-api,openai-api",
        alias="ARAGORA_STREAMING_AGENTS"
    )

    # Streaming configuration
    stream_buffer_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        le=100 * 1024 * 1024,  # 100MB max
        alias="ARAGORA_STREAM_BUFFER_SIZE",
        description="Maximum buffer size for streaming responses (bytes)"
    )
    stream_chunk_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        alias="ARAGORA_STREAM_CHUNK_TIMEOUT",
        description="Timeout between stream chunks (seconds)"
    )

    # Context limits (for truncation)
    max_context_chars: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        alias="ARAGORA_MAX_CONTEXT_CHARS",
        description="Maximum characters for context/history"
    )
    max_message_chars: int = Field(
        default=50000,
        ge=1000,
        le=500000,
        alias="ARAGORA_MAX_MESSAGE_CHARS",
        description="Maximum characters per message"
    )

    # Local LLM fallback configuration
    local_fallback_enabled: bool = Field(
        default=False,
        alias="ARAGORA_LOCAL_FALLBACK_ENABLED",
        description="Enable local LLM (Ollama/LM Studio) as fallback before OpenRouter"
    )
    local_fallback_priority: bool = Field(
        default=False,
        alias="ARAGORA_LOCAL_FALLBACK_PRIORITY",
        description="Prioritize local LLMs over cloud providers when available"
    )

    @property
    def default_agent_list(self) -> list[str]:
        """Get default agents as a list."""
        return [a.strip() for a in self.default_agents.split(",") if a.strip()]

    @property
    def streaming_agent_list(self) -> list[str]:
        """Get streaming-capable agents as a list."""
        return [a.strip() for a in self.streaming_agents.split(",") if a.strip()]


class CacheSettings(BaseSettings):
    """Cache TTL configuration (all values in seconds)."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_CACHE_")

    # Leaderboard & Rankings
    leaderboard: int = Field(default=300, ge=1, alias="ARAGORA_CACHE_LEADERBOARD")
    rankings: int = Field(default=300, ge=1, alias="ARAGORA_CACHE_LB_RANKINGS")
    matches: int = Field(default=120, ge=1, alias="ARAGORA_CACHE_LB_MATCHES")
    reputation: int = Field(default=300, ge=1, alias="ARAGORA_CACHE_LB_REPUTATION")
    recent_matches: int = Field(default=120, ge=1, alias="ARAGORA_CACHE_RECENT_MATCHES")

    # Agent Data
    agent_profile: int = Field(default=600, ge=1, alias="ARAGORA_CACHE_AGENT_PROFILE")
    agent_h2h: int = Field(default=600, ge=1, alias="ARAGORA_CACHE_AGENT_H2H")

    # Analytics
    analytics: int = Field(default=600, ge=1, alias="ARAGORA_CACHE_ANALYTICS")

    # Consensus
    consensus: int = Field(default=240, ge=1, alias="ARAGORA_CACHE_CONSENSUS")

    # Memory & Learning
    replays_list: int = Field(default=120, ge=1, alias="ARAGORA_CACHE_REPLAYS_LIST")

    # Generic tiers
    method_default: int = Field(default=300, ge=1, alias="ARAGORA_CACHE_METHOD")
    query_default: int = Field(default=60, ge=1, alias="ARAGORA_CACHE_QUERY")

    # Embeddings (expensive)
    embeddings: int = Field(default=3600, ge=60, alias="ARAGORA_CACHE_EMBEDDINGS")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_DB_")

    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, alias="ARAGORA_DB_TIMEOUT")
    mode: str = Field(default="legacy", alias="ARAGORA_DB_MODE")
    nomic_dir: str = Field(default=".nomic", alias="ARAGORA_NOMIC_DIR")

    # PostgreSQL configuration
    url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    backend: str = Field(default="sqlite", alias="ARAGORA_DB_BACKEND")
    pool_size: int = Field(default=5, ge=1, le=100, alias="ARAGORA_DB_POOL_SIZE")
    pool_max_overflow: int = Field(default=10, ge=0, le=100, alias="ARAGORA_DB_POOL_OVERFLOW")

    # Legacy paths (for backwards compatibility)
    elo_path: str = Field(default="agent_elo.db", alias="ARAGORA_DB_ELO")
    memory_path: str = Field(default="continuum.db", alias="ARAGORA_DB_MEMORY")
    insights_path: str = Field(default="aragora_insights.db", alias="ARAGORA_DB_INSIGHTS")
    consensus_path: str = Field(default="consensus_memory.db", alias="ARAGORA_DB_CONSENSUS")
    personas_path: str = Field(default="agent_personas.db", alias="ARAGORA_DB_PERSONAS")
    positions_path: str = Field(default="grounded_positions.db", alias="ARAGORA_DB_POSITIONS")
    genesis_path: str = Field(default="genesis.db", alias="ARAGORA_DB_GENESIS")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid = {"legacy", "consolidated"}
        if v.lower() not in valid:
            raise ValueError(f"Database mode must be one of {valid}")
        return v.lower()

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        valid = {"sqlite", "postgresql"}
        if v.lower() not in valid:
            raise ValueError(f"Database backend must be one of {valid}")
        return v.lower()

    @property
    def nomic_path(self) -> Path:
        """Get nomic directory as Path."""
        return Path(self.nomic_dir)

    @property
    def is_postgresql(self) -> bool:
        """Check if PostgreSQL backend is configured."""
        return self.backend == "postgresql" and self.url is not None


class WebSocketSettings(BaseSettings):
    """WebSocket configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_WS_")

    max_message_size: int = Field(default=64 * 1024, ge=1024, le=10 * 1024 * 1024, alias="ARAGORA_WS_MAX_MESSAGE_SIZE")
    heartbeat_interval: int = Field(default=30, ge=5, le=300, alias="ARAGORA_WS_HEARTBEAT")


class EloSettings(BaseSettings):
    """ELO rating system configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_ELO_")

    initial_rating: int = Field(default=1500, ge=100, le=3000, alias="ARAGORA_ELO_INITIAL")
    k_factor: int = Field(default=32, ge=1, le=100, alias="ARAGORA_ELO_K_FACTOR")
    calibration_min_count: int = Field(default=10, ge=1, le=100, alias="ARAGORA_ELO_CALIBRATION_MIN_COUNT")


class BeliefSettings(BaseSettings):
    """Belief network configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_BELIEF_")

    max_iterations: int = Field(default=100, ge=10, le=1000, alias="ARAGORA_BELIEF_MAX_ITERATIONS")
    convergence_threshold: float = Field(default=0.001, ge=0.0001, le=0.1, alias="ARAGORA_BELIEF_CONVERGENCE_THRESHOLD")


class SSLSettings(BaseSettings):
    """SSL/TLS configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_SSL_")

    enabled: bool = Field(default=False, alias="ARAGORA_SSL_ENABLED")
    cert_path: str = Field(default="", alias="ARAGORA_SSL_CERT")
    key_path: str = Field(default="", alias="ARAGORA_SSL_KEY")

    @field_validator("cert_path", "key_path")
    @classmethod
    def validate_ssl_paths(cls, v: str, info) -> str:
        # Only validate if SSL is being enabled
        # Can't check enabled here since it's validated after
        return v


class StorageSettings(BaseSettings):
    """Storage configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    storage_dir: str = Field(default=".aragora", alias="ARAGORA_STORAGE_DIR")
    max_log_bytes: int = Field(default=100 * 1024, ge=1024, alias="ARAGORA_MAX_LOG_BYTES")


class EvidenceSettings(BaseSettings):
    """Evidence collection configuration."""

    model_config = SettingsConfigDict(env_prefix="ARAGORA_")

    max_snippets_per_connector: int = Field(default=3, ge=1, le=20, alias="ARAGORA_MAX_SNIPPETS_CONNECTOR")
    max_total_snippets: int = Field(default=8, ge=1, le=50, alias="ARAGORA_MAX_TOTAL_SNIPPETS")
    snippet_max_length: int = Field(default=1000, ge=100, le=10000, alias="ARAGORA_SNIPPET_MAX_LENGTH")


class Settings(BaseSettings):
    """
    Main settings class aggregating all configuration sections.

    Access nested settings via properties:
        settings = get_settings()
        settings.database.timeout_seconds
        settings.rate_limit.default_limit
        settings.debate.max_rounds
    """

    model_config = SettingsConfigDict(
        env_prefix="ARAGORA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Nested settings (loaded lazily on first access)
    _auth: Optional[AuthSettings] = None
    _rate_limit: Optional[RateLimitSettings] = None
    _api_limit: Optional[APILimitSettings] = None
    _debate: Optional[DebateSettings] = None
    _agent: Optional[AgentSettings] = None
    _cache: Optional[CacheSettings] = None
    _database: Optional[DatabaseSettings] = None
    _websocket: Optional[WebSocketSettings] = None
    _elo: Optional[EloSettings] = None
    _belief: Optional[BeliefSettings] = None
    _ssl: Optional[SSLSettings] = None
    _storage: Optional[StorageSettings] = None
    _evidence: Optional[EvidenceSettings] = None

    @property
    def auth(self) -> AuthSettings:
        if self._auth is None:
            self._auth = AuthSettings()
        return self._auth

    @property
    def rate_limit(self) -> RateLimitSettings:
        if self._rate_limit is None:
            self._rate_limit = RateLimitSettings()
        return self._rate_limit

    @property
    def api_limit(self) -> APILimitSettings:
        if self._api_limit is None:
            self._api_limit = APILimitSettings()
        return self._api_limit

    @property
    def debate(self) -> DebateSettings:
        if self._debate is None:
            self._debate = DebateSettings()
        return self._debate

    @property
    def agent(self) -> AgentSettings:
        if self._agent is None:
            self._agent = AgentSettings()
        return self._agent

    @property
    def cache(self) -> CacheSettings:
        if self._cache is None:
            self._cache = CacheSettings()
        return self._cache

    @property
    def database(self) -> DatabaseSettings:
        if self._database is None:
            self._database = DatabaseSettings()
        return self._database

    @property
    def websocket(self) -> WebSocketSettings:
        if self._websocket is None:
            self._websocket = WebSocketSettings()
        return self._websocket

    @property
    def elo(self) -> EloSettings:
        if self._elo is None:
            self._elo = EloSettings()
        return self._elo

    @property
    def belief(self) -> BeliefSettings:
        if self._belief is None:
            self._belief = BeliefSettings()
        return self._belief

    @property
    def ssl(self) -> SSLSettings:
        if self._ssl is None:
            self._ssl = SSLSettings()
        return self._ssl

    @property
    def storage(self) -> StorageSettings:
        if self._storage is None:
            self._storage = StorageSettings()
        return self._storage

    @property
    def evidence(self) -> EvidenceSettings:
        if self._evidence is None:
            self._evidence = EvidenceSettings()
        return self._evidence


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure the same settings instance is returned
    throughout the application lifecycle.

    Returns:
        Settings instance with all configuration loaded from environment.
    """
    return Settings()


def reset_settings() -> None:
    """
    Reset the cached settings instance.

    Useful for testing or when environment variables change.
    """
    get_settings.cache_clear()


# Valid agent types (allowlist for security)
ALLOWED_AGENT_TYPES: FrozenSet[str] = frozenset({
    # CLI-based
    "codex", "claude", "openai", "gemini-cli", "grok-cli",
    "qwen-cli", "deepseek-cli", "kilocode",
    # API-based (direct)
    "gemini", "ollama", "anthropic-api", "openai-api", "grok",
    # API-based (via OpenRouter)
    "deepseek", "deepseek-r1", "llama", "mistral", "openrouter",
})
