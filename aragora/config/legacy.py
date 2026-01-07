"""
Aragora Configuration.

Centralized configuration with environment variable overrides.
Import these values instead of hardcoding throughout the codebase.
"""

import os
from typing import Optional


def _env_int(key: str, default: int) -> int:
    """Get integer from environment with fallback."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment with fallback."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment with fallback."""
    return os.getenv(key, default)


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment with fallback."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def get_api_key(*env_vars: str, required: bool = True) -> Optional[str]:
    """Get and validate API key from environment variables.

    Checks each environment variable in order, returning the first valid
    (non-empty, non-whitespace) value found. Strips whitespace from the result.

    Args:
        *env_vars: Environment variable names to check (in order of preference)
        required: If True, raise ValueError when no valid key found

    Returns:
        The stripped API key, or None if not required and not found

    Raises:
        ValueError: If required=True and no valid key found

    Example:
        >>> api_key = get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        >>> optional_key = get_api_key("BACKUP_KEY", required=False)
    """
    for var in env_vars:
        value = os.getenv(var)
        if value and value.strip():
            return value.strip()

    if required:
        var_names = " or ".join(env_vars)
        raise ValueError(f"{var_names} environment variable required")
    return None


# === Authentication ===
TOKEN_TTL_SECONDS = _env_int("ARAGORA_TOKEN_TTL", 3600)
SHAREABLE_LINK_TTL = _env_int("ARAGORA_SHAREABLE_LINK_TTL", 3600)

# === Rate Limiting ===
DEFAULT_RATE_LIMIT = _env_int("ARAGORA_RATE_LIMIT", 60)  # requests per minute
IP_RATE_LIMIT = _env_int("ARAGORA_IP_RATE_LIMIT", 120)

# === API Limits ===
MAX_API_LIMIT = _env_int("ARAGORA_MAX_API_LIMIT", 100)
DEFAULT_PAGINATION = _env_int("ARAGORA_DEFAULT_PAGINATION", 20)
MAX_CONTENT_LENGTH = _env_int("ARAGORA_MAX_CONTENT_LENGTH", 100 * 1024 * 1024)  # 100MB
MAX_QUESTION_LENGTH = _env_int("ARAGORA_MAX_QUESTION_LENGTH", 10000)

# === Debate Defaults ===
DEFAULT_ROUNDS = _env_int("ARAGORA_DEFAULT_ROUNDS", 3)
MAX_ROUNDS = _env_int("ARAGORA_MAX_ROUNDS", 10)
DEFAULT_CONSENSUS = _env_str("ARAGORA_DEFAULT_CONSENSUS", "hybrid")
DEBATE_TIMEOUT_SECONDS = _env_int("ARAGORA_DEBATE_TIMEOUT", 600)  # 10 minutes

# === Agents ===
DEFAULT_AGENTS = _env_str(
    "ARAGORA_DEFAULT_AGENTS",
    "grok,anthropic-api,openai-api,deepseek,gemini"
)
STREAMING_CAPABLE_AGENTS = _env_str(
    "ARAGORA_STREAMING_AGENTS",
    "grok,anthropic-api,openai-api"
)

# Valid agent types (allowlist for security)
# Single source of truth - import this instead of duplicating
ALLOWED_AGENT_TYPES = frozenset({
    # CLI-based
    "codex", "claude", "openai", "gemini-cli", "grok-cli",
    "qwen-cli", "deepseek-cli", "kilocode",
    # API-based (direct)
    "gemini", "ollama", "anthropic-api", "openai-api", "grok",
    # API-based (via OpenRouter)
    "deepseek", "deepseek-r1", "llama", "mistral", "openrouter",
})

# === Caching TTLs (seconds) ===
# Standard tiers - use these for most cases:
#   SHORT=60s   - Real-time or frequently-changing data
#   MEDIUM=120s - Frequently queried, moderate volatility
#   DEFAULT=300s - Standard cache duration
#   LONG=600s   - Stable data, less frequent queries
#   EXTENDED=900s - Aggregate statistics
#   VERY_LONG=1800s - Expensive computation, rarely changes
#   EMBEDDING=3600s - Very expensive, rarely invalidated

# --- Leaderboard & Rankings ---
CACHE_TTL_LEADERBOARD = _env_int("ARAGORA_CACHE_LEADERBOARD", 300)  # 5 min
CACHE_TTL_LB_RANKINGS = _env_int("ARAGORA_CACHE_LB_RANKINGS", 300)  # 5 min
CACHE_TTL_LB_MATCHES = _env_int("ARAGORA_CACHE_LB_MATCHES", 120)  # 2 min
CACHE_TTL_LB_REPUTATION = _env_int("ARAGORA_CACHE_LB_REPUTATION", 300)  # 5 min
CACHE_TTL_LB_TEAMS = _env_int("ARAGORA_CACHE_LB_TEAMS", 600)  # 10 min
CACHE_TTL_LB_STATS = _env_int("ARAGORA_CACHE_LB_STATS", 900)  # 15 min
CACHE_TTL_LB_INTROSPECTION = _env_int("ARAGORA_CACHE_LB_INTROSPECTION", 600)  # 10 min

# --- Agent Data ---
CACHE_TTL_AGENT_PROFILE = _env_int("ARAGORA_CACHE_AGENT_PROFILE", 600)  # 10 min
CACHE_TTL_AGENT_H2H = _env_int("ARAGORA_CACHE_AGENT_H2H", 600)  # 10 min
CACHE_TTL_AGENT_FLIPS = _env_int("ARAGORA_CACHE_AGENT_FLIPS", 300)  # 5 min
CACHE_TTL_AGENT_REPUTATION = _env_int("ARAGORA_CACHE_AGENT_REPUTATION", 120)  # 2 min
CACHE_TTL_RECENT_MATCHES = _env_int("ARAGORA_CACHE_RECENT_MATCHES", 120)  # 2 min
CACHE_TTL_CALIBRATION_LB = _env_int("ARAGORA_CACHE_CALIBRATION_LB", 300)  # 5 min
CACHE_TTL_FLIPS_RECENT = _env_int("ARAGORA_CACHE_FLIPS_RECENT", 300)  # 5 min
CACHE_TTL_FLIPS_SUMMARY = _env_int("ARAGORA_CACHE_FLIPS_SUMMARY", 600)  # 10 min

# --- Analytics ---
CACHE_TTL_ANALYTICS = _env_int("ARAGORA_CACHE_ANALYTICS", 600)  # 10 min
CACHE_TTL_ANALYTICS_RANKING = _env_int("ARAGORA_CACHE_ANALYTICS_RANKING", 300)  # 5 min
CACHE_TTL_ANALYTICS_DEBATES = _env_int("ARAGORA_CACHE_ANALYTICS_DEBATES", 300)  # 5 min
CACHE_TTL_ANALYTICS_MEMORY = _env_int("ARAGORA_CACHE_ANALYTICS_MEMORY", 1800)  # 30 min

# --- Consensus ---
CACHE_TTL_CONSENSUS = _env_int("ARAGORA_CACHE_CONSENSUS", 240)  # 4 min
CACHE_TTL_CONSENSUS_SIMILAR = _env_int("ARAGORA_CACHE_CONSENSUS_SIMILAR", 240)  # 4 min
CACHE_TTL_CONSENSUS_SETTLED = _env_int("ARAGORA_CACHE_CONSENSUS_SETTLED", 600)  # 10 min
CACHE_TTL_CONSENSUS_STATS = _env_int("ARAGORA_CACHE_CONSENSUS_STATS", 600)  # 10 min
CACHE_TTL_RECENT_DISSENTS = _env_int("ARAGORA_CACHE_RECENT_DISSENTS", 300)  # 5 min
CACHE_TTL_CONTRARIAN_VIEWS = _env_int("ARAGORA_CACHE_CONTRARIAN_VIEWS", 300)  # 5 min
CACHE_TTL_RISK_WARNINGS = _env_int("ARAGORA_CACHE_RISK_WARNINGS", 300)  # 5 min

# --- Memory & Learning ---
CACHE_TTL_REPLAYS_LIST = _env_int("ARAGORA_CACHE_REPLAYS_LIST", 120)  # 2 min
CACHE_TTL_LEARNING_EVOLUTION = _env_int("ARAGORA_CACHE_LEARNING_EVOLUTION", 600)  # 10 min
CACHE_TTL_META_LEARNING = _env_int("ARAGORA_CACHE_META_LEARNING", 60)  # 1 min
CACHE_TTL_CRITIQUE_PATTERNS = _env_int("ARAGORA_CACHE_CRITIQUE_PATTERNS", 120)  # 2 min
CACHE_TTL_CRITIQUE_STATS = _env_int("ARAGORA_CACHE_CRITIQUE_STATS", 300)  # 5 min
CACHE_TTL_ARCHIVE_STATS = _env_int("ARAGORA_CACHE_ARCHIVE_STATS", 600)  # 10 min
CACHE_TTL_ALL_REPUTATIONS = _env_int("ARAGORA_CACHE_ALL_REPUTATIONS", 300)  # 5 min

# --- Dashboard ---
CACHE_TTL_DASHBOARD_DEBATES = _env_int("ARAGORA_CACHE_DASHBOARD_DEBATES", 600)  # 10 min

# --- Embeddings (expensive computation) ---
CACHE_TTL_EMBEDDINGS = _env_int("ARAGORA_CACHE_EMBEDDINGS", 3600)  # 1 hour

# --- Generic cache tiers (for utils/cache.py) ---
CACHE_TTL_METHOD = _env_int("ARAGORA_CACHE_METHOD", 300)  # 5 min
CACHE_TTL_QUERY = _env_int("ARAGORA_CACHE_QUERY", 60)  # 1 min

# === WebSocket ===
# Note: 64KB default prevents memory exhaustion from malicious large messages
# Increase for deployments with trusted clients/large message payloads
WS_MAX_MESSAGE_SIZE = _env_int("ARAGORA_WS_MAX_MESSAGE_SIZE", 64 * 1024)  # 64KB default
WS_HEARTBEAT_INTERVAL = _env_int("ARAGORA_WS_HEARTBEAT", 30)

# === Storage ===
DEFAULT_STORAGE_DIR = _env_str("ARAGORA_STORAGE_DIR", ".aragora")
MAX_LOG_BYTES = _env_int("ARAGORA_MAX_LOG_BYTES", 100 * 1024)  # 100KB

# === Database ===
DB_TIMEOUT_SECONDS = _env_float("ARAGORA_DB_TIMEOUT", 30.0)

# Database mode: "legacy" (individual DBs) or "consolidated" (4 combined DBs)
# See aragora.persistence.db_config for full configuration
DB_MODE = _env_str("ARAGORA_DB_MODE", "legacy")

# Nomic directory for databases (relative to working directory)
NOMIC_DIR = _env_str("ARAGORA_NOMIC_DIR", ".nomic")

# Legacy database path constants (for backwards compatibility)
# These are deprecated - prefer using get_db_path() from aragora.persistence.db_config
DB_ELO_PATH = _env_str("ARAGORA_DB_ELO", "agent_elo.db")
DB_MEMORY_PATH = _env_str("ARAGORA_DB_MEMORY", "continuum.db")
DB_INSIGHTS_PATH = _env_str("ARAGORA_DB_INSIGHTS", "aragora_insights.db")
DB_CONSENSUS_PATH = _env_str("ARAGORA_DB_CONSENSUS", "consensus_memory.db")
DB_CALIBRATION_PATH = _env_str("ARAGORA_DB_CALIBRATION", "agent_calibration.db")
DB_LAB_PATH = _env_str("ARAGORA_DB_LAB", "persona_lab.db")
DB_PERSONAS_PATH = _env_str("ARAGORA_DB_PERSONAS", "agent_personas.db")
DB_POSITIONS_PATH = _env_str("ARAGORA_DB_POSITIONS", "grounded_positions.db")
DB_GENESIS_PATH = _env_str("ARAGORA_DB_GENESIS", "genesis.db")

# === Evidence Collection ===
MAX_SNIPPETS_PER_CONNECTOR = _env_int("ARAGORA_MAX_SNIPPETS_CONNECTOR", 3)
MAX_TOTAL_SNIPPETS = _env_int("ARAGORA_MAX_TOTAL_SNIPPETS", 8)
SNIPPET_MAX_LENGTH = _env_int("ARAGORA_SNIPPET_MAX_LENGTH", 1000)

# === Deep Audit ===
DEEP_AUDIT_ROUNDS = _env_int("ARAGORA_DEEP_AUDIT_ROUNDS", 6)
CROSS_EXAMINATION_DEPTH = _env_int("ARAGORA_CROSS_EXAM_DEPTH", 3)
RISK_THRESHOLD = _env_float("ARAGORA_RISK_THRESHOLD", 0.7)

# === ELO System ===
ELO_INITIAL_RATING = _env_int("ARAGORA_ELO_INITIAL", 1500)
ELO_K_FACTOR = _env_int("ARAGORA_ELO_K_FACTOR", 32)
ELO_CALIBRATION_MIN_COUNT = _env_int("ARAGORA_ELO_CALIBRATION_MIN_COUNT", 10)

# === Debate Limits ===
MAX_AGENTS_PER_DEBATE = _env_int("ARAGORA_MAX_AGENTS_PER_DEBATE", 10)
MAX_CONCURRENT_DEBATES = _env_int("ARAGORA_MAX_CONCURRENT_DEBATES", 10)
USER_EVENT_QUEUE_SIZE = _env_int("ARAGORA_USER_EVENT_QUEUE_SIZE", 10000)

# === Belief Network ===
BELIEF_MAX_ITERATIONS = _env_int("ARAGORA_BELIEF_MAX_ITERATIONS", 100)
BELIEF_CONVERGENCE_THRESHOLD = _env_float("ARAGORA_BELIEF_CONVERGENCE_THRESHOLD", 0.001)

# === SSL/TLS ===
SSL_ENABLED = _env_bool("ARAGORA_SSL_ENABLED", False)
SSL_CERT_PATH = _env_str("ARAGORA_SSL_CERT", "")
SSL_KEY_PATH = _env_str("ARAGORA_SSL_KEY", "")


# ============================================================================
# Configuration Validation
# ============================================================================

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_configuration(strict: bool = False) -> dict:
    """
    Validate configuration at startup.

    Checks that:
    - Numeric values are in valid ranges
    - Required paths exist (if SSL enabled)
    - At least one API provider is configured (in strict mode)

    Args:
        strict: If True, require at least one API key to be set

    Returns:
        Dict with validation results:
        {
            "valid": True/False,
            "errors": [...],
            "warnings": [...],
            "config_summary": {...}
        }

    Raises:
        ConfigurationError: If strict=True and critical errors found
    """
    import logging
    logger = logging.getLogger(__name__)

    errors = []
    warnings = []

    # Validate numeric ranges
    if DEFAULT_RATE_LIMIT <= 0:
        errors.append(f"ARAGORA_RATE_LIMIT must be positive, got {DEFAULT_RATE_LIMIT}")

    if MAX_ROUNDS < 1:
        errors.append(f"ARAGORA_MAX_ROUNDS must be >= 1, got {MAX_ROUNDS}")

    if DEFAULT_ROUNDS > MAX_ROUNDS:
        warnings.append(f"ARAGORA_DEFAULT_ROUNDS ({DEFAULT_ROUNDS}) > MAX_ROUNDS ({MAX_ROUNDS})")

    if DB_TIMEOUT_SECONDS <= 0:
        errors.append(f"ARAGORA_DB_TIMEOUT must be positive, got {DB_TIMEOUT_SECONDS}")

    if DEBATE_TIMEOUT_SECONDS < 30:
        warnings.append(f"ARAGORA_DEBATE_TIMEOUT is very low ({DEBATE_TIMEOUT_SECONDS}s)")

    if WS_MAX_MESSAGE_SIZE < 1024:
        warnings.append(f"ARAGORA_WS_MAX_MESSAGE_SIZE is very low ({WS_MAX_MESSAGE_SIZE} bytes)")

    if MAX_AGENTS_PER_DEBATE > 20:
        warnings.append(f"ARAGORA_MAX_AGENTS_PER_DEBATE is high ({MAX_AGENTS_PER_DEBATE}), may cause performance issues")

    # Validate SSL configuration if enabled
    if SSL_ENABLED:
        if not SSL_CERT_PATH:
            errors.append("ARAGORA_SSL_ENABLED=true but ARAGORA_SSL_CERT not set")
        elif not os.path.exists(SSL_CERT_PATH):
            errors.append(f"SSL certificate not found: {SSL_CERT_PATH}")

        if not SSL_KEY_PATH:
            errors.append("ARAGORA_SSL_ENABLED=true but ARAGORA_SSL_KEY not set")
        elif not os.path.exists(SSL_KEY_PATH):
            errors.append(f"SSL key not found: {SSL_KEY_PATH}")

    # Check API keys (in strict mode)
    api_keys_found = []
    api_keys_checked = [
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("GEMINI_API_KEY", "Gemini"),
        ("GOOGLE_API_KEY", "Google"),
        ("XAI_API_KEY", "xAI/Grok"),
        ("GROK_API_KEY", "Grok"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
    ]

    for env_var, provider in api_keys_checked:
        if os.getenv(env_var):
            api_keys_found.append(provider)

    if strict and not api_keys_found:
        errors.append("No API keys configured. Set at least one of: " +
                     ", ".join(var for var, _ in api_keys_checked))
    elif not api_keys_found:
        warnings.append("No API keys configured - agent functionality will be limited")

    # Build config summary
    config_summary = {
        "rate_limit": DEFAULT_RATE_LIMIT,
        "debate_timeout": DEBATE_TIMEOUT_SECONDS,
        "max_rounds": MAX_ROUNDS,
        "default_rounds": DEFAULT_ROUNDS,
        "max_agents_per_debate": MAX_AGENTS_PER_DEBATE,
        "ws_max_message_size": WS_MAX_MESSAGE_SIZE,
        "db_timeout": DB_TIMEOUT_SECONDS,
        "ssl_enabled": SSL_ENABLED,
        "api_providers": api_keys_found,
    }

    # Log configuration at startup
    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Configuration validated successfully")
        logger.info(f"  API providers: {', '.join(api_keys_found) if api_keys_found else 'none'}")
        logger.info(f"  Rate limit: {DEFAULT_RATE_LIMIT} req/min")
        logger.info(f"  Debate timeout: {DEBATE_TIMEOUT_SECONDS}s")
        logger.info(f"  SSL: {'enabled' if SSL_ENABLED else 'disabled'}")
    else:
        for error in errors:
            logger.error(f"Configuration error: {error}")
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    result = {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "config_summary": config_summary,
    }

    if strict and errors:
        raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

    return result
