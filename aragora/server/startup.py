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
    from aragora.knowledge.mound.types import MoundConfig

logger = logging.getLogger(__name__)


def _get_config_value(name: str) -> str | None:
    """Get configuration value from environment or secrets manager."""
    import os

    # First check environment
    value = os.environ.get(name)
    if value:
        return value

    # Try secrets manager as fallback
    try:
        from aragora.config.secrets import get_secret

        return get_secret(name)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 - Secret fetch fallback
        return None


def check_connector_dependencies() -> list[str]:
    """Check if connector dependencies are available.

    SECURITY: Connectors fail-closed when dependencies are missing, but this
    function provides early warnings at startup to help operators identify
    misconfiguration before runtime failures occur.

    Returns:
        List of warnings for missing connector dependencies
    """
    import os

    warnings = []

    # Discord webhook verification requires PyNaCl
    if os.environ.get("DISCORD_PUBLIC_KEY") or os.environ.get("DISCORD_WEBHOOK_URL"):
        try:
            import nacl.signing  # noqa: F401
        except ImportError:
            warnings.append(
                "Discord connector configured but PyNaCl not installed. "
                "Webhook signature verification will fail-closed. "
                "Install with: pip install pynacl"
            )

    # Teams/Google Chat webhook verification requires PyJWT
    teams_configured = os.environ.get("TEAMS_TENANT_ID") or os.environ.get("TEAMS_WEBHOOK_URL")
    gchat_configured = os.environ.get("GOOGLE_CHAT_PROJECT") or os.environ.get(
        "GOOGLE_CHAT_WEBHOOK_URL"
    )
    if teams_configured or gchat_configured:
        try:
            import jwt  # noqa: F401
        except ImportError:
            connectors = []
            if teams_configured:
                connectors.append("Teams")
            if gchat_configured:
                connectors.append("Google Chat")
            warnings.append(
                f"{'/'.join(connectors)} connector configured but PyJWT not installed. "
                "Webhook signature verification will fail-closed. "
                "Install with: pip install pyjwt"
            )

    # Slack webhook verification requires signing secret
    if os.environ.get("SLACK_WEBHOOK_URL") and not os.environ.get("SLACK_SIGNING_SECRET"):
        warnings.append(
            "Slack webhook configured but SLACK_SIGNING_SECRET not set. "
            "Webhook signature verification will fail-closed unless "
            "ARAGORA_WEBHOOK_ALLOW_UNVERIFIED=1 is set (not recommended for production)."
        )

    # Slack OAuth configuration validation
    slack_oauth_configured = os.environ.get("SLACK_CLIENT_ID") or os.environ.get(
        "SLACK_CLIENT_SECRET"
    )
    if slack_oauth_configured:
        slack_oauth_issues = []

        if not os.environ.get("SLACK_CLIENT_ID"):
            slack_oauth_issues.append("SLACK_CLIENT_ID")
        if not os.environ.get("SLACK_CLIENT_SECRET"):
            slack_oauth_issues.append("SLACK_CLIENT_SECRET")

        if slack_oauth_issues:
            warnings.append(
                f"Slack OAuth partially configured - missing: {', '.join(slack_oauth_issues)}. "
                "OAuth installation flow will fail without both variables set."
            )

        # Validate SLACK_REDIRECT_URI in production
        is_production = os.environ.get("ARAGORA_ENV", "development") == "production"
        redirect_uri = os.environ.get("SLACK_REDIRECT_URI", "")

        if is_production and not redirect_uri:
            warnings.append(
                "SLACK_REDIRECT_URI not set in production. "
                "Slack OAuth flow may fail or redirect to unexpected URLs."
            )
        elif redirect_uri and not redirect_uri.startswith("https://"):
            if is_production:
                warnings.append(
                    "SLACK_REDIRECT_URI must use HTTPS in production. "
                    f"Current value: {redirect_uri}"
                )

        # Encryption key is recommended for token storage
        if not os.environ.get("ARAGORA_ENCRYPTION_KEY") and is_production:
            warnings.append(
                "ARAGORA_ENCRYPTION_KEY not set - Slack OAuth tokens will be stored "
                "UNENCRYPTED. This is a security risk in production."
            )

    return warnings


def check_agent_credentials(default_agents: str | None = None) -> list[str]:
    """Check for missing API keys required by default agents.

    This validates that environment variables or AWS Secrets Manager provide
    the credentials needed to instantiate the configured default agents.
    """
    from aragora.config.settings import get_settings

    warnings: list[str] = []
    agents_str = default_agents or get_settings().agent.default_agents
    agent_names = [a.strip() for a in agents_str.split(",") if a.strip()]

    # Agents backed by OpenRouter (single key)
    openrouter_agents = {
        "deepseek",
        "kimi",
        "mistral",
        "qwen",
        "qwen-max",
    }
    if any(agent in openrouter_agents for agent in agent_names):
        if not _get_config_value("OPENROUTER_API_KEY"):
            warnings.append(
                "OPENROUTER_API_KEY missing for OpenRouter-backed agents "
                f"({', '.join(sorted(openrouter_agents & set(agent_names)))}). "
                "Set via env or AWS Secrets Manager (ARAGORA_USE_SECRETS_MANAGER=1)."
            )

    # Direct API providers
    if "openai-api" in agent_names and not _get_config_value("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY missing for openai-api agent (env or Secrets Manager).")
    if "anthropic-api" in agent_names and not _get_config_value("ANTHROPIC_API_KEY"):
        warnings.append(
            "ANTHROPIC_API_KEY missing for anthropic-api agent (env or Secrets Manager)."
        )
    if "gemini" in agent_names and not _get_config_value("GEMINI_API_KEY"):
        warnings.append("GEMINI_API_KEY missing for gemini agent (env or Secrets Manager).")
    if "grok" in agent_names and not _get_config_value("XAI_API_KEY"):
        warnings.append("XAI_API_KEY missing for grok agent (env or Secrets Manager).")
    if "mistral-api" in agent_names and not _get_config_value("MISTRAL_API_KEY"):
        warnings.append("MISTRAL_API_KEY missing for mistral-api agent (env or Secrets Manager).")

    return warnings


def check_production_requirements() -> list[str]:
    """Check if production requirements are met.

    SECURITY: This function performs fail-fast validation of production
    configuration to prevent runtime failures and security misconfigurations.

    Environment Variables:
        ARAGORA_ENV: Set to "production" to enable production checks
        ARAGORA_MULTI_INSTANCE: Set to "true" to require Redis for HA
        ARAGORA_REQUIRE_DATABASE: Set to "true" to require PostgreSQL

    Returns:
        List of missing requirements (empty if all met)
    """
    import os

    from aragora.control_plane.leader import is_distributed_state_required

    missing = []
    warnings = []
    env = os.environ.get("ARAGORA_ENV", "development")
    is_production = env == "production"
    distributed_state_required = is_distributed_state_required()
    require_database = os.environ.get("ARAGORA_REQUIRE_DATABASE", "").lower() in (
        "true",
        "1",
        "yes",
    )

    if is_production:
        # =====================================================================
        # HARD REQUIREMENTS (fail startup)
        # =====================================================================

        # Encryption key is required for production
        if not _get_config_value("ARAGORA_ENCRYPTION_KEY"):
            missing.append(
                "ARAGORA_ENCRYPTION_KEY required in production "
                "(32-byte hex string for AES-256 encryption)"
            )

        # Distributed state mode requires Redis
        if distributed_state_required:
            if not os.environ.get("REDIS_URL"):
                missing.append(
                    "REDIS_URL required for distributed state (multi-instance or production). "
                    "Redis is needed for: session store, control-plane leader election, "
                    "debate origins, and distributed caching. "
                    "Set ARAGORA_SINGLE_INSTANCE=true if running single-node."
                )

        # Database requirement (optional flag for strict deployments)
        if require_database:
            if not os.environ.get("DATABASE_URL"):
                missing.append(
                    "DATABASE_URL required when ARAGORA_REQUIRE_DATABASE=true. "
                    "PostgreSQL is needed for: governance store, audit logs, "
                    "and enterprise connector sync."
                )

        # =====================================================================
        # SOFT REQUIREMENTS (warnings)
        # =====================================================================

        # Redis recommended for durable state
        if not distributed_state_required and not os.environ.get("REDIS_URL"):
            warnings.append(
                "REDIS_URL not set - using in-memory state for sessions, "
                "debate origins, and control plane. Data will be lost on restart. "
                "Set ARAGORA_MULTI_INSTANCE=true to make Redis mandatory."
            )

        # PostgreSQL recommended for governance store
        if not require_database and not os.environ.get("DATABASE_URL"):
            warnings.append(
                "DATABASE_URL not set - using SQLite for governance store. "
                "PostgreSQL recommended for production. "
                "Set ARAGORA_REQUIRE_DATABASE=true to make it mandatory."
            )

        # JWT secret should be set for auth
        if not _get_config_value("JWT_SECRET") and not _get_config_value("ARAGORA_JWT_SECRET"):
            warnings.append(
                "JWT_SECRET not set - using derived key from encryption key. "
                "Consider setting JWT_SECRET for independent key rotation."
            )

    # Check connector dependencies (warnings, not errors)
    connector_warnings = check_connector_dependencies()
    warnings.extend(connector_warnings)

    # Check agent API keys for default agents (warnings, not errors)
    agent_warnings = check_agent_credentials()
    warnings.extend(agent_warnings)

    # Log all warnings
    for warning in warnings:
        logger.warning(f"[PRODUCTION CONFIG] {warning}")

    # Log summary
    if is_production:
        if missing:
            logger.error(
                f"[PRODUCTION CONFIG] {len(missing)} critical requirement(s) missing. "
                "Server startup will fail."
            )
        elif warnings:
            logger.warning(
                f"[PRODUCTION CONFIG] {len(warnings)} recommendation(s) not met. "
                "Server will start but may have reduced durability."
            )
        else:
            logger.info("[PRODUCTION CONFIG] All production requirements met.")

    return missing


async def validate_redis_connectivity(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Test Redis connectivity with a PING command.

    This function validates that Redis is actually reachable when required,
    not just that REDIS_URL is configured. This catches common issues like:
    - Network connectivity problems
    - Authentication failures
    - Redis server not running

    Args:
        timeout_seconds: Connection timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    import os

    redis_url = os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL")
    if not redis_url:
        return True, "Redis not configured (skipping connectivity check)"

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            redis_url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        try:
            result = await asyncio.wait_for(client.ping(), timeout=timeout_seconds)
            if result:
                # Check if we can get server info (validates auth)
                info = await asyncio.wait_for(client.info("server"), timeout=timeout_seconds)
                redis_version = info.get("redis_version", "unknown")
                return True, f"Redis connected (version {redis_version})"
            return False, "Redis PING failed"
        finally:
            await client.aclose()
    except ImportError:
        return False, "redis package not installed - run: pip install redis"
    except asyncio.TimeoutError:
        return False, f"Redis connection timed out after {timeout_seconds}s"
    except Exception as e:
        return False, f"Redis connection failed: {e}"


async def init_redis_ha() -> dict[str, Any]:
    """Initialize Redis High-Availability connection.

    Configures the Redis HA client based on environment variables.
    Supports three modes:
    - Standalone: Single Redis instance (development)
    - Sentinel: Redis Sentinel for automatic failover (production HA)
    - Cluster: Redis Cluster for horizontal scaling (enterprise)

    The mode is determined by ARAGORA_REDIS_MODE or auto-detected from
    available configuration (ARAGORA_REDIS_SENTINEL_HOSTS or
    ARAGORA_REDIS_CLUSTER_NODES).

    Environment Variables:
        ARAGORA_REDIS_MODE: Redis mode ("standalone", "sentinel", "cluster")
        ARAGORA_REDIS_SENTINEL_HOSTS: Comma-separated sentinel hosts
        ARAGORA_REDIS_SENTINEL_MASTER: Sentinel master name (default: mymaster)
        ARAGORA_REDIS_CLUSTER_NODES: Comma-separated cluster nodes

    Returns:
        Dictionary with Redis HA initialization status:
        {
            "enabled": bool,
            "mode": str,
            "healthy": bool,
            "description": str,
            "error": Optional[str],
        }
    """
    result: dict[str, Any] = {
        "enabled": False,
        "mode": "standalone",
        "healthy": False,
        "description": "Redis HA not configured",
        "error": None,
    }

    try:
        from aragora.config.redis import get_redis_ha_config
        from aragora.storage.redis_ha import (
            RedisHAConfig,
            RedisMode,
            check_redis_health,
            get_redis_client,
            reset_cached_clients,
        )

        # Get configuration from environment
        config = get_redis_ha_config()
        result["mode"] = config.mode.value
        result["enabled"] = config.enabled

        if not config.enabled and not config.is_configured:
            logger.debug("Redis HA not configured (no Redis URL or HA hosts set)")
            return result

        # Build RedisHAConfig from our settings
        ha_config = RedisHAConfig(
            mode=RedisMode(config.mode.value),
            host=config.host,
            port=config.port,
            password=config.password,
            db=config.db,
            url=config.url,
            sentinel_hosts=config.sentinel_hosts,
            sentinel_master=config.sentinel_master,
            sentinel_password=config.sentinel_password,
            cluster_nodes=config.cluster_nodes,
            cluster_read_from_replicas=config.cluster_read_from_replicas,
            cluster_skip_full_coverage_check=config.cluster_skip_full_coverage_check,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            max_connections=config.max_connections,
            retry_on_timeout=config.retry_on_timeout,
            health_check_interval=config.health_check_interval,
            decode_responses=config.decode_responses,
            ssl=config.ssl,
            ssl_cert_reqs=config.ssl_cert_reqs,
            ssl_ca_certs=config.ssl_ca_certs,
        )

        # Reset any cached clients to pick up new configuration
        reset_cached_clients()

        # Test connection
        health = check_redis_health(ha_config)
        result["healthy"] = health.get("healthy", False)
        result["description"] = config.get_mode_description()

        if health.get("healthy"):
            # Store the client for reuse
            client = get_redis_client(ha_config)
            if client:
                result["enabled"] = True
                logger.info(
                    f"Redis HA initialized: {config.get_mode_description()} "
                    f"(latency={health.get('latency_ms', 'unknown')}ms)"
                )
            else:
                result["error"] = "Failed to create Redis client"
                logger.warning("Redis HA client creation failed")
        else:
            result["error"] = health.get("error", "Unknown error")
            logger.warning(f"Redis HA health check failed: {result['error']}")

    except ImportError as e:
        result["error"] = f"Redis package not installed: {e}"
        logger.debug(f"Redis HA not available: {e}")
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Redis HA initialization failed: {e}")

    return result


async def validate_database_connectivity(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Test PostgreSQL connectivity with a simple query.

    This function validates that PostgreSQL is actually reachable when required,
    not just that DATABASE_URL is configured. This catches common issues like:
    - Network connectivity problems
    - Authentication failures
    - Database server not running
    - Database doesn't exist

    Args:
        timeout_seconds: Connection timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    import os

    database_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
    if not database_url:
        return True, "PostgreSQL not configured (skipping connectivity check)"

    try:
        import asyncpg

        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(database_url, timeout=timeout_seconds),
                timeout=timeout_seconds,
            )
            try:
                # Run a simple query to validate connection
                version = await conn.fetchval("SELECT version()")
                # Extract just the version string (e.g., "PostgreSQL 15.4")
                version_short = version.split(",")[0] if version else "unknown"
                return True, f"PostgreSQL connected ({version_short})"
            finally:
                await conn.close()
        except asyncio.TimeoutError:
            return False, f"PostgreSQL connection timed out after {timeout_seconds}s"
    except ImportError:
        return False, "asyncpg package not installed - run: pip install asyncpg"
    except Exception as e:
        return False, f"PostgreSQL connection failed: {e}"


async def validate_backend_connectivity(
    require_redis: bool = False,
    require_database: bool = False,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Validate connectivity to all configured backends.

    This function should be called during startup after environment validation
    to ensure that configured backends are actually reachable.

    Args:
        require_redis: If True, fail if Redis is not reachable
        require_database: If True, fail if PostgreSQL is not reachable
        timeout_seconds: Timeout for each connectivity test

    Returns:
        Dictionary with connectivity status:
        {
            "valid": True/False,
            "redis": {"connected": bool, "message": str},
            "database": {"connected": bool, "message": str},
            "errors": [str, ...]
        }
    """
    errors: list[str] = []

    # Test Redis connectivity
    redis_ok, redis_msg = await validate_redis_connectivity(timeout_seconds)
    if not redis_ok and require_redis:
        errors.append(f"Redis connectivity required but failed: {redis_msg}")

    # Test database connectivity
    db_ok, db_msg = await validate_database_connectivity(timeout_seconds)
    if not db_ok and require_database:
        errors.append(f"PostgreSQL connectivity required but failed: {db_msg}")

    # Log results
    if redis_ok and "connected" in redis_msg.lower():
        logger.info(f"[BACKEND CHECK] {redis_msg}")
    elif not redis_ok:
        logger.warning(f"[BACKEND CHECK] Redis: {redis_msg}")

    if db_ok and "connected" in db_msg.lower():
        logger.info(f"[BACKEND CHECK] {db_msg}")
    elif not db_ok:
        logger.warning(f"[BACKEND CHECK] PostgreSQL: {db_msg}")

    return {
        "valid": len(errors) == 0,
        "redis": {"connected": redis_ok, "message": redis_msg},
        "database": {"connected": db_ok, "message": db_msg},
        "errors": errors,
    }


def validate_storage_backend() -> dict[str, Any]:
    """Validate storage backend configuration for production.

    This function ensures that the correct storage backend is being used
    in production environments. SQLite is not suitable for multi-instance
    deployments as each server would have its own isolated database.

    Returns:
        Dictionary with validation results:
        {
            "valid": True/False,
            "backend": "supabase" | "postgres" | "sqlite",
            "is_production": True/False,
            "warnings": [str, ...],
            "errors": [str, ...]
        }
    """
    import os

    from aragora.storage.factory import get_storage_backend, StorageBackend

    errors: list[str] = []
    warnings: list[str] = []

    env = os.environ.get("ARAGORA_ENV", "development")
    is_production = env == "production"
    backend = get_storage_backend()
    allow_sqlite = os.environ.get("ARAGORA_ALLOW_SQLITE_FALLBACK", "").lower() in (
        "true",
        "1",
    )

    if is_production and backend == StorageBackend.SQLITE:
        if allow_sqlite:
            warnings.append(
                "SQLite backend used in production with ARAGORA_ALLOW_SQLITE_FALLBACK=true. "
                "This is not recommended for multi-instance deployments. "
                "Users created on one server will not be visible on other servers."
            )
        else:
            errors.append(
                "Production environment requires distributed storage (Supabase or PostgreSQL). "
                "SQLite is not suitable for multi-instance deployments. "
                "Configure SUPABASE_URL + SUPABASE_DB_PASSWORD or ARAGORA_POSTGRES_DSN, "
                "or set ARAGORA_ALLOW_SQLITE_FALLBACK=true to override (not recommended)."
            )

    # Log results
    backend_name = backend.value
    if backend == StorageBackend.SUPABASE:
        logger.info("[STORAGE BACKEND] Using Supabase PostgreSQL (recommended)")
    elif backend == StorageBackend.POSTGRES:
        logger.info("[STORAGE BACKEND] Using self-hosted PostgreSQL")
    else:
        if is_production:
            logger.warning("[STORAGE BACKEND] Using SQLite in production (not recommended)")
        else:
            logger.info("[STORAGE BACKEND] Using SQLite (development mode)")

    for warning in warnings:
        logger.warning(f"[STORAGE BACKEND] {warning}")
    for error in errors:
        logger.error(f"[STORAGE BACKEND] {error}")

    return {
        "valid": len(errors) == 0,
        "backend": backend_name,
        "is_production": is_production,
        "warnings": warnings,
        "errors": errors,
    }


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


async def init_otlp_exporter() -> bool:
    """Initialize OpenTelemetry OTLP exporter for distributed tracing.

    Configures trace export to external backends like Jaeger, Zipkin,
    or Datadog via the OTLP protocol. This is separate from the basic
    OpenTelemetry setup and provides more flexible backend options.

    Environment Variables:
        ARAGORA_OTLP_EXPORTER: Exporter type (none, jaeger, zipkin, otlp_grpc, otlp_http, datadog)
        ARAGORA_OTLP_ENDPOINT: Collector endpoint URL
        ARAGORA_SERVICE_NAME: Service name for traces (default: aragora)
        ARAGORA_ENVIRONMENT: Deployment environment (default: development)
        ARAGORA_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
        See docs/ENVIRONMENT.md for full configuration reference.

    Returns:
        True if OTLP exporter was configured, False otherwise
    """
    try:
        from aragora.observability.config import is_otlp_enabled
        from aragora.observability.otlp_export import configure_otlp_exporter, get_otlp_config

        if not is_otlp_enabled():
            logger.debug(
                "OTLP exporter disabled (set ARAGORA_OTLP_EXPORTER to jaeger/zipkin/otlp_grpc/otlp_http/datadog)"
            )
            return False

        config = get_otlp_config()
        provider = configure_otlp_exporter(config)

        if provider:
            logger.info(
                f"OTLP exporter initialized: type={config.exporter_type.value}, "
                f"endpoint={config.get_effective_endpoint()}"
            )
            return True
        else:
            logger.warning("OTLP exporter configuration failed")
            return False

    except ImportError as e:
        logger.debug(f"OTLP exporter not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize OTLP exporter: {e}")

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
    - Policy management (automatic policy sync from compliance store)

    Policy sync is enabled by default and controlled by ARAGORA_POLICY_SYNC_ON_STARTUP
    (or CP_ENABLE_POLICY_SYNC for backward compatibility). Set to "false" to disable.

    Returns:
        Connected ControlPlaneCoordinator, or None if initialization fails
    """
    try:
        from aragora.control_plane.coordinator import ControlPlaneCoordinator

        coordinator = await ControlPlaneCoordinator.create()

        # Log policy manager status
        if coordinator.policy_manager:
            policy_count = (
                len(coordinator.policy_manager._policies)
                if hasattr(coordinator.policy_manager, "_policies")
                else 0
            )
            logger.info(
                f"Control Plane coordinator initialized and connected "
                f"(policies_loaded={policy_count})"
            )
        else:
            logger.info("Control Plane coordinator initialized and connected (no policy manager)")

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
        from aragora.control_plane.shared_state import get_shared_state

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


# Global witness instance for handlers to access
_witness_behavior: Optional[Any] = None


def get_witness_behavior() -> Optional[Any]:
    """Get the global witness behavior instance.

    Returns:
        WitnessBehavior instance if initialized, None otherwise
    """
    return _witness_behavior


async def init_witness_patrol() -> bool:
    """Initialize and start the Witness Patrol for Gas Town monitoring.

    Creates the WitnessBehavior with an AgentHierarchy and starts the patrol
    loop which monitors:
    - Agent health and heartbeats
    - Bead progress and stuck detection
    - Convoy completion rates
    - Automatic escalation to MAYOR on critical issues

    The witness instance is stored globally and can be accessed via
    get_witness_behavior() for the status endpoint.

    Returns:
        True if patrol started successfully, False otherwise
    """
    global _witness_behavior

    try:
        from aragora.nomic.witness_behavior import WitnessBehavior, WitnessConfig
        from aragora.nomic.agent_roles import AgentHierarchy

        # Create hierarchy with default persistence directory
        hierarchy = AgentHierarchy()

        # Configure witness with reasonable defaults
        config = WitnessConfig(
            patrol_interval_seconds=30,
            heartbeat_timeout_seconds=120,
            notify_mayor_on_critical=True,
        )

        # Create witness behavior
        witness = WitnessBehavior(
            hierarchy=hierarchy,
            config=config,
        )

        # Start the patrol loop
        await witness.start_patrol()

        _witness_behavior = witness
        logger.info("Witness patrol started successfully")
        return True

    except ImportError as e:
        logger.debug(f"Witness behavior not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Witness patrol initialization failed: {e}")
        return False


# Global mayor coordinator instance
_mayor_coordinator: Optional[Any] = None


def get_mayor_coordinator() -> Optional[Any]:
    """Get the global mayor coordinator instance.

    Returns:
        MayorCoordinator instance if initialized, None otherwise
    """
    return _mayor_coordinator


async def init_mayor_coordinator() -> bool:
    """Initialize the Mayor Coordinator for distributed leadership.

    Creates the MayorCoordinator which bridges leader election with the
    Gas Town agent hierarchy:
    - When this node wins election, it becomes MAYOR
    - When it loses election, it demotes to WITNESS
    - Provides current mayor info via get_mayor_coordinator()

    Returns:
        True if coordinator started successfully, False otherwise
    """
    global _mayor_coordinator

    try:
        from aragora.nomic.mayor_coordinator import MayorCoordinator
        from aragora.nomic.agent_roles import AgentHierarchy
        import os

        # Get node ID from environment or generate one
        node_id = os.environ.get("ARAGORA_NODE_ID")
        region = os.environ.get("ARAGORA_REGION")

        # Create hierarchy (will be shared with witness if both are initialized)
        hierarchy = AgentHierarchy()

        # Create and start coordinator
        coordinator = MayorCoordinator(
            hierarchy=hierarchy,
            node_id=node_id,
            region=region,
        )

        if await coordinator.start():
            _mayor_coordinator = coordinator
            is_mayor = "yes" if coordinator.is_mayor else "no"
            logger.info(
                f"Mayor coordinator started (node={coordinator.node_id}, "
                f"is_mayor={is_mayor}, region={region or 'global'})"
            )
            return True

        return False

    except ImportError as e:
        logger.debug(f"Mayor coordinator not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Mayor coordinator initialization failed: {e}")
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
            PersistentTaskQueue,  # noqa: F401
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

            create_km_bridge()
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


async def init_notification_worker() -> bool:
    """Initialize the notification dispatcher worker for queue processing.

    Starts the background worker that processes queued notifications with
    retry logic, circuit breakers, and dead letter queue support.

    Requires Redis for queue persistence. If Redis is not available, the
    worker will not start but notifications can still be sent synchronously.

    Environment Variables:
        REDIS_URL: Redis connection URL for queue persistence
        ARAGORA_NOTIFICATION_WORKER: Set to "0" to disable (enabled by default)

    Returns:
        True if worker was started, False otherwise
    """
    import os

    # Enabled by default - set to "0" to disable
    if os.environ.get("ARAGORA_NOTIFICATION_WORKER", "1").lower() in ("0", "false", "no"):
        logger.debug("Notification worker not started (ARAGORA_NOTIFICATION_WORKER disabled)")
        return False

    # Check if Redis is available
    redis_url = os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL")
    if not redis_url:
        logger.debug("Notification worker not started (no REDIS_URL configured)")
        return False

    try:
        import redis.asyncio as aioredis

        from aragora.control_plane.notifications import (
            create_notification_dispatcher,
            NotificationDispatcherConfig,
            set_default_notification_dispatcher,
        )
        from aragora.control_plane.channels import NotificationManager

        # Connect to Redis
        redis_client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

        # Test connection
        await redis_client.ping()

        # Create manager with Redis persistence and load persisted channel configs
        manager = NotificationManager(redis_client=redis_client)
        channel_count = await manager.load_channels()

        config = NotificationDispatcherConfig(
            queue_enabled=True,
            max_concurrent_deliveries=int(os.environ.get("ARAGORA_NOTIFICATION_CONCURRENCY", "20")),
        )
        dispatcher = create_notification_dispatcher(
            manager=manager,
            redis=redis_client,
            config=config,
        )

        # Start the worker
        await dispatcher.start_worker()
        set_default_notification_dispatcher(dispatcher)

        # Wire dispatcher to task events
        from aragora.control_plane.task_events import set_task_event_dispatcher

        set_task_event_dispatcher(dispatcher)
        logger.info("task_event_dispatcher_wired")

        logger.info(
            f"Notification worker started "
            f"(concurrency={config.max_concurrent_deliveries}, channels_loaded={channel_count})"
        )
        return True

    except ImportError as e:
        logger.debug(f"Notification worker dependencies not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start notification worker: {e}")

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


async def init_backup_scheduler() -> bool:
    """Initialize the backup scheduler for automated backups and DR drills.

    Starts the backup scheduler that runs scheduled backups at configured
    intervals and performs disaster recovery verification drills.

    Environment Variables:
        BACKUP_ENABLED: Set to "true" to enable backup scheduling (default: false)
        BACKUP_DIR: Directory for backup storage (default: ~/.aragora/backups)
        BACKUP_DAILY_TIME: Time for daily backups in HH:MM format (default: 02:00)
        BACKUP_DR_DRILL_ENABLED: Enable DR drills (default: true)
        BACKUP_DR_DRILL_INTERVAL_DAYS: Days between DR drills (default: 30)

    Returns:
        True if scheduler was started, False otherwise
    """
    import os
    from datetime import time as dt_time

    # Check if backup scheduling is enabled
    if os.environ.get("BACKUP_ENABLED", "false").lower() not in ("true", "1", "yes"):
        logger.debug("Backup scheduler disabled (set BACKUP_ENABLED=true to enable)")
        return False

    try:
        from aragora.backup.manager import get_backup_manager
        from aragora.backup.scheduler import (
            BackupSchedule,
            start_backup_scheduler,
        )

        # Parse configuration from environment
        backup_dir = os.environ.get("BACKUP_DIR")

        # Parse daily backup time (HH:MM format)
        daily_time_str = os.environ.get("BACKUP_DAILY_TIME", "02:00")
        try:
            hour, minute = map(int, daily_time_str.split(":"))
            daily_time = dt_time(hour, minute)
        except (ValueError, TypeError):
            logger.warning(f"Invalid BACKUP_DAILY_TIME '{daily_time_str}', using 02:00")
            daily_time = dt_time(2, 0)

        # Parse DR drill settings
        dr_drills_enabled = os.environ.get("BACKUP_DR_DRILL_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        dr_drill_interval = int(os.environ.get("BACKUP_DR_DRILL_INTERVAL_DAYS", "30"))

        # Get or create the backup manager
        manager = get_backup_manager(backup_dir)

        # Create schedule configuration
        schedule = BackupSchedule(
            daily=daily_time,
            enable_dr_drills=dr_drills_enabled,
            dr_drill_interval_days=dr_drill_interval,
        )

        # Start the scheduler
        await start_backup_scheduler(manager, schedule)

        logger.info(
            f"Backup scheduler started "
            f"(daily={daily_time_str}, "
            f"dr_drills={'enabled' if dr_drills_enabled else 'disabled'}, "
            f"dr_interval={dr_drill_interval}d)"
        )
        return True

    except ImportError as e:
        logger.debug(f"Backup scheduler not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start backup scheduler: {e}")

    return False


def _get_degraded_status() -> dict[str, Any]:
    """Return a minimal status dict for degraded mode startup.

    This allows the server to start in degraded mode where it can
    respond to health checks but returns 503 for other endpoints.
    """
    return {
        "degraded": True,
        "backend_connectivity": {"valid": False, "errors": ["Server in degraded mode"]},
        "error_monitoring": False,
        "opentelemetry": False,
        "otlp_exporter": False,
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
        "key_rotation_scheduler": False,
        "access_review_scheduler": False,
        "rbac_distributed_cache": False,
        "notification_worker": False,
        "graphql": False,
        "backup_scheduler": False,
    }


async def run_startup_sequence(
    nomic_dir: Optional[Path] = None,
    stream_emitter: Optional[Any] = None,
    graceful_degradation: bool = True,
) -> dict:
    """Run the full server startup sequence.

    Args:
        nomic_dir: Path to nomic directory
        stream_emitter: Optional event emitter for debates
        graceful_degradation: If True, enter degraded mode on failure instead of crashing.
            The server will start but return 503 for most endpoints until the issue is resolved.
            Defaults to True for production resilience.

    Returns:
        Dictionary with startup status for each component. If graceful_degradation is True
        and startup fails, returns a minimal status dict with degraded=True.

    Raises:
        RuntimeError: If production requirements are not met AND graceful_degradation is False
    """
    import os

    from aragora.control_plane.leader import is_distributed_state_required
    from aragora.server.degraded_mode import (
        set_degraded,
        DegradedErrorCode,
    )

    # Check production requirements first (fail fast or enter degraded mode)
    missing_requirements = check_production_requirements()
    if missing_requirements:
        for req in missing_requirements:
            logger.error(f"Missing production requirement: {req}")

        error_msg = f"Production requirements not met: {', '.join(missing_requirements)}"

        if graceful_degradation:
            # Determine the error code based on what's missing
            error_code = DegradedErrorCode.CONFIG_ERROR
            if any("ENCRYPTION_KEY" in r for r in missing_requirements):
                error_code = DegradedErrorCode.ENCRYPTION_KEY_MISSING
            elif any("REDIS" in r for r in missing_requirements):
                error_code = DegradedErrorCode.REDIS_UNAVAILABLE
            elif any("DATABASE" in r for r in missing_requirements):
                error_code = DegradedErrorCode.DATABASE_UNAVAILABLE

            set_degraded(
                reason=error_msg,
                error_code=error_code,
                details={"missing_requirements": missing_requirements},
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Validate actual backend connectivity (not just config presence)
    env = os.environ.get("ARAGORA_ENV", "development")
    is_production = env == "production"
    distributed_required = is_distributed_state_required()
    require_database = os.environ.get("ARAGORA_REQUIRE_DATABASE", "").lower() in (
        "true",
        "1",
        "yes",
    )

    connectivity = await validate_backend_connectivity(
        require_redis=distributed_required,
        require_database=require_database,
        timeout_seconds=10.0 if is_production else 5.0,
    )

    if not connectivity["valid"]:
        for error in connectivity["errors"]:
            logger.error(f"Backend connectivity failure: {error}")

        error_msg = f"Backend connectivity validation failed: {'; '.join(connectivity['errors'])}"

        if graceful_degradation:
            # Determine error code based on what failed
            error_code = DegradedErrorCode.BACKEND_CONNECTIVITY
            if any("Redis" in e for e in connectivity["errors"]):
                error_code = DegradedErrorCode.REDIS_UNAVAILABLE
            elif any("PostgreSQL" in e for e in connectivity["errors"]):
                error_code = DegradedErrorCode.DATABASE_UNAVAILABLE

            set_degraded(
                reason=error_msg,
                error_code=error_code,
                details={
                    "connectivity": connectivity,
                    "distributed_required": distributed_required,
                    "require_database": require_database,
                },
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Validate storage backend configuration
    storage_backend = validate_storage_backend()
    if not storage_backend["valid"]:
        for error in storage_backend["errors"]:
            logger.error(f"Storage backend error: {error}")

        error_msg = f"Storage backend validation failed: {'; '.join(storage_backend['errors'])}"

        if graceful_degradation:
            set_degraded(
                reason=error_msg,
                error_code=DegradedErrorCode.DATABASE_UNAVAILABLE,
                details={"storage_backend": storage_backend},
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Validate database schema (in consolidated mode)
    schema_validation = {"success": True, "errors": [], "warnings": []}
    try:
        from aragora.persistence.validator import validate_consolidated_schema

        schema_result = validate_consolidated_schema()
        schema_validation = {
            "success": schema_result.success,
            "errors": schema_result.errors,
            "warnings": schema_result.warnings,
        }

        if not schema_result.success:
            for error in schema_result.errors:
                logger.error(f"Database schema validation error: {error}")

            # Only fail if explicitly required (production environments may need migration)
            require_valid_schema = os.environ.get("ARAGORA_REQUIRE_VALID_SCHEMA", "").lower() in (
                "true",
                "1",
                "yes",
            )

            if require_valid_schema:
                error_msg = f"Database schema validation failed: {'; '.join(schema_result.errors)}"

                if graceful_degradation:
                    set_degraded(
                        reason=error_msg,
                        error_code=DegradedErrorCode.DATABASE_UNAVAILABLE,
                        details={"schema_validation": schema_validation},
                    )
                    return _get_degraded_status()

                raise RuntimeError(error_msg)
            else:
                logger.warning(
                    "Database schema validation failed but ARAGORA_REQUIRE_VALID_SCHEMA not set. "
                    "Server will start with potentially incomplete schema. "
                    "Run: python -m aragora.persistence.migrations.consolidate --migrate"
                )
        else:
            for warning in schema_result.warnings:
                logger.warning(f"Database schema: {warning}")

    except ImportError:
        logger.debug("Database validator not available - skipping schema validation")

    import time as time_mod

    status: dict[str, Any] = {
        "_startup_start_time": time_mod.time(),  # For duration calculation
        "backend_connectivity": connectivity,
        "storage_backend": storage_backend,
        "schema_validation": schema_validation,
        "redis_ha": {"enabled": False, "mode": "standalone", "healthy": False},
        "error_monitoring": False,
        "opentelemetry": False,
        "otlp_exporter": False,
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
        "key_rotation_scheduler": False,
        "access_review_scheduler": False,
        "rbac_distributed_cache": False,
        "notification_worker": False,
        "graphql": False,
        "backup_scheduler": False,
        "witness_patrol": False,
        "mayor_coordinator": False,
    }

    # Initialize Redis HA early (other components may depend on it)
    status["redis_ha"] = await init_redis_ha()

    # Initialize in parallel where possible
    status["error_monitoring"] = await init_error_monitoring()
    status["opentelemetry"] = await init_opentelemetry()
    status["otlp_exporter"] = await init_otlp_exporter()
    status["prometheus"] = await init_prometheus_metrics()

    # Sequential initialization for components with dependencies
    status["circuit_breakers"] = init_circuit_breaker_persistence(nomic_dir)
    status["background_tasks"] = init_background_tasks(nomic_dir)
    status["pulse_scheduler"] = await init_pulse_scheduler(stream_emitter)
    status["state_cleanup"] = init_state_cleanup_task()
    status["watchdog_task"] = await init_stuck_debate_watchdog()
    status["control_plane_coordinator"] = await init_control_plane_coordinator()
    status["shared_control_plane_state"] = await init_shared_control_plane_state()

    # Initialize Witness Patrol for Gas Town agent monitoring
    status["witness_patrol"] = await init_witness_patrol()

    # Initialize Mayor Coordinator for distributed leadership
    status["mayor_coordinator"] = await init_mayor_coordinator()

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

    # Start backup scheduler for automated backups and DR drills (if enabled)
    status["backup_scheduler"] = await init_backup_scheduler()

    # Start notification dispatcher worker for queue processing
    status["notification_worker"] = await init_notification_worker()

    # Initialize Redis state backend for horizontal scaling
    status["redis_state_backend"] = await init_redis_state_backend()

    # Initialize DecisionRouter with platform response handlers
    status["decision_router"] = await init_decision_router()

    # Initialize key rotation scheduler for automated encryption key management
    status["key_rotation_scheduler"] = await init_key_rotation_scheduler()

    # Initialize access review scheduler for SOC 2 CC6.1/CC6.2 compliance
    status["access_review_scheduler"] = await init_access_review_scheduler()

    # Initialize RBAC distributed cache for horizontal scaling
    status["rbac_distributed_cache"] = await init_rbac_distributed_cache()

    # Recover pending approval requests from governance store
    status["approval_gate_recovery"] = await init_approval_gate_recovery()

    # Initialize GraphQL API routes (if enabled)
    status["graphql"] = init_graphql_routes(None)

    # Run comprehensive deployment validation and log results
    status["deployment_validation"] = await init_deployment_validation()

    # Record startup completion time and store report
    import time as time_mod

    startup_end_time = time_mod.time()
    startup_duration = startup_end_time - (status.get("_startup_start_time", startup_end_time))

    try:
        from aragora.server.startup_transaction import (
            StartupReport,
            set_last_startup_report,
        )

        # Generate report from status
        components_initialized = [k for k, v in status.items() if v and not k.startswith("_")]
        components_failed = [k for k, v in status.items() if v is False]

        report = StartupReport(
            success=len(components_failed) == 0,
            total_duration_seconds=startup_duration,
            slo_seconds=30.0,
            slo_met=startup_duration <= 30.0,
            components_initialized=len(components_initialized),
            components_failed=components_failed,
            checkpoints=[],
            error=None,
        )
        set_last_startup_report(report)

        if startup_duration > 30.0:
            logger.warning(
                f"[STARTUP] Completed in {startup_duration:.2f}s (exceeds 30s SLO target)"
            )
        else:
            logger.info(
                f"[STARTUP] Completed in {startup_duration:.2f}s "
                f"({len(components_initialized)} components)"
            )

        status["startup_report"] = report.to_dict()

    except ImportError:
        logger.debug("startup_transaction module not available - skipping report")
        status["startup_duration_seconds"] = round(startup_duration, 2)

    return status


async def init_deployment_validation() -> dict:
    """Run comprehensive deployment validation and log results.

    This validates all production requirements including:
    - JWT secret strength and uniqueness
    - AI provider API key configuration
    - Database connectivity (Supabase/PostgreSQL)
    - Redis configuration for distributed state
    - CORS and security settings
    - Rate limiting configuration
    - TLS/HTTPS settings
    - Encryption key configuration

    Returns:
        Dictionary with validation results summary
    """
    try:
        from aragora.ops.deployment_validator import validate_deployment, Severity

        result = await validate_deployment()

        # Log validation results
        critical_count = sum(1 for i in result.issues if i.severity == Severity.CRITICAL)
        warning_count = sum(1 for i in result.issues if i.severity == Severity.WARNING)
        info_count = sum(1 for i in result.issues if i.severity == Severity.INFO)

        if result.ready:
            if warning_count > 0:
                logger.info(
                    f"[DEPLOYMENT VALIDATION] Passed with {warning_count} warning(s), "
                    f"{info_count} info message(s). Duration: {result.validation_duration_ms:.1f}ms"
                )
            else:
                logger.info(
                    f"[DEPLOYMENT VALIDATION] All checks passed. "
                    f"Duration: {result.validation_duration_ms:.1f}ms"
                )
        else:
            logger.warning(
                f"[DEPLOYMENT VALIDATION] {critical_count} critical issue(s), "
                f"{warning_count} warning(s). Server may not function correctly."
            )

        # Log critical issues
        for issue in result.issues:
            if issue.severity == Severity.CRITICAL:
                logger.error(
                    f"[DEPLOYMENT VALIDATION] CRITICAL - {issue.component}: {issue.message}"
                )
                if issue.suggestion:
                    logger.error(f"  Suggestion: {issue.suggestion}")
            elif issue.severity == Severity.WARNING:
                logger.warning(
                    f"[DEPLOYMENT VALIDATION] WARNING - {issue.component}: {issue.message}"
                )

        return {
            "ready": result.ready,
            "live": result.live,
            "critical_issues": critical_count,
            "warnings": warning_count,
            "info_messages": info_count,
            "validation_duration_ms": result.validation_duration_ms,
            "components_checked": len(result.components),
        }

    except ImportError as e:
        logger.debug(f"Deployment validator not available: {e}")
        return {"available": False, "error": str(e)}
    except Exception as e:
        logger.warning(f"Deployment validation failed: {e}")
        return {"available": True, "error": str(e)}


def init_graphql_routes(app: Any) -> bool:
    """Initialize GraphQL routes and mount endpoints.

    Mounts the GraphQL API endpoint at /graphql and the GraphiQL playground
    at /graphiql (when enabled). The routes are only mounted if GraphQL is
    enabled via ARAGORA_GRAPHQL_ENABLED environment variable.

    Args:
        app: The application or handler registry to mount routes on.
              For UnifiedServer, this is typically the handler registry.

    Environment Variables:
        ARAGORA_GRAPHQL_ENABLED: Enable GraphQL API (default: true)
        ARAGORA_GRAPHQL_INTROSPECTION: Allow schema introspection (default: true in dev, false in prod)
        ARAGORA_GRAPHIQL_ENABLED: Enable GraphiQL playground (default: same as dev mode)

    Returns:
        True if GraphQL routes were mounted, False otherwise
    """
    import os

    # Check if GraphQL is enabled
    graphql_enabled = os.environ.get("ARAGORA_GRAPHQL_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    if not graphql_enabled:
        logger.info("GraphQL API disabled (ARAGORA_GRAPHQL_ENABLED=false)")
        return False

    try:
        from aragora.server.graphql import GraphQLHandler, GraphQLSchemaHandler  # noqa: F401

        # Determine environment for defaults
        env = os.environ.get("ARAGORA_ENV", "development")
        is_production = env == "production"

        # Check introspection and GraphiQL settings
        introspection_default = "false" if is_production else "true"
        introspection_enabled = os.environ.get(
            "ARAGORA_GRAPHQL_INTROSPECTION", introspection_default
        ).lower() in ("true", "1", "yes", "on")

        graphiql_default = "false" if is_production else "true"
        graphiql_enabled = os.environ.get("ARAGORA_GRAPHIQL_ENABLED", graphiql_default).lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

        # Log configuration
        logger.info(
            f"GraphQL API enabled (introspection={introspection_enabled}, "
            f"graphiql={graphiql_enabled})"
        )

        # The handlers are auto-registered via the handler registry pattern
        # when they define ROUTES class attribute. We just need to ensure
        # they can be imported and the module is loaded.

        # Log mounted endpoints
        logger.info("  POST /graphql - GraphQL query endpoint")
        logger.info("  POST /api/graphql - GraphQL query endpoint (alternate)")
        logger.info("  POST /api/v1/graphql - GraphQL query endpoint (versioned)")

        if introspection_enabled:
            logger.info("  GET /graphql/schema - Schema introspection endpoint")

        if graphiql_enabled:
            logger.info("  GET /graphql - GraphiQL playground")

        return True

    except ImportError as e:
        logger.warning(f"GraphQL module not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize GraphQL routes: {e}")
        return False


async def init_rbac_distributed_cache() -> bool:
    """Initialize Redis-backed RBAC cache for distributed deployments.

    Enables cross-instance RBAC decision caching for horizontal scaling.
    Only initializes if Redis is available.

    Environment Variables:
        REDIS_URL: Redis connection URL
        RBAC_CACHE_ENABLED: Set to "false" to disable (default: true)
        RBAC_CACHE_DECISION_TTL: Cache TTL for decisions (default: 300s)
        RBAC_CACHE_L1_ENABLED: Enable local L1 cache (default: true)

    Returns:
        True if distributed cache was initialized, False otherwise
    """
    import os

    # Check if RBAC cache is enabled
    if os.environ.get("RBAC_CACHE_ENABLED", "true").lower() == "false":
        logger.debug("RBAC distributed cache disabled (RBAC_CACHE_ENABLED=false)")
        return False

    # Check if Redis URL is configured
    redis_url = os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL")
    if not redis_url:
        logger.debug("RBAC distributed cache not initialized (no REDIS_URL)")
        return False

    try:
        from aragora.rbac.cache import (
            RBACCacheConfig,
            RBACDistributedCache,  # noqa: F401
            get_rbac_cache,
        )
        from aragora.rbac.checker import (
            PermissionChecker,
            get_permission_checker,
            set_permission_checker,
        )

        # Create cache config from environment
        config = RBACCacheConfig.from_env()

        # Initialize distributed cache
        cache = get_rbac_cache(config)
        cache.start()

        # Check if Redis is actually available
        if not cache.is_distributed:
            logger.debug("RBAC cache Redis not available, using local-only")
            return False

        # Create new permission checker with distributed cache backend
        current_checker = get_permission_checker()
        new_checker = PermissionChecker(
            auditor=current_checker._auditor if hasattr(current_checker, "_auditor") else None,
            cache_ttl=config.decision_ttl_seconds,
            enable_cache=True,
            cache_backend=cache,
        )
        set_permission_checker(new_checker)

        logger.info(
            f"RBAC distributed cache initialized "
            f"(decision_ttl={config.decision_ttl_seconds}s, "
            f"l1={'enabled' if config.l1_enabled else 'disabled'})"
        )
        return True

    except ImportError as e:
        logger.debug(f"RBAC distributed cache not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize RBAC distributed cache: {e}")

    return False


async def init_approval_gate_recovery() -> int:
    """Recover pending approval requests from the governance store.

    Restores any pending approval requests that were active when the server
    last stopped. Approvals that have expired since then are automatically
    marked as expired.

    Returns:
        Number of pending approvals recovered
    """
    try:
        from aragora.server.middleware.approval_gate import recover_pending_approvals

        recovered = await recover_pending_approvals()
        if recovered > 0:
            logger.info(f"Recovered {recovered} pending approval requests")
        return recovered

    except ImportError as e:
        logger.debug(f"Approval gate recovery not available: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Failed to recover pending approvals: {e}")
        return 0


async def init_access_review_scheduler() -> bool:
    """Initialize the access review scheduler for SOC 2 compliance.

    Starts the scheduler that runs periodic access reviews:
    - Monthly user access reviews
    - Weekly stale credential detection (90+ days unused)
    - Role certification workflows
    - Manager sign-off requirements

    SOC 2 Compliance: CC6.1, CC6.2 (Access Control)

    Environment Variables:
        ARAGORA_ACCESS_REVIEW_ENABLED: Set to "true" to enable (default: true in production)
        ARAGORA_ACCESS_REVIEW_STORAGE: Path for SQLite storage (default: data/access_reviews.db)

    Returns:
        True if scheduler was started, False otherwise
    """
    import os

    enabled = os.environ.get("ARAGORA_ACCESS_REVIEW_ENABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    is_production = os.environ.get("ARAGORA_ENV", "development") == "production"

    # Enable by default in production
    if not enabled and is_production:
        enabled = True

    if not enabled:
        logger.debug(
            "Access review scheduler disabled (set ARAGORA_ACCESS_REVIEW_ENABLED=true to enable)"
        )
        return False

    try:
        from aragora.scheduler.access_review_scheduler import (
            AccessReviewConfig,
            get_access_review_scheduler,
        )

        # Configure storage path
        storage_path = os.environ.get("ARAGORA_ACCESS_REVIEW_STORAGE")
        if not storage_path:
            from aragora.persistence.db_config import get_nomic_dir

            data_dir = get_nomic_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            storage_path = str(data_dir / "access_reviews.db")

        config = AccessReviewConfig(storage_path=storage_path)

        # Get or create the global scheduler
        scheduler = get_access_review_scheduler(config)

        # Start the scheduler
        await scheduler.start()

        logger.info(
            f"Access review scheduler started "
            f"(storage={storage_path}, monthly_review_day={config.monthly_review_day})"
        )
        return True

    except ImportError as e:
        logger.debug(f"Access review scheduler not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to start access review scheduler: {e}")
        return False


async def init_key_rotation_scheduler() -> bool:
    """Initialize the key rotation scheduler for automated key management.

    Starts the scheduler that automatically rotates encryption keys based on
    configured intervals and handles re-encryption of data if enabled.

    Environment Variables:
        ARAGORA_KEY_ROTATION_ENABLED: Set to "true" to enable (default: false in dev, true in prod)
        ARAGORA_KEY_ROTATION_INTERVAL_DAYS: Days between rotations (default: 90)
        ARAGORA_KEY_ROTATION_OVERLAP_DAYS: Days to keep old keys valid (default: 7)
        ARAGORA_KEY_ROTATION_RE_ENCRYPT: Re-encrypt data after rotation (default: false)
        ARAGORA_KEY_ROTATION_ALERT_DAYS: Days before rotation to alert (default: 7)

    Returns:
        True if scheduler was started, False otherwise
    """
    import os

    # In production, enabled by default; in development, disabled by default
    env = os.environ.get("ARAGORA_ENV", "development")
    default_enabled = "true" if env == "production" else "false"
    enabled = os.environ.get("ARAGORA_KEY_ROTATION_ENABLED", default_enabled).lower() == "true"

    if not enabled:
        logger.debug(
            "Key rotation scheduler disabled (set ARAGORA_KEY_ROTATION_ENABLED=true to enable)"
        )
        return False

    # Check if encryption key is configured
    if not os.environ.get("ARAGORA_ENCRYPTION_KEY"):
        logger.debug("Key rotation scheduler not started (no ARAGORA_ENCRYPTION_KEY configured)")
        return False

    try:
        from aragora.operations.key_rotation import (
            get_key_rotation_scheduler,
            KeyRotationConfig,
        )
        from aragora.observability.metrics.security import set_active_keys

        # Create scheduler with config from environment
        config = KeyRotationConfig.from_env()
        scheduler = get_key_rotation_scheduler()
        scheduler.config = config

        # Set up alert callback to integrate with notification systems
        def alert_callback(severity: str, message: str, details: dict) -> None:
            """Forward key rotation alerts to notification systems."""
            try:
                from aragora.integrations.webhooks import get_webhook_dispatcher

                dispatcher = get_webhook_dispatcher()
                if dispatcher:
                    dispatcher.enqueue(
                        {
                            "type": "security.key_rotation",
                            "severity": severity,
                            "message": message,
                            **details,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to dispatch key rotation alert: {e}")

        scheduler.alert_callback = alert_callback

        # Start the scheduler
        await scheduler.start()

        # Set initial key metrics
        try:
            from aragora.security.encryption import get_encryption_service

            service = get_encryption_service()
            active_key_id = service.get_active_key_id()
            if active_key_id:
                set_active_keys(master=1)
        except Exception:
            pass

        # Get initial status for logging
        status = await scheduler.get_status()
        next_rotation = status.get("next_rotation", "unknown")

        logger.info(
            f"Key rotation scheduler started "
            f"(interval={config.rotation_interval_days}d, "
            f"overlap={config.key_overlap_days}d, "
            f"re_encrypt={config.re_encrypt_on_rotation})"
        )

        if next_rotation and next_rotation != "unknown":
            logger.info(f"Next key rotation scheduled: {next_rotation}")

        return True

    except ImportError as e:
        logger.debug(f"Key rotation scheduler not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to start key rotation scheduler: {e}")

    return False


async def init_decision_router() -> bool:
    """Initialize the DecisionRouter with platform response handlers.

    Registers response handlers for all supported platforms so the router
    can deliver debate results back to the originating channel.

    Returns:
        True if initialization succeeded
    """
    try:
        from aragora.core.decision import get_decision_router
        from aragora.server.debate_origin import route_debate_result

        router = get_decision_router()

        # Register platform response handlers
        # These handlers use route_debate_result to deliver results
        # back to the originating channel

        async def telegram_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "telegram":
                await route_debate_result(result.request_id, result.to_dict())

        async def slack_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "slack":
                await route_debate_result(result.request_id, result.to_dict())

        async def discord_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "discord":
                await route_debate_result(result.request_id, result.to_dict())

        async def whatsapp_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "whatsapp":
                await route_debate_result(result.request_id, result.to_dict())

        async def teams_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "teams":
                await route_debate_result(result.request_id, result.to_dict())

        async def email_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform == "email":
                await route_debate_result(result.request_id, result.to_dict())

        async def google_chat_handler(result, channel):
            from aragora.server.debate_origin import get_debate_origin

            origin = get_debate_origin(result.request_id)
            if origin and origin.platform in ("google_chat", "gchat"):
                await route_debate_result(result.request_id, result.to_dict())

        # Register all handlers
        router.register_response_handler("telegram", telegram_handler)
        router.register_response_handler("slack", slack_handler)
        router.register_response_handler("discord", discord_handler)
        router.register_response_handler("whatsapp", whatsapp_handler)
        router.register_response_handler("teams", teams_handler)
        router.register_response_handler("email", email_handler)
        router.register_response_handler("google_chat", google_chat_handler)
        router.register_response_handler("gchat", google_chat_handler)

        logger.info("DecisionRouter initialized with 8 platform response handlers")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize DecisionRouter: {e}")
        return False


__all__ = [
    "check_connector_dependencies",
    "check_agent_credentials",
    "check_production_requirements",
    "validate_redis_connectivity",
    "validate_database_connectivity",
    "validate_backend_connectivity",
    "validate_storage_backend",
    "init_redis_ha",
    "init_error_monitoring",
    "init_opentelemetry",
    "init_otlp_exporter",
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
    "init_backup_scheduler",
    "init_webhook_dispatcher",
    "init_slo_webhooks",
    "init_gauntlet_run_recovery",
    "init_durable_job_queue_recovery",
    "init_gauntlet_worker",
    "init_redis_state_backend",
    "init_decision_router",
    "init_key_rotation_scheduler",
    "init_access_review_scheduler",
    "init_rbac_distributed_cache",
    "init_approval_gate_recovery",
    "init_notification_worker",
    "init_graphql_routes",
    "init_deployment_validation",
    "init_witness_patrol",
    "get_witness_behavior",
    "init_mayor_coordinator",
    "get_mayor_coordinator",
    "run_startup_sequence",
]
