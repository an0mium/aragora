"""
Server startup Redis initialization.

This module handles Redis HA and Redis state backend initialization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
            "error": str | None,
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
    except (OSError, RuntimeError, ConnectionError, ValueError, TimeoutError) as e:
        result["error"] = str(e)
        logger.warning(f"Redis HA initialization failed: {e}")

    return result


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
    except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Failed to initialize Redis state backend: {e}")

    return False
