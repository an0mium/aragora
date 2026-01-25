"""
Redis High-Availability Configuration.

Centralized configuration for Redis HA deployment modes:
- Standalone: Single Redis instance (development/testing)
- Sentinel: Redis Sentinel for automatic failover (production HA)
- Cluster: Redis Cluster for horizontal scaling (enterprise)

This module provides environment-based configuration that is consumed
by aragora.storage.redis_ha for client creation.

Usage:
    from aragora.config.redis import get_redis_ha_config, RedisHASettings

    # Get configuration from environment
    config = get_redis_ha_config()

    # Check mode
    if config.mode == "sentinel":
        print(f"Sentinel master: {config.sentinel_master}")
        print(f"Sentinel hosts: {config.sentinel_hosts}")

Environment Variables:
    ARAGORA_REDIS_MODE: Redis mode ("standalone", "sentinel", "cluster")
    ARAGORA_REDIS_URL: Standalone Redis URL
    ARAGORA_REDIS_HOST: Standalone Redis host (default: localhost)
    ARAGORA_REDIS_PORT: Standalone Redis port (default: 6379)
    ARAGORA_REDIS_PASSWORD: Redis authentication password
    ARAGORA_REDIS_DB: Redis database number (default: 0)

    # Sentinel mode
    ARAGORA_REDIS_SENTINEL_HOSTS: Comma-separated sentinel hosts (host:port)
    ARAGORA_REDIS_SENTINEL_MASTER: Sentinel master name (default: mymaster)
    ARAGORA_REDIS_SENTINEL_PASSWORD: Sentinel authentication password

    # Cluster mode
    ARAGORA_REDIS_CLUSTER_NODES: Comma-separated cluster nodes (host:port)
    ARAGORA_REDIS_CLUSTER_READ_FROM_REPLICAS: Enable read from replicas (default: true)

    # Common settings
    ARAGORA_REDIS_SOCKET_TIMEOUT: Socket timeout in seconds (default: 5.0)
    ARAGORA_REDIS_MAX_CONNECTIONS: Max pool connections (default: 50)
    ARAGORA_REDIS_HEALTH_CHECK_INTERVAL: Health check interval (default: 30)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RedisMode(str, Enum):
    """Redis deployment mode."""

    STANDALONE = "standalone"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


@dataclass
class RedisHASettings:
    """
    Configuration settings for Redis High-Availability.

    This dataclass holds all Redis HA configuration loaded from
    environment variables. Use get_redis_ha_config() to obtain
    a populated instance.

    Attributes:
        mode: Redis deployment mode (standalone/sentinel/cluster)
        enabled: Whether Redis HA is explicitly enabled

        # Standalone configuration
        host: Redis server hostname
        port: Redis server port
        password: Redis authentication password
        db: Redis database number
        url: Full Redis URL (overrides host/port if provided)

        # Sentinel configuration
        sentinel_hosts: List of sentinel host:port strings
        sentinel_master: Name of the master in Sentinel
        sentinel_password: Password for Sentinel connections

        # Cluster configuration
        cluster_nodes: List of cluster node host:port strings
        cluster_read_from_replicas: Enable reading from replicas

        # Common connection settings
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Connection timeout in seconds
        max_connections: Maximum connections in pool
        retry_on_timeout: Retry operations on timeout
        health_check_interval: Seconds between health checks
        decode_responses: Decode byte responses to strings

        # SSL/TLS settings
        ssl: Enable SSL/TLS connections
        ssl_cert_reqs: SSL certificate requirements
        ssl_ca_certs: Path to CA certificates
    """

    mode: RedisMode = RedisMode.STANDALONE
    enabled: bool = False

    # Standalone configuration
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    url: Optional[str] = None

    # Sentinel configuration
    sentinel_hosts: List[str] = field(default_factory=list)
    sentinel_master: str = "mymaster"
    sentinel_password: Optional[str] = None

    # Cluster configuration
    cluster_nodes: List[str] = field(default_factory=list)
    cluster_read_from_replicas: bool = True
    cluster_skip_full_coverage_check: bool = False

    # Common connection settings
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    decode_responses: bool = True

    # SSL/TLS settings
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None

    @property
    def is_ha_mode(self) -> bool:
        """Check if running in high-availability mode (Sentinel or Cluster)."""
        return self.mode in (RedisMode.SENTINEL, RedisMode.CLUSTER)

    @property
    def is_configured(self) -> bool:
        """Check if Redis is configured (URL, Sentinel hosts, or Cluster nodes)."""
        if self.url:
            return True
        if self.sentinel_hosts:
            return True
        if self.cluster_nodes:
            return True
        # Check for standalone with non-default host
        if self.host != "localhost":
            return True
        return False

    def get_mode_description(self) -> str:
        """Get a human-readable description of the current mode."""
        if self.mode == RedisMode.SENTINEL:
            return f"Sentinel ({self.sentinel_master}, {len(self.sentinel_hosts)} nodes)"
        elif self.mode == RedisMode.CLUSTER:
            return f"Cluster ({len(self.cluster_nodes)} nodes)"
        else:
            return f"Standalone ({self.host}:{self.port})"


from aragora.config.env_helpers import (
    env_str as _env_str,
    env_int as _env_int,
    env_float as _env_float,
    env_bool as _env_bool,
)


def _parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated string into list of stripped strings.

    Note: Consider using env_list() from env_helpers for new code.
    """
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_redis_ha_config() -> RedisHASettings:
    """
    Load Redis HA configuration from environment variables.

    This function reads all Redis-related environment variables and
    constructs a RedisHASettings instance. It also performs auto-detection
    of the Redis mode based on available configuration.

    Auto-detection logic:
    - If ARAGORA_REDIS_SENTINEL_HOSTS is set -> Sentinel mode
    - If ARAGORA_REDIS_CLUSTER_NODES is set -> Cluster mode
    - Otherwise -> Standalone mode

    Returns:
        RedisHASettings populated from environment variables

    Example:
        >>> config = get_redis_ha_config()
        >>> if config.is_ha_mode:
        ...     print(f"Running in HA mode: {config.get_mode_description()}")
    """
    # Determine explicit mode setting
    mode_str = _env_str("ARAGORA_REDIS_MODE", "standalone").lower()
    try:
        mode = RedisMode(mode_str)
    except ValueError:
        mode = RedisMode.STANDALONE

    # Parse sentinel hosts
    sentinel_hosts_str = _env_str("ARAGORA_REDIS_SENTINEL_HOSTS")
    sentinel_hosts = _parse_comma_separated(sentinel_hosts_str)

    # Parse cluster nodes
    cluster_nodes_str = _env_str("ARAGORA_REDIS_CLUSTER_NODES")
    cluster_nodes = _parse_comma_separated(cluster_nodes_str)

    # Auto-detect mode if not explicitly set
    if mode == RedisMode.STANDALONE:
        if sentinel_hosts:
            mode = RedisMode.SENTINEL
        elif cluster_nodes:
            mode = RedisMode.CLUSTER

    # Get URL from multiple possible environment variables
    url = _env_str("ARAGORA_REDIS_URL") or _env_str("REDIS_URL") or None

    # Determine if Redis is enabled
    enabled = bool(url or sentinel_hosts or cluster_nodes or mode != RedisMode.STANDALONE)

    return RedisHASettings(
        mode=mode,
        enabled=enabled,
        # Standalone
        host=_env_str("ARAGORA_REDIS_HOST", "localhost"),
        port=_env_int("ARAGORA_REDIS_PORT", 6379),
        password=_env_str("ARAGORA_REDIS_PASSWORD") or None,
        db=_env_int("ARAGORA_REDIS_DB", 0),
        url=url,
        # Sentinel
        sentinel_hosts=sentinel_hosts,
        sentinel_master=_env_str("ARAGORA_REDIS_SENTINEL_MASTER", "mymaster"),
        sentinel_password=_env_str("ARAGORA_REDIS_SENTINEL_PASSWORD") or None,
        # Cluster
        cluster_nodes=cluster_nodes,
        cluster_read_from_replicas=_env_bool("ARAGORA_REDIS_CLUSTER_READ_FROM_REPLICAS", True),
        cluster_skip_full_coverage_check=_env_bool(
            "ARAGORA_REDIS_CLUSTER_SKIP_FULL_COVERAGE", False
        ),
        # Common
        socket_timeout=_env_float("ARAGORA_REDIS_SOCKET_TIMEOUT", 5.0),
        socket_connect_timeout=_env_float("ARAGORA_REDIS_SOCKET_CONNECT_TIMEOUT", 5.0),
        max_connections=_env_int("ARAGORA_REDIS_MAX_CONNECTIONS", 50),
        retry_on_timeout=_env_bool("ARAGORA_REDIS_RETRY_ON_TIMEOUT", True),
        health_check_interval=_env_int("ARAGORA_REDIS_HEALTH_CHECK_INTERVAL", 30),
        decode_responses=_env_bool("ARAGORA_REDIS_DECODE_RESPONSES", True),
        # SSL
        ssl=_env_bool("ARAGORA_REDIS_SSL", False),
        ssl_cert_reqs=_env_str("ARAGORA_REDIS_SSL_CERT_REQS") or None,
        ssl_ca_certs=_env_str("ARAGORA_REDIS_SSL_CA_CERTS") or None,
    )


# Module-level constants for direct import
REDIS_MODE = _env_str("ARAGORA_REDIS_MODE", "standalone")
REDIS_SENTINEL_HOSTS = _env_str("ARAGORA_REDIS_SENTINEL_HOSTS", "")
REDIS_SENTINEL_MASTER = _env_str("ARAGORA_REDIS_SENTINEL_MASTER", "mymaster")
REDIS_CLUSTER_NODES = _env_str("ARAGORA_REDIS_CLUSTER_NODES", "")


__all__ = [
    "RedisMode",
    "RedisHASettings",
    "get_redis_ha_config",
    "REDIS_MODE",
    "REDIS_SENTINEL_HOSTS",
    "REDIS_SENTINEL_MASTER",
    "REDIS_CLUSTER_NODES",
]
