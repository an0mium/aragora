"""
Connection Router - Read replica routing for PostgreSQL.

Provides connection routing to read replicas for load distribution:
- Primary pool for write operations (INSERT, UPDATE, DELETE)
- Replica pools for read operations (SELECT)
- Automatic failover from replicas to primary
- Round-robin load balancing across replicas

Configuration:
    ARAGORA_POSTGRES_PRIMARY_DSN: Primary database (writes)
    ARAGORA_POSTGRES_REPLICA_DSNS: Comma-separated replica DSNs (reads)
    ARAGORA_REPLICA_POOL_SIZE: Per-replica pool size (default: 10)
    ARAGORA_REPLICA_FAILOVER: Enable failover to primary (default: true)

Usage:
    from aragora.storage.connection_router import (
        get_connection_router,
        initialize_connection_router,
    )

    # In server startup:
    router = await initialize_connection_router()

    # In store operations:
    async with router.connection(read_only=True) as conn:
        await conn.fetch("SELECT * FROM items")

    async with router.connection(read_only=False) as conn:
        await conn.execute("INSERT INTO items ...")
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import AsyncGenerator

if TYPE_CHECKING:
    from asyncpg import Connection, Pool

logger = logging.getLogger(__name__)


@dataclass
class ReplicaConfig:
    """Configuration for a read replica."""

    dsn: str
    name: str = ""
    pool_size: int = 10
    priority: int = 0  # Lower = higher priority


@dataclass
class RouterConfig:
    """Configuration for the connection router."""

    primary_dsn: str | None = None
    replicas: list[ReplicaConfig] = field(default_factory=list)
    failover_to_primary: bool = True
    pool_min_size: int = 5
    pool_max_size: int = 20
    replica_pool_size: int = 10
    command_timeout: float = 60.0
    statement_timeout: int = 60

    @classmethod
    def from_environment(cls) -> RouterConfig:
        """Create configuration from environment variables."""
        primary_dsn = os.environ.get(
            "ARAGORA_POSTGRES_PRIMARY_DSN",
            os.environ.get("ARAGORA_POSTGRES_DSN", os.environ.get("DATABASE_URL")),
        )

        replica_dsns_str = os.environ.get("ARAGORA_POSTGRES_REPLICA_DSNS", "")
        replicas = []
        if replica_dsns_str:
            for i, dsn in enumerate(replica_dsns_str.split(",")):
                dsn = dsn.strip()
                if dsn:
                    replicas.append(
                        ReplicaConfig(
                            dsn=dsn,
                            name=f"replica-{i}",
                            pool_size=int(os.environ.get("ARAGORA_REPLICA_POOL_SIZE", "10")),
                            priority=i,
                        )
                    )

        return cls(
            primary_dsn=primary_dsn,
            replicas=replicas,
            failover_to_primary=os.environ.get("ARAGORA_REPLICA_FAILOVER", "true").lower()
            in ("true", "1", "yes"),
            pool_min_size=int(os.environ.get("ARAGORA_POOL_MIN_SIZE", "5")),
            pool_max_size=int(os.environ.get("ARAGORA_POOL_MAX_SIZE", "20")),
            replica_pool_size=int(os.environ.get("ARAGORA_REPLICA_POOL_SIZE", "10")),
            command_timeout=float(os.environ.get("ARAGORA_POOL_COMMAND_TIMEOUT", "60.0")),
            statement_timeout=int(os.environ.get("ARAGORA_POOL_STATEMENT_TIMEOUT", "60")),
        )


@dataclass
class RouterMetrics:
    """Metrics for connection router."""

    primary_requests: int = 0
    replica_requests: int = 0
    replica_failovers: int = 0
    total_requests: int = 0
    read_ratio: float = 0.0


class ConnectionRouter:
    """Routes database connections between primary and replicas."""

    def __init__(self, config: RouterConfig | None = None):
        """Initialize the connection router.

        Args:
            config: Router configuration. If None, loads from environment.
        """
        self.config = config or RouterConfig.from_environment()
        self._primary_pool: Pool | None = None
        self._replica_pools: list[Pool] = []
        self._replica_index = 0  # For round-robin
        self._initialized = False
        self._metrics = RouterMetrics()
        self._lock = asyncio.Lock()

    @property
    def has_replicas(self) -> bool:
        """Check if read replicas are configured."""
        return len(self._replica_pools) > 0

    @property
    def replica_count(self) -> int:
        """Get number of active replica pools."""
        return len(self._replica_pools)

    async def initialize(self) -> bool:
        """Initialize connection pools.

        Returns:
            True if at least primary pool initialized successfully.
        """
        if self._initialized:
            return True

        try:
            import asyncpg
        except ImportError:
            logger.error("[router] asyncpg not available")
            return False

        # Initialize primary pool
        if self.config.primary_dsn:
            try:
                self._primary_pool = await asyncpg.create_pool(
                    self.config.primary_dsn,
                    min_size=self.config.pool_min_size,
                    max_size=self.config.pool_max_size,
                    command_timeout=self.config.command_timeout,
                    server_settings={
                        "statement_timeout": f"{self.config.statement_timeout * 1000}",
                    },
                )
                logger.info(
                    f"[router] Primary pool initialized (size: {self._primary_pool.get_size()})"
                )
            except Exception as e:
                logger.error(f"[router] Failed to initialize primary pool: {e}")
                return False

        # Initialize replica pools
        for replica_config in self.config.replicas:
            try:
                pool = await asyncpg.create_pool(
                    replica_config.dsn,
                    min_size=1,
                    max_size=replica_config.pool_size,
                    command_timeout=self.config.command_timeout,
                    server_settings={
                        "statement_timeout": f"{self.config.statement_timeout * 1000}",
                        "default_transaction_read_only": "on",
                    },
                )
                self._replica_pools.append(pool)
                logger.info(
                    f"[router] Replica pool '{replica_config.name}' initialized "
                    f"(size: {pool.get_size()})"
                )
            except Exception as e:
                logger.warning(
                    f"[router] Failed to initialize replica '{replica_config.name}': {e}"
                )
                # Continue with other replicas

        self._initialized = True
        logger.info(f"[router] Initialized with 1 primary + {len(self._replica_pools)} replicas")
        return True

    async def _select_replica_pool(self) -> Pool | None:
        """Select a replica pool using round-robin."""
        if not self._replica_pools:
            return None

        async with self._lock:
            pool = self._replica_pools[self._replica_index]
            self._replica_index = (self._replica_index + 1) % len(self._replica_pools)
            return pool

    @asynccontextmanager
    async def connection(self, read_only: bool = False) -> AsyncGenerator[Connection, None]:
        """Get a connection, routing to replica for reads if available.

        Args:
            read_only: If True, prefer replica pools for the connection.

        Yields:
            Database connection from appropriate pool.
        """
        self._metrics.total_requests += 1

        if read_only and self._replica_pools:
            # Try to get from replica pool
            pool = await self._select_replica_pool()
            if pool:
                try:
                    self._metrics.replica_requests += 1
                    async with pool.acquire() as conn:
                        yield conn
                    return
                except Exception as e:
                    logger.warning(f"[router] Replica connection failed: {e}")
                    self._metrics.replica_failovers += 1
                    # Fall through to primary if failover enabled
                    if not self.config.failover_to_primary:
                        raise

        # Use primary pool
        if self._primary_pool is None:
            raise RuntimeError("Primary pool not initialized")

        self._metrics.primary_requests += 1
        async with self._primary_pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """Get a transactional connection (always uses primary).

        Transactions always go to primary since replicas are read-only.

        Yields:
            Database connection from primary pool with transaction.
        """
        if self._primary_pool is None:
            raise RuntimeError("Primary pool not initialized")

        self._metrics.primary_requests += 1
        self._metrics.total_requests += 1

        async with self._primary_pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    def get_metrics(self) -> RouterMetrics:
        """Get current router metrics."""
        metrics = self._metrics
        if metrics.total_requests > 0:
            metrics.read_ratio = metrics.replica_requests / metrics.total_requests
        return metrics

    def get_info(self) -> dict[str, Any]:
        """Get router information for diagnostics."""
        return {
            "initialized": self._initialized,
            "has_replicas": self.has_replicas,
            "replica_count": self.replica_count,
            "primary_pool_size": (self._primary_pool.get_size() if self._primary_pool else 0),
            "primary_free": (self._primary_pool.get_idle_size() if self._primary_pool else 0),
            "replica_pools": [
                {"size": p.get_size(), "free": p.get_idle_size()} for p in self._replica_pools
            ],
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "primary_requests": self._metrics.primary_requests,
                "replica_requests": self._metrics.replica_requests,
                "replica_failovers": self._metrics.replica_failovers,
            },
        }

    async def close(self) -> None:
        """Close all connection pools."""
        if self._primary_pool:
            await self._primary_pool.close()
            logger.info("[router] Primary pool closed")

        for i, pool in enumerate(self._replica_pools):
            await pool.close()
            logger.info(f"[router] Replica pool {i} closed")

        self._initialized = False
        self._replica_pools = []
        self._primary_pool = None


# Global router instance
_router: ConnectionRouter | None = None


async def initialize_connection_router(
    config: RouterConfig | None = None,
) -> ConnectionRouter | None:
    """Initialize the global connection router.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        ConnectionRouter if initialization succeeds, None otherwise.
    """
    global _router

    if _router is not None and _router._initialized:
        logger.debug("[router] Router already initialized")
        return _router

    router_config = config or RouterConfig.from_environment()

    # Only initialize if replicas are configured
    if not router_config.replicas:
        logger.debug("[router] No replicas configured, router not needed")
        return None

    _router = ConnectionRouter(router_config)
    success = await _router.initialize()

    if not success:
        _router = None
        return None

    return _router


def get_connection_router() -> ConnectionRouter | None:
    """Get the global connection router if initialized."""
    return _router


def is_router_initialized() -> bool:
    """Check if the connection router is initialized."""
    return _router is not None and _router._initialized


async def close_connection_router() -> None:
    """Close the global connection router."""
    global _router

    if _router:
        await _router.close()
        _router = None


def reset_connection_router() -> None:
    """Reset router state (for testing only)."""
    global _router
    _router = None


__all__ = [
    "ConnectionRouter",
    "RouterConfig",
    "ReplicaConfig",
    "RouterMetrics",
    "initialize_connection_router",
    "get_connection_router",
    "is_router_initialized",
    "close_connection_router",
    "reset_connection_router",
]
