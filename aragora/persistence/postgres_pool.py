"""
PostgreSQL Connection Pool with Read Replica Support.

Provides connection pooling with automatic routing of read queries to replicas
when available, while maintaining write consistency through the primary.

Usage:
    from aragora.persistence.postgres_pool import (
        ReplicaAwarePool,
        get_pool,
        configure_pool,
    )

    # Configure with replicas
    configure_pool(
        primary_dsn="postgresql://primary:5432/db",
        replica_dsns=["postgresql://replica1:5432/db"],
    )

    # Get a connection (auto-routes reads to replicas)
    async with get_pool().acquire(readonly=True) as conn:
        result = await conn.fetch("SELECT * FROM table")

Configuration:
    ARAGORA_POSTGRES_PRIMARY: Primary database DSN
    ARAGORA_POSTGRES_REPLICAS: Comma-separated list of replica DSNs
    ARAGORA_POSTGRES_POOL_MIN: Minimum connections per pool (default: 2)
    ARAGORA_POSTGRES_POOL_MAX: Maximum connections per pool (default: 10)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration from environment
PRIMARY_DSN = os.environ.get("ARAGORA_POSTGRES_PRIMARY", "")
REPLICA_DSNS_RAW = os.environ.get("ARAGORA_POSTGRES_REPLICAS", "")
POOL_MIN_SIZE = int(os.environ.get("ARAGORA_POSTGRES_POOL_MIN", "2"))
POOL_MAX_SIZE = int(os.environ.get("ARAGORA_POSTGRES_POOL_MAX", "10"))


def _parse_replica_dsns(raw: str) -> List[str]:
    """Parse comma-separated replica DSNs."""
    if not raw.strip():
        return []
    return [dsn.strip() for dsn in raw.split(",") if dsn.strip()]


@dataclass
class PoolStats:
    """Statistics for a connection pool."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    wait_count: int = 0
    total_queries: int = 0
    read_queries: int = 0
    write_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "wait_count": self.wait_count,
            "total_queries": self.total_queries,
            "read_queries": self.read_queries,
            "write_queries": self.write_queries,
        }


@dataclass
class ReplicaHealth:
    """Health status of a replica."""

    dsn: str
    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    latency_ms: float = 0.0


class ConnectionWrapper:
    """Wrapper around a database connection for metrics and tracing."""

    def __init__(self, conn: Any, pool: "ReplicaAwarePool", is_replica: bool = False):
        """
        Initialize connection wrapper.

        Args:
            conn: Underlying database connection
            pool: Parent pool for stats
            is_replica: Whether this is a replica connection
        """
        self._conn = conn
        self._pool = pool
        self._is_replica = is_replica

    @property
    def connection(self) -> Any:
        """Get underlying connection."""
        return self._conn

    async def fetch(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a fetch query."""
        self._pool._record_query(readonly=self._is_replica)
        return await self._conn.fetch(query, *args, **kwargs)

    async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a fetchrow query."""
        self._pool._record_query(readonly=self._is_replica)
        return await self._conn.fetchrow(query, *args, **kwargs)

    async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a fetchval query."""
        self._pool._record_query(readonly=self._is_replica)
        return await self._conn.fetchval(query, *args, **kwargs)

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a query."""
        self._pool._record_query(readonly=False)
        return await self._conn.execute(query, *args, **kwargs)

    async def executemany(self, query: str, args: Any, **kwargs: Any) -> Any:
        """Execute many queries."""
        self._pool._record_query(readonly=False)
        return await self._conn.executemany(query, args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to underlying connection."""
        return getattr(self._conn, name)


class ReplicaAwarePool:
    """
    Connection pool with automatic read replica routing.

    Routes readonly queries to replicas when available, while ensuring
    write operations always go to the primary.
    """

    def __init__(
        self,
        primary_dsn: str = "",
        replica_dsns: Optional[List[str]] = None,
        min_size: int = POOL_MIN_SIZE,
        max_size: int = POOL_MAX_SIZE,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize the replica-aware pool.

        Args:
            primary_dsn: DSN for primary database
            replica_dsns: List of DSNs for read replicas
            min_size: Minimum connections per pool
            max_size: Maximum connections per pool
            health_check_interval: Seconds between replica health checks
        """
        self._primary_dsn = primary_dsn or PRIMARY_DSN
        self._replica_dsns = replica_dsns or _parse_replica_dsns(REPLICA_DSNS_RAW)
        self._min_size = min_size
        self._max_size = max_size
        self._health_check_interval = health_check_interval

        self._primary_pool: Optional[Any] = None
        self._replica_pools: List[Any] = []
        self._replica_health: Dict[str, ReplicaHealth] = {}
        self._stats = PoolStats()
        self._initialized = False
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize connection pools."""
        async with self._lock:
            if self._initialized:
                return

            try:
                import asyncpg  # noqa: F401 - type hint only
            except ImportError:
                logger.warning(
                    "asyncpg not installed, PostgreSQL pool will not be available. "
                    "Install with: pip install asyncpg"
                )
                return

            # Create primary pool
            if self._primary_dsn:
                try:
                    import asyncpg

                    self._primary_pool = await asyncpg.create_pool(
                        self._primary_dsn,
                        min_size=self._min_size,
                        max_size=self._max_size,
                    )
                    logger.info("Primary PostgreSQL pool initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize primary pool: {e}")
                    raise

            # Create replica pools
            for dsn in self._replica_dsns:
                try:
                    import asyncpg

                    pool = await asyncpg.create_pool(
                        dsn,
                        min_size=self._min_size,
                        max_size=self._max_size,
                    )
                    self._replica_pools.append(pool)
                    self._replica_health[dsn] = ReplicaHealth(dsn=dsn)
                    logger.info(f"Replica pool initialized: {dsn[:30]}...")
                except Exception as e:
                    logger.warning(f"Failed to initialize replica pool {dsn[:30]}...: {e}")

            if self._replica_pools:
                # Start background health checks
                self._health_task = asyncio.create_task(self._health_check_loop())

            self._initialized = True
            logger.info(f"PostgreSQL pool ready: 1 primary, {len(self._replica_pools)} replicas")

    async def close(self) -> None:
        """Close all connection pools."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._primary_pool:
            await self._primary_pool.close()

        for pool in self._replica_pools:
            await pool.close()

        self._initialized = False
        logger.info("PostgreSQL pools closed")

    @asynccontextmanager
    async def acquire(
        self,
        readonly: bool = False,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[ConnectionWrapper, None]:
        """
        Acquire a database connection.

        Args:
            readonly: If True, prefer replica for read operations
            timeout: Optional timeout for acquiring connection

        Yields:
            ConnectionWrapper with database connection

        Raises:
            RuntimeError: If pool not initialized or no connections available
        """
        if not self._initialized:
            await self.initialize()

        pool = None
        is_replica = False

        # Try to use a healthy replica for readonly operations
        if readonly and self._replica_pools:
            pool = self._select_healthy_replica()
            if pool:
                is_replica = True
                logger.debug("Routing read query to replica")

        # Fall back to primary
        if pool is None:
            if not self._primary_pool:
                raise RuntimeError("Primary pool not initialized")
            pool = self._primary_pool
            if readonly:
                logger.debug("No healthy replica, using primary for read")

        self._stats.wait_count += 1

        try:
            if timeout:
                conn = await asyncio.wait_for(pool.acquire(), timeout=timeout)
            else:
                conn = await pool.acquire()

            self._stats.active_connections += 1
            wrapper = ConnectionWrapper(conn, self, is_replica=is_replica)

            try:
                yield wrapper
            finally:
                self._stats.active_connections -= 1
                await pool.release(conn)

        except asyncio.TimeoutError:
            logger.warning(f"Connection acquire timeout after {timeout}s")
            raise

    def _select_healthy_replica(self) -> Optional[Any]:
        """Select a healthy replica using weighted random selection."""
        if not self._replica_pools:
            return None

        # Filter healthy replicas
        healthy_pools = []
        for pool, health in zip(self._replica_pools, self._replica_health.values()):
            if health.healthy:
                healthy_pools.append(pool)

        if not healthy_pools:
            return None

        # Simple random selection (could be enhanced with latency-based weighting)
        return random.choice(healthy_pools)

    def _record_query(self, readonly: bool) -> None:
        """Record query statistics."""
        self._stats.total_queries += 1
        if readonly:
            self._stats.read_queries += 1
        else:
            self._stats.write_queries += 1

    async def _health_check_loop(self) -> None:
        """Background task to check replica health."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_replica_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    async def _check_replica_health(self) -> None:
        """Check health of all replicas."""
        import time

        for pool, dsn in zip(self._replica_pools, self._replica_dsns):
            health = self._replica_health.get(dsn)
            if not health:
                continue

            start = time.perf_counter()
            try:
                async with asyncio.timeout(5.0):
                    conn = await pool.acquire()
                    try:
                        await conn.fetchval("SELECT 1")
                    finally:
                        await pool.release(conn)

                health.healthy = True
                health.consecutive_failures = 0
                health.latency_ms = (time.perf_counter() - start) * 1000
                health.last_check = time.time()

            except Exception as e:
                health.consecutive_failures += 1
                if health.consecutive_failures >= 3:
                    health.healthy = False
                    logger.warning(f"Replica marked unhealthy: {dsn[:30]}... ({e})")
                health.last_check = time.time()

    @property
    def stats(self) -> PoolStats:
        """Get pool statistics."""
        if self._primary_pool:
            self._stats.total_connections = self._primary_pool.get_size()
            self._stats.idle_connections = self._primary_pool.get_idle_size()
        return self._stats

    @property
    def replica_count(self) -> int:
        """Get number of configured replicas."""
        return len(self._replica_pools)

    @property
    def healthy_replica_count(self) -> int:
        """Get number of healthy replicas."""
        return sum(1 for h in self._replica_health.values() if h.healthy)

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "initialized": self._initialized,
            "primary_healthy": self._primary_pool is not None,
            "replica_count": self.replica_count,
            "healthy_replicas": self.healthy_replica_count,
            "stats": self._stats.to_dict(),
            "replicas": {
                dsn[:30]: {
                    "healthy": h.healthy,
                    "latency_ms": round(h.latency_ms, 2),
                    "consecutive_failures": h.consecutive_failures,
                }
                for dsn, h in self._replica_health.items()
            },
        }


# Global pool instance
_pool: Optional[ReplicaAwarePool] = None


def configure_pool(
    primary_dsn: str = "",
    replica_dsns: Optional[List[str]] = None,
    min_size: int = POOL_MIN_SIZE,
    max_size: int = POOL_MAX_SIZE,
) -> ReplicaAwarePool:
    """
    Configure the global PostgreSQL pool.

    Args:
        primary_dsn: Primary database DSN
        replica_dsns: List of replica DSNs
        min_size: Minimum pool size
        max_size: Maximum pool size

    Returns:
        Configured pool instance
    """
    global _pool
    _pool = ReplicaAwarePool(
        primary_dsn=primary_dsn,
        replica_dsns=replica_dsns,
        min_size=min_size,
        max_size=max_size,
    )
    return _pool


def get_pool() -> ReplicaAwarePool:
    """
    Get the global PostgreSQL pool.

    Returns:
        The configured pool (creates default if not configured)
    """
    global _pool
    if _pool is None:
        _pool = ReplicaAwarePool()
    return _pool


async def close_pool() -> None:
    """Close the global pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


__all__ = [
    "ReplicaAwarePool",
    "PoolStats",
    "ReplicaHealth",
    "ConnectionWrapper",
    "configure_pool",
    "get_pool",
    "close_pool",
]
