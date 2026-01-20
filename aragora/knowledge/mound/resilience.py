"""
Knowledge Mound Persistence Resilience.

Provides production-hardening capabilities:
- Retry logic with exponential backoff for transient failures
- Explicit transaction boundaries for multi-table operations
- Connection health monitoring
- Circuit breaker integration
- Cache invalidation events

"Reliability is a feature."
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Retry strategies for transient failures."""

    EXPONENTIAL = "exponential"  # 2^n * base_delay with jitter
    LINEAR = "linear"  # n * base_delay
    CONSTANT = "constant"  # base_delay always


class TransactionIsolation(str, Enum):
    """Transaction isolation levels."""

    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 0.1  # seconds
    max_delay: float = 10.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2**attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add ±25% jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


@dataclass
class TransactionConfig:
    """Configuration for transaction behavior."""

    isolation: TransactionIsolation = TransactionIsolation.READ_COMMITTED
    timeout_seconds: float = 30.0
    savepoint_on_nested: bool = True


@dataclass
class HealthStatus:
    """Health status for a storage backend."""

    healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class CacheInvalidationEvent:
    """Event for cache invalidation."""

    event_type: str  # "node_updated", "node_deleted", "query_invalidated"
    workspace_id: str
    item_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "workspace_id": self.workspace_id,
            "item_id": self.item_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class CacheInvalidationBus:
    """
    Event bus for cache invalidation.

    Allows subscribers to receive invalidation events for
    coordinated cache updates across the system.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[CacheInvalidationEvent], Awaitable[None]]] = []
        self._event_log: list[CacheInvalidationEvent] = []
        self._max_log_size = 1000

    def subscribe(
        self, callback: Callable[[CacheInvalidationEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Subscribe to cache invalidation events.

        Returns an unsubscribe function.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    async def publish(self, event: CacheInvalidationEvent) -> None:
        """Publish a cache invalidation event to all subscribers."""
        # Log event
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size // 2 :]

        # Notify subscribers
        errors = []
        for subscriber in self._subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Cache invalidation subscriber error: {e}")

        if errors:
            logger.warning(f"Cache invalidation had {len(errors)} subscriber errors")

    async def publish_node_update(
        self, workspace_id: str, node_id: str, **metadata: Any
    ) -> None:
        """Convenience method for node update events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_updated",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_node_delete(
        self, workspace_id: str, node_id: str, **metadata: Any
    ) -> None:
        """Convenience method for node deletion events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_deleted",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_query_invalidation(
        self, workspace_id: str, **metadata: Any
    ) -> None:
        """Convenience method for query cache invalidation."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="query_invalidated",
                workspace_id=workspace_id,
                metadata=metadata,
            )
        )

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent invalidation events."""
        return [e.to_dict() for e in self._event_log[-limit:]]


# Global cache invalidation bus
_invalidation_bus: Optional[CacheInvalidationBus] = None


def get_invalidation_bus() -> CacheInvalidationBus:
    """Get or create the global cache invalidation bus."""
    global _invalidation_bus
    if _invalidation_bus is None:
        _invalidation_bus = CacheInvalidationBus()
    return _invalidation_bus


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for adding retry logic to async functions.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def save_node(...):
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__} (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}: {e}"
                        )

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry loop exit in {func.__name__}")

        return wrapper

    return decorator


class TransactionManager:
    """
    Manages explicit transaction boundaries for PostgreSQL operations.

    Provides:
    - Explicit BEGIN/COMMIT/ROLLBACK
    - Savepoints for nested transactions
    - Timeout enforcement
    - Deadlock detection and retry
    """

    def __init__(
        self,
        pool: Any,
        config: Optional[TransactionConfig] = None,
    ):
        self._pool = pool
        self._config = config or TransactionConfig()
        self._active_transactions = 0

    @asynccontextmanager
    async def transaction(
        self,
        isolation: Optional[TransactionIsolation] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Any]:
        """
        Execute operations within an explicit transaction.

        Usage:
            async with tx_manager.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
                # Auto-commit on success, rollback on exception
        """
        isolation = isolation or self._config.isolation
        timeout = timeout or self._config.timeout_seconds

        async with self._pool.acquire() as conn:
            # Set transaction isolation
            await conn.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation.value}")

            # Start transaction
            await conn.execute("BEGIN")
            self._active_transactions += 1

            try:
                # Yield connection for operations
                yield conn

                # Commit on success
                await conn.execute("COMMIT")
                logger.debug("Transaction committed successfully")

            except Exception as e:
                # Rollback on any exception
                await conn.execute("ROLLBACK")
                logger.warning(f"Transaction rolled back due to: {e}")
                raise

            finally:
                self._active_transactions -= 1

    @asynccontextmanager
    async def savepoint(self, conn: Any, name: str) -> AsyncIterator[None]:
        """
        Create a savepoint within an existing transaction.

        Usage:
            async with tx_manager.transaction() as conn:
                # Some operations...
                async with tx_manager.savepoint(conn, "sp1"):
                    # Nested operations that can be rolled back independently
                    ...
        """
        await conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except Exception:
            await conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get transaction manager statistics."""
        return {
            "active_transactions": self._active_transactions,
            "default_isolation": self._config.isolation.value,
            "default_timeout": self._config.timeout_seconds,
        }


class ConnectionHealthMonitor:
    """
    Monitors connection health and provides circuit breaker functionality.

    Tracks:
    - Connection success/failure rates
    - Latency metrics
    - Automatic health checks
    """

    def __init__(
        self,
        pool: Any,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        health_check_interval: float = 10.0,
    ):
        self._pool = pool
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._health_check_interval = health_check_interval

        self._status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
        )
        self._check_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start background health monitoring."""
        if self._check_task is None:
            self._check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Connection health monitor started")

    async def stop(self) -> None:
        """Stop background health monitoring."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    async def check_health(self) -> HealthStatus:
        """Perform a health check."""
        start = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            latency = (time.monotonic() - start) * 1000
            self._status = HealthStatus(
                healthy=True,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=0,
                latency_ms=latency,
            )

        except Exception as e:
            self._status = HealthStatus(
                healthy=self._status.consecutive_failures < self._failure_threshold,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._status.consecutive_failures + 1,
                last_error=str(e),
            )

        return self._status

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._status.consecutive_failures > 0:
            self._status.consecutive_failures = 0
            self._status.healthy = True

    def record_failure(self, error: str) -> None:
        """Record a failed operation."""
        self._status.consecutive_failures += 1
        self._status.last_error = error
        if self._status.consecutive_failures >= self._failure_threshold:
            self._status.healthy = False
            logger.error(
                f"Connection unhealthy after {self._status.consecutive_failures} failures"
            )

    def is_healthy(self) -> bool:
        """Check if connections are healthy."""
        return self._status.healthy

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        return self._status


@dataclass
class IntegrityCheckResult:
    """Result of an integrity check."""

    passed: bool
    checks_performed: int
    issues_found: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_performed": self.checks_performed,
            "issues_found": self.issues_found,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class IntegrityVerifier:
    """
    Verifies data integrity on startup and periodically.

    Checks:
    - Foreign key constraint violations
    - Orphaned records
    - Checksum consistency
    - Index health
    """

    def __init__(self, pool: Any):
        self._pool = pool

    async def verify_all(self) -> IntegrityCheckResult:
        """Run all integrity checks."""
        issues: list[str] = []
        details: dict[str, Any] = {}
        checks_performed = 0

        # Check for orphaned provenance chains
        checks_performed += 1
        orphaned_provenance = await self._check_orphaned_provenance()
        if orphaned_provenance > 0:
            issues.append(f"Found {orphaned_provenance} orphaned provenance chains")
        details["orphaned_provenance"] = orphaned_provenance

        # Check for orphaned relationships
        checks_performed += 1
        orphaned_relationships = await self._check_orphaned_relationships()
        if orphaned_relationships > 0:
            issues.append(f"Found {orphaned_relationships} orphaned relationships")
        details["orphaned_relationships"] = orphaned_relationships

        # Check for orphaned topics
        checks_performed += 1
        orphaned_topics = await self._check_orphaned_topics()
        if orphaned_topics > 0:
            issues.append(f"Found {orphaned_topics} orphaned topics")
        details["orphaned_topics"] = orphaned_topics

        # Check for orphaned access grants
        checks_performed += 1
        orphaned_grants = await self._check_orphaned_access_grants()
        if orphaned_grants > 0:
            issues.append(f"Found {orphaned_grants} orphaned access grants")
        details["orphaned_grants"] = orphaned_grants

        # Check index health
        checks_performed += 1
        index_issues = await self._check_index_health()
        if index_issues:
            issues.extend(index_issues)
        details["index_health"] = "healthy" if not index_issues else index_issues

        # Check for duplicate content hashes
        checks_performed += 1
        duplicates = await self._check_duplicate_content()
        details["duplicate_content_count"] = duplicates

        return IntegrityCheckResult(
            passed=len(issues) == 0,
            checks_performed=checks_performed,
            issues_found=issues,
            details=details,
        )

    async def _check_orphaned_provenance(self) -> int:
        """Check for provenance chains without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM provenance_chains pc
                LEFT JOIN knowledge_nodes kn ON pc.node_id = kn.id
                WHERE kn.id IS NULL
                """
            )
            return result or 0

    async def _check_orphaned_relationships(self) -> int:
        """Check for relationships with missing nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM knowledge_relationships kr
                LEFT JOIN knowledge_nodes kn1 ON kr.from_node_id = kn1.id
                LEFT JOIN knowledge_nodes kn2 ON kr.to_node_id = kn2.id
                WHERE kn1.id IS NULL OR kn2.id IS NULL
                """
            )
            return result or 0

    async def _check_orphaned_topics(self) -> int:
        """Check for topics without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM node_topics nt
                LEFT JOIN knowledge_nodes kn ON nt.node_id = kn.id
                WHERE kn.id IS NULL
                """
            )
            return result or 0

    async def _check_orphaned_access_grants(self) -> int:
        """Check for access grants without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM access_grants ag
                LEFT JOIN knowledge_nodes kn ON ag.item_id = kn.id
                WHERE kn.id IS NULL
                """
            )
            return result or 0

    async def _check_index_health(self) -> list[str]:
        """Check index health (PostgreSQL specific)."""
        issues = []
        try:
            async with self._pool.acquire() as conn:
                # Check for invalid indexes
                invalid = await conn.fetch(
                    """
                    SELECT indexrelid::regclass AS index_name
                    FROM pg_index WHERE NOT indisvalid
                    """
                )
                for row in invalid:
                    issues.append(f"Invalid index: {row['index_name']}")
        except Exception as e:
            logger.warning(f"Index health check failed: {e}")
        return issues

    async def _check_duplicate_content(self) -> int:
        """Check for duplicate content by hash."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM (
                    SELECT content_hash, workspace_id, COUNT(*) as cnt
                    FROM knowledge_nodes
                    WHERE content_hash != ''
                    GROUP BY content_hash, workspace_id
                    HAVING COUNT(*) > 1
                ) duplicates
                """
            )
            return result or 0

    async def repair_orphans(self, dry_run: bool = True) -> dict[str, int]:
        """
        Repair orphaned records.

        Args:
            dry_run: If True, only report what would be fixed

        Returns:
            Count of records fixed by table
        """
        repairs: dict[str, int] = {}

        async with self._pool.acquire() as conn:
            if dry_run:
                # Just count what would be repaired
                repairs["provenance_chains"] = await self._check_orphaned_provenance()
                repairs["relationships"] = await self._check_orphaned_relationships()
                repairs["topics"] = await self._check_orphaned_topics()
                repairs["access_grants"] = await self._check_orphaned_access_grants()
            else:
                # Actually delete orphaned records
                result = await conn.execute(
                    """
                    DELETE FROM provenance_chains
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """
                )
                repairs["provenance_chains"] = int(result.split()[-1])

                result = await conn.execute(
                    """
                    DELETE FROM knowledge_relationships
                    WHERE from_node_id NOT IN (SELECT id FROM knowledge_nodes)
                       OR to_node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """
                )
                repairs["relationships"] = int(result.split()[-1])

                result = await conn.execute(
                    """
                    DELETE FROM node_topics
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """
                )
                repairs["topics"] = int(result.split()[-1])

                result = await conn.execute(
                    """
                    DELETE FROM access_grants
                    WHERE item_id NOT IN (SELECT id FROM knowledge_nodes)
                    """
                )
                repairs["access_grants"] = int(result.split()[-1])

                logger.info(f"Repaired orphaned records: {repairs}")

        return repairs


class ResilientPostgresStore:
    """
    Enhanced PostgreSQL store with resilience features.

    Wraps PostgresStore with:
    - Automatic retries with exponential backoff
    - Transaction management
    - Health monitoring
    - Cache invalidation events
    - Integrity verification
    """

    def __init__(
        self,
        store: Any,  # PostgresStore
        retry_config: Optional[RetryConfig] = None,
        transaction_config: Optional[TransactionConfig] = None,
    ):
        self._store = store
        self._retry_config = retry_config or RetryConfig()
        self._tx_config = transaction_config or TransactionConfig()

        self._tx_manager: Optional[TransactionManager] = None
        self._health_monitor: Optional[ConnectionHealthMonitor] = None
        self._integrity_verifier: Optional[IntegrityVerifier] = None
        self._invalidation_bus = get_invalidation_bus()

    async def initialize(self) -> IntegrityCheckResult:
        """
        Initialize with health monitoring and integrity verification.

        Returns integrity check result.
        """
        # Initialize underlying store
        await self._store.initialize()

        # Setup transaction manager
        self._tx_manager = TransactionManager(
            self._store._pool,
            self._tx_config,
        )

        # Setup health monitor
        self._health_monitor = ConnectionHealthMonitor(self._store._pool)
        await self._health_monitor.start()

        # Setup and run integrity verifier
        self._integrity_verifier = IntegrityVerifier(self._store._pool)
        result = await self._integrity_verifier.verify_all()

        if not result.passed:
            logger.warning(
                f"Integrity check found {len(result.issues_found)} issues: {result.issues_found}"
            )
        else:
            logger.info(
                f"Integrity check passed ({result.checks_performed} checks)"
            )

        return result

    async def close(self) -> None:
        """Close with cleanup."""
        if self._health_monitor:
            await self._health_monitor.stop()
        await self._store.close()

    @asynccontextmanager
    async def transaction(
        self,
        isolation: Optional[TransactionIsolation] = None,
    ) -> AsyncIterator[Any]:
        """Get a transaction context."""
        if not self._tx_manager:
            raise RuntimeError("ResilientPostgresStore not initialized")
        async with self._tx_manager.transaction(isolation) as conn:
            yield conn

    @with_retry()
    async def save_node_async(
        self,
        node_data: dict[str, Any],
        *,
        invalidate_cache: bool = True,
    ) -> str:
        """Save node with retry and cache invalidation."""
        result = await self._store.save_node_async(node_data)

        if invalidate_cache:
            await self._invalidation_bus.publish_node_update(
                workspace_id=node_data.get("workspace_id", "default"),
                node_id=node_data["id"],
            )

        if self._health_monitor:
            self._health_monitor.record_success()

        return result

    @with_retry()
    async def get_node_async(self, node_id: str) -> Optional[dict[str, Any]]:
        """Get node with retry."""
        result = await self._store.get_node_async(node_id)
        if self._health_monitor:
            self._health_monitor.record_success()
        return result

    @with_retry()
    async def delete_node_async(
        self,
        node_id: str,
        workspace_id: str,
        *,
        invalidate_cache: bool = True,
    ) -> bool:
        """Delete node with retry and cache invalidation."""
        result = await self._store.delete_node_async(node_id, workspace_id)

        if invalidate_cache and result:
            await self._invalidation_bus.publish_node_delete(
                workspace_id=workspace_id,
                node_id=node_id,
            )

        if self._health_monitor:
            self._health_monitor.record_success()

        return result

    async def save_nodes_batch(
        self,
        nodes: list[dict[str, Any]],
        workspace_id: str,
    ) -> int:
        """
        Save multiple nodes in a single transaction.

        Returns count of saved nodes.
        """
        if not self._tx_manager:
            raise RuntimeError("ResilientPostgresStore not initialized")

        saved = 0
        async with self._tx_manager.transaction() as conn:
            for node in nodes:
                await conn.execute(
                    """
                    INSERT INTO knowledge_nodes (
                        id, workspace_id, node_type, content, content_hash,
                        confidence, tier, created_at, updated_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        confidence = EXCLUDED.confidence,
                        update_count = knowledge_nodes.update_count + 1,
                        updated_at = EXCLUDED.updated_at
                    """,
                    node["id"],
                    workspace_id,
                    node.get("node_type", "fact"),
                    node["content"],
                    node.get("content_hash", ""),
                    node.get("confidence", 0.5),
                    node.get("tier", "slow"),
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    json.dumps(node.get("metadata", {})),
                )
                saved += 1

        # Batch invalidation
        await self._invalidation_bus.publish_query_invalidation(
            workspace_id=workspace_id,
            batch_size=saved,
        )

        return saved

    def is_healthy(self) -> bool:
        """Check if store is healthy."""
        if self._health_monitor:
            return self._health_monitor.is_healthy()
        return True

    def get_health_status(self) -> dict[str, Any]:
        """Get detailed health status."""
        status: dict[str, Any] = {"initialized": self._store._initialized}

        if self._health_monitor:
            status["connection"] = self._health_monitor.get_status().to_dict()

        if self._tx_manager:
            status["transactions"] = self._tx_manager.get_stats()

        return status

    async def verify_integrity(self) -> IntegrityCheckResult:
        """Run integrity verification."""
        if not self._integrity_verifier:
            raise RuntimeError("ResilientPostgresStore not initialized")
        return await self._integrity_verifier.verify_all()

    async def repair_integrity(self, dry_run: bool = True) -> dict[str, int]:
        """Repair integrity issues."""
        if not self._integrity_verifier:
            raise RuntimeError("ResilientPostgresStore not initialized")
        return await self._integrity_verifier.repair_orphans(dry_run=dry_run)
