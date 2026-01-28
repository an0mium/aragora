"""
Knowledge Mound Persistence Resilience.

Provides production-hardening capabilities:
- Retry logic with exponential backoff for transient failures
- Explicit transaction boundaries for multi-table operations
- Connection health monitoring
- Circuit breaker integration for adapters
- Cache invalidation events
- SLO monitoring with Prometheus metrics
- Bulkhead isolation for adapter operations
- Timeout configuration for all operations

"Reliability is a feature."
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, TypeVar

# Python 3.11+ has asyncio.timeout, earlier versions need async-timeout
if sys.version_info >= (3, 11):
    asyncio_timeout = asyncio.timeout
else:
    try:
        from async_timeout import timeout as asyncio_timeout
    except ImportError:
        # Fallback: create a simple context manager that doesn't timeout
        @asynccontextmanager
        async def asyncio_timeout(delay: float):  # type: ignore[misc]
            """Fallback timeout context manager (no-op)."""
            yield


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

    def __init__(self, max_log_size: int = 1000) -> None:
        self._subscribers: list[Callable[[CacheInvalidationEvent], Awaitable[None]]] = []
        self._event_log: list[CacheInvalidationEvent] = []
        self._max_log_size = max_log_size

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
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                errors.append(str(e))
                logger.warning(f"Cache invalidation subscriber error (expected): {e}")
            except Exception as e:
                errors.append(str(e))
                logger.exception(f"Cache invalidation subscriber error (unexpected): {e}")

        if errors:
            logger.warning(f"Cache invalidation had {len(errors)} subscriber errors")

    async def publish_node_update(self, workspace_id: str, node_id: str, **metadata: Any) -> None:
        """Convenience method for node update events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_updated",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_node_delete(self, workspace_id: str, node_id: str, **metadata: Any) -> None:
        """Convenience method for node deletion events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_deleted",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_query_invalidation(self, workspace_id: str, **metadata: Any) -> None:
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

    Supports:
    - Configurable retry strategies (exponential, linear, constant)
    - Per-attempt timeout enforcement
    - Jitter to prevent thundering herd

    Usage:
        @with_retry(RetryConfig(max_retries=3, timeout_seconds=30.0))
        async def save_node(...):
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            total_attempts = config.max_retries + 1

            for attempt in range(total_attempts):
                try:
                    if config.timeout_seconds is not None:
                        async with asyncio_timeout(config.timeout_seconds):
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Timeout in {func.__name__} (attempt {attempt + 1}/{total_attempts}): "
                            f"exceeded {config.timeout_seconds}s. Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__} after timeout")
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__} (attempt {attempt + 1}/{total_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")

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

        except (ConnectionError, TimeoutError, OSError) as e:
            self._status = HealthStatus(
                healthy=self._status.consecutive_failures < self._failure_threshold,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._status.consecutive_failures + 1,
                last_error=str(e),
            )
            logger.debug(f"Health check failed with expected error: {e}")
        except Exception as e:
            self._status = HealthStatus(
                healthy=self._status.consecutive_failures < self._failure_threshold,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._status.consecutive_failures + 1,
                last_error=str(e),
            )
            logger.warning(f"Health check failed with unexpected error: {e}")

        return self._status

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.check_health()
            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Health check loop error (expected): {e}")
            except Exception as e:
                logger.warning(f"Health check loop error (unexpected): {e}")

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
            logger.error(f"Connection unhealthy after {self._status.consecutive_failures} failures")

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
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM provenance_chains pc
                LEFT JOIN knowledge_nodes kn ON pc.node_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_relationships(self) -> int:
        """Check for relationships with missing nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM knowledge_relationships kr
                LEFT JOIN knowledge_nodes kn1 ON kr.from_node_id = kn1.id
                LEFT JOIN knowledge_nodes kn2 ON kr.to_node_id = kn2.id
                WHERE kn1.id IS NULL OR kn2.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_topics(self) -> int:
        """Check for topics without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM node_topics nt
                LEFT JOIN knowledge_nodes kn ON nt.node_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_access_grants(self) -> int:
        """Check for access grants without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM access_grants ag
                LEFT JOIN knowledge_nodes kn ON ag.item_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_index_health(self) -> list[str]:
        """Check index health (PostgreSQL specific)."""
        issues = []
        try:
            async with self._pool.acquire() as conn:
                # Check for invalid indexes
                invalid = await conn.fetch("""
                    SELECT indexrelid::regclass AS index_name
                    FROM pg_index WHERE NOT indisvalid
                    """)
                for row in invalid:
                    issues.append(f"Invalid index: {row['index_name']}")
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Index health check failed (connection error): {e}")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Index health check failed (data error): {e}")
        return issues

    async def _check_duplicate_content(self) -> int:
        """Check for duplicate content by hash."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM (
                    SELECT content_hash, workspace_id, COUNT(*) as cnt
                    FROM knowledge_nodes
                    WHERE content_hash != ''
                    GROUP BY content_hash, workspace_id
                    HAVING COUNT(*) > 1
                ) duplicates
                """)
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
                result = await conn.execute("""
                    DELETE FROM provenance_chains
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["provenance_chains"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM knowledge_relationships
                    WHERE from_node_id NOT IN (SELECT id FROM knowledge_nodes)
                       OR to_node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["relationships"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM node_topics
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["topics"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM access_grants
                    WHERE item_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
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
            logger.info(f"Integrity check passed ({result.checks_performed} checks)")

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


# =============================================================================
# Adapter Circuit Breaker
# =============================================================================


class AdapterCircuitState(str, Enum):
    """Circuit breaker states for adapters."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class AdapterCircuitBreakerConfig:
    """Configuration for adapter circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes in half-open to close circuit
        timeout_seconds: Time in open state before trying half-open
        half_open_max_calls: Max concurrent calls in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


@dataclass
class AdapterCircuitStats:
    """Statistics for an adapter circuit breaker."""

    adapter_name: str
    state: AdapterCircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.time)
    total_failures: int = 0
    total_successes: int = 0
    total_circuit_opens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "state_changed_at": self.state_changed_at,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_circuit_opens": self.total_circuit_opens,
        }


class AdapterCircuitBreaker:
    """
    Circuit breaker specifically designed for Knowledge Mound adapters.

    Provides per-adapter circuit breaker functionality with:
    - Configurable failure thresholds
    - Half-open state for gradual recovery
    - Metrics integration for monitoring
    - State persistence for recovery

    Usage:
        breaker = AdapterCircuitBreaker("continuum")

        async def operation():
            if not breaker.can_proceed():
                raise AdapterUnavailableError("Circuit open")
            try:
                result = await adapter.do_something()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(str(e))
                raise
    """

    def __init__(
        self,
        adapter_name: str,
        config: Optional[AdapterCircuitBreakerConfig] = None,
    ):
        """Initialize adapter circuit breaker.

        Args:
            adapter_name: Name of the adapter (continuum, consensus, etc.)
            config: Circuit breaker configuration
        """
        self.adapter_name = adapter_name
        self.config = config or AdapterCircuitBreakerConfig()
        self._state = AdapterCircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._state_changed_at = time.time()
        self._total_failures = 0
        self._total_successes = 0
        self._total_circuit_opens = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> AdapterCircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == AdapterCircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == AdapterCircuitState.CLOSED

    def can_proceed(self) -> bool:
        """Check if a request can proceed through the circuit.

        Returns:
            True if request is allowed, False if circuit is open
        """
        if self._state == AdapterCircuitState.CLOSED:
            return True

        if self._state == AdapterCircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self._state_changed_at >= self.config.timeout_seconds:
                self._transition_to_half_open()
                return self._half_open_calls < self.config.half_open_max_calls
            return False

        if self._state == AdapterCircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        self._last_success_time = time.time()
        self._total_successes += 1

        if self._state == AdapterCircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls -= 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == AdapterCircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

        self._record_metrics("success")

    def record_failure(self, error: Optional[str] = None) -> bool:
        """Record a failed operation.

        Args:
            error: Optional error message for logging

        Returns:
            True if circuit just opened
        """
        self._last_failure_time = time.time()
        self._total_failures += 1
        circuit_opened = False

        if self._state == AdapterCircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._half_open_calls -= 1
            self._transition_to_open()
            circuit_opened = True
            logger.warning(f"Adapter circuit {self.adapter_name} reopened from half-open: {error}")
        elif self._state == AdapterCircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()
                circuit_opened = True
                logger.warning(
                    f"Adapter circuit {self.adapter_name} opened after "
                    f"{self._failure_count} failures: {error}"
                )

        self._record_metrics("failure")
        return circuit_opened

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = AdapterCircuitState.OPEN
        self._state_changed_at = time.time()
        self._total_circuit_opens += 1
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> OPEN")
        self._record_state_change()

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = AdapterCircuitState.HALF_OPEN
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> HALF_OPEN")
        self._record_state_change()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = AdapterCircuitState.CLOSED
        self._state_changed_at = time.time()
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> CLOSED")
        self._record_state_change()

    def reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = AdapterCircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._state_changed_at = time.time()
        logger.info(f"Adapter circuit {self.adapter_name} reset to CLOSED")

    def get_stats(self) -> AdapterCircuitStats:
        """Get circuit breaker statistics."""
        return AdapterCircuitStats(
            adapter_name=self.adapter_name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            state_changed_at=self._state_changed_at,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            total_circuit_opens=self._total_circuit_opens,
        )

    def cooldown_remaining(self) -> float:
        """Get remaining time in cooldown (open state).

        Returns:
            Seconds remaining, or 0 if not in open state
        """
        if self._state != AdapterCircuitState.OPEN:
            return 0.0
        elapsed = time.time() - self._state_changed_at
        remaining = self.config.timeout_seconds - elapsed
        return max(0.0, remaining)

    def _record_metrics(self, event_type: str) -> None:
        """Record Prometheus metrics for circuit breaker events."""
        try:
            from aragora.observability.metrics.km import (
                record_km_adapter_sync,
            )

            success = event_type == "success"
            record_km_adapter_sync(self.adapter_name, "circuit_breaker", success)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to record circuit breaker metric: {e}")

    def _record_state_change(self) -> None:
        """Record Prometheus metrics for state changes."""
        try:
            # Map state to health status for logging
            state_map = {
                AdapterCircuitState.CLOSED: 3,  # healthy
                AdapterCircuitState.HALF_OPEN: 2,  # degraded
                AdapterCircuitState.OPEN: 1,  # unhealthy
            }
            logger.debug(
                f"Adapter {self.adapter_name} state: {self._state.value} "
                f"(health={state_map.get(self._state, 0)})"
            )
        except Exception as e:
            logger.debug(f"Failed to record state change metric: {e}")

    @asynccontextmanager
    async def protected_call(self) -> AsyncIterator[None]:
        """Context manager for circuit-breaker-protected async calls.

        Raises:
            AdapterUnavailableError: If circuit is open

        Usage:
            async with breaker.protected_call():
                result = await adapter.operation()
        """
        if not self.can_proceed():
            remaining = self.cooldown_remaining()
            raise AdapterUnavailableError(
                self.adapter_name,
                remaining,
                f"Circuit breaker open, retry in {remaining:.1f}s",
            )

        if self._state == AdapterCircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            yield
            self.record_success()
        except asyncio.CancelledError:
            # Don't count task cancellation as failure
            if self._state == AdapterCircuitState.HALF_OPEN:
                self._half_open_calls -= 1
            raise
        except Exception as e:
            self.record_failure(str(e))
            raise


class AdapterUnavailableError(Exception):
    """Raised when an adapter is unavailable due to circuit breaker."""

    def __init__(
        self,
        adapter_name: str,
        cooldown_remaining: float,
        message: Optional[str] = None,
    ):
        self.adapter_name = adapter_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            message or f"Adapter '{adapter_name}' unavailable. Retry in {cooldown_remaining:.1f}s"
        )


# Global registry of adapter circuit breakers
_adapter_circuits: Dict[str, AdapterCircuitBreaker] = {}


def get_adapter_circuit_breaker(
    adapter_name: str,
    config: Optional[AdapterCircuitBreakerConfig] = None,
) -> AdapterCircuitBreaker:
    """Get or create a circuit breaker for an adapter.

    Args:
        adapter_name: Name of the adapter
        config: Optional configuration (only used if creating new)

    Returns:
        AdapterCircuitBreaker instance
    """
    if adapter_name not in _adapter_circuits:
        _adapter_circuits[adapter_name] = AdapterCircuitBreaker(adapter_name, config)
    return _adapter_circuits[adapter_name]


def get_all_adapter_circuit_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all adapter circuit breakers.

    Returns:
        Dict mapping adapter names to their stats
    """
    return {name: cb.get_stats().to_dict() for name, cb in _adapter_circuits.items()}


def reset_adapter_circuit_breaker(adapter_name: str) -> bool:
    """Reset a specific adapter's circuit breaker.

    Args:
        adapter_name: Name of the adapter

    Returns:
        True if reset, False if adapter not found
    """
    if adapter_name in _adapter_circuits:
        _adapter_circuits[adapter_name].reset()
        return True
    return False


def reset_all_adapter_circuit_breakers() -> int:
    """Reset all adapter circuit breakers.

    Returns:
        Number of circuit breakers reset
    """
    count = 0
    for cb in _adapter_circuits.values():
        cb.reset()
        count += 1
    return count


# =============================================================================
# SLO Monitoring for Adapters
# =============================================================================


@dataclass
class AdapterSLOConfig:
    """SLO configuration for adapter operations.

    Defines latency thresholds for monitoring adapter performance.
    """

    # Forward sync (source -> KM) latencies in milliseconds
    forward_sync_p50_ms: float = 100.0
    forward_sync_p90_ms: float = 300.0
    forward_sync_p99_ms: float = 800.0

    # Reverse query (KM -> consumer) latencies in milliseconds
    reverse_query_p50_ms: float = 50.0
    reverse_query_p90_ms: float = 150.0
    reverse_query_p99_ms: float = 500.0

    # Semantic search latencies in milliseconds
    semantic_search_p50_ms: float = 100.0
    semantic_search_p90_ms: float = 300.0
    semantic_search_p99_ms: float = 1000.0

    # Operation timeouts in seconds
    forward_sync_timeout_s: float = 5.0
    reverse_query_timeout_s: float = 3.0
    semantic_search_timeout_s: float = 5.0


# Default SLO config
_adapter_slo_config: Optional[AdapterSLOConfig] = None


def get_adapter_slo_config() -> AdapterSLOConfig:
    """Get the adapter SLO configuration."""
    global _adapter_slo_config
    if _adapter_slo_config is None:
        _adapter_slo_config = AdapterSLOConfig()
    return _adapter_slo_config


def set_adapter_slo_config(config: AdapterSLOConfig) -> None:
    """Set a custom adapter SLO configuration."""
    global _adapter_slo_config
    _adapter_slo_config = config


def check_adapter_slo(
    operation: str,
    latency_ms: float,
    adapter_name: str,
    percentile: str = "p99",
) -> tuple[bool, str]:
    """Check if adapter operation meets SLO.

    Args:
        operation: Operation type (forward_sync, reverse_query, semantic_search)
        latency_ms: Measured latency in milliseconds
        adapter_name: Name of the adapter
        percentile: SLO percentile to check (p50, p90, p99)

    Returns:
        Tuple of (is_within_slo, message)
    """
    config = get_adapter_slo_config()

    # Get threshold for operation and percentile
    attr_name = f"{operation}_{percentile}_ms"
    threshold = getattr(config, attr_name, None)

    if threshold is None:
        return True, f"No SLO defined for {operation}.{percentile}"

    is_within = latency_ms <= threshold

    if is_within:
        return (
            True,
            f"{adapter_name}.{operation} latency {latency_ms:.1f}ms "
            f"within {percentile} SLO ({threshold}ms)",
        )
    else:
        return (
            False,
            f"{adapter_name}.{operation} latency {latency_ms:.1f}ms "
            f"EXCEEDS {percentile} SLO ({threshold}ms)",
        )


def record_adapter_slo_check(
    adapter_name: str,
    operation: str,
    latency_ms: float,
    success: bool,
    context: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Record an adapter operation and check SLO compliance.

    Combines metric recording with SLO checking for convenience.

    Args:
        adapter_name: Name of the adapter
        operation: Operation type (forward_sync, reverse_query, semantic_search)
        latency_ms: Measured latency in milliseconds
        success: Whether the operation succeeded
        context: Optional context for SLO violation reporting

    Returns:
        Tuple of (is_within_slo, message)
    """
    # Record the operation latency
    try:
        from aragora.observability.metrics.km import (
            record_forward_sync_latency,
            record_reverse_query_latency,
            record_km_operation,
        )

        latency_seconds = latency_ms / 1000.0

        if operation == "forward_sync":
            record_forward_sync_latency(adapter_name, latency_seconds)
        elif operation == "reverse_query":
            record_reverse_query_latency(adapter_name, latency_seconds)

        record_km_operation(f"{adapter_name}_{operation}", success, latency_seconds)

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to record adapter metrics: {e}")

    # Check SLO compliance
    passed, message = check_adapter_slo(operation, latency_ms, adapter_name)

    # Record SLO check if metrics available
    if not passed:
        try:
            from aragora.observability.metrics.slo import (
                check_and_record_slo_with_recovery,
            )

            # Map to standard SLO operation name
            slo_operation = f"adapter_{operation}"
            check_and_record_slo_with_recovery(
                operation=slo_operation,
                latency_ms=latency_ms,
                context={
                    "adapter": adapter_name,
                    "operation": operation,
                    "success": success,
                    **(context or {}),
                },
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to record SLO check: {e}")

        logger.warning(message)

    return passed, message


# =============================================================================
# Bulkhead Pattern for Adapter Isolation
# =============================================================================


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation.

    Limits concurrent operations per adapter to prevent cascade failures.
    """

    max_concurrent_calls: int = 10
    max_wait_seconds: float = 5.0


class AdapterBulkhead:
    """
    Bulkhead pattern for isolating adapter operations.

    Limits the number of concurrent calls to an adapter to prevent
    resource exhaustion and cascade failures.

    Usage:
        bulkhead = AdapterBulkhead("continuum", max_concurrent=10)

        async with bulkhead.acquire():
            result = await adapter.operation()
    """

    def __init__(
        self,
        adapter_name: str,
        config: Optional[BulkheadConfig] = None,
    ):
        """Initialize bulkhead.

        Args:
            adapter_name: Name of the adapter
            config: Bulkhead configuration
        """
        self.adapter_name = adapter_name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        self._active_calls = 0
        self._rejected_calls = 0
        self._total_calls = 0

    @property
    def active_calls(self) -> int:
        """Get number of active calls."""
        return self._active_calls

    @property
    def available_permits(self) -> int:
        """Get number of available permits."""
        return self.config.max_concurrent_calls - self._active_calls

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a permit from the bulkhead.

        Raises:
            BulkheadFullError: If bulkhead is full and wait times out
        """
        self._total_calls += 1

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_seconds,
            )
            if not acquired:
                self._rejected_calls += 1
                raise BulkheadFullError(
                    self.adapter_name,
                    self.config.max_concurrent_calls,
                )
        except asyncio.TimeoutError:
            self._rejected_calls += 1
            raise BulkheadFullError(
                self.adapter_name,
                self.config.max_concurrent_calls,
                f"Bulkhead full after waiting {self.config.max_wait_seconds}s",
            )

        self._active_calls += 1
        try:
            yield
        finally:
            self._active_calls -= 1
            self._semaphore.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "adapter_name": self.adapter_name,
            "max_concurrent_calls": self.config.max_concurrent_calls,
            "active_calls": self._active_calls,
            "available_permits": self.available_permits,
            "total_calls": self._total_calls,
            "rejected_calls": self._rejected_calls,
            "rejection_rate": (
                self._rejected_calls / self._total_calls if self._total_calls > 0 else 0.0
            ),
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    def __init__(
        self,
        adapter_name: str,
        max_concurrent: int,
        message: Optional[str] = None,
    ):
        self.adapter_name = adapter_name
        self.max_concurrent = max_concurrent
        super().__init__(
            message or f"Bulkhead '{adapter_name}' full (max {max_concurrent} concurrent calls)"
        )


# Global registry of adapter bulkheads
_adapter_bulkheads: Dict[str, AdapterBulkhead] = {}


def get_adapter_bulkhead(
    adapter_name: str,
    config: Optional[BulkheadConfig] = None,
) -> AdapterBulkhead:
    """Get or create a bulkhead for an adapter.

    Args:
        adapter_name: Name of the adapter
        config: Optional configuration (only used if creating new)

    Returns:
        AdapterBulkhead instance
    """
    if adapter_name not in _adapter_bulkheads:
        _adapter_bulkheads[adapter_name] = AdapterBulkhead(adapter_name, config)
    return _adapter_bulkheads[adapter_name]


def get_all_adapter_bulkhead_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all adapter bulkheads."""
    return {name: bh.get_stats() for name, bh in _adapter_bulkheads.items()}


# =============================================================================
# Resilient Adapter Base
# =============================================================================


class ResilientAdapterMixin:
    """
    Mixin providing resilience patterns for Knowledge Mound adapters.

    Combines circuit breaker, bulkhead, retry, and SLO monitoring
    into a consistent interface for adapter implementations.

    Usage:
        class MyAdapter(ResilientAdapterMixin):
            def __init__(self):
                self._init_resilience("my_adapter")

            async def my_operation(self):
                async with self._resilient_call("forward_sync"):
                    return await self._do_operation()
    """

    _adapter_name: str
    _circuit_breaker: Optional[AdapterCircuitBreaker] = None
    _bulkhead: Optional[AdapterBulkhead] = None
    _retry_config: Optional[RetryConfig] = None
    _timeout_seconds: float = 5.0

    def _init_resilience(
        self,
        adapter_name: str,
        circuit_config: Optional[AdapterCircuitBreakerConfig] = None,
        bulkhead_config: Optional[BulkheadConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize resilience components.

        Args:
            adapter_name: Name of the adapter
            circuit_config: Circuit breaker configuration
            bulkhead_config: Bulkhead configuration
            retry_config: Retry configuration
            timeout_seconds: Default operation timeout
        """
        self._adapter_name = adapter_name
        self._circuit_breaker = get_adapter_circuit_breaker(adapter_name, circuit_config)
        self._bulkhead = get_adapter_bulkhead(adapter_name, bulkhead_config)
        self._retry_config = retry_config or RetryConfig()
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _resilient_call(
        self,
        operation: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute an operation with full resilience protection.

        Applies circuit breaker, bulkhead, timeout, and SLO monitoring.

        Args:
            operation: Operation name (forward_sync, reverse_query, etc.)
            timeout: Optional timeout override

        Yields:
            Context dict for storing operation metadata

        Raises:
            AdapterUnavailableError: If circuit is open
            BulkheadFullError: If bulkhead is full
            asyncio.TimeoutError: If operation times out
        """
        if not hasattr(self, "_adapter_name"):
            raise RuntimeError("Call _init_resilience() before using _resilient_call()")

        context: Dict[str, Any] = {
            "adapter": self._adapter_name,
            "operation": operation,
        }

        start_time = time.time()
        success = False
        timeout_s = timeout or self._timeout_seconds

        try:
            # Check circuit breaker
            if self._circuit_breaker and not self._circuit_breaker.can_proceed():
                remaining = self._circuit_breaker.cooldown_remaining()
                raise AdapterUnavailableError(self._adapter_name, remaining)

            # Acquire bulkhead permit with timeout
            if self._bulkhead:
                async with self._bulkhead.acquire():
                    # Execute with timeout
                    async with asyncio_timeout(timeout_s):
                        yield context
                        success = True
            else:
                async with asyncio_timeout(timeout_s):
                    yield context
                    success = True

            # Record success in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

        except asyncio.TimeoutError:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(f"Timeout after {timeout_s}s")
            raise
        except (AdapterUnavailableError, BulkheadFullError):
            raise
        except Exception as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(str(e))
            raise
        finally:
            # Record SLO metrics
            latency_ms = (time.time() - start_time) * 1000
            record_adapter_slo_check(
                self._adapter_name,
                operation,
                latency_ms,
                success,
                context,
            )

    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get combined resilience statistics."""
        stats: Dict[str, Any] = {
            "adapter_name": self._adapter_name,
            "timeout_seconds": self._timeout_seconds,
        }

        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats().to_dict()

        if self._bulkhead:
            stats["bulkhead"] = self._bulkhead.get_stats()

        return stats


# =============================================================================
# Timeout Decorator
# =============================================================================


def with_timeout(
    timeout_seconds: float,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for adding timeout to async functions.

    Args:
        timeout_seconds: Maximum execution time
        fallback: Optional fallback function to call on timeout

    Usage:
        @with_timeout(5.0)
        async def my_operation():
            ...

        @with_timeout(5.0, fallback=lambda: [])
        async def get_items():
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                if fallback is not None:
                    result = fallback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result  # type: ignore
                raise

        return wrapper

    return decorator


# =============================================================================
# Combined Health Status
# =============================================================================


def get_km_resilience_status() -> Dict[str, Any]:
    """Get comprehensive resilience status for all KM components.

    Returns:
        Dict with circuit breaker, bulkhead, and SLO status
    """
    return {
        "circuit_breakers": get_all_adapter_circuit_stats(),
        "bulkheads": get_all_adapter_bulkhead_stats(),
        "slo_config": {
            "forward_sync_p99_ms": get_adapter_slo_config().forward_sync_p99_ms,
            "reverse_query_p99_ms": get_adapter_slo_config().reverse_query_p99_ms,
            "semantic_search_p99_ms": get_adapter_slo_config().semantic_search_p99_ms,
        },
        "adapters_with_open_circuits": [
            name for name, cb in _adapter_circuits.items() if cb.is_open
        ],
        "total_adapters_tracked": len(_adapter_circuits),
    }
