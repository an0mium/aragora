"""Enhanced PostgreSQL store with resilience features."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from aragora.knowledge.mound.resilience.cache_invalidation import get_invalidation_bus
from aragora.knowledge.mound.resilience.health import ConnectionHealthMonitor
from aragora.knowledge.mound.resilience.integrity import IntegrityCheckResult, IntegrityVerifier
from aragora.knowledge.mound.resilience.retry import RetryConfig, with_retry
from aragora.knowledge.mound.resilience.transaction import (
    TransactionConfig,
    TransactionIsolation,
    TransactionManager,
)

logger = logging.getLogger(__name__)


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
        retry_config: RetryConfig | None = None,
        transaction_config: TransactionConfig | None = None,
    ):
        self._store = store
        self._retry_config = retry_config or RetryConfig()
        self._tx_config = transaction_config or TransactionConfig()

        self._tx_manager: TransactionManager | None = None
        self._health_monitor: ConnectionHealthMonitor | None = None
        self._integrity_verifier: IntegrityVerifier | None = None
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
        isolation: TransactionIsolation | None = None,
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
    async def get_node_async(self, node_id: str) -> dict[str, Any] | None:
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
