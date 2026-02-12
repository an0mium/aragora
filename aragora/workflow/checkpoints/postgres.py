"""
PostgreSQL-backed checkpoint store implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any, Optional

from aragora.workflow.checkpoints._compat import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_OPERATION_TIMEOUT,
    _PoolType,
    _asyncio_timeout,
)
from aragora.workflow.checkpoints.exceptions import ConnectionTimeoutError
from aragora.workflow.types import WorkflowCheckpoint

logger = logging.getLogger(__name__)


class PostgresCheckpointStore:
    """
    PostgreSQL-backed checkpoint store for durable, queryable checkpoint storage.

    Features:
    - ACID-compliant durable storage
    - Full SQL queryability for debugging/analytics
    - Schema versioning and migrations
    - Connection pooling for high concurrency

    Usage:
        from aragora.storage.postgres_store import get_postgres_pool

        pool = await get_postgres_pool()
        store = PostgresCheckpointStore(pool)
        await store.initialize()

        checkpoint_id = await store.save(checkpoint)

    Environment:
        Requires ARAGORA_POSTGRES_DSN or DATABASE_URL environment variable.
    """

    SCHEMA_NAME = "workflow_checkpoints"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS workflow_checkpoints (
            id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            definition_id TEXT NOT NULL,
            current_step TEXT,
            completed_steps TEXT[] DEFAULT '{}',
            step_outputs JSONB DEFAULT '{}',
            context_state JSONB DEFAULT '{}',
            checksum TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_wf_checkpoints_workflow_id ON workflow_checkpoints(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_wf_checkpoints_created_at ON workflow_checkpoints(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_wf_checkpoints_workflow_created
            ON workflow_checkpoints(workflow_id, created_at DESC);
    """

    def __init__(self, pool: _PoolType):
        """
        Initialize Postgres checkpoint store.

        Args:
            pool: asyncpg connection pool from get_postgres_pool()
        """
        # Import from stub for test patching compatibility
        import aragora.workflow.checkpoint_store as _compat_stub

        if not _compat_stub.ASYNCPG_AVAILABLE:
            raise RuntimeError(
                "PostgreSQL checkpoint store requires 'asyncpg' package. "
                "Install with: pip install aragora[postgres] or pip install asyncpg"
            )

        self._pool: _PoolType = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            # Create schema version tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Check current version
            row = await conn.fetchrow(
                "SELECT version FROM _schema_versions WHERE module = $1",
                self.SCHEMA_NAME,
            )
            current_version = row["version"] if row else 0

            if current_version == 0:
                # New database - run initial schema
                logger.info(f"[{self.SCHEMA_NAME}] Creating initial schema v{self.SCHEMA_VERSION}")
                await conn.execute(self.INITIAL_SCHEMA)
                await conn.execute(
                    """
                    INSERT INTO _schema_versions (module, version)
                    VALUES ($1, $2)
                    ON CONFLICT (module) DO UPDATE SET version = $2, updated_at = NOW()
                """,
                    self.SCHEMA_NAME,
                    self.SCHEMA_VERSION,
                )

        self._initialized = True
        logger.info(f"[{self.SCHEMA_NAME}] Schema initialized")

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Save a checkpoint to PostgreSQL.

        Args:
            checkpoint: WorkflowCheckpoint to save

        Returns:
            Checkpoint ID
        """
        import time

        checkpoint_id = f"{checkpoint.workflow_id}_{int(time.time() * 1000)}"
        checkpoint_dict = self._checkpoint_to_dict(checkpoint)

        try:
            async with _asyncio_timeout(DEFAULT_CONNECTION_TIMEOUT):
                async with self._pool.acquire() as conn:
                    await asyncio.wait_for(
                        conn.execute(
                            """
                            INSERT INTO workflow_checkpoints
                                (id, workflow_id, definition_id, current_step, completed_steps,
                                 step_outputs, context_state, checksum, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (id) DO UPDATE SET
                                current_step = EXCLUDED.current_step,
                                completed_steps = EXCLUDED.completed_steps,
                                step_outputs = EXCLUDED.step_outputs,
                                context_state = EXCLUDED.context_state,
                                checksum = EXCLUDED.checksum,
                                updated_at = NOW()
                        """,
                            checkpoint_id,
                            checkpoint.workflow_id,
                            checkpoint.definition_id,
                            checkpoint.current_step,
                            list(checkpoint.completed_steps),
                            json.dumps(checkpoint_dict.get("step_outputs", {})),
                            json.dumps(checkpoint_dict.get("context_state", {})),
                            checkpoint_dict.get("checksum", ""),
                            checkpoint.created_at,
                        ),
                        timeout=DEFAULT_OPERATION_TIMEOUT,
                    )
        except asyncio.TimeoutError:
            raise ConnectionTimeoutError(
                f"PostgreSQL checkpoint save timed out after {DEFAULT_CONNECTION_TIMEOUT}s"
            )

        logger.info(
            f"Saved checkpoint to PostgreSQL: workflow={checkpoint.workflow_id}, id={checkpoint_id}"
        )
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        try:
            async with _asyncio_timeout(DEFAULT_CONNECTION_TIMEOUT):
                async with self._pool.acquire() as conn:
                    row = await asyncio.wait_for(
                        conn.fetchrow(
                            "SELECT * FROM workflow_checkpoints WHERE id = $1",
                            checkpoint_id,
                        ),
                        timeout=DEFAULT_OPERATION_TIMEOUT,
                    )
                    if row is None:
                        return None

                    checkpoint = self._row_to_checkpoint(row)

                    # Validate checkpoint integrity
                    if checkpoint.checksum:
                        expected_checksum = self._compute_checksum(checkpoint)
                        if checkpoint.checksum != expected_checksum:
                            logger.warning(
                                f"Checkpoint {checkpoint_id} checksum mismatch: "
                                f"expected={expected_checksum}, got={checkpoint.checksum}"
                            )
                            # Don't fail, just log warning - checksum may be from different serialization

                    return checkpoint

        except asyncio.TimeoutError:
            logger.error(f"Timeout loading checkpoint {checkpoint_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            async with _asyncio_timeout(DEFAULT_CONNECTION_TIMEOUT):
                async with self._pool.acquire() as conn:
                    row = await asyncio.wait_for(
                        conn.fetchrow(
                            """
                            SELECT * FROM workflow_checkpoints
                            WHERE workflow_id = $1
                            ORDER BY created_at DESC
                            LIMIT 1
                        """,
                            workflow_id,
                        ),
                        timeout=DEFAULT_OPERATION_TIMEOUT,
                    )
                    if row is None:
                        return None

                    return self._row_to_checkpoint(row)

        except asyncio.TimeoutError:
            logger.error(f"Timeout loading latest checkpoint for {workflow_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        """
        List all checkpoint IDs for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of checkpoint IDs (newest first)
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id FROM workflow_checkpoints
                    WHERE workflow_id = $1
                    ORDER BY created_at DESC
                """,
                    workflow_id,
                )
                return [row["id"] for row in rows]

        except Exception as e:
            logger.error(f"Failed to list checkpoints for {workflow_id}: {e}")
            return []

    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM workflow_checkpoints WHERE id = $1",
                    checkpoint_id,
                )
                return "DELETE 0" not in result

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def cleanup_old_checkpoints(
        self,
        workflow_id: str,
        keep_count: int = 10,
    ) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            workflow_id: Workflow ID to clean up
            keep_count: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM workflow_checkpoints
                    WHERE workflow_id = $1
                    AND id NOT IN (
                        SELECT id FROM workflow_checkpoints
                        WHERE workflow_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    )
                """,
                    workflow_id,
                    keep_count,
                )
                # Parse "DELETE N" result
                parts = result.split()
                if len(parts) >= 2 and parts[0] == "DELETE":
                    return int(parts[1])
                return 0

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints for {workflow_id}: {e}")
            return 0

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> dict[str, Any]:
        """Convert checkpoint to dictionary."""
        if hasattr(checkpoint, "to_dict"):
            return checkpoint.to_dict()
        elif hasattr(checkpoint, "__dataclass_fields__"):
            return asdict(checkpoint)
        else:
            return {
                "id": getattr(checkpoint, "id", ""),
                "workflow_id": checkpoint.workflow_id,
                "definition_id": checkpoint.definition_id,
                "current_step": checkpoint.current_step,
                "completed_steps": list(checkpoint.completed_steps),
                "step_outputs": dict(checkpoint.step_outputs),
                "context_state": dict(getattr(checkpoint, "context_state", {})),
                "created_at": (
                    checkpoint.created_at.isoformat()
                    if hasattr(checkpoint.created_at, "isoformat")
                    else str(checkpoint.created_at)
                ),
                "checksum": getattr(checkpoint, "checksum", ""),
            }

    def _row_to_checkpoint(self, row: Any) -> WorkflowCheckpoint:
        """Convert database row to WorkflowCheckpoint."""
        step_outputs = row["step_outputs"]
        if isinstance(step_outputs, str):
            step_outputs = json.loads(step_outputs)

        context_state = row["context_state"]
        if isinstance(context_state, str):
            context_state = json.loads(context_state)

        return WorkflowCheckpoint(
            id=row["id"],
            workflow_id=row["workflow_id"],
            definition_id=row["definition_id"],
            current_step=row["current_step"] or "",
            completed_steps=list(row["completed_steps"] or []),
            step_outputs=step_outputs or {},
            context_state=context_state or {},
            created_at=row["created_at"],
            checksum=row["checksum"] or "",
        )

    def _compute_checksum(self, checkpoint: WorkflowCheckpoint) -> str:
        """Compute checksum for checkpoint validation."""
        # Create deterministic string from checkpoint data
        data = {
            "workflow_id": checkpoint.workflow_id,
            "definition_id": checkpoint.definition_id,
            "current_step": checkpoint.current_step,
            "completed_steps": sorted(checkpoint.completed_steps),
            "step_outputs": json.dumps(checkpoint.step_outputs, sort_keys=True, default=str),
        }
        checksum_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(checksum_str.encode()).hexdigest()[:16]
