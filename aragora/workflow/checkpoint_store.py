"""
Checkpoint Store implementations for Workflow Engine.

Provides persistent storage for workflow checkpoints:
- RedisCheckpointStore: Fast distributed storage with TTL (production)
- PostgresCheckpointStore: Durable SQL storage with schema management (production)
- KnowledgeMoundCheckpointStore: Stores checkpoints in KnowledgeMound
- FileCheckpointStore: Stores checkpoints as local files (fallback)

Checkpoints enable:
- Crash recovery and resume
- Long-running workflow persistence
- Audit trail of workflow progress

Production Deployment:
    For production, prefer RedisCheckpointStore (fast) or PostgresCheckpointStore (durable).
    Use get_checkpoint_store() which auto-detects available backends.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

from aragora.workflow.types import WorkflowCheckpoint

logger = logging.getLogger(__name__)

# Optional Redis import - graceful degradation
try:
    from aragora.server.redis_config import get_redis_client

    REDIS_AVAILABLE = True
except ImportError:
    get_redis_client = None  # type: ignore
    REDIS_AVAILABLE = False
    logger.debug("Redis not available for checkpoint store")

# Optional asyncpg import - graceful degradation
try:
    import asyncpg
    from asyncpg import Pool

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    Pool = Any  # type: ignore
    ASYNCPG_AVAILABLE = False
    logger.debug("asyncpg not available for checkpoint store")


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint and return its ID."""
        ...

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a checkpoint by ID."""
        ...

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load the most recent checkpoint for a workflow."""
        ...

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoint IDs for a workflow."""
        ...

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        ...


class RedisCheckpointStore:
    """
    Redis-backed checkpoint store for distributed, fast checkpoint storage.

    Features:
    - Fast distributed storage across multiple server instances
    - TTL-based automatic cleanup of old checkpoints
    - JSON serialization with compression for large checkpoints
    - Atomic operations for checkpoint updates

    Usage:
        store = RedisCheckpointStore(ttl_hours=24)
        checkpoint_id = await store.save(checkpoint)
        checkpoint = await store.load_latest("workflow_123")

    Environment:
        Requires REDIS_URL environment variable or Redis configuration
        in aragora.server.redis_config.
    """

    # Key prefixes for Redis
    PREFIX = "aragora:workflow:checkpoint"
    WORKFLOW_INDEX_PREFIX = "aragora:workflow:index"

    def __init__(
        self,
        ttl_hours: float = 24.0,
        compress_threshold: int = 4096,
    ):
        """
        Initialize Redis checkpoint store.

        Args:
            ttl_hours: Time-to-live for checkpoints in hours (default 24h)
            compress_threshold: Compress checkpoints larger than this (bytes)
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis checkpoint store requires Redis configuration. "
                "Ensure aragora.server.redis_config is available and REDIS_URL is set."
            )

        self._ttl_seconds = int(ttl_hours * 3600)
        self._compress_threshold = compress_threshold
        self._redis = None

    def _get_redis(self) -> Any:
        """Get Redis client (lazy initialization)."""
        if self._redis is None:
            self._redis = get_redis_client()  # type: ignore
            if self._redis is None:
                raise RuntimeError("Redis client not available")
        return self._redis

    def _checkpoint_key(self, checkpoint_id: str) -> str:
        """Build Redis key for a checkpoint."""
        return f"{self.PREFIX}:{checkpoint_id}"

    def _workflow_index_key(self, workflow_id: str) -> str:
        """Build Redis key for workflow checkpoint index."""
        return f"{self.WORKFLOW_INDEX_PREFIX}:{workflow_id}"

    async def save(self, checkpoint: "WorkflowCheckpoint") -> str:
        """
        Save a checkpoint to Redis.

        Args:
            checkpoint: WorkflowCheckpoint to save

        Returns:
            Checkpoint ID
        """
        import time
        import zlib

        redis = self._get_redis()
        checkpoint_id = f"{checkpoint.workflow_id}_{int(time.time() * 1000)}"

        # Serialize checkpoint
        checkpoint_dict = self._checkpoint_to_dict(checkpoint)
        checkpoint_dict["checkpoint_id"] = checkpoint_id
        data = json.dumps(checkpoint_dict, default=str)

        # Compress if large
        data_bytes: bytes
        if len(data) > self._compress_threshold:
            data_bytes = zlib.compress(data.encode("utf-8"))
            is_compressed = True
        else:
            data_bytes = data.encode("utf-8")
            is_compressed = False

        # Store checkpoint
        key = self._checkpoint_key(checkpoint_id)
        redis.setex(key, self._ttl_seconds, data_bytes)

        # Store compression flag
        redis.setex(f"{key}:meta", self._ttl_seconds, json.dumps({"compressed": is_compressed}))

        # Add to workflow index (sorted set by timestamp)
        index_key = self._workflow_index_key(checkpoint.workflow_id)
        redis.zadd(index_key, {checkpoint_id: time.time()})
        redis.expire(index_key, self._ttl_seconds)

        logger.info(
            f"Saved checkpoint to Redis: workflow={checkpoint.workflow_id}, "
            f"id={checkpoint_id}, size={len(data_bytes)}, compressed={is_compressed}"
        )
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> Optional["WorkflowCheckpoint"]:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        import zlib

        try:
            redis = self._get_redis()
            key = self._checkpoint_key(checkpoint_id)

            # Get data
            data = redis.get(key)
            if data is None:
                return None

            # Check if compressed
            meta = redis.get(f"{key}:meta")
            is_compressed = False
            if meta:
                meta_dict = json.loads(meta)
                is_compressed = meta_dict.get("compressed", False)

            # Decompress if needed
            if is_compressed:
                data = zlib.decompress(data).decode("utf-8")
            elif isinstance(data, bytes):
                data = data.decode("utf-8")

            checkpoint_dict = json.loads(data)
            return self._dict_to_checkpoint(checkpoint_dict)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id} from Redis: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> Optional["WorkflowCheckpoint"]:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            redis = self._get_redis()
            index_key = self._workflow_index_key(workflow_id)

            # Get latest checkpoint ID from sorted set (highest score = most recent)
            results = redis.zrevrange(index_key, 0, 0)
            if not results:
                return None

            checkpoint_id = results[0]
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")

            return await self.load(checkpoint_id)

        except Exception as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """
        List all checkpoint IDs for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of checkpoint IDs (newest first)
        """
        try:
            redis = self._get_redis()
            index_key = self._workflow_index_key(workflow_id)

            results = redis.zrevrange(index_key, 0, -1)
            return [r.decode("utf-8") if isinstance(r, bytes) else r for r in results]

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
            redis = self._get_redis()
            key = self._checkpoint_key(checkpoint_id)

            # Delete checkpoint and metadata
            deleted = redis.delete(key, f"{key}:meta")

            # Remove from workflow index
            # Extract workflow_id from checkpoint_id (format: workflow_id_timestamp)
            parts = checkpoint_id.rsplit("_", 1)
            if len(parts) == 2:
                workflow_id = parts[0]
                index_key = self._workflow_index_key(workflow_id)
                redis.zrem(index_key, checkpoint_id)

            return deleted > 0

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def _checkpoint_to_dict(self, checkpoint: "WorkflowCheckpoint") -> Dict[str, Any]:
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

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Convert dictionary to WorkflowCheckpoint."""
        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return WorkflowCheckpoint(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step=data.get("current_step", ""),
            completed_steps=list(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            context_state=data.get("context_state", {}),
            created_at=created_at,
            checksum=data.get("checksum", ""),
        )


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

    def __init__(self, pool: "Pool"):
        """
        Initialize Postgres checkpoint store.

        Args:
            pool: asyncpg connection pool from get_postgres_pool()
        """
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError(
                "PostgreSQL checkpoint store requires 'asyncpg' package. "
                "Install with: pip install aragora[postgres] or pip install asyncpg"
            )

        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            # Create schema version tracking table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """
            )

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

    async def save(self, checkpoint: "WorkflowCheckpoint") -> str:
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

        async with self._pool.acquire() as conn:
            await conn.execute(
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
            )

        logger.info(
            f"Saved checkpoint to PostgreSQL: workflow={checkpoint.workflow_id}, "
            f"id={checkpoint_id}"
        )
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> Optional["WorkflowCheckpoint"]:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM workflow_checkpoints WHERE id = $1",
                    checkpoint_id,
                )
                if row is None:
                    return None

                return self._row_to_checkpoint(row)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> Optional["WorkflowCheckpoint"]:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM workflow_checkpoints
                    WHERE workflow_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    workflow_id,
                )
                if row is None:
                    return None

                return self._row_to_checkpoint(row)

        except Exception as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
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

    def _checkpoint_to_dict(self, checkpoint: "WorkflowCheckpoint") -> Dict[str, Any]:
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

    def _row_to_checkpoint(self, row: Any) -> "WorkflowCheckpoint":
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


class KnowledgeMoundCheckpointStore:
    """
    Stores workflow checkpoints in KnowledgeMound.

    Checkpoints are stored as KnowledgeNodes with:
    - node_type: "workflow_checkpoint"
    - content: Serialized checkpoint state
    - provenance: Workflow ID, step ID, timestamp
    - tier: MEDIUM (balance between persistence and cleanup)

    This enables:
    - Unified storage with other knowledge
    - Cross-workspace checkpoint access
    - Automatic tier-based cleanup
    - Semantic search over checkpoints

    Usage:
        from aragora.knowledge.mound import KnowledgeMound
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mound = KnowledgeMound(workspace_id="my_workspace")
        store = KnowledgeMoundCheckpointStore(mound)

        # Save checkpoint
        checkpoint_id = await store.save(checkpoint)

        # Resume from checkpoint
        checkpoint = await store.load_latest("workflow_123")
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize checkpoint store with KnowledgeMound backend.

        Args:
            mound: KnowledgeMound instance for storage
            workspace_id: Optional workspace override (defaults to mound's workspace)
        """
        self.mound = mound
        self.workspace_id = workspace_id or getattr(mound, "_workspace_id", "default")

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Save a checkpoint to KnowledgeMound.

        Args:
            checkpoint: WorkflowCheckpoint to save

        Returns:
            Checkpoint node ID
        """
        try:
            from aragora.knowledge.mound import KnowledgeNode, MemoryTier, ProvenanceChain  # type: ignore[attr-defined]

            # Serialize checkpoint to JSON
            checkpoint_dict = self._checkpoint_to_dict(checkpoint)
            content = json.dumps(checkpoint_dict, indent=2, default=str)

            # Build provenance chain
            provenance = ProvenanceChain(  # type: ignore[call-arg]
                source_type="workflow_engine",
                source_id=checkpoint.workflow_id,
                timestamp=datetime.now().isoformat(),
                chain=[
                    {
                        "workflow_id": checkpoint.workflow_id,
                        "step_id": checkpoint.current_step,
                        "steps_completed": len(checkpoint.completed_steps),
                    }
                ],
                metadata={
                    "checkpoint_type": "workflow",
                    "steps_completed": len(checkpoint.completed_steps),
                },
            )

            # Create knowledge node
            node = KnowledgeNode(  # type: ignore[call-arg]
                node_type="workflow_checkpoint",
                content=content,
                confidence=1.0,  # Checkpoints are authoritative
                provenance=provenance,
                tier=MemoryTier.MEDIUM,  # Balance persistence and cleanup
                workspace_id=self.workspace_id,
            )

            # Store in mound
            node_id = await self.mound.add_node(node)  # type: ignore[arg-type]
            logger.info(
                f"Saved workflow checkpoint: workflow={checkpoint.workflow_id}, "
                f"step={checkpoint.current_step}, node_id={node_id}"
            )
            return node_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint to KnowledgeMound: {e}")
            raise

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Load a checkpoint by its node ID.

        Args:
            checkpoint_id: KnowledgeNode ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        try:
            node = await self.mound.get_node(checkpoint_id)  # type: ignore[arg-type]
            if node is None:
                return None

            if node.node_type != "workflow_checkpoint":
                logger.warning(f"Node {checkpoint_id} is not a checkpoint")
                return None

            checkpoint_dict = json.loads(node.content)
            return self._dict_to_checkpoint(checkpoint_dict)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID to find checkpoint for

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            # Query for checkpoints with this workflow ID
            nodes = await self.mound.query_by_provenance(  # type: ignore[attr-defined]
                source_type="workflow_engine",
                source_id=workflow_id,
                node_type="workflow_checkpoint",
                limit=1,
            )

            if not nodes:
                return None

            # Get the most recent (should be first due to ordering)
            latest_node = nodes[0]
            checkpoint_dict = json.loads(latest_node.content)
            return self._dict_to_checkpoint(checkpoint_dict)

        except Exception as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """
        List all checkpoint IDs for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of checkpoint node IDs
        """
        try:
            nodes = await self.mound.query_by_provenance(  # type: ignore[attr-defined]
                source_type="workflow_engine",
                source_id=workflow_id,
                node_type="workflow_checkpoint",
                limit=100,
            )
            return [node.id for node in nodes]

        except Exception as e:
            logger.error(f"Failed to list checkpoints for {workflow_id}: {e}")
            return []

    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint node ID to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            return await self.mound.delete_node(checkpoint_id)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        if hasattr(checkpoint, "to_dict"):
            return checkpoint.to_dict()
        elif hasattr(checkpoint, "__dataclass_fields__"):
            return asdict(checkpoint)
        else:
            # Fallback for objects without to_dict - matches WorkflowCheckpoint fields
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

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary back to WorkflowCheckpoint."""
        from datetime import datetime

        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return WorkflowCheckpoint(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step=data.get("current_step", ""),
            completed_steps=list(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            context_state=data.get("context_state", {}),
            created_at=created_at,
            checksum=data.get("checksum", ""),
        )


class FileCheckpointStore:
    """
    Fallback checkpoint store using local files.

    Stores checkpoints as JSON files in a specified directory.
    Useful for development or when KnowledgeMound is unavailable.

    Usage:
        store = FileCheckpointStore("/path/to/checkpoints")
        checkpoint_id = await store.save(checkpoint)
    """

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize file-based checkpoint store.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint to a file."""
        checkpoint_id = f"{checkpoint.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        checkpoint_dict = self._checkpoint_to_dict(checkpoint)
        checkpoint_dict["checkpoint_id"] = checkpoint_id

        file_path.write_text(json.dumps(checkpoint_dict, indent=2, default=str))
        logger.info(f"Saved checkpoint to file: {file_path}")
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a checkpoint from a file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text())
        return self._dict_to_checkpoint(data)

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load the most recent checkpoint for a workflow."""
        matching_files = sorted(
            self.checkpoint_dir.glob(f"{workflow_id}_*.json"),
            reverse=True,
        )
        if not matching_files:
            return None

        data = json.loads(matching_files[0].read_text())
        return self._dict_to_checkpoint(data)

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoint IDs for a workflow."""
        return [f.stem for f in self.checkpoint_dir.glob(f"{workflow_id}_*.json")]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        if hasattr(checkpoint, "to_dict"):
            return checkpoint.to_dict()
        elif hasattr(checkpoint, "__dataclass_fields__"):
            return asdict(checkpoint)
        else:
            # Fallback for objects without to_dict - matches WorkflowCheckpoint fields
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

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary to WorkflowCheckpoint."""
        from datetime import datetime

        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return WorkflowCheckpoint(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step=data.get("current_step", ""),
            completed_steps=list(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            context_state=data.get("context_state", {}),
            created_at=created_at,
            checksum=data.get("checksum", ""),
        )


# Module-level default KnowledgeMound for checkpoint storage
_default_mound: Optional["KnowledgeMound"] = None


def set_default_knowledge_mound(mound: "KnowledgeMound") -> None:
    """
    Set the default KnowledgeMound for checkpoint storage.

    When set, get_checkpoint_store() will use KnowledgeMoundCheckpointStore
    instead of FileCheckpointStore, enabling durable checkpoint persistence
    with the Knowledge Mound backend.

    Usage:
        from aragora.knowledge.mound import KnowledgeMound
        from aragora.workflow.checkpoint_store import set_default_knowledge_mound

        mound = KnowledgeMound(workspace_id="production")
        await mound.initialize()
        set_default_knowledge_mound(mound)

    Args:
        mound: KnowledgeMound instance to use for checkpoints
    """
    global _default_mound
    _default_mound = mound
    logger.info("Set default KnowledgeMound for workflow checkpoints")


def get_default_knowledge_mound() -> Optional["KnowledgeMound"]:
    """Get the default KnowledgeMound for checkpoint storage."""
    return _default_mound


def get_checkpoint_store(
    mound: Optional["KnowledgeMound"] = None,
    fallback_dir: str = ".checkpoints",
    use_default_mound: bool = True,
    prefer_redis: bool = True,
    prefer_postgres: bool = False,
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store based on availability.

    Priority order:
    1. Explicitly provided KnowledgeMound
    2. Default KnowledgeMound (if set via set_default_knowledge_mound)
    3. RedisCheckpointStore (if prefer_redis=True and Redis is available)
    4. PostgresCheckpointStore (if prefer_postgres=True and Postgres is available)
    5. FileCheckpointStore (persistent file-based fallback)

    Args:
        mound: Optional KnowledgeMound instance (highest priority)
        fallback_dir: Fallback directory for file-based storage
        use_default_mound: Whether to use the default mound if no mound provided
        prefer_redis: Try Redis before file fallback (default True)
        prefer_postgres: Try Postgres before file fallback (default False)

    Returns:
        CheckpointStore implementation
    """
    # Use explicitly provided mound
    if mound is not None:
        logger.debug("Using provided KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(mound)

    # Try default mound
    if use_default_mound and _default_mound is not None:
        logger.debug("Using default KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(_default_mound)

    # Try Redis if preferred
    if prefer_redis and REDIS_AVAILABLE:
        try:
            store = RedisCheckpointStore()
            # Test Redis availability
            redis = store._get_redis()
            if redis is not None:
                logger.info("Using RedisCheckpointStore for checkpoints")
                return store
        except Exception as e:
            logger.debug(f"Redis checkpoint store not available: {e}")

    # Try Postgres if preferred
    if prefer_postgres and ASYNCPG_AVAILABLE:
        try:
            # Import here to avoid circular imports
            import asyncio
            from aragora.storage.postgres_store import get_postgres_pool

            # Get pool synchronously if possible
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync context, skip Postgres
                    logger.debug("Postgres pool requires async context, skipping")
                else:
                    pool = loop.run_until_complete(get_postgres_pool())  # type: ignore[arg-type]
                    store = PostgresCheckpointStore(pool)  # type: ignore[arg-type]
                    loop.run_until_complete(store.initialize())
                    logger.info("Using PostgresCheckpointStore for checkpoints")
                    return store
            except RuntimeError:
                # No event loop
                logger.debug("No event loop for Postgres initialization")
        except Exception as e:
            logger.debug(f"Postgres checkpoint store not available: {e}")

    # Fall back to file-based storage
    # SECURITY: Check production guards before allowing file fallback
    try:
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "checkpoint_store",
            StorageMode.FILE,
            "No Redis or PostgreSQL available for checkpoint storage",
        )
    except ImportError:
        pass  # Guards not available, allow fallback

    logger.debug(f"Using FileCheckpointStore in {fallback_dir}")
    return FileCheckpointStore(fallback_dir)


async def get_checkpoint_store_async(
    mound: Optional["KnowledgeMound"] = None,
    fallback_dir: str = ".checkpoints",
    use_default_mound: bool = True,
    prefer_redis: bool = True,
    prefer_postgres: bool = True,
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store (async version).

    This is the recommended function for async contexts as it properly
    initializes Postgres connection pools.

    Priority order:
    1. Explicitly provided KnowledgeMound
    2. Default KnowledgeMound (if set via set_default_knowledge_mound)
    3. RedisCheckpointStore (if prefer_redis=True and Redis is available)
    4. PostgresCheckpointStore (if prefer_postgres=True and Postgres is available)
    5. FileCheckpointStore (persistent file-based fallback)

    Args:
        mound: Optional KnowledgeMound instance (highest priority)
        fallback_dir: Fallback directory for file-based storage
        use_default_mound: Whether to use the default mound if no mound provided
        prefer_redis: Try Redis before file fallback (default True)
        prefer_postgres: Try Postgres before file fallback (default True)

    Returns:
        CheckpointStore implementation
    """
    # Use explicitly provided mound
    if mound is not None:
        logger.debug("Using provided KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(mound)

    # Try default mound
    if use_default_mound and _default_mound is not None:
        logger.debug("Using default KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(_default_mound)

    # Try Redis if preferred
    if prefer_redis and REDIS_AVAILABLE:
        try:
            store = RedisCheckpointStore()
            redis = store._get_redis()
            if redis is not None:
                logger.info("Using RedisCheckpointStore for checkpoints")
                return store
        except Exception as e:
            logger.debug(f"Redis checkpoint store not available: {e}")

    # Try Postgres if preferred
    if prefer_postgres and ASYNCPG_AVAILABLE:
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            pool = await get_postgres_pool()  # type: ignore[arg-type]
            store = PostgresCheckpointStore(pool)  # type: ignore[arg-type]
            await store.initialize()
            logger.info("Using PostgresCheckpointStore for checkpoints")
            return store
        except Exception as e:
            logger.debug(f"Postgres checkpoint store not available: {e}")

    # Fall back to file-based storage
    # SECURITY: Check production guards before allowing file fallback
    try:
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "checkpoint_store",
            StorageMode.FILE,
            "No Redis or PostgreSQL available for checkpoint storage (async)",
        )
    except ImportError:
        pass  # Guards not available, allow fallback

    logger.debug(f"Using FileCheckpointStore in {fallback_dir}")
    return FileCheckpointStore(fallback_dir)
