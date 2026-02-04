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

from aragora.workflow.checkpoints._compat import (
    ASYNCPG_AVAILABLE,
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_OPERATION_TIMEOUT,
    MAX_CHECKPOINT_CACHE_SIZE,
    REDIS_AVAILABLE,
)
from aragora.workflow.checkpoints.cache import CachingCheckpointStore, LRUCheckpointCache
from aragora.workflow.checkpoints.exceptions import (
    CheckpointValidationError,
    ConnectionTimeoutError,
)
from aragora.workflow.checkpoints.factory import (
    get_checkpoint_store,
    get_checkpoint_store_async,
    get_default_knowledge_mound,
    set_default_knowledge_mound,
)
from aragora.workflow.checkpoints.file import FileCheckpointStore
from aragora.workflow.checkpoints.mound import KnowledgeMoundCheckpointStore
from aragora.workflow.checkpoints.postgres import PostgresCheckpointStore
from aragora.workflow.checkpoints.protocol import CheckpointStore
from aragora.workflow.checkpoints.redis import RedisCheckpointStore

__all__ = [
    # Exceptions
    "CheckpointValidationError",
    "ConnectionTimeoutError",
    # Protocol
    "CheckpointStore",
    # Cache
    "LRUCheckpointCache",
    "CachingCheckpointStore",
    # Store implementations
    "RedisCheckpointStore",
    "PostgresCheckpointStore",
    "KnowledgeMoundCheckpointStore",
    "FileCheckpointStore",
    # Factory functions
    "get_checkpoint_store",
    "get_checkpoint_store_async",
    "set_default_knowledge_mound",
    "get_default_knowledge_mound",
    # Constants
    "DEFAULT_CONNECTION_TIMEOUT",
    "DEFAULT_OPERATION_TIMEOUT",
    "MAX_CHECKPOINT_CACHE_SIZE",
    "REDIS_AVAILABLE",
    "ASYNCPG_AVAILABLE",
]
