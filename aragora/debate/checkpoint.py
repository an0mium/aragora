"""
Incremental Consensus Checkpointing.

Enables pause/resume for long-running debates:
- Durable checkpoints at configurable intervals
- Resume from last checkpoint on crash/timeout
- Async human participation (review + intervene + resume)
- Distributed debates across sessions

Key concepts:
- DebateCheckpoint: Full state snapshot at a point in time
- CheckpointStore: Persistence layer (file, S3, git)
- CheckpointManager: Orchestrates checkpointing lifecycle
- ResumedDebate: Context for continuing from checkpoint
"""

import asyncio
import gzip
import hashlib
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

# Git-safe ID pattern: alphanumeric, dash, underscore only (no path traversal or special chars)
SAFE_CHECKPOINT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

logger = logging.getLogger(__name__)

from aragora.core import Critique, Message, Vote
from aragora.exceptions import ConfigurationError


class CheckpointStatus(Enum):
    """Status of a checkpoint."""

    CREATING = "creating"
    COMPLETE = "complete"
    RESUMING = "resuming"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"


@dataclass
class AgentState:
    """Serialized state of an agent at checkpoint time."""

    agent_name: str
    agent_model: str
    agent_role: str
    system_prompt: str
    stance: str
    memory_snapshot: Optional[dict] = None


@dataclass
class DebateCheckpoint:
    """
    Complete state snapshot for debate resumption.

    Captures everything needed to continue a debate from
    exactly where it left off.
    """

    checkpoint_id: str
    debate_id: str
    task: str

    # Progress
    current_round: int
    total_rounds: int
    phase: str  # "proposal", "critique", "vote", "synthesis"

    # Message history
    messages: list[dict]  # Serialized Message objects
    critiques: list[dict]  # Serialized Critique objects
    votes: list[dict]  # Serialized Vote objects

    # Agent states
    agent_states: list[AgentState]

    # Consensus state
    current_consensus: Optional[str] = None
    consensus_confidence: float = 0.0
    convergence_status: str = ""

    # Claims kernel state (if using)
    claims_kernel_state: Optional[dict] = None

    # Belief network state (if using)
    belief_network_state: Optional[dict] = None

    # Continuum memory state (if using)
    continuum_memory_state: Optional[dict] = None

    # Metadata
    status: CheckpointStatus = CheckpointStatus.COMPLETE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None
    checksum: str = ""

    # Resumption info
    resume_count: int = 0
    last_resumed_at: Optional[str] = None
    resumed_by: Optional[str] = None  # User/system that resumed

    # Human intervention
    pending_intervention: bool = False
    intervention_notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = f"{self.debate_id}:{self.current_round}:{len(self.messages)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify checkpoint integrity."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "debate_id": self.debate_id,
            "task": self.task,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "phase": self.phase,
            "messages": self.messages,
            "critiques": self.critiques,
            "votes": self.votes,
            "agent_states": [
                {
                    "agent_name": s.agent_name,
                    "agent_model": s.agent_model,
                    "agent_role": s.agent_role,
                    "system_prompt": s.system_prompt,
                    "stance": s.stance,
                    "memory_snapshot": s.memory_snapshot,
                }
                for s in self.agent_states
            ],
            "current_consensus": self.current_consensus,
            "consensus_confidence": self.consensus_confidence,
            "convergence_status": self.convergence_status,
            "claims_kernel_state": self.claims_kernel_state,
            "belief_network_state": self.belief_network_state,
            "continuum_memory_state": self.continuum_memory_state,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "checksum": self.checksum,
            "resume_count": self.resume_count,
            "last_resumed_at": self.last_resumed_at,
            "resumed_by": self.resumed_by,
            "pending_intervention": self.pending_intervention,
            "intervention_notes": self.intervention_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebateCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            debate_id=data["debate_id"],
            task=data["task"],
            current_round=data["current_round"],
            total_rounds=data["total_rounds"],
            phase=data["phase"],
            messages=data["messages"],
            critiques=data["critiques"],
            votes=data["votes"],
            agent_states=[
                AgentState(
                    agent_name=s["agent_name"],
                    agent_model=s["agent_model"],
                    agent_role=s["agent_role"],
                    system_prompt=s["system_prompt"],
                    stance=s["stance"],
                    memory_snapshot=s.get("memory_snapshot"),
                )
                for s in data["agent_states"]
            ],
            current_consensus=data.get("current_consensus"),
            consensus_confidence=data.get("consensus_confidence", 0.0),
            convergence_status=data.get("convergence_status", ""),
            claims_kernel_state=data.get("claims_kernel_state"),
            belief_network_state=data.get("belief_network_state"),
            continuum_memory_state=data.get("continuum_memory_state"),
            status=CheckpointStatus(data.get("status", "complete")),
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            checksum=data["checksum"],
            resume_count=data.get("resume_count", 0),
            last_resumed_at=data.get("last_resumed_at"),
            resumed_by=data.get("resumed_by"),
            pending_intervention=data.get("pending_intervention", False),
            intervention_notes=data.get("intervention_notes", []),
        )


@dataclass
class ResumedDebate:
    """Context for a debate resumed from checkpoint."""

    checkpoint: DebateCheckpoint
    original_debate_id: str
    resumed_at: str
    resumed_by: str

    # Restored state
    messages: list[Message]
    votes: list[Vote]

    # Reconciliation
    context_drift_detected: bool = False
    drift_notes: list[str] = field(default_factory=list)


class CheckpointStore(ABC):
    """Abstract base for checkpoint persistence."""

    @abstractmethod
    async def save(self, checkpoint: DebateCheckpoint) -> str:
        """Save checkpoint, return storage path."""
        raise NotImplementedError("Subclasses must implement save")

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[DebateCheckpoint]:
        """Load checkpoint by ID."""
        raise NotImplementedError("Subclasses must implement load")

    @abstractmethod
    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List available checkpoints."""
        raise NotImplementedError("Subclasses must implement list_checkpoints")

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        raise NotImplementedError("Subclasses must implement delete")


class FileCheckpointStore(CheckpointStore):
    """File-based checkpoint storage."""

    def __init__(
        self,
        base_dir: str = ".checkpoints",
        compress: bool = True,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress

    def _sanitize_checkpoint_id(self, checkpoint_id: str) -> str:
        """Sanitize checkpoint ID to prevent path traversal attacks."""
        # Remove any path separators and parent directory references
        sanitized = checkpoint_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        # Only allow alphanumeric characters, hyphens, and underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
        if not sanitized:
            raise ValueError("Invalid checkpoint ID")
        return sanitized

    def _get_path(self, checkpoint_id: str) -> Path:
        ext = ".json.gz" if self.compress else ".json"
        sanitized_id = self._sanitize_checkpoint_id(checkpoint_id)
        path = self.base_dir / f"{sanitized_id}{ext}"
        # Ensure the resolved path is within base_dir (defense in depth)
        if not path.resolve().is_relative_to(self.base_dir):
            raise ValueError("Invalid checkpoint path")
        return path

    async def save(self, checkpoint: DebateCheckpoint) -> str:
        path = self._get_path(checkpoint.checkpoint_id)
        data = json.dumps(checkpoint.to_dict(), indent=2)

        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            path.write_text(data)

        return str(path)

    async def load(self, checkpoint_id: str) -> Optional[DebateCheckpoint]:
        path = self._get_path(checkpoint_id)

        if not path.exists():
            return None

        try:
            if self.compress:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(path.read_text())

            return DebateCheckpoint.from_dict(data)

        except (json.JSONDecodeError, gzip.BadGzipFile) as e:
            logger.warning(f"Corrupted checkpoint data {checkpoint_id}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid checkpoint structure {checkpoint_id}: {e}")
            return None
        except OSError as e:
            logger.debug(f"Cannot read checkpoint file {checkpoint_id}: {e}")
            return None

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        checkpoints = []
        pattern = "*.json.gz" if self.compress else "*.json"

        for path in sorted(self.base_dir.glob(pattern), reverse=True)[:limit]:
            try:
                cp = await self.load(path.stem.replace(".json", ""))
                if cp and (debate_id is None or cp.debate_id == debate_id):
                    checkpoints.append(
                        {
                            "checkpoint_id": cp.checkpoint_id,
                            "debate_id": cp.debate_id,
                            "task": cp.task[:100],
                            "current_round": cp.current_round,
                            "created_at": cp.created_at,
                            "status": cp.status.value,
                        }
                    )
            except (
                json.JSONDecodeError,
                gzip.BadGzipFile,
                KeyError,
                ValueError,
                TypeError,
                OSError,
            ) as e:
                logger.debug(f"Skipping invalid checkpoint file {path}: {e}")
                continue

        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        path = self._get_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False


class S3CheckpointStore(CheckpointStore):
    """S3-based checkpoint storage for distributed deployments."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "checkpoints/",
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise ConfigurationError(
                    component="S3CheckpointStore",
                    reason="boto3 required but not installed. Run: pip install boto3",
                )
        return self._client

    def _get_key(self, checkpoint_id: str) -> str:
        return f"{self.prefix}{checkpoint_id}.json.gz"

    async def save(self, checkpoint: DebateCheckpoint) -> str:
        client = self._get_client()
        key = self._get_key(checkpoint.checkpoint_id)

        data = json.dumps(checkpoint.to_dict())
        compressed = gzip.compress(data.encode())

        client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=compressed,
            ContentType="application/json",
            ContentEncoding="gzip",
        )

        return f"s3://{self.bucket}/{key}"

    async def load(self, checkpoint_id: str) -> Optional[DebateCheckpoint]:
        try:
            client = self._get_client()
            key = self._get_key(checkpoint_id)

            response = client.get_object(Bucket=self.bucket, Key=key)
            compressed = response["Body"].read()
            data = json.loads(gzip.decompress(compressed))

            return DebateCheckpoint.from_dict(data)

        except (json.JSONDecodeError, gzip.BadGzipFile) as e:
            logger.warning(f"Corrupted S3 checkpoint data {checkpoint_id}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid S3 checkpoint structure {checkpoint_id}: {e}")
            return None
        except ImportError:
            logger.error("boto3 required for S3CheckpointStore")
            return None
        except OSError as e:
            logger.warning(f"S3 connection error for {checkpoint_id}: {e}")
            return None

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        client = self._get_client()
        checkpoints = []

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                checkpoint_id = obj["Key"].replace(self.prefix, "").replace(".json.gz", "")
                cp = await self.load(checkpoint_id)
                if cp and (debate_id is None or cp.debate_id == debate_id):
                    checkpoints.append(
                        {
                            "checkpoint_id": cp.checkpoint_id,
                            "debate_id": cp.debate_id,
                            "task": cp.task[:100],
                            "current_round": cp.current_round,
                            "created_at": cp.created_at,
                            "status": cp.status.value,
                        }
                    )

                if len(checkpoints) >= limit:
                    break

        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        try:
            client = self._get_client()
            key = self._get_key(checkpoint_id)
            client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ImportError:
            logger.error("boto3 required for S3CheckpointStore")
            return False
        except OSError as e:
            logger.warning(f"S3 connection error deleting {checkpoint_id}: {e}")
            return False


class GitCheckpointStore(CheckpointStore):
    """Git branch-based checkpoint storage for version control."""

    def __init__(
        self,
        repo_path: str = ".",
        branch_prefix: str = "checkpoint/",
    ):
        self.repo_path = Path(repo_path)
        self.branch_prefix = branch_prefix
        self.checkpoint_dir = self.repo_path / ".checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    async def _run_git(self, args: list[str]) -> tuple[bool, str]:
        """Run git command asynchronously (non-blocking).

        Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30.0,
                )
                return proc.returncode == 0, stdout.decode("utf-8").strip()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "git command timed out"
        except FileNotFoundError:
            return False, "git not found in PATH"
        except (OSError, PermissionError) as e:
            logger.warning(f"Git command OS error: {e}")
            return False, str(e)
        except Exception as e:
            logger.exception(f"Unexpected git command error: {e}")
            return False, str(e)

    async def save(self, checkpoint: DebateCheckpoint) -> str:
        # Validate checkpoint ID for git safety
        if not SAFE_CHECKPOINT_ID.match(checkpoint.checkpoint_id):
            raise ValueError(f"Invalid checkpoint ID format: {checkpoint.checkpoint_id}")

        # Save to file
        path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))

        # Create git branch
        branch_name = f"{self.branch_prefix}{checkpoint.checkpoint_id}"
        await self._run_git(["checkout", "-b", branch_name])
        await self._run_git(["add", str(path)])
        await self._run_git(["commit", "-m", f"Checkpoint: {checkpoint.checkpoint_id}"])
        await self._run_git(["checkout", "-"])  # Return to previous branch

        return f"git:{branch_name}"

    async def load(self, checkpoint_id: str) -> Optional[DebateCheckpoint]:
        # Validate checkpoint ID for git safety
        if not SAFE_CHECKPOINT_ID.match(checkpoint_id):
            logger.warning(f"Invalid checkpoint ID format rejected: {checkpoint_id[:50]}")
            return None

        path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if path.exists():
            data = json.loads(path.read_text())
            return DebateCheckpoint.from_dict(data)

        # Try loading from git branch
        branch_name = f"{self.branch_prefix}{checkpoint_id}"
        success, _ = await self._run_git(
            ["show", f"{branch_name}:.checkpoints/{checkpoint_id}.json"]
        )

        if success:
            success, content = await self._run_git(
                ["show", f"{branch_name}:.checkpoints/{checkpoint_id}.json"]
            )
            if success:
                data = json.loads(content)
                return DebateCheckpoint.from_dict(data)

        return None

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        success, branches = await self._run_git(["branch", "-a"])
        checkpoints = []

        if success:
            for line in branches.split("\n"):
                branch = line.strip().replace("* ", "")
                if branch.startswith(self.branch_prefix):
                    checkpoint_id = branch.replace(self.branch_prefix, "")
                    cp = await self.load(checkpoint_id)
                    if cp and (debate_id is None or cp.debate_id == debate_id):
                        checkpoints.append(
                            {
                                "checkpoint_id": cp.checkpoint_id,
                                "debate_id": cp.debate_id,
                                "task": cp.task[:100],
                                "current_round": cp.current_round,
                                "created_at": cp.created_at,
                                "status": cp.status.value,
                            }
                        )

        return checkpoints[:limit]

    async def delete(self, checkpoint_id: str) -> bool:
        branch_name = f"{self.branch_prefix}{checkpoint_id}"
        success, _ = await self._run_git(["branch", "-D", branch_name])

        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if path.exists():
            path.unlink()

        return success


class DatabaseCheckpointStore(CheckpointStore):
    """
    SQLite-based checkpoint storage for single-machine deployments.

    Advantages over file storage:
    - Atomic writes (no partial checkpoints on crash)
    - Efficient queries (indexed by debate_id, created_at)
    - Built-in expiry with DELETE queries
    - Concurrent read access
    - Connection pooling via SQLiteStore

    For distributed deployments, use PostgreSQL with a connection pool
    by passing a PostgreSQL connection string.

    Uses SQLiteStore internally for standardized schema management.
    """

    SCHEMA_NAME = "checkpoints"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            task TEXT NOT NULL,
            current_round INTEGER NOT NULL,
            total_rounds INTEGER NOT NULL,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            data BLOB NOT NULL,
            checksum TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            compressed INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_checkpoints_debate_id
        ON checkpoints(debate_id);

        CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at
        ON checkpoints(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_checkpoints_expires_at
        ON checkpoints(expires_at);
    """

    def __init__(
        self,
        db_path: str = ".checkpoints/checkpoints.db",
        compress: bool = True,
        pool_size: int = 5,
    ):
        """
        Initialize database checkpoint store.

        Args:
            db_path: Path to SQLite database file
            compress: Whether to gzip checkpoint data before storing
            pool_size: Maximum number of connections (for backward compatibility)
        """
        from aragora.storage.base_store import SQLiteStore

        # Create SQLiteStore-based database wrapper
        class _CheckpointDB(SQLiteStore):
            SCHEMA_NAME = DatabaseCheckpointStore.SCHEMA_NAME
            SCHEMA_VERSION = DatabaseCheckpointStore.SCHEMA_VERSION
            INITIAL_SCHEMA = DatabaseCheckpointStore.INITIAL_SCHEMA

        self._db = _CheckpointDB(db_path, timeout=30.0)
        self.compress = compress
        self._pool_size = pool_size  # Kept for API compatibility

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics.

        Returns:
            Dict with pool stats and db_path
        """
        return {
            "available_connections": "managed_by_sqlitestore",
            "max_pool_size": self._pool_size,
            "db_path": str(self._db.db_path),
        }

    async def save(self, checkpoint: DebateCheckpoint) -> str:
        """Save checkpoint to database."""
        data = json.dumps(checkpoint.to_dict())

        if self.compress:
            data_bytes = gzip.compress(data.encode("utf-8"))
            compressed = 1
        else:
            data_bytes = data.encode("utf-8")
            compressed = 0

        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, debate_id, task, current_round, total_rounds,
                    phase, status, data, checksum, created_at, expires_at, compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.debate_id,
                    checkpoint.task[:500],  # Truncate for index efficiency
                    checkpoint.current_round,
                    checkpoint.total_rounds,
                    checkpoint.phase,
                    checkpoint.status.value,
                    data_bytes,
                    checkpoint.checksum,
                    checkpoint.created_at,
                    checkpoint.expires_at,
                    compressed,
                ),
            )

        return f"db:{checkpoint.checkpoint_id}"

    async def load(self, checkpoint_id: str) -> Optional[DebateCheckpoint]:
        """Load checkpoint from database."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT data, compressed FROM checkpoints
                WHERE checkpoint_id = ?
            """,
                (checkpoint_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        data_bytes, compressed = row

        try:
            if compressed:
                data = gzip.decompress(data_bytes).decode("utf-8")
            else:
                data = data_bytes.decode("utf-8")

            return DebateCheckpoint.from_dict(json.loads(data))

        except (json.JSONDecodeError, gzip.BadGzipFile, UnicodeDecodeError) as e:
            logger.warning(f"Corrupted checkpoint data {checkpoint_id}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid checkpoint structure {checkpoint_id}: {e}")
            return None

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List available checkpoints."""
        with self._db.connection() as conn:
            if debate_id:
                cursor = conn.execute(
                    """
                    SELECT checkpoint_id, debate_id, task, current_round,
                           created_at, status
                    FROM checkpoints
                    WHERE debate_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (debate_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT checkpoint_id, debate_id, task, current_round,
                           created_at, status
                    FROM checkpoints
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append(
                    {
                        "checkpoint_id": row[0],
                        "debate_id": row[1],
                        "task": row[2][:100],
                        "current_round": row[3],
                        "created_at": row[4],
                        "status": row[5],
                    }
                )

        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from database."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM checkpoints WHERE checkpoint_id = ?
            """,
                (checkpoint_id,),
            )
            return cursor.rowcount > 0

    async def cleanup_expired(self) -> int:
        """Delete expired checkpoints. Returns count deleted."""
        now = datetime.now().isoformat()
        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM checkpoints
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
                (now,),
            )
            return cursor.rowcount

    async def get_stats(self) -> dict:
        """Get checkpoint store statistics."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(DISTINCT debate_id) as debates,
                    SUM(LENGTH(data)) as total_bytes
                FROM checkpoints
            """
            )
            row = cursor.fetchone()

        pool_stats = self.get_pool_stats()
        return {
            "total_checkpoints": row[0],
            "unique_debates": row[1],
            "total_bytes": row[2] or 0,
            "db_path": str(self._db.db_path),
            "pool": pool_stats,
        }


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing behavior."""

    interval_rounds: int = 1  # Checkpoint every N rounds
    interval_seconds: float = 300.0  # Or every N seconds
    max_checkpoints: int = 10  # Keep at most N checkpoints per debate
    expiry_hours: float = 72.0  # Delete checkpoints after N hours
    compress: bool = True
    auto_cleanup: bool = True


class CheckpointManager:
    """
    Manages checkpoint lifecycle for debates.

    Handles creation, storage, resumption, and cleanup.
    """

    def __init__(
        self,
        store: Optional[CheckpointStore] = None,
        config: Optional[CheckpointConfig] = None,
    ):
        self.store = store or FileCheckpointStore()
        self.config = config or CheckpointConfig()

        self._last_checkpoint_time: dict[str, datetime] = {}
        self._checkpoint_count: dict[str, int] = {}

    def should_checkpoint(
        self,
        debate_id: str,
        current_round: int,
    ) -> bool:
        """Determine if a checkpoint should be created."""
        # Check round interval
        if current_round % self.config.interval_rounds == 0:
            return True

        # Check time interval
        last_time = self._last_checkpoint_time.get(debate_id)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed >= self.config.interval_seconds:
                return True

        return False

    async def create_checkpoint(
        self,
        debate_id: str,
        task: str,
        current_round: int,
        total_rounds: int,
        phase: str,
        messages: list[Message],
        critiques: list[Critique],
        votes: list[Vote],
        agents: list,  # Agent objects
        current_consensus: Optional[str] = None,
        claims_kernel_state: Optional[dict] = None,
        belief_network_state: Optional[dict] = None,
        continuum_memory_state: Optional[dict] = None,
    ) -> DebateCheckpoint:
        """Create and save a checkpoint."""
        checkpoint_id = f"cp-{debate_id[:8]}-{current_round:03d}-{uuid.uuid4().hex[:4]}"

        # Serialize messages
        messages_dict = [
            {
                "role": m.role,
                "agent": m.agent,
                "content": m.content,
                "timestamp": (
                    m.timestamp.isoformat()
                    if hasattr(m.timestamp, "isoformat")
                    else str(m.timestamp)
                ),
                "round": m.round,
            }
            for m in messages
        ]

        # Serialize critiques
        critiques_dict = [
            {
                "agent": c.agent,
                "target_agent": c.target_agent,
                "target_content": c.target_content,
                "issues": c.issues,
                "suggestions": c.suggestions,
                "severity": c.severity,
                "reasoning": c.reasoning,
            }
            for c in critiques
        ]

        # Serialize votes
        votes_dict = [
            {
                "agent": v.agent,
                "choice": v.choice,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
                "continue_debate": v.continue_debate,
            }
            for v in votes
        ]

        # Serialize agent states
        agent_states = [
            AgentState(
                agent_name=a.name,
                agent_model=a.model,
                agent_role=a.role,
                system_prompt=getattr(a, "system_prompt", ""),
                stance=getattr(a, "stance", "neutral"),
            )
            for a in agents
        ]

        # Calculate expiry
        expiry = None
        if self.config.expiry_hours > 0:
            expiry = (datetime.now() + timedelta(hours=self.config.expiry_hours)).isoformat()

        checkpoint = DebateCheckpoint(
            checkpoint_id=checkpoint_id,
            debate_id=debate_id,
            task=task,
            current_round=current_round,
            total_rounds=total_rounds,
            phase=phase,
            messages=messages_dict,
            critiques=critiques_dict,
            votes=votes_dict,
            agent_states=agent_states,
            current_consensus=current_consensus,
            claims_kernel_state=claims_kernel_state,
            belief_network_state=belief_network_state,
            continuum_memory_state=continuum_memory_state,
            expires_at=expiry,
        )

        # Save
        await self.store.save(checkpoint)

        # Track
        self._last_checkpoint_time[debate_id] = datetime.now()
        self._checkpoint_count[debate_id] = self._checkpoint_count.get(debate_id, 0) + 1

        # Cleanup old checkpoints if needed
        if self.config.auto_cleanup:
            await self._cleanup_old_checkpoints(debate_id)

        return checkpoint

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        resumed_by: str = "system",
    ) -> Optional[ResumedDebate]:
        """Resume a debate from a checkpoint."""
        checkpoint = await self.store.load(checkpoint_id)

        if not checkpoint:
            return None

        if not checkpoint.verify_integrity():
            checkpoint.status = CheckpointStatus.CORRUPTED
            return None

        # Restore messages
        messages = [
            Message(
                role=m["role"],
                agent=m["agent"],
                content=m["content"],
                timestamp=(
                    datetime.fromisoformat(m["timestamp"])
                    if isinstance(m["timestamp"], str)
                    else m["timestamp"]
                ),
                round=m["round"],
            )
            for m in checkpoint.messages
        ]

        # Restore votes
        votes = [
            Vote(
                agent=v["agent"],
                choice=v["choice"],
                confidence=v["confidence"],
                reasoning=v["reasoning"],
                continue_debate=v.get("continue_debate", True),
            )
            for v in checkpoint.votes
        ]

        # Update checkpoint
        checkpoint.resume_count += 1
        checkpoint.last_resumed_at = datetime.now().isoformat()
        checkpoint.resumed_by = resumed_by
        checkpoint.status = CheckpointStatus.RESUMING

        await self.store.save(checkpoint)

        return ResumedDebate(
            checkpoint=checkpoint,
            original_debate_id=checkpoint.debate_id,
            resumed_at=datetime.now().isoformat(),
            resumed_by=resumed_by,
            messages=messages,
            votes=votes,
        )

    async def add_intervention(
        self,
        checkpoint_id: str,
        note: str,
        by: str = "human",
    ) -> bool:
        """Add an intervention note to a checkpoint."""
        checkpoint = await self.store.load(checkpoint_id)

        if not checkpoint:
            return False

        checkpoint.pending_intervention = True
        checkpoint.intervention_notes.append(f"[{by}] {note}")

        await self.store.save(checkpoint)
        return True

    async def list_debates_with_checkpoints(self) -> list[dict]:
        """List all debates that have checkpoints."""
        all_checkpoints = await self.store.list_checkpoints()

        debates = {}
        for cp in all_checkpoints:
            debate_id = cp["debate_id"]
            if debate_id not in debates:
                debates[debate_id] = {
                    "debate_id": debate_id,
                    "task": cp["task"],
                    "checkpoint_count": 0,
                    "latest_checkpoint": None,
                    "latest_round": 0,
                }

            debates[debate_id]["checkpoint_count"] += 1
            if cp["current_round"] > debates[debate_id]["latest_round"]:
                debates[debate_id]["latest_round"] = cp["current_round"]
                debates[debate_id]["latest_checkpoint"] = cp["checkpoint_id"]

        return list(debates.values())

    async def _cleanup_old_checkpoints(self, debate_id: str):
        """Remove old checkpoints beyond the limit."""
        checkpoints = await self.store.list_checkpoints(debate_id=debate_id)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)

        # Delete extras
        for cp in checkpoints[self.config.max_checkpoints :]:
            await self.store.delete(cp["checkpoint_id"])


class CheckpointWebhook:
    """Webhook notifications for checkpoint events."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.handlers: dict[str, list[Callable]] = {
            "on_checkpoint": [],
            "on_resume": [],
            "on_intervention": [],
        }

    def on_checkpoint(self, handler: Callable) -> Callable:
        """Register checkpoint creation handler."""
        self.handlers["on_checkpoint"].append(handler)
        return handler

    def on_resume(self, handler: Callable) -> Callable:
        """Register resume handler."""
        self.handlers["on_resume"].append(handler)
        return handler

    def on_intervention(self, handler: Callable) -> Callable:
        """Register intervention handler."""
        self.handlers["on_intervention"].append(handler)
        return handler

    async def emit(self, event: str, data: dict) -> None:
        """Emit event to all handlers."""
        for handler in self.handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Checkpoint webhook handler failed for event '{event}': {e}")
            except Exception as e:
                logger.exception(
                    f"Unexpected error in checkpoint webhook handler for event '{event}': {e}"
                )

        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook(event, data)

    async def _send_webhook(self, event: str, data: dict):
        """Send webhook notification."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.webhook_url,
                    json={"event": event, "data": data},
                    timeout=aiohttp.ClientTimeout(total=10),
                )
        except ImportError as e:
            logger.debug(f"Webhook notification failed - aiohttp not available: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"Webhook notification failed - connection error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected webhook notification error: {e}")


# Convenience function for quick checkpointing
async def checkpoint_debate(
    debate_id: str,
    task: str,
    round_num: int,
    total_rounds: int,
    phase: str,
    messages: list[Message],
    agents: list,
    store_path: str = ".checkpoints",
) -> DebateCheckpoint:
    """Quick checkpoint creation."""
    manager = CheckpointManager(
        store=FileCheckpointStore(store_path),
        config=CheckpointConfig(),
    )

    return await manager.create_checkpoint(
        debate_id=debate_id,
        task=task,
        current_round=round_num,
        total_rounds=total_rounds,
        phase=phase,
        messages=messages,
        critiques=[],
        votes=[],
        agents=agents,
    )
