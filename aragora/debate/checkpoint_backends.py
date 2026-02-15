"""
Checkpoint storage backends for debate checkpointing.

Extracted from checkpoint.py for modularity.

Backends:
- FileCheckpointStore: Local filesystem storage with optional gzip compression
- S3CheckpointStore: AWS S3 storage for distributed deployments
- GitCheckpointStore: Git branch-based storage with continuous commit mode
- DatabaseCheckpointStore: SQLite-based storage with atomic writes
- RecoveryNarrator: Git history summarizer for debate resumption context
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from .checkpoint import (
    SAFE_CHECKPOINT_ID,
    CheckpointStore,
    DebateCheckpoint,
)

logger = logging.getLogger(__name__)


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

    async def load(self, checkpoint_id: str) -> DebateCheckpoint | None:
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
        debate_id: str | None = None,
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
                raise RuntimeError("boto3 required but not installed. Run: pip install boto3")
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

    async def load(self, checkpoint_id: str) -> DebateCheckpoint | None:
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
        debate_id: str | None = None,
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
    """Git branch-based checkpoint storage for version control.

    Enhanced with Gastown-inspired continuous commit mode for crash recovery.

    Modes:
    - sparse (default): Create checkpoints at configured intervals
    - continuous: Commit after every round for maximum crash resilience

    The continuous mode enables any crash recovery scenario - debates can
    be resumed from the exact round where they were interrupted.
    """

    def __init__(
        self,
        repo_path: str = ".",
        branch_prefix: str = "checkpoint/",
        continuous_mode: bool = False,
        commit_message_template: str = "Debate {debate_id} round {round}",
    ):
        self.repo_path = Path(repo_path)
        self.branch_prefix = branch_prefix
        self.checkpoint_dir = self.repo_path / ".checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.continuous_mode = continuous_mode
        self.commit_message_template = commit_message_template
        self._commit_history: dict[str, list[str]] = {}  # debate_id -> commit hashes

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

    async def load(self, checkpoint_id: str) -> DebateCheckpoint | None:
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
        debate_id: str | None = None,
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

    async def commit_round(
        self,
        debate_id: str,
        round_num: int,
        checkpoint: DebateCheckpoint,
        message: str | None = None,
    ) -> str | None:
        """Commit a single round checkpoint (continuous mode).

        Returns the commit hash if successful, None otherwise.
        """
        if not self.continuous_mode:
            # In sparse mode, use standard save
            await self.save(checkpoint)
            return None

        # Validate checkpoint ID
        if not SAFE_CHECKPOINT_ID.match(checkpoint.checkpoint_id):
            raise ValueError(f"Invalid checkpoint ID format: {checkpoint.checkpoint_id}")

        # Save to file
        path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))

        # Commit with round-specific message
        commit_msg = message or self.commit_message_template.format(
            debate_id=debate_id,
            round=round_num,
        )

        await self._run_git(["add", str(path)])
        success, output = await self._run_git(["commit", "-m", commit_msg])

        if success:
            # Get commit hash
            _, commit_hash = await self._run_git(["rev-parse", "HEAD"])
            if debate_id not in self._commit_history:
                self._commit_history[debate_id] = []
            self._commit_history[debate_id].append(commit_hash)
            return commit_hash

        return None

    async def get_commit_history(
        self,
        debate_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get git commit history for a debate.

        Returns list of commits with hash, message, timestamp.
        """
        # Search for commits mentioning this debate
        success, log_output = await self._run_git(
            [
                "log",
                f"--grep=Debate {debate_id}",
                f"-{limit}",
                "--format=%H|%s|%ai",
            ]
        )

        if not success:
            return []

        commits = []
        for line in log_output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) >= 3:
                commits.append(
                    {
                        "hash": parts[0],
                        "message": parts[1],
                        "timestamp": parts[2],
                    }
                )

        return commits

    async def restore_to_round(
        self,
        debate_id: str,
        target_round: int,
    ) -> DebateCheckpoint | None:
        """Restore debate state to a specific round from git history.

        Searches commit history to find the checkpoint for the target round.
        """
        commits = await self.get_commit_history(debate_id)

        for commit in commits:
            if f"round {target_round}" in commit["message"]:
                # Checkout the checkpoint file at that commit
                success, content = await self._run_git(
                    [
                        "show",
                        f"{commit['hash']}:.checkpoints/cp-{debate_id[:8]}-{target_round:03d}*.json",
                    ]
                )
                if success and content:
                    try:
                        data = json.loads(content)
                        return DebateCheckpoint.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue

        return None


class RecoveryNarrator:
    """Summarizes git history for resuming debates (Gastown pattern).

    The RecoveryNarrator reads checkpoint commit history and generates
    a human-readable summary that can be injected into agent prompts
    when resuming a debate, providing context about what happened before
    the interruption.
    """

    def __init__(self, git_store: GitCheckpointStore):
        self.git_store = git_store

    async def generate_recovery_summary(
        self,
        debate_id: str,
        include_agent_states: bool = True,
        include_consensus_progress: bool = True,
        max_rounds_detail: int = 5,
    ) -> str:
        """Generate a narrative summary of debate progress for recovery.

        Args:
            debate_id: The debate to summarize
            include_agent_states: Include agent stance/position info
            include_consensus_progress: Include consensus trajectory
            max_rounds_detail: Number of recent rounds to detail

        Returns:
            Human-readable summary suitable for prompt injection
        """
        commits = await self.git_store.get_commit_history(debate_id)

        if not commits:
            return "No previous debate history found."

        # Load most recent checkpoint for full context
        latest_checkpoint = None
        for commit in commits:
            # Try to extract checkpoint from commit
            success, content = await self.git_store._run_git(
                [
                    "show",
                    f"{commit['hash']}:.checkpoints/*.json",
                ]
            )
            if success and content:
                try:
                    data = json.loads(content)
                    latest_checkpoint = DebateCheckpoint.from_dict(data)
                    break
                except (json.JSONDecodeError, KeyError):
                    continue

        # Build narrative
        lines = [
            "## Debate Recovery Context",
            "",
            f"**Debate ID:** {debate_id}",
            f"**Total Commits:** {len(commits)}",
        ]

        if latest_checkpoint:
            lines.extend(
                [
                    f"**Last Round:** {latest_checkpoint.current_round} of {latest_checkpoint.total_rounds}",
                    f"**Phase at Interruption:** {latest_checkpoint.phase}",
                    f"**Task:** {latest_checkpoint.task[:200]}",
                    "",
                ]
            )

            if include_consensus_progress and latest_checkpoint.current_consensus:
                lines.extend(
                    [
                        "### Consensus Progress",
                        f"**Current Working Consensus:** {latest_checkpoint.current_consensus[:500]}",
                        f"**Confidence:** {latest_checkpoint.consensus_confidence:.1%}",
                        f"**Convergence Status:** {latest_checkpoint.convergence_status}",
                        "",
                    ]
                )

            if include_agent_states and latest_checkpoint.agent_states:
                lines.append("### Agent Positions at Interruption")
                for agent in latest_checkpoint.agent_states:
                    lines.append(f"- **{agent.agent_name}** ({agent.agent_role}): {agent.stance}")
                lines.append("")

            # Recent round summaries
            if latest_checkpoint.messages:
                lines.append(f"### Recent Discussion (Last {max_rounds_detail} Rounds)")
                recent_messages = latest_checkpoint.messages[-max_rounds_detail * 3 :]
                for msg in recent_messages:
                    agent = msg.get("agent", "unknown")
                    content = msg.get("content", "")[:200]
                    lines.append(f"- **{agent}**: {content}...")
                lines.append("")

            if latest_checkpoint.intervention_notes:
                lines.append("### Human Interventions")
                for note in latest_checkpoint.intervention_notes:
                    lines.append(f"- {note}")
                lines.append("")

        # Commit timeline
        lines.append("### Checkpoint Timeline")
        for commit in commits[:10]:
            lines.append(f"- {commit['timestamp']}: {commit['message']}")

        return "\n".join(lines)

    async def get_resumption_prompt(
        self,
        debate_id: str,
        agent_name: str,
    ) -> str:
        """Generate a prompt injection for an agent resuming a debate.

        This provides the agent with context about:
        - What was being discussed
        - Where consensus stood
        - What their position was
        - What happened before interruption
        """
        summary = await self.generate_recovery_summary(debate_id)

        return f"""
You are resuming a debate that was interrupted. Here is the context:

{summary}

You are agent **{agent_name}**. Please continue the debate from where it left off,
maintaining consistency with your previous positions while remaining open to
new arguments. The debate will now continue.
"""


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

    async def load(self, checkpoint_id: str) -> DebateCheckpoint | None:
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
        debate_id: str | None = None,
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
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(DISTINCT debate_id) as debates,
                    SUM(LENGTH(data)) as total_bytes
                FROM checkpoints
            """)
            row = cursor.fetchone()

        pool_stats = self.get_pool_stats()
        return {
            "total_checkpoints": row[0],
            "unique_debates": row[1],
            "total_bytes": row[2] or 0,
            "db_path": str(self._db.db_path),
            "pool": pool_stats,
        }
