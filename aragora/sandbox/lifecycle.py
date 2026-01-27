"""
Session Container Lifecycle Manager.

Manages per-session container lifecycle with:
- Session-bound container allocation
- Policy-based execution control
- Resource tracking and cleanup
- Tenant isolation enforcement

Usage:
    from aragora.sandbox.lifecycle import SessionContainerManager

    manager = SessionContainerManager()
    await manager.start()

    # Create session container
    session = await manager.create_session("session-123", "tenant-456")

    # Execute in session
    result = await manager.execute("session-123", "print('hello')", "python")

    # Cleanup
    await manager.terminate_session("session-123")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from aragora.sandbox.executor import ExecutionStatus
from aragora.sandbox.policies import (
    ToolPolicy,
    ToolPolicyChecker,
    create_default_policy,
)
from aragora.sandbox.pool import (
    ContainerPool,
    PooledContainer,
    get_container_pool,
)

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """State of a container session."""

    CREATING = "creating"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


@dataclass
class SessionConfig:
    """Configuration for a container session."""

    max_execution_time_seconds: float = 300.0
    """Maximum time for a single execution."""

    max_session_duration_seconds: float = 3600.0
    """Maximum session lifetime (1 hour default)."""

    max_executions: int = 1000
    """Maximum executions per session."""

    max_memory_mb: int = 512
    """Memory limit for executions."""

    max_output_bytes: int = 1024 * 1024  # 1MB
    """Maximum output size per execution."""

    network_enabled: bool = False
    """Whether network access is allowed."""

    file_persistence: bool = True
    """Whether files persist between executions."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_session_duration_seconds": self.max_session_duration_seconds,
            "max_executions": self.max_executions,
            "max_memory_mb": self.max_memory_mb,
            "max_output_bytes": self.max_output_bytes,
            "network_enabled": self.network_enabled,
            "file_persistence": self.file_persistence,
        }


@dataclass
class ContainerSession:
    """A container session for code execution."""

    session_id: str
    """Unique session identifier."""

    tenant_id: str
    """Tenant this session belongs to."""

    user_id: Optional[str] = None
    """User who owns this session."""

    container: Optional[PooledContainer] = None
    """Container bound to this session."""

    state: SessionState = SessionState.CREATING
    """Current session state."""

    config: SessionConfig = field(default_factory=SessionConfig)
    """Session configuration."""

    policy: Optional[ToolPolicy] = None
    """Execution policy for this session."""

    created_at: float = field(default_factory=time.time)
    """When the session was created."""

    last_activity_at: float = field(default_factory=time.time)
    """When the session was last used."""

    execution_count: int = 0
    """Number of executions in this session."""

    total_execution_time_seconds: float = 0.0
    """Total execution time."""

    files_created: List[str] = field(default_factory=list)
    """Files created in the session workspace."""

    environment: Dict[str, str] = field(default_factory=dict)
    """Environment variables for executions."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def is_expired(self) -> bool:
        """Check if session has expired."""
        age = time.time() - self.created_at
        return age > self.config.max_session_duration_seconds

    def is_execution_allowed(self) -> bool:
        """Check if execution is allowed."""
        if self.state != SessionState.ACTIVE:
            return False
        if self.execution_count >= self.config.max_executions:
            return False
        return not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "container_id": self.container.container_id if self.container else None,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "execution_count": self.execution_count,
            "total_execution_time_seconds": self.total_execution_time_seconds,
            "files_created": self.files_created,
            "metadata": self.metadata,
        }


@dataclass
class SessionExecutionResult:
    """Result of a session execution."""

    session_id: str
    execution_id: str
    status: ExecutionStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    files_created: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "files_created": self.files_created,
            "policy_violations": self.policy_violations,
            "error_message": self.error_message,
        }


class SessionError(Exception):
    """Base exception for session errors."""

    pass


class SessionNotFoundError(SessionError):
    """Session not found."""

    pass


class SessionExpiredError(SessionError):
    """Session has expired."""

    pass


class ExecutionDeniedError(SessionError):
    """Execution denied by policy."""

    pass


class SessionContainerManager:
    """
    Manages per-session container lifecycle.

    Features:
    - Session-bound container allocation
    - Policy-based execution control
    - Resource tracking
    - Tenant isolation
    - Automatic cleanup
    """

    def __init__(
        self,
        pool: Optional[ContainerPool] = None,
        default_policy: Optional[ToolPolicy] = None,
    ):
        """Initialize the session container manager."""
        self._pool = pool
        self._default_policy = default_policy or create_default_policy()
        self._sessions: Dict[str, ContainerSession] = {}
        self._tenant_sessions: Dict[str, Set[str]] = {}  # tenant_id -> session_ids
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

    @property
    def pool(self) -> ContainerPool:
        """Get the container pool."""
        if self._pool is None:
            self._pool = get_container_pool()
        return self._pool

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    async def start(self) -> None:
        """Start the session manager."""
        if self._started:
            return

        logger.info("Starting session container manager")

        # Ensure pool is started
        if self.pool.state.value == "stopped":
            await self.pool.start()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._started = True

        logger.info("Session container manager started")

    async def stop(self, graceful: bool = True) -> None:
        """Stop the session manager."""
        if not self._started:
            return

        logger.info("Stopping session container manager")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Terminate all sessions
        async with self._lock:
            session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            await self.terminate_session(session_id)

        self._started = False
        logger.info("Session container manager stopped")

    # ==========================================================================
    # Session Management
    # ==========================================================================

    async def create_session(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
        policy: Optional[ToolPolicy] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContainerSession:
        """
        Create a new container session.

        Args:
            tenant_id: Tenant ID for isolation
            user_id: Optional user ID
            config: Session configuration
            policy: Execution policy
            session_id: Optional session ID (generated if not provided)
            metadata: Optional session metadata

        Returns:
            Created ContainerSession
        """
        session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"

        session = ContainerSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            config=config or SessionConfig(),
            policy=policy or self._default_policy,
            metadata=metadata or {},
        )

        logger.debug(f"Creating session {session_id} for tenant {tenant_id}")

        try:
            # Acquire container from pool
            container = await self.pool.acquire(session_id)
            session.container = container
            session.state = SessionState.ACTIVE

            # Register session
            async with self._lock:
                self._sessions[session_id] = session
                if tenant_id not in self._tenant_sessions:
                    self._tenant_sessions[tenant_id] = set()
                self._tenant_sessions[tenant_id].add(session_id)

            logger.info(f"Created session {session_id} with container {container.container_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise SessionError(f"Failed to create session: {e}") from e

    async def get_session(self, session_id: str) -> ContainerSession:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            ContainerSession

        Raises:
            SessionNotFoundError: If session not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)

        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        return session

    async def list_sessions(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        state: Optional[SessionState] = None,
    ) -> List[ContainerSession]:
        """
        List sessions with optional filtering.

        Args:
            tenant_id: Filter by tenant
            user_id: Filter by user
            state: Filter by state

        Returns:
            List of matching sessions
        """
        async with self._lock:
            sessions = list(self._sessions.values())

        # Apply filters
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if state:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    async def suspend_session(self, session_id: str) -> None:
        """
        Suspend a session (keep container but mark inactive).

        Args:
            session_id: Session ID
        """
        session = await self.get_session(session_id)

        if session.state != SessionState.ACTIVE:
            logger.warning(f"Cannot suspend session {session_id} in state {session.state}")
            return

        session.state = SessionState.SUSPENDED
        logger.info(f"Suspended session {session_id}")

    async def resume_session(self, session_id: str) -> None:
        """
        Resume a suspended session.

        Args:
            session_id: Session ID
        """
        session = await self.get_session(session_id)

        if session.state != SessionState.SUSPENDED:
            logger.warning(f"Cannot resume session {session_id} in state {session.state}")
            return

        if session.is_expired():
            raise SessionExpiredError(f"Session {session_id} has expired")

        session.state = SessionState.ACTIVE
        session.last_activity_at = time.time()
        logger.info(f"Resumed session {session_id}")

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a session and release its container.

        Args:
            session_id: Session ID
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                if session.tenant_id in self._tenant_sessions:
                    self._tenant_sessions[session.tenant_id].discard(session_id)

        if not session:
            return

        session.state = SessionState.TERMINATING

        # Release container back to pool
        if session.container:
            try:
                await self.pool.release(session_id)
            except Exception as e:
                logger.warning(f"Failed to release container for {session_id}: {e}")
                # Force destroy if release fails
                await self.pool.destroy(session_id)

        session.state = SessionState.TERMINATED
        logger.info(f"Terminated session {session_id}")

    # ==========================================================================
    # Execution
    # ==========================================================================

    async def execute(
        self,
        session_id: str,
        code: str,
        language: str = "python",
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> SessionExecutionResult:
        """
        Execute code in a session's container.

        Args:
            session_id: Session ID
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            env: Additional environment variables

        Returns:
            SessionExecutionResult
        """
        session = await self.get_session(session_id)

        # Validate session state
        if not session.is_execution_allowed():
            if session.is_expired():
                raise SessionExpiredError(f"Session {session_id} has expired")
            raise ExecutionDeniedError(
                f"Execution not allowed in session state {session.state.value}"
            )

        # Check policy
        if session.policy:
            checker = ToolPolicyChecker(session.policy)
            allowed, reason = checker.check_tool(self._language_to_tool(language))
            if not allowed:
                return SessionExecutionResult(
                    session_id=session_id,
                    execution_id=f"exec-denied-{uuid.uuid4().hex[:8]}",
                    status=ExecutionStatus.POLICY_DENIED,
                    policy_violations=[f"Tool denied: {reason}"],
                    error_message=f"Execution denied by policy: {reason}",
                )

        execution_id = f"exec-{uuid.uuid4().hex[:8]}"
        timeout = timeout or session.config.max_execution_time_seconds

        logger.debug(
            f"Executing {language} code in session {session_id} (execution {execution_id})"
        )

        start = time.time()

        try:
            result = await self._execute_in_container(
                session=session,
                execution_id=execution_id,
                code=code,
                language=language,
                timeout=timeout,
                env=env,
            )

            # Update session stats
            duration = time.time() - start
            session.execution_count += 1
            session.total_execution_time_seconds += duration
            session.last_activity_at = time.time()

            return result

        except asyncio.TimeoutError:
            return SessionExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {timeout:.1f}s",
            )

        except Exception as e:
            logger.error(f"Execution error in session {session_id}: {e}")
            return SessionExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )

    async def _execute_in_container(
        self,
        session: ContainerSession,
        execution_id: str,
        code: str,
        language: str,
        timeout: float,
        env: Optional[Dict[str, str]],
    ) -> SessionExecutionResult:
        """Execute code in the session's container."""
        if not session.container:
            raise SessionError("Session has no container")

        container_id = session.container.container_id

        # Build execution command
        cmd = self._build_execution_command(language)
        if not cmd:
            return SessionExecutionResult(
                session_id=session.session_id,
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Unsupported language: {language}",
            )

        # Merge environment
        exec_env = {**session.environment, **(env or {})}
        env_args = []
        for key, value in exec_env.items():
            env_args.extend(["-e", f"{key}={value}"])

        # Execute using docker exec
        full_cmd = [
            "docker",
            "exec",
            *env_args,
            "-i",  # Keep stdin open
            container_id,
            *cmd,
        ]

        start = time.time()

        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=code.encode()),
                timeout=timeout,
            )

            duration = time.time() - start

            # Truncate output if too large
            max_output = session.config.max_output_bytes
            stdout_str = stdout.decode("utf-8", errors="replace")[:max_output]
            stderr_str = stderr.decode("utf-8", errors="replace")[:max_output]

            return SessionExecutionResult(
                session_id=session.session_id,
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED
                if proc.returncode == 0
                else ExecutionStatus.FAILED,
                exit_code=proc.returncode or 0,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            # Kill the process
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise

    def _build_execution_command(self, language: str) -> Optional[List[str]]:
        """Build the execution command for a language."""
        commands = {
            "python": ["python3", "-c"],
            "python3": ["python3", "-c"],
            "javascript": ["node", "-e"],
            "node": ["node", "-e"],
            "bash": ["bash", "-c"],
            "shell": ["sh", "-c"],
            "sh": ["sh", "-c"],
        }
        return commands.get(language.lower())

    def _language_to_tool(self, language: str) -> str:
        """Map language to tool name for policy checking."""
        mapping = {
            "python": "python3",
            "python3": "python3",
            "javascript": "node",
            "node": "node",
            "bash": "bash",
            "shell": "sh",
            "sh": "sh",
        }
        return mapping.get(language.lower(), language)

    # ==========================================================================
    # Tenant Operations
    # ==========================================================================

    async def get_tenant_sessions(self, tenant_id: str) -> List[ContainerSession]:
        """Get all sessions for a tenant."""
        async with self._lock:
            session_ids = self._tenant_sessions.get(tenant_id, set()).copy()

        sessions = []
        for session_id in session_ids:
            try:
                session = await self.get_session(session_id)
                sessions.append(session)
            except SessionNotFoundError:
                pass

        return sessions

    async def terminate_tenant_sessions(self, tenant_id: str) -> int:
        """
        Terminate all sessions for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Number of sessions terminated
        """
        sessions = await self.get_tenant_sessions(tenant_id)

        for session in sessions:
            await self.terminate_session(session.session_id)

        logger.info(f"Terminated {len(sessions)} sessions for tenant {tenant_id}")
        return len(sessions)

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions."""
        async with self._lock:
            expired = [
                s.session_id
                for s in self._sessions.values()
                if s.is_expired() and s.state == SessionState.ACTIVE
            ]

        if expired:
            logger.debug(f"Cleaning up {len(expired)} expired sessions")
            for session_id in expired:
                await self.terminate_session(session_id)

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        async with self._lock:
            sessions = list(self._sessions.values())

        total = len(sessions)
        by_state = {}
        by_tenant = {}

        for session in sessions:
            state = session.state.value
            by_state[state] = by_state.get(state, 0) + 1

            tenant = session.tenant_id
            by_tenant[tenant] = by_tenant.get(tenant, 0) + 1

        total_executions = sum(s.execution_count for s in sessions)
        total_exec_time = sum(s.total_execution_time_seconds for s in sessions)

        return {
            "total_sessions": total,
            "by_state": by_state,
            "by_tenant": by_tenant,
            "total_executions": total_executions,
            "total_execution_time_seconds": total_exec_time,
            "pool_stats": self.pool.stats.to_dict() if self._pool else None,
        }


# ==========================================================================
# Global Instance
# ==========================================================================

_manager_instance: Optional[SessionContainerManager] = None


def get_session_manager() -> SessionContainerManager:
    """Get or create the global session container manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SessionContainerManager()
    return _manager_instance


def set_session_manager(manager: SessionContainerManager) -> None:
    """Set the global session container manager."""
    global _manager_instance
    _manager_instance = manager


__all__ = [
    "ContainerSession",
    "ExecutionDeniedError",
    "SessionConfig",
    "SessionContainerManager",
    "SessionError",
    "SessionExecutionResult",
    "SessionExpiredError",
    "SessionNotFoundError",
    "SessionState",
    "get_session_manager",
    "set_session_manager",
]
