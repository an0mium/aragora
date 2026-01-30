"""
Tests for Session Container Lifecycle Manager.

Tests cover:
- SessionContainerManager lifecycle (start, stop)
- Session creation and termination
- Session states (active, suspended, terminated)
- Execution within sessions
- Policy enforcement in sessions
- Tenant isolation
- Session expiration
- Statistics collection
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.sandbox.executor import ExecutionStatus
from aragora.sandbox.lifecycle import (
    ContainerSession,
    ExecutionDeniedError,
    SessionConfig,
    SessionContainerManager,
    SessionError,
    SessionExecutionResult,
    SessionExpiredError,
    SessionNotFoundError,
    SessionState,
    get_session_manager,
    set_session_manager,
)
from aragora.sandbox.policies import (
    PolicyAction,
    ToolPolicy,
    create_default_policy,
    create_strict_policy,
)
from aragora.sandbox.pool import (
    ContainerPool,
    ContainerPoolConfig,
    PooledContainer,
    PoolState,
)


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_config(self):
        """Test default session configuration."""
        config = SessionConfig()

        assert config.max_execution_time_seconds == 300.0
        assert config.max_session_duration_seconds == 3600.0
        assert config.max_executions == 1000
        assert config.max_memory_mb == 512
        assert config.max_output_bytes == 1024 * 1024
        assert config.network_enabled is False
        assert config.file_persistence is True

    def test_custom_config(self):
        """Test custom session configuration."""
        config = SessionConfig(
            max_execution_time_seconds=60.0,
            max_session_duration_seconds=1800.0,
            max_executions=100,
            network_enabled=True,
        )

        assert config.max_execution_time_seconds == 60.0
        assert config.max_session_duration_seconds == 1800.0
        assert config.max_executions == 100
        assert config.network_enabled is True

    def test_config_to_dict(self):
        """Test session config serialization."""
        config = SessionConfig(max_memory_mb=256)
        data = config.to_dict()

        assert data["max_memory_mb"] == 256
        assert "max_execution_time_seconds" in data
        assert "network_enabled" in data


class TestContainerSession:
    """Tests for ContainerSession dataclass."""

    def test_session_creation(self):
        """Test creating a container session."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            user_id="user-789",
        )

        assert session.session_id == "sess-123"
        assert session.tenant_id == "tenant-456"
        assert session.user_id == "user-789"
        assert session.state == SessionState.CREATING
        assert session.execution_count == 0
        assert session.container is None

    def test_session_is_expired(self):
        """Test session expiration check."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            created_at=time.time() - 4000,  # Over 1 hour ago
        )

        assert session.is_expired() is True

    def test_session_not_expired(self):
        """Test session that is not expired."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            created_at=time.time() - 100,  # Just created
        )

        assert session.is_expired() is False

    def test_session_execution_allowed(self):
        """Test checking if execution is allowed."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            state=SessionState.ACTIVE,
        )

        assert session.is_execution_allowed() is True

    def test_session_execution_not_allowed_wrong_state(self):
        """Test execution not allowed in wrong state."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            state=SessionState.SUSPENDED,
        )

        assert session.is_execution_allowed() is False

    def test_session_execution_not_allowed_max_reached(self):
        """Test execution not allowed when max reached."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            state=SessionState.ACTIVE,
            config=SessionConfig(max_executions=10),
            execution_count=10,
        )

        assert session.is_execution_allowed() is False

    def test_session_execution_not_allowed_expired(self):
        """Test execution not allowed when expired."""
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            state=SessionState.ACTIVE,
            created_at=time.time() - 4000,
        )

        assert session.is_execution_allowed() is False

    def test_session_to_dict(self):
        """Test session serialization."""
        container = PooledContainer(
            container_id="c-123",
            container_name="test-container",
        )
        session = ContainerSession(
            session_id="sess-123",
            tenant_id="tenant-456",
            container=container,
            state=SessionState.ACTIVE,
        )

        data = session.to_dict()

        assert data["session_id"] == "sess-123"
        assert data["tenant_id"] == "tenant-456"
        assert data["state"] == "active"
        assert data["container_id"] == "c-123"


class TestSessionState:
    """Tests for SessionState enum."""

    def test_session_states(self):
        """Test session state values."""
        assert SessionState.CREATING.value == "creating"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.SUSPENDED.value == "suspended"
        assert SessionState.TERMINATING.value == "terminating"
        assert SessionState.TERMINATED.value == "terminated"


class TestSessionExecutionResult:
    """Tests for SessionExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = SessionExecutionResult(
            session_id="sess-123",
            execution_id="exec-456",
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="Hello, World!",
            duration_seconds=0.5,
        )

        assert result.session_id == "sess-123"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.stdout == "Hello, World!"
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed execution result."""
        result = SessionExecutionResult(
            session_id="sess-123",
            execution_id="exec-456",
            status=ExecutionStatus.FAILED,
            exit_code=1,
            stderr="Error occurred",
            error_message="Execution failed",
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.error_message == "Execution failed"

    def test_policy_denied_result(self):
        """Test policy denied result."""
        result = SessionExecutionResult(
            session_id="sess-123",
            execution_id="exec-456",
            status=ExecutionStatus.POLICY_DENIED,
            policy_violations=["Network access denied"],
        )

        assert result.status == ExecutionStatus.POLICY_DENIED
        assert len(result.policy_violations) == 1

    def test_result_to_dict(self):
        """Test result serialization."""
        result = SessionExecutionResult(
            session_id="sess-123",
            execution_id="exec-456",
            status=ExecutionStatus.COMPLETED,
        )

        data = result.to_dict()

        assert data["session_id"] == "sess-123"
        assert data["status"] == "completed"


class TestSessionErrors:
    """Tests for session error classes."""

    def test_session_error(self):
        """Test SessionError."""
        error = SessionError("Session error")
        assert str(error) == "Session error"

    def test_session_not_found_error(self):
        """Test SessionNotFoundError."""
        error = SessionNotFoundError("Session not found")
        assert str(error) == "Session not found"
        assert isinstance(error, SessionError)

    def test_session_expired_error(self):
        """Test SessionExpiredError."""
        error = SessionExpiredError("Session expired")
        assert str(error) == "Session expired"
        assert isinstance(error, SessionError)

    def test_execution_denied_error(self):
        """Test ExecutionDeniedError."""
        error = ExecutionDeniedError("Execution denied")
        assert str(error) == "Execution denied"
        assert isinstance(error, SessionError)


class TestSessionContainerManager:
    """Tests for SessionContainerManager class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock container pool."""
        pool = MagicMock(spec=ContainerPool)
        pool.state = PoolState.RUNNING
        pool.stats = MagicMock()
        pool.stats.to_dict.return_value = {}
        pool.acquire = AsyncMock(
            return_value=PooledContainer(
                container_id="c-123",
                container_name="test-container",
            )
        )
        pool.release = AsyncMock()
        pool.destroy = AsyncMock()
        pool.start = AsyncMock()
        return pool

    @pytest.fixture
    def manager(self, mock_pool):
        """Create a session manager with mock pool."""
        manager = SessionContainerManager(pool=mock_pool)
        return manager

    def test_manager_init(self):
        """Test manager initialization."""
        manager = SessionContainerManager()

        assert manager._sessions == {}
        assert manager._tenant_sessions == {}
        assert manager._started is False

    def test_manager_init_with_policy(self):
        """Test manager initialization with custom policy."""
        policy = create_strict_policy()
        manager = SessionContainerManager(default_policy=policy)

        assert manager._default_policy == policy

    @pytest.mark.asyncio
    async def test_manager_start(self, manager, mock_pool):
        """Test starting the manager."""
        mock_pool.state = MagicMock()
        mock_pool.state.value = "stopped"

        await manager.start()

        assert manager._started is True
        mock_pool.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_start_already_started(self, manager):
        """Test starting manager that's already started."""
        manager._started = True

        await manager.start()  # Should return early

    @pytest.mark.asyncio
    async def test_manager_stop(self, manager):
        """Test stopping the manager."""
        manager._started = True
        manager._cleanup_task = asyncio.create_task(asyncio.sleep(100))

        await manager.stop()

        assert manager._started is False

    @pytest.mark.asyncio
    async def test_create_session(self, manager, mock_pool):
        """Test creating a session."""
        session = await manager.create_session(
            tenant_id="tenant-123",
            user_id="user-456",
        )

        assert session.tenant_id == "tenant-123"
        assert session.user_id == "user-456"
        assert session.state == SessionState.ACTIVE
        assert session.container is not None
        mock_pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_with_config(self, manager, mock_pool):
        """Test creating a session with custom config."""
        config = SessionConfig(max_executions=50)

        session = await manager.create_session(
            tenant_id="tenant-123",
            config=config,
        )

        assert session.config.max_executions == 50

    @pytest.mark.asyncio
    async def test_create_session_with_policy(self, manager, mock_pool):
        """Test creating a session with custom policy."""
        policy = create_strict_policy()

        session = await manager.create_session(
            tenant_id="tenant-123",
            policy=policy,
        )

        assert session.policy == policy

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, manager, mock_pool):
        """Test creating a session with metadata."""
        session = await manager.create_session(
            tenant_id="tenant-123",
            metadata={"source": "api", "version": "1.0"},
        )

        assert session.metadata["source"] == "api"

    @pytest.mark.asyncio
    async def test_create_session_registers_tenant(self, manager, mock_pool):
        """Test creating session registers with tenant."""
        session = await manager.create_session(tenant_id="tenant-123")

        assert "tenant-123" in manager._tenant_sessions
        assert session.session_id in manager._tenant_sessions["tenant-123"]

    @pytest.mark.asyncio
    async def test_get_session(self, manager, mock_pool):
        """Test getting a session."""
        created = await manager.create_session(tenant_id="tenant-123")

        session = await manager.get_session(created.session_id)

        assert session == created

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager):
        """Test getting non-existent session."""
        with pytest.raises(SessionNotFoundError):
            await manager.get_session("nonexistent-session")

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager, mock_pool):
        """Test listing sessions."""
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-2")

        sessions = await manager.list_sessions()

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_tenant(self, manager, mock_pool):
        """Test listing sessions filtered by tenant."""
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-2")

        sessions = await manager.list_sessions(tenant_id="tenant-1")

        assert len(sessions) == 1
        assert sessions[0].tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_state(self, manager, mock_pool):
        """Test listing sessions filtered by state."""
        session1 = await manager.create_session(tenant_id="tenant-1")
        session2 = await manager.create_session(tenant_id="tenant-1")
        session2.state = SessionState.SUSPENDED

        sessions = await manager.list_sessions(state=SessionState.ACTIVE)

        assert len(sessions) == 1
        assert sessions[0] == session1

    @pytest.mark.asyncio
    async def test_suspend_session(self, manager, mock_pool):
        """Test suspending a session."""
        session = await manager.create_session(tenant_id="tenant-123")

        await manager.suspend_session(session.session_id)

        assert session.state == SessionState.SUSPENDED

    @pytest.mark.asyncio
    async def test_suspend_non_active_session(self, manager, mock_pool):
        """Test suspending a non-active session."""
        session = await manager.create_session(tenant_id="tenant-123")
        session.state = SessionState.TERMINATED

        await manager.suspend_session(session.session_id)  # Should log warning

        # State should not change
        assert session.state == SessionState.TERMINATED

    @pytest.mark.asyncio
    async def test_resume_session(self, manager, mock_pool):
        """Test resuming a suspended session."""
        session = await manager.create_session(tenant_id="tenant-123")
        session.state = SessionState.SUSPENDED

        await manager.resume_session(session.session_id)

        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_resume_expired_session(self, manager, mock_pool):
        """Test resuming an expired session."""
        session = await manager.create_session(tenant_id="tenant-123")
        session.state = SessionState.SUSPENDED
        session.created_at = time.time() - 4000  # Make it expired

        with pytest.raises(SessionExpiredError):
            await manager.resume_session(session.session_id)

    @pytest.mark.asyncio
    async def test_terminate_session(self, manager, mock_pool):
        """Test terminating a session."""
        session = await manager.create_session(tenant_id="tenant-123")

        await manager.terminate_session(session.session_id)

        assert session.state == SessionState.TERMINATED
        assert session.session_id not in manager._sessions
        mock_pool.release.assert_called()

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_session(self, manager):
        """Test terminating a non-existent session."""
        # Should not raise
        await manager.terminate_session("nonexistent-session")


class TestSessionExecution:
    """Tests for session execution functionality."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock container pool."""
        pool = MagicMock(spec=ContainerPool)
        pool.state = PoolState.RUNNING
        pool.acquire = AsyncMock(
            return_value=PooledContainer(
                container_id="c-123",
                container_name="test-container",
            )
        )
        pool.release = AsyncMock()
        return pool

    @pytest.fixture
    def manager(self, mock_pool):
        """Create a session manager with mock pool."""
        return SessionContainerManager(pool=mock_pool)

    @pytest.mark.asyncio
    async def test_execute_policy_denied(self, manager, mock_pool):
        """Test execution denied by policy."""
        # Create strict policy that denies python
        policy = ToolPolicy(name="deny-python")
        policy.add_tool_denylist([r"^python3$"], reason="Python denied")

        session = await manager.create_session(
            tenant_id="tenant-123",
            policy=policy,
        )

        result = await manager.execute(
            session_id=session.session_id,
            code='print("hello")',
            language="python",
        )

        assert result.status == ExecutionStatus.POLICY_DENIED
        assert len(result.policy_violations) > 0

    @pytest.mark.asyncio
    async def test_execute_not_allowed_state(self, manager, mock_pool):
        """Test execution not allowed in wrong state."""
        session = await manager.create_session(tenant_id="tenant-123")
        session.state = SessionState.SUSPENDED

        with pytest.raises(ExecutionDeniedError):
            await manager.execute(
                session_id=session.session_id,
                code='print("hello")',
                language="python",
            )

    @pytest.mark.asyncio
    async def test_execute_expired_session(self, manager, mock_pool):
        """Test execution on expired session."""
        session = await manager.create_session(tenant_id="tenant-123")
        session.created_at = time.time() - 4000  # Make it expired

        with pytest.raises(SessionExpiredError):
            await manager.execute(
                session_id=session.session_id,
                code='print("hello")',
                language="python",
            )

    @pytest.mark.asyncio
    async def test_execute_updates_stats(self, manager, mock_pool):
        """Test execution updates session stats."""
        session = await manager.create_session(tenant_id="tenant-123")
        original_count = session.execution_count

        with patch.object(manager, "_execute_in_container", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SessionExecutionResult(
                session_id=session.session_id,
                execution_id="exec-123",
                status=ExecutionStatus.COMPLETED,
            )

            await manager.execute(
                session_id=session.session_id,
                code='print("hello")',
                language="python",
            )

            assert session.execution_count == original_count + 1


class TestSessionTenantIsolation:
    """Tests for tenant isolation in sessions."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock container pool."""
        pool = MagicMock(spec=ContainerPool)
        pool.state = PoolState.RUNNING
        pool.acquire = AsyncMock(
            return_value=PooledContainer(
                container_id="c-123",
                container_name="test-container",
            )
        )
        pool.release = AsyncMock()
        return pool

    @pytest.fixture
    def manager(self, mock_pool):
        """Create a session manager with mock pool."""
        return SessionContainerManager(pool=mock_pool)

    @pytest.mark.asyncio
    async def test_get_tenant_sessions(self, manager, mock_pool):
        """Test getting sessions for a tenant."""
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-2")

        sessions = await manager.get_tenant_sessions("tenant-1")

        assert len(sessions) == 2
        assert all(s.tenant_id == "tenant-1" for s in sessions)

    @pytest.mark.asyncio
    async def test_terminate_tenant_sessions(self, manager, mock_pool):
        """Test terminating all sessions for a tenant."""
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-2")

        count = await manager.terminate_tenant_sessions("tenant-1")

        assert count == 2

        # Verify tenant-1 sessions are gone
        tenant1_sessions = await manager.get_tenant_sessions("tenant-1")
        assert len(tenant1_sessions) == 0

        # Verify tenant-2 sessions remain
        tenant2_sessions = await manager.get_tenant_sessions("tenant-2")
        assert len(tenant2_sessions) == 1


class TestSessionStatistics:
    """Tests for session statistics collection."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock container pool."""
        pool = MagicMock(spec=ContainerPool)
        pool.state = PoolState.RUNNING
        pool.stats = MagicMock()
        pool.stats.to_dict.return_value = {"total_containers": 5}
        pool.acquire = AsyncMock(
            return_value=PooledContainer(
                container_id="c-123",
                container_name="test-container",
            )
        )
        return pool

    @pytest.fixture
    def manager(self, mock_pool):
        """Create a session manager with mock pool."""
        return SessionContainerManager(pool=mock_pool)

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, mock_pool):
        """Test getting session manager statistics."""
        session1 = await manager.create_session(tenant_id="tenant-1")
        session2 = await manager.create_session(tenant_id="tenant-2")
        session1.execution_count = 5
        session1.total_execution_time_seconds = 10.0
        session2.execution_count = 3
        session2.total_execution_time_seconds = 5.0

        stats = await manager.get_stats()

        assert stats["total_sessions"] == 2
        assert stats["total_executions"] == 8
        assert stats["total_execution_time_seconds"] == 15.0
        assert "by_state" in stats
        assert "by_tenant" in stats
        assert stats["pool_stats"]["total_containers"] == 5

    @pytest.mark.asyncio
    async def test_stats_by_tenant(self, manager, mock_pool):
        """Test statistics by tenant."""
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-1")
        await manager.create_session(tenant_id="tenant-2")

        stats = await manager.get_stats()

        assert stats["by_tenant"]["tenant-1"] == 2
        assert stats["by_tenant"]["tenant-2"] == 1

    @pytest.mark.asyncio
    async def test_stats_by_state(self, manager, mock_pool):
        """Test statistics by state."""
        session1 = await manager.create_session(tenant_id="tenant-1")
        session2 = await manager.create_session(tenant_id="tenant-1")
        session2.state = SessionState.SUSPENDED

        stats = await manager.get_stats()

        assert stats["by_state"]["active"] == 1
        assert stats["by_state"]["suspended"] == 1


class TestGlobalSessionManager:
    """Tests for global session manager instance."""

    def test_get_session_manager_creates_instance(self):
        """Test getting global manager creates instance."""
        set_session_manager(None)

        manager = get_session_manager()

        assert manager is not None
        assert isinstance(manager, SessionContainerManager)

    def test_get_session_manager_returns_same(self):
        """Test getting global manager returns same instance."""
        set_session_manager(None)

        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_set_session_manager(self):
        """Test setting global manager."""
        custom_manager = SessionContainerManager()
        set_session_manager(custom_manager)

        manager = get_session_manager()

        assert manager is custom_manager

    def test_set_session_manager_none(self):
        """Test resetting global manager."""
        set_session_manager(None)

        # Getting manager should create new one
        manager = get_session_manager()
        assert manager is not None


class TestSessionLanguageSupport:
    """Tests for language support in sessions."""

    @pytest.fixture
    def manager(self):
        """Create a session manager."""
        return SessionContainerManager()

    def test_language_to_tool_python(self, manager):
        """Test Python language mapping."""
        assert manager._language_to_tool("python") == "python3"
        assert manager._language_to_tool("python3") == "python3"

    def test_language_to_tool_javascript(self, manager):
        """Test JavaScript language mapping."""
        assert manager._language_to_tool("javascript") == "node"
        assert manager._language_to_tool("node") == "node"

    def test_language_to_tool_bash(self, manager):
        """Test bash language mapping."""
        assert manager._language_to_tool("bash") == "bash"
        assert manager._language_to_tool("shell") == "sh"
        assert manager._language_to_tool("sh") == "sh"

    def test_language_to_tool_unknown(self, manager):
        """Test unknown language mapping."""
        assert manager._language_to_tool("rust") == "rust"

    def test_build_execution_command_python(self, manager):
        """Test building Python execution command."""
        cmd = manager._build_execution_command("python")

        assert cmd == ["python3", "-c"]

    def test_build_execution_command_javascript(self, manager):
        """Test building JavaScript execution command."""
        cmd = manager._build_execution_command("javascript")

        assert cmd == ["node", "-e"]

    def test_build_execution_command_bash(self, manager):
        """Test building bash execution command."""
        cmd = manager._build_execution_command("bash")

        assert cmd == ["bash", "-c"]

    def test_build_execution_command_unsupported(self, manager):
        """Test building command for unsupported language."""
        cmd = manager._build_execution_command("rust")

        assert cmd is None
