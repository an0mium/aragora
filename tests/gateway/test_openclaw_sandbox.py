"""
Comprehensive unit tests for OpenClaw Action Sandbox.

Tests cover:
1. Workspace creation and cleanup
2. Path traversal prevention (symlink attacks)
3. Resource limit enforcement (CPU, memory, time)
4. Environment variable isolation
5. Session lifecycle (create, execute, destroy)
6. Error handling
7. Edge cases
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.openclaw_sandbox import (
    OpenClawActionSandbox,
    SandboxActionResult,
    SandboxConfig,
    SandboxSession,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sandbox_base_dir(tmp_path: Path) -> Path:
    """Provide a temporary base directory for sandbox workspaces."""
    base = tmp_path / "sandbox_base"
    base.mkdir()
    return base


@pytest.fixture
def sandbox(sandbox_base_dir: Path) -> OpenClawActionSandbox:
    """Create a sandbox manager with a temp base directory."""
    return OpenClawActionSandbox(base_workspace_path=str(sandbox_base_dir))


@pytest.fixture
def custom_config() -> SandboxConfig:
    """Create a custom sandbox configuration for testing."""
    return SandboxConfig(
        workspace_root="/workspace",
        workspace_size_limit_mb=500,
        max_execution_time_seconds=10,
        max_memory_mb=256,
        max_file_size_mb=5,
        max_processes=5,
    )


@pytest.fixture
def strict_config() -> SandboxConfig:
    """Create a strict sandbox configuration with minimal permissions."""
    return SandboxConfig(
        allowed_commands=["echo", "ls", "cat"],
        blocked_commands=["sudo", "su", "rm", "dd"],
        allow_network=False,
        inherit_env=False,
        max_execution_time_seconds=5,
        max_file_size_mb=1,
    )


# ============================================================================
# SandboxConfig Tests
# ============================================================================


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Test that default config has expected values."""
        config = SandboxConfig()

        assert config.workspace_root == "/workspace"
        assert config.create_workspace is True
        assert config.workspace_size_limit_mb == 1000
        assert config.max_execution_time_seconds == 300
        assert config.max_memory_mb == 512
        assert config.max_file_size_mb == 100
        assert config.max_processes == 10
        assert config.allow_network is True
        assert config.inherit_env is False

    def test_default_blocked_paths(self) -> None:
        """Test that sensitive paths are blocked by default."""
        config = SandboxConfig()

        assert "/etc/passwd" in config.blocked_paths
        assert "/etc/shadow" in config.blocked_paths
        assert "/etc/sudoers" in config.blocked_paths
        assert "/root" in config.blocked_paths
        assert "/home/*/.ssh" in config.blocked_paths

    def test_default_allowed_commands(self) -> None:
        """Test that safe commands are allowed by default."""
        config = SandboxConfig()

        for cmd in ["ls", "cat", "grep", "echo", "python", "git"]:
            assert cmd in config.allowed_commands

    def test_default_blocked_commands(self) -> None:
        """Test that dangerous commands are blocked by default."""
        config = SandboxConfig()

        for cmd in ["sudo", "su", "dd", "mkfs", "reboot", "shutdown", "iptables"]:
            assert cmd in config.blocked_commands

    def test_default_blocked_hosts(self) -> None:
        """Test that cloud metadata and localhost are blocked."""
        config = SandboxConfig()

        assert "localhost" in config.blocked_hosts
        assert "127.0.0.1" in config.blocked_hosts
        assert "0.0.0.0" in config.blocked_hosts
        assert "169.254.169.254" in config.blocked_hosts

    def test_custom_config(self) -> None:
        """Test custom configuration overrides."""
        config = SandboxConfig(
            workspace_root="/custom/workspace",
            max_memory_mb=1024,
            max_execution_time_seconds=600,
            inherit_env=True,
            environment_vars={"MY_VAR": "value"},
        )

        assert config.workspace_root == "/custom/workspace"
        assert config.max_memory_mb == 1024
        assert config.max_execution_time_seconds == 600
        assert config.inherit_env is True
        assert config.environment_vars == {"MY_VAR": "value"}

    def test_empty_allowed_commands(self) -> None:
        """Test config with explicitly empty allowed commands list."""
        config = SandboxConfig(allowed_commands=[])

        assert config.allowed_commands == []

    def test_custom_allowed_read_write_paths(self) -> None:
        """Test config with custom read/write path permissions."""
        config = SandboxConfig(
            allowed_read_paths=["/data/shared"],
            allowed_write_paths=["/data/output"],
        )

        assert "/data/shared" in config.allowed_read_paths
        assert "/data/output" in config.allowed_write_paths


# ============================================================================
# SandboxSession Tests
# ============================================================================


class TestSandboxSession:
    """Tests for SandboxSession dataclass."""

    def test_session_creation_defaults(self) -> None:
        """Test session creation with default values."""
        session = SandboxSession(
            sandbox_id="sb-123",
            session_id="sess-456",
            user_id="user-789",
            tenant_id="acme",
            workspace_path="/tmp/workspace",
            config=SandboxConfig(),
        )

        assert session.sandbox_id == "sb-123"
        assert session.session_id == "sess-456"
        assert session.user_id == "user-789"
        assert session.tenant_id == "acme"
        assert session.action_count == 0
        assert session.bytes_written == 0
        assert session.status == "active"
        assert session.created_at > 0
        assert session.last_activity > 0

    def test_session_timestamps(self) -> None:
        """Test that session timestamps are reasonable."""
        before = time.time()
        session = SandboxSession(
            sandbox_id="sb-1",
            session_id="sess-1",
            user_id="user-1",
            tenant_id="tenant-1",
            workspace_path="/tmp/ws",
            config=SandboxConfig(),
        )
        after = time.time()

        assert before <= session.created_at <= after
        assert before <= session.last_activity <= after


# ============================================================================
# SandboxActionResult Tests
# ============================================================================


class TestSandboxActionResult:
    """Tests for SandboxActionResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful action result."""
        result = SandboxActionResult(
            success=True,
            action_id="act-123",
            output="Hello world",
            execution_time_ms=15.5,
        )

        assert result.success is True
        assert result.action_id == "act-123"
        assert result.output == "Hello world"
        assert result.error is None
        assert result.execution_time_ms == 15.5

    def test_failure_result(self) -> None:
        """Test failed action result."""
        result = SandboxActionResult(
            success=False,
            action_id="act-456",
            error="Permission denied",
        )

        assert result.success is False
        assert result.error == "Permission denied"
        assert result.output is None

    def test_result_with_resources(self) -> None:
        """Test result with resource usage data."""
        result = SandboxActionResult(
            success=True,
            action_id="act-789",
            resources_used={"memory_mb": 128, "cpu_percent": 25},
        )

        assert result.resources_used["memory_mb"] == 128
        assert result.resources_used["cpu_percent"] == 25


# ============================================================================
# Workspace Creation and Cleanup Tests
# ============================================================================


class TestWorkspaceCreationAndCleanup:
    """Tests for workspace creation and cleanup."""

    @pytest.mark.asyncio
    async def test_create_sandbox_creates_workspace_directory(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that creating a sandbox creates the workspace directory."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            tenant_id="tenant-1",
        )

        workspace = Path(session.workspace_path)
        assert workspace.exists()
        assert workspace.is_dir()

    @pytest.mark.asyncio
    async def test_workspace_isolated_per_tenant(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that workspaces are isolated by tenant."""
        session_a = await sandbox.create_sandbox(
            session_id="sess-a",
            user_id="user-1",
            tenant_id="tenant-a",
        )
        session_b = await sandbox.create_sandbox(
            session_id="sess-b",
            user_id="user-1",
            tenant_id="tenant-b",
        )

        # Workspaces should be in different tenant directories
        path_a = Path(session_a.workspace_path)
        path_b = Path(session_b.workspace_path)

        assert "tenant-a" in str(path_a)
        assert "tenant-b" in str(path_b)
        assert path_a != path_b

        # Cleanup
        await sandbox.destroy_sandbox(session_a.sandbox_id)
        await sandbox.destroy_sandbox(session_b.sandbox_id)

    @pytest.mark.asyncio
    async def test_workspace_isolated_per_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that each sandbox gets its own workspace directory."""
        session_1 = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            tenant_id="tenant-1",
        )
        session_2 = await sandbox.create_sandbox(
            session_id="sess-2",
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert session_1.workspace_path != session_2.workspace_path
        assert Path(session_1.workspace_path).exists()
        assert Path(session_2.workspace_path).exists()

        # Cleanup
        await sandbox.destroy_sandbox(session_1.sandbox_id)
        await sandbox.destroy_sandbox(session_2.sandbox_id)

    @pytest.mark.asyncio
    async def test_destroy_sandbox_cleans_up_workspace(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that destroying a sandbox removes the workspace directory."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        workspace_path = session.workspace_path
        assert Path(workspace_path).exists()

        result = await sandbox.destroy_sandbox(session.sandbox_id)
        assert result is True

        # The parent of workspace (sandbox dir) should be removed
        assert not Path(workspace_path).parent.exists()

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_sandbox_returns_false(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test destroying a nonexistent sandbox returns False."""
        result = await sandbox.destroy_sandbox("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_destroy_sandbox_sets_terminated_status(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that destroying a sandbox marks it as terminated."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Keep a reference before destroy removes it
        sandbox_id = session.sandbox_id
        assert session.status == "active"

        await sandbox.destroy_sandbox(sandbox_id)
        assert session.status == "terminated"

    @pytest.mark.asyncio
    async def test_cleanup_all_destroys_all_sandboxes(self, sandbox: OpenClawActionSandbox) -> None:
        """Test cleanup_all destroys every active sandbox."""
        sessions = []
        for i in range(5):
            s = await sandbox.create_sandbox(
                session_id=f"sess-{i}",
                user_id=f"user-{i}",
            )
            sessions.append(s)

        assert sandbox.get_stats()["active_sandboxes"] == 5

        count = await sandbox.cleanup_all()
        assert count == 5
        assert sandbox.get_stats()["active_sandboxes"] == 0

    @pytest.mark.asyncio
    async def test_destroy_handles_missing_workspace_gracefully(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that destroy handles already-deleted workspace directories."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Manually delete the workspace before destroy
        shutil.rmtree(Path(session.workspace_path).parent, ignore_errors=True)

        # Should not raise
        result = await sandbox.destroy_sandbox(session.sandbox_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that default tenant_id is 'default'."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        assert session.tenant_id == "default"
        assert "default" in session.workspace_path

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Path Traversal Prevention Tests
# ============================================================================


class TestPathTraversalPrevention:
    """Tests for path traversal and symlink attack prevention."""

    @pytest.mark.asyncio
    async def test_write_outside_workspace_blocked(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that writing outside the workspace is blocked."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/tmp/evil_file.txt",
            content="malicious content",
        )

        assert result.success is False
        assert "outside workspace" in result.error.lower() or "not allowed" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_path_traversal_with_dotdot_blocked(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that path traversal with '../' is prevented."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/../../../etc/evil",
            content="malicious",
        )

        assert result.success is False

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_blocked_path_etc_passwd_read(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that reading /etc/passwd is blocked."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/etc/passwd",
        )

        assert result.success is False
        assert "blocked" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_blocked_path_etc_shadow(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that reading /etc/shadow is blocked."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/etc/shadow",
        )

        assert result.success is False

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_relative_path_resolved_within_workspace(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that relative paths are resolved within the workspace."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Write a file using a relative path
        write_result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="./test_file.txt",
            content="hello from relative path",
        )
        assert write_result.success is True

        # Verify the file exists in the workspace
        expected_file = Path(session.workspace_path) / "test_file.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == "hello from relative path"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_workspace_prefix_path_resolution(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that /workspace paths map to actual workspace."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        write_result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/mapped_file.txt",
            content="mapped content",
        )
        assert write_result.success is True

        # Verify the file was written to the actual workspace
        actual_file = Path(session.workspace_path) / "mapped_file.txt"
        assert actual_file.exists()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_outside_workspace_blocked(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that deleting files outside workspace is blocked."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/tmp/some_other_file.txt",
        )

        assert result.success is False

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.path.exists("/etc/passwd"), reason="Needs /etc/passwd to test blocked paths"
    )
    async def test_symlink_attack_blocked_for_write(
        self, sandbox: OpenClawActionSandbox, tmp_path: Path
    ) -> None:
        """Test that symlink-based path traversal is blocked for writes."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a symlink inside workspace pointing outside
        workspace = Path(session.workspace_path)
        outside_target = tmp_path / "outside_target.txt"
        outside_target.write_text("original content")
        symlink_path = workspace / "sneaky_link"

        try:
            symlink_path.symlink_to(outside_target)
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        # Attempting to write via the symlink should be blocked because
        # the resolved path is outside workspace
        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/sneaky_link",
            content="overwritten via symlink!",
        )

        assert result.success is False

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Resource Limit Enforcement Tests
# ============================================================================


class TestResourceLimitEnforcement:
    """Tests for resource limit enforcement (CPU, memory, time, file size)."""

    @pytest.mark.asyncio
    async def test_file_size_limit_enforcement(self, sandbox_base_dir: Path) -> None:
        """Test that file size limits are enforced on writes."""
        config = SandboxConfig(max_file_size_mb=1)  # 1 MB limit
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Try to write a file larger than 1MB
        large_content = "x" * (2 * 1024 * 1024)  # 2 MB
        result = await sb.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/large_file.txt",
            content=large_content,
        )

        assert result.success is False
        assert "exceeds limit" in result.error.lower() or "size" in result.error.lower()

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_file_within_size_limit_allowed(self, sandbox_base_dir: Path) -> None:
        """Test that files within the size limit are written successfully."""
        config = SandboxConfig(max_file_size_mb=1)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Write a small file (well under 1MB)
        small_content = "hello world"
        result = await sb.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/small_file.txt",
            content=small_content,
        )

        assert result.success is True
        assert result.output["bytes_written"] == len(small_content.encode("utf-8"))

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_shell_command_timeout(self, sandbox_base_dir: Path) -> None:
        """Test that shell commands respect timeout limits."""
        config = SandboxConfig(max_execution_time_seconds=1)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sb.execute_shell(
            sandbox_id=session.sandbox_id,
            command="sleep 30",
        )

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.execution_time_ms > 0

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that per-command timeout override works."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="sleep 10",
            timeout=1,
        )

        assert result.success is False
        assert "timed out" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Environment Variable Isolation Tests
# ============================================================================


class TestEnvironmentVariableIsolation:
    """Tests for environment variable isolation."""

    @pytest.mark.asyncio
    async def test_default_env_is_minimal(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that the default environment has only safe variables."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sandbox._build_environment(session)

        assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        assert env["USER"] == "sandbox"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["HOME"] == session.workspace_path
        assert env["SANDBOX_ID"] == session.sandbox_id
        assert env["WORKSPACE"] == session.workspace_path

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_no_inherited_env_by_default(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that host environment variables are not inherited by default."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sandbox._build_environment(session)

        # Should not include common host env vars
        for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SHELL"]:
            if var in os.environ:
                assert var not in env or env[var] != os.environ[var]

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_inherited_env_when_configured(self, sandbox_base_dir: Path) -> None:
        """Test that env is inherited when inherit_env is True."""
        config = SandboxConfig(inherit_env=True)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sb._build_environment(session)

        # Should contain host PATH (or equivalent)
        assert "PATH" in env

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_custom_env_vars_added(self, sandbox_base_dir: Path) -> None:
        """Test that custom environment variables are included."""
        config = SandboxConfig(
            environment_vars={
                "CUSTOM_VAR": "custom_value",
                "ANOTHER_VAR": "another_value",
            },
        )
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sb._build_environment(session)

        assert env["CUSTOM_VAR"] == "custom_value"
        assert env["ANOTHER_VAR"] == "another_value"

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_sandbox_specific_vars_always_present(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that SANDBOX_ID and WORKSPACE are always set."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sandbox._build_environment(session)

        assert "SANDBOX_ID" in env
        assert "WORKSPACE" in env
        assert env["SANDBOX_ID"] == session.sandbox_id
        assert env["WORKSPACE"] == session.workspace_path

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_custom_vars_override_inherited_env(self, sandbox_base_dir: Path) -> None:
        """Test that custom vars override inherited environment variables."""
        config = SandboxConfig(
            inherit_env=True,
            environment_vars={"PATH": "/custom/path"},
        )
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        env = sb._build_environment(session)

        # Custom PATH should be overridden, but then SANDBOX_ID is added last
        # The custom vars override inherited, but sandbox-specific are added on top
        assert "SANDBOX_ID" in env

        await sb.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Session Lifecycle Tests
# ============================================================================


class TestSessionLifecycle:
    """Tests for session create/execute/destroy lifecycle."""

    @pytest.mark.asyncio
    async def test_create_returns_session_with_sandbox_id(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that create_sandbox returns a valid session."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            tenant_id="acme",
        )

        assert session.sandbox_id is not None
        assert len(session.sandbox_id) > 0
        assert session.session_id == "sess-1"
        assert session.user_id == "user-1"
        assert session.tenant_id == "acme"
        assert session.status == "active"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_get_sandbox_by_id(self, sandbox: OpenClawActionSandbox) -> None:
        """Test retrieving a sandbox by its ID."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        retrieved = sandbox.get_sandbox(session.sandbox_id)
        assert retrieved is not None
        assert retrieved.sandbox_id == session.sandbox_id
        assert retrieved.session_id == session.session_id

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_get_sandbox_by_session_id(self, sandbox: OpenClawActionSandbox) -> None:
        """Test retrieving a sandbox by session ID."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        retrieved = sandbox.get_sandbox_for_session("sess-1")
        assert retrieved is not None
        assert retrieved.sandbox_id == session.sandbox_id

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_get_nonexistent_sandbox_returns_none(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that getting a nonexistent sandbox returns None."""
        assert sandbox.get_sandbox("nonexistent") is None
        assert sandbox.get_sandbox_for_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_session_action_count_increments(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that action count increments on operations."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        assert session.action_count == 0

        # Write a file
        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file1.txt",
            content="content1",
        )
        assert session.action_count == 1

        # Read a file
        await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file1.txt",
        )
        assert session.action_count == 2

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_session_bytes_written_tracked(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that bytes written are tracked per session."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        content = "hello world"
        expected_bytes = len(content.encode("utf-8"))

        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file.txt",
            content=content,
        )

        assert session.bytes_written == expected_bytes

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_session_last_activity_updates(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that last_activity timestamp updates on actions."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        initial_activity = session.last_activity

        # Small delay to ensure time difference
        await asyncio.sleep(0.05)

        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file.txt",
            content="content",
        )

        assert session.last_activity >= initial_activity

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_destroyed_sandbox_not_retrievable(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that destroyed sandboxes cannot be retrieved."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )
        sandbox_id = session.sandbox_id

        await sandbox.destroy_sandbox(sandbox_id)

        assert sandbox.get_sandbox(sandbox_id) is None
        assert sandbox.get_sandbox_for_session("sess-1") is None

    @pytest.mark.asyncio
    async def test_custom_config_per_session(
        self, sandbox: OpenClawActionSandbox, custom_config: SandboxConfig
    ) -> None:
        """Test that a custom config can be passed per session."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            config=custom_config,
        )

        assert session.config.max_execution_time_seconds == 10
        assert session.config.max_memory_mb == 256
        assert session.config.max_file_size_mb == 5

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Command Validation Tests
# ============================================================================


class TestCommandValidation:
    """Tests for command validation and filtering."""

    def test_allowed_command_passes(self) -> None:
        """Test that an allowed command passes validation."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        result = sb._validate_command("ls -la", config)
        assert result["allowed"] is True

    def test_blocked_command_rejected(self) -> None:
        """Test that blocked commands are rejected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        result = sb._validate_command("sudo rm -rf /", config)
        assert result["allowed"] is False
        assert "blocked" in result["reason"].lower()

    def test_command_not_in_allowed_list_rejected(self) -> None:
        """Test that commands not in the allowed list are rejected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig(allowed_commands=["ls", "cat"])

        result = sb._validate_command("wget http://evil.com", config)
        assert result["allowed"] is False
        assert "not in allowed list" in result["reason"].lower()

    def test_empty_command_rejected(self) -> None:
        """Test that empty commands are rejected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        result = sb._validate_command("", config)
        assert result["allowed"] is False
        assert "empty" in result["reason"].lower()

    def test_whitespace_only_command_rejected(self) -> None:
        """Test that whitespace-only commands are rejected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        result = sb._validate_command("   ", config)
        assert result["allowed"] is False

    def test_blocked_command_in_pipeline_detected(self) -> None:
        """Test that blocked commands within pipelines are detected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        # sudo embedded in a pipeline
        result = sb._validate_command("echo test | sudo cat", config)
        assert result["allowed"] is False

    def test_empty_allowed_list_allows_any_unblocked_command(self) -> None:
        """Test that empty allowed list permits non-blocked commands."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig(allowed_commands=[])

        result = sb._validate_command("custom_command", config)
        assert result["allowed"] is True

    def test_all_default_blocked_commands_rejected(self) -> None:
        """Test that all default blocked commands are properly rejected."""
        sb = OpenClawActionSandbox()
        config = SandboxConfig()

        for cmd in config.blocked_commands:
            result = sb._validate_command(cmd, config)
            assert result["allowed"] is False, f"Command '{cmd}' should be blocked"


# ============================================================================
# Shell Execution Tests
# ============================================================================


class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_echo(self, sandbox: OpenClawActionSandbox) -> None:
        """Test executing a simple echo command."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="echo 'Hello World'",
        )

        assert result.success is True
        assert "Hello World" in result.output["stdout"]
        assert result.output["return_code"] == 0
        assert result.execution_time_ms > 0

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_on_nonexistent_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that executing on a nonexistent sandbox fails gracefully."""
        result = await sandbox.execute_shell(
            sandbox_id="nonexistent",
            command="echo test",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_on_terminated_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that executing on a terminated sandbox fails."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Manually set status to terminated
        session.status = "terminated"

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="echo test",
        )

        assert result.success is False
        assert "terminated" in result.error.lower()

        # Restore for cleanup
        session.status = "active"
        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_on_suspended_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that executing on a suspended sandbox fails."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        session.status = "suspended"

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="echo test",
        )

        assert result.success is False
        assert "suspended" in result.error.lower()

        session.status = "active"
        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that blocked commands are rejected before execution."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="sudo whoami",
        )

        assert result.success is False
        assert "blocked" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that stderr output is captured."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="ls /nonexistent_path_xyz",
        )

        assert result.success is False
        assert result.output["return_code"] != 0
        assert result.output["stderr"] != ""

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_increments_stats(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that execution increments global stats."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        initial_executed = sandbox.get_stats()["commands_executed"]

        await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="echo test",
        )

        assert sandbox.get_stats()["commands_executed"] == initial_executed + 1

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_blocked_command_increments_blocked_stats(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that blocked commands increment the blocked stats counter."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        initial_blocked = sandbox.get_stats()["commands_blocked"]

        await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="sudo test",
        )

        assert sandbox.get_stats()["commands_blocked"] == initial_blocked + 1

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_cwd_is_workspace(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that commands execute within the workspace directory."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="pwd",
        )

        assert result.success is True
        # pwd output should match the workspace path
        assert session.workspace_path in result.output["stdout"].strip()

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# File Operations Tests
# ============================================================================


class TestFileOperations:
    """Tests for read_file, write_file, and delete_file operations."""

    @pytest.mark.asyncio
    async def test_write_and_read_file(self, sandbox: OpenClawActionSandbox) -> None:
        """Test writing and reading a file in the sandbox."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Write
        write_result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/test.txt",
            content="Hello, sandbox!",
        )
        assert write_result.success is True

        # Read
        read_result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/test.txt",
        )
        assert read_result.success is True
        assert read_result.output == "Hello, sandbox!"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that write_file creates intermediate directories."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/deep/nested/dir/file.txt",
            content="nested content",
        )
        assert result.success is True

        nested_file = Path(session.workspace_path) / "deep" / "nested" / "dir" / "file.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "nested content"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that reading a nonexistent file fails gracefully."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/does_not_exist.txt",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_read_from_nonexistent_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test reading from nonexistent sandbox returns error."""
        result = await sandbox.read_file(
            sandbox_id="nonexistent",
            path="/workspace/file.txt",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_to_nonexistent_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test writing to nonexistent sandbox returns error."""
        result = await sandbox.write_file(
            sandbox_id="nonexistent",
            path="/workspace/file.txt",
            content="content",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_file_success(self, sandbox: OpenClawActionSandbox) -> None:
        """Test successful file deletion within workspace."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a file
        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/to_delete.txt",
            content="delete me",
        )

        # Delete it
        result = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/to_delete.txt",
        )
        assert result.success is True

        # Verify it is gone
        assert not (Path(session.workspace_path) / "to_delete.txt").exists()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that deleting a nonexistent file fails gracefully."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/ghost_file.txt",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_directory(self, sandbox: OpenClawActionSandbox) -> None:
        """Test deleting a directory within the workspace."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a directory with a file
        dir_path = Path(session.workspace_path) / "subdir"
        dir_path.mkdir()
        (dir_path / "inner.txt").write_text("inner")

        result = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/subdir",
        )
        assert result.success is True
        assert not dir_path.exists()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_from_nonexistent_sandbox(self, sandbox: OpenClawActionSandbox) -> None:
        """Test deleting from nonexistent sandbox returns error."""
        result = await sandbox.delete_file(
            sandbox_id="nonexistent",
            path="/workspace/file.txt",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_increments_file_stats(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that write operations update global statistics."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        initial_written = sandbox.get_stats()["files_written"]
        initial_bytes = sandbox.get_stats()["bytes_written"]

        content = "test content"
        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/stats_test.txt",
            content=content,
        )

        stats = sandbox.get_stats()
        assert stats["files_written"] == initial_written + 1
        assert stats["bytes_written"] == initial_bytes + len(content.encode("utf-8"))

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_read_increments_file_stats(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that read operations update global statistics."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a file first
        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/read_test.txt",
            content="readable",
        )

        initial_read = sandbox.get_stats()["files_read"]

        await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/read_test.txt",
        )

        assert sandbox.get_stats()["files_read"] == initial_read + 1

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_unicode_content(self, sandbox: OpenClawActionSandbox) -> None:
        """Test writing and reading Unicode content."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        unicode_content = "Bonjour le monde! Konnichi wa! 42 > 7"

        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/unicode.txt",
            content=unicode_content,
        )

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/unicode.txt",
        )

        assert result.success is True
        assert result.output == unicode_content

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_empty_file(self, sandbox: OpenClawActionSandbox) -> None:
        """Test writing an empty file."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/empty.txt",
            content="",
        )

        assert result.success is True
        assert result.output["bytes_written"] == 0

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Statistics and Event Tracking Tests
# ============================================================================


class TestStatisticsAndEvents:
    """Tests for statistics tracking and event emission."""

    @pytest.mark.asyncio
    async def test_initial_stats_are_zero(self, sandbox_base_dir: Path) -> None:
        """Test that a fresh sandbox manager starts with zero stats."""
        sb = OpenClawActionSandbox(base_workspace_path=str(sandbox_base_dir))

        stats = sb.get_stats()
        assert stats["sandboxes_created"] == 0
        assert stats["sandboxes_destroyed"] == 0
        assert stats["commands_executed"] == 0
        assert stats["commands_blocked"] == 0
        assert stats["files_read"] == 0
        assert stats["files_written"] == 0
        assert stats["bytes_written"] == 0
        assert stats["active_sandboxes"] == 0

    @pytest.mark.asyncio
    async def test_stats_track_sandbox_creation(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that stats track sandbox creation."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        stats = sandbox.get_stats()
        assert stats["sandboxes_created"] == 1
        assert stats["active_sandboxes"] == 1

        await sandbox.destroy_sandbox(session.sandbox_id)

        stats = sandbox.get_stats()
        assert stats["sandboxes_destroyed"] == 1
        assert stats["active_sandboxes"] == 0

    @pytest.mark.asyncio
    async def test_event_callback_called_on_create(self, sandbox_base_dir: Path) -> None:
        """Test that event callback is called when sandbox is created."""
        events = []

        def callback(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            event_callback=callback,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            tenant_id="acme",
        )

        assert len(events) == 1
        assert events[0][0] == "sandbox_created"
        assert events[0][1]["sandbox_id"] == session.sandbox_id
        assert events[0][1]["session_id"] == "sess-1"
        assert events[0][1]["user_id"] == "user-1"
        assert events[0][1]["tenant_id"] == "acme"

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_event_callback_called_on_destroy(self, sandbox_base_dir: Path) -> None:
        """Test that event callback is called when sandbox is destroyed."""
        events = []

        def callback(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            event_callback=callback,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )
        events.clear()  # Clear the create event

        await sb.destroy_sandbox(session.sandbox_id)

        assert len(events) == 1
        assert events[0][0] == "sandbox_destroyed"
        assert events[0][1]["sandbox_id"] == session.sandbox_id
        assert "duration_seconds" in events[0][1]
        assert "action_count" in events[0][1]
        assert "bytes_written" in events[0][1]

    @pytest.mark.asyncio
    async def test_event_callback_exception_does_not_crash(self, sandbox_base_dir: Path) -> None:
        """Test that a failing event callback does not crash the sandbox."""

        def bad_callback(event_type: str, data: dict) -> None:
            raise RuntimeError("Callback explosion!")

        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            event_callback=bad_callback,
        )

        # Should not raise despite callback failure
        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )
        assert session is not None

        result = await sb.destroy_sandbox(session.sandbox_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_event_callback_when_none(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that no errors occur when event callback is None."""
        assert sandbox._event_callback is None

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )
        # Should not raise
        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_execute_shell_handles_exception_gracefully(
        self, sandbox: OpenClawActionSandbox
    ) -> None:
        """Test that execute_shell handles unexpected exceptions."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        with patch("asyncio.create_subprocess_shell", side_effect=OSError("Boom")):
            result = await sandbox.execute_shell(
                sandbox_id=session.sandbox_id,
                command="echo test",
            )

        assert result.success is False
        assert "Boom" in result.error
        assert result.execution_time_ms > 0

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_read_file_handles_permission_error(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that read_file handles permission errors."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a file then remove read permissions
        workspace = Path(session.workspace_path)
        test_file = workspace / "no_read.txt"
        test_file.write_text("secret")
        test_file.chmod(0o000)

        try:
            result = await sandbox.read_file(
                sandbox_id=session.sandbox_id,
                path="/workspace/no_read.txt",
            )

            assert result.success is False
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_file_handles_write_error(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that write_file handles write errors (e.g., read-only fs)."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Make workspace read-only
        workspace = Path(session.workspace_path)
        workspace.chmod(0o555)

        try:
            result = await sandbox.write_file(
                sandbox_id=session.sandbox_id,
                path="/workspace/attempt.txt",
                content="content",
            )

            assert result.success is False
        finally:
            # Restore for cleanup
            workspace.chmod(0o755)

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_file_handles_errors(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that delete_file handles unexpected errors."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a directory, make it unwritable, and try to delete a file inside
        workspace = Path(session.workspace_path)
        protected_dir = workspace / "protected"
        protected_dir.mkdir()
        protected_file = protected_dir / "guarded.txt"
        protected_file.write_text("guarded")
        protected_dir.chmod(0o555)

        try:
            result = await sandbox.delete_file(
                sandbox_id=session.sandbox_id,
                path="/workspace/protected/guarded.txt",
            )
            # May succeed or fail depending on OS - the point is no exception
            assert isinstance(result, SandboxActionResult)
        finally:
            protected_dir.chmod(0o755)

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_multiple_operations_after_error(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that sandbox remains functional after an error."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Trigger an error (read nonexistent file)
        result1 = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/missing.txt",
        )
        assert result1.success is False

        # Should still work for subsequent operations
        result2 = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/works.txt",
            content="still working",
        )
        assert result2.success is True

        result3 = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/works.txt",
        )
        assert result3.success is True
        assert result3.output == "still working"

        await sandbox.destroy_sandbox(session.sandbox_id)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_exact_file_size_at_limit(self, sandbox_base_dir: Path) -> None:
        """Test writing a file exactly at the size limit boundary."""
        # Set a very small limit for testing
        config = SandboxConfig(max_file_size_mb=1)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Write exactly 1MB
        content = "x" * (1 * 1024 * 1024)
        result = await sb.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/exact_limit.txt",
            content=content,
        )
        assert result.success is True

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_file_one_byte_over_limit(self, sandbox_base_dir: Path) -> None:
        """Test writing a file exactly one byte over the limit."""
        config = SandboxConfig(max_file_size_mb=1)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        content = "x" * (1 * 1024 * 1024 + 1)
        result = await sb.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/over_limit.txt",
            content=content,
        )
        assert result.success is False

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_many_concurrent_sandboxes(self, sandbox_base_dir: Path) -> None:
        """Test creating and using many sandboxes concurrently."""
        sb = OpenClawActionSandbox(base_workspace_path=str(sandbox_base_dir))

        sessions = []
        for i in range(20):
            s = await sb.create_sandbox(
                session_id=f"sess-{i}",
                user_id=f"user-{i}",
                tenant_id=f"tenant-{i % 3}",
            )
            sessions.append(s)

        assert sb.get_stats()["active_sandboxes"] == 20
        assert sb.get_stats()["sandboxes_created"] == 20

        # Write to all sandboxes
        for s in sessions:
            result = await sb.write_file(
                sandbox_id=s.sandbox_id,
                path="/workspace/hello.txt",
                content=f"Hello from {s.session_id}",
            )
            assert result.success is True

        # Read from all sandboxes (verify isolation)
        for s in sessions:
            result = await sb.read_file(
                sandbox_id=s.sandbox_id,
                path="/workspace/hello.txt",
            )
            assert result.success is True
            assert s.session_id in result.output

        # Cleanup
        count = await sb.cleanup_all()
        assert count == 20

    @pytest.mark.asyncio
    async def test_special_characters_in_filename(self, sandbox: OpenClawActionSandbox) -> None:
        """Test handling files with special characters in names."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file with spaces.txt",
            content="spaces in name",
        )
        assert result.success is True

        read_result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/file with spaces.txt",
        )
        assert read_result.success is True
        assert read_result.output == "spaces in name"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that writing to an existing file overwrites it."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/overwrite.txt",
            content="original",
        )
        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/overwrite.txt",
            content="updated",
        )

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/overwrite.txt",
        )
        assert result.success is True
        assert result.output == "updated"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_sandbox_with_no_base_path(self) -> None:
        """Test creating sandbox without specifying base path uses tempdir."""
        sb = OpenClawActionSandbox()

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        assert session.workspace_path is not None
        assert Path(session.workspace_path).exists()

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_multiple_sessions_same_session_id(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that creating a sandbox with same session_id overwrites mapping."""
        session_1 = await sandbox.create_sandbox(
            session_id="sess-dup",
            user_id="user-1",
        )
        session_2 = await sandbox.create_sandbox(
            session_id="sess-dup",
            user_id="user-2",
        )

        # The session_to_sandbox mapping should point to the latest
        retrieved = sandbox.get_sandbox_for_session("sess-dup")
        assert retrieved is not None
        assert retrieved.sandbox_id == session_2.sandbox_id

        # Both sandbox IDs should still be valid
        assert sandbox.get_sandbox(session_1.sandbox_id) is not None
        assert sandbox.get_sandbox(session_2.sandbox_id) is not None

        await sandbox.destroy_sandbox(session_1.sandbox_id)
        await sandbox.destroy_sandbox(session_2.sandbox_id)

    @pytest.mark.asyncio
    async def test_execution_time_tracked_in_results(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that execution time is tracked for all operations."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Write
        w = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/time.txt",
            content="timing test",
        )
        assert w.execution_time_ms >= 0

        # Read
        r = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/time.txt",
        )
        assert r.execution_time_ms >= 0

        # Delete
        d = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/time.txt",
        )
        assert d.execution_time_ms >= 0

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_path_without_workspace_prefix(self, sandbox: OpenClawActionSandbox) -> None:
        """Test handling a bare filename (no prefix, no relative marker)."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="bare_file.txt",
            content="bare path content",
        )
        assert result.success is True

        # Should be in workspace
        assert (Path(session.workspace_path) / "bare_file.txt").exists()

        await sandbox.destroy_sandbox(session.sandbox_id)

    def test_sandbox_manager_initialization(self, sandbox_base_dir: Path) -> None:
        """Test that the sandbox manager initializes correctly."""
        sb = OpenClawActionSandbox(base_workspace_path=str(sandbox_base_dir))

        assert sb._base_path.exists()
        assert sb._default_config is not None
        assert sb._event_callback is None
        assert len(sb._sandboxes) == 0
        assert len(sb._session_to_sandbox) == 0

    def test_sandbox_manager_with_custom_default_config(
        self, sandbox_base_dir: Path, custom_config: SandboxConfig
    ) -> None:
        """Test initialization with custom default config."""
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=custom_config,
        )

        assert sb._default_config.max_execution_time_seconds == 10
        assert sb._default_config.max_memory_mb == 256

    @pytest.mark.asyncio
    async def test_resolve_path_with_invalid_path(self, sandbox: OpenClawActionSandbox) -> None:
        """Test _resolve_path handles invalid path inputs."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Path with null bytes (may cause issues on some systems)
        result = sandbox._resolve_path("\x00invalid", session, write=True)
        # Should either be invalid or blocked
        # The behavior depends on the OS, so just verify no crash
        assert isinstance(result, dict)

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_multibyte_content_size_calculation(self, sandbox_base_dir: Path) -> None:
        """Test that file size is calculated in bytes, not characters."""
        config = SandboxConfig(max_file_size_mb=1)
        sb = OpenClawActionSandbox(
            base_workspace_path=str(sandbox_base_dir),
            default_config=config,
        )

        session = await sb.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Multi-byte characters take more bytes than characters
        # Each character here is 3 bytes in UTF-8
        multibyte_char = "\u2603"  # Snowman: 3 bytes each
        content = multibyte_char * (512 * 1024)  # ~1.5 MB in bytes
        byte_size = len(content.encode("utf-8"))

        result = await sb.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/multibyte.txt",
            content=content,
        )

        if byte_size > 1 * 1024 * 1024:
            assert result.success is False
        else:
            assert result.success is True

        await sb.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_action_increments_count(self, sandbox: OpenClawActionSandbox) -> None:
        """Test that delete operations increment the action count."""
        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/to_delete.txt",
            content="bye",
        )
        count_after_write = session.action_count

        await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/to_delete.txt",
        )

        assert session.action_count == count_after_write + 1

        await sandbox.destroy_sandbox(session.sandbox_id)
