"""Tests for external agent sandbox isolation."""

import pytest
import asyncio
import os
import tempfile

from aragora.gateway.external_agents.base import IsolationLevel
from aragora.gateway.external_agents.sandbox import (
    SandboxConfig,
    SandboxExecution,
    SandboxState,
    SandboxManager,
    ProcessSandbox,
)


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        assert config.isolation_level == IsolationLevel.CONTAINER
        assert config.max_memory_mb == 512
        assert config.max_cpu_cores == 1.0
        assert config.max_execution_seconds == 300
        assert config.allow_network is False
        assert config.read_only_root is True
        assert config.no_new_privileges is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            isolation_level=IsolationLevel.PROCESS,
            max_memory_mb=1024,
            max_cpu_cores=2.0,
            allow_network=True,
            allowed_hosts=["api.example.com"],
        )
        assert config.isolation_level == IsolationLevel.PROCESS
        assert config.max_memory_mb == 1024
        assert config.max_cpu_cores == 2.0
        assert config.allow_network is True
        assert "api.example.com" in config.allowed_hosts

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = SandboxConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_memory_too_low(self):
        """Test validation fails for memory too low."""
        config = SandboxConfig(max_memory_mb=32)
        errors = config.validate()
        assert len(errors) > 0
        assert any("memory" in e.lower() for e in errors)

    def test_validate_memory_too_high(self):
        """Test validation fails for memory too high."""
        config = SandboxConfig(max_memory_mb=32768)
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_cpu_too_low(self):
        """Test validation fails for CPU too low."""
        config = SandboxConfig(max_cpu_cores=0.01)
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_execution_time_too_low(self):
        """Test validation fails for execution time too low."""
        config = SandboxConfig(max_execution_seconds=0)
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_execution_time_too_high(self):
        """Test validation fails for execution time too high."""
        config = SandboxConfig(max_execution_seconds=7200)
        errors = config.validate()
        assert len(errors) > 0


class TestSandboxExecution:
    """Tests for SandboxExecution."""

    def test_success_execution(self):
        """Test successful execution result."""
        execution = SandboxExecution(
            execution_id="exec-123",
            success=True,
            output="Hello World",
            exit_code=0,
            execution_time_ms=100.0,
        )
        assert execution.success is True
        assert execution.output == "Hello World"
        assert execution.exit_code == 0

    def test_failed_execution(self):
        """Test failed execution result."""
        execution = SandboxExecution(
            execution_id="exec-123",
            success=False,
            error="Command not found",
            exit_code=127,
            state=SandboxState.FAILED,
        )
        assert execution.success is False
        assert execution.error == "Command not found"
        assert execution.exit_code == 127
        assert execution.state == SandboxState.FAILED


class TestSandboxState:
    """Tests for SandboxState enum."""

    def test_state_values(self):
        """Test state values."""
        assert SandboxState.CREATED.value == "created"
        assert SandboxState.RUNNING.value == "running"
        assert SandboxState.STOPPED.value == "stopped"
        assert SandboxState.FAILED.value == "failed"


class TestProcessSandbox:
    """Tests for ProcessSandbox backend."""

    @pytest.mark.asyncio
    async def test_is_available(self):
        """Test process sandbox is always available."""
        sandbox = ProcessSandbox()
        available = await sandbox.is_available()
        assert available is True

    @pytest.mark.asyncio
    async def test_create_sandbox(self):
        """Test creating a process sandbox."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await sandbox.create(config)

        assert instance_id.startswith("proc-")
        assert instance_id in sandbox._instances

        # Cleanup
        await sandbox.destroy(instance_id)

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        """Test executing a simple command."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await sandbox.create(config)

        result = await sandbox.execute(
            instance_id,
            ["echo", "Hello, World!"],
            {},
        )

        assert result.success is True
        assert "Hello, World!" in result.output
        assert result.exit_code == 0

        # Cleanup
        await sandbox.destroy(instance_id)

    @pytest.mark.asyncio
    async def test_execute_with_env_vars(self):
        """Test executing with environment variables."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await sandbox.create(config)

        result = await sandbox.execute(
            instance_id,
            ["sh", "-c", "echo $TEST_VAR"],
            {"TEST_VAR": "test_value"},
        )

        assert result.success is True
        assert "test_value" in result.output

        # Cleanup
        await sandbox.destroy(instance_id)

    @pytest.mark.asyncio
    async def test_execute_with_stdin(self):
        """Test executing with stdin input."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await sandbox.create(config)

        result = await sandbox.execute(
            instance_id,
            ["cat"],
            {},
            stdin="Input from stdin",
        )

        assert result.success is True
        assert "Input from stdin" in result.output

        # Cleanup
        await sandbox.destroy(instance_id)

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test command timeout."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(
            isolation_level=IsolationLevel.PROCESS,
            max_execution_seconds=1,
        )
        instance_id = await sandbox.create(config)

        result = await sandbox.execute(
            instance_id,
            ["sleep", "10"],
            {},
        )

        assert result.success is False
        assert "timeout" in result.error.lower()

        # Cleanup
        await sandbox.destroy(instance_id)

    @pytest.mark.asyncio
    async def test_execute_nonexistent_instance(self):
        """Test executing on nonexistent instance."""
        sandbox = ProcessSandbox()
        result = await sandbox.execute(
            "nonexistent-id",
            ["echo", "test"],
            {},
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_destroy_sandbox(self):
        """Test destroying a sandbox."""
        sandbox = ProcessSandbox()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await sandbox.create(config)

        result = await sandbox.destroy(instance_id)
        assert result is True
        assert instance_id not in sandbox._instances

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_sandbox(self):
        """Test destroying nonexistent sandbox."""
        sandbox = ProcessSandbox()
        result = await sandbox.destroy("nonexistent-id")
        assert result is False


class TestSandboxManager:
    """Tests for SandboxManager."""

    @pytest.mark.asyncio
    async def test_create_process_sandbox(self):
        """Test creating a process sandbox through manager."""
        manager = SandboxManager()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await manager.create_sandbox(config)

        assert instance_id.startswith("proc-")
        assert instance_id in manager.get_active_sandboxes()

        # Cleanup
        await manager.destroy_sandbox(instance_id)

    @pytest.mark.asyncio
    async def test_execute_through_manager(self):
        """Test executing through sandbox manager."""
        manager = SandboxManager()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)
        instance_id = await manager.create_sandbox(config)

        result = await manager.execute(
            instance_id,
            ["echo", "managed execution"],
        )

        assert result.success is True
        assert "managed execution" in result.output

        # Cleanup
        await manager.destroy_sandbox(instance_id)

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all sandboxes."""
        manager = SandboxManager()
        config = SandboxConfig(isolation_level=IsolationLevel.PROCESS)

        # Create multiple sandboxes
        ids = []
        for _ in range(3):
            instance_id = await manager.create_sandbox(config)
            ids.append(instance_id)

        assert len(manager.get_active_sandboxes()) == 3

        # Cleanup all
        count = await manager.cleanup_all()
        assert count == 3
        assert len(manager.get_active_sandboxes()) == 0

    @pytest.mark.asyncio
    async def test_invalid_config_raises(self):
        """Test that invalid config raises error."""
        manager = SandboxManager()
        config = SandboxConfig(max_memory_mb=10)  # Too low

        with pytest.raises(ValueError):
            await manager.create_sandbox(config)

    @pytest.mark.asyncio
    async def test_execute_nonexistent_sandbox(self):
        """Test executing on nonexistent sandbox."""
        manager = SandboxManager()
        result = await manager.execute(
            "nonexistent",
            ["echo", "test"],
        )
        assert result.success is False
