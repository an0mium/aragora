"""
Tests for Computer Use Sandbox component.

Tests sandbox creation, lifecycle, and isolation.
"""

import pytest
from pathlib import Path

from aragora.computer_use.sandbox import (
    ProcessSandboxProvider,
    SandboxConfig,
    SandboxInstance,
    SandboxManager,
    SandboxStatus,
    SandboxType,
)


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default sandbox configuration."""
        config = SandboxConfig()

        assert config.sandbox_type == SandboxType.DOCKER
        assert config.memory_limit_mb == 2048
        assert config.cpu_limit_cores == 2.0
        assert config.timeout_seconds == 300.0
        assert config.network_enabled is True

    def test_custom_config(self):
        """Test custom sandbox configuration."""
        config = SandboxConfig(
            sandbox_type=SandboxType.PROCESS,
            memory_limit_mb=1024,
            network_enabled=False,
            timeout_seconds=60.0,
        )

        assert config.sandbox_type == SandboxType.PROCESS
        assert config.memory_limit_mb == 1024
        assert config.network_enabled is False


class TestSandboxTypes:
    """Tests for SandboxType enum."""

    def test_sandbox_types(self):
        """Test sandbox type values."""
        assert SandboxType.NONE.value == "none"
        assert SandboxType.PROCESS.value == "process"
        assert SandboxType.DOCKER.value == "docker"
        assert SandboxType.FIREJAIL.value == "firejail"


class TestSandboxStatus:
    """Tests for SandboxStatus enum."""

    def test_sandbox_statuses(self):
        """Test sandbox status values."""
        assert SandboxStatus.CREATED.value == "created"
        assert SandboxStatus.STARTING.value == "starting"
        assert SandboxStatus.RUNNING.value == "running"
        assert SandboxStatus.STOPPING.value == "stopping"
        assert SandboxStatus.STOPPED.value == "stopped"
        assert SandboxStatus.ERROR.value == "error"


class TestProcessSandboxProvider:
    """Tests for ProcessSandboxProvider."""

    @pytest.fixture
    def provider(self) -> ProcessSandboxProvider:
        """Create a process sandbox provider."""
        return ProcessSandboxProvider()

    @pytest.mark.asyncio
    async def test_create_sandbox(self, provider: ProcessSandboxProvider):
        """Test creating a process sandbox."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)

        assert instance is not None
        assert instance.id is not None
        assert instance.status == SandboxStatus.CREATED
        assert instance.temp_dir is not None
        assert Path(instance.temp_dir).exists()

        # Cleanup
        await provider.destroy(instance)

    @pytest.mark.asyncio
    async def test_start_sandbox(self, provider: ProcessSandboxProvider):
        """Test starting a process sandbox."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)

        await provider.start(instance)

        assert instance.status == SandboxStatus.RUNNING
        assert instance.started_at is not None

        await provider.destroy(instance)

    @pytest.mark.asyncio
    async def test_stop_sandbox(self, provider: ProcessSandboxProvider):
        """Test stopping a process sandbox."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)
        await provider.start(instance)

        await provider.stop(instance)

        assert instance.status == SandboxStatus.STOPPED

        await provider.destroy(instance)

    @pytest.mark.asyncio
    async def test_execute_command(self, provider: ProcessSandboxProvider):
        """Test executing a command in sandbox."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)
        await provider.start(instance)

        exit_code, stdout, stderr = await provider.execute(
            instance, ["echo", "hello"]
        )

        assert exit_code == 0
        assert "hello" in stdout
        assert stderr == ""

        await provider.destroy(instance)

    @pytest.mark.asyncio
    async def test_execute_timeout(self, provider: ProcessSandboxProvider):
        """Test command timeout in sandbox."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)
        await provider.start(instance)

        exit_code, stdout, stderr = await provider.execute(
            instance, ["sleep", "5"], timeout=0.5
        )

        assert exit_code == -1
        assert "timed out" in stderr.lower()

        await provider.destroy(instance)

    @pytest.mark.asyncio
    async def test_health_check(self, provider: ProcessSandboxProvider):
        """Test sandbox health check."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await provider.create(config)

        # Not running yet
        assert await provider.health_check(instance) is False

        await provider.start(instance)
        assert await provider.health_check(instance) is True

        await provider.stop(instance)
        assert await provider.health_check(instance) is False

        await provider.destroy(instance)


class TestSandboxManager:
    """Tests for SandboxManager."""

    @pytest.fixture
    def manager(self) -> SandboxManager:
        """Create a sandbox manager."""
        return SandboxManager()

    @pytest.mark.asyncio
    async def test_create_sandbox(self, manager: SandboxManager):
        """Test creating a sandbox via manager."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await manager.create_sandbox(config)

        assert instance is not None
        assert instance.id is not None

        await manager.destroy_sandbox(instance.id)

    @pytest.mark.asyncio
    async def test_start_sandbox(self, manager: SandboxManager):
        """Test starting a sandbox via manager."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await manager.create_sandbox(config)

        await manager.start_sandbox(instance.id)

        updated = await manager.get_sandbox(instance.id)
        assert updated is not None
        assert updated.status == SandboxStatus.RUNNING

        await manager.destroy_sandbox(instance.id)

    @pytest.mark.asyncio
    async def test_list_sandboxes(self, manager: SandboxManager):
        """Test listing sandboxes."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)

        # Create multiple sandboxes
        instance1 = await manager.create_sandbox(config)
        instance2 = await manager.create_sandbox(config)

        sandboxes = await manager.list_sandboxes()
        assert len(sandboxes) == 2

        await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_cleanup_all(self, manager: SandboxManager):
        """Test cleaning up all sandboxes."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)

        await manager.create_sandbox(config)
        await manager.create_sandbox(config)

        await manager.cleanup_all()

        sandboxes = await manager.list_sandboxes()
        assert len(sandboxes) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, manager: SandboxManager):
        """Test getting sandbox stats."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)

        instance = await manager.create_sandbox(config)
        await manager.start_sandbox(instance.id)

        stats = await manager.get_stats()

        assert stats["total_sandboxes"] == 1
        assert "running" in stats["by_status"]
        assert "process" in stats["by_type"]

        await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_execute_in_sandbox(self, manager: SandboxManager):
        """Test executing a command via manager."""
        config = SandboxConfig(sandbox_type=SandboxType.PROCESS)
        instance = await manager.create_sandbox(config)
        await manager.start_sandbox(instance.id)

        exit_code, stdout, stderr = await manager.execute_in_sandbox(
            instance.id, ["echo", "test"]
        )

        assert exit_code == 0
        assert "test" in stdout

        await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_execute_nonexistent_sandbox(self, manager: SandboxManager):
        """Test executing in nonexistent sandbox."""
        with pytest.raises(ValueError, match="not found"):
            await manager.execute_in_sandbox("nonexistent", ["echo"])

    @pytest.mark.asyncio
    async def test_start_nonexistent_sandbox(self, manager: SandboxManager):
        """Test starting nonexistent sandbox."""
        with pytest.raises(ValueError, match="not found"):
            await manager.start_sandbox("nonexistent")
