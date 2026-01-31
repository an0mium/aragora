"""Tests for external agent models."""

from datetime import datetime, timezone

import pytest

from aragora.agents.external.models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
    ToolInvocation,
    ToolPermission,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected task statuses exist."""
        expected = [
            "PENDING",
            "INITIALIZING",
            "RUNNING",
            "PAUSED",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
        ]
        for status in expected:
            assert hasattr(TaskStatus, status)

    def test_status_values(self) -> None:
        """Verify status values are lowercase strings."""
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.RUNNING.value == "running"


class TestToolPermission:
    """Tests for ToolPermission enum."""

    def test_permission_key_mapping(self) -> None:
        """Test conversion to RBAC permission keys."""
        assert ToolPermission.FILE_READ.to_permission_key() == "computer_use.file_read"
        assert ToolPermission.SHELL_EXECUTE.to_permission_key() == "computer_use.shell"
        assert ToolPermission.BROWSER_USE.to_permission_key() == "computer_use.browser"
        assert ToolPermission.NETWORK_ACCESS.to_permission_key() == "computer_use.network"

    def test_all_permissions_have_mapping(self) -> None:
        """Verify all permissions have a key mapping."""
        for perm in ToolPermission:
            if perm.name.startswith("_"):
                continue
            key = perm.to_permission_key()
            assert key.startswith("computer_use.")


class TestTaskRequest:
    """Tests for TaskRequest dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        request = TaskRequest(task_type="code", prompt="Write hello world")
        assert request.task_type == "code"
        assert request.prompt == "Write hello world"
        assert request.timeout_seconds == 3600.0
        assert request.max_steps == 100
        assert request.tool_permissions == set()
        assert request.id is not None
        assert request.created_at is not None

    def test_with_tool_permissions(self) -> None:
        """Test creating request with tool permissions."""
        request = TaskRequest(
            task_type="code",
            prompt="Create a file",
            tool_permissions={ToolPermission.FILE_WRITE, ToolPermission.SHELL_EXECUTE},
        )
        assert ToolPermission.FILE_WRITE in request.tool_permissions
        assert ToolPermission.SHELL_EXECUTE in request.tool_permissions

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        request = TaskRequest(
            task_type="code",
            prompt="Test prompt",
            user_id="user-123",
            tenant_id="tenant-456",
        )
        data = request.to_dict()
        assert data["task_type"] == "code"
        assert data["prompt"] == "Test prompt"
        assert data["user_id"] == "user-123"
        assert data["tenant_id"] == "tenant-456"
        assert "id" in data
        assert "created_at" in data


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_terminal_status_check(self) -> None:
        """Test is_terminal property."""
        completed = TaskResult(task_id="1", status=TaskStatus.COMPLETED)
        assert completed.is_terminal

        failed = TaskResult(task_id="2", status=TaskStatus.FAILED)
        assert failed.is_terminal

        running = TaskResult(task_id="3", status=TaskStatus.RUNNING)
        assert not running.is_terminal

    def test_duration_calculation(self) -> None:
        """Test duration_seconds calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
        result = TaskResult(
            task_id="1",
            status=TaskStatus.COMPLETED,
            started_at=start,
            completed_at=end,
        )
        assert result.duration_seconds == 300.0

    def test_duration_none_when_incomplete(self) -> None:
        """Test duration is None when task is incomplete."""
        result = TaskResult(
            task_id="1",
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        assert result.duration_seconds is None

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = TaskResult(
            task_id="task-123",
            status=TaskStatus.COMPLETED,
            output="Hello, World!",
            tokens_used=100,
            cost_usd=0.05,
        )
        data = result.to_dict()
        assert data["task_id"] == "task-123"
        assert data["status"] == "completed"
        assert data["output"] == "Hello, World!"
        assert data["tokens_used"] == 100
        assert data["cost_usd"] == 0.05


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_progress_percent_with_total(self) -> None:
        """Test progress percent calculation."""
        progress = TaskProgress(
            task_id="1",
            status=TaskStatus.RUNNING,
            current_step=5,
            total_steps=10,
            message="Processing",
        )
        assert progress.progress_percent == 50.0

    def test_progress_percent_without_total(self) -> None:
        """Test progress percent is None without total."""
        progress = TaskProgress(
            task_id="1",
            status=TaskStatus.RUNNING,
            current_step=5,
            total_steps=None,
            message="Processing",
        )
        assert progress.progress_percent is None

    def test_to_dict(self) -> None:
        """Test serialization."""
        progress = TaskProgress(
            task_id="1",
            status=TaskStatus.RUNNING,
            current_step=3,
            total_steps=10,
            message="Working",
        )
        data = progress.to_dict()
        assert data["current_step"] == 3
        assert data["total_steps"] == 10
        assert data["progress_percent"] == 30.0


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy_status(self) -> None:
        """Test healthy status creation."""
        status = HealthStatus(
            adapter_name="openhands",
            healthy=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=50.0,
            framework_version="1.0.0",
        )
        assert status.healthy
        assert status.adapter_name == "openhands"
        assert status.framework_version == "1.0.0"

    def test_unhealthy_status_with_error(self) -> None:
        """Test unhealthy status with error message."""
        status = HealthStatus(
            adapter_name="openhands",
            healthy=False,
            last_check=datetime.now(timezone.utc),
            response_time_ms=0.0,
            error="Connection refused",
        )
        assert not status.healthy
        assert status.error == "Connection refused"


class TestToolInvocation:
    """Tests for ToolInvocation dataclass."""

    def test_successful_invocation(self) -> None:
        """Test successful tool invocation."""
        inv = ToolInvocation(
            tool_name="TerminalTool",
            arguments={"command": "ls -la"},
            result="file1.txt\nfile2.txt",
            success=True,
            duration_ms=150.0,
        )
        assert inv.success
        assert not inv.blocked
        assert inv.tool_name == "TerminalTool"

    def test_blocked_invocation(self) -> None:
        """Test blocked tool invocation."""
        inv = ToolInvocation(
            tool_name="TerminalTool",
            arguments={"command": "rm -rf /"},
            blocked=True,
            block_reason="Dangerous command detected",
            success=False,
        )
        assert inv.blocked
        assert not inv.success
        assert "Dangerous" in inv.block_reason

    def test_to_dict(self) -> None:
        """Test serialization."""
        inv = ToolInvocation(
            tool_name="FileEditorTool",
            arguments={"path": "/tmp/test.txt"},
            result="File written",
            success=True,
        )
        data = inv.to_dict()
        assert data["tool_name"] == "FileEditorTool"
        assert data["success"] is True
        assert "id" in data
        assert "timestamp" in data
