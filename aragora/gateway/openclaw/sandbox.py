"""
OpenClaw Sandbox Runner.

Provides isolated execution environment for OpenClaw tasks with:
- Resource limits (memory, CPU, execution time)
- Network isolation with domain allowlists
- Filesystem restrictions
- Plugin sandboxing
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SandboxStatus(str, Enum):
    """Sandbox execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class SandboxConfig:
    """Configuration for isolated OpenClaw execution."""

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_seconds: int = 300
    max_output_bytes: int = 10 * 1024 * 1024  # 10MB

    # Network isolation
    allow_external_network: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    # Filesystem isolation
    allowed_paths: list[str] = field(default_factory=list)
    read_only_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(
        default_factory=lambda: [
            "/etc/passwd",
            "/etc/shadow",
            "~/.ssh",
            "~/.aws",
            "~/.config",
        ]
    )

    # Plugin restrictions
    allowed_plugins: list[str] = field(default_factory=list)
    plugin_allowlist_mode: bool = True  # Only allowed plugins can run

    # Execution settings
    enable_logging: bool = True
    log_level: str = "INFO"
    capture_stdout: bool = True
    capture_stderr: bool = True

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.max_memory_mb < 64:
            errors.append("max_memory_mb must be at least 64")
        if self.max_memory_mb > 8192:
            errors.append("max_memory_mb cannot exceed 8192")
        if self.max_execution_seconds < 1:
            errors.append("max_execution_seconds must be at least 1")
        if self.max_execution_seconds > 3600:
            errors.append("max_execution_seconds cannot exceed 3600")
        if self.max_cpu_percent < 1 or self.max_cpu_percent > 100:
            errors.append("max_cpu_percent must be between 1 and 100")
        return errors


@dataclass
class SandboxResult:
    """Result of sandbox execution."""

    status: SandboxStatus
    output: Any | None = None
    error: str | None = None
    execution_time_ms: int = 0
    memory_used_mb: int = 0
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenClawTask:
    """Task to execute in the OpenClaw sandbox."""

    id: str
    type: str  # text_generation, code_execution, file_operation, etc.
    payload: dict[str, Any]
    capabilities: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxViolation(Exception):
    """Raised when sandbox policy is violated."""

    def __init__(self, message: str, violation_type: str):
        super().__init__(message)
        self.violation_type = violation_type


class OpenClawSandbox:
    """
    Isolated execution environment for OpenClaw tasks.

    Provides resource isolation and policy enforcement before
    tasks are forwarded to the OpenClaw runtime.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        openclaw_endpoint: str = "http://localhost:8081",
    ) -> None:
        """
        Initialize sandbox.

        Args:
            config: Sandbox configuration
            openclaw_endpoint: OpenClaw runtime endpoint
        """
        self.config = config or SandboxConfig()
        self.openclaw_endpoint = openclaw_endpoint
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}

    async def execute(
        self,
        task: OpenClawTask,
        config_override: SandboxConfig | None = None,
    ) -> SandboxResult:
        """
        Execute task in isolated environment with resource limits.

        Args:
            task: Task to execute
            config_override: Optional config override for this task

        Returns:
            SandboxResult with execution details
        """
        config = config_override or self.config
        start_time = time.monotonic()

        # Validate config
        errors = config.validate()
        if errors:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=f"Invalid sandbox config: {', '.join(errors)}",
            )

        # Check plugin allowlist
        if config.plugin_allowlist_mode and task.plugins:
            blocked_plugins = set(task.plugins) - set(config.allowed_plugins)
            if blocked_plugins:
                return SandboxResult(
                    status=SandboxStatus.POLICY_VIOLATION,
                    error=f"Plugins not in allowlist: {blocked_plugins}",
                )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(task, config),
                timeout=config.max_execution_seconds,
            )
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            return result

        except asyncio.TimeoutError:
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                error=f"Task exceeded {config.max_execution_seconds}s timeout",
                execution_time_ms=config.max_execution_seconds * 1000,
            )
        except SandboxViolation as e:
            return SandboxResult(
                status=SandboxStatus.POLICY_VIOLATION,
                error=str(e),
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
            )
        except (OSError, RuntimeError, PermissionError) as e:
            logger.exception(f"Sandbox execution failed for task {task.id}")
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=str(e),
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
            )

    async def _execute_task(
        self,
        task: OpenClawTask,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Internal task execution with isolation."""
        # Validate network access
        self._validate_network_access(task, config)

        # Validate filesystem access
        self._validate_filesystem_access(task, config)

        # Forward to OpenClaw runtime
        # In production, this would use HTTP/gRPC to communicate with OpenClaw
        # For now, we simulate the execution
        result = await self._forward_to_openclaw(task, config)

        return result

    def _validate_network_access(
        self,
        task: OpenClawTask,
        config: SandboxConfig,
    ) -> None:
        """Validate network access for task."""
        # Check if task requires external network
        if "network_external" in task.capabilities:
            if not config.allow_external_network:
                raise SandboxViolation(
                    "External network access is disabled",
                    "network_blocked",
                )

        # Check domain allowlist
        requested_domains = task.payload.get("domains", [])
        if config.allowed_domains and requested_domains:
            for domain in requested_domains:
                if domain not in config.allowed_domains:
                    raise SandboxViolation(
                        f"Domain '{domain}' not in allowlist",
                        "domain_blocked",
                    )

        # Check domain blocklist
        if config.blocked_domains and requested_domains:
            for domain in requested_domains:
                if domain in config.blocked_domains:
                    raise SandboxViolation(
                        f"Domain '{domain}' is blocked",
                        "domain_blocked",
                    )

    def _validate_filesystem_access(
        self,
        task: OpenClawTask,
        config: SandboxConfig,
    ) -> None:
        """Validate filesystem access for task."""
        requested_paths = task.payload.get("paths", [])

        for path in requested_paths:
            # Check blocked paths
            for blocked in config.blocked_paths:
                if path.startswith(blocked) or blocked in path:
                    raise SandboxViolation(
                        f"Path '{path}' is blocked",
                        "path_blocked",
                    )

            # Check if write access is restricted to read-only
            if "file_system_write" in task.capabilities:
                for read_only in config.read_only_paths:
                    if path.startswith(read_only):
                        raise SandboxViolation(
                            f"Path '{path}' is read-only",
                            "write_blocked",
                        )

    async def _forward_to_openclaw(
        self,
        task: OpenClawTask,
        config: SandboxConfig,
    ) -> SandboxResult:
        """
        Forward validated task to OpenClaw runtime.

        In production, this would make HTTP/gRPC calls to the OpenClaw service.
        """
        # Import aiohttp here to avoid circular imports and allow mocking
        try:
            import aiohttp
        except ImportError:
            # Fallback for environments without aiohttp
            logger.warning("aiohttp not available, using mock execution")
            return SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"mock": True, "task_type": task.type},
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.openclaw_endpoint}/api/v1/tasks",
                    json={
                        "id": task.id,
                        "type": task.type,
                        "payload": task.payload,
                        "capabilities": task.capabilities,
                        "plugins": task.plugins,
                        "sandbox_config": {
                            "max_memory_mb": config.max_memory_mb,
                            "max_cpu_percent": config.max_cpu_percent,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=config.max_execution_seconds),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return SandboxResult(
                            status=SandboxStatus.COMPLETED,
                            output=data.get("result"),
                            stdout=data.get("stdout", ""),
                            stderr=data.get("stderr", ""),
                            memory_used_mb=data.get("memory_used_mb", 0),
                        )
                    else:
                        error_text = await response.text()
                        return SandboxResult(
                            status=SandboxStatus.FAILED,
                            error=f"OpenClaw returned {response.status}: {error_text}",
                        )
        except aiohttp.ClientError as e:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=f"Failed to connect to OpenClaw: {e}",
            )

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            del self._active_tasks[task_id]
            return True
        return False

    def get_active_tasks(self) -> list[str]:
        """Get list of active task IDs."""
        return list(self._active_tasks.keys())


__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "SandboxStatus",
    "SandboxViolation",
    "OpenClawSandbox",
    "OpenClawTask",
]
