"""
Sandbox Executor for Safe Code Execution.

Provides isolated execution environments using Docker or subprocess isolation:
- Resource limits (CPU, memory, time)
- Tool policy enforcement
- Secure file system boundaries
- Network restrictions
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from aragora.sandbox.policies import (
    ToolPolicy,
    ToolPolicyChecker,
    create_default_policy,
)

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Mode of sandbox execution."""

    DOCKER = "docker"
    SUBPROCESS = "subprocess"
    MOCK = "mock"  # For testing


class ExecutionStatus(str, Enum):
    """Status of sandbox execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    POLICY_DENIED = "policy_denied"


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    execution_id: str
    status: ExecutionStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    memory_used_mb: float = 0.0
    files_created: list[str] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "memory_used_mb": self.memory_used_mb,
            "files_created": self.files_created,
            "policy_violations": self.policy_violations,
            "error_message": self.error_message,
        }


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    mode: ExecutionMode = ExecutionMode.SUBPROCESS
    policy: Optional[ToolPolicy] = None
    docker_image: str = "python:3.11-slim"
    workspace_base: str = "/tmp/sandbox"
    cleanup_on_complete: bool = True
    capture_output: bool = True
    network_enabled: bool = False


class SandboxExecutor:
    """
    Executes code in a sandboxed environment.

    Supports multiple isolation modes:
    - Docker: Full container isolation (most secure)
    - Subprocess: Process-level isolation with resource limits
    - Mock: For testing without actual execution
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.policy = self.config.policy or create_default_policy()
        self.checker = ToolPolicyChecker(self.policy)
        self._active_executions: dict[str, asyncio.subprocess.Process] = {}

        # Ensure workspace exists
        Path(self.config.workspace_base).mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
        files: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)
            timeout: Execution timeout in seconds
            env: Environment variables
            files: Additional files to create {filename: content}

        Returns:
            ExecutionResult with stdout, stderr, and status
        """
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"
        timeout = timeout or self.policy.resource_limits.max_execution_seconds

        # Create isolated workspace
        workspace = Path(self.config.workspace_base) / execution_id
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            # Write code file
            code_file = self._write_code_file(workspace, code, language)

            # Write additional files
            if files:
                for filename, content in files.items():
                    file_path = workspace / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)

            # Execute based on mode
            if self.config.mode == ExecutionMode.DOCKER:
                result = await self._execute_docker(
                    execution_id, workspace, code_file, language, timeout, env
                )
            elif self.config.mode == ExecutionMode.SUBPROCESS:
                result = await self._execute_subprocess(
                    execution_id, workspace, code_file, language, timeout, env
                )
            else:
                result = self._execute_mock(execution_id, code)

            # Collect created files
            if workspace.exists():
                result.files_created = [
                    str(f.relative_to(workspace)) for f in workspace.rglob("*") if f.is_file()
                ]

            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            logger.exception(f"Sandbox execution error: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )
        finally:
            # Cleanup
            if self.config.cleanup_on_complete and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    def _write_code_file(
        self,
        workspace: Path,
        code: str,
        language: str,
    ) -> Path:
        """Write code to a file in the workspace."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "shell": ".sh",
        }
        ext = extensions.get(language, ".txt")
        code_file = workspace / f"main{ext}"
        code_file.write_text(code)
        return code_file

    async def _execute_subprocess(
        self,
        execution_id: str,
        workspace: Path,
        code_file: Path,
        language: str,
        timeout: float,
        env: Optional[dict[str, str]],
    ) -> ExecutionResult:
        """Execute code using subprocess with resource limits."""
        import time

        # Build command based on language
        commands = {
            "python": ["python3", str(code_file)],
            "javascript": ["node", str(code_file)],
            "bash": ["bash", str(code_file)],
            "shell": ["sh", str(code_file)],
        }
        cmd = commands.get(language)
        if not cmd:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Unsupported language: {language}",
            )

        # Check tool policy
        tool_name = cmd[0]
        allowed, reason = self.checker.check_tool(tool_name)
        if not allowed:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.POLICY_DENIED,
                policy_violations=[f"Tool '{tool_name}' denied: {reason}"],
                error_message=f"Execution denied by policy: {reason}",
            )

        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # Apply resource limits via ulimit (Unix only)
        limits = self.policy.resource_limits
        preexec = None
        if os.name != "nt":  # Not Windows
            import resource

            def set_limits():
                # Memory limit
                mem_bytes = limits.max_memory_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, resource.error):
                    pass

                # CPU time limit
                try:
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (limits.max_execution_seconds, limits.max_execution_seconds),
                    )
                except (ValueError, resource.error):
                    pass

            preexec = set_limits

        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if self.config.capture_output else None,
                stderr=asyncio.subprocess.PIPE if self.config.capture_output else None,
                cwd=workspace,
                env=exec_env,
                preexec_fn=preexec,
            )

            self._active_executions[execution_id] = proc

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            duration = time.time() - start_time

            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED
                if proc.returncode == 0
                else ExecutionStatus.FAILED,
                exit_code=proc.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                duration_seconds=duration,
            )

        finally:
            self._active_executions.pop(execution_id, None)

    async def _execute_docker(
        self,
        execution_id: str,
        workspace: Path,
        code_file: Path,
        language: str,
        timeout: float,
        env: Optional[dict[str, str]],
    ) -> ExecutionResult:
        """Execute code in a Docker container."""
        import time

        limits = self.policy.resource_limits

        # Build docker run command
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"sandbox-{execution_id}",
            # Resource limits
            f"--memory={limits.max_memory_mb}m",
            f"--cpus={limits.max_cpu_percent / 100}",
            "--pids-limit",
            str(limits.max_processes),
            # Security
            "--security-opt=no-new-privileges",
            "--read-only",
            # Network
            *([] if self.config.network_enabled else ["--network=none"]),
            # Mount workspace
            "-v",
            f"{workspace}:/workspace:rw",
            "-w",
            "/workspace",
        ]

        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        cmd.append(self.config.docker_image)

        # Add execution command
        exec_cmds = {
            "python": ["python3", f"/workspace/{code_file.name}"],
            "javascript": ["node", f"/workspace/{code_file.name}"],
            "bash": ["bash", f"/workspace/{code_file.name}"],
        }
        cmd.extend(exec_cmds.get(language, ["cat", f"/workspace/{code_file.name}"]))

        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout + 5,  # Extra time for container overhead
            )

            duration = time.time() - start_time

            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED
                if proc.returncode == 0
                else ExecutionStatus.FAILED,
                exit_code=proc.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                duration_seconds=duration,
            )

        except FileNotFoundError:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message="Docker not found. Install Docker or use subprocess mode.",
            )

    def _execute_mock(
        self,
        execution_id: str,
        code: str,
    ) -> ExecutionResult:
        """Mock execution for testing."""
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout=f"[MOCK] Would execute:\n{code[:500]}",
            stderr="",
            duration_seconds=0.1,
        )

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        proc = self._active_executions.get(execution_id)
        if proc:
            proc.kill()
            await proc.wait()
            return True

        # Try to kill Docker container
        try:
            await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                f"sandbox-{execution_id}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

        return False

    def get_policy_checker(self) -> ToolPolicyChecker:
        """Get the policy checker for manual checks."""
        return self.checker

    def update_policy(self, policy: ToolPolicy) -> None:
        """Update the execution policy."""
        self.policy = policy
        self.checker = ToolPolicyChecker(policy)


# Convenience function
async def execute_sandboxed(
    code: str,
    language: str = "python",
    timeout: float = 60.0,
) -> ExecutionResult:
    """Execute code in a default sandbox."""
    executor = SandboxExecutor()
    return await executor.execute(code, language, timeout)


__all__ = [
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionStatus",
    "SandboxConfig",
    "SandboxExecutor",
    "execute_sandboxed",
]
