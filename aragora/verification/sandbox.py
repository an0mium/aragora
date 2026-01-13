"""
Proof Execution Sandbox - Isolated subprocess execution for formal verification.

Security Features:
- Subprocess isolation with resource limits
- Timeout enforcement (hard kill after timeout)
- Memory limits (ulimit or resource module)
- Temporary directory cleanup
- No network access (where possible)

Usage:
    sandbox = ProofSandbox(timeout=30.0, memory_mb=512)
    result = await sandbox.execute_lean(lean_code)
    result = await sandbox.execute_z3(smtlib_code)
"""

import asyncio
import logging
import os
import resource
import shutil
import signal
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SandboxStatus(Enum):
    """Status of sandbox execution."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    EXECUTION_ERROR = "execution_error"
    SETUP_FAILED = "setup_failed"
    KILLED = "killed"


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    status: SandboxStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    error_message: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == SandboxStatus.SUCCESS and self.exit_code == 0


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    timeout_seconds: float = 30.0
    memory_mb: int = 512
    max_output_bytes: int = 1024 * 1024  # 1MB
    cleanup_on_exit: bool = True
    allow_network: bool = False
    working_dir: Optional[Path] = None


class ProofSandbox:
    """
    Sandbox for executing formal verification proofs.

    Provides isolated subprocess execution with:
    - Hard timeout enforcement
    - Memory limits
    - Temporary file isolation
    - Cleanup after execution
    """

    def __init__(
        self,
        timeout: float = 30.0,
        memory_mb: int = 512,
        max_output_bytes: int = 1024 * 1024,
    ):
        """
        Initialize the sandbox.

        Args:
            timeout: Maximum execution time in seconds
            memory_mb: Maximum memory in megabytes
            max_output_bytes: Maximum output size in bytes
        """
        self.config = SandboxConfig(
            timeout_seconds=timeout,
            memory_mb=memory_mb,
            max_output_bytes=max_output_bytes,
        )
        self._temp_dirs: list[Path] = []
        self._closed = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up all temporary directories (idempotent)."""
        if self._closed:
            return
        self._cleanup_temp_dirs()
        self._closed = True

    def __del__(self):
        """Destructor - fallback cleanup if context manager not used.

        Note: Bare except is intentional here - during garbage collection,
        any exception (including AttributeError for partially-constructed objects)
        must be suppressed to avoid runtime errors during interpreter shutdown.
        """
        try:
            if not self._closed and self.config.cleanup_on_exit:
                self._cleanup_temp_dirs()
        except Exception:
            pass  # Suppress all errors during garbage collection (standard Python pattern)

    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for sandboxed execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix="aragora_sandbox_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def _cleanup_temp_dirs(self):
        """Clean up all temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self._temp_dirs.clear()

    def _set_resource_limits(self):
        """
        Set resource limits for the subprocess.

        Called in the subprocess via preexec_fn.
        """
        # Memory limit (soft and hard)
        memory_bytes = self.config.memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, resource.error) as e:
            logger.debug(f"Could not set memory limit: {e}")

        # CPU time limit (backup for timeout)
        cpu_seconds = int(self.config.timeout_seconds * 2)  # 2x margin
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, resource.error) as e:
            logger.debug(f"Could not set CPU limit: {e}")

        # Limit file descriptor count
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except (ValueError, resource.error) as e:
            logger.debug(f"Could not set file descriptor limit: {e}")

        # Limit number of processes/threads
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, resource.error) as e:
            logger.debug(f"Could not set process limit: {e}")

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        stdin_data: Optional[str] = None,
    ) -> SandboxResult:
        """
        Run a subprocess with resource limits and timeout.

        Args:
            cmd: Command and arguments
            cwd: Working directory
            env: Environment variables
            stdin_data: Data to write to stdin

        Returns:
            SandboxResult with execution details
        """
        import time

        start_time = time.time()

        # Prepare environment
        if env is None:
            env = os.environ.copy()

        # Restrict PATH to essential directories
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

        # Disable network access where possible
        if not self.config.allow_network:
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                preexec_fn=self._set_resource_limits,
                start_new_session=True,  # Create new process group for clean kill
            )
        except FileNotFoundError:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Command not found: {cmd[0]}",
            )
        except PermissionError:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Permission denied: {cmd[0]}",
            )
        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Failed to start process: {type(e).__name__}: {e}",
            )

        try:
            # Run with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_data.encode() if stdin_data else None),
                timeout=self.config.timeout_seconds,
            )

            execution_time = (time.time() - start_time) * 1000

            # Truncate output if too large
            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[: self.config.max_output_bytes] + b"\n... (truncated)"
            if len(stderr) > self.config.max_output_bytes:
                stderr = stderr[: self.config.max_output_bytes] + b"\n... (truncated)"

            return SandboxResult(
                status=SandboxStatus.SUCCESS,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            # Kill the process group
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                execution_time_ms=self.config.timeout_seconds * 1000,
                error_message=f"Execution exceeded timeout of {self.config.timeout_seconds}s",
            )

        except Exception as e:
            # Kill if still running
            try:
                process.kill()
            except ProcessLookupError:
                pass

            return SandboxResult(
                status=SandboxStatus.EXECUTION_ERROR,
                error_message=f"Execution error: {type(e).__name__}: {e}",
            )

    async def execute_lean(
        self,
        lean_code: str,
        project_dir: Optional[Path] = None,
    ) -> SandboxResult:
        """
        Execute Lean 4 code in a sandboxed environment.

        Args:
            lean_code: Lean 4 source code to check
            project_dir: Optional Lean project directory (with lakefile.lean)

        Returns:
            SandboxResult with type-checking results
        """
        # Check if lean is available
        lean_path = shutil.which("lean")
        if not lean_path:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message="Lean 4 not installed. Install from https://leanprover.github.io/",
            )

        # Create temp directory and file
        temp_dir = self._create_temp_dir()
        source_file = temp_dir / "Proof.lean"

        try:
            source_file.write_text(lean_code, encoding="utf-8")
        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Failed to write source file: {e}",
            )

        # Run lean type checker
        cmd = [lean_path, str(source_file)]

        result = await self._run_subprocess(cmd, cwd=temp_dir)

        # Cleanup
        if self.config.cleanup_on_exit:
            self._cleanup_temp_dirs()

        return result

    async def execute_z3(
        self,
        smtlib_code: str,
    ) -> SandboxResult:
        """
        Execute Z3 SMT solver in a sandboxed environment.

        Args:
            smtlib_code: SMT-LIB2 code to check

        Returns:
            SandboxResult with solver results
        """
        # Check if z3 is available
        z3_path = shutil.which("z3")
        if not z3_path:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message="Z3 not installed. Install with: pip install z3-solver",
            )

        # Create temp directory and file
        temp_dir = self._create_temp_dir()
        source_file = temp_dir / "query.smt2"

        try:
            source_file.write_text(smtlib_code, encoding="utf-8")
        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Failed to write source file: {e}",
            )

        # Run z3 solver
        cmd = [
            z3_path,
            f"-T:{int(self.config.timeout_seconds)}",  # Timeout in seconds
            f"-memory:{self.config.memory_mb}",  # Memory limit in MB
            str(source_file),
        ]

        result = await self._run_subprocess(cmd, cwd=temp_dir)

        # Cleanup
        if self.config.cleanup_on_exit:
            self._cleanup_temp_dirs()

        return result

    async def execute(
        self,
        code: str,
        language: str = "z3",
    ) -> SandboxResult:
        """
        Execute code in the appropriate sandbox.

        Args:
            code: Source code to execute
            language: "z3", "lean", "lean4"

        Returns:
            SandboxResult with execution results
        """
        language = language.lower()

        if language in ("z3", "smt", "smtlib", "smt2"):
            return await self.execute_z3(code)
        elif language in ("lean", "lean4"):
            return await self.execute_lean(code)
        else:
            return SandboxResult(
                status=SandboxStatus.SETUP_FAILED,
                error_message=f"Unknown language: {language}. Supported: z3, lean",
            )

# Convenience function
async def run_sandboxed(
    code: str,
    language: str = "z3",
    timeout: float = 30.0,
    memory_mb: int = 512,
) -> SandboxResult:
    """
    Run code in a sandboxed environment.

    Args:
        code: Source code to execute
        language: "z3" or "lean"
        timeout: Timeout in seconds
        memory_mb: Memory limit in MB

    Returns:
        SandboxResult with execution results
    """
    sandbox = ProofSandbox(timeout=timeout, memory_mb=memory_mb)
    return await sandbox.execute(code, language)
