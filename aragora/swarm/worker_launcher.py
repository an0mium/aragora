"""Worker launcher for supervised swarm runs.

Spawns Claude Code or Codex CLI processes in provisioned worktrees,
monitors their lifecycle, and collects completion receipts.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass(slots=True)
class WorkerProcess:
    """Tracks a running worker subprocess."""

    work_order_id: str
    agent: str
    worktree_path: str
    branch: str
    pid: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    diff: str = ""

    @property
    def is_running(self) -> bool:
        return self.exit_code is None and self.pid is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "agent": self.agent,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "pid": self.pid,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
        }


@dataclass(slots=True)
class LaunchConfig:
    """Configuration for worker launches."""

    claude_path: str = "claude"
    codex_path: str = "codex"
    timeout_seconds: float = 600.0
    claude_model: str | None = None
    codex_model: str | None = None
    auto_commit: bool = True


class WorkerLauncher:
    """Launch and monitor Claude Code / Codex worker processes.

    Usage::

        launcher = WorkerLauncher()
        proc = await launcher.launch(work_order, worktree_path="/path/to/wt")
        # ... later ...
        result = await launcher.wait(proc.work_order_id)
    """

    def __init__(self, config: LaunchConfig | None = None) -> None:
        self.config = config or LaunchConfig()
        self._workers: dict[str, WorkerProcess] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def launch(
        self,
        work_order: dict[str, Any],
        *,
        worktree_path: str,
        branch: str = "main",
    ) -> WorkerProcess:
        """Launch a worker process for a work order.

        Args:
            work_order: Dict with work_order_id, title, description, target_agent,
                        file_scope, expected_tests, etc.
            worktree_path: Absolute path to the provisioned worktree.
            branch: Git branch the worktree is on.

        Returns:
            WorkerProcess tracking the launched subprocess.
        """
        work_order_id = str(work_order.get("work_order_id", "unknown"))
        agent = str(work_order.get("target_agent", "claude")).strip() or "claude"
        prompt = self._build_prompt(work_order)

        cmd = self._build_command(agent, prompt, worktree_path)
        if not cmd:
            raise RuntimeError(f"Cannot build launch command for agent={agent}")

        cli_path = cmd[0]
        if not shutil.which(cli_path):
            raise FileNotFoundError(f"{cli_path} CLI not found on PATH")

        logger.info(
            "Launching %s worker for %s in %s",
            agent,
            work_order_id,
            worktree_path,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        worker = WorkerProcess(
            work_order_id=work_order_id,
            agent=agent,
            worktree_path=worktree_path,
            branch=branch,
            pid=proc.pid,
        )
        self._workers[work_order_id] = worker
        self._processes[work_order_id] = proc
        return worker

    async def wait(
        self,
        work_order_id: str,
        *,
        timeout: float | None = None,
    ) -> WorkerProcess:
        """Wait for a worker to complete and collect results.

        Args:
            work_order_id: The work order to wait for.
            timeout: Override timeout (defaults to config.timeout_seconds).

        Returns:
            Completed WorkerProcess with exit_code, stdout, stderr, diff.
        """
        worker = self._workers.get(work_order_id)
        proc = self._processes.get(work_order_id)
        if worker is None or proc is None:
            raise KeyError(f"No running worker for {work_order_id}")

        effective_timeout = timeout or self.config.timeout_seconds

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
            worker.exit_code = proc.returncode
            worker.stdout = stdout_bytes.decode(errors="replace")
            worker.stderr = stderr_bytes.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            worker.exit_code = -1
            worker.stderr = f"Timed out after {effective_timeout}s"
            logger.warning("Worker %s timed out", work_order_id)

        worker.completed_at = datetime.now(UTC).isoformat()

        # Collect git diff from worktree
        worker.diff = await self._collect_diff(worker.worktree_path)

        # Auto-commit if configured and there are changes
        if self.config.auto_commit and worker.diff and worker.exit_code == 0:
            await self._auto_commit(worker)

        logger.info(
            "Worker %s completed: exit=%s diff_lines=%d",
            work_order_id,
            worker.exit_code,
            worker.diff.count("\n"),
        )

        # Clean up process references
        self._processes.pop(work_order_id, None)
        return worker

    async def launch_and_wait(
        self,
        work_order: dict[str, Any],
        *,
        worktree_path: str,
        branch: str = "main",
    ) -> WorkerProcess:
        """Launch a worker and wait for it to complete."""
        worker = await self.launch(
            work_order,
            worktree_path=worktree_path,
            branch=branch,
        )
        return await self.wait(worker.work_order_id)

    def get_worker(self, work_order_id: str) -> WorkerProcess | None:
        return self._workers.get(work_order_id)

    def active_workers(self) -> list[WorkerProcess]:
        return [w for w in self._workers.values() if w.is_running]

    def _build_command(
        self,
        agent: str,
        prompt: str,
        worktree_path: str,
    ) -> list[str]:
        """Build the CLI command for the given agent type.

        All commands use create_subprocess_exec (no shell) for safety.
        """
        if agent == "claude":
            cmd = [
                self.config.claude_path,
                "-p",  # non-interactive
                prompt,
                "--yes",  # auto-approve edits
            ]
            if self.config.claude_model:
                cmd.extend(["--model", self.config.claude_model])
            return cmd

        if agent == "codex":
            cmd = [
                self.config.codex_path,
                "exec",
                prompt,
                "--full-auto",
            ]
            if self.config.codex_model:
                cmd.extend(["--model", self.config.codex_model])
            return cmd

        # Unknown agent — try as claude with warning
        logger.warning("Unknown agent %r, falling back to claude", agent)
        return [
            self.config.claude_path,
            "-p",
            prompt,
            "--yes",
        ]

    @staticmethod
    def _build_prompt(work_order: dict[str, Any]) -> str:
        """Build the task prompt from a work order dict."""
        parts: list[str] = []

        title = str(work_order.get("title", "")).strip()
        if title:
            parts.append(f"# Task: {title}")

        description = str(work_order.get("description", "")).strip()
        if description:
            parts.append(description)

        file_scope = work_order.get("file_scope", [])
        if file_scope:
            scope_list = ", ".join(str(f) for f in file_scope)
            parts.append(f"Files in scope: {scope_list}")

        expected_tests = work_order.get("expected_tests", [])
        if expected_tests:
            tests_text = "\n".join(f"  - {t}" for t in expected_tests)
            parts.append(f"Run these tests to verify:\n{tests_text}")

        metadata = work_order.get("metadata", {})
        acceptance = metadata.get("acceptance_criteria", [])
        if acceptance:
            criteria_text = "\n".join(f"  - {c}" for c in acceptance)
            parts.append(f"Acceptance criteria:\n{criteria_text}")

        constraints = metadata.get("constraints", [])
        if constraints:
            constraints_text = "\n".join(f"  - {c}" for c in constraints)
            parts.append(f"Constraints:\n{constraints_text}")

        parts.append(
            "After completing the task, run the tests listed above to verify "
            "your changes work correctly. Commit your changes with a descriptive message."
        )

        return "\n\n".join(parts)

    @staticmethod
    async def _collect_diff(worktree_path: str) -> str:
        """Collect git diff (staged + unstaged) from the worktree."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "HEAD",
                cwd=worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            return stdout.decode(errors="replace")
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return ""

    @staticmethod
    async def _auto_commit(worker: WorkerProcess) -> None:
        """Auto-commit changes in the worktree if any."""
        try:
            # Stage all changes
            add_proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "-A",
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(add_proc.communicate(), timeout=10)

            # Commit
            msg = f"feat(swarm): {worker.agent} completed {worker.work_order_id}"
            commit_proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                msg,
                "--allow-empty",
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(commit_proc.communicate(), timeout=10)
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
            logger.warning("Auto-commit failed for %s: %s", worker.work_order_id, exc)
