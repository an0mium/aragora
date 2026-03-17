"""Worker launcher for supervised swarm runs.

Spawns Claude Code or Codex CLI processes in provisioned worktrees,
reusing the managed-session wrapper so worktree locks/logs stay coherent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Any

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Session artifacts that autonomous workers should never treat as deliverable
# output.  These are infrastructure metadata files created by the harness, not
# user work product.  They must be stripped from changed_paths before any
# result is qualified.
SESSION_ARTIFACTS: frozenset[str] = frozenset(
    {
        ".codex_session_meta.json",
        ".codex_session.log",
        ".codex_session_active",
    }
)

# Exit codes where the worker likely completed its work but the process was
# terminated by a transport-level signal (e.g. broken pipe).  Only these
# codes are eligible for auto-commit salvage — all other non-zero exits are
# treated as genuine failures whose partial output must NOT become a
# concrete deliverable.
_SALVAGEABLE_EXIT_CODES: frozenset[int] = frozenset(
    {
        141,  # SIGPIPE — stdout pipe closed before process finished writing
    }
)

DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 900.0


@dataclass(slots=True)
class WorkerProcess:
    """Tracks a running worker subprocess."""

    work_order_id: str
    agent: str
    worktree_path: str
    branch: str
    pid: int | None = None
    session_id: str = ""
    lease_id: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    diff: str = ""
    initial_head: str = ""
    head_sha: str = ""
    commit_shas: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    expected_tests: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)

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
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "head_sha": self.head_sha,
            "commit_shas": list(self.commit_shas),
            "changed_paths": list(self.changed_paths),
            "expected_tests": list(self.expected_tests),
            "tests_run": list(self.tests_run),
            "verification_results": [dict(item) for item in self.verification_results],
        }


@dataclass(slots=True)
class LaunchConfig:
    """Configuration for worker launches."""

    claude_path: str = "claude"
    codex_path: str = "codex"
    timeout_seconds: float = 2400.0
    no_progress_timeout_seconds: float = 1800.0
    claude_model: str | None = None
    codex_model: str | None = None
    auto_commit: bool = True
    use_managed_session_script: bool = True
    base_branch: str = "main"
    detach: bool = False


class WorkerLauncher:
    """Launch and monitor Claude Code / Codex worker processes."""

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
        """Launch a worker process for a work order."""
        work_order_id = str(work_order.get("work_order_id", "unknown"))
        agent = str(work_order.get("target_agent", "claude")).strip() or "claude"
        prompt = self._build_prompt(work_order)
        session_id = str(work_order.get("owner_session_id", "")).strip()
        lease_id = str(work_order.get("lease_id", "")).strip()

        cmd = self._build_command(
            agent,
            prompt,
            worktree_path,
            session_id=session_id,
        )
        if not cmd:
            raise RuntimeError(f"Cannot build launch command for agent={agent}")

        self._validate_launch_command(cmd, agent)
        initial_head = await self._git_output(worktree_path, "rev-parse", "HEAD")

        logger.info(
            "Launching %s worker for %s in %s",
            agent,
            work_order_id,
            worktree_path,
        )

        if self.config.detach:
            log_dir = Path(worktree_path)
            stdout_file = open(log_dir / ".swarm_worker_stdout.log", "w")  # noqa: SIM115
            stderr_file = open(log_dir / ".swarm_worker_stderr.log", "w")  # noqa: SIM115
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=worktree_path,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        else:
            stdout_file = None
            stderr_file = None
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
            session_id=session_id,
            lease_id=lease_id,
            initial_head=initial_head,
            expected_tests=[
                str(item) for item in work_order.get("expected_tests", []) if str(item).strip()
            ],
            command=list(cmd),
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
        """Wait for a worker to complete and collect results."""
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
            # In detached mode, stdout/stderr are file handles, not PIPE —
            # communicate() returns (None, None).
            if stdout_bytes is not None:
                worker.stdout = stdout_bytes.decode(errors="replace")
            else:
                worker.stdout = self._read_log_file(worker.worktree_path, "stdout")
            if stderr_bytes is not None:
                worker.stderr = stderr_bytes.decode(errors="replace")
            else:
                worker.stderr = self._read_log_file(worker.worktree_path, "stderr")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            worker.exit_code = -1
            worker.stderr = f"Timed out after {effective_timeout}s"
            logger.warning("Worker %s timed out", work_order_id)

        worker.completed_at = datetime.now(UTC).isoformat()
        try:
            worker.diff = await self._collect_diff(worker.worktree_path)

            _exit_ok = worker.exit_code == 0 or worker.exit_code in _SALVAGEABLE_EXIT_CODES
            # ``git diff HEAD`` only detects modifications to tracked files.
            # Workers that create NEW (untracked) files show no diff.
            # Always fall back to ``git status --porcelain`` which detects
            # both untracked and modified files.
            _has_changes = bool(worker.diff) or (
                _exit_ok and await self._has_working_tree_changes(worker.worktree_path)
            )
            if self.config.auto_commit and _has_changes and _exit_ok:
                await self._auto_commit(worker)

            worker.head_sha = await self._git_output(worker.worktree_path, "rev-parse", "HEAD")
            worker.commit_shas = await self._collect_commit_shas(
                worker.worktree_path,
                initial_head=worker.initial_head,
                head_sha=worker.head_sha,
            )
            worker.changed_paths = await self._collect_changed_paths(
                worker.worktree_path,
                initial_head=worker.initial_head,
                head_sha=worker.head_sha,
            )
            if worker.exit_code == 0 and worker.expected_tests:
                worker.verification_results = await self._run_verification_commands(
                    worker.worktree_path,
                    worker.expected_tests,
                )
                worker.tests_run = [
                    str(item.get("command", "")).strip()
                    for item in worker.verification_results
                    if str(item.get("command", "")).strip()
                ]
        finally:
            self._cleanup_session_artifacts(worker.worktree_path)

        logger.info(
            "Worker %s completed: exit=%s commits=%d changed_paths=%d",
            work_order_id,
            worker.exit_code,
            len(worker.commit_shas),
            len(worker.changed_paths),
        )

        self._processes.pop(work_order_id, None)
        return worker

    async def collect_finished(
        self,
        *,
        work_order_ids: list[str] | None = None,
        poll_timeout: float = 0.01,
    ) -> list[WorkerProcess]:
        """Collect only workers that have already finished."""
        completed: list[WorkerProcess] = []
        ids = work_order_ids or list(self._processes.keys())
        for work_order_id in ids:
            proc = self._processes.get(work_order_id)
            if proc is None:
                continue
            finished = proc.returncode is not None
            if not finished:
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=poll_timeout)
                    finished = True
                except asyncio.TimeoutError:
                    finished = False
            if finished:
                # Use the configured timeout for result collection — the short
                # poll_timeout is only for checking whether the process exited.
                # communicate() on a finished process is normally fast, but
                # auto_commit (called inside wait) needs 30-60 s for git ops.
                completed.append(await self.wait(work_order_id))
        return completed

    async def snapshot_progress(self, work_order: dict[str, Any]) -> dict[str, Any]:
        """Capture lightweight progress state for a dispatched worker."""
        worktree_path = str(work_order.get("worktree_path", "")).strip()
        initial_head = str(work_order.get("initial_head", "")).strip()
        raw_pid = work_order.get("pid")
        try:
            pid = int(raw_pid) if raw_pid is not None else None
        except (TypeError, ValueError):
            pid = None

        snapshot: dict[str, Any] = {
            "pid_alive": self._is_pid_running(pid) if pid is not None else False,
            "head_sha": "",
            "changed_paths": [],
            "diff_lines": 0,
        }
        if not worktree_path:
            return snapshot

        head_sha = await self._git_output(worktree_path, "rev-parse", "HEAD")
        diff = await self._collect_diff(worktree_path)
        changed_paths = await self._collect_changed_paths(
            worktree_path,
            initial_head=initial_head,
            head_sha=head_sha,
        )
        snapshot.update(
            {
                "head_sha": head_sha,
                "changed_paths": list(changed_paths),
                "diff_lines": diff.count("\n") if diff else 0,
            }
        )
        return snapshot

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
        *,
        session_id: str = "",
    ) -> list[str]:
        """Build the launch command for the given agent type."""
        inner = self._build_agent_command(agent, prompt)
        if not self.config.use_managed_session_script:
            return inner

        session_script = Path(worktree_path).resolve() / "scripts" / "codex_session.sh"
        managed_dir = str(Path(worktree_path).resolve().parent)
        cmd = [
            "bash",
            str(session_script),
            "--agent",
            agent,
            "--base",
            self.config.base_branch,
            "--managed-dir",
            managed_dir,
            "--no-maintain",
            "--no-reconcile",
        ]
        if session_id:
            cmd.extend(["--session-id", session_id])
        cmd.append("--")
        cmd.extend(inner)
        return cmd

    def _build_agent_command(self, agent: str, prompt: str) -> list[str]:
        if agent == "claude":
            cmd = [self.config.claude_path, "-p", prompt, "--yes"]
            if self.config.claude_model:
                cmd.extend(["--model", self.config.claude_model])
            return cmd

        if agent == "codex":
            cmd = [self.config.codex_path, "exec", prompt, "--full-auto"]
            if self.config.codex_model:
                cmd.extend(["--model", self.config.codex_model])
            return cmd

        logger.warning("Unknown agent %r, falling back to claude", agent)
        return [self.config.claude_path, "-p", prompt, "--yes"]

    def _validate_launch_command(self, cmd: list[str], agent: str) -> None:
        if not cmd:
            raise RuntimeError("Empty launch command")
        if self.config.use_managed_session_script:
            inner_cli = self.config.claude_path if agent == "claude" else self.config.codex_path
            if agent not in {"claude", "codex"}:
                inner_cli = self.config.claude_path
            if not shutil.which(inner_cli):
                raise FileNotFoundError(f"{inner_cli} CLI not found on PATH")
            session_script = Path(cmd[1]) if len(cmd) > 1 else None
            if session_script is None or not session_script.exists():
                raise FileNotFoundError(f"session script not found: {session_script}")
            return

        cli_path = cmd[0]
        if not shutil.which(cli_path):
            raise FileNotFoundError(f"{cli_path} CLI not found on PATH")

    @staticmethod
    def _build_prompt(work_order: dict[str, Any]) -> str:
        """Build the task prompt from a work order dict."""
        parts: list[str] = []
        metadata = work_order.get("metadata", {})

        title = str(work_order.get("title", "")).strip()
        if title:
            parts.append(f"# Task: {title}")

        parts.append(
            "You are one Aragora-managed CLI worker lane inside a supervised swarm run. "
            "Do only the bounded work for this lane and leave coordination, integration, "
            "and human escalation to the boss lane."
        )

        description = str(work_order.get("description", "")).strip()
        if description:
            parts.append(description)

        file_scope = work_order.get("file_scope", [])
        if file_scope:
            scope_list = "\n".join(f"  - {f}" for f in file_scope)
            parts.append(
                "MANDATORY FILE SCOPE CONSTRAINT:\n"
                "You MUST only create, modify, or delete files within these paths:\n"
                f"{scope_list}\n"
                "DO NOT edit any files outside this scope. Edits outside scope will be "
                "rejected and your work will be discarded. This is enforced automatically."
            )

        expected_tests = work_order.get("expected_tests", [])
        if expected_tests:
            tests_text = "\n".join(f"  - {t}" for t in expected_tests)
            parts.append(f"Expected validation:\n{tests_text}")

        acceptance = metadata.get("acceptance_criteria", [])
        if acceptance:
            criteria_text = "\n".join(f"  - {c}" for c in acceptance)
            parts.append(f"Acceptance criteria:\n{criteria_text}")

        constraints = metadata.get("constraints", [])
        if constraints:
            constraints_text = "\n".join(f"  - {c}" for c in constraints)
            parts.append(f"Constraints:\n{constraints_text}")

        approval_required = bool(work_order.get("approval_required", False))
        if approval_required:
            parts.append(
                "Decision boundary:\n"
                "  - If you hit a real ambiguity, approval boundary, or blocker, stop cleanly and "
                "report the exact reason instead of widening scope."
            )

        lease_id = str(work_order.get("lease_id", "")).strip()
        if lease_id:
            parts.append(
                "Receipt expectation:\n"
                f"  - Lease id: {lease_id}\n"
                "  - Aragora will record the completion receipt after a successful exit from this lane."
            )

        parts.append(
            "Stop condition:\n"
            "  - Finish the bounded lane or stop at a real blocker.\n"
            "  - Run the expected validation commands when possible, or state exactly why they could not run.\n"
            "  - Stage only the files you intentionally changed with `git add <file> ...`.\n"
            "  - Do NOT use `git add -A` or `git add .` — session metadata files must not be committed.\n"
            '  - Commit with a descriptive message using `git commit -m "..."` before exiting.\n'
            "  - After committing, push your branch: `git push origin HEAD`.\n"
            "  - If `git push` fails (e.g. no remote, permission error), that is acceptable — "
            "the harness will attempt to push for you.\n"
            "  - Exit with a truthful final state; do not claim integration or approval work is done unless it happened in this lane."
        )

        return "\n\n".join(parts)

    @staticmethod
    async def _git_output(worktree_path: str, *args: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                return ""
            return stdout.decode(errors="replace").rstrip()
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return ""

    @classmethod
    async def _has_working_tree_changes(cls, worktree_path: str) -> bool:
        """Check for real (non-artifact) working-tree changes via git status.

        This is a robust fallback for ``_collect_diff`` which relies on
        ``git diff HEAD`` — that command can return empty on timeout, error,
        or when only binary files changed.  ``git status --porcelain`` is
        cheaper and more reliable for a yes/no dirty-tree check.
        """
        # Expand untracked directories into file paths so docs-only tasks that
        # create new trees still qualify as concrete deliverables.
        status = await cls._git_output(
            worktree_path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        for line in status.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if path and path not in SESSION_ARTIFACTS:
                return True
        return False

    @classmethod
    async def _collect_diff(cls, worktree_path: str) -> str:
        return await cls._git_output(worktree_path, "diff", "HEAD")

    @classmethod
    async def _collect_commit_shas(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        if not initial_head or not head_sha or initial_head == head_sha:
            return []
        output = await cls._git_output(
            worktree_path, "rev-list", "--reverse", f"{initial_head}..{head_sha}"
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    @classmethod
    async def _collect_changed_paths(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        changed: set[str] = set()
        if initial_head and head_sha and initial_head != head_sha:
            diff_names = await cls._git_output(
                worktree_path,
                "diff",
                "--name-only",
                f"{initial_head}..{head_sha}",
            )
            changed.update(line.strip() for line in diff_names.splitlines() if line.strip())

        # _git_output() strips the whole output, which can eat the leading
        # space from porcelain status lines like " M docs/file.py".  Parse
        # robustly: skip the first two status characters if the line matches
        # the XY+space pattern, otherwise fall back to lstrip-after-status.
        status_output = await cls._git_output(
            worktree_path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        for line in status_output.splitlines():
            if not line or len(line) < 2:
                continue
            # Porcelain v1: XY<space>PATH  (3 chars prefix when both X,Y present)
            # After .strip(), a leading-space status " M" becomes "M" — only
            # 2 chars before the path.  Detect by checking whether position 2
            # is a space (full prefix) or part of the path (stripped prefix).
            if len(line) > 2 and line[2] == " ":
                path = line[3:].strip()
            else:
                # Stripped leading space: "M docs/..." or "?? docs/..."
                # Find the first space after the status chars.
                first_space = line.find(" ")
                if first_space < 0:
                    continue
                path = line[first_space + 1 :].strip()
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            if path:
                changed.add(path)
        # Strip session artifacts — these are harness metadata, not deliverables
        changed -= SESSION_ARTIFACTS
        return sorted(changed)

    @classmethod
    async def collect_detached_result(
        cls,
        *,
        work_order_id: str,
        agent: str,
        worktree_path: str,
        branch: str,
        pid: int | None = None,
        initial_head: str = "",
        auto_commit: bool = True,
        expected_tests: list[str] | None = None,
    ) -> WorkerProcess | None:
        """Collect results from a detached worker by checking PID and worktree state.

        Returns None if the worker is still running (PID alive).
        Returns a WorkerProcess with collected results if finished.
        """
        session_meta = cls._read_session_meta(worktree_path)
        session_exit_code, session_completed_at = cls._terminal_session_result(session_meta)

        if session_exit_code is None and pid is not None and cls._is_pid_running(pid):
            return None

        worker = WorkerProcess(
            work_order_id=work_order_id,
            agent=agent,
            worktree_path=worktree_path,
            branch=branch,
            pid=pid,
            initial_head=initial_head,
            expected_tests=[
                str(item).strip() for item in expected_tests or [] if str(item).strip()
            ],
        )

        worker.stdout = cls._read_log_file(worktree_path, "stdout")
        worker.stderr = cls._read_log_file(worktree_path, "stderr")
        worker.exit_code = 0 if session_exit_code is None else session_exit_code
        worker.completed_at = session_completed_at or datetime.now(UTC).isoformat()
        try:
            worker.diff = await cls._collect_diff(worktree_path)

            _exit_ok = worker.exit_code == 0 or worker.exit_code in _SALVAGEABLE_EXIT_CODES
            # ``git diff HEAD`` only detects modifications to tracked files.
            # Workers that create NEW (untracked) files show no diff.
            # Always fall back to ``git status --porcelain`` which detects
            # both untracked and modified files.
            _has_changes = bool(worker.diff) or (
                _exit_ok and await cls._has_working_tree_changes(worktree_path)
            )
            if auto_commit and _has_changes and _exit_ok:
                await cls._auto_commit(worker)

            worker.head_sha = await cls._git_output(worktree_path, "rev-parse", "HEAD")
            worker.commit_shas = await cls._collect_commit_shas(
                worktree_path,
                initial_head=initial_head,
                head_sha=worker.head_sha,
            )
            worker.changed_paths = await cls._collect_changed_paths(
                worktree_path,
                initial_head=initial_head,
                head_sha=worker.head_sha,
            )
            if worker.exit_code == 0 and worker.expected_tests:
                worker.verification_results = await cls._run_verification_commands(
                    worktree_path,
                    worker.expected_tests,
                )
                worker.tests_run = [
                    str(item.get("command", "")).strip()
                    for item in worker.verification_results
                    if str(item.get("command", "")).strip()
                ]
        finally:
            # Wait for the worker process (including its shell EXIT trap) to
            # fully terminate before removing artifacts.  Without this wait
            # the codex_session.sh trap can recreate .codex_session_meta.json
            # and append to .codex_session.log after Python-side cleanup (#902).
            _cleanup_pid = pid
            if _cleanup_pid is None:
                raw_pid = session_meta.get("pid")
                if raw_pid is not None:
                    try:
                        _cleanup_pid = int(raw_pid)
                    except (TypeError, ValueError):
                        pass
            if _cleanup_pid is not None:
                await cls._wait_for_pid_exit(_cleanup_pid)
            cls._cleanup_session_artifacts(worktree_path)

        logger.info(
            "Collected detached worker %s: commits=%d changed_paths=%d",
            work_order_id,
            len(worker.commit_shas),
            len(worker.changed_paths),
        )
        return worker

    @staticmethod
    def _read_session_meta(worktree_path: str) -> dict[str, Any]:
        meta_path = Path(worktree_path) / ".codex_session_meta.json"
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _cleanup_session_artifacts(worktree_path: str) -> None:
        """Remove harness session artifacts after result collection.

        The launcher reads these files to qualify terminal state, but they must
        not remain behind as future worktree dirt once collection completes.
        """
        root = Path(worktree_path)
        for artifact in SESSION_ARTIFACTS:
            artifact_path = root / artifact
            try:
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                else:
                    artifact_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.debug("Could not remove session artifact %s: %s", artifact_path, exc)

    @staticmethod
    def _terminal_session_result(session_meta: dict[str, Any]) -> tuple[int | None, str | None]:
        ended_at = str(session_meta.get("ended_at", "")).strip()
        if not ended_at:
            return None, None
        raw_exit_code = session_meta.get("exit_code")
        try:
            exit_code = int(raw_exit_code)
        except (TypeError, ValueError):
            return None, None
        return exit_code, ended_at

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        """Check if a process with the given PID is still alive."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    @classmethod
    async def _wait_for_pid_exit(cls, pid: int, timeout: float = 5.0) -> None:
        """Wait for a process to fully terminate, including trap handlers.

        The shell wrapper ``codex_session.sh`` uses an EXIT trap that rewrites
        session artifacts.  After sending SIGTERM the trap handler may still be
        running when Python-side cleanup fires.  Polling until the PID is gone
        ensures the trap has completed before we delete those files.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if not cls._is_pid_running(pid):
                return
            await asyncio.sleep(0.1)
        logger.debug("PID %d still alive after %.1fs wait — proceeding with cleanup", pid, timeout)

    @classmethod
    async def _run_verification_commands(
        cls,
        worktree_path: str,
        commands: list[str],
        *,
        timeout: float = DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Run required verification commands and capture structured results."""
        results: list[dict[str, Any]] = []
        for raw_command in commands:
            command = str(raw_command).strip()
            if not command:
                continue

            started = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash",
                    "-lc",
                    command,
                    cwd=worktree_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, OSError) as exc:
                results.append(
                    {
                        "command": command,
                        "exit_code": -2,
                        "passed": False,
                        "stdout": "",
                        "stderr": str(exc),
                        "duration_seconds": round(time.monotonic() - started, 3),
                    }
                )
                continue

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                exit_code = proc.returncode if proc.returncode is not None else -1
                stdout = (stdout_bytes or b"").decode(errors="replace")
                stderr = (stderr_bytes or b"").decode(errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                exit_code = -1
                stdout = ""
                stderr = f"Timed out after {int(timeout)}s"

            results.append(
                {
                    "command": command,
                    "exit_code": exit_code,
                    "passed": exit_code == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            )
        return results

    @staticmethod
    def _read_log_file(worktree_path: str, stream: str) -> str:
        """Read a detached worker's log file (stdout or stderr)."""
        log_path = Path(worktree_path) / f".swarm_worker_{stream}.log"
        try:
            return log_path.read_text(errors="replace")
        except (FileNotFoundError, OSError):
            return ""

    @staticmethod
    async def _auto_commit(worker: WorkerProcess) -> None:
        """Auto-commit changes in the worktree, excluding session artifacts."""
        try:
            # Stage everything, then unstage session artifacts so they are
            # never committed by the harness.
            add_proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "-A",
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, add_stderr = await asyncio.wait_for(add_proc.communicate(), timeout=10)
            if add_proc.returncode != 0:
                logger.warning(
                    "git add -A failed for %s (rc=%s): %s",
                    worker.work_order_id,
                    add_proc.returncode,
                    (add_stderr or b"").decode(errors="replace").strip(),
                )
                return

            # Unstage session artifacts — ignore errors if files are not staged
            for artifact in SESSION_ARTIFACTS:
                reset_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "reset",
                    "HEAD",
                    "--",
                    artifact,
                    cwd=worker.worktree_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(reset_proc.communicate(), timeout=5)

            # Verify something is actually staged before committing
            diff_index_proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--cached",
                "--quiet",
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(diff_index_proc.communicate(), timeout=5)
            if diff_index_proc.returncode == 0:
                # returncode 0 means no staged changes — nothing to commit
                logger.info(
                    "No staged changes for %s after git add — skipping commit",
                    worker.work_order_id,
                )
                return

            if worker.exit_code is not None and worker.exit_code != 0:
                msg = (
                    f"fix(swarm): salvage {worker.agent} work for "
                    f"{worker.work_order_id} (exit {worker.exit_code})"
                )
            else:
                msg = f"feat(swarm): {worker.agent} completed {worker.work_order_id}"
            commit_proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "--no-verify",
                "-m",
                msg,
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, commit_stderr = await asyncio.wait_for(commit_proc.communicate(), timeout=30)
            if commit_proc.returncode != 0:
                logger.warning(
                    "git commit failed for %s (rc=%s): %s",
                    worker.work_order_id,
                    commit_proc.returncode,
                    (commit_stderr or b"").decode(errors="replace").strip(),
                )
                return

            # Push the branch so the deliverable is visible upstream.
            # Best-effort: if push fails (no remote, auth, etc.) the local
            # commit is still collected by _collect_commit_shas.
            await WorkerLauncher._auto_push(worker)
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
            logger.warning("Auto-commit failed for %s: %s", worker.work_order_id, exc)

    @staticmethod
    async def _auto_push(worker: WorkerProcess) -> None:
        """Best-effort push of the worker branch to origin.

        Called after auto-commit to ensure the deliverable is available for
        review and PR creation.  If push fails the local commit is still
        collected — the deliverable gate accepts branch+commit_shas from
        the local worktree.
        """
        try:
            push_proc = await asyncio.create_subprocess_exec(
                "git",
                "push",
                "origin",
                "HEAD",
                cwd=worker.worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, push_stderr = await asyncio.wait_for(push_proc.communicate(), timeout=30)
            if push_proc.returncode != 0:
                logger.info(
                    "Auto-push failed for %s (rc=%s): %s — local commit preserved",
                    worker.work_order_id,
                    push_proc.returncode,
                    (push_stderr or b"").decode(errors="replace").strip()[:200],
                )
            else:
                logger.info("Auto-pushed branch for %s", worker.work_order_id)
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
            logger.info(
                "Auto-push skipped for %s: %s — local commit preserved",
                worker.work_order_id,
                exc,
            )
